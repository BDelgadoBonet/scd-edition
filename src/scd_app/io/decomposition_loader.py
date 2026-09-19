"""Load decomposition pickles and adapt upstream SCD output for the editor."""

from __future__ import annotations

import io
import pickle
import re
from pathlib import Path
from typing import Any
import warnings

import numpy as np

from scd_app.core.utils import to_numpy


GUI_FORMAT = "scd-edition"
UPSTREAM_SCD_FORMAT = "swarm-contrastive-decomposition"
_GUI_REQUIRED_KEYS = {"ports", "discharge_times", "pulse_trains"}
_SCD_TIMESTAMP_KEYS = ("timestamps", "MUPulses")
_SCD_SOURCE_KEYS = ("source", "sources")
_SCD_METRIC_KEYS = ("silhouettes", "RoA", "fr", "cov", "best_exp")

class UnsupportedDecompositionFormat(ValueError):
    """Raised when a pickle is neither an SCD Edition file nor raw SCD output."""

class UnsupportedDecompositionFormatWarning(UserWarning):
    """Raised when a decomposition value is not supported but can be replaced with default values."""

def detect_decomposition_format(data: Any) -> str:
    """Return the supported format name for an unpickled object."""
    if not isinstance(data, dict):
        raise UnsupportedDecompositionFormat(
            f"Expected a dictionary, found {type(data).__name__}."
        )
    if _GUI_REQUIRED_KEYS.issubset(data):
        return GUI_FORMAT
    has_timestamps = any(key in data for key in _SCD_TIMESTAMP_KEYS)
    has_sources = any(key in data for key in _SCD_SOURCE_KEYS)
    if has_timestamps and has_sources:
        return UPSTREAM_SCD_FORMAT
    keys = ", ".join(sorted(str(key) for key in data)) or "<none>"
    raise UnsupportedDecompositionFormat(
        "The pickle is not a supported SCD Edition or raw "
        f"swarm-contrastive-decomposition result. Found keys: {keys}"
    )


def load_decomposition_file(path: Path) -> dict:
    """Load a trusted pickle and return the normalized SCD Edition structure.

    Pickle files can execute code while loading. This function is intended only
    for decomposition files created by the user or another trusted source.
    """
    path = Path(path)
    with path.open("rb") as handle:
        data = _CPUCompatibleUnpickler(handle).load()
    file_format = detect_decomposition_format(data)
    if file_format == GUI_FORMAT:
        return data
    return convert_scd_output(data, source_path=path)


class _CPUCompatibleUnpickler(pickle.Unpickler):
    """Load torch-backed pickle values without requiring their original GPU."""

    def find_class(self, module: str, name: str):
        if module == "torch.storage" and name == "_load_from_bytes":
            import torch

            def load_torch_storage(value):
                return torch.load(
                    io.BytesIO(value),
                    map_location="cpu",
                    weights_only=False,
                )

            return load_torch_storage
        return super().find_class(module, name)


def convert_scd_output(data: dict, source_path: Path | None = None) -> dict:
    """Convert one raw upstream SCD dictionary into a one-port editor file."""
    if detect_decomposition_format(data) != UPSTREAM_SCD_FORMAT:
        raise UnsupportedDecompositionFormat(
            "The supplied dictionary is not raw SCD output."
        )

    timestamps_raw = _first_present(data, _SCD_TIMESTAMP_KEYS)
    timestamps = _timestamp_list(timestamps_raw)
    if not timestamps:
        raise UnsupportedDecompositionFormat("The SCD result contains no motor units.")

    n_units = len(timestamps)
    sources = [
        np.asarray(item).flatten()
        for item in _unit_list(
            _first_present(data, _SCD_SOURCE_KEYS), n_units, "source"
        )
    ]
    if any(source.size == 0 for source in sources):
        raise UnsupportedDecompositionFormat("One or more SCD sources are empty.")
    source_lengths = {int(source.size) for source in sources}
    if len(source_lengths) != 1:
        raise UnsupportedDecompositionFormat(
            f"SCD sources have inconsistent lengths: {sorted(source_lengths)}"
        )
    source_length = source_lengths.pop()

    for unit_idx, unit_timestamps in enumerate(timestamps):
        if unit_timestamps.size and (
            int(unit_timestamps.min()) < 0
            or int(unit_timestamps.max()) >= source_length
        ):
            raise UnsupportedDecompositionFormat(
                f"SCD timestamps for unit {unit_idx} fall outside its source."
            )

    filters = _filter_list(data.get("filters", data.get("mu_filters")), n_units)
    editor_filters = None if all(item is None for item in filters) else filters
    preprocessing_config = data.get("preprocessing_config") or {}
    if not isinstance(preprocessing_config, dict):
        raise UnsupportedDecompositionFormat(
            "SCD preprocessing_config must be a dictionary."
        )

    sampling_rate = data.get(
        "sampling_rate",
        data.get("fsamp", preprocessing_config.get("sampling_frequency")),
    )
    try:
        sampling_rate = float(sampling_rate)
    except (TypeError, ValueError) as exc:
        raise UnsupportedDecompositionFormat(
            "The SCD result does not record a valid sampling frequency."
        ) from exc
    if sampling_rate <= 0:
        raise UnsupportedDecompositionFormat(
            "The SCD sampling frequency must be positive."
        )

    # Plateau coordinates (start and end times for the decomposed signal segment, in samples)
    start_time = preprocessing_config.get("start_time", None)
    end_time = preprocessing_config.get("end_time", None)
    if (start_time is None) or (end_time is None):
        start_time = 0
        end_time = -1
        warnings.warn(
            "The SCD result did not record start or end time; start is set to 0, end to -1.",
            UnsupportedDecompositionFormatWarning)

    # - Convert to float, then samples
    try:
        start_time = float(start_time)
    except (TypeError, ValueError):
        start_time = 0.0
        warnings.warn(
            "The SCD result does not record a valid start time; start is set to 0.",
            UnsupportedDecompositionFormatWarning)
    try:
        end_time = float(end_time)
    except (TypeError, ValueError):
        end_time = -1.0
        warnings.warn(
            "The SCD result does not record a valid end time; end is set to -1.",
            UnsupportedDecompositionFormatWarning)

    plateau_coords = np.round(np.array([start_time, end_time]) * sampling_rate).astype(int)

    # - Force within valid decomposed signal length (as recorded by sources)
    if plateau_coords[0] < 0:
        plateau_coords[0] = 0
        warnings.warn(
            "The SCD result does not record a valid start time; start is set to 0.",
            UnsupportedDecompositionFormatWarning)
    elif plateau_coords[0] > source_length:
        plateau_coords[0] = 0
        warnings.warn(
            "The SCD result records a start time past the existing sources; start is set to 0.",
            UnsupportedDecompositionFormatWarning)

    if plateau_coords[1] < 0:
        plateau_coords[1] = source_length
    elif plateau_coords[1] > source_length:
        plateau_coords[1] = source_length
        warnings.warn(
            "The SCD result records an end time past the existing sources; end is set to the maximum length recorded by sources.",
            UnsupportedDecompositionFormatWarning)

    if plateau_coords[0] == plateau_coords[1]:
        plateau_coords[0] = 0
        plateau_coords[1] = source_length
        warnings.warn(
            "The SCD result records the same start and end time; re-setting to full length recorded by sources.",
            UnsupportedDecompositionFormatWarning)
    elif plateau_coords[0] > plateau_coords[1]:
        plateau_coords = plateau_coords[[1, 0]]
        warnings.warn(
            "The SCD result records start time later than end time; times are reversed.",
            UnsupportedDecompositionFormatWarning)

    w_mat_raw = data.get("w_mat")
    w_mat = to_numpy(w_mat_raw) if w_mat_raw is not None else None
    if w_mat is not None and w_mat.size == 0:
        w_mat = None
    n_channels = _infer_channel_count(filters, w_mat, preprocessing_config)
    port_name = _infer_port_name(source_path)

    provenance = {
        "format": UPSTREAM_SCD_FORMAT,
        "source_file": source_path.name if source_path is not None else None,
        "scd_commit": _read_scd_commit(source_path),
        "converted_by": "scd-edition",
    }
    provenance = {key: value for key, value in provenance.items() if value is not None}

    scd_metadata = {
        key: _portable_value(data[key]) for key in _SCD_METRIC_KEYS if key in data
    }

    # Data given as input to swarm-contrastive-decomposition pipeline (optional)
    emg = data.get("data", None)
    if emg is not None:
        # Format to numpy array (channels x samples)
        emg_formatted = to_numpy(emg)
        if emg_formatted.ndim == 2 and emg_formatted.shape[0] > emg_formatted.shape[1]:
            emg_formatted = emg_formatted.T
        emg = emg_formatted  # and refresh

    return {
        "version": 1.1,
        "data": emg,
        "ports": [port_name],
        "sampling_rate": sampling_rate,
        "discharge_times": [timestamps],
        "pulse_trains": [sources],
        "mu_filters": [editor_filters],
        "w_mat": [w_mat],
        "peel_off_sequence": [_portable_value(data.get("peel_off_sequence", []))],
        "preprocessing_config": [_portable_value(preprocessing_config)],
        "plateau_coords": plateau_coords.tolist(),
        "chans_per_electrode": [n_channels],
        "channel_indices": [list(range(n_channels))],
        "emg_mask": [[0] * n_channels],
        "electrodes": [None],
        "aux_channels": [],
        "aux_configs": [],
        "decomposition_params": [{}],
        "acquisition_metadata": {"format": "scd-output"},
        "import_provenance": provenance,
        "scd_metadata": scd_metadata,
        "skip_filter_recalc": False
    }


def _first_present(data: dict, keys: tuple[str, ...]):
    for key in keys:
        if key in data:
            return data[key]
    return None


def _timestamp_list(value: Any) -> list[np.ndarray]:
    if isinstance(value, (list, tuple)):
        items = list(value)
    else:
        arr = to_numpy(value)
        if arr.dtype == object:
            items = list(arr.flatten())
        elif arr.ndim <= 1:
            items = [arr]
        else:
            items = [arr[index] for index in range(arr.shape[0])]
    result = []
    for item in items:
        arr = to_numpy(item).flatten()
        if arr.size and not np.all(np.isfinite(arr)):
            raise UnsupportedDecompositionFormat(
                "SCD timestamps contain non-finite values."
            )
        result.append(arr.astype(np.int64, copy=False))
    return result


def _unit_list(value: Any, n_units: int, field_name: str) -> list[np.ndarray]:
    if isinstance(value, (list, tuple)):
        items = list(value)
    else:
        arr = to_numpy(value)
        if arr.dtype == object:
            items = list(arr.flatten())
        elif n_units == 1:
            items = [arr]
        elif arr.ndim >= 2 and arr.shape[0] == n_units:
            items = [arr[index] for index in range(n_units)]
        elif arr.ndim == 2 and arr.shape[1] == n_units:
            items = [arr[:, index] for index in range(n_units)]
        else:
            raise UnsupportedDecompositionFormat(
                f"Could not separate SCD {field_name} data into {n_units} units."
            )
    if len(items) != n_units:
        raise UnsupportedDecompositionFormat(
            f"SCD has {n_units} timestamp sets but {len(items)} {field_name} entries."
        )
    return [to_numpy(item) for item in items]


def _filter_list(value: Any, n_units: int) -> list[np.ndarray | None]:
    if value is None:
        return [None] * n_units
    return [np.asarray(item).flatten() for item in _unit_list(value, n_units, "filter")]


def _infer_channel_count(
    filters: list[np.ndarray | None],
    w_mat: np.ndarray | None,
    preprocessing_config: dict,
) -> int:
    feature_count = 0
    for mu_filter in filters:
        if mu_filter is not None and mu_filter.size:
            feature_count = int(mu_filter.size)
            break
    if not feature_count and w_mat is not None and w_mat.ndim >= 1:
        feature_count = int(w_mat.shape[0])
    try:
        extension_factor = int(preprocessing_config.get("extension_factor", 0))
    except (TypeError, ValueError):
        extension_factor = 0
    if extension_factor > 0 and feature_count % extension_factor == 0:
        return feature_count // extension_factor
    return 0


def _infer_port_name(source_path: Path | None) -> str:
    if source_path is None:
        return "SCD"
    match = re.search(r"(?:^|_)muscle-([^_]+)", source_path.stem, re.IGNORECASE)
    if match:
        return match.group(1)
    stem = re.sub(r"_scddict$", "", source_path.stem, flags=re.IGNORECASE)
    return stem or "SCD"


def _read_scd_commit(source_path: Path | None) -> str | None:
    if source_path is None:
        return None
    stem = re.sub(r"_scddict$", "", source_path.stem, flags=re.IGNORECASE)
    commit_path = source_path.with_name(f"{stem}_scdcommit.txt")
    if not commit_path.is_file():
        return None
    commit = commit_path.read_text(encoding="utf-8").strip()
    return commit if re.fullmatch(r"[0-9a-fA-F]{7,40}", commit) else None


def _portable_value(value: Any):
    if isinstance(value, dict):
        return {key: _portable_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_portable_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.copy()
    if hasattr(value, "detach"):
        arr = to_numpy(value)
        return arr.item() if arr.ndim == 0 else arr
    return value
