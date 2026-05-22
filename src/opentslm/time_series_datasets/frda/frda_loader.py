# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""Utilities for loading FRDA AIM JSON recordings and creating reproducible splits."""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

CHANNEL_NAMES = ["AccX", "AccY", "AccZ", "GyrX", "GyrY", "GyrZ", "Force"]
TIME_SERIES_LABELS = [
    "Accelerometer X-axis",
    "Accelerometer Y-axis",
    "Accelerometer Z-axis",
    "Gyroscope X-axis",
    "Gyroscope Y-axis",
    "Gyroscope Z-axis",
    "Grip force",
]

FILE_COLUMNS = {
    "file_name_01": 1,
    "file_name_02": 2,
    "file_name_03": 3,
}

FILENAME_COLUMN_ALIASES = (
    "filename",
    "file_name",
    "json_file",
)


def _safe_float(value: Any) -> float:
    """Best-effort float conversion returning NaN for invalid values."""
    if value is None:
        return float("nan")
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _safe_int(value: Any, default: int = -1) -> int:
    val = _safe_float(value)
    if np.isnan(val):
        return default
    return int(val)


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    txt = str(value).strip()
    if txt.lower() == "nan":
        return ""
    return txt


def _derive_group_id_from_filename(file_name: str) -> str:
    """
    Derive a stable group ID when explicit patient_id is unavailable.

    For the merged FRDA filename schema (15-digit stem), the first 12 digits
    identify the patient while the final 3 digits identify test/device variant.
    """
    stem = Path(file_name).stem.strip()
    if not stem:
        return ""

    digits = "".join(ch for ch in stem if ch.isdigit())
    if len(digits) >= 13:
        return digits[:12]
    if len(digits) >= 9:
        return digits

    tokens = [tok for tok in re.split(r"[_.\-/ ]+", stem) if tok]
    if len(tokens) >= 2 and tokens[1].isdigit():
        return f"{tokens[0]}_{tokens[1]}"
    if tokens:
        return tokens[0]
    return stem


def _derive_group_id(
    *,
    explicit_patient_id: str,
    file_name: str,
    group_id_strategy: str,
    filename_group_prefix_len: int,
) -> str:
    if group_id_strategy == "patient_id":
        return explicit_patient_id

    stem = Path(file_name).stem.strip()
    if group_id_strategy == "filename_full":
        return stem

    if group_id_strategy == "filename_prefix":
        if filename_group_prefix_len <= 0:
            return stem
        digits = "".join(ch for ch in stem if ch.isdigit())
        if len(digits) >= filename_group_prefix_len:
            return digits[:filename_group_prefix_len]
        if len(stem) >= filename_group_prefix_len:
            return stem[:filename_group_prefix_len]
        return stem

    if group_id_strategy != "auto":
        raise ValueError(
            "group_id_strategy must be one of "
            "['auto', 'patient_id', 'filename_prefix', 'filename_full']"
        )

    if explicit_patient_id != "":
        return explicit_patient_id

    if filename_group_prefix_len > 0:
        digits = "".join(ch for ch in stem if ch.isdigit())
        if len(digits) >= filename_group_prefix_len:
            return digits[:filename_group_prefix_len]

    return _derive_group_id_from_filename(file_name)


def _median_filter_1d(signal: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply a median filter to a 1D signal using reflect padding."""
    if kernel_size <= 1:
        return signal.astype(np.float32, copy=True)

    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd, got {kernel_size}")

    if signal.size == 0:
        return signal.astype(np.float32, copy=True)

    pad = kernel_size // 2
    if signal.size == 1:
        padded = np.pad(signal, (pad, pad), mode="constant", constant_values=signal[0])
    else:
        padded = np.pad(signal, (pad, pad), mode="reflect")

    windows = np.lib.stride_tricks.sliding_window_view(padded, kernel_size)
    filtered = np.median(windows, axis=-1)
    return filtered.astype(np.float32, copy=False)


def median_filter_multichannel(signal: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Apply median filtering independently on each channel."""
    filtered_channels = [_median_filter_1d(signal[:, i], kernel_size=kernel_size) for i in range(signal.shape[1])]
    return np.stack(filtered_channels, axis=1).astype(np.float32, copy=False)


def zscore_per_channel(signal: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Z-score normalize each channel independently with epsilon guard."""
    means = signal.mean(axis=0, keepdims=True)
    stds = signal.std(axis=0, keepdims=True)
    denom = np.where(stds < eps, 1.0, stds)
    normalized = (signal - means) / denom

    # Force near-constant channels exactly to zero for numerical stability.
    constant_mask = (stds < eps).reshape(-1)
    if constant_mask.any():
        normalized[:, constant_mask] = 0.0

    return normalized.astype(np.float32, copy=False)


def upsample_signal_linear(signal: np.ndarray, factor: int = 2) -> np.ndarray:
    """Linearly upsample time-series by an integer factor."""
    if factor <= 1:
        return signal.astype(np.float32, copy=True)

    n = signal.shape[0]
    if n <= 1:
        return signal.astype(np.float32, copy=True)

    x_old = np.arange(n, dtype=np.float64)
    x_new = np.linspace(0.0, float(n - 1), num=n * factor, endpoint=True)
    upsampled = np.stack(
        [np.interp(x_new, x_old, signal[:, i]) for i in range(signal.shape[1])],
        axis=1,
    )
    return upsampled.astype(np.float32, copy=False)


def parse_record_rows(record_rows: Iterable[Any], test_id: int) -> np.ndarray:
    """
    Parse raw JSON `record` rows into [N, 7] float array.

    Output channels are fixed as:
    [AccX, AccY, AccZ, GyrX, GyrY, GyrZ, Force].
    """
    parsed: List[List[float]] = []

    for row in record_rows:
        accx = accy = accz = gyrx = gyry = gyrz = force = float("nan")

        if isinstance(row, list):
            if len(row) == 9:
                # [timestamp, marker, accx, accy, accz, gyrx, gyry, gyrz, force]
                accx = _safe_float(row[2])
                accy = _safe_float(row[3])
                accz = _safe_float(row[4])
                gyrx = _safe_float(row[5])
                gyry = _safe_float(row[6])
                gyrz = _safe_float(row[7])
                force = _safe_float(row[8])
            elif len(row) == 8:
                # [timestamp, marker, accx, accy, accz, gyrx, gyry, gyrz]
                accx = _safe_float(row[2])
                accy = _safe_float(row[3])
                accz = _safe_float(row[4])
                gyrx = _safe_float(row[5])
                gyry = _safe_float(row[6])
                gyrz = _safe_float(row[7])
                force = 0.0
            else:
                continue

        elif isinstance(row, dict):
            accx = _safe_float(row.get("accx"))
            accy = _safe_float(row.get("accy"))
            accz = _safe_float(row.get("accz"))
            gyrx = _safe_float(row.get("gyrx"))
            gyry = _safe_float(row.get("gyry"))
            gyrz = _safe_float(row.get("gyrz"))
            force = _safe_float(row.get("force", 0.0))

        else:
            continue

        core = np.array([accx, accy, accz, gyrx, gyry, gyrz], dtype=np.float64)
        if np.isnan(core).any() or np.isinf(core).any():
            continue

        # Force is only meaningful for TestID 1 (AIM-C); pad zeros otherwise.
        if test_id != 1:
            force = 0.0
        if np.isnan(force) or np.isinf(force):
            force = 0.0

        parsed.append([
            float(accx),
            float(accy),
            float(accz),
            float(gyrx),
            float(gyry),
            float(gyrz),
            float(force),
        ])

    if not parsed:
        raise ValueError("No valid record rows were parsed from JSON payload")

    return np.asarray(parsed, dtype=np.float32)


def load_json_signal(json_path: str) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    """Load one AIM JSON file and return parsed raw signal with test ID."""
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {json_path}")

    record_rows = payload.get("record")
    if not isinstance(record_rows, list):
        raise ValueError(f"Missing or invalid 'record' list in {json_path}")

    test_id = _safe_int(payload.get("TestID"), default=-1)
    if test_id <= 0:
        # Best-effort fallback if TestID is absent.
        first = record_rows[0] if record_rows else None
        if isinstance(first, list) and len(first) == 9:
            test_id = 1
        else:
            test_id = 2

    signal = parse_record_rows(record_rows, test_id=test_id)
    return signal, test_id, payload


def preprocess_signal(
    signal: np.ndarray,
    test_id: int,
    *,
    median_kernel_size: int = 5,
    normalize: bool = True,
    upsample_test3: bool = True,
) -> np.ndarray:
    """Apply FRDA preprocessing steps to raw [N,7] signal."""
    processed = signal.astype(np.float32, copy=True)

    # AIM-P (TestID 3) recorded at 50Hz and needs upsampling to 100Hz.
    if upsample_test3 and int(test_id) == 3:
        processed = upsample_signal_linear(processed, factor=2)

    processed = median_filter_multichannel(processed, kernel_size=median_kernel_size)

    if normalize:
        processed = zscore_per_channel(processed)

    if not np.isfinite(processed).all():
        raise ValueError("Preprocessing produced non-finite values")

    return processed.astype(np.float32, copy=False)


def window_signal(
    signal: np.ndarray,
    *,
    window_size: int = 3000,
    stride: int = 1500,
) -> List[np.ndarray]:
    """Window [N,7] signal into fixed-length windows with overlap and short padding."""
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")

    n = signal.shape[0]
    if n < window_size:
        pad = window_size - n
        if n <= 1:
            padded = np.pad(signal, ((0, pad), (0, 0)), mode="constant", constant_values=0.0)
        else:
            try:
                padded = np.pad(signal, ((0, pad), (0, 0)), mode="reflect")
            except ValueError:
                padded = np.pad(signal, ((0, pad), (0, 0)), mode="constant", constant_values=0.0)
        return [padded.astype(np.float32, copy=False)]

    starts = list(range(0, n - window_size + 1, stride))
    last_start = n - window_size
    if starts[-1] != last_start:
        starts.append(last_start)

    windows = [signal[s : s + window_size].astype(np.float32, copy=False) for s in starts]
    return windows


def build_windowed_signal_from_json(
    json_path: str,
    *,
    window_size: int = 3000,
    stride: int = 1500,
    median_kernel_size: int = 5,
    normalize: bool = True,
    upsample_test3: bool = True,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """Load, preprocess, and window one JSON recording."""
    raw_signal, test_id, payload = load_json_signal(json_path)
    processed = preprocess_signal(
        raw_signal,
        test_id,
        median_kernel_size=median_kernel_size,
        normalize=normalize,
        upsample_test3=upsample_test3,
    )
    windows = window_signal(processed, window_size=window_size, stride=stride)

    metadata = {
        "json_path": str(json_path),
        "test_id": int(test_id),
        "raw_length": int(raw_signal.shape[0]),
        "processed_length": int(processed.shape[0]),
        "record_id": str(payload.get("RecordID", "")),
    }
    return windows, metadata


def window_to_model_inputs(window: np.ndarray) -> Tuple[List[str], List[List[float]]]:
    """Convert one [T,7] window into OpenTSLM-SP compatible prompt inputs."""
    if window.ndim != 2 or window.shape[1] != 7:
        raise ValueError(f"Expected window shape [T,7], got {window.shape}")

    means = window.mean(axis=0)
    stds = window.std(axis=0)

    texts: List[str] = []
    series: List[List[float]] = []

    for i, label in enumerate(TIME_SERIES_LABELS):
        texts.append(f"{label}, mean={means[i]:.4f}, std={stds[i]:.4f}:")
        series.append(window[:, i].astype(np.float32).tolist())

    return texts, series


def _normalize_metadata_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        norm = col.strip().lower()
        if norm in {"patient id", "patient_id"}:
            rename_map[col] = "patient_id"
        elif norm in {"date_of_collection", "visit_date"}:
            rename_map[col] = "visit_date"
        elif norm in {"file name", "filename"}:
            rename_map[col] = "filename"
        elif norm in {"device type"}:
            rename_map[col] = "device"
    return df.rename(columns=rename_map)


def load_file_records_from_metadata(
    metadata_csv: str,
    json_root: str,
    *,
    target_col: str = "mFARS",
    group_id_strategy: str = "auto",
    filename_group_prefix_len: int = 12,
) -> List[Dict[str, Any]]:
    """Create file-level records from metadata rows for FRDA regression."""
    df = pd.read_csv(metadata_csv)
    df = _normalize_metadata_columns(df)

    if target_col not in df.columns:
        raise ValueError(f"Missing required metadata column: {target_col}")

    records: List[Dict[str, Any]] = []
    json_root_path = Path(json_root)

    filename_col = next((c for c in FILENAME_COLUMN_ALIASES if c in df.columns), None)
    file_cols_present = [c for c in FILE_COLUMNS if c in df.columns]
    if filename_col is None and not file_cols_present:
        raise ValueError(
            "No filename column found. Expected one of "
            f"{list(FILENAME_COLUMN_ALIASES)} or legacy columns {sorted(FILE_COLUMNS)}"
        )

    for row in df.to_dict(orient="records"):
        target = _safe_float(row.get(target_col))
        if np.isnan(target):
            continue

        explicit_patient_id = _safe_str(row.get("patient_id"))
        visit_date = _safe_str(row.get("visit_date"))
        device = _safe_str(row.get("device"))

        file_entries: List[Tuple[str, Any, int]] = []
        if filename_col is not None:
            file_entries.append((filename_col, row.get(filename_col), _safe_int(row.get("test_id"), default=-1)))
        else:
            for col in file_cols_present:
                file_entries.append((col, row.get(col), FILE_COLUMNS[col]))

        for source_col, file_name_raw, source_test_id in file_entries:
            file_name = _safe_str(file_name_raw)
            if file_name == "":
                continue

            json_path = json_root_path / file_name
            if not json_path.exists():
                continue

            group_id = _derive_group_id(
                explicit_patient_id=explicit_patient_id,
                file_name=file_name,
                group_id_strategy=group_id_strategy,
                filename_group_prefix_len=filename_group_prefix_len,
            )
            if group_id == "":
                continue

            patient_id = explicit_patient_id or group_id

            records.append(
                {
                    "patient_id": patient_id,
                    "group_id": group_id,
                    "visit_date": visit_date,
                    "device": device,
                    "file_name": file_name,
                    "json_path": str(json_path),
                    "target": float(target),
                    "source_column": source_col,
                    "source_test_id": int(source_test_id),
                }
            )

    if not records:
        raise ValueError("No valid file records found in metadata CSV")

    return records


def _split_groups(
    group_ids: List[str],
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[List[str], List[str], List[str]]:
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(
            f"train/val/test ratios must sum to 1.0, got {train_ratio}, {val_ratio}, {test_ratio}"
        )

    if len(group_ids) < 3:
        raise ValueError(f"Need at least 3 unique groups for train/val/test split, got {len(group_ids)}")

    rng = np.random.default_rng(seed)
    shuffled = list(group_ids)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))

    n_train = max(1, min(n_train, n - 2))
    n_val = max(1, min(n_val, n - n_train - 1))
    n_test = n - n_train - n_val

    if n_test < 1:
        if n_val > 1:
            n_val -= 1
        elif n_train > 1:
            n_train -= 1
        n_test = n - n_train - n_val

    train_ids = shuffled[:n_train]
    val_ids = shuffled[n_train : n_train + n_val]
    test_ids = shuffled[n_train + n_val :]

    return sorted(train_ids), sorted(val_ids), sorted(test_ids)


def create_split_manifest(
    metadata_csv: str,
    json_root: str,
    *,
    output_path: str,
    target_col: str = "mFARS",
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    group_id_strategy: str = "auto",
    filename_group_prefix_len: int = 12,
) -> Dict[str, Any]:
    """Create and persist a deterministic patient-group split manifest."""
    records = load_file_records_from_metadata(
        metadata_csv,
        json_root,
        target_col=target_col,
        group_id_strategy=group_id_strategy,
        filename_group_prefix_len=filename_group_prefix_len,
    )

    group_ids = sorted({_safe_str(r.get("group_id")) or _safe_str(r.get("patient_id")) for r in records})
    train_ids, val_ids, test_ids = _split_groups(
        group_ids,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

    split_records = {"train": [], "validation": [], "test": []}
    for record in records:
        gid = _safe_str(record.get("group_id")) or _safe_str(record.get("patient_id"))
        if gid in train_set:
            split_records["train"].append(record)
        elif gid in val_set:
            split_records["validation"].append(record)
        elif gid in test_set:
            split_records["test"].append(record)

    manifest = {
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "metadata_csv": str(metadata_csv),
        "json_root": str(json_root),
        "target_col": target_col,
        "seed": int(seed),
        "group_id_strategy": str(group_id_strategy),
        "filename_group_prefix_len": int(filename_group_prefix_len),
        "ratios": {
            "train": float(train_ratio),
            "validation": float(val_ratio),
            "test": float(test_ratio),
        },
        "group_key": "group_id",
        "groups": {
            "train": train_ids,
            "validation": val_ids,
            "test": test_ids,
        },
        "records": split_records,
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def create_split_manifest_from_split_csv(
    metadata_csv: str,
    split_csv: str,
    json_root: str,
    *,
    output_path: str,
    target_col: str = "mfars_total",
) -> Dict[str, Any]:
    """Build a split manifest from master_adults.csv + split_adults.csv.

    Uses the canonical participant-level split instead of a random re-split.
    ``split_csv`` must have columns ``participant_id`` and ``split``
    (values: train / val / test).
    """
    meta_df = pd.read_csv(metadata_csv)
    split_df = pd.read_csv(split_csv)

    if target_col not in meta_df.columns:
        raise ValueError(f"Missing target column '{target_col}' in {metadata_csv}")
    if "participant_id" not in meta_df.columns:
        raise ValueError(f"Missing 'participant_id' column in {metadata_csv}")
    if "participant_id" not in split_df.columns or "split" not in split_df.columns:
        raise ValueError(f"split_csv must have 'participant_id' and 'split' columns: {split_csv}")

    split_map: Dict[str, str] = {
        str(row["participant_id"]): str(row["split"])
        for _, row in split_df.iterrows()
    }

    json_root_path = Path(json_root)
    split_records: Dict[str, List[Dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    groups: Dict[str, List[str]] = {"train": [], "validation": [], "test": []}

    seen_groups: Dict[str, str] = {}
    for row in meta_df.to_dict(orient="records"):
        target = _safe_float(row.get(target_col))
        if np.isnan(target):
            continue

        pid = str(_safe_int(row.get("participant_id"), default=-1))
        if pid == "-1":
            continue

        split_label = split_map.get(pid)
        if split_label is None:
            continue
        # Remap split CSV label "val" → manifest key "validation"
        if split_label == "val":
            split_label = "validation"
        if split_label not in split_records:
            continue

        file_name = _safe_str(row.get("filename"))
        if file_name == "":
            continue

        json_path = json_root_path / file_name
        if not json_path.exists():
            continue

        patient_id = _safe_str(row.get("participant_id"))
        split_records[split_label].append(
            {
                "patient_id": patient_id,
                "group_id": patient_id,
                "visit_date": _safe_str(row.get("recording_date")),
                "device": _safe_str(row.get("device_type")),
                "file_name": file_name,
                "json_path": str(json_path),
                "target": float(target),
                "source_column": "filename",
                "source_test_id": -1,
            }
        )

        if pid not in seen_groups:
            seen_groups[pid] = split_label
            groups[split_label].append(pid)

    manifest = {
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "metadata_csv": str(metadata_csv),
        "split_csv": str(split_csv),
        "json_root": str(json_root),
        "target_col": target_col,
        "group_key": "participant_id",
        "groups": groups,
        "records": split_records,
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def load_split_manifest(path: str) -> Dict[str, Any]:
    """Load a split manifest from disk and perform minimal validation."""
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if "records" not in manifest:
        raise ValueError(f"Invalid split manifest at {path}: missing 'records'")

    for split in ("train", "validation", "test"):
        if split not in manifest["records"]:
            raise ValueError(f"Invalid split manifest at {path}: missing records.{split}")

    return manifest


def get_records_for_split(manifest: Dict[str, Any], split: str) -> List[Dict[str, Any]]:
    """Return file-level records for one split."""
    valid = {"train", "validation", "test"}
    if split not in valid:
        raise ValueError(f"split must be one of {sorted(valid)}, got {split}")

    records = manifest.get("records", {}).get(split, [])
    if not isinstance(records, list):
        raise ValueError(f"Invalid records for split {split}")
    return records
