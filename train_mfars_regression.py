# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""Train FRDA mFARS regression models (OpenTSLM or ridge-window baseline)."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from scipy.stats import kurtosis, skew
from sklearn.ensemble import HistGradientBoostingRegressor
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from opentslm.model.llm.OpenTSLMRegressionSP import OpenTSLMRegressionSP
from opentslm.model.regression.ridge_window import RidgeWindowRegressor
from opentslm.time_series_datasets.frda.FRDAMFARSDataset import FRDAMFARSDataset
from opentslm.time_series_datasets.frda.frda_loader import create_split_manifest
from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate


def _sanitize_llm_id(llm_id: str) -> str:
    safe = llm_id.split("/")[-1].replace(".", "_").replace("-", "_")
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe


def _get_device(device_arg: str | None) -> str:
    if device_arg:
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    mse = float(np.mean((preds - targets) ** 2))
    mae = float(np.mean(np.abs(preds - targets)))
    rmse = float(np.sqrt(mse))

    target_mean = float(np.mean(targets))
    ss_res = float(np.sum((targets - preds) ** 2))
    ss_tot = float(np.sum((targets - target_mean) ** 2))
    r2 = 0.0 if ss_tot <= 0.0 else float(1.0 - ss_res / ss_tot)

    return {"r2": r2, "mae": mae, "rmse": rmse}


def _build_collate_fn(patch_size: int):
    def _collate(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return extend_time_series_to_match_patch_size_and_aggregate(batch, patch_size=patch_size)

    return _collate


def _compute_weighted_loss_from_preds(
    preds: torch.Tensor,
    targets: torch.Tensor,
    loss_type: str,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if loss_type == "huber":
        per_sample = torch.nn.functional.smooth_l1_loss(preds, targets, beta=1.0, reduction="none")
    elif loss_type == "mse":
        per_sample = torch.nn.functional.mse_loss(preds, targets, reduction="none")
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    if sample_weights is None:
        return per_sample.mean()

    weights = sample_weights.to(per_sample.dtype)
    weights = weights / weights.sum().clamp(min=1e-8)
    return (per_sample * weights).sum()


def _compute_weighted_np_loss(
    preds: np.ndarray,
    targets: np.ndarray,
    loss_type: str,
    sample_weights: np.ndarray | None,
) -> float:
    if loss_type == "huber":
        err = preds - targets
        abs_err = np.abs(err)
        delta = 1.0
        per_sample = np.where(abs_err <= delta, 0.5 * (err**2), delta * (abs_err - 0.5 * delta))
    elif loss_type == "mse":
        per_sample = (preds - targets) ** 2
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    if sample_weights is None:
        return float(np.mean(per_sample))
    weights = sample_weights.astype(np.float64)
    weights = weights / max(np.sum(weights), 1e-8)
    return float(np.sum(per_sample * weights))


def _save_json(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _save_jsonl(path: str, rows: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _aggregate_file_rows(window_rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in window_rows:
        grouped[row["json_path"]].append(row)

    file_rows: List[Dict[str, Any]] = []
    for json_path, rows in grouped.items():
        pred_values = np.asarray([r["prediction"] for r in rows], dtype=np.float64)
        target_values = np.asarray([r["target"] for r in rows], dtype=np.float64)
        first = rows[0]
        file_rows.append(
            {
                "json_path": json_path,
                "file_name": first["file_name"],
                "patient_id": first["patient_id"],
                "visit_date": first.get("visit_date", ""),
                "test_id": int(first["test_id"]),
                "num_windows": len(rows),
                "prediction": float(np.mean(pred_values)),
                "prediction_std": float(np.std(pred_values)),
                "target": float(target_values[0]),
            }
        )

    file_preds = np.asarray([r["prediction"] for r in file_rows], dtype=np.float64)
    file_targets = np.asarray([r["target"] for r in file_rows], dtype=np.float64)
    return file_rows, _metrics(file_preds, file_targets)


def _compute_target_stats(dataset: FRDAMFARSDataset) -> Tuple[float, float]:
    per_file: Dict[str, float] = {}
    for sample in dataset.samples:
        per_file[sample["json_path"]] = float(sample["target"])
    targets = np.asarray(list(per_file.values()), dtype=np.float64)
    mean = float(np.mean(targets))
    std = float(np.std(targets))
    if std < 1e-8:
        std = 1.0
    return mean, std


def _window_matrix_from_sample(sample: Dict[str, Any]) -> np.ndarray:
    ts = np.asarray(sample["time_series"], dtype=np.float64)
    if ts.ndim != 2:
        raise ValueError(f"Expected time_series with ndim=2, got shape {ts.shape}")

    # Dataset stores channels as [7, T]; convert to [T, 7].
    if ts.shape[0] == 7:
        return ts.T
    if ts.shape[1] == 7:
        return ts
    raise ValueError(f"Expected 7 channels in one axis, got shape {ts.shape}")


def _extract_rich_window_features(window: np.ndarray, test_id: int) -> np.ndarray:
    w = np.asarray(window, dtype=np.float64)
    if w.ndim != 2 or w.shape[1] != 7:
        raise ValueError(f"Expected window [T,7], got {w.shape}")

    mean = np.mean(w, axis=0)
    std = np.std(w, axis=0)
    med = np.median(w, axis=0)
    p10 = np.percentile(w, 10.0, axis=0)
    p25 = np.percentile(w, 25.0, axis=0)
    p75 = np.percentile(w, 75.0, axis=0)
    p90 = np.percentile(w, 90.0, axis=0)
    iqr = p75 - p25
    abs_mean = np.mean(np.abs(w), axis=0)
    rms = np.sqrt(np.mean(w * w, axis=0))

    diff = np.diff(w, axis=0)
    if diff.size == 0:
        mean_abs_diff = np.zeros((7,), dtype=np.float64)
        std_diff = np.zeros((7,), dtype=np.float64)
    else:
        mean_abs_diff = np.mean(np.abs(diff), axis=0)
        std_diff = np.std(diff, axis=0)

    sk = np.nan_to_num(skew(w, axis=0, bias=False, nan_policy="omit"), nan=0.0, posinf=0.0, neginf=0.0)
    ku = np.nan_to_num(kurtosis(w, axis=0, bias=False, nan_policy="omit"), nan=0.0, posinf=0.0, neginf=0.0)

    return np.concatenate(
        [
            mean,
            std,
            med,
            p10,
            p25,
            p75,
            p90,
            iqr,
            abs_mean,
            rms,
            mean_abs_diff,
            std_diff,
            sk,
            ku,
            np.asarray([float(test_id)], dtype=np.float64),
        ],
        axis=0,
    )


def _aggregate_rich_windows_to_file(
    rich_window_x: np.ndarray,
    meta_rows: List[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    grouped: Dict[str, List[int]] = defaultdict(list)
    file_targets: Dict[str, float] = {}
    file_meta: Dict[str, Dict[str, Any]] = {}

    for idx, meta in enumerate(meta_rows):
        path = meta["json_path"]
        grouped[path].append(idx)
        file_targets[path] = float(meta["target"])
        file_meta[path] = meta

    file_keys = sorted(grouped.keys())
    file_x = []
    file_y = []

    for path in file_keys:
        idxs = grouped[path]
        win_feats = rich_window_x[idxs]
        agg = np.concatenate(
            [
                np.mean(win_feats, axis=0),
                np.std(win_feats, axis=0),
                np.min(win_feats, axis=0),
                np.max(win_feats, axis=0),
                np.median(win_feats, axis=0),
                np.asarray(
                    [
                        float(file_meta[path]["num_windows_for_file"]),
                        float(file_meta[path]["raw_length"]),
                        float(file_meta[path]["processed_length"]),
                    ],
                    dtype=np.float64,
                ),
            ],
            axis=0,
        )
        file_x.append(agg)
        file_y.append(file_targets[path])

    return np.stack(file_x, axis=0), np.asarray(file_y, dtype=np.float64), file_keys


def _build_ridge_arrays(
    dataset: FRDAMFARSDataset,
    *,
    include_test_id: bool,
    use_file_balanced_weights: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None, List[Dict[str, Any]]]:
    x_rows: List[np.ndarray] = []
    y_rows: List[float] = []
    w_rows: List[float] = []
    metadata_rows: List[Dict[str, Any]] = []

    for sample in dataset.samples:
        window = _window_matrix_from_sample(sample)
        feat = RidgeWindowRegressor.extract_window_features(
            window,
            test_id=int(sample["test_id"]),
            include_test_id=include_test_id,
        )

        x_rows.append(feat)
        y_rows.append(float(sample["target"]))
        if use_file_balanced_weights:
            n = max(int(sample["num_windows_for_file"]), 1)
            w_rows.append(1.0 / float(n))
        metadata_rows.append(
            {
                "json_path": sample["json_path"],
                "file_name": sample["file_name"],
                "patient_id": sample["patient_id"],
                "visit_date": sample.get("visit_date", ""),
                "window_index": int(sample["window_index"]),
                "num_windows_for_file": int(sample["num_windows_for_file"]),
                "test_id": int(sample["test_id"]),
                "raw_length": int(sample["raw_length"]),
                "processed_length": int(sample["processed_length"]),
                "target": float(sample["target"]),
            }
        )

    if not x_rows:
        raise RuntimeError("Dataset produced no samples for ridge training")

    x = np.stack(x_rows, axis=0)
    y = np.asarray(y_rows, dtype=np.float64)
    weights = np.asarray(w_rows, dtype=np.float64) if use_file_balanced_weights else None
    return x, y, weights, metadata_rows


def _build_rich_window_arrays(dataset: FRDAMFARSDataset) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    rich_rows: List[np.ndarray] = []
    meta_rows: List[Dict[str, Any]] = []

    for sample in dataset.samples:
        window = _window_matrix_from_sample(sample)
        rich_rows.append(_extract_rich_window_features(window, test_id=int(sample["test_id"])))
        meta_rows.append(
            {
                "json_path": sample["json_path"],
                "target": float(sample["target"]),
                "num_windows_for_file": int(sample["num_windows_for_file"]),
                "raw_length": int(sample["raw_length"]),
                "processed_length": int(sample["processed_length"]),
            }
        )

    if not rich_rows:
        raise RuntimeError("Dataset produced no samples for rich feature extraction")

    return np.stack(rich_rows, axis=0), meta_rows


def _rows_with_predictions(meta_rows: List[Dict[str, Any]], preds: np.ndarray) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for meta, pred in zip(meta_rows, preds.tolist()):
        row = dict(meta)
        row["prediction"] = float(pred)
        rows.append(row)
    return rows


def _aggregate_window_features_to_file(
    x: np.ndarray,
    meta_rows: List[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    grouped: Dict[str, List[int]] = defaultdict(list)
    targets: Dict[str, float] = {}
    for idx, meta in enumerate(meta_rows):
        path = meta["json_path"]
        grouped[path].append(idx)
        targets[path] = float(meta["target"])

    file_keys = sorted(grouped.keys())
    file_x = []
    file_y = []
    for key in file_keys:
        idxs = grouped[key]
        file_x.append(np.mean(x[idxs], axis=0))
        file_y.append(targets[key])

    return np.stack(file_x, axis=0), np.asarray(file_y, dtype=np.float64), file_keys


def _broadcast_file_predictions_to_windows(
    meta_rows: List[Dict[str, Any]],
    file_keys: List[str],
    file_preds: np.ndarray,
) -> np.ndarray:
    pred_map = {k: float(v) for k, v in zip(file_keys, file_preds.tolist())}
    return np.asarray([pred_map[meta["json_path"]] for meta in meta_rows], dtype=np.float64)


def _find_best_blend_weight(
    window_preds: np.ndarray,
    file_as_window_preds: np.ndarray,
    y_window: np.ndarray,
    meta_rows: List[Dict[str, Any]],
) -> Tuple[float, np.ndarray, Dict[str, float], Dict[str, float]]:
    best_w = 1.0
    best_val_rows = _rows_with_predictions(meta_rows, window_preds)
    best_file_rows, best_file_metrics = _aggregate_file_rows(best_val_rows)
    best_window_metrics = _metrics(window_preds, y_window)

    for w in np.linspace(0.0, 1.0, 101):
        blend_preds = w * window_preds + (1.0 - w) * file_as_window_preds
        blend_rows = _rows_with_predictions(meta_rows, blend_preds)
        _, file_metrics = _aggregate_file_rows(blend_rows)
        window_metrics = _metrics(blend_preds, y_window)

        better_r2 = file_metrics["r2"] > (best_file_metrics["r2"] + 1e-12)
        tied_r2 = abs(file_metrics["r2"] - best_file_metrics["r2"]) <= 1e-12
        better_mae = file_metrics["mae"] < (best_file_metrics["mae"] - 1e-12)
        if better_r2 or (tied_r2 and better_mae):
            best_w = float(w)
            best_val_rows = blend_rows
            best_file_metrics = file_metrics
            best_window_metrics = window_metrics

    return best_w, np.asarray([r["prediction"] for r in best_val_rows], dtype=np.float64), best_window_metrics, best_file_metrics


def _evaluate_opentslm(
    model: OpenTSLMRegressionSP,
    loader: DataLoader,
    *,
    loss_type: str,
    target_mean: float,
    target_std: float,
    use_file_balanced_weights: bool,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    model.eval()
    all_preds: List[float] = []
    all_targets: List[float] = []
    rows: List[Dict[str, Any]] = []
    losses: List[float] = []

    with torch.no_grad():
        for batch in loader:
            preds_norm = model.predict_batch(batch)
            targets_raw = torch.tensor(
                [float(sample["target"]) for sample in batch],
                dtype=torch.float32,
                device=model.device,
            )
            targets_norm = (targets_raw - target_mean) / target_std

            if use_file_balanced_weights:
                weights = torch.tensor(
                    [1.0 / max(float(sample["num_windows_for_file"]), 1.0) for sample in batch],
                    dtype=torch.float32,
                    device=model.device,
                )
            else:
                weights = None

            loss = _compute_weighted_loss_from_preds(preds_norm, targets_norm, loss_type=loss_type, sample_weights=weights)
            losses.append(float(loss.item()))

            preds_raw = (preds_norm * target_std + target_mean).detach().cpu().numpy()
            targets_np = targets_raw.detach().cpu().numpy()

            all_preds.extend(preds_raw.tolist())
            all_targets.extend(targets_np.tolist())

            for sample, pred_val, target_val in zip(batch, preds_raw, targets_np):
                rows.append(
                    {
                        "json_path": sample["json_path"],
                        "file_name": sample["file_name"],
                        "patient_id": sample["patient_id"],
                        "visit_date": sample.get("visit_date", ""),
                        "window_index": int(sample["window_index"]),
                        "num_windows_for_file": int(sample["num_windows_for_file"]),
                        "test_id": int(sample["test_id"]),
                        "raw_length": int(sample["raw_length"]),
                        "processed_length": int(sample["processed_length"]),
                        "prediction": float(pred_val),
                        "target": float(target_val),
                    }
                )

    if not all_preds:
        raise RuntimeError("Validation loader produced no predictions")

    window_metrics = _metrics(np.asarray(all_preds, dtype=np.float64), np.asarray(all_targets, dtype=np.float64))
    file_rows, file_metrics = _aggregate_file_rows(rows)
    metrics = {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "window": window_metrics,
        "file": file_metrics,
    }
    return metrics, rows, file_rows


def _parse_alpha_grid(alpha_grid_str: str) -> List[float]:
    values: List[float] = []
    for chunk in alpha_grid_str.split(","):
        txt = chunk.strip()
        if not txt:
            continue
        values.append(float(txt))
    if not values:
        raise ValueError("--ridge-alpha-grid produced no valid alpha values")
    return values


def _train_ridge_backend(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    split_manifest_path: str,
    train_dataset: FRDAMFARSDataset,
    val_dataset: FRDAMFARSDataset,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    x_train, y_train, w_train, train_meta = _build_ridge_arrays(
        train_dataset,
        include_test_id=args.ridge_include_test_id,
        use_file_balanced_weights=args.file_balanced_loss,
    )
    x_val, y_val, w_val, val_meta = _build_ridge_arrays(
        val_dataset,
        include_test_id=args.ridge_include_test_id,
        use_file_balanced_weights=args.file_balanced_loss,
    )

    alpha_grid = _parse_alpha_grid(args.ridge_alpha_grid)
    history: List[Dict[str, Any]] = []

    best_alpha = None
    best_file_r2 = -float("inf")
    best_file_mae = float("inf")
    best_window_rows: List[Dict[str, Any]] = []
    best_file_rows: List[Dict[str, Any]] = []
    best_window_state: Dict[str, Any] | None = None
    best_file_state: Dict[str, Any] | None = None
    best_blend_weight = 1.0
    best_hgbr_blend_weight = 1.0
    best_hgbr_model: HistGradientBoostingRegressor | None = None

    best_checkpoint_path = output_dir / "best_model.pt"
    best_val_predictions_path = output_dir / "val_predictions.jsonl"

    x_train_file, y_train_file, _ = _aggregate_window_features_to_file(x_train, train_meta)
    x_val_file, _, val_file_keys = _aggregate_window_features_to_file(x_val, val_meta)

    rich_train_x, rich_train_meta = _build_rich_window_arrays(train_dataset)
    rich_val_x, rich_val_meta = _build_rich_window_arrays(val_dataset)
    rich_train_file_x, rich_train_file_y, _ = _aggregate_rich_windows_to_file(rich_train_x, rich_train_meta)
    rich_val_file_x, _, rich_val_file_keys = _aggregate_rich_windows_to_file(rich_val_x, rich_val_meta)

    hgbr_model = None
    hgbr_val_as_window_preds = None
    if args.ridge_use_hgbr_blend:
        hgbr_model = HistGradientBoostingRegressor(
            random_state=args.seed,
            learning_rate=args.hgbr_learning_rate,
            max_leaf_nodes=args.hgbr_max_leaf_nodes,
            max_depth=args.hgbr_max_depth,
            l2_regularization=args.hgbr_l2,
        )
        hgbr_model.fit(rich_train_file_x, rich_train_file_y)
        hgbr_val_file_preds = hgbr_model.predict(rich_val_file_x)
        hgbr_val_as_window_preds = _broadcast_file_predictions_to_windows(
            val_meta,
            rich_val_file_keys,
            hgbr_val_file_preds,
        )

    for alpha in alpha_grid:
        window_model = RidgeWindowRegressor(alpha=alpha, include_test_id=args.ridge_include_test_id)
        window_model.fit(x_train, y_train, sample_weight=w_train)
        window_preds = window_model.predict(x_val)

        file_model = RidgeWindowRegressor(
            alpha=alpha,
            include_test_id=args.ridge_include_test_id,
            standardize=False,
        )
        file_model.fit(x_train_file, y_train_file, sample_weight=None)
        file_preds_val = file_model.predict(x_val_file)
        file_as_window_preds = _broadcast_file_predictions_to_windows(val_meta, val_file_keys, file_preds_val)

        blend_weight, blend_preds, val_window_metrics, val_file_metrics = _find_best_blend_weight(
            window_preds=window_preds,
            file_as_window_preds=file_as_window_preds,
            y_window=y_val,
            meta_rows=val_meta,
        )
        hgbr_blend_weight = 1.0
        final_preds = blend_preds
        final_window_metrics = val_window_metrics
        final_file_metrics = val_file_metrics

        if hgbr_val_as_window_preds is not None:
            hgbr_blend_weight, final_preds, final_window_metrics, final_file_metrics = _find_best_blend_weight(
                window_preds=blend_preds,
                file_as_window_preds=hgbr_val_as_window_preds,
                y_window=y_val,
                meta_rows=val_meta,
            )

        val_window_rows = _rows_with_predictions(val_meta, final_preds)
        val_file_rows, _ = _aggregate_file_rows(val_window_rows)
        val_loss = _compute_weighted_np_loss(final_preds, y_val, args.loss, w_val)

        entry = {
            "alpha": float(alpha),
            "blend_weight": float(blend_weight),
            "hgbr_blend_weight": float(hgbr_blend_weight),
            "val_loss": float(val_loss),
            "val_window_r2": float(final_window_metrics["r2"]),
            "val_window_mae": float(final_window_metrics["mae"]),
            "val_window_rmse": float(final_window_metrics["rmse"]),
            "val_file_r2": float(final_file_metrics["r2"]),
            "val_file_mae": float(final_file_metrics["mae"]),
            "val_file_rmse": float(final_file_metrics["rmse"]),
        }
        history.append(entry)

        print(
            f"alpha={alpha:g}: val_file_r2={final_file_metrics['r2']:.4f}, "
            f"val_file_mae={final_file_metrics['mae']:.4f}, val_window_r2={final_window_metrics['r2']:.4f}, "
            f"blend_w={blend_weight:.2f}, hgbr_w={hgbr_blend_weight:.2f}"
        )

        better_r2 = final_file_metrics["r2"] > (best_file_r2 + 1e-12)
        tied_r2 = abs(final_file_metrics["r2"] - best_file_r2) <= 1e-12
        better_mae = final_file_metrics["mae"] < (best_file_mae - 1e-12)
        is_better = better_r2 or (tied_r2 and better_mae)

        if is_better:
            best_alpha = float(alpha)
            best_file_r2 = float(final_file_metrics["r2"])
            best_file_mae = float(final_file_metrics["mae"])
            best_window_rows = val_window_rows
            best_file_rows = val_file_rows
            best_window_state = window_model.to_state_dict()
            best_file_state = file_model.to_state_dict()
            best_blend_weight = float(blend_weight)
            best_hgbr_blend_weight = float(hgbr_blend_weight)
            best_hgbr_model = hgbr_model

            checkpoint = {
                "model_type": "ridge_window",
                "ridge_state": best_window_state,
                "ridge_file_state": best_file_state,
                "ridge_blend_weight": best_blend_weight,
                "ridge_hgbr_blend_weight": best_hgbr_blend_weight,
                "hgbr_file_model": best_hgbr_model,
                "val_metrics": {
                    "loss": val_loss,
                    "window": final_window_metrics,
                    "file": final_file_metrics,
                },
                "training_config": vars(args),
                "split_manifest_path": split_manifest_path,
            }
            torch.save(checkpoint, best_checkpoint_path)
            _save_jsonl(str(best_val_predictions_path), best_window_rows)
            print(f"Saved new best ridge checkpoint (alpha={alpha:g}) to {best_checkpoint_path}")

    if best_window_state is None or best_alpha is None:
        raise RuntimeError("Ridge training did not produce a valid model")

    run_summary = {
        "best_alpha": best_alpha,
        "best_blend_weight": float(best_blend_weight),
        "best_hgbr_blend_weight": float(best_hgbr_blend_weight),
        "ridge_use_hgbr_blend": bool(args.ridge_use_hgbr_blend),
        "best_val_file_r2": best_file_r2,
        "best_val_file_mae": best_file_mae,
        "num_alpha_trials": len(alpha_grid),
        "best_val_num_windows": len(best_window_rows),
        "best_val_num_files": len(best_file_rows),
    }
    return history, run_summary


def _train_opentslm_backend(
    args: argparse.Namespace,
    *,
    device: str,
    output_dir: Path,
    split_manifest_path: str,
    train_dataset: FRDAMFARSDataset,
    val_dataset: FRDAMFARSDataset,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=_build_collate_fn(args.encoder_patch_size),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(1, min(args.batch_size, 8)),
        shuffle=False,
        collate_fn=_build_collate_fn(args.encoder_patch_size),
    )

    if len(train_loader) == 0:
        raise RuntimeError("Training loader is empty")

    model = OpenTSLMRegressionSP(
        llm_id=args.llm_id,
        device=device,
        encoder_patch_size=args.encoder_patch_size,
        encoder_max_patches=args.encoder_max_patches,
    )

    target_mean, target_std = _compute_target_stats(train_dataset) if args.target_normalize else (0.0, 1.0)

    param_groups = [
        {
            "params": list(model.encoder.parameters()),
            "lr": args.lr_encoder,
            "weight_decay": args.weight_decay,
        },
        {
            "params": list(model.projector.projector.parameters()),
            "lr": args.lr_projector,
            "weight_decay": args.weight_decay,
        },
        {
            "params": list(model.regression_head.parameters()),
            "lr": args.lr_regression_head,
            "weight_decay": args.weight_decay,
        },
    ]

    optimizer = AdamW(param_groups)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(args.warmup_frac * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    history: List[Dict[str, Any]] = []
    best_epoch = -1
    best_file_r2 = -float("inf")
    best_file_mae = float("inf")
    epochs_without_improvement = 0

    best_checkpoint_path = output_dir / "best_model.pt"
    best_val_predictions_path = output_dir / "val_predictions.jsonl"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_losses: List[float] = []

        for batch in tqdm(train_loader, desc=f"Train epoch {epoch}/{args.epochs}"):
            optimizer.zero_grad()
            preds_norm = model.predict_batch(batch)
            targets_raw = torch.tensor(
                [float(sample["target"]) for sample in batch],
                dtype=torch.float32,
                device=model.device,
            )
            targets_norm = (targets_raw - target_mean) / target_std
            if args.file_balanced_loss:
                weights = torch.tensor(
                    [1.0 / max(float(sample["num_windows_for_file"]), 1.0) for sample in batch],
                    dtype=torch.float32,
                    device=model.device,
                )
            else:
                weights = None

            loss = _compute_weighted_loss_from_preds(preds_norm, targets_norm, args.loss, sample_weights=weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            running_losses.append(float(loss.item()))

        train_loss = float(np.mean(running_losses)) if running_losses else float("nan")

        val_metrics, val_window_rows, _ = _evaluate_opentslm(
            model,
            val_loader,
            loss_type=args.loss,
            target_mean=target_mean,
            target_std=target_std,
            use_file_balanced_weights=args.file_balanced_loss,
        )

        epoch_summary = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_window_r2": val_metrics["window"]["r2"],
            "val_window_mae": val_metrics["window"]["mae"],
            "val_window_rmse": val_metrics["window"]["rmse"],
            "val_file_r2": val_metrics["file"]["r2"],
            "val_file_mae": val_metrics["file"]["mae"],
            "val_file_rmse": val_metrics["file"]["rmse"],
            "lr_encoder": optimizer.param_groups[0]["lr"],
            "lr_projector": optimizer.param_groups[1]["lr"],
            "lr_regression_head": optimizer.param_groups[2]["lr"],
        }
        history.append(epoch_summary)

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f}, "
            f"val_file_r2={val_metrics['file']['r2']:.4f}, val_file_mae={val_metrics['file']['mae']:.4f}, "
            f"val_window_r2={val_metrics['window']['r2']:.4f}"
        )

        better_r2 = val_metrics["file"]["r2"] > (best_file_r2 + 1e-12)
        tied_r2 = abs(val_metrics["file"]["r2"] - best_file_r2) <= 1e-12
        better_mae = val_metrics["file"]["mae"] < (best_file_mae - 1e-12)
        is_better = better_r2 or (tied_r2 and better_mae)

        if is_better:
            best_epoch = epoch
            best_file_r2 = float(val_metrics["file"]["r2"])
            best_file_mae = float(val_metrics["file"]["mae"])
            epochs_without_improvement = 0

            model.store_to_file(
                str(best_checkpoint_path),
                extra_state={
                    "model_type": "opentslm",
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "target_stats": {
                        "mean": float(target_mean),
                        "std": float(target_std),
                        "normalized": bool(args.target_normalize),
                    },
                    "training_config": vars(args),
                    "split_manifest_path": split_manifest_path,
                },
            )
            _save_jsonl(str(best_val_predictions_path), val_window_rows)
            print(f"Saved new best checkpoint at epoch {epoch} to {best_checkpoint_path}")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement}/{args.patience} epoch(s)")

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    run_summary = {
        "best_epoch": best_epoch,
        "best_val_file_r2": best_file_r2,
        "best_val_file_mae": best_file_mae,
        "target_mean": float(target_mean),
        "target_std": float(target_std),
    }
    return history, run_summary


def main():
    parser = argparse.ArgumentParser(description="Train FRDA mFARS regression model")
    parser.add_argument("--metadata-csv", type=str, required=True)
    parser.add_argument("--json-root", type=str, required=True)
    parser.add_argument("--target", type=str, default="mFARS")
    parser.add_argument("--split-manifest", type=str, default=None)
    parser.add_argument(
        "--group-id-strategy",
        choices=["auto", "patient_id", "filename_prefix", "filename_full"],
        default="auto",
        help="How to derive patient/group IDs when creating split manifests.",
    )
    parser.add_argument(
        "--filename-group-prefix-len",
        type=int,
        default=12,
        help="Prefix length used when --group-id-strategy includes filename_prefix logic.",
    )

    parser.add_argument("--model-backend", choices=["ridge_window", "opentslm"], default="ridge_window")
    parser.add_argument("--llm-id", type=str, default="/home/ben/pretrained/gemma-3-270m-it")
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--warmup-frac", type=float, default=0.03)

    parser.add_argument("--lr-encoder", type=float, default=2e-4)
    parser.add_argument("--lr-projector", type=float, default=1e-4)
    parser.add_argument("--lr-regression-head", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--loss", choices=["huber", "mse"], default="huber")
    parser.add_argument("--encoder-patch-size", type=int, default=50)
    parser.add_argument("--encoder-max-patches", type=int, default=256)

    parser.add_argument("--ridge-alpha-grid", type=str, default="0.01,0.1,1,3,10,30,100,300,1000")
    parser.add_argument("--ridge-include-test-id", dest="ridge_include_test_id", action="store_true")
    parser.add_argument("--ridge-no-test-id", dest="ridge_include_test_id", action="store_false")
    parser.set_defaults(ridge_include_test_id=True)
    parser.add_argument("--ridge-use-hgbr-blend", dest="ridge_use_hgbr_blend", action="store_true")
    parser.add_argument("--ridge-no-hgbr-blend", dest="ridge_use_hgbr_blend", action="store_false")
    parser.set_defaults(ridge_use_hgbr_blend=True)
    parser.add_argument("--hgbr-learning-rate", type=float, default=0.1)
    parser.add_argument("--hgbr-max-leaf-nodes", type=int, default=15)
    parser.add_argument("--hgbr-max-depth", type=int, default=3)
    parser.add_argument("--hgbr-l2", type=float, default=0.0)

    parser.add_argument("--file-balanced-loss", dest="file_balanced_loss", action="store_true")
    parser.add_argument("--no-file-balanced-loss", dest="file_balanced_loss", action="store_false")
    parser.set_defaults(file_balanced_loss=True)

    parser.add_argument("--target-normalize", dest="target_normalize", action="store_true")
    parser.add_argument("--no-target-normalize", dest="target_normalize", action="store_false")
    parser.set_defaults(target_normalize=True)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)

    parser.add_argument("--window-size", type=int, default=3000)
    parser.add_argument("--stride", type=int, default=1500)
    parser.add_argument("--median-kernel-size", type=int, default=5)
    parser.add_argument("--normalize", dest="normalize", action="store_true")
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")
    parser.set_defaults(normalize=None)
    parser.add_argument("--no-upsample-test3", action="store_true", default=False)

    parser.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = _get_device(args.device)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = f"{args.model_backend}_{_sanitize_llm_id(args.llm_id)}"
        output_dir = Path("results") / "mfars_regression" / f"{suffix}_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)

    split_manifest_path = args.split_manifest
    if split_manifest_path is None:
        split_manifest_path = str(output_dir / "split_manifest.json")

    if not os.path.exists(split_manifest_path):
        create_split_manifest(
            args.metadata_csv,
            args.json_root,
            output_path=split_manifest_path,
            target_col=args.target,
            seed=args.seed,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            group_id_strategy=args.group_id_strategy,
            filename_group_prefix_len=args.filename_group_prefix_len,
        )

    if args.normalize is None:
        resolved_normalize = False if args.model_backend == "ridge_window" else True
    else:
        resolved_normalize = bool(args.normalize)
    args.resolved_normalize = resolved_normalize

    train_dataset = FRDAMFARSDataset(
        "train",
        split_manifest_path,
        window_size=args.window_size,
        stride=args.stride,
        median_kernel_size=args.median_kernel_size,
        normalize=resolved_normalize,
        upsample_test3=not args.no_upsample_test3,
        max_samples=args.max_samples,
    )
    val_dataset = FRDAMFARSDataset(
        "validation",
        split_manifest_path,
        window_size=args.window_size,
        stride=args.stride,
        median_kernel_size=args.median_kernel_size,
        normalize=resolved_normalize,
        upsample_test3=not args.no_upsample_test3,
        max_samples=args.max_samples,
    )

    if args.model_backend == "ridge_window":
        history, run_summary = _train_ridge_backend(
            args,
            output_dir=output_dir,
            split_manifest_path=split_manifest_path,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
        )
    else:
        history, run_summary = _train_opentslm_backend(
            args,
            device=device,
            output_dir=output_dir,
            split_manifest_path=split_manifest_path,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
        )

    run_config = {
        **vars(args),
        "resolved_normalize": resolved_normalize,
        "device": device,
        "output_dir": str(output_dir),
        "split_manifest_path": split_manifest_path,
        **run_summary,
    }

    _save_json(str(output_dir / "run_config.json"), run_config)
    _save_json(str(output_dir / "train_history.json"), history)

    print("Training complete")
    print(f"Output directory: {output_dir}")
    if "best_val_file_r2" in run_summary:
        print(f"Best val file-level R2: {run_summary['best_val_file_r2']:.6f}")
    if "best_val_file_mae" in run_summary:
        print(f"Best val file-level MAE: {run_summary['best_val_file_mae']:.6f}")
    if "best_alpha" in run_summary:
        print(f"Best alpha: {run_summary['best_alpha']}")
    if "best_epoch" in run_summary:
        print(f"Best epoch: {run_summary['best_epoch']}")


if __name__ == "__main__":
    main()
