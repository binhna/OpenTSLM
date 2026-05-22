# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""Evaluate FRDA mFARS regression checkpoints on the test split."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from scipy.stats import kurtosis, pearsonr, skew
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from opentslm.model.llm.OpenTSLMRegressionSP import OpenTSLMRegressionSP
from opentslm.model.regression.ridge_window import RidgeWindowRegressor
from opentslm.time_series_datasets.frda.FRDAMFARSDataset import FRDAMFARSDataset
from opentslm.time_series_datasets.frda.frda_loader import (
    create_split_manifest,
    create_split_manifest_from_split_csv,
)
from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate


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
    mean_target = float(np.mean(targets))
    ss_res = float(np.sum((targets - preds) ** 2))
    ss_tot = float(np.sum((targets - mean_target) ** 2))
    r2 = 0.0 if ss_tot <= 0.0 else float(1.0 - ss_res / ss_tot)
    if preds.size >= 2:
        pr, pp = pearsonr(targets.astype(np.float64), preds.astype(np.float64))
        pearson_r = float(pr)
        pearson_p = float(pp)
    else:
        pearson_r = float("nan")
        pearson_p = float("nan")
    return {"r2": r2, "mae": mae, "rmse": rmse, "pearson_r": pearson_r, "pearson_p": pearson_p}


def _build_collate_fn(patch_size: int):
    def _collate(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return extend_time_series_to_match_patch_size_and_aggregate(batch, patch_size=patch_size)

    return _collate


def _window_matrix_from_sample(sample: Dict[str, Any]) -> np.ndarray:
    ts = np.asarray(sample["time_series"], dtype=np.float64)
    if ts.ndim != 2:
        raise ValueError(f"Expected time_series with ndim=2, got {ts.shape}")
    if ts.shape[0] == 7:
        return ts.T
    if ts.shape[1] == 7:
        return ts
    raise ValueError(f"Expected 7 channels in one axis, got {ts.shape}")


def _extract_rich_window_features(window: np.ndarray, test_id: int) -> np.ndarray:
    w = np.asarray(window, dtype=np.float64)
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


def _build_ridge_eval_arrays(
    dataset: FRDAMFARSDataset,
    *,
    include_test_id: bool,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    x_rows: List[np.ndarray] = []
    y_rows: List[float] = []
    meta_rows: List[Dict[str, Any]] = []

    for sample in dataset.samples:
        window = _window_matrix_from_sample(sample)
        feat = RidgeWindowRegressor.extract_window_features(
            window,
            test_id=int(sample["test_id"]),
            include_test_id=include_test_id,
        )

        x_rows.append(feat)
        y_rows.append(float(sample["target"]))
        meta_rows.append(
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
        raise RuntimeError("Test dataset produced no samples")

    return np.stack(x_rows, axis=0), np.asarray(y_rows, dtype=np.float64), meta_rows


def _build_rich_eval_arrays(
    dataset: FRDAMFARSDataset,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    rich_rows: List[np.ndarray] = []
    meta_rows: List[Dict[str, Any]] = []

    for sample in dataset.samples:
        window = _window_matrix_from_sample(sample)
        rich_rows.append(_extract_rich_window_features(window, int(sample["test_id"])))
        meta_rows.append(
            {
                "json_path": sample["json_path"],
                "target": float(sample["target"]),
                "num_windows_for_file": int(sample["num_windows_for_file"]),
                "raw_length": int(sample["raw_length"]),
                "processed_length": int(sample["processed_length"]),
            }
        )

    return np.stack(rich_rows, axis=0), meta_rows


def _aggregate_window_features_to_file(
    x: np.ndarray,
    meta_rows: List[Dict[str, Any]],
) -> Tuple[np.ndarray, List[str]]:
    grouped: Dict[str, List[int]] = defaultdict(list)
    for idx, meta in enumerate(meta_rows):
        grouped[meta["json_path"]].append(idx)

    file_keys = sorted(grouped.keys())
    file_x = [np.mean(x[grouped[key]], axis=0) for key in file_keys]
    return np.stack(file_x, axis=0), file_keys


def _aggregate_rich_windows_to_file(
    rich_x: np.ndarray,
    meta_rows: List[Dict[str, Any]],
) -> Tuple[np.ndarray, List[str]]:
    grouped: Dict[str, List[int]] = defaultdict(list)
    file_meta: Dict[str, Dict[str, Any]] = {}
    for idx, meta in enumerate(meta_rows):
        grouped[meta["json_path"]].append(idx)
        file_meta[meta["json_path"]] = meta

    file_keys = sorted(grouped.keys())
    file_x = []
    for key in file_keys:
        idxs = grouped[key]
        win = rich_x[idxs]
        agg = np.concatenate(
            [
                np.mean(win, axis=0),
                np.std(win, axis=0),
                np.min(win, axis=0),
                np.max(win, axis=0),
                np.median(win, axis=0),
                np.asarray(
                    [
                        float(file_meta[key]["num_windows_for_file"]),
                        float(file_meta[key]["raw_length"]),
                        float(file_meta[key]["processed_length"]),
                    ],
                    dtype=np.float64,
                ),
            ],
            axis=0,
        )
        file_x.append(agg)
    return np.stack(file_x, axis=0), file_keys


def _broadcast_file_predictions_to_windows(
    meta_rows: List[Dict[str, Any]],
    file_keys: List[str],
    file_preds: np.ndarray,
) -> np.ndarray:
    pred_map = {k: float(v) for k, v in zip(file_keys, file_preds.tolist())}
    return np.asarray([pred_map[meta["json_path"]] for meta in meta_rows], dtype=np.float64)


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


def _infer_model_settings(
    checkpoint: Dict[str, Any],
    llm_override: str | None,
    patch_override: int | None,
) -> Tuple[str, int]:
    llm_id = llm_override
    encoder_patch_size = patch_override

    training_config = checkpoint.get("training_config", {})
    regression_config = checkpoint.get("regression_config", {})

    if llm_id is None:
        llm_id = training_config.get("llm_id")
    if llm_id is None:
        llm_id = regression_config.get("llm_id")
    if llm_id is None:
        llm_id = "/home/ben/pretrained/gemma-3-270m-it"

    if encoder_patch_size is None:
        encoder_patch_size = training_config.get("encoder_patch_size")
    if encoder_patch_size is None:
        encoder_patch_size = regression_config.get("encoder_patch_size")
    if encoder_patch_size is None:
        encoder_patch_size = 50

    return str(llm_id), int(encoder_patch_size)


def _resolve_preproc(
    args: argparse.Namespace,
    training_config: Dict[str, Any],
    model_type: str,
) -> Tuple[int, int, int, bool, bool]:
    window_size = int(args.window_size if args.window_size is not None else training_config.get("window_size", 3000))
    stride = int(args.stride if args.stride is not None else training_config.get("stride", 1500))
    median_kernel_size = int(
        args.median_kernel_size
        if args.median_kernel_size is not None
        else training_config.get("median_kernel_size", 5)
    )

    if args.normalize is None:
        if "resolved_normalize" in training_config:
            normalize = bool(training_config["resolved_normalize"])
        elif "normalize" in training_config and training_config["normalize"] is not None:
            normalize = bool(training_config["normalize"])
        elif model_type == "ridge_window":
            normalize = False
        else:
            normalize = not bool(training_config.get("no_normalize", False))
    else:
        normalize = bool(args.normalize)

    if args.upsample_test3 is None:
        upsample_test3 = not bool(training_config.get("no_upsample_test3", False))
    else:
        upsample_test3 = bool(args.upsample_test3)

    return window_size, stride, median_kernel_size, normalize, upsample_test3


def _save_json(path: Path, obj: Any):
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _save_jsonl(path: Path, rows: List[Dict[str, Any]]):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Test FRDA mFARS regression model")
    parser.add_argument("--checkpoint", type=str, required=True)

    parser.add_argument("--split-manifest", type=str, default=None)
    parser.add_argument("--split-adults-csv", type=str, default=None,
                        help="Path to split_adults.csv for canonical participant-level split.")
    parser.add_argument("--metadata-csv", type=str, default=None)
    parser.add_argument("--json-root", type=str, default=None)
    parser.add_argument("--target", type=str, default="mfars_total")
    parser.add_argument("--seed", type=int, default=42)
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

    parser.add_argument("--llm-id", type=str, default=None)
    parser.add_argument("--encoder-patch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=4)

    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--median-kernel-size", type=int, default=None)
    parser.add_argument("--normalize", dest="normalize", action="store_true")
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")
    parser.set_defaults(normalize=None)
    parser.add_argument("--upsample-test3", dest="upsample_test3", action="store_true")
    parser.add_argument("--no-upsample-test3", dest="upsample_test3", action="store_false")
    parser.set_defaults(upsample_test3=None)

    parser.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    model_type = str(checkpoint.get("model_type", "opentslm"))
    training_config = checkpoint.get("training_config", {})

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = checkpoint_path.parent / "test_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    split_manifest_path = args.split_manifest
    if split_manifest_path is None:
        if args.metadata_csv is None or args.json_root is None:
            raise ValueError(
                "Provide --split-manifest, or provide both --metadata-csv and --json-root to create one"
            )
        split_manifest_path = str(output_dir / "split_manifest.json")
        if not Path(split_manifest_path).exists():
            if args.split_adults_csv is not None:
                create_split_manifest_from_split_csv(
                    args.metadata_csv,
                    args.split_adults_csv,
                    args.json_root,
                    output_path=split_manifest_path,
                    target_col=args.target,
                )
            else:
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

    window_size, stride, median_kernel_size, normalize, upsample_test3 = _resolve_preproc(
        args,
        training_config,
        model_type,
    )
    test_dataset = FRDAMFARSDataset(
        "test",
        split_manifest_path,
        window_size=window_size,
        stride=stride,
        median_kernel_size=median_kernel_size,
        normalize=normalize,
        upsample_test3=upsample_test3,
    )

    device = _get_device(args.device)
    window_rows: List[Dict[str, Any]] = []

    if model_type == "ridge_window":
        if "ridge_state" not in checkpoint:
            raise RuntimeError("Ridge checkpoint missing 'ridge_state'")
        ridge = RidgeWindowRegressor.from_state_dict(checkpoint["ridge_state"])

        x_test, _, meta_rows = _build_ridge_eval_arrays(
            test_dataset,
            include_test_id=bool(ridge.include_test_id),
        )
        preds = ridge.predict(x_test).astype(np.float64)

        if "ridge_file_state" in checkpoint:
            file_model = RidgeWindowRegressor.from_state_dict(checkpoint["ridge_file_state"])
            blend_weight = float(checkpoint.get("ridge_blend_weight", 1.0))
            x_file, file_keys = _aggregate_window_features_to_file(x_test, meta_rows)
            file_preds = file_model.predict(x_file)
            file_as_window_preds = _broadcast_file_predictions_to_windows(meta_rows, file_keys, file_preds)
            preds = blend_weight * preds + (1.0 - blend_weight) * file_as_window_preds

        if "hgbr_file_model" in checkpoint:
            hgbr_model = checkpoint["hgbr_file_model"]
            hgbr_blend_weight = float(checkpoint.get("ridge_hgbr_blend_weight", 1.0))
            rich_x, rich_meta = _build_rich_eval_arrays(test_dataset)
            rich_file_x, rich_file_keys = _aggregate_rich_windows_to_file(rich_x, rich_meta)
            hgbr_file_preds = np.asarray(hgbr_model.predict(rich_file_x), dtype=np.float64)
            hgbr_as_window = _broadcast_file_predictions_to_windows(meta_rows, rich_file_keys, hgbr_file_preds)
            preds = hgbr_blend_weight * preds + (1.0 - hgbr_blend_weight) * hgbr_as_window

        for meta, pred in zip(meta_rows, preds.tolist()):
            row = dict(meta)
            row["prediction"] = float(pred)
            window_rows.append(row)
    else:
        llm_id, encoder_patch_size = _infer_model_settings(
            checkpoint,
            args.llm_id,
            args.encoder_patch_size,
        )
        target_stats = checkpoint.get("target_stats", {})
        target_mean = float(target_stats.get("mean", 0.0))
        target_std = float(target_stats.get("std", 1.0))
        if target_std <= 0.0:
            target_std = 1.0

        test_loader = DataLoader(
            test_dataset,
            batch_size=max(1, args.batch_size),
            shuffle=False,
            collate_fn=_build_collate_fn(encoder_patch_size),
        )

        model = OpenTSLMRegressionSP(
            llm_id=llm_id,
            device=device,
            encoder_patch_size=encoder_patch_size,
        )
        model.load_from_file(str(checkpoint_path))
        model.eval()

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                preds_norm = model.predict_batch(batch).detach().cpu().numpy()
                preds = preds_norm * target_std + target_mean
                for sample, pred in zip(batch, preds.tolist()):
                    window_rows.append(
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
                            "prediction": float(pred),
                            "target": float(sample["target"]),
                        }
                    )

    if not window_rows:
        raise RuntimeError("No test predictions were generated")

    window_preds = np.asarray([r["prediction"] for r in window_rows], dtype=np.float64)
    window_targets = np.asarray([r["target"] for r in window_rows], dtype=np.float64)
    window_metrics = _metrics(window_preds, window_targets)

    file_rows, file_metrics = _aggregate_file_rows(window_rows)

    metrics = {
        "window": window_metrics,
        "file": file_metrics,
        "counts": {
            "window_samples": len(window_rows),
            "file_samples": len(file_rows),
        },
        "primary_metric": "r2",
        "model_type": model_type,
    }

    _save_jsonl(output_dir / "test_predictions_window.jsonl", window_rows)
    _save_jsonl(output_dir / "test_predictions_file.jsonl", file_rows)
    _save_json(output_dir / "test_metrics.json", metrics)

    summary_lines = [
        "FRDA mFARS Test Summary",
        "",
        f"Checkpoint: {checkpoint_path}",
        f"Model type: {model_type}",
        f"Split manifest: {split_manifest_path}",
        "",
        "Window-level metrics:",
        f"  R2:   {window_metrics['r2']:.6f}",
        f"  MAE:  {window_metrics['mae']:.6f}",
        f"  RMSE: {window_metrics['rmse']:.6f}",
        f"  Pearson r: {window_metrics['pearson_r']:.6f}",
        f"  Pearson p: {window_metrics['pearson_p']:.6g}",
        "",
        "File-level metrics:",
        f"  R2:   {file_metrics['r2']:.6f}",
        f"  MAE:  {file_metrics['mae']:.6f}",
        f"  RMSE: {file_metrics['rmse']:.6f}",
        f"  Pearson r: {file_metrics['pearson_r']:.6f}",
        f"  Pearson p: {file_metrics['pearson_p']:.6g}",
        "",
        f"Window samples: {len(window_rows)}",
        f"File samples: {len(file_rows)}",
    ]
    (output_dir / "test_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    print("Test evaluation complete")
    print(f"Output directory: {output_dir}")
    print(f"Window-level R2: {window_metrics['r2']:.6f}")
    print(f"File-level R2: {file_metrics['r2']:.6f}")


if __name__ == "__main__":
    main()
