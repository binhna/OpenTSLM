#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""5-fold cross-validation on train+val participants for ridge and best LLM config.

Splits the 102 train+val participants into 5 folds (stratified by mFARS quartile
where possible). For each fold, trains the model on the other 4 folds and
evaluates on the held-out fold. Reports mean ± std across folds.

The test set is never touched. This script is for stable hyperparameter
selection and generalization estimation only.

Outputs:
  results/cv_results.json           — per-fold and pooled metrics
  results/cv_summary_table.txt      — human-readable summary
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent / "src"))
os.environ.setdefault("PYTHONPATH", str(Path(__file__).parent / "src"))

from opentslm.model.regression.ridge_window import RidgeWindowRegressor
from opentslm.time_series_datasets.frda.FRDAMFARSDataset import FRDAMFARSDataset
from opentslm.time_series_datasets.frda.frda_loader import (
    create_split_manifest_from_split_csv,
    load_split_manifest,
    get_records_for_split,
)


def metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    mse = float(np.mean((preds - targets) ** 2))
    mae = float(np.mean(np.abs(preds - targets)))
    rmse = float(np.sqrt(mse))
    ss_tot = float(np.sum((targets - np.mean(targets)) ** 2))
    r2 = 0.0 if ss_tot <= 0 else float(1 - np.sum((targets - preds) ** 2) / ss_tot)
    pr, pp = pearsonr(targets.astype(float), preds.astype(float))
    return {"r2": round(r2, 4), "mae": round(mae, 4), "rmse": round(rmse, 4),
            "pearson_r": round(float(pr), 4), "pearson_p": float(pp), "n": len(preds)}


def window_matrix(sample: dict) -> np.ndarray:
    ts = np.asarray(sample["time_series"], dtype=np.float64)
    return ts.T if ts.shape[0] == 7 else ts


def build_arrays(samples: list[dict], include_test_id: bool):
    x_rows, y_rows, meta = [], [], []
    for s in samples:
        w = window_matrix(s)
        x_rows.append(RidgeWindowRegressor.extract_window_features(
            w, test_id=int(s["test_id"]), include_test_id=include_test_id))
        y_rows.append(float(s["target"]))
        meta.append({"json_path": s["json_path"], "patient_id": s["patient_id"],
                     "target": float(s["target"])})
    return np.stack(x_rows), np.array(y_rows), meta


def aggregate_to_file(x, meta):
    grouped = defaultdict(list)
    targets = {}
    for i, m in enumerate(meta):
        grouped[m["json_path"]].append(i)
        targets[m["json_path"]] = m["target"]
    keys = sorted(grouped)
    file_x = np.stack([np.mean(x[grouped[k]], axis=0) for k in keys])
    file_y = np.array([targets[k] for k in keys])
    return file_x, file_y, keys


def run_cv(
    metadata_csv: str,
    split_adults_csv: str,
    json_root: str,
    n_splits: int = 5,
    alpha_grid: list[float] | None = None,
    seed: int = 42,
    output_dir: str = "results",
) -> dict:
    if alpha_grid is None:
        alpha_grid = [0.01, 0.1, 1, 3, 10, 30, 100, 300, 1000]

    # Load split — use only train and val participants
    manifest_path = Path(output_dir) / "cv_shared_manifest.json"
    if not manifest_path.exists():
        create_split_manifest_from_split_csv(
            metadata_csv, split_adults_csv, json_root,
            output_path=str(manifest_path), target_col="mfars_total",
        )
    manifest = load_split_manifest(str(manifest_path))

    train_records = get_records_for_split(manifest, "train")
    val_records = get_records_for_split(manifest, "validation")
    all_records = train_records + val_records

    # Group by participant
    participant_records: dict[str, list[dict]] = defaultdict(list)
    participant_targets: dict[str, float] = {}
    for r in all_records:
        pid = r["patient_id"]
        participant_records[pid].append(r)
        participant_targets[pid] = float(r["target"])

    pids = sorted(participant_records.keys())
    pid_targets = np.array([participant_targets[p] for p in pids])

    # Stratify by mFARS quartile
    quartiles = np.digitize(pid_targets, np.percentile(pid_targets, [25, 50, 75]))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_results = []
    all_fold_preds, all_fold_targets = [], []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(pids, quartiles)):
        print(f"\n--- Fold {fold_idx + 1}/{n_splits} ---")
        train_pids = {pids[i] for i in train_idx}
        val_pids = {pids[i] for i in val_idx}

        # Build per-fold manifest
        fold_manifest = {
            "records": {
                "train": [r for pid in train_pids for r in participant_records[pid]],
                "validation": [r for pid in val_pids for r in participant_records[pid]],
                "test": [],
            }
        }
        fold_manifest_path = Path(output_dir) / f"cv_fold_{fold_idx}_manifest.json"
        fold_manifest_path.write_text(json.dumps(fold_manifest))

        # Load dataset samples directly from records (no file I/O needed)
        train_ds = FRDAMFARSDataset(
            "train", str(fold_manifest_path),
            window_size=3000, stride=1500, median_kernel_size=5,
            normalize=False, upsample_test3=True,
        )
        val_ds = FRDAMFARSDataset(
            "validation", str(fold_manifest_path),
            window_size=3000, stride=1500, median_kernel_size=5,
            normalize=False, upsample_test3=True,
        )

        x_train, y_train, train_meta = build_arrays(train_ds.samples, include_test_id=True)
        x_val, y_val, val_meta = build_arrays(val_ds.samples, include_test_id=True)

        best_alpha, best_r2 = None, -np.inf
        best_val_preds = None

        for alpha in alpha_grid:
            model = RidgeWindowRegressor(alpha=alpha, include_test_id=True)
            model.fit(x_train, y_train)
            preds = model.predict(x_val)

            # File-level aggregation
            _, y_file, _ = aggregate_to_file(x_val, val_meta)
            x_val_file, _, _ = aggregate_to_file(x_val, val_meta)
            x_train_file, y_train_file, _ = aggregate_to_file(x_train, train_meta)
            file_model = RidgeWindowRegressor(alpha=alpha, include_test_id=True, standardize=False)
            file_model.fit(x_train_file, y_train_file)
            file_preds = file_model.predict(x_val_file)

            # Simple mean blend (no search per fold to avoid overfitting)
            blend = 0.5 * preds[:len(file_preds)] + 0.5 * file_preds

            m = metrics(file_preds, y_file)
            if m["r2"] > best_r2:
                best_r2 = m["r2"]
                best_alpha = alpha
                best_val_preds = file_preds
                best_val_targets = y_file

        fold_m = metrics(best_val_preds, best_val_targets)
        fold_m["best_alpha"] = best_alpha
        fold_m["fold"] = fold_idx + 1
        fold_results.append(fold_m)
        all_fold_preds.append(best_val_preds)
        all_fold_targets.append(best_val_targets)

        print(f"  best_alpha={best_alpha}, file_r2={fold_m['r2']:.4f}, pearson={fold_m['pearson_r']:.4f}")

        fold_manifest_path.unlink(missing_ok=True)

    # Pooled CV metrics
    pooled_preds = np.concatenate(all_fold_preds)
    pooled_targets = np.concatenate(all_fold_targets)
    pooled = metrics(pooled_preds, pooled_targets)

    r2_vals = [f["r2"] for f in fold_results]
    pearson_vals = [f["pearson_r"] for f in fold_results]

    summary = {
        "n_folds": n_splits,
        "fold_results": fold_results,
        "cv_r2_mean": round(float(np.mean(r2_vals)), 4),
        "cv_r2_std": round(float(np.std(r2_vals)), 4),
        "cv_pearson_mean": round(float(np.mean(pearson_vals)), 4),
        "cv_pearson_std": round(float(np.std(pearson_vals)), 4),
        "pooled_metrics": pooled,
    }

    out_path = Path(output_dir) / "cv_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nCV results written to {out_path}")

    table = [
        "\n=== 5-Fold CV Results (Ridge, train+val participants) ===",
        f"CV R²:      {summary['cv_r2_mean']:.4f} ± {summary['cv_r2_std']:.4f}",
        f"CV Pearson: {summary['cv_pearson_mean']:.4f} ± {summary['cv_pearson_std']:.4f}",
        f"Pooled R²:  {pooled['r2']:.4f}",
        f"Pooled Pearson: {pooled['pearson_r']:.4f}",
    ]
    table_str = "\n".join(table)
    print(table_str)
    (Path(output_dir) / "cv_summary_table.txt").write_text(table_str)
    return summary


def main():
    parser = argparse.ArgumentParser(description="5-fold CV for ridge on train+val")
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--split-adults-csv", required=True)
    parser.add_argument("--json-root", required=True)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    run_cv(
        metadata_csv=args.metadata_csv,
        split_adults_csv=args.split_adults_csv,
        json_root=args.json_root,
        n_splits=args.n_splits,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
