#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compute device-stratified test metrics from existing prediction files.

Reads test_predictions_file.jsonl from every completed experiment under
results/mfars_experiments/, groups by device (test_id 1=cup, 2=spoon,
3=pendant), and computes R², MAE, RMSE, Pearson r per device per model.

Outputs:
  results/device_stratified_results.json  — machine-readable
  results/device_stratified_table.txt     — human-readable table
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr

DEVICE_NAMES = {1: "cup", 2: "spoon", 3: "pendant"}
RESULTS_DIR = Path("results/mfars_experiments")
OUT_JSON = Path("results/device_stratified_results.json")
OUT_TABLE = Path("results/device_stratified_table.txt")


def metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    if len(preds) < 2:
        return {"r2": None, "mae": None, "rmse": None, "pearson_r": None, "n": len(preds)}
    mse = float(np.mean((preds - targets) ** 2))
    mae = float(np.mean(np.abs(preds - targets)))
    rmse = float(np.sqrt(mse))
    ss_res = float(np.sum((targets - preds) ** 2))
    ss_tot = float(np.sum((targets - np.mean(targets)) ** 2))
    r2 = 0.0 if ss_tot <= 0 else float(1 - ss_res / ss_tot)
    pr, _ = pearsonr(targets.astype(float), preds.astype(float))
    return {"r2": round(r2, 4), "mae": round(mae, 4), "rmse": round(rmse, 4),
            "pearson_r": round(float(pr), 4), "n": len(preds)}


def load_predictions(run_dir: Path) -> list[dict] | None:
    pred_file = run_dir / "test_results" / "test_predictions_file.jsonl"
    if not pred_file.exists():
        return None
    rows = [json.loads(l) for l in pred_file.read_text().splitlines() if l.strip()]
    return rows


def main():
    all_results = {}

    for run_dir in sorted(RESULTS_DIR.iterdir()):
        if not run_dir.is_dir() or run_dir.name == "shared_split_manifest.json":
            continue
        rows = load_predictions(run_dir)
        if not rows:
            continue

        run_name = run_dir.name
        by_device: dict[int, list[dict]] = defaultdict(list)
        for r in rows:
            tid = int(r.get("test_id", -1))
            by_device[tid].append(r)

        device_metrics = {}
        # Overall (all devices)
        all_preds = np.array([r["prediction"] for r in rows])
        all_targets = np.array([r["target"] for r in rows])
        device_metrics["all"] = metrics(all_preds, all_targets)

        for tid, device_rows in sorted(by_device.items()):
            name = DEVICE_NAMES.get(tid, f"test_id_{tid}")
            preds = np.array([r["prediction"] for r in device_rows])
            targets = np.array([r["target"] for r in device_rows])
            device_metrics[name] = metrics(preds, targets)

        all_results[run_name] = device_metrics

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(all_results, indent=2))
    print(f"Written: {OUT_JSON}")

    # Human-readable table
    devices = ["all", "cup", "spoon", "pendant"]
    metric_key = "r2"
    pearson_key = "pearson_r"
    lines = []
    header = f"{'Run':<45}" + "".join(f"  {d+' R²':>12}" for d in devices) + \
             "".join(f"  {d+' r':>11}" for d in devices)
    lines.append(header)
    lines.append("-" * len(header))

    for run_name, dm in all_results.items():
        row = f"{run_name:<45}"
        for d in devices:
            v = dm.get(d, {}).get(metric_key)
            row += f"  {f'{v:.4f}' if v is not None else 'N/A':>12}"
        for d in devices:
            v = dm.get(d, {}).get(pearson_key)
            row += f"  {f'{v:.4f}' if v is not None else 'N/A':>11}"
        lines.append(row)

    table_str = "\n".join(lines)
    OUT_TABLE.write_text(table_str)
    print(f"Written: {OUT_TABLE}")
    print()
    print(table_str)


if __name__ == "__main__":
    main()
