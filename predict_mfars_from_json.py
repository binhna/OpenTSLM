# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""Predict mFARS from a single AIM JSON recording."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from scipy.stats import kurtosis, skew
from torch.utils.data import DataLoader

from opentslm.model.llm.OpenTSLMRegressionSP import OpenTSLMRegressionSP
from opentslm.model.regression.ridge_window import RidgeWindowRegressor
from opentslm.time_series_datasets.frda.FRDAMFARSDataset import (
    DEFAULT_POST_PROMPT,
    DEFAULT_PRE_PROMPT,
)
from opentslm.time_series_datasets.frda.frda_loader import (
    build_windowed_signal_from_json,
    window_to_model_inputs,
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


def _build_collate_fn(patch_size: int):
    def _collate(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return extend_time_series_to_match_patch_size_and_aggregate(batch, patch_size=patch_size)

    return _collate


def _build_window_samples(
    json_path: str,
    *,
    window_size: int,
    stride: int,
    median_kernel_size: int,
    normalize: bool,
    upsample_test3: bool,
    pre_prompt: str,
    post_prompt: str,
) -> List[Dict[str, Any]]:
    windows, meta = build_windowed_signal_from_json(
        json_path,
        window_size=window_size,
        stride=stride,
        median_kernel_size=median_kernel_size,
        normalize=normalize,
        upsample_test3=upsample_test3,
    )

    samples: List[Dict[str, Any]] = []
    for idx, window in enumerate(windows):
        ts_text, ts_data = window_to_model_inputs(window)
        samples.append(
            {
                "pre_prompt": pre_prompt,
                "time_series_text": ts_text,
                "time_series": ts_data,
                # dummy target not used in predict path
                "target": 0.0,
                "post_prompt": post_prompt,
                "json_path": json_path,
                "window_index": idx,
                "num_windows_for_file": len(windows),
                "test_id": int(meta["test_id"]),
                "raw_length": int(meta["raw_length"]),
                "processed_length": int(meta["processed_length"]),
            }
        )
    return samples


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


def _aggregate_window_features_to_file(x: np.ndarray, samples: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[str]]:
    grouped: Dict[str, List[int]] = {}
    for idx, sample in enumerate(samples):
        grouped.setdefault(sample["json_path"], []).append(idx)
    file_keys = sorted(grouped.keys())
    file_x = [np.mean(x[grouped[key]], axis=0) for key in file_keys]
    return np.stack(file_x, axis=0), file_keys


def _aggregate_rich_windows_to_file(
    rich_x: np.ndarray,
    samples: List[Dict[str, Any]],
) -> Tuple[np.ndarray, List[str]]:
    grouped: Dict[str, List[int]] = {}
    for idx, sample in enumerate(samples):
        grouped.setdefault(sample["json_path"], []).append(idx)

    file_keys = sorted(grouped.keys())
    file_x = []
    for key in file_keys:
        idxs = grouped[key]
        win = rich_x[idxs]
        first = samples[idxs[0]]
        agg = np.concatenate(
            [
                np.mean(win, axis=0),
                np.std(win, axis=0),
                np.min(win, axis=0),
                np.max(win, axis=0),
                np.median(win, axis=0),
                np.asarray(
                    [
                        float(first["num_windows_for_file"]),
                        float(first["raw_length"]),
                        float(first["processed_length"]),
                    ],
                    dtype=np.float64,
                ),
            ],
            axis=0,
        )
        file_x.append(agg)
    return np.stack(file_x, axis=0), file_keys


def _broadcast_file_predictions_to_windows(
    samples: List[Dict[str, Any]],
    file_keys: List[str],
    file_preds: np.ndarray,
) -> np.ndarray:
    pred_map = {k: float(v) for k, v in zip(file_keys, file_preds.tolist())}
    return np.asarray([pred_map[s["json_path"]] for s in samples], dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(description="Predict mFARS from a single AIM JSON recording")
    parser.add_argument("--json-file", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--llm-id", type=str, default=None)
    parser.add_argument("--encoder-patch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=8)

    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--median-kernel-size", type=int, default=None)
    parser.add_argument("--normalize", dest="normalize", action="store_true")
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")
    parser.set_defaults(normalize=None)
    parser.add_argument("--upsample-test3", dest="upsample_test3", action="store_true")
    parser.add_argument("--no-upsample-test3", dest="upsample_test3", action="store_false")
    parser.set_defaults(upsample_test3=None)

    parser.add_argument("--pre-prompt", type=str, default=DEFAULT_PRE_PROMPT)
    parser.add_argument("--post-prompt", type=str, default=DEFAULT_POST_PROMPT)
    parser.add_argument("--output-json", type=str, default=None)

    args = parser.parse_args()

    json_path = Path(args.json_file)
    checkpoint_path = Path(args.checkpoint)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    model_type = str(checkpoint.get("model_type", "opentslm"))
    training_config = checkpoint.get("training_config", {})
    window_size, stride, median_kernel_size, normalize, upsample_test3 = _resolve_preproc(
        args,
        training_config,
        model_type,
    )

    samples = _build_window_samples(
        str(json_path),
        window_size=window_size,
        stride=stride,
        median_kernel_size=median_kernel_size,
        normalize=normalize,
        upsample_test3=upsample_test3,
        pre_prompt=args.pre_prompt,
        post_prompt=args.post_prompt,
    )

    if not samples:
        raise RuntimeError(f"No valid windows extracted from {json_path}")

    window_predictions: List[float] = []
    device = _get_device(args.device)

    if model_type == "ridge_window":
        if "ridge_state" not in checkpoint:
            raise RuntimeError("Ridge checkpoint missing 'ridge_state'")
        ridge = RidgeWindowRegressor.from_state_dict(checkpoint["ridge_state"])

        x_rows = []
        for sample in samples:
            window = _window_matrix_from_sample(sample)
            feat = RidgeWindowRegressor.extract_window_features(
                window,
                test_id=int(sample["test_id"]),
                include_test_id=bool(ridge.include_test_id),
            )
            x_rows.append(feat)
        x = np.stack(x_rows, axis=0)
        preds = ridge.predict(x).astype(np.float64)

        if "ridge_file_state" in checkpoint:
            file_model = RidgeWindowRegressor.from_state_dict(checkpoint["ridge_file_state"])
            blend_weight = float(checkpoint.get("ridge_blend_weight", 1.0))
            x_file, file_keys = _aggregate_window_features_to_file(x, samples)
            file_preds = file_model.predict(x_file)
            file_as_window_preds = _broadcast_file_predictions_to_windows(samples, file_keys, file_preds)
            preds = blend_weight * preds + (1.0 - blend_weight) * file_as_window_preds

        if "hgbr_file_model" in checkpoint:
            hgbr_model = checkpoint["hgbr_file_model"]
            hgbr_blend_weight = float(checkpoint.get("ridge_hgbr_blend_weight", 1.0))
            rich_x = np.stack(
                [
                    _extract_rich_window_features(_window_matrix_from_sample(sample), int(sample["test_id"]))
                    for sample in samples
                ],
                axis=0,
            )
            rich_file_x, rich_file_keys = _aggregate_rich_windows_to_file(rich_x, samples)
            hgbr_file_preds = np.asarray(hgbr_model.predict(rich_file_x), dtype=np.float64)
            hgbr_as_window = _broadcast_file_predictions_to_windows(samples, rich_file_keys, hgbr_file_preds)
            preds = hgbr_blend_weight * preds + (1.0 - hgbr_blend_weight) * hgbr_as_window

        window_predictions = preds.tolist()
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

        loader = DataLoader(
            samples,
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
            for batch in loader:
                preds_norm = model.predict_batch(batch).detach().cpu().numpy()
                preds = preds_norm * target_std + target_mean
                window_predictions.extend(preds.tolist())

    pred_arr = np.asarray(window_predictions, dtype=np.float64)
    final_pred = float(np.mean(pred_arr))

    first = samples[0]
    result = {
        "json_file": str(json_path),
        "model_type": model_type,
        "predicted_mfars": final_pred,
        "window_predictions": [float(v) for v in window_predictions],
        "prediction_std": float(np.std(pred_arr)),
        "num_windows": int(len(window_predictions)),
        "test_id": int(first["test_id"]),
        "raw_length": int(first["raw_length"]),
        "processed_length": int(first["processed_length"]),
    }

    print(json.dumps(result, indent=2))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
