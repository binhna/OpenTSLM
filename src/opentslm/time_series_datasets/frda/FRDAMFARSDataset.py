# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""PyTorch dataset for FRDA mFARS regression using AIM JSON recordings."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from torch.utils.data import Dataset

from opentslm.time_series_datasets.frda.frda_loader import (
    build_windowed_signal_from_json,
    get_records_for_split,
    load_split_manifest,
    window_to_model_inputs,
)


DEFAULT_PRE_PROMPT = (
    "You are a clinical time-series assistant for Friedreich's Ataxia. "
    "Analyze motion sensor and force signals and infer disease severity."
)

DEFAULT_POST_PROMPT = (
    "Predict the patient's mFARS score as a single numeric value (0-93)."
)


class FRDAMFARSDataset(Dataset):
    """Windowed FRDA regression dataset that emits OpenTSLM-SP compatible samples."""

    def __init__(
        self,
        split: Literal["train", "validation", "test"],
        split_manifest_path: str,
        *,
        pre_prompt: str = DEFAULT_PRE_PROMPT,
        post_prompt: str = DEFAULT_POST_PROMPT,
        window_size: int = 3000,
        stride: int = 1500,
        median_kernel_size: int = 5,
        normalize: bool = True,
        upsample_test3: bool = True,
        max_samples: Optional[int] = None,
    ):
        self.split = split
        self.split_manifest_path = split_manifest_path
        self.pre_prompt = pre_prompt
        self.post_prompt = post_prompt
        self.window_size = window_size
        self.stride = stride
        self.median_kernel_size = median_kernel_size
        self.normalize = normalize
        self.upsample_test3 = upsample_test3

        manifest = load_split_manifest(split_manifest_path)
        records = get_records_for_split(manifest, split)

        self.samples: List[Dict[str, Any]] = []

        for record in records:
            windows, signal_meta = build_windowed_signal_from_json(
                record["json_path"],
                window_size=window_size,
                stride=stride,
                median_kernel_size=median_kernel_size,
                normalize=normalize,
                upsample_test3=upsample_test3,
            )

            total_windows = len(windows)
            for idx, window in enumerate(windows):
                time_series_text, time_series = window_to_model_inputs(window)
                sample = {
                    "pre_prompt": pre_prompt,
                    "time_series_text": time_series_text,
                    "time_series": time_series,
                    "post_prompt": post_prompt,
                    "target": float(record["target"]),
                    "file_name": record["file_name"],
                    "json_path": record["json_path"],
                    "patient_id": record["patient_id"],
                    "group_id": record.get("group_id", record["patient_id"]),
                    "visit_date": record.get("visit_date", ""),
                    "device": record.get("device", ""),
                    "source_column": record.get("source_column", ""),
                    "window_index": idx,
                    "num_windows_for_file": total_windows,
                    "test_id": signal_meta["test_id"],
                    "raw_length": signal_meta["raw_length"],
                    "processed_length": signal_meta["processed_length"],
                }
                self.samples.append(sample)

                if max_samples is not None and len(self.samples) >= max_samples:
                    break

            if max_samples is not None and len(self.samples) >= max_samples:
                break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]
