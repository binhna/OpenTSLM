#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""Tests for FRDA split manifest generation and dataset split safety."""

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from opentslm.time_series_datasets.frda.FRDAMFARSDataset import FRDAMFARSDataset
from opentslm.time_series_datasets.frda.frda_loader import create_split_manifest


def _make_fake_json(path: Path, test_id: int = 2, n: int = 200):
    if test_id == 1:
        rows = [
            [100 + i // 10, 0, 0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i, 0.5 * i, 0.6 * i, 0.01 * i]
            for i in range(n)
        ]
    else:
        rows = [
            [100 + i // (5 if test_id == 3 else 10), 0, 0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i, 0.5 * i, 0.6 * i]
            for i in range(n)
        ]

    payload = {
        "RecordID": path.stem,
        "TestID": test_id,
        "record": rows,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class TestMFARSSplit(unittest.TestCase):
    def test_split_is_patient_disjoint_and_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            json_root = tmp_path / "json"
            json_root.mkdir(parents=True, exist_ok=True)

            rows = []
            for patient_id in range(1, 13):
                # one file per patient is sufficient for split determinism test
                file_name = f"patient_{patient_id:02d}_01.json"
                _make_fake_json(json_root / file_name, test_id=1, n=180)
                rows.append(
                    {
                        "patient ID": patient_id,
                        "date_of_collection": "2025-01-01",
                        "mFARS": float(20 + patient_id),
                        "file_name_01": file_name,
                        "file_name_02": "",
                        "file_name_03": "",
                    }
                )

            metadata_csv = tmp_path / "metadata.csv"
            pd.DataFrame(rows).to_csv(metadata_csv, index=False)

            manifest_a_path = tmp_path / "split_a.json"
            manifest_b_path = tmp_path / "split_b.json"

            manifest_a = create_split_manifest(
                str(metadata_csv),
                str(json_root),
                output_path=str(manifest_a_path),
                target_col="mFARS",
                seed=42,
                train_ratio=0.70,
                val_ratio=0.15,
                test_ratio=0.15,
            )
            manifest_b = create_split_manifest(
                str(metadata_csv),
                str(json_root),
                output_path=str(manifest_b_path),
                target_col="mFARS",
                seed=42,
                train_ratio=0.70,
                val_ratio=0.15,
                test_ratio=0.15,
            )

            train_ids = set(manifest_a["groups"]["train"])
            val_ids = set(manifest_a["groups"]["validation"])
            test_ids = set(manifest_a["groups"]["test"])

            self.assertTrue(train_ids.isdisjoint(val_ids))
            self.assertTrue(train_ids.isdisjoint(test_ids))
            self.assertTrue(val_ids.isdisjoint(test_ids))

            # Deterministic split with same seed.
            self.assertEqual(manifest_a["groups"], manifest_b["groups"])

    def test_dataset_builds_samples_from_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            json_root = tmp_path / "json"
            json_root.mkdir(parents=True, exist_ok=True)

            rows = []
            for patient_id in range(1, 7):
                file_name_01 = f"patient_{patient_id:02d}_01.json"
                file_name_02 = f"patient_{patient_id:02d}_02.json"
                _make_fake_json(json_root / file_name_01, test_id=1, n=220)
                _make_fake_json(json_root / file_name_02, test_id=2, n=220)
                rows.append(
                    {
                        "patient ID": patient_id,
                        "date_of_collection": "2025-01-01",
                        "mFARS": float(30 + patient_id),
                        "file_name_01": file_name_01,
                        "file_name_02": file_name_02,
                        "file_name_03": "",
                    }
                )

            metadata_csv = tmp_path / "metadata.csv"
            pd.DataFrame(rows).to_csv(metadata_csv, index=False)

            manifest_path = tmp_path / "split.json"
            create_split_manifest(
                str(metadata_csv),
                str(json_root),
                output_path=str(manifest_path),
                target_col="mFARS",
                seed=42,
                train_ratio=0.70,
                val_ratio=0.15,
                test_ratio=0.15,
            )

            dataset = FRDAMFARSDataset(
                "train",
                str(manifest_path),
                window_size=128,
                stride=64,
                median_kernel_size=5,
                normalize=True,
                upsample_test3=True,
            )

            self.assertGreater(len(dataset), 0)
            sample = dataset[0]
            self.assertIn("target", sample)
            self.assertIn("time_series", sample)
            self.assertEqual(len(sample["time_series"]), 7)
            self.assertIn("window_index", sample)

    def test_split_supports_filename_device_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            json_root = tmp_path / "json"
            json_root.mkdir(parents=True, exist_ok=True)

            rows = []
            for patient_idx in range(1, 13):
                # 12-digit patient-like prefix + 3-digit trial suffix.
                group_prefix = f"900100{patient_idx:06d}"
                mfars = float(15 + patient_idx)
                for device, suffix in [("cup", "101"), ("spoon", "201"), ("pendant", "301")]:
                    file_name = f"{group_prefix}{suffix}.json"
                    _make_fake_json(json_root / file_name, test_id=1, n=200)
                    rows.append(
                        {
                            "filename": file_name,
                            "device": device,
                            "mFARS": mfars,
                        }
                    )

            metadata_csv = tmp_path / "metadata_filename_device_mfars.csv"
            pd.DataFrame(rows).to_csv(metadata_csv, index=False)

            manifest_path = tmp_path / "split.json"
            manifest = create_split_manifest(
                str(metadata_csv),
                str(json_root),
                output_path=str(manifest_path),
                target_col="mFARS",
                seed=42,
                train_ratio=0.70,
                val_ratio=0.15,
                test_ratio=0.15,
            )

            train_ids = set(manifest["groups"]["train"])
            val_ids = set(manifest["groups"]["validation"])
            test_ids = set(manifest["groups"]["test"])

            self.assertTrue(train_ids.isdisjoint(val_ids))
            self.assertTrue(train_ids.isdisjoint(test_ids))
            self.assertTrue(val_ids.isdisjoint(test_ids))

            # Ensure records carry derived grouping + device metadata.
            train_records = manifest["records"]["train"]
            self.assertGreater(len(train_records), 0)
            rec0 = train_records[0]
            self.assertIn("group_id", rec0)
            self.assertIn("device", rec0)
            self.assertEqual(rec0["patient_id"], rec0["group_id"])

    def test_filename_prefix_group_strategy(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            json_root = tmp_path / "json"
            json_root.mkdir(parents=True, exist_ok=True)

            rows = []
            # 13-digit prefixes represent separate groups; each has 3 files.
            for visit_idx in range(1, 7):
                prefix13 = f"8002000000{visit_idx:03d}"
                mfars = float(40 + visit_idx)
                for suffix in ("101", "201", "301"):
                    file_name = f"{prefix13}{suffix}.json"
                    _make_fake_json(json_root / file_name, test_id=1, n=180)
                    rows.append({"filename": file_name, "device": "cup", "mFARS": mfars})

            metadata_csv = tmp_path / "metadata_filename_device_mfars.csv"
            pd.DataFrame(rows).to_csv(metadata_csv, index=False)

            manifest_path = tmp_path / "split_prefix13.json"
            manifest = create_split_manifest(
                str(metadata_csv),
                str(json_root),
                output_path=str(manifest_path),
                target_col="mFARS",
                seed=42,
                train_ratio=0.70,
                val_ratio=0.15,
                test_ratio=0.15,
                group_id_strategy="filename_prefix",
                filename_group_prefix_len=13,
            )

            all_records = (
                manifest["records"]["train"]
                + manifest["records"]["validation"]
                + manifest["records"]["test"]
            )
            self.assertGreater(len(all_records), 0)
            group_lengths = {len(str(r["group_id"])) for r in all_records}
            self.assertEqual(group_lengths, {13})


if __name__ == "__main__":
    unittest.main()
