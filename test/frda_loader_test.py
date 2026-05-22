#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""Unit tests for FRDA loader utilities."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from opentslm.time_series_datasets.frda.frda_loader import (
    build_windowed_signal_from_json,
    parse_record_rows,
    preprocess_signal,
    window_signal,
)


class TestFRDALoader(unittest.TestCase):
    def test_parse_record_rows_list_with_force(self):
        rows = [
            [100, 0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            [101, 0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, 8.0],
        ]
        parsed = parse_record_rows(rows, test_id=1)
        self.assertEqual(parsed.shape, (2, 7))
        self.assertAlmostEqual(parsed[0, 0], 1.0)
        self.assertAlmostEqual(parsed[0, 6], 7.0)
        self.assertAlmostEqual(parsed[1, 6], 8.0)

    def test_parse_record_rows_without_force(self):
        rows = [
            [100, 0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [101, 0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0],
        ]
        parsed = parse_record_rows(rows, test_id=2)
        self.assertEqual(parsed.shape, (2, 7))
        self.assertTrue(np.allclose(parsed[:, 6], 0.0))

    def test_parse_record_rows_dict_format(self):
        rows = [
            {
                "timestamp": 100,
                "marker": 0,
                "accx": 1.1,
                "accy": 1.2,
                "accz": 1.3,
                "gyrx": 1.4,
                "gyry": 1.5,
                "gyrz": 1.6,
            },
            {
                "timestamp": 101,
                "marker": 0,
                "accx": 2.1,
                "accy": 2.2,
                "accz": 2.3,
                "gyrx": 2.4,
                "gyry": 2.5,
                "gyrz": 2.6,
            },
        ]
        parsed = parse_record_rows(rows, test_id=3)
        self.assertEqual(parsed.shape, (2, 7))
        self.assertTrue(np.allclose(parsed[:, 6], 0.0))

    def test_preprocess_signal_upsamples_test3(self):
        signal = np.random.randn(25, 7).astype(np.float32)
        processed = preprocess_signal(
            signal,
            test_id=3,
            median_kernel_size=1,
            normalize=False,
            upsample_test3=True,
        )
        self.assertEqual(processed.shape[0], 50)

    def test_preprocess_signal_output_finite(self):
        signal = np.random.randn(120, 7).astype(np.float32)
        processed = preprocess_signal(
            signal,
            test_id=2,
            median_kernel_size=5,
            normalize=True,
            upsample_test3=False,
        )
        self.assertTrue(np.isfinite(processed).all())

    def test_window_signal_short_and_long(self):
        short_signal = np.random.randn(1000, 7).astype(np.float32)
        windows_short = window_signal(short_signal, window_size=3000, stride=1500)
        self.assertEqual(len(windows_short), 1)
        self.assertEqual(windows_short[0].shape, (3000, 7))

        long_signal = np.random.randn(4200, 7).astype(np.float32)
        windows_long = window_signal(long_signal, window_size=3000, stride=1500)
        self.assertEqual(len(windows_long), 2)
        self.assertEqual(windows_long[0].shape, (3000, 7))
        self.assertEqual(windows_long[1].shape, (3000, 7))

    def test_build_windowed_signal_from_json(self):
        payload = {
            "TestID": 2,
            "RecordID": "x",
            "record": [
                [100 + i // 10, 0, 0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i, 0.5 * i, 0.6 * i]
                for i in range(3100)
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            windows, meta = build_windowed_signal_from_json(
                str(path),
                window_size=3000,
                stride=1500,
                median_kernel_size=5,
                normalize=True,
                upsample_test3=True,
            )

            self.assertGreaterEqual(len(windows), 1)
            self.assertEqual(windows[0].shape, (3000, 7))
            self.assertEqual(meta["test_id"], 2)
            self.assertEqual(meta["raw_length"], 3100)


if __name__ == "__main__":
    unittest.main()
