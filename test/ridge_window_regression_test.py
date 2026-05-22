# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import unittest

import numpy as np

from opentslm.model.regression.ridge_window import RidgeWindowRegressor


class TestRidgeWindowRegressor(unittest.TestCase):
    def test_extract_window_features_shape(self):
        rng = np.random.default_rng(123)
        window = rng.normal(size=(3000, 7)).astype(np.float32)

        f_with_tid = RidgeWindowRegressor.extract_window_features(window, test_id=2, include_test_id=True)
        f_no_tid = RidgeWindowRegressor.extract_window_features(window, include_test_id=False)

        self.assertEqual(f_with_tid.shape[0], 57)
        self.assertEqual(f_no_tid.shape[0], 56)

    def test_fit_predict_and_roundtrip(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(200, 57))
        true_w = rng.normal(size=(57,))
        y = x @ true_w + 2.5 + rng.normal(scale=0.05, size=(200,))
        weights = rng.uniform(0.2, 1.5, size=(200,))

        model = RidgeWindowRegressor(alpha=0.1, include_test_id=True)
        model.fit(x, y, sample_weight=weights)
        preds = model.predict(x)

        r2 = 1.0 - np.sum((y - preds) ** 2) / np.sum((y - np.mean(y)) ** 2)
        self.assertGreater(r2, 0.99)

        state = model.to_state_dict()
        loaded = RidgeWindowRegressor.from_state_dict(state)
        preds_loaded = loaded.predict(x)
        self.assertTrue(np.allclose(preds, preds_loaded))


if __name__ == "__main__":
    unittest.main()
