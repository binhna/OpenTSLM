# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""Ridge regression on handcrafted window-level IMU features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


def _ensure_2d_float(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}")
    return arr


@dataclass
class RidgeWindowRegressor:
    """Weighted ridge regressor with feature standardization and serializable state."""

    alpha: float = 1.0
    include_test_id: bool = True
    standardize: bool = True
    eps: float = 1e-8

    coef_: Optional[np.ndarray] = None
    intercept_: float = 0.0
    feature_mean_: Optional[np.ndarray] = None
    feature_std_: Optional[np.ndarray] = None

    @staticmethod
    def extract_window_features(
        window: np.ndarray,
        *,
        test_id: Optional[int] = None,
        include_test_id: bool = True,
    ) -> np.ndarray:
        """
        Build a compact statistical feature vector from one window [T, 7].

        Features per channel:
        - mean
        - std
        - percentiles [5, 25, 50, 75, 95]
        - mean absolute first difference
        """
        win = np.asarray(window, dtype=np.float64)
        if win.ndim != 2 or win.shape[1] != 7:
            raise ValueError(f"Expected window shape [T, 7], got {win.shape}")

        means = np.mean(win, axis=0)
        stds = np.std(win, axis=0)
        quantiles = np.percentile(win, [5.0, 25.0, 50.0, 75.0, 95.0], axis=0).reshape(-1)
        if win.shape[0] > 1:
            madiff = np.mean(np.abs(np.diff(win, axis=0)), axis=0)
        else:
            madiff = np.zeros((7,), dtype=np.float64)

        parts = [means, stds, quantiles, madiff]
        if include_test_id:
            tid = 0.0 if test_id is None else float(test_id)
            parts.append(np.asarray([tid], dtype=np.float64))
        return np.concatenate(parts, axis=0)

    def _transform(self, x: np.ndarray) -> np.ndarray:
        x2 = _ensure_2d_float(x)
        if self.feature_mean_ is None or self.feature_std_ is None:
            raise RuntimeError("Model is not fitted")
        return (x2 - self.feature_mean_) / self.feature_std_

    def fit(self, x: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> "RidgeWindowRegressor":
        x2 = _ensure_2d_float(x)
        y1 = np.asarray(y, dtype=np.float64).reshape(-1)
        if x2.shape[0] != y1.shape[0]:
            raise ValueError(f"Mismatched sample counts: X={x2.shape[0]}, y={y1.shape[0]}")

        if self.standardize:
            feat_mean = np.mean(x2, axis=0)
            feat_std = np.std(x2, axis=0)
            feat_std = np.where(feat_std < self.eps, 1.0, feat_std)
            x_std = (x2 - feat_mean) / feat_std
        else:
            feat_mean = np.zeros((x2.shape[1],), dtype=np.float64)
            feat_std = np.ones((x2.shape[1],), dtype=np.float64)
            x_std = x2

        x_aug = np.concatenate([x_std, np.ones((x_std.shape[0], 1), dtype=np.float64)], axis=1)

        if sample_weight is not None:
            w = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
            if w.shape[0] != x_aug.shape[0]:
                raise ValueError(
                    f"Mismatched sample counts for sample_weight: got {w.shape[0]}, expected {x_aug.shape[0]}"
                )
            w = np.clip(w, a_min=self.eps, a_max=None)
            sw = np.sqrt(w)
            xw = x_aug * sw[:, None]
            yw = y1 * sw
        else:
            xw = x_aug
            yw = y1

        d_aug = xw.shape[1]
        reg = np.eye(d_aug, dtype=np.float64)
        reg[-1, -1] = 0.0  # Do not regularize intercept
        a = xw.T @ xw + float(self.alpha) * reg
        b = xw.T @ yw
        theta = np.linalg.solve(a, b)

        self.coef_ = theta[:-1]
        self.intercept_ = float(theta[-1])
        self.feature_mean_ = feat_mean
        self.feature_std_ = feat_std
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted")
        x_std = self._transform(x)
        return x_std @ self.coef_ + self.intercept_

    def to_state_dict(self) -> Dict[str, Any]:
        if self.coef_ is None or self.feature_mean_ is None or self.feature_std_ is None:
            raise RuntimeError("Cannot serialize an unfitted model")
        return {
            "alpha": float(self.alpha),
            "include_test_id": bool(self.include_test_id),
            "standardize": bool(self.standardize),
            "eps": float(self.eps),
            "coef": self.coef_.tolist(),
            "intercept": float(self.intercept_),
            "feature_mean": self.feature_mean_.tolist(),
            "feature_std": self.feature_std_.tolist(),
            "feature_dim": int(self.coef_.shape[0]),
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "RidgeWindowRegressor":
        model = cls(
            alpha=float(state.get("alpha", 1.0)),
            include_test_id=bool(state.get("include_test_id", True)),
            standardize=bool(state.get("standardize", True)),
            eps=float(state.get("eps", 1e-8)),
        )
        model.coef_ = np.asarray(state["coef"], dtype=np.float64)
        model.intercept_ = float(state["intercept"])
        model.feature_mean_ = np.asarray(state["feature_mean"], dtype=np.float64)
        model.feature_std_ = np.asarray(state["feature_std"], dtype=np.float64)
        return model
