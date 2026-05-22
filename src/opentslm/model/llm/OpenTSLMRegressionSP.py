# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""Regression variant of OpenTSLM-SP for continuous mFARS prediction."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from opentslm.model.encoder.TransformerCNNEncoder import TransformerCNNEncoder
from opentslm.model.llm.OpenTSLMSP import OpenTSLMSP


class OpenTSLMRegressionSP(OpenTSLMSP):
    """OpenTSLM-SP with a lightweight regression head."""

    def __init__(
        self,
        llm_id: str = "meta-llama/Llama-3.2-1B",
        device: str = "cuda",
        regression_hidden_dim: int = 512,
        regression_dropout: float = 0.1,
        encoder_patch_size: int = 50,
        encoder_max_patches: int = 256,
    ):
        super().__init__(llm_id=llm_id, device=device)
        hidden_size = self.llm.config.hidden_size

        # Use larger patches by default for manageable token lengths in regression.
        self.encoder = TransformerCNNEncoder(
            patch_size=encoder_patch_size,
            max_patches=encoder_max_patches,
        ).to(device)
        self.patch_size = encoder_patch_size

        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, regression_hidden_dim),
            nn.ReLU(),
            nn.Dropout(regression_dropout),
            nn.Linear(regression_hidden_dim, 1),
        ).to(device)

        self.regression_config = {
            "regression_hidden_dim": regression_hidden_dim,
            "regression_dropout": regression_dropout,
            "llm_id": llm_id,
            "encoder_patch_size": encoder_patch_size,
            "encoder_max_patches": encoder_max_patches,
        }

    def predict_batch(self, batch: List[Dict[str, Any]]) -> torch.Tensor:
        """Predict a scalar target for each sample in the batch."""
        inputs_embeds, attention_mask = self.pad_and_apply_batch(batch)

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden = outputs.hidden_states[-1]  # [B, L, H]
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)

        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        preds = self.regression_head(pooled.float()).squeeze(-1)
        return preds

    def compute_loss(
        self,
        batch: List[Dict[str, Any]],
        loss_type: str = "huber",
    ) -> torch.Tensor:
        """Compute regression loss for the current batch."""
        preds = self.predict_batch(batch)
        targets = torch.tensor(
            [float(sample["target"]) for sample in batch],
            dtype=torch.float32,
            device=self.device,
        )

        if loss_type == "huber":
            return F.smooth_l1_loss(preds, targets, beta=1.0)
        if loss_type == "mse":
            return F.mse_loss(preds, targets)

        raise ValueError(f"Unsupported loss_type '{loss_type}'. Choose from ['huber', 'mse']")

    def store_to_file(self, path: str, extra_state: Optional[Dict[str, Any]] = None):
        """Store encoder/projector/regression states and optional metadata."""
        checkpoint: Dict[str, Any] = {
            "encoder_state": self.encoder.state_dict(),
            "projector_state": self.projector.state_dict(),
            "regression_head_state": self.regression_head.state_dict(),
            "regression_config": self.regression_config,
        }

        self.save_lora_state_to_checkpoint(checkpoint)

        if extra_state:
            checkpoint.update(extra_state)

        torch.save(checkpoint, path)

    def load_from_file(self, path: str) -> Dict[str, Any]:
        """Load checkpoint and return raw checkpoint dictionary."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        self.encoder.load_state_dict(ckpt["encoder_state"])
        self.projector.load_state_dict(ckpt["projector_state"])

        if "regression_head_state" not in ckpt:
            raise RuntimeError(
                f"Checkpoint at {path} is missing regression_head_state. "
                "Expected a regression checkpoint produced by OpenTSLMRegressionSP."
            )
        self.regression_head.load_state_dict(ckpt["regression_head_state"])

        self.load_lora_state_from_checkpoint(ckpt, allow_missing=True)
        return ckpt
