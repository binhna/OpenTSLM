# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""Frozen MOMENT-1-large encoder + MLP regression head for mFARS prediction.

MOMENT expects fixed-length input of seq_len=512 with one channel at a time.
Our windows are [T=3000, C=7]. Strategy:
  1. Split each window into 512-sample chunks (6 chunks, last zero-padded).
  2. For each chunk, run all 7 channels independently through MOMENT's
     patch_embedding + encoder → pool patch tokens → [d_model].
  3. Mean-pool across 7 channels and 6 chunks → single [d_model] vector.
  4. Regression head: Linear(d_model, 512) → ReLU → Dropout → Linear(512, 1).

Only channel_proj and regression_head parameters are trained.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

MOMENT_SEQ_LEN = 512  # fixed input length required by MOMENT


class MOMENTRegressionSP(nn.Module):
    def __init__(
        self,
        moment_path: str,
        device: str = "cuda",
        regression_hidden_dim: int = 512,
        regression_dropout: float = 0.1,
    ):
        super().__init__()
        self.device = device
        self.moment_path = moment_path

        from momentfm import MOMENTPipeline

        pipeline = MOMENTPipeline.from_pretrained(
            moment_path,
            model_kwargs={"task_name": "reconstruction"},
        )
        pipeline.eval()

        # Extract the frozen submodules we need
        self.normalizer = pipeline.normalizer.to(device)
        self.tokenizer = pipeline.tokenizer.to(device)
        self.patch_embedding = pipeline.patch_embedding.to(device)
        self.encoder = pipeline.encoder.to(device)

        for mod in [self.normalizer, self.tokenizer, self.patch_embedding, self.encoder]:
            for p in mod.parameters():
                p.requires_grad = False

        d_model = pipeline.config.d_model  # 1024
        self.d_model = d_model
        n_patches = MOMENT_SEQ_LEN // pipeline.config.patch_len  # 512/8 = 64

        # Project 7 channels of pooled patch embeddings → single d_model vector
        self.channel_proj = nn.Linear(7 * d_model, d_model).to(device)

        self.regression_head = nn.Sequential(
            nn.Linear(d_model, regression_hidden_dim),
            nn.ReLU(),
            nn.Dropout(regression_dropout),
            nn.Linear(regression_hidden_dim, 1),
        ).to(device)

        self.regression_config = {
            "moment_path": moment_path,
            "d_model": d_model,
            "n_patches": n_patches,
            "regression_hidden_dim": regression_hidden_dim,
            "regression_dropout": regression_dropout,
        }

    def _encode_chunk(self, chunk: torch.Tensor) -> torch.Tensor:
        """Encode one [B, 7, 512] chunk → [B, d_model].

        Runs each channel independently through MOMENT's encoder,
        then projects the 7 channel embeddings into one vector.
        """
        B, C, T = chunk.shape
        # Process all channels at once by treating them as batch dimension
        x = chunk.reshape(B * C, 1, T).float()  # [B*C, 1, 512]

        # RevIN normalization (per-channel) — returns tensor directly
        mask = torch.ones(x.shape[0], x.shape[2], device=x.device)
        x = self.normalizer(x, mask=mask, mode="norm")

        # Patch tokenization: [B*C, 1, 512] → [B*C, 1, n_patches, patch_len]
        x = self.tokenizer(x)

        # patch_embedding needs a mask: [B*C, seq_len] of ones
        n = x.shape[0]
        patch_mask = torch.ones(n, MOMENT_SEQ_LEN, device=x.device)

        # Patch embedding: → [B*C, 1, n_patches, d_model] then squeeze channel dim
        x = self.patch_embedding(x, mask=patch_mask)
        x = x.squeeze(1)  # [B*C, n_patches, d_model]

        # T5 encoder: → [B*C, n_patches, d_model]
        enc_out = self.encoder(inputs_embeds=x).last_hidden_state

        # Mean-pool patches → [B*C, d_model]
        pooled = enc_out.mean(dim=1)

        # Reshape back: [B*C, d_model] → [B, C, d_model]
        pooled = pooled.reshape(B, C, self.d_model)

        # Flatten channels and project → [B, d_model]
        return self.channel_proj(pooled.reshape(B, C * self.d_model))

    def _encode_window(self, window: torch.Tensor) -> torch.Tensor:
        """Encode [B, T=3000, C=7] → [B, d_model] by chunking T into 512-sample pieces."""
        # Rearrange to [B, C, T]
        x = window.permute(0, 2, 1).float().to(self.device)  # [B, 7, 3000]
        T = x.shape[2]

        n_full = T // MOMENT_SEQ_LEN  # 5 full chunks of 512 from 3000 (=2560)
        remainder = T % MOMENT_SEQ_LEN  # 440 remaining samples

        chunks = [x[:, :, i * MOMENT_SEQ_LEN:(i + 1) * MOMENT_SEQ_LEN] for i in range(n_full)]
        if remainder > 0:
            last = x[:, :, n_full * MOMENT_SEQ_LEN:]
            pad = MOMENT_SEQ_LEN - remainder
            last = torch.nn.functional.pad(last, (0, pad))
            chunks.append(last)

        # Encode each chunk and mean-pool across chunks → [B, d_model]
        chunk_embs = torch.stack([self._encode_chunk(c) for c in chunks], dim=1)
        return chunk_embs.mean(dim=1)

    def predict_batch(self, batch: List[Dict[str, Any]]) -> torch.Tensor:
        # time_series stored as [C=7, T] in dataset samples
        windows = torch.tensor(
            np.array([np.array(s["time_series"], dtype=np.float32) for s in batch]),
            dtype=torch.float32,
        )  # [B, 7, 3000]
        windows = windows.permute(0, 2, 1)  # [B, 3000, 7] for _encode_window
        emb = self._encode_window(windows)   # [B, d_model]
        return self.regression_head(emb).squeeze(-1)  # [B]

    def parameters(self, recurse: bool = True):
        yield from self.channel_proj.parameters(recurse=recurse)
        yield from self.regression_head.parameters(recurse=recurse)

    def train(self, mode: bool = True):
        super().train(mode)
        # MOMENT stays frozen
        self.normalizer.eval()
        self.tokenizer.eval()
        self.patch_embedding.eval()
        self.encoder.eval()
        return self

    def eval(self):
        return self.train(False)

    def store_to_file(self, path: str, extra_state: Optional[Dict[str, Any]] = None):
        checkpoint: Dict[str, Any] = {
            "channel_proj_state": self.channel_proj.state_dict(),
            "regression_head_state": self.regression_head.state_dict(),
            "regression_config": self.regression_config,
        }
        if extra_state:
            checkpoint.update(extra_state)
        torch.save(checkpoint, path)

    def load_from_file(self, path: str) -> Dict[str, Any]:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.channel_proj.load_state_dict(ckpt["channel_proj_state"])
        self.regression_head.load_state_dict(ckpt["regression_head_state"])
        return ckpt
