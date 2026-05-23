# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors
# SPDX-License-Identifier: MIT

"""Frozen MOMENT encoder + MLP regression head for mFARS prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn


# MOMENT processes sequences of fixed length 512. We segment each 3000-sample
# window into 512-sample chunks (with the last chunk zero-padded if needed),
# run each through the encoder, then mean-pool across chunks before the head.
MOMENT_SEQ_LEN = 512


def _load_moment(model_path: str, device: str) -> Any:
    """Load MOMENT encoder from local cache via momentfm or transformers."""
    try:
        from momentfm import MOMENTConfig, MOMENTPipeline

        config = MOMENTConfig.from_pretrained(model_path)
        model = MOMENTPipeline.from_pretrained(
            model_path,
            model_kwargs={"task_name": "embedding"},
        )
        model = model.to(device)
        return model, int(config.d_model or 1024)
    except Exception:
        pass

    # Fallback: load as a plain T5 encoder (MOMENT backbone is flan-t5-large)
    from transformers import T5EncoderModel, AutoConfig

    cfg = AutoConfig.from_pretrained(model_path)
    hidden = int(getattr(cfg, "d_model", 1024))
    encoder = T5EncoderModel.from_pretrained(model_path)
    encoder = encoder.to(device)
    return encoder, hidden


class _T5EncoderWrapper(nn.Module):
    """Thin wrapper so T5EncoderModel works like a MOMENT pipeline."""

    def __init__(self, encoder: Any, patch_embed: nn.Linear, device: str):
        super().__init__()
        self.encoder = encoder
        self.patch_embed = patch_embed
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T] → project patches → T5 encoder → pool
        B, C, T = x.shape
        # Flatten channels into the patch dimension: [B, num_patches * C, patch_len]
        # Simple approach: reshape to [B*C, 1, T] and embed
        x_flat = x.reshape(B * C, T)  # [B*C, T]
        # Split into n_patches of MOMENT_SEQ_LEN → treat each as a token
        n_patches = max(1, T // 8)  # patch_len=8
        # Truncate to multiple of patch_len
        usable = n_patches * 8
        x_flat = x_flat[:, :usable].reshape(B * C, n_patches, 8)  # [B*C, P, 8]
        emb = self.patch_embed(x_flat)  # [B*C, P, d_model]
        out = self.encoder(inputs_embeds=emb).last_hidden_state  # [B*C, P, d_model]
        pooled = out.mean(dim=1)  # [B*C, d_model]
        pooled = pooled.reshape(B, C, -1).mean(dim=1)  # [B, d_model]
        return pooled


class MOMENTRegressionSP(nn.Module):
    """Frozen MOMENT-1-large encoder with a lightweight regression head.

    Input: windows of shape [T=3000, C=7]. Each window is segmented into
    512-sample chunks (6 chunks), encoded independently, then averaged before
    the regression head.
    """

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

        self._moment, d_model = _load_moment(moment_path, device)
        self.d_model = d_model

        # Freeze all MOMENT parameters
        for p in self._moment.parameters():
            p.requires_grad = False

        # Input projection: 7 channels × MOMENT_SEQ_LEN → patch embeddings
        # MOMENT patch_len=8; we project [B, 7, MOMENT_SEQ_LEN] per chunk
        self._use_momentfm = hasattr(self._moment, "encoder")
        if not self._use_momentfm:
            # T5 fallback: need a patch embedding layer
            self._patch_embed = nn.Linear(8, d_model).to(device)
        else:
            self._patch_embed = None

        # Channel mixer: project 7 channels of d_model → 1 × d_model
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
            "regression_hidden_dim": regression_hidden_dim,
            "regression_dropout": regression_dropout,
        }

    def _encode_chunk(self, chunk: torch.Tensor) -> torch.Tensor:
        """Encode one [B, 7, MOMENT_SEQ_LEN] chunk → [B, d_model]."""
        B, C, T = chunk.shape

        if self._use_momentfm:
            # momentfm pipeline expects [B, C, T]
            out = self._moment(x_enc=chunk.float())
            if hasattr(out, "embeddings"):
                emb = out.embeddings  # [B, d_model] or [B, C, d_model]
            else:
                emb = out
            if emb.ndim == 3:
                # [B, C, d_model] → mean over channels here, then channel_proj below
                emb = emb  # keep [B, C, d_model]
            else:
                # [B, d_model] → expand to [B, C, d_model] by repeating
                emb = emb.unsqueeze(1).expand(-1, C, -1)
        else:
            # T5 fallback
            chunk_flat = chunk.reshape(B * C, T)
            n_patches = T // 8
            x_p = chunk_flat[:, : n_patches * 8].reshape(B * C, n_patches, 8)
            x_p = self._patch_embed(x_p.to(self.device))
            enc_out = self._moment.encoder(inputs_embeds=x_p).last_hidden_state
            pooled = enc_out.mean(dim=1).reshape(B, C, self.d_model)
            emb = pooled

        # emb: [B, C, d_model] → flatten channels → [B, C*d_model] → project
        emb_flat = emb.reshape(B, C * self.d_model)
        return self.channel_proj(emb_flat)  # [B, d_model]

    def _encode_window(self, window: torch.Tensor) -> torch.Tensor:
        """Encode [B, T=3000, C=7] → [B, d_model] by chunking over T."""
        # Rearrange to [B, C, T]
        x = window.permute(0, 2, 1).float().to(self.device)  # [B, 7, 3000]

        n_full = x.shape[2] // MOMENT_SEQ_LEN
        remainder = x.shape[2] % MOMENT_SEQ_LEN
        chunks = []
        for i in range(n_full):
            chunks.append(x[:, :, i * MOMENT_SEQ_LEN : (i + 1) * MOMENT_SEQ_LEN])
        if remainder > 0:
            pad_size = MOMENT_SEQ_LEN - remainder
            last = x[:, :, n_full * MOMENT_SEQ_LEN :]
            last = torch.nn.functional.pad(last, (0, pad_size))
            chunks.append(last)

        if not chunks:
            chunks = [x]

        chunk_embs = torch.stack(
            [self._encode_chunk(c) for c in chunks], dim=1
        )  # [B, n_chunks, d_model]
        return chunk_embs.mean(dim=1)  # [B, d_model]

    def predict_batch(self, batch: List[Dict[str, Any]]) -> torch.Tensor:
        windows = torch.tensor(
            np.array([np.array(s["time_series"], dtype=np.float32).T for s in batch]),
            dtype=torch.float32,
        )  # [B, C=7, T=3000] — note: time_series stored as [C, T] in dataset
        # Permute to [B, T, C] for _encode_window
        windows = windows.permute(0, 2, 1)  # [B, 3000, 7]
        emb = self._encode_window(windows)  # [B, d_model]
        return self.regression_head(emb).squeeze(-1)  # [B]

    def parameters(self, recurse: bool = True):
        """Only yield trainable parameters (regression head + channel_proj)."""
        for p in self.channel_proj.parameters(recurse=recurse):
            yield p
        for p in self.regression_head.parameters(recurse=recurse):
            yield p
        if self._patch_embed is not None:
            for p in self._patch_embed.parameters(recurse=recurse):
                yield p

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep MOMENT frozen regardless
        if self._use_momentfm:
            self._moment.eval()
        else:
            self._moment.eval()
        return self

    def eval(self):
        return self.train(False)

    def store_to_file(self, path: str, extra_state: Optional[Dict[str, Any]] = None):
        checkpoint: Dict[str, Any] = {
            "channel_proj_state": self.channel_proj.state_dict(),
            "regression_head_state": self.regression_head.state_dict(),
            "regression_config": self.regression_config,
        }
        if self._patch_embed is not None:
            checkpoint["patch_embed_state"] = self._patch_embed.state_dict()
        if extra_state:
            checkpoint.update(extra_state)
        torch.save(checkpoint, path)

    def load_from_file(self, path: str) -> Dict[str, Any]:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.channel_proj.load_state_dict(ckpt["channel_proj_state"])
        self.regression_head.load_state_dict(ckpt["regression_head_state"])
        if "patch_embed_state" in ckpt and self._patch_embed is not None:
            self._patch_embed.load_state_dict(ckpt["patch_embed_state"])
        return ckpt
