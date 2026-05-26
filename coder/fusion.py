"""
Cross-Modal Gated Exchange Fusion (CM-GEF) for the MER Mental Health architecture.

This is the core novel operator in the architecture. It fuses audio, video, and
text modalities with adaptive gating that substitutes unreliable modalities
with cross-modal context.

Components:
  - MultiHeadExchangeAttention:  each modality attends to the other two
  - GatedSubstitution:           per-dimension gated interpolation between self and context
  - CrossModalGatedExchangeFusion: full fusion block combining everything

Shape convention: all modality features are (B, proj_dim) after projection.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from abc import ABC, abstractmethod

from layers import LayerNorm, FeedForward, CrossModalAttention
from config import ModelConfig


# ---------------------------------------------------------------------------
# Base operator abstract class (for swappable components)
# ---------------------------------------------------------------------------

class BaseFusionOperator(ABC, nn.Module):
    """Base class for the novel fusion operator in the architecture."""

    @abstractmethod
    def forward(self, audio_feat: torch.Tensor, video_feat: torch.Tensor,
                text_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            audio_feat: (B, proj_dim)
            video_feat: (B, proj_dim)
            text_feat:  (B, proj_dim)
        Returns:
            fused:          (B, proj_dim) — fused utterance representation
            gate_weights:   (B, 3) — [gate_a, gate_v, gate_t] mean per-dim gate
            modality_weights: (B, 3) — softmax reliability weights
        """
        pass


# ---------------------------------------------------------------------------
# Per-dimension gated substitution
# ---------------------------------------------------------------------------

class GatedSubstitution(nn.Module):
    """Per-feature-dimension gated interpolation between self and context.

    Given a modality's self-features, its cross-modal context, and its
    reliability score, computes a per-dimension gate that interpolates:
        output = gate * self_feat + (1 - gate) * ctx_feat

    Higher reliability -> gate closer to 1 (trust self).
    Lower reliability  -> gate closer to 0 (substitute with context).
    """

    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(3 * d_model, hidden_dim, bias=False),   # concat[self, ctx, reliability]
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, d_model, bias=False),       # per-dimension gate
        )
        self._init_weights()

    def _init_weights(self):
        """Initialize gate network to produce ~0.5 gates initially."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)

    def forward(self, self_feat: torch.Tensor, ctx_feat: torch.Tensor,
                reliability: torch.Tensor) -> torch.Tensor:
        """
        Args:
            self_feat: (B, d_model) — modality's own projected features
            ctx_feat:  (B, d_model) — cross-modal context from other modalities
            reliability: (B,) — scalar reliability in [0, 1]
        Returns:
            fused: (B, d_model) — gated interpolation
            gate:  (B, d_model) — per-dimension gate values in [0, 1]
        """
        r = reliability.unsqueeze(-1)                           # (B, 1)
        r_expanded = r.expand(-1, self_feat.size(-1))           # (B, d_model)

        gate_input = torch.cat([self_feat, ctx_feat, r_expanded], dim=-1)  # (B, 3*d_model)
        gate_logits = self.gate_net(gate_input)                  # (B, d_model)
        gate = torch.sigmoid(gate_logits)                        # (B, d_model) in [0, 1]

        fused = gate * self_feat + (1 - gate) * ctx_feat         # (B, d_model)

        # Safety check (training only)
        if self.training:
            assert not torch.isnan(fused).any(), "NaN in GatedSubstitution output"
            # Gate entropy regularization target (not applied here, done at loss level)

        return fused, gate


# ---------------------------------------------------------------------------
# Cross-Modal Gated Exchange Fusion
# ---------------------------------------------------------------------------

class CrossModalGatedExchangeFusion(BaseFusionOperator):
    """Cross-Modal Gated Exchange Fusion (CM-GEF).

    Stages:
      1. Multi-head exchange attention: each modality attends to the other two
      2. Per-dimension gated substitution: gate·self + (1-gate)·context
      3. Reliability-weighted aggregation: softmax(r_a, r_v, r_t) · fused_feats
      4. Optional: stacked exchange layers with residual + FFN
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        d = config.proj_dim

        # Exchange attention — one per modality (query = this modality, KV = others)
        self.exchange_attn_a = CrossModalAttention(d, config.n_exchange_heads,
                                                    config.exchange_dropout)
        self.exchange_attn_v = CrossModalAttention(d, config.n_exchange_heads,
                                                    config.exchange_dropout)
        self.exchange_attn_t = CrossModalAttention(d, config.n_exchange_heads,
                                                    config.exchange_dropout)

        # Per-dimension gated substitution — one per modality
        self.gate_a = GatedSubstitution(d, config.exchange_hidden_dim)
        self.gate_v = GatedSubstitution(d, config.exchange_hidden_dim)
        self.gate_t = GatedSubstitution(d, config.exchange_hidden_dim)

        # Post-exchange FFN and norm (for stacked layers)
        self.post_norm = LayerNorm(d)
        self.post_ffn = FeedForward(d, config.exchange_hidden_dim, config.exchange_dropout)

        # Dropout for final aggregation
        self.dropout = nn.Dropout(config.exchange_dropout)

    def forward(self, audio_feat: torch.Tensor, video_feat: torch.Tensor,
                text_feat: torch.Tensor,
                r_audio: Optional[torch.Tensor] = None,
                r_video: Optional[torch.Tensor] = None,
                r_text: Optional[torch.Tensor] = None,
                audio_mask: Optional[torch.Tensor] = None,
                video_mask: Optional[torch.Tensor] = None,
                text_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            audio_feat: (B, proj_dim) — projected audio utterance features
            video_feat: (B, proj_dim) — projected video utterance features
            text_feat:  (B, proj_dim) — projected text utterance features
            r_audio, r_video, r_text: (B,) reliability scores or None (default 1.0)
            audio_mask, video_mask, text_mask: (B,) bool — True if modality present
        Returns:
            fused:          (B, proj_dim) — fused utterance representation
            gate_weights:   (B, 3) — [gate_a, gate_v, gate_t] mean per-dim gate
            modality_weights: (B, 3) — softmax reliability weights
        """
        B = audio_feat.shape[0]
        d = audio_feat.shape[1]

        # ── Default reliability to 1.0 if not provided ──
        if r_audio is None:
            r_audio = torch.ones(B, device=audio_feat.device, dtype=audio_feat.dtype)
        if r_video is None:
            r_video = torch.ones(B, device=video_feat.device, dtype=video_feat.dtype)
        if r_text is None:
            r_text = torch.ones(B, device=text_feat.device, dtype=text_feat.dtype)

        # ── Apply masks for missing modalities ──
        if audio_mask is not None:
            audio_feat = audio_feat * audio_mask.unsqueeze(-1).float()
            r_audio = r_audio * audio_mask.float() + (1 - audio_mask.float()) * 0.0
        if video_mask is not None:
            video_feat = video_feat * video_mask.unsqueeze(-1).float()
            r_video = r_video * video_mask.float() + (1 - video_mask.float()) * 0.0
        if text_mask is not None:
            text_feat = text_feat * text_mask.unsqueeze(-1).float()
            r_text = r_text * text_mask.float() + (1 - text_mask.float()) * 0.0

        # ── Stage 1: Exchange attention ──
        # Each modality, as a single query, attends to the other two as key-value sources
        # Audio query attends to video + text
        ctx_audio, _ = self.exchange_attn_a(
            query=audio_feat.unsqueeze(1),          # (B, 1, d)
            keys=[video_feat.unsqueeze(1), text_feat.unsqueeze(1)],
            values=[video_feat.unsqueeze(1), text_feat.unsqueeze(1)],
        )                                            # (B, d)

        # Video query attends to audio + text
        ctx_video, _ = self.exchange_attn_v(
            query=video_feat.unsqueeze(1),          # (B, 1, d)
            keys=[audio_feat.unsqueeze(1), text_feat.unsqueeze(1)],
            values=[audio_feat.unsqueeze(1), text_feat.unsqueeze(1)],
        )                                            # (B, d)

        # Text query attends to audio + video
        ctx_text, _ = self.exchange_attn_t(
            query=text_feat.unsqueeze(1),           # (B, 1, d)
            keys=[audio_feat.unsqueeze(1), video_feat.unsqueeze(1)],
            values=[audio_feat.unsqueeze(1), video_feat.unsqueeze(1)],
        )                                            # (B, d)

        # ── Stage 2: Per-dimension gated substitution ──
        audio_fused, gate_a = self.gate_a(audio_feat, ctx_audio, r_audio)  # (B, d), (B, d)
        video_fused, gate_v = self.gate_v(video_feat, ctx_video, r_video)  # (B, d), (B, d)
        text_fused, gate_t = self.gate_t(text_feat, ctx_text, r_text)      # (B, d), (B, d)

        # ── Stage 3: Reliability-weighted aggregation ──
        # Stack reliability scores and softmax
        r_stack = torch.stack([r_audio, r_video, r_text], dim=-1)   # (B, 3)
        # Ensure no all-zero weights (add epsilon)
        modality_weights = F.softmax(r_stack.float() + 1e-8, dim=-1).to(r_stack.dtype)  # (B, 3)

        fused = (
            modality_weights[:, 0:1] * audio_fused +
            modality_weights[:, 1:2] * video_fused +
            modality_weights[:, 2:3] * text_fused
        )                                                            # (B, d)

        # ── Stage 4: Stacked exchange blocks (optional) ──
        for _ in range(1, self.config.n_exchange_layers):
            residual = fused
            fused = self.post_norm(fused)                            # (B, d)
            fused = fused + self.post_ffn(fused)                     # (B, d)
            if self.config.exchange_residual:
                fused = residual + fused

        fused = self.dropout(fused)                                  # (B, d)

        # Mean gate values for monitoring
        gate_weights = torch.stack([
            gate_a.mean(dim=-1),   # (B,)
            gate_v.mean(dim=-1),   # (B,)
            gate_t.mean(dim=-1),   # (B,)
        ], dim=-1)                  # (B, 3)

        # Safety
        if self.training:
            assert not torch.isnan(fused).any(), "NaN in CM-GEF fused output"

        return fused, gate_weights, modality_weights

    def forward_with_checkpoint(self, audio_feat: torch.Tensor,
                                 video_feat: torch.Tensor,
                                 text_feat: torch.Tensor,
                                 **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Wrapper for gradient checkpointing support."""
        return torch.utils.checkpoint.checkpoint(
            self.forward, audio_feat, video_feat, text_feat,
            *[kwargs.get(k) for k in ["r_audio", "r_video", "r_text",
                                        "audio_mask", "video_mask", "text_mask"]
              if kwargs.get(k) is not None],
            use_reentrant=False,
        )
