"""
Session-Level Hierarchical Transformer (SLHT) for the MER Mental Health architecture.

Models session-level emotion dynamics with therapeutic phase awareness.

Components:
  - PhaseEmbedding:       learned embedding for each therapeutic phase
  - SessionTransformer:   bidirectional multi-layer Transformer encoder over utterances
  - PhaseLevelPooling:    aggregate utterances within each therapeutic phase
  - SessionLevelTransformer: full SLHT block combining all components

Phase structure (DAIC-WOZ / clinical interviews):
  0 — Rapport-building (introductory questions)
  1 — Exploration (open-ended probes)
  2 — Intervention (targeted clinical questions)
  3 — Closure (wrap-up)

Shape convention: (B, T, d) where T = number of utterances in session.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from abc import ABC, abstractmethod

from layers import LayerNorm, FeedForward, MultiHeadSelfAttention, SinusoidalPositionEncoding
from config import ModelConfig


# ---------------------------------------------------------------------------
# Base operator abstract class
# ---------------------------------------------------------------------------

class BaseSessionOperator(ABC, nn.Module):
    """Base class for the session-level temporal modeling operator."""

    @abstractmethod
    def forward(self, utterance_features: torch.Tensor,
                phase_labels: Optional[torch.Tensor] = None,
                timestamps: Optional[torch.Tensor] = None,
                utterance_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            utterance_features: (B, T, d_in)
            phase_labels:       (B, T) or None — therapeutic phase index per utterance
            timestamps:         (B, T) or None — seconds from session start
            utterance_mask:     (B, T) bool — True = valid utterance
        Returns:
            session_feat: (B, d_session_model) — session-level representation
            phase_repr:   (B, n_phases, d_session_model) — per-phase representations
        """
        pass


# ---------------------------------------------------------------------------
# Phase embedding
# ---------------------------------------------------------------------------

class PhaseEmbedding(nn.Module):
    """Learned embedding for therapeutic phases."""

    def __init__(self, n_phases: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(n_phases, embed_dim)

    def forward(self, phase_labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            phase_labels: (B, T) — integer phase indices
        Returns:
            phase_emb: (B, T, embed_dim)
        """
        return self.embedding(phase_labels)              # (B, T, embed_dim)


# ---------------------------------------------------------------------------
# Session Transformer Encoder Block
# ---------------------------------------------------------------------------

class SessionTransformerBlock(nn.Module):
    """Single transformer block for session-level encoding.

    Architecture: MHA -> residual -> LayerNorm -> FFN -> residual -> LayerNorm
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm1 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                use_checkpoint: bool = False) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
            attn_mask: optional (B, T, T) or (T, T) bool mask
            use_checkpoint: enable gradient checkpointing
        Returns:
            out: (B, T, d_model)
        """
        # Self-attention with residual
        residual = x
        x = self.norm1(x)                                 # (B, T, d_model)

        if use_checkpoint:
            attn_out, _ = torch.utils.checkpoint.checkpoint(
                self.attention, x, attn_mask, False, use_reentrant=False
            )
        else:
            attn_out, _ = self.attention(x, attn_mask)     # (B, T, d_model)

        x = residual + self.dropout(attn_out)              # (B, T, d_model)

        # FFN with residual
        residual = x
        x = self.norm2(x)                                  # (B, T, d_model)
        x = residual + self.dropout(self.ffn(x))           # (B, T, d_model)

        return x


# ---------------------------------------------------------------------------
# Phase-level pooling
# ---------------------------------------------------------------------------

class PhaseLevelPooling(nn.Module):
    """Aggregate utterances within each therapeutic phase.

    Uses masked mean as the base representation, then refines with
    cross-attention between the phase mean and all phase utterances.
    """

    def __init__(self, d_model: int, n_phases: int, dropout: float = 0.1):
        super().__init__()
        self.n_phases = n_phases
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, phase_labels: torch.Tensor,
                utterance_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model) — session transformer output
            phase_labels: (B, T) — integer phase indices [0, n_phases)
            utterance_mask: (B, T) bool — True = valid utterance
        Returns:
            phase_repr: (B, n_phases, d_model) — per-phase representations
        """
        B, T, D = x.shape                                  # batch, seq_len, d_model

        phase_repr_list = []
        for phase_idx in range(self.n_phases):
            # Create phase mask
            phase_mask = (phase_labels == phase_idx).float()  # (B, T)

            # Apply utterance mask
            if utterance_mask is not None:
                phase_mask = phase_mask * utterance_mask.float()

            # Masked mean (fallback if no utterances in this phase)
            phase_sum = (x * phase_mask.unsqueeze(-1)).sum(dim=1)   # (B, d_model)
            phase_count = phase_mask.sum(dim=-1, keepdim=True) + 1e-8  # (B, 1)
            phase_mean = phase_sum / phase_count                    # (B, d_model)

            # Cross-attention refinement: phase_mean attends to phase utterances
            # Direct scaled dot-product attention over utterances in this phase
            # query: (B, 1, D), keys: (B, T, D), values: (B, T, D)
            scale = 1.0 / math.sqrt(D)
            attn_scores = torch.matmul(
                phase_mean.unsqueeze(1).float(),                     # (B, 1, D)
                x.float().transpose(-2, -1)                          # (B, D, T)
            ) * scale                                                # (B, 1, T)

            # Apply phase mask to attention scores (mask out non-phase utterances)
            attn_mask = phase_mask.unsqueeze(1).float()              # (B, 1, T)
            attn_scores = attn_scores + (1.0 - attn_mask) * (-1e9)   # (B, 1, T)

            attn_weights = F.softmax(attn_scores, dim=-1).to(x.dtype)  # (B, 1, T)
            ctx = torch.matmul(attn_weights, x)                       # (B, 1, D)
            ctx = ctx.squeeze(1)                                      # (B, D)

            # Blend masked mean with cross-attended (residual)
            phase_repr = phase_mean + self.dropout(ctx)             # (B, d_model)
            phase_repr_list.append(phase_repr)

        phase_repr = torch.stack(phase_repr_list, dim=1)           # (B, n_phases, d_model)
        return phase_repr


# ---------------------------------------------------------------------------
# Session-Level Hierarchical Transformer
# ---------------------------------------------------------------------------

class SessionLevelTransformer(BaseSessionOperator):
    """Session-Level Hierarchical Transformer (SLHT).

    Architecture:
      1. Phase embedding (learned) + position encoding (learned or sinusoidal)
      2. Project concatenated utterance + phase features to session model dim
      3. Bidirectional multi-layer Transformer over all utterances
      4. Phase-level pooling: cross-attend phase means to utterances
      5. Phase-weighted session pooling -> session representation
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        d_in = config.proj_dim
        d_model = config.d_session_model
        n_phases = config.n_therapeutic_phases
        phase_dim = config.phase_embed_dim

        # Phase embeddings
        self.phase_embed = PhaseEmbedding(n_phases, phase_dim)

        # Position encoding (learned)
        if config.session_pos_encoding == "learned":
            self.pos_embed = nn.Embedding(config.max_utterances, d_model)
        else:
            self.pos_embed = SinusoidalPositionEncoding(config.max_utterances, d_model)

        # Projection: (d_in + phase_dim) -> d_model
        self.input_proj = nn.Linear(d_in + phase_dim, d_model, bias=False)

        # Session transformer blocks
        self.blocks = nn.ModuleList([
            SessionTransformerBlock(
                d_model, config.n_session_heads, config.session_d_model_ff,
                config.session_dropout,
            ) for _ in range(config.n_session_layers)
        ])

        # Phase-level pooling
        self.phase_pooling = PhaseLevelPooling(d_model, n_phases, config.session_dropout)

        # Phase importance scoring for final session representation
        self.phase_importance = nn.Sequential(
            nn.Linear(d_model, 1, bias=False),        # per-phase importance logit
        )

        # Final layer norm
        self.norm = LayerNorm(d_model)

        self.dropout = nn.Dropout(config.session_dropout)

    def forward(self, utterance_features: torch.Tensor,
                phase_labels: Optional[torch.Tensor] = None,
                timestamps: Optional[torch.Tensor] = None,
                utterance_mask: Optional[torch.Tensor] = None,
                use_checkpoint: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            utterance_features: (B, T, proj_dim) — fused utterance features
            phase_labels: (B, T) — therapeutic phase index per utterance
            timestamps: (B, T) — seconds from session start (optional)
            utterance_mask: (B, T) bool — True = valid utterance (not padding)
            use_checkpoint: enable gradient checkpointing
        Returns:
            session_feat: (B, d_session_model) — session-level representation
            phase_repr: (B, n_therapeutic_phases, d_session_model) — per-phase repr
        """
        B, T, _ = utterance_features.shape               # batch, seq_len, proj_dim
        device = utterance_features.device
        d_model = self.config.d_session_model

        # ── 1. Phase embedding ──
        if phase_labels is None:
            # Default: assume all utterances are in phase 0 (no phase info)
            phase_labels = torch.zeros(B, T, device=device, dtype=torch.long)

        phase_emb = self.phase_embed(phase_labels)       # (B, T, phase_embed_dim)

        # ── 2. Position encoding ──
        if self.config.session_pos_encoding == "learned":
            positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)  # (B, T)
            positions = positions.clamp(max=self.config.max_utterances - 1)
            pos_enc = self.pos_embed(positions)           # (B, T, d_model)
        else:
            pos_enc = self.pos_embed(utterance_features)  # (1, T, d_model) broadcast

        # ── 3. Project to session model dimension ──
        # Concatenate utterance features with phase embeddings, then project
        x = torch.cat([utterance_features, phase_emb], dim=-1)  # (B, T, d_in + phase_dim)
        x = self.input_proj(x)                              # (B, T, d_model)
        x = x + pos_enc                                      # (B, T, d_model)
        x = self.dropout(x)

        # ── 4. Bidirectional session transformer ──
        # Create attention mask from utterance mask (causal only if streaming_mode)
        attn_mask = None
        if utterance_mask is not None and not self.config.streaming_mode:
            # Bidirectional: mask = utterance_mask[i] & utterance_mask[j]
            attn_mask = utterance_mask.unsqueeze(1) & utterance_mask.unsqueeze(2)  # (B, T, T)

        for block in self.blocks:
            x = block(x, attn_mask=attn_mask,
                     use_checkpoint=use_checkpoint)        # (B, T, d_model)

        x = self.norm(x)                                    # (B, T, d_model)

        # ── 5. Phase-level pooling ──
        phase_repr = self.phase_pooling(x, phase_labels, utterance_mask)  # (B, n_phases, d_model)

        # ── 6. Session representation via phase-weighted pooling ──
        phase_importance_logits = self.phase_importance(phase_repr).squeeze(-1)  # (B, n_phases)
        phase_importance = F.softmax(phase_importance_logits.float(), dim=-1).to(
            phase_importance_logits.dtype
        )                                                           # (B, n_phases)

        session_feat = (phase_importance.unsqueeze(-1) * phase_repr).sum(dim=1)  # (B, d_model)

        # Residual: also add global utterance context
        if utterance_mask is not None:
            global_context = (x * utterance_mask.unsqueeze(-1).float()).sum(dim=1) / \
                             (utterance_mask.sum(dim=-1, keepdim=True).float() + 1e-8)  # (B, d_model)
        else:
            global_context = x.mean(dim=1)                    # (B, d_model)

        session_feat = session_feat + global_context          # (B, d_model)

        # Safety
        if self.training:
            assert not torch.isnan(session_feat).any(), "NaN in SLHT output"

        return session_feat, phase_repr

    def forward_with_checkpoint(self, utterance_features: torch.Tensor,
                                 **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Wrapper for gradient checkpointing."""
        return torch.utils.checkpoint.checkpoint(
            self.forward, utterance_features,
            kwargs.get("phase_labels"), kwargs.get("timestamps"),
            kwargs.get("utterance_mask"), False,  # use_checkpoint inside blocks
            use_reentrant=False,
        )
