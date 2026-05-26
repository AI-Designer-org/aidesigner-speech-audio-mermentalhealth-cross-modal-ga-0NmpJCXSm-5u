"""
Shared neural network layers for the MER Mental Health architecture.

Contains:
  - ProjectionMLP:  projects a modality feature to a common latent dimension
  - FeedForward:    position-wise FFN (linear -> activation -> dropout -> linear)
  - MultiHeadSelfAttention: standard multi-head scaled dot-product attention
  - LayerNorm:      safe layernorm (float32 cast for numerical stability)
  - LoRALinear:     LoRA-decorated linear layer for parameter-efficient fine-tuning
  - SinusoidalPositionEncoding: fixed sinusoidal position encoding
  - AttentionPooling: attention-weighted mean pooling over sequence dimension
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from abc import ABC, abstractmethod

# ---------------------------------------------------------------------------
# Safe normalization
# ---------------------------------------------------------------------------

class LayerNorm(nn.Module):
    """LayerNorm that always runs in float32 for numerical stability."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply float32-stable layer normalization.

        Casts to float32 internally regardless of input dtype to avoid
        bf16/fp16 numerical instability. Always safe for training.

        Args:
            x: (..., d_model) — any tensor with final dim = d_model.

        Returns:
            (..., d_model) — normalized output with same dtype as input.

        Shape invariants:
            - Input and output have identical shape.
            - dtype in {float32, bfloat16, float16}; float32 internal.
        """
        dtype = x.dtype
        x = x.float()                                    # (B, ..., d_model) float32
        mean = x.mean(dim=-1, keepdim=True)               # (B, ..., 1)
        var = (x - mean).pow(2).mean(dim=-1, keepdim=True)  # (B, ..., 1)
        out = (x - mean) / torch.sqrt(var + self.eps)     # (B, ..., d_model)
        out = out * self.weight.float() + self.bias.float()  # (B, ..., d_model)
        return out.to(dtype)


# ---------------------------------------------------------------------------
# Feed-forward network
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network: linear -> activation -> dropout -> linear."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        act = {"gelu": nn.GELU(approximate="tanh"), "relu": nn.ReLU(), "silu": nn.SiLU()}
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            act[activation],
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """FFN forward: linear -> activation -> dropout -> linear.

        Args:
            x: (..., d_model) — input hidden states.

        Returns:
            (..., d_model) — FFN output, residual NOT yet added.

        Shape invariants:
            - Input and output shapes match exactly.
            - d_model must match constructor argument.
        """
        return self.net(x)


# ---------------------------------------------------------------------------
# Projection MLP
# ---------------------------------------------------------------------------

class ProjectionMLP(nn.Module):
    """Project features from d_in to d_out through a shallow MLP."""

    def __init__(self, d_in: int, d_out: int, n_layers: int = 2,
                 dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        assert n_layers >= 1, "ProjectionMLP requires at least 1 layer"

        layers = []
        dims = [d_in] + [d_out] * n_layers
        for i in range(n_layers):
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=False))
            if i < n_layers - 1:
                act = {"gelu": nn.GELU(approximate="tanh"),
                       "relu": nn.ReLU(), "silu": nn.SiLU()}
                layers.append(act[activation])
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, d_in) -> (B, d_out)
        return self.net(x)


# ---------------------------------------------------------------------------
# Multi-head self-attention
# ---------------------------------------------------------------------------

class MultiHeadSelfAttention(nn.Module):
    """Standard multi-head scaled dot-product self-attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, bias: bool = False):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None,
                need_weights: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (B, T, d_model)
            attn_mask: optional (B, T, T) or (T, T) bool mask (True = keep)
            need_weights: return attention weights if True
        Returns:
            out: (B, T, d_model)
            attn_weights: (B, n_heads, T, T) or None
        """
        B, T, D = x.shape                                # batch, seq_len, d_model
        d_h = self.d_head

        q = self.q_proj(x)                               # (B, T, D)
        k = self.k_proj(x)                               # (B, T, D)
        v = self.v_proj(x)                               # (B, T, D)

        q = q.view(B, T, self.n_heads, d_h).transpose(1, 2)   # (B, n_heads, T, d_h)
        k = k.view(B, T, self.n_heads, d_h).transpose(1, 2)   # (B, n_heads, T, d_h)
        v = v.view(B, T, self.n_heads, d_h).transpose(1, 2)   # (B, n_heads, T, d_h)

        # Scaled dot-product attention (float32 for safety)
        scale = 1.0 / math.sqrt(d_h)
        attn = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale  # (B, n_heads, T, T)

        if attn_mask is not None:
            # mask: True = keep, False = mask out
            # attn_mask can be (T, T) or (B, T, T); broadcast to (B, 1, T, T)
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)   # (1, 1, T, T)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)                # (B, 1, T, T)
            attn = attn.masked_fill(~attn_mask, float("-inf"))

        attn_weights = F.softmax(attn, dim=-1).to(x.dtype)   # (B, n_heads, T, T)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)                   # (B, n_heads, T, d_h)
        out = out.transpose(1, 2).contiguous().view(B, T, D)  # (B, T, D)
        out = self.out_proj(out)                              # (B, T, D)

        if need_weights:
            return out, attn_weights.detach()
        return out, None


# ---------------------------------------------------------------------------
# Cross-modal multi-head attention (single query attends to multiple key-value sources)
# ---------------------------------------------------------------------------

class CrossModalAttention(nn.Module):
    """Multi-head cross-modal attention where a query attends to multiple key/value sources."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, bias: bool = False):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, keys: List[torch.Tensor],
                values: List[torch.Tensor],
                need_weights: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            query: (B, 1, d_model) — single query vector (e.g., one modality)
            keys:  list of (B, 1, d_model) — one key per source modality
            values: list of (B, 1, d_model) — one value per source modality
        Returns:
            out: (B, d_model) — attended output
            attn_weights: (B, n_heads, 1, n_sources) or None
        """
        B = query.shape[0]
        n_src = len(keys)
        d_h = self.d_head

        # Stack key/value sources
        K = torch.stack(keys, dim=1)                       # (B, n_src, d_model)
        V = torch.stack(values, dim=1)                      # (B, n_src, d_model)

        Q = self.q_proj(query)                              # (B, 1, d_model)
        K = self.k_proj(K)                                  # (B, n_src, d_model)
        V = self.v_proj(V)                                  # (B, n_src, d_model)

        # Reshape for multi-head
        Q = Q.view(B, 1, self.n_heads, d_h).transpose(1, 2)     # (B, n_heads, 1, d_h)
        K = K.view(B, n_src, self.n_heads, d_h).permute(0, 2, 1, 3)  # (B, n_heads, n_src, d_h)
        V = V.view(B, n_src, self.n_heads, d_h).permute(0, 2, 1, 3)  # (B, n_heads, n_src, d_h)

        scale = 1.0 / math.sqrt(d_h)
        attn = torch.matmul(Q.float(), K.float().transpose(-2, -1)) * scale  # (B, n_heads, 1, n_src)
        attn_weights = F.softmax(attn, dim=-1).to(query.dtype)               # (B, n_heads, 1, n_src)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V)                  # (B, n_heads, 1, d_h)
        out = out.transpose(1, 2).contiguous().view(B, self.n_heads * d_h)  # (B, d_model)
        out = self.out_proj(out)                             # (B, d_model)

        if need_weights:
            return out, attn_weights.detach()
        return out, None


# ---------------------------------------------------------------------------
# Sinusoidal position encoding
# ---------------------------------------------------------------------------

class SinusoidalPositionEncoding(nn.Module):
    """Fixed sinusoidal position encoding."""

    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)                  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                             -(math.log(10000.0) / d_model))  # (d_model/2,)
        pe[:, 0::2] = torch.sin(position * div_term)         # (max_len, d_model/2)
        pe[:, 1::2] = torch.cos(position * div_term)         # (max_len, d_model/2)
        self.register_buffer("pe", pe.unsqueeze(0))          # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model) — used only to get T
        # returns: (1, T, d_model)
        T = x.size(1)
        return self.pe[:, :T, :].to(x.dtype)                 # (1, T, d_model)


# ---------------------------------------------------------------------------
# Attention pooling (learned query attends to sequence)
# ---------------------------------------------------------------------------

class AttentionPooling(nn.Module):
    """Attention-weighted mean pooling over a sequence dimension."""

    def __init__(self, d_model: int):
        super().__init__()
        self.score_net = nn.Linear(d_model, 1, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
            mask: (B, T) boolean — True = valid, False = padded
        Returns:
            pooled: (B, d_model)
        """
        scores = self.score_net(x)                           # (B, T, 1)
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        attn_weights = F.softmax(scores.float(), dim=1).to(x.dtype)  # (B, T, 1)
        pooled = (attn_weights * x).sum(dim=1)                # (B, d_model)
        return pooled


# ---------------------------------------------------------------------------
# LoRA Linear layer
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Linear layer with Low-Rank Adaptation (LoRA).

    During training, computes: output = Wx + (x @ A @ B) * (alpha / r)
    where A and B are the low-rank matrices.
    During inference in eval mode, can merge LoRA weights into W for zero-overhead.
    """

    def __init__(self, in_features: int, out_features: int,
                 r: int = 8, alpha: float = 16, dropout: float = 0.05,
                 bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # Pretrained weight (frozen)
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # LoRA low-rank matrices (trainable)
        if r > 0:
            self.lora_A = nn.Parameter(torch.randn(in_features, r) * 0.01)
            self.lora_B = nn.Parameter(torch.zeros(r, out_features))
            self.dropout = nn.Dropout(dropout)
        else:
            # r=0 means no LoRA (identity)
            self.register_buffer("lora_A", None)
            self.register_buffer("lora_B", None)
            self.dropout = nn.Identity()

        self._merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_features) -> (..., out_features)
        result = self.linear(x)                             # (..., out_features)

        if self.r > 0 and not self._merged:
            # LoRA branch
            lora_input = self.dropout(x)                     # (..., in_features)
            lora_update = (lora_input @ self.lora_A @ self.lora_B) * self.scaling  # (..., out_features)
            result = result + lora_update

        return result

    def merge_weights(self):
        """Merge LoRA weights into the base linear layer (for inference)."""
        if self.r == 0 or self._merged:
            return
        delta_w = (self.lora_A @ self.lora_B) * self.scaling  # (in_features, out_features)
        self.linear.weight.data.add_(delta_w.T)
        self._merged = True

    def unmerge_weights(self):
        """Reverse merge_weights for fine-tuning."""
        if self.r == 0 or not self._merged:
            return
        delta_w = (self.lora_A @ self.lora_B) * self.scaling
        self.linear.weight.data.sub_(delta_w.T)
        self._merged = False


def apply_lora_to_linear(module: nn.Module, config, prefix: str = "",
                         current_depth: int = 0, max_depth: int = 10) -> None:
    """Recursively replace Linear layers in a module with LoRALinear.

    Matches target module names against config.lora_target_modules.
    """
    if current_depth > max_depth:
        return

    target_set = set(config.lora_target_modules)
    for name, child in module.named_children():
        child_path = f"{prefix}.{name}" if prefix else name

        # Check if this is a Linear layer we want to replace
        is_target = any(t in child_path for t in target_set)
        if isinstance(child, nn.Linear) and is_target:
            lora_linear = LoRALinear(
                in_features=child.in_features,
                out_features=child.out_features,
                r=config.lora_r,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout,
                bias=child.bias is not None,
            )
            # Copy pretrained weights
            lora_linear.linear.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                lora_linear.linear.bias.data.copy_(child.bias.data)
            setattr(module, name, lora_linear)
        else:
            apply_lora_to_linear(child, config, child_path, current_depth + 1, max_depth)


# ---------------------------------------------------------------------------
# Parameter count utility
# ---------------------------------------------------------------------------

def count_params(model: nn.Module) -> None:
    """Print total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,} | Trainable: {trainable:,} ({100 * trainable / total:.1f}%)")
    return total, trainable
