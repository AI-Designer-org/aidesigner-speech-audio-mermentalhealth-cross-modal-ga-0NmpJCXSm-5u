"""
Unimodal encoder wrappers for the MER Mental Health architecture.

Provides:
  - BaseEncoder (ABC):   abstract interface for all modality encoders
  - WavLMEncoder:        audio encoder (speech prosody, tone, hesitations)
  - DINOv2Encoder:       video encoder (facial expressions, Action Units)
  - MentalBERTEncoder:   text encoder (semantic content, sentiment)
  - ModalityQualityEstimator: predicts per-utterance modality reliability

Shape conventions:
  Audio: (B, T_samples) waveform -> (B, audio_feat_dim)
  Video: (B, n_frames, C, H, W) frames -> (B, video_feat_dim)
  Text:  (B, T_tokens) token ids -> (B, text_feat_dim)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from abc import ABC, abstractmethod

from layers import LayerNorm, AttentionPooling, ProjectionMLP
from config import ModelConfig


# ---------------------------------------------------------------------------
# Abstract base encoder
# ---------------------------------------------------------------------------

class BaseEncoder(ABC, nn.Module):
    """Abstract base class for all unimodal encoders.

    Every encoder produces a fixed-size utterance-level feature vector
    via attention pooling over time/frame/token dimension.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.feature_dim: int = 0  # set by subclass
        self.frozen: bool = True   # default: frozen during stage 1

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Returns:
            pooled_feat: (B, feature_dim) — attention-pooled utterance representation
        """
        pass

    def freeze(self):
        """Freeze all parameters."""
        for p in self.parameters():
            p.requires_grad = False
        self.frozen = True

    def unfreeze(self):
        """Unfreeze all parameters."""
        for p in self.parameters():
            p.requires_grad = True
        self.frozen = False


# ---------------------------------------------------------------------------
# WavLM Audio Encoder
# ---------------------------------------------------------------------------

class WavLMEncoder(BaseEncoder):
    """Audio encoder based on WavLM Base Plus.

    Processes raw waveforms and produces utterance-level prosodic features.
    Uses attention-weighted mean pooling over the transformer output frames.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.feature_dim = config.audio_feat_dim  # 768 for base-plus

        if config.pretrained:
            try:
                from transformers import WavLMModel
                self.model = WavLMModel.from_pretrained(config.audio_encoder_name)
            except Exception as e:
                print(f"[WavLMEncoder] Could not load pretrained model ({e}), using random init")
                self.model = self._build_dummy_wavlm(config)
        else:
            self.model = self._build_dummy_wavlm(config)

        self.pooling = AttentionPooling(config.audio_feat_dim)

        if config.audio_freeze_encoder:
            self.freeze()

    def _build_dummy_wavlm(self, config: ModelConfig) -> nn.Module:
        """Build a dummy WavLM-compatible module for smoke testing."""
        from transformers import WavLMConfig, WavLMModel
        feat_dim = config.audio_feat_dim
        for n_heads in [8, 4, 2, 1]:
            if feat_dim % n_heads == 0:
                break
        else:
            n_heads = 1
        dummy_cfg = WavLMConfig(
            hidden_size=feat_dim,
            num_hidden_layers=2,
            num_attention_heads=n_heads,
            intermediate_size=feat_dim * 4,
            layer_norm_eps=1e-5,
        )
        return WavLMModel(dummy_cfg)

    def forward(self, waveform: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            waveform: (B, T_samples) — raw audio at 16kHz
            mask: optional (B, T_frames) — True = valid, False = padded
        Returns:
            pooled: (B, audio_feat_dim) — attention-pooled utterance representation
        """
        B = waveform.shape[0]

        # WavLM forward
        outputs = self.model(waveform, attention_mask=mask)
        # outputs.last_hidden_state: (B, T_frames, audio_feat_dim)

        # Pool over time dimension
        pooled = self.pooling(outputs.last_hidden_state)  # (B, audio_feat_dim)

        # Safety check (training only)
        if self.training:
            assert not torch.isnan(pooled).any(), "NaN in WavLMEncoder output"

        return pooled


# ---------------------------------------------------------------------------
# DINOv2 Video Encoder
# ---------------------------------------------------------------------------

class DINOv2Encoder(BaseEncoder):
    """Video encoder based on DINOv2.

    Processes uniformly sampled facial frames per utterance.
    Each frame is independently encoded by DINOv2; frames are then
    attention-pooled into a single utterance vector.

    For real clinical use, frames should be face-detected and
    landmark-aligned before input.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.feature_dim = config.video_feat_dim  # 384 for small
        self.n_frames = config.video_n_frames
        self.frame_size = config.video_frame_size

        if config.pretrained:
            try:
                from transformers import Dinov2Model
                self.model = Dinov2Model.from_pretrained(
                    f"facebook/{config.video_encoder_name}"
                )
            except Exception as e:
                print(f"[DINOv2Encoder] Could not load pretrained model ({e}), using random init")
                self.model = self._build_dummy_dinov2(config)
        else:
            self.model = self._build_dummy_dinov2(config)

        self.frame_pooling = AttentionPooling(config.video_feat_dim)
        self.utterance_pooling = AttentionPooling(config.video_feat_dim)

        if config.video_freeze_encoder:
            self.freeze()

    def _build_dummy_dinov2(self, config: ModelConfig) -> nn.Module:
        """Build a dummy DINOv2-compatible module for smoke testing."""
        from transformers import Dinov2Config, Dinov2Model
        feat_dim = config.video_feat_dim
        # Choose num_heads that divides feat_dim evenly
        for n_heads in [8, 4, 2, 1]:
            if feat_dim % n_heads == 0:
                break
        else:
            n_heads = 1
        dummy_cfg = Dinov2Config(
            hidden_size=feat_dim,
            num_hidden_layers=2,
            num_attention_heads=n_heads,
            intermediate_size=feat_dim * 4,
            patch_size=config.video_patch_size,
            image_size=config.video_frame_size,
        )
        return Dinov2Model(dummy_cfg)

    def forward(self, frames: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            frames: (B, n_frames, C, H, W) — RGB frames, C=3, H=W=224
            mask: optional (B, n_frames) — True if frame is valid (not occluded)
        Returns:
            pooled: (B, video_feat_dim) — attention-pooled utterance representation
        """
        B, F, C, H, W = frames.shape                    # batch, frames, channels, height, width

        # Reshape to process all frames through DINOv2: (B*F, C, H, W)
        x = frames.view(B * F, C, H, W)                 # (B*F, C, H, W)

        outputs = self.model(x)
        # outputs.last_hidden_state: (B*F, N_patches+1, video_feat_dim)
        # Take [CLS] token: index 0
        frame_feats = outputs.last_hidden_state[:, 0, :]   # (B*F, video_feat_dim)
        frame_feats = frame_feats.view(B, F, -1)          # (B, F, video_feat_dim)

        # Pool frames into utterance vector
        pooled = self.utterance_pooling(frame_feats, mask)  # (B, video_feat_dim)

        if self.training:
            assert not torch.isnan(pooled).any(), "NaN in DINOv2Encoder output"

        return pooled


# ---------------------------------------------------------------------------
# MentalBERT Text Encoder
# ---------------------------------------------------------------------------

class MentalBERTEncoder(BaseEncoder):
    """Text encoder based on MentalBERT (BERT fine-tuned on mental health text).

    Uses the [CLS] token representation as the utterance embedding,
    followed by an attention pooling refinement.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.feature_dim = config.text_feat_dim  # 768

        if config.pretrained:
            try:
                from transformers import AutoModel, AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained("mentalbert")
                self.model = AutoModel.from_pretrained("mentalbert")
            except Exception as e:
                print(f"[MentalBERTEncoder] Could not load pretrained model ({e}), using random init")
                self.model = self._build_dummy_bert(config)
                self.tokenizer = None
        else:
            self.model = self._build_dummy_bert(config)
            self.tokenizer = None

        # Refinement pooling over the sequence
        self.refine_pooling = AttentionPooling(config.text_feat_dim)
        self.projection = nn.Linear(config.text_feat_dim, config.text_feat_dim, bias=False)

        if config.text_freeze_encoder:
            self.freeze()

    def _build_dummy_bert(self, config: ModelConfig) -> nn.Module:
        """Build a dummy BERT-compatible module for smoke testing."""
        from transformers import BertConfig, BertModel
        feat_dim = config.text_feat_dim
        for n_heads in [8, 4, 2, 1]:
            if feat_dim % n_heads == 0:
                break
        else:
            n_heads = 1
        dummy_cfg = BertConfig(
            hidden_size=feat_dim,
            num_hidden_layers=2,
            num_attention_heads=n_heads,
            intermediate_size=feat_dim * 4,
            max_position_embeddings=config.text_max_length,
        )
        return BertModel(dummy_cfg)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            input_ids: (B, T_tokens) — token IDs
            attention_mask: (B, T_tokens) — 1 = valid, 0 = padding
        Returns:
            pooled: (B, text_feat_dim) — utterance representation
        """
        B, T = input_ids.shape                          # batch, seq_len

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # outputs.last_hidden_state: (B, T_tokens, text_feat_dim)
        # outputs.pooler_output: (B, text_feat_dim) — [CLS] via tanh

        # Use both CLS token and attention-weighted pooling over sequence
        cls_feat = outputs.pooler_output                 # (B, text_feat_dim)
        seq_feats = outputs.last_hidden_state            # (B, T, text_feat_dim)
        pool_mask = attention_mask.bool() if attention_mask is not None else None
        pooled_seq = self.refine_pooling(seq_feats, pool_mask)  # (B, text_feat_dim)

        # Combine CLS and pooled sequence
        combined = cls_feat + self.projection(pooled_seq)  # (B, text_feat_dim)

        if self.training:
            assert not torch.isnan(combined).any(), "NaN in MentalBERTEncoder output"

        return combined


# ---------------------------------------------------------------------------
# Modality Quality Estimator
# ---------------------------------------------------------------------------

class ModalityQualityEstimator(nn.Module):
    """Predict per-utterance modality reliability from encoder features.

    A lightweight MLP head on pooled encoder features that predicts
    a quality class (e.g., clean/noisy/very_noisy). The reliability
    score is the probability of the highest-quality class.

    Pre-trained on synthetic corruptions. During clinical fine-tuning,
    can run in inference mode or be fine-tuned end-to-end.
    """

    def __init__(self, d_feat: int, n_classes: int, hidden_dim: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.n_classes = n_classes

        self.norm = LayerNorm(d_feat)
        self.net = nn.Sequential(
            nn.Linear(d_feat, hidden_dim, bias=False),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, n_classes, bias=False)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: (B, d_feat) — pooled unimodal encoder output
        Returns:
            reliability: (B,) — scalar in [0, 1], 1 = fully reliable
            logits: (B, n_classes) — raw class logits (for training loss)
        """
        h = self.norm(features)                          # (B, d_feat) float32-safe
        h = self.net(h)                                   # (B, hidden_dim)

        logits = self.classifier(h)                       # (B, n_classes)

        # Reliability = softmax probability of class 0 (= highest quality)
        probs = F.softmax(logits.float(), dim=-1).to(logits.dtype)  # (B, n_classes)
        reliability = probs[:, 0]                          # (B,)

        return reliability, logits

    def estimate(self, features: torch.Tensor) -> torch.Tensor:
        """Convenience: returns only reliability score (no grad needed)."""
        with torch.no_grad():
            reliability, _ = self.forward(features)
        return reliability
