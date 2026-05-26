"""
Full MER Mental Health Model — Multimodal Emotion Recognition for Mental Health Diagnostics.

Architecture pipeline:
  Raw Inputs (Audio + Video + Text)
    -> Unimodal Encoders (WavLM / DINOv2 / MentalBERT)
    -> Modality Quality Estimators
    -> Projection MLPs (to common dimension)
    -> Cross-Modal Gated Exchange Fusion (CM-GEF)
    -> Session-Level Hierarchical Transformer (SLHT)
    -> Clinical Concept Bottleneck (CCB)
    -> PHQ-8 Predictions (total score + items + binary depression)

Training pipeline:
  Stage 1: Fusion pre-training on in-the-wild data (MOSEI/MELD/IEMOCAP)
  Stage 2: Clinical adaptation on DAIC-WOZ/E-DAIC with LoRA

Usage:
    from model import MERMentalHealthModel
    from config import ModelConfig

    cfg = ModelConfig(pretrained=False)  # random init for testing
    model = MERMentalHealthModel(cfg)
    outputs = model(
        waveforms=batch["waveform"],        # (B, T_samples)
        frames=batch["frames"],              # (B, n_frames, C, H, W)
        input_ids=batch["input_ids"],        # (B, T_tokens)
    )
    # outputs contains: phq8_total, phq8_items, concept_scores, depression_prob
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass

from config import ModelConfig
from layers import (
    ProjectionMLP, LayerNorm, FeedForward, count_params,
    apply_lora_to_linear,
)
from encoders import (
    WavLMEncoder, DINOv2Encoder, MentalBERTEncoder,
    ModalityQualityEstimator,
)
from fusion import CrossModalGatedExchangeFusion
from session import SessionLevelTransformer
from concept import ClinicalConceptBottleneck


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class MEROutput:
    """Structured output from the MER model."""

    phq8_total: torch.Tensor              # (B,) predicted PHQ-8 total [0, 24]
    phq8_items: torch.Tensor              # (B, 8) per-item scores [0, 3]
    concept_scores: torch.Tensor          # (B, n_concepts) concept activations [0, 1]
    depression_prob: torch.Tensor         # (B,) depression probability
    gate_weights: Optional[torch.Tensor] = None      # (B, 3) mean gate per modality
    modality_weights: Optional[torch.Tensor] = None  # (B, 3) reliability weights
    utterance_feats: Optional[torch.Tensor] = None   # (B, T, proj_dim) per-utterance
    session_feat: Optional[torch.Tensor] = None      # (B, d_session_model)
    phase_repr: Optional[torch.Tensor] = None         # (B, n_phases, d_session_model)
    reliability_scores: Optional[Dict[str, torch.Tensor]] = None
    loss: Optional[torch.Tensor] = None               # scalar loss

    def detach(self):
        """Return a new MEROutput with all tensors detached."""
        return MEROutput(
            phq8_total=self.phq8_total.detach(),
            phq8_items=self.phq8_items.detach(),
            concept_scores=self.concept_scores.detach(),
            depression_prob=self.depression_prob.detach(),
            gate_weights=self.gate_weights.detach() if self.gate_weights is not None else None,
            modality_weights=self.modality_weights.detach() if self.modality_weights is not None else None,
            utterance_feats=self.utterance_feats.detach() if self.utterance_feats is not None else None,
            session_feat=self.session_feat.detach() if self.session_feat is not None else None,
            phase_repr=self.phase_repr.detach() if self.phase_repr is not None else None,
            reliability_scores={k: v.detach() for k, v in self.reliability_scores.items()}
            if self.reliability_scores is not None else None,
            loss=self.loss.detach() if self.loss is not None else None,
        )


# ---------------------------------------------------------------------------
# MER Mental Health Full Model
# ---------------------------------------------------------------------------

class MERMentalHealthModel(nn.Module):
    """Full Multimodal Emotion Recognition model for mental health diagnostics.

    Combines audio (WavLM), video (DINOv2), and text (MentalBERT) encoders
    with cross-modal gated fusion, session-level temporal modeling, and a
    clinical concept bottleneck for interpretable PHQ-8 prediction.

    Usage modes:
      1. Per-utterance: forward_utterance(waveform, frames, input_ids) -> fused_feat
      2. Full session:  forward(waveforms, frames, input_ids, ...) -> MEROutput
      3. Streaming:     forward_streaming(utterance_feat) -> MEROutput (no session context)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.device: Optional[torch.device] = None

        # ── Unimodal encoders ──
        self.audio_encoder = WavLMEncoder(config)
        self.video_encoder = DINOv2Encoder(config)
        self.text_encoder = MentalBERTEncoder(config)

        # ── Attention pooling for each encoder ──
        # (already built into each encoder; fine)

        # ── Modality quality estimators ──
        self.audio_quality_estimator = ModalityQualityEstimator(
            d_feat=config.audio_feat_dim,
            n_classes=config.n_audio_quality_classes,
            hidden_dim=config.quality_estimator_hidden,
            dropout=config.quality_estimator_dropout,
        )
        self.video_quality_estimator = ModalityQualityEstimator(
            d_feat=config.video_feat_dim,
            n_classes=config.n_video_quality_classes,
            hidden_dim=config.quality_estimator_hidden,
            dropout=config.quality_estimator_dropout,
        )
        self.text_quality_estimator = ModalityQualityEstimator(
            d_feat=config.text_feat_dim,
            n_classes=config.n_text_quality_classes,
            hidden_dim=config.quality_estimator_hidden,
            dropout=config.quality_estimator_dropout,
        )

        # ── Projection MLPs (d_feat -> proj_dim) ──
        self.audio_projection = ProjectionMLP(
            config.audio_feat_dim, config.proj_dim,
            n_layers=config.proj_n_layers, dropout=config.proj_dropout,
            activation=config.proj_activation,
        )
        self.video_projection = ProjectionMLP(
            config.video_feat_dim, config.proj_dim,
            n_layers=config.proj_n_layers, dropout=config.proj_dropout,
            activation=config.proj_activation,
        )
        self.text_projection = ProjectionMLP(
            config.text_feat_dim, config.proj_dim,
            n_layers=config.proj_n_layers, dropout=config.proj_dropout,
            activation=config.proj_activation,
        )

        # ── Cross-Modal Gated Exchange Fusion ──
        self.fusion = CrossModalGatedExchangeFusion(config)

        # ── Session-Level Hierarchical Transformer ──
        self.session_transformer = SessionLevelTransformer(config)

        # ── Clinical Concept Bottleneck ──
        self.concept_bottleneck = ClinicalConceptBottleneck(config)

        # ── Keep track of which parameters are trainable for LoRA ──
        self._lora_applied = False

        # ── Init weights for non-pretrained components ──
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for all non-encoder components."""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) and not any(
                enc in name for enc in ["audio_encoder", "video_encoder", "text_encoder"]
            ):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def apply_lora(self, config: Optional[ModelConfig] = None) -> None:
        """Apply LoRA adapters to target modules for stage 2 fine-tuning.

        Replaces linear layers matching lora_target_modules with LoRALinear
        wrappers in the specified components.
        """
        cfg = config or self.config
        if self._lora_applied:
            return

        if cfg.lora_adapt_encoders:
            apply_lora_to_linear(self.audio_encoder, cfg, "audio_encoder")
            apply_lora_to_linear(self.video_encoder, cfg, "video_encoder")
            apply_lora_to_linear(self.text_encoder, cfg, "text_encoder")

        if cfg.lora_adapt_fusion:
            apply_lora_to_linear(self.fusion, cfg, "fusion")

        if cfg.lora_adapt_session:
            apply_lora_to_linear(self.session_transformer, cfg, "session_transformer")

        # CCB is small; typically trained fully (lora_adapt_concept defaults to False)

        self._lora_applied = True

    def freeze_encoders(self):
        """Freeze unimodal encoder backbones."""
        self.audio_encoder.freeze()
        self.video_encoder.freeze()
        self.text_encoder.freeze()

    def unfreeze_encoders(self):
        """Unfreeze unimodal encoder backbones."""
        self.audio_encoder.unfreeze()
        self.video_encoder.unfreeze()
        self.text_encoder.unfreeze()

    def set_trainable_params(self, stage: int = 1):
        """Configure which parameters are trainable for each training stage.

        Stage 1: Train projection MLPs, quality estimators, fusion, session, CCB.
                 Encoders are frozen.

        Stage 2: Train LoRA adapters + fusion + session + CCB.
                 Encoders are frozen except for LoRA adapters.
        """
        if stage == 1:
            # Freeze all first, then selectively unfreeze
            for p in self.parameters():
                p.requires_grad = False

            # Unfreeze trainable components
            for module in [self.audio_projection, self.video_projection,
                           self.text_projection,
                           self.audio_quality_estimator, self.video_quality_estimator,
                           self.text_quality_estimator,
                           self.fusion, self.session_transformer,
                           self.concept_bottleneck]:
                for p in module.parameters():
                    p.requires_grad = True

        elif stage == 2:
            # Freeze encoders
            self.freeze_encoders()

            # Apply LoRA if not yet applied
            self.apply_lora()

            # Ensure LoRA parameters are trainable
            for name, p in self.named_parameters():
                # LoRA parameters (lora_A, lora_B) should be trainable
                if "lora_" in name:
                    p.requires_grad = True
                # Fusion, session, CCB stay trainable
                elif any(m in name for m in ["fusion", "session_transformer",
                                              "concept_bottleneck",
                                              "quality_estimator",
                                              "projection"]):
                    p.requires_grad = True

        else:
            # Stage 0: everything frozen (inference mode)
            for p in self.parameters():
                p.requires_grad = False

    def encode_utterance(self, waveform: torch.Tensor, frames: torch.Tensor,
                         input_ids: torch.Tensor,
                         audio_mask: Optional[torch.Tensor] = None,
                         video_mask: Optional[torch.Tensor] = None,
                         text_mask: Optional[torch.Tensor] = None,
                         audio_attn_mask: Optional[torch.Tensor] = None,
                         text_attn_mask: Optional[torch.Tensor] = None
                         ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Encode a single utterance through all three unimodal encoders.

        Args:
            waveform: (B, T_samples) raw audio
            frames: (B, n_frames, C, H, W) video frames
            input_ids: (B, T_tokens) text token IDs
            audio_mask: (B,) bool — True if audio modality present
            video_mask: (B,) bool — True if video modality present
            text_mask: (B,) bool — True if text modality present
            audio_attn_mask: (B, T_frames) WavLM attention mask
            text_attn_mask: (B, T_tokens) BERT attention mask

        Returns:
            fused_feat: (B, proj_dim) — fused utterance representation
            reliability: dict of modality -> (B,) reliability scores
            gate_info: dict of "gate_weights" (B,3) and "modality_weights" (B,3)
        """
        B = waveform.shape[0]
        device = waveform.device
        self.device = device

        # Default masks to True if not provided
        if audio_mask is None:
            audio_mask = torch.ones(B, device=device, dtype=torch.bool)
        if video_mask is None:
            video_mask = torch.ones(B, device=device, dtype=torch.bool)
        if text_mask is None:
            text_mask = torch.ones(B, device=device, dtype=torch.bool)

        # ── 1a. Audio encoding ──
        audio_feat_raw = torch.zeros(B, self.config.audio_feat_dim, device=device)
        if audio_mask.any():
            audio_subset = waveform[audio_mask]
            audio_attn_subset = audio_attn_mask[audio_mask] if audio_attn_mask is not None else None
            audio_encoded = self.audio_encoder(audio_subset, mask=audio_attn_subset)  # (n_audio, audio_feat_dim)
            audio_feat_raw[audio_mask] = audio_encoded

        # ── 1b. Video encoding ──
        video_feat_raw = torch.zeros(B, self.config.video_feat_dim, device=device)
        if video_mask.any():
            frames_subset = frames[video_mask]
            video_encoded = self.video_encoder(frames_subset)  # (n_video, video_feat_dim)
            video_feat_raw[video_mask] = video_encoded

        # ── 1c. Text encoding ──
        text_feat_raw = torch.zeros(B, self.config.text_feat_dim, device=device)
        if text_mask.any():
            text_input_ids = input_ids[text_mask]
            text_attn_subset = text_attn_mask[text_mask] if text_attn_mask is not None else None
            text_encoded = self.text_encoder(text_input_ids, attention_mask=text_attn_subset)  # (n_text, text_feat_dim)
            text_feat_raw[text_mask] = text_encoded

        # ── 2. Quality estimation ──
        r_audio, audio_qual_logits = self.audio_quality_estimator(audio_feat_raw)  # (B,), (B, 3)
        r_video, video_qual_logits = self.video_quality_estimator(video_feat_raw)  # (B,), (B, 3)
        r_text, text_qual_logits = self.text_quality_estimator(text_feat_raw)      # (B,), (B, 3)

        # Set reliability to 0 for missing modalities
        r_audio = r_audio * audio_mask.float()
        r_video = r_video * video_mask.float()
        r_text = r_text * text_mask.float()

        reliability = {
            "audio": r_audio,
            "video": r_video,
            "text": r_text,
            "audio_logits": audio_qual_logits,
            "video_logits": video_qual_logits,
            "text_logits": text_qual_logits,
        }

        # ── 3. Projection to common dimension ──
        audio_proj = self.audio_projection(audio_feat_raw)     # (B, proj_dim)
        video_proj = self.video_projection(video_feat_raw)     # (B, proj_dim)
        text_proj = self.text_projection(text_feat_raw)        # (B, proj_dim)

        # ── 4. Cross-modal fusion ──
        fused_feat, gate_weights, modality_weights = self.fusion(
            audio_proj, video_proj, text_proj,
            r_audio=r_audio, r_video=r_video, r_text=r_text,
            audio_mask=audio_mask, video_mask=video_mask, text_mask=text_mask,
        )                                                      # (B, proj_dim), (B, 3), (B, 3)

        gate_info = {
            "gate_weights": gate_weights,
            "modality_weights": modality_weights,
        }

        return fused_feat, reliability, gate_info

    def forward_utterance(self, waveform: torch.Tensor, frames: torch.Tensor,
                          input_ids: torch.Tensor,
                          audio_mask: Optional[torch.Tensor] = None,
                          video_mask: Optional[torch.Tensor] = None,
                          text_mask: Optional[torch.Tensor] = None,
                          audio_attn_mask: Optional[torch.Tensor] = None,
                          text_attn_mask: Optional[torch.Tensor] = None
                          ) -> torch.Tensor:
        """Encode a single utterance and return fused features.

        For use in streaming mode or per-utterance processing
        without session context.

        Returns:
            fused_feat: (B, proj_dim)
        """
        fused_feat, _, _ = self.encode_utterance(
            waveform, frames, input_ids,
            audio_mask, video_mask, text_mask,
            audio_attn_mask, text_attn_mask,
        )
        return fused_feat

    def forward(self,
                waveforms: torch.Tensor,
                frames: torch.Tensor,
                input_ids: torch.Tensor,
                phase_labels: Optional[torch.Tensor] = None,
                timestamps: Optional[torch.Tensor] = None,
                utterance_mask: Optional[torch.Tensor] = None,
                audio_mask: Optional[torch.Tensor] = None,
                video_mask: Optional[torch.Tensor] = None,
                text_mask: Optional[torch.Tensor] = None,
                audio_attn_mask: Optional[torch.Tensor] = None,
                text_attn_mask: Optional[torch.Tensor] = None,
                phq8_labels: Optional[torch.Tensor] = None,
                concept_labels: Optional[torch.Tensor] = None,
                use_checkpoint: bool = False,
                return_all: bool = False,
                ) -> MEROutput:
        """
        Full forward pass: encodes utterances, fuses modalities, models session,
        then predicts PHQ-8 through the clinical concept bottleneck.

        Args:
            waveforms: (B, T_utterances, T_samples) — raw audio per utterance
            frames: (B, T_utterances, n_frames, C, H, W) — video frames per utterance
            input_ids: (B, T_utterances, T_tokens) — text token IDs per utterance
            phase_labels: (B, T_utterances) — therapeutic phase index per utterance
            timestamps: (B, T_utterances) — seconds from session start
            utterance_mask: (B, T_utterances) bool — True if utterance is valid
            audio_mask: (B, T_utterances) bool — True if audio present
            video_mask: (B, T_utterances) bool — True if video present
            text_mask: (B, T_utterances) bool — True if text present
            audio_attn_mask: (B, T_utterances, T_frames) — WavLM attention mask per utt
            text_attn_mask: (B, T_utterances, T_tokens) — BERT attention mask per utt
            phq8_labels: (B,) optional — ground truth PHQ-8 total score for loss
            concept_labels: (B, n_concepts) optional — ground truth concept scores
            use_checkpoint: enable gradient checkpointing
            return_all: return intermediate representations

        Returns:
            MEROutput with predictions and optional intermediate features
        """
        B, T_utt = waveforms.shape[:2]
        device = waveforms.device
        self.device = device
        cfg = self.config

        # ── Flatten utterances for batch processing ──
        # (B, T_utt, ...) -> (B*T_utt, ...)
        flat_waveforms = waveforms.view(-1, waveforms.shape[-1])             # (B*T, T_samples)
        flat_frames = frames.view(-1, *frames.shape[2:])                     # (B*T, n_frames, C, H, W)
        flat_input_ids = input_ids.view(-1, input_ids.shape[-1])             # (B*T, T_tokens)

        # Flatten masks
        if audio_mask is not None:
            flat_audio_mask = audio_mask.view(-1)                            # (B*T,)
        else:
            flat_audio_mask = None
        if video_mask is not None:
            flat_video_mask = video_mask.view(-1)                            # (B*T,)
        else:
            flat_video_mask = None
        if text_mask is not None:
            flat_text_mask = text_mask.view(-1)                              # (B*T,)
        else:
            flat_text_mask = None
        if audio_attn_mask is not None:
            try:
                flat_audio_attn_mask = audio_attn_mask.view(-1, audio_attn_mask.shape[-1])
            except RuntimeError:
                flat_audio_attn_mask = None
        else:
            flat_audio_attn_mask = None
        if text_attn_mask is not None:
            try:
                flat_text_attn_mask = text_attn_mask.view(-1, text_attn_mask.shape[-1])
            except RuntimeError:
                flat_text_attn_mask = None
        else:
            flat_text_attn_mask = None

        # ── Encode all utterances ──
        fused_feat, reliability, gate_info = self.encode_utterance(
            flat_waveforms, flat_frames, flat_input_ids,
            audio_mask=flat_audio_mask,
            video_mask=flat_video_mask,
            text_mask=flat_text_mask,
            audio_attn_mask=flat_audio_attn_mask,
            text_attn_mask=flat_text_attn_mask,
        )                                                                     # (B*T, proj_dim)

        # ── Reshape back to session format ──
        utt_feats = fused_feat.view(B, T_utt, cfg.proj_dim)                  # (B, T, proj_dim)

        # ── Session-level modeling ──
        if cfg.streaming_mode:
            # Per-utterance: use utt_feats directly, no session context
            session_feat = utt_feats.mean(dim=1)                              # (B, proj_dim)
            phase_repr = session_feat.unsqueeze(1).unsqueeze(1)               # placeholder
            # Need to project to d_session_model for CCB
            session_feat_proj = nn.Linear(cfg.proj_dim, cfg.d_session_model,
                                          bias=False).to(device, utt_feats.dtype)(
                session_feat
            )                                                                 # (B, d_session_model)
        else:
            session_feat, phase_repr = self.session_transformer(
                utt_feats,
                phase_labels=phase_labels,
                timestamps=timestamps,
                utterance_mask=utterance_mask,
                use_checkpoint=use_checkpoint,
            )                                                                 # (B, d_session_model), (B, n_phases, d_session_model)
            session_feat_proj = session_feat

        # ── Clinical concept bottleneck ──
        phq8_total, phq8_items, concept_scores, depression_prob = \
            self.concept_bottleneck(session_feat_proj, concept_labels=concept_labels)

        # ── Assemble output ──
        output = MEROutput(
            phq8_total=phq8_total,
            phq8_items=phq8_items,
            concept_scores=concept_scores,
            depression_prob=depression_prob,
            gate_weights=gate_info["gate_weights"],
            modality_weights=gate_info["modality_weights"],
            utterance_feats=utt_feats if return_all else None,
            session_feat=session_feat_proj if return_all else None,
            phase_repr=phase_repr if return_all else None,
            reliability_scores=reliability if return_all else None,
        )

        # ── Compute loss if labels provided ──
        if phq8_labels is not None:
            output.loss = self.compute_loss(
                phq8_total, phq8_items, depression_prob, concept_scores,
                phq8_labels, concept_labels,
            )

        return output

    @torch.no_grad()
    def forward_streaming(self, fused_feat: torch.Tensor) -> MEROutput:
        """Streaming inference: predict from a single utterance without session context.

        Args:
            fused_feat: (B, proj_dim) — per-utterance fused features
        Returns:
            MEROutput with predictions (placeholder session features)
        """
        B = fused_feat.shape[0]
        device = fused_feat.device

        # Simple projection to session dim (in streaming mode, we skip the
        # full session transformer and use a lightweight MLP)
        if not hasattr(self, '_streaming_proj'):
            self._streaming_proj = nn.Linear(
                self.config.proj_dim, self.config.d_session_model, bias=False
            ).to(device, fused_feat.dtype)

        session_feat = self._streaming_proj(fused_feat)  # (B, d_session_model)

        phq8_total, phq8_items, concept_scores, depression_prob = \
            self.concept_bottleneck(session_feat)

        return MEROutput(
            phq8_total=phq8_total,
            phq8_items=phq8_items,
            concept_scores=concept_scores,
            depression_prob=depression_prob,
        )

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def compute_loss(self, phq8_total: torch.Tensor, phq8_items: torch.Tensor,
                     depression_prob: torch.Tensor, concept_scores: torch.Tensor,
                     phq8_labels: torch.Tensor,
                     concept_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute multi-task loss.

        Loss components:
          - L_mae:  MAE on PHQ-8 total score
          - L_bce:  Binary cross-entropy for depression classification
          - L_item: MSE on per-item PHQ-8 scores (if available)
          - L_concept: Binary cross-entropy on concept labels (if available)

        Args:
            phq8_total: (B,) predicted PHQ-8 total
            phq8_items: (B, 8) predicted PHQ-8 item scores
            depression_prob: (B,) predicted depression probability
            concept_scores: (B, n_concepts) concept activations
            phq8_labels: (B,) ground truth PHQ-8 total scores
            concept_labels: (B, n_concepts) or None

        Returns:
            loss: scalar tensor
        """
        cfg = self.config
        loss = torch.tensor(0.0, device=phq8_total.device)

        # PHQ-8 regression loss (MAE)
        loss_mae = F.l1_loss(phq8_total, phq8_labels)                     # scalar
        loss = loss + cfg.loss_mae_weight * loss_mae

        # Binary depression classification (BCE)
        if cfg.classification_head:
            depression_labels = (phq8_labels >= cfg.phq8_threshold).float()  # (B,)
            loss_bce = F.binary_cross_entropy(
                depression_prob, depression_labels
            )                                                               # scalar
            loss = loss + cfg.loss_bce_weight * loss_bce

        # PHQ-8 item-level loss (if item labels available)
        # In DAIC-WOZ, item-level labels may not be available;
        # this is optional and can be disabled.
        if cfg.item_prediction_head and phq8_labels.dim() > 1 and phq8_labels.shape[-1] == cfg.n_phq8_items:
            loss_item = F.mse_loss(phq8_items, phq8_labels)                 # scalar
            loss = loss + cfg.loss_item_weight * loss_item

        # Concept supervision (if concept labels available)
        if concept_labels is not None:
            loss_concept = F.binary_cross_entropy(concept_scores, concept_labels)  # scalar
            loss = loss + cfg.loss_concept_supervision_weight * loss_concept

        return loss


# ---------------------------------------------------------------------------
# Training pipeline helpers
# ---------------------------------------------------------------------------

def create_optimizer(model: MERMentalHealthModel, config: ModelConfig,
                     stage: int = 1) -> torch.optim.AdamW:
    """Create AdamW optimizer with separate LR for LoRA params (optional).

    Stage-specific learning rates and weight decay handling:
    - LoRA parameters: no weight decay (standard LoRA practice)
    - Bias parameters: no weight decay
    - All other trainable params: weight_decay from config
    """
    lr = config.stage_1_lr if stage == 1 else config.stage_2_lr

    # Separate param groups
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # No weight decay for bias and LoRA parameters
        if "bias" in name or "lora_" in name or "norm" in name or "LayerNorm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=lr, betas=(0.9, 0.999), eps=1e-8)

    return optimizer


def create_scheduler(optimizer: torch.optim.AdamW, config: ModelConfig,
                     steps_per_epoch: int, stage: int = 1) -> torch.optim.lr_scheduler.LambdaLR:
    """Create cosine learning rate scheduler with linear warmup."""
    n_epochs = config.stage_1_epochs if stage == 1 else config.stage_2_epochs
    total_steps = steps_per_epoch * n_epochs
    warmup_steps = config.warmup_steps

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def modality_dropout_augmentation(
    audio_mask: torch.Tensor, video_mask: torch.Tensor, text_mask: torch.Tensor,
    random_dropout_prob: float = 0.15,
    clinical_correlated_dropout_prob: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply modality dropout augmentation during training.

    Simulates both random modality dropout and clinically-correlated
    dropout patterns (e.g., when patient turns away, both video and audio
    may be affected).

    Args:
        audio_mask: (B, T) bool — original audio availability
        video_mask: (B, T) bool — original video availability
        text_mask: (B, T) bool — original text availability
        random_dropout_prob: per-modality random dropout probability
        clinical_correlated_dropout_prob: correlated dropout probability

    Returns:
        Augmented masks (audio, video, text) with some entries set to False
    """
    B, T = audio_mask.shape
    device = audio_mask.device

    # Random per-modality dropout
    if random_dropout_prob > 0:
        rand_a = torch.rand(B, T, device=device) < random_dropout_prob
        rand_v = torch.rand(B, T, device=device) < random_dropout_prob
        rand_t = torch.rand(B, T, device=device) < random_dropout_prob

        audio_mask = audio_mask & ~rand_a
        video_mask = video_mask & ~rand_v
        text_mask = text_mask & ~rand_t

    # Clinically-correlated dropout patterns
    if clinical_correlated_dropout_prob > 0:
        # Pattern 1: Vision drops during emotional distress (VIDEO only)
        # Simulated: random utterances lose vision
        pattern_v = torch.rand(B, T, device=device) < clinical_correlated_dropout_prob
        video_mask = video_mask & ~pattern_v

        # Pattern 2: Audio drops in noisy environment (AUDIO only)
        pattern_a = torch.rand(B, T, device=device) < clinical_correlated_dropout_prob
        audio_mask = audio_mask & ~pattern_a

        # Pattern 3: Text drops when patient non-verbal (TEXT + potential VIDEO)
        pattern_t = torch.rand(B, T, device=device) < clinical_correlated_dropout_prob
        text_mask = text_mask & ~pattern_t

    return audio_mask, video_mask, text_mask
