"""
Model configuration for Multimodal Emotion Recognition (MER)
for Mental Health Diagnostics.

Architecture: Cross-Modal Gated Exchange Fusion (CM-GEF)
              + Session-Level Hierarchical Transformer (SLHT)
              + Clinical Concept Bottleneck (CCB)

Domain: Speech/Audio + CV + LM (Multimodal Fusion)
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class ModelConfig:
    # ──────────────────────────────────────────────────────────
    # Task specification
    # ──────────────────────────────────────────────────────────
    n_phq8_items: int = 8
    phq8_threshold: int = 10                         # PHQ-8 >= 10 -> depression positive
    n_clinical_concepts: int = 8                      # PHQ-8-aligned intermediate concepts
    task_type: str = "regression"                     # "regression" | "classification" | "both"

    # ──────────────────────────────────────────────────────────
    # Audio encoder (WavLM)
    # ──────────────────────────────────────────────────────────
    audio_encoder_name: str = "wavlm-base-plus"       # "wavlm-base-plus" | "wavlm-small"
    audio_feat_dim: int = 768                         # output dimension of audio encoder
    audio_sample_rate: int = 16000
    audio_max_seconds: float = 10.0                   # max utterance length in seconds
    audio_n_mels: int = 80
    audio_n_fft: int = 400
    audio_hop_length: int = 160
    audio_freeze_encoder: bool = True                 # freeze backbone in stage 2

    # ──────────────────────────────────────────────────────────
    # Video encoder (DINOv2)
    # ──────────────────────────────────────────────────────────
    video_encoder_name: str = "dinov2-small"          # "dinov2-small" | "dinov2-base"
    video_feat_dim: int = 384                         # output dimension of video encoder
    video_n_frames: int = 30                          # uniformly sampled frames per utterance
    video_frame_size: int = 224
    video_patch_size: int = 14
    video_freeze_encoder: bool = True

    # ──────────────────────────────────────────────────────────
    # Text encoder (MentalBERT)
    # ──────────────────────────────────────────────────────────
    text_encoder_name: str = "mentalbert"             # "mentalbert" | "deberta-v3-base"
    text_feat_dim: int = 768
    text_max_length: int = 128
    text_freeze_encoder: bool = True

    # ──────────────────────────────────────────────────────────
    # Common projection (unify all modalities)
    # ──────────────────────────────────────────────────────────
    proj_dim: int = 256                                # common latent dimension
    proj_n_layers: int = 2                             # MLP depth for projection
    proj_dropout: float = 0.1
    proj_activation: str = "gelu"

    # ──────────────────────────────────────────────────────────
    # Cross-Modal Gated Exchange Fusion (CM-GEF)
    # ──────────────────────────────────────────────────────────
    n_exchange_heads: int = 4                          # multi-head exchange attention
    exchange_hidden_dim: int = 128                     # gate MLP hidden size
    exchange_dropout: float = 0.1
    n_exchange_layers: int = 2                         # stacked exchange blocks
    exchange_norm: str = "layer_norm"                  # "layer_norm" | "batch_norm"
    exchange_residual: bool = True                     # residual connections in exchange

    # ──────────────────────────────────────────────────────────
    # Modality Quality Estimators
    # ──────────────────────────────────────────────────────────
    n_audio_quality_classes: int = 3                   # 0=clean, 1=noisy, 2=very_noisy
    n_video_quality_classes: int = 3                   # 0=visible, 1=occluded, 2=absent
    n_text_quality_classes: int = 3                    # 0=informative, 1=short, 2=empty
    quality_estimator_hidden: int = 64
    quality_estimator_dropout: float = 0.1

    # ──────────────────────────────────────────────────────────
    # Session-Level Hierarchical Transformer (SLHT)
    # ──────────────────────────────────────────────────────────
    n_therapeutic_phases: int = 4                      # rapport, exploration, intervention, closure
    phase_embed_dim: int = 64
    d_session_model: int = 256                         # session transformer hidden dim
    n_session_layers: int = 4
    n_session_heads: int = 8
    session_d_model_ff: int = 1024                     # session FFN expansion
    session_dropout: float = 0.1
    max_utterances: int = 512                          # max utterances per session

    # Position encoding for session
    session_pos_encoding: str = "learned"              # "learned" | "sinusoidal"
    max_session_length_minutes: int = 60

    # ──────────────────────────────────────────────────────────
    # Clinical Concept Bottleneck (CCB)
    # ──────────────────────────────────────────────────────────
    concept_names: Tuple[str, ...] = (
        "anhedonia",                                    # PHQ-8 item 1: loss of interest/pleasure
        "depressed_mood",                               # PHQ-8 item 2: feeling down/depressed
        "sleep_disturbance",                            # PHQ-8 item 3: sleep problems
        "fatigue",                                      # PHQ-8 item 4: low energy/fatigue
        "appetite_change",                              # PHQ-8 item 5: appetite changes
        "guilt_worthlessness",                          # PHQ-8 item 6: guilt/worthlessness
        "concentration_problems",                       # PHQ-8 item 7: trouble concentrating
        "psychomotor_change",                           # PHQ-8 item 8: slow/restless movement
    )
    concept_hidden_dim: int = 128
    concept_dropout: float = 0.15

    # ──────────────────────────────────────────────────────────
    # Prediction heads
    # ──────────────────────────────────────────────────────────
    regression_head: bool = True                        # predict PHQ-8 total score
    classification_head: bool = True                    # predict binary depression
    item_prediction_head: bool = True                   # predict per-item PHQ-8 scores
    head_hidden_dim: int = 64

    # ──────────────────────────────────────────────────────────
    # LoRA adaptation
    # ──────────────────────────────────────────────────────────
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = (
        "query", "value", "key",
        "output.dense", "intermediate.dense",
    )
    # Which components receive LoRA in stage 2
    lora_adapt_encoders: bool = True
    lora_adapt_fusion: bool = True
    lora_adapt_session: bool = True
    lora_adapt_concept: bool = False                    # CCB is small; train fully

    # ──────────────────────────────────────────────────────────
    # Two-stage training
    # ──────────────────────────────────────────────────────────
    stage_1_lr: float = 1e-4
    stage_1_epochs: int = 50
    stage_1_batch_size: int = 32                        # per-GPU
    stage_2_lr: float = 5e-5
    stage_2_epochs: int = 30
    stage_2_batch_size: int = 16
    warmup_steps: int = 1000
    weight_decay: float = 0.01

    # ──────────────────────────────────────────────────────────
    # Loss weights
    # ──────────────────────────────────────────────────────────
    loss_mae_weight: float = 1.0
    loss_bce_weight: float = 0.5
    loss_item_weight: float = 0.3
    loss_concept_supervision_weight: float = 0.2        # only if concept labels exist
    loss_dropout_reconstruction_weight: float = 0.1     # auxiliary in stage 1

    # ──────────────────────────────────────────────────────────
    # Modality dropout simulation (training)
    # ──────────────────────────────────────────────────────────
    random_dropout_prob: float = 0.15                   # per-modality random dropout during training
    clinical_correlated_dropout_prob: float = 0.1       # clinically-correlated pattern dropout

    # ──────────────────────────────────────────────────────────
    # Inference / deployment
    # ──────────────────────────────────────────────────────────
    streaming_mode: bool = False                        # True = per-utterance without session context
    dtype: str = "bfloat16"
    use_bias: bool = False
    seed: int = 42

    # ──────────────────────────────────────────────────────────
    # Test / debug — load pretrained or use random init
    # ──────────────────────────────────────────────────────────
    pretrained: bool = False                            # False = random init for smoke test
