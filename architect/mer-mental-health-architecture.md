# Multimodal Emotion Recognition for Mental Health Diagnostics — Architecture Design

**Date:** 2026-05-26
**Architect:** ml-architect
**Upstream:** ml-research → `/artifacts/j_0NmpJCXSm-5u/work/research/mer-mental-health-research.md` (read and incorporated)

---

## Step 0 — Domain Identification

| Domain | Role | Design concerns |
|---|---|---|
| **Speech/Audio** | Unimodal encoder (prosody, tone, hesitations) | WavLM backbone; noise robustness; quality estimation; streaming-capable |
| **Computer Vision** | Unimodal encoder (facial expressions, AUs) | DINOv2 backbone; occlusion handling; pose robustness; quality estimation |
| **Language Models** | Unimodal encoder (semantic content, sentiment) | MentalBERT backbone; short-utterance robustness; clinical vocabulary |
| **Multimodal Fusion** | Central architecture problem | Cross-modal gated exchange; missing-modality robustness; session-level temporal modeling; clinical concept bottleneck |

**Design constraint:** The architecture must be **lightweight enough for real-time inference** on a single A100 (or edge deployment with smaller backbones) while producing **clinically-grounded, interpretable** depression severity predictions.

---

## Step 1 — Upstream Research Contract Summary

Read from `mer-mental-health-research.md`. Key binding items:

| Item | Content |
|---|---|
| **task_level** | level_1 |
| **research_question** | Can a lightweight cross-modal gated fusion architecture with hierarchical session-level temporal modeling achieve clinically viable emotion recognition (MAE < 3.0 on E-DAIC) while maintaining robustness to clinically-correlated modality dropout and demographic fairness? |
| **novelty_claims** | (1) Cross-modal gated exchange on clinical data with correlated dropout (grounded); (2) Session-level therapeutic-phase temporal modeling (grounded); (3) Two-stage SSL+LoRA adaptation (hypothesis); (4) Fairness disaggregation (grounded gap); (5) Clinical concept bottleneck interpretability (grounded gap) |
| **baseline_requirements** | Unimodal (text/audio/video), late fusion, MDD-MARF, early fusion, ablated gating |
| **evaluation_requirements** | MAE/RMSE on DAIC-WOZ/E-DAIC; binary depression metrics; 6 dropout + 3 clinical dropout conditions; session-level sustained affect F1; demographic fairness; clinician interpretability study; latency/memory/FLOPs |
| **blocking_unknowns** | DAIC-WOZ demographic diversity; clinician availability; data privacy/IRB; no established clinically-correlated dropout benchmark |
| **expected_targets** | MAE < 3.0, RMSE < 3.5 on E-DAIC; <15% degradation under 40% dropout; ≥10% F1 gain on sustained affect |

---

## Step 2 — ModelConfig Dataclass

```python
from dataclasses import dataclass, field
from typing import Optional, Tuple

@dataclass
class ModelConfig:
    # ──────────────────────────────────────────
    # Task specification
    # ──────────────────────────────────────────
    n_phq8_items: int = 8
    phq8_threshold: int = 10                     # PHQ-8 >= 10 → depression positive
    n_clinical_concepts: int = 8                  # PHQ-8-aligned intermediate concepts
    task_type: str = "regression"                 # "regression" | "classification" | "both"

    # ──────────────────────────────────────────
    # Audio encoder (WavLM)
    # ──────────────────────────────────────────
    audio_encoder_name: str = "wavlm-base-plus"   # "wavlm-base-plus" | "wavlm-small"
    audio_feat_dim: int = 768                     # output dimension of audio encoder
    audio_sample_rate: int = 16000
    audio_max_seconds: float = 10.0               # max utterance length in seconds
    audio_n_mels: int = 80
    audio_n_fft: int = 400
    audio_hop_length: int = 160
    audio_freeze_encoder: bool = True             # freeze backbone in stage 2

    # ──────────────────────────────────────────
    # Video encoder (DINOv2)
    # ──────────────────────────────────────────
    video_encoder_name: str = "dinov2-small"      # "dinov2-small" | "dinov2-base"
    video_feat_dim: int = 384                     # output dimension of video encoder
    video_n_frames: int = 30                      # uniformly sampled frames per utterance
    video_frame_size: int = 224
    video_patch_size: int = 14
    video_freeze_encoder: bool = True

    # ──────────────────────────────────────────
    # Text encoder (MentalBERT)
    # ──────────────────────────────────────────
    text_encoder_name: str = "mentalbert"          # "mentalbert" | "deberta-v3-base"
    text_feat_dim: int = 768
    text_max_length: int = 128
    text_freeze_encoder: bool = True

    # ──────────────────────────────────────────
    # Common projection (unify all modalities)
    # ──────────────────────────────────────────
    proj_dim: int = 256                            # common latent dimension
    proj_n_layers: int = 2                         # MLP depth for projection
    proj_dropout: float = 0.1
    proj_activation: str = "gelu"

    # ──────────────────────────────────────────
    # Cross-Modal Gated Exchange Fusion (CM-GEF)
    # ──────────────────────────────────────────
    n_exchange_heads: int = 4                      # multi-head exchange attention
    exchange_hidden_dim: int = 128                 # gate MLP hidden size
    exchange_dropout: float = 0.1
    n_exchange_layers: int = 2                     # stacked exchange blocks
    exchange_norm: str = "layer_norm"              # "layer_norm" | "batch_norm"
    exchange_residual: bool = True                 # residual connections in exchange

    # ──────────────────────────────────────────
    # Modality Quality Estimators
    # ──────────────────────────────────────────
    n_audio_quality_classes: int = 3               # 0=clean, 1=noisy, 2=very_noisy
    n_video_quality_classes: int = 3               # 0=visible, 1=occluded, 2=absent
    n_text_quality_classes: int = 3                # 0=informative, 1=short, 2=empty
    quality_estimator_hidden: int = 64
    quality_estimator_dropout: float = 0.1

    # ──────────────────────────────────────────
    # Session-Level Hierarchical Transformer (SLHT)
    # ──────────────────────────────────────────
    n_therapeutic_phases: int = 4                  # rapport, exploration, intervention, closure
    phase_embed_dim: int = 64
    d_session_model: int = 256                     # session transformer hidden dim
    n_session_layers: int = 4
    n_session_heads: int = 8
    session_d_model_ff: int = 1024                 # session FFN expansion
    session_dropout: float = 0.1
    max_utterances: int = 512                      # max utterances per session

    # Position encoding for session
    session_pos_encoding: str = "learned"          # "learned" | "sinusoidal"
    max_session_length_minutes: int = 60

    # ──────────────────────────────────────────
    # Clinical Concept Bottleneck (CCB)
    # ──────────────────────────────────────────
    concept_names: Tuple[str, ...] = (
        "anhedonia",                                # PHQ-8 item 1: loss of interest/pleasure
        "depressed_mood",                           # PHQ-8 item 2: feeling down/depressed
        "sleep_disturbance",                        # PHQ-8 item 3: sleep problems
        "fatigue",                                  # PHQ-8 item 4: low energy/fatigue
        "appetite_change",                          # PHQ-8 item 5: appetite changes
        "guilt_worthlessness",                      # PHQ-8 item 6: guilt/worthlessness
        "concentration_problems",                   # PHQ-8 item 7: trouble concentrating
        "psychomotor_change",                       # PHQ-8 item 8: slow/restless movement
    )
    concept_hidden_dim: int = 128
    concept_dropout: float = 0.15

    # ──────────────────────────────────────────
    # Prediction heads
    # ──────────────────────────────────────────
    regression_head: bool = True                   # predict PHQ-8 total score
    classification_head: bool = True               # predict binary depression
    item_prediction_head: bool = True              # predict per-item PHQ-8 scores
    head_hidden_dim: int = 64

    # ──────────────────────────────────────────
    # LoRA adaptation
    # ──────────────────────────────────────────
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
    lora_adapt_concept: bool = False               # CCB is small; train fully

    # ──────────────────────────────────────────
    # Two-stage training
    # ──────────────────────────────────────────
    stage_1_lr: float = 1e-4
    stage_1_epochs: int = 50
    stage_1_batch_size: int = 32                   # per-GPU
    stage_2_lr: float = 5e-5
    stage_2_epochs: int = 30
    stage_2_batch_size: int = 16
    warmup_steps: int = 1000
    weight_decay: float = 0.01

    # ──────────────────────────────────────────
    # Loss weights
    # ──────────────────────────────────────────
    loss_mae_weight: float = 1.0
    loss_bce_weight: float = 0.5
    loss_item_weight: float = 0.3
    loss_concept_supervision_weight: float = 0.2   # only if concept labels exist
    loss_dropout_reconstruction_weight: float = 0.1  # auxiliary in stage 1

    # ──────────────────────────────────────────
    # Modality dropout simulation (training)
    # ──────────────────────────────────────────
    random_dropout_prob: float = 0.15              # per-modality random dropout during training
    clinical_correlated_dropout_prob: float = 0.1  # clinically-correlated pattern dropout

    # ──────────────────────────────────────────
    # Inference / deployment
    # ──────────────────────────────────────────
    streaming_mode: bool = False                   # True = per-utterance without session context
    dtype: str = "bfloat16"
    use_bias: bool = False
    seed: int = 42
```

---

## Step 3 — Core Block Pseudocode

### Block 1: Modality Quality Estimator

```python
def estimate_modality_quality(features: Tensor, modality: str, config: ModelConfig) -> Tensor:
    """
    Predict modality reliability from encoder features.
    
    Args:
        features: (B, d_feat) — pooled unimodal encoder output
        modality: "audio" | "video" | "text"
        config: ModelConfig
    
    Returns:
        reliability: (B,) — scalar in [0, 1], 1 = fully reliable
    """
    n_classes = {
        "audio": config.n_audio_quality_classes,       # 3: clean, noisy, very_noisy
        "video": config.n_video_quality_classes,       # 3: visible, occluded, absent
        "text":  config.n_text_quality_classes,        # 3: informative, short, empty
    }[modality]
    
    hidden_dim = config.quality_estimator_hidden
    dropout = config.quality_estimator_dropout
    
    # Lightweight MLP head on pooled encoder features
    h = layer_norm(features)
    h = linear(h, d_feat -> hidden_dim)
    h = gelu(h)
    h = dropout(h)
    logits = linear(h, hidden_dim -> n_classes)     # (B, n_classes)
    
    # Quality score = probability of the "clean/visible/informative" class
    probs = softmax(logits, dim=-1)                 # (B, n_classes)
    reliability = probs[:, 0]                       # (B,) class 0 = highest quality
    
    return reliability
```

**Training:** The quality estimator is pre-trained on synthetic data — take clean utterances,
apply known corruptions (noise injection, face occlusion, text truncation), and supervise
with the corruption label. During clinical fine-tuning, the estimator runs in inference mode
or is fine-tuned with end-to-end signal via the gating mechanism.

---

### Block 2: Cross-Modal Gated Exchange Fusion (CM-GEF)

```python
def cross_modal_gated_exchange(
    audio_feat: Tensor,   # (B, d_audio_feat) — pooled WavLM utterance features
    video_feat: Tensor,   # (B, d_video_feat) — pooled DINOv2 utterance features
    text_feat: Tensor,    # (B, d_text_feat)  — pooled MentalBERT utterance features
    config: ModelConfig,
    training: bool = True,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Fuse three modalities with adaptive gating that substitutes unreliable
    modalities with cross-modal context.
    
    Returns:
        utterance_feat: (B, config.proj_dim) — fused representation
        gate_weights: (B, 3) — [gate_a, gate_v, gate_t] each in [0,1]
        modality_weights: (B, 3) — softmax over reliability scores
    """
    d = config.proj_dim
    
    # ── Stage 1: Project to common dimension ──
    audio_proj = projection_mlp(audio_feat, config.audio_feat_dim, d, 
                                n_layers=config.proj_n_layers, dropout=config.proj_dropout)
    video_proj = projection_mlp(video_feat, config.video_feat_dim, d,
                                n_layers=config.proj_n_layers, dropout=config.proj_dropout)
    text_proj  = projection_mlp(text_feat,  config.text_feat_dim,  d,
                                n_layers=config.proj_n_layers, dropout=config.proj_dropout)
    # Each: (B, d)
    
    # ── Stage 2: Estimate modality reliability ──
    r_audio = estimate_modality_quality(audio_feat, "audio", config)   # (B,)
    r_video = estimate_modality_quality(video_feat, "video", config)   # (B,)
    r_text  = estimate_modality_quality(text_feat,  "text",  config)   # (B,)
    
    # ── Stage 3: Multi-head exchange attention ──
    # For each modality, compute cross-modal context from other two modalities
    # We use multi-head dot-product attention where query = current modality,
    # keys/values = other modalities
    
    def exchange_attention(query: Tensor, keys: List[Tensor], values: List[Tensor],
                           n_heads: int) -> Tensor:
        """Multi-head cross-modal attention with one query and multiple key/value sources."""
        # Stack key-value pairs from all OTHER modalities
        K = torch.stack(keys, dim=1)     # (B, n_sources, d)
        V = torch.stack(values, dim=1)   # (B, n_sources, d)
        Q = query.unsqueeze(1)           # (B, 1, d)
        
        # Multi-head
        head_dim = d // n_heads
        Q = reshape_for_heads(Q, n_heads)   # (B, n_heads, 1, head_dim)
        K = reshape_for_heads(K, n_heads)   # (B, n_heads, n_sources, head_dim)
        V = reshape_for_heads(V, n_heads)   # (B, n_heads, n_sources, head_dim)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / sqrt(head_dim)  # (B, n_heads, 1, n_sources)
        attn = softmax(scores, dim=-1)
        out = torch.matmul(attn, V)                                     # (B, n_heads, 1, head_dim)
        out = merge_heads(out)                                          # (B, d)
        return out, attn
    
    # Audio gets context from video + text
    ctx_audio, _ = exchange_attention(
        audio_proj, [video_proj, text_proj], [video_proj, text_proj],
        config.n_exchange_heads
    )
    # Video gets context from audio + text
    ctx_video, _ = exchange_attention(
        video_proj, [audio_proj, text_proj], [audio_proj, text_proj],
        config.n_exchange_heads
    )
    # Text gets context from audio + video
    ctx_text, _ = exchange_attention(
        text_proj, [audio_proj, video_proj], [audio_proj, video_proj],
        config.n_exchange_heads
    )
    # Each: (B, d)
    
    # ── Stage 4: Compute per-modality gates ──
    # Gate = f(reliability, self_features, cross_context)
    # Higher reliability → gate closer to 1 (trust self)
    # Lower reliability  → gate closer to 0 (substitute with context)
    
    def compute_gate(reliability: Tensor, self_feat: Tensor, ctx_feat: Tensor, 
                     hidden_dim: int) -> Tensor:
        """Compute per-feature-dimension gate from reliability + features."""
        r = reliability.unsqueeze(-1).expand(-1, self_feat.size(-1))   # (B, d)
        gate_input = torch.stack([self_feat, ctx_feat, r], dim=-1)     # (B, d, 3)
        gate_input = gate_input.flatten(-2, -1)                        # (B, d*3) — but we want per-dim
        # Better: concat along feature dim
        gate_input = torch.cat([self_feat, ctx_feat, r], dim=-1)      # (B, 2d + d) = (B, 3d)
        h = linear(gate_input, 3*self_feat.size(-1), hidden_dim)
        h = gelu(h)
        h = linear(h, hidden_dim, self_feat.size(-1))
        gate = torch.sigmoid(h)                                         # (B, d) ∈ [0, 1]
        return gate
    
    gate_a = compute_gate(r_audio, audio_proj, ctx_audio, config.exchange_hidden_dim)
    gate_v = compute_gate(r_video, video_proj, ctx_video, config.exchange_hidden_dim)
    gate_t = compute_gate(r_text,  text_proj,  ctx_text,  config.exchange_hidden_dim)
    # Each: (B, d) — per-feature-dimension gate
    
    # ── Stage 5: Gated substitution ──
    audio_fused = gate_a * audio_proj + (1 - gate_a) * ctx_audio   # (B, d)
    video_fused = gate_v * video_proj + (1 - gate_v) * ctx_video   # (B, d)
    text_fused  = gate_t * text_proj  + (1 - gate_t) * ctx_text    # (B, d)
    
    # ── Stage 6: Adaptive reliability-weighted aggregation ──
    modality_weights = softmax(
        torch.stack([r_audio, r_video, r_text], dim=-1), dim=-1
    )                                                               # (B, 3)
    
    fused = (
        modality_weights[:, 0:1] * audio_fused +
        modality_weights[:, 1:2] * video_fused +
        modality_weights[:, 2:3] * text_fused
    )                                                               # (B, d)
    
    # Optional: stack multiple exchange layers
    for _ in range(1, config.n_exchange_layers):
        if config.exchange_residual:
            residual = fused
        fused = layer_norm(fused)
        fused = fused + feed_forward(fused, d, config.exchange_hidden_dim, 
                                      config.exchange_dropout)
        if config.exchange_residual:
            fused = residual + fused
    
    return fused, torch.stack([gate_a.mean(-1), gate_v.mean(-1), gate_t.mean(-1)], dim=-1), modality_weights


def feed_forward(x: Tensor, d_model: int, d_ff: int, dropout: float) -> Tensor:
    """Standard FFN: linear → activation → dropout → linear."""
    h = linear(x, d_model, d_ff)
    h = gelu(h)
    h = dropout(h)
    h = linear(h, d_ff, d_model)
    return h
```

**Key design rationale:**
- **Per-feature-dimension gating** (not scalar): allows the gate to selectively suppress/keep
  individual feature dimensions, providing finer-grained control than modality-level scalar gates.
- **Multi-head exchange attention**: enables each modality to attend to different aspects of the
  other modalities (e.g., audio attends to lip movements from video and sentiment from text).
- **Reliability-weighted aggregation**: final fusion weights are proportional to estimated quality,
  so unreliable modalities contribute minimally even to the combined representation.

---

### Block 3: Session-Level Hierarchical Transformer (SLHT)

```python
def session_level_hierarchical_transformer(
    utterance_features: Tensor,    # (B, T, proj_dim) — T utterances per session
    phase_labels: Tensor,          # (B, T) — therapeutic phase index for each utterance ∈ {0,1,2,3}
    timestamps: Tensor,            # (B, T) — timestamp in seconds from session start
    config: ModelConfig,
) -> Tuple[Tensor, Tensor]:
    """
    Model session-level emotion dynamics with therapeutic phase awareness.
    
    Returns:
        session_feat: (B, d_session_model) — session-level representation
        phase_repr: (B, n_phases, d_session_model) — per-phase representations
    """
    B, T, d_in = utterance_features.shape
    d_model = config.d_session_model
    
    # ── 1. Phase embedding ──
    phase_emb = nn.Embedding(config.n_therapeutic_phases, config.phase_embed_dim)
    phase_emb = phase_emb(phase_labels)                          # (B, T, phase_embed_dim)
    
    # ── 2. Temporal position encoding (utterance index & timestamps) ──
    if config.session_pos_encoding == "learned":
        pos_emb = nn.Embedding(config.max_utterances, d_model)
        positions = torch.arange(T, device=utterance_features.device).unsqueeze(0).expand(B, -1)
        pos_enc = pos_emb(positions)                              # (B, T, d_model)
    else:
        pos_enc = sinusoidal_position_encoding(T, d_model)        # (B, T, d_model)
    
    # ── 3. Project utterance features to session model dim ──
    x = linear(utterance_features, d_in + config.phase_embed_dim, d_model)
    x = x + pos_enc                                               # (B, T, d_model)
    
    # ── 4. Session-level Transformer encoder ──
    # Note: Bidirectional (not causal) — full session is available at inference
    for layer_idx in range(config.n_session_layers):
        residual = x
        x = layer_norm(x)
        
        # Multi-head self-attention over all utterances
        x = multi_head_self_attention(
            x, n_heads=config.n_session_heads,
            d_model=d_model, dropout=config.session_dropout,
        )
        x = residual + x
        
        residual = x
        x = layer_norm(x)
        x = feed_forward(x, d_model, config.session_d_model_ff, config.session_dropout)
        x = residual + x
    
    # ── 5. Phase-level aggregation ──
    # Pool utterances within each therapeutic phase via learned attention
    phase_repr_list = []
    for phase_idx in range(config.n_therapeutic_phases):
        phase_mask = (phase_labels == phase_idx).float()                  # (B, T)
        
        # Masked mean (fallback if no utterances in this phase)
        phase_mean = (x * phase_mask.unsqueeze(-1)).sum(dim=1) / \
                     (phase_mask.sum(dim=-1, keepdim=True) + 1e-8)        # (B, d_model)
        
        # Cross-attend phase_mean to all utterances for fine-grained context
        phase_query = phase_mean.unsqueeze(1)                             # (B, 1, d_model)
        phase_attended = cross_attention(
            query=phase_query, key=x, value=x,
            key_mask=(phase_mask > 0)
        )                                                                 # (B, 1, d_model)
        phase_repr_list.append(phase_attended.squeeze(1))
    
    phase_repr = torch.stack(phase_repr_list, dim=1)                     # (B, n_phases, d_model)
    
    # ── 6. Session representation via phase-weighted pooling ──
    # Learnable importance weighting over phases
    phase_importance = linear(phase_repr, d_model, 1).squeeze(-1)        # (B, n_phases)
    phase_importance = softmax(phase_importance, dim=-1)
    
    session_feat = (phase_importance.unsqueeze(-1) * phase_repr).sum(dim=1)  # (B, d_model)
    
    # Also retain global utterance context
    global_context = x.mean(dim=1)                                       # (B, d_model)
    session_feat = session_feat + global_context                         # residual connection
    
    return session_feat, phase_repr
```

**Phase assignment strategy:** Phase labels can be obtained via:
1. **Manual** — therapist annotates phase boundaries (gold standard, small-scale)
2. **Heuristic** — based on time windows (rapport = first 5 min, closure = last 5 min, etc.)
3. **Learned** — train a lightweight phase classifier on utterance features (self-supervised
   via transition detection)

For DAIC-WOZ structure, phases correspond to: (0) rapport-building / introductory questions,
(1) exploration / open-ended probes, (2) intervention / targeted clinical questions,
(3) closure / wrap-up.

---

### Block 4: Clinical Concept Bottleneck (CCB)

```python
def clinical_concept_bottleneck(
    session_feat: Tensor,        # (B, d_session_model)
    config: ModelConfig,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Map session representation through clinically-grounded concept space
    to PHQ-8 item scores and total depression severity.
    
    Returns:
        phq8_total:      (B,) — predicted PHQ-8 total score [0, 24]
        phq8_items:      (B, 8) — per-item severity [0, 3] each
        concept_scores:  (B, n_clinical_concepts) — concept activation ∈ [0, 1]
        depression_prob: (B,) — probability PHQ-8 >= 10
    """
    d = config.d_session_model
    n_concepts = config.n_clinical_concepts  # = 8, aligned to PHQ-8 items
    n_items = config.n_phq8_items            # = 8
    
    # ── 1. Concept predictor ──
    h = layer_norm(session_feat)
    h = linear(h, d, config.concept_hidden_dim)
    h = gelu(h)
    h = dropout(h, config.concept_dropout)
    h = linear(h, config.concept_hidden_dim, config.concept_hidden_dim)
    h = gelu(h)
    h = dropout(h, config.concept_dropout)
    
    concept_logits = linear(h, config.concept_hidden_dim, n_concepts)  # (B, n_concepts)
    concept_scores = torch.sigmoid(concept_logits)                     # (B, n_concepts) ∈ [0, 1]
    
    # ── 2. PHQ-8 item predictor (from concepts) ──
    # Each concept primarily influences a specific PHQ-8 item, with cross-talk
    # via a learned weight matrix W ∈ R^(n_items × n_concepts)
    item_logits = linear(concept_scores, n_concepts, n_items)          # (B, 8)
    phq8_items = 3.0 * torch.sigmoid(item_logits)                      # (B, 8) ∈ [0, 3]
    
    # ── 3. PHQ-8 total score ──
    if config.item_prediction_head:
        phq8_total = phq8_items.sum(dim=-1)                            # (B,) [0, 24]
    else:
        phq8_total = linear(session_feat, d, 1).squeeze(-1)            # (B,)
    
    # ── 4. Binary depression classification ──
    if config.classification_head:
        # Classify from concept scores (clinical prior: depression = multiple concepts)
        depression_logit = linear(concept_scores, n_concepts, 1).squeeze(-1)  # (B,)
        depression_prob = torch.sigmoid(depression_logit)                     # (B,)
    else:
        depression_prob = None
    
    return phq8_total, phq8_items, concept_scores, depression_prob
```

**Interpretability via concept scoring:** Each concept score ∈ [0, 1] corresponds to a
PHQ-8-aligned clinical construct. Clinicians can inspect which concepts contributed most
to the prediction, making the model's reasoning transparent and falsifiable.

**Weight regularization:** The concept→item weight matrix `W` can be initialized with a
diagonal-dominant prior (each concept → corresponding item), encouraging interpretable
mapping. Sparsity penalty (L1) on off-diagonal elements enforces clinical grounding.

---

## Step 4 — Full Architecture Diagram (ASCII)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MER Mental Health Architecture                         │
│         Cross-Modal Gated Exchange + Session-Level Transformer            │
│                      + Clinical Concept Bottleneck                         │
└─────────────────────────────────────────────────────────────────────────┘

                            ┌─────────────────┐
                            │   PHQ-8 Total    │ ← Regression target [0,24]
                            │  (B,)            │
                            └────────┬────────┘
                                     ▲
                            ┌────────┴────────┐
                            │  PHQ-8 Items     │ ← 8 item scores [0,3] each
                            │  (B, 8)          │
                            └────────┬────────┘
                                     ▲
                            ┌────────┴────────┐
                            │ Clinical Concept  │ ← Interpretable bottleneck
                            │ Bottleneck (CCB) │   8 sigmoid-activated concepts
                            │  (B, 8)          │
                            └────────┬────────┘
                                     ▲
                   ┌─────────────────┼─────────────────┐
                   │     Session Representation        │
                   │     (B, d_session_model)          │
                   └─────────────────┬─────────────────┘
                                     ▲
        ┌────────────────────────────┼────────────────────────────┐
        │        Session-Level Hierarchical Transformer (SLHT)    │
        │              ┌────────────────────────────┐            │
        │              │  Position Encoding         │            │
        │              │  (utterance index + time)   │            │
        │              └────────────┬───────────────┘            │
        │              ┌────────────┴───────────────┐            │
        │              │  Multi-Head Self-Attention │  × 4       │
        │              │  + FFN (bidirectional)     │   layers   │
        │              └────────────┬───────────────┘            │
        │              ┌────────────┴───────────────┐            │
        │              │ Phase-Level Pooling         │            │
        │              │ (rapport, exploration,      │            │
        │              │  intervention, closure)      │            │
        │              └────────────┬───────────────┘            │
        └───────────────────────────┼────────────────────────────┘
                                    ▲
     ┌──────────────────────────────┼──────────────────────────────┐
     │      ┌───────────────────────┴───────────────────────┐     │
     │      │     Cross-Modal Gated Exchange Fusion         │     │
     │      │                (CM-GEF)                        │     │
     │      │                                               │     │
     │      │  ┌──────────┐   ┌──────────┐   ┌──────────┐  │     │
     │      │  │ quality  │   │ quality  │   │ quality  │  │     │
     │      │  │estimator │   │estimator │   │estimator │  │     │
     │      │  │ (audio)  │   │ (video)  │   │ (text)   │  │     │
     │      │  └────┬─────┘   └────┬─────┘   └────┬─────┘  │     │
     │      │       │              │              │        │     │
     │      │  ┌────┴────┐   ┌────┴────┐   ┌────┴────┐    │     │
     │      │  │ Proj.   │   │ Proj.   │   │ Proj.   │    │     │
     │      │  │ MLP     │   │ MLP     │   │ MLP     │    │     │
     │      │  │ d→256   │   │ d→256   │   │ d→256   │    │     │
     │      │  └────┬────┘   └────┬────┘   └────┬────┘    │     │
     │      │       │              │              │        │     │
     │      │  ┌────┴──────────────┴──────────────┴────┐   │     │
     │      │  │    Multi-Head Exchange Attention      │   │     │
     │      │  │  (each modality → cross-modal context)│   │     │
     │      │  └──────────────────┬───────────────────┘   │     │
     │      │                     │                        │     │
     │      │  ┌──────────────────┴───────────────────┐   │     │
     │      │  │    Per-dimension Gated Substitution   │   │     │
     │      │  │    gate·self + (1-gate)·context       │   │     │
     │      │  └──────────────────┬───────────────────┘   │     │
     │      │                     │                        │     │
     │      │  ┌──────────────────┴───────────────────┐   │     │
     │      │  │  Reliability-Weighted Aggregation     │   │     │
     │      │  │  softmax(r_a, r_v, r_t) · fused_feats │   │     │
     │      │  └──────────────────┬───────────────────┘   │     │
     │      └─────────────────────┼───────────────────────┘     │
     │                            │                             │
     │                            ▼                             │
     │                    Utterance Feature                     │
     │                       (B, 256)                           │
     └───────────────────────────┬──────────────────────────────┘
                                 ▲
         ┌───────────────────────┴───────────────────────┐
         │                                               │
     ┌───┴───┐                                     ┌───┴───┐
     │ Audio  │                                     │ Video │
     │Encoder │                                     │Encoder│
     │(WavLM) │                                     │(DINOv2│
     │Base+)  │                                     │ Small)│
     │ d=768  │                                     │ d=384 │
     └───┬───┘                                     └───┬───┘
         │                                               │
     ┌───┴───────────────────────────────────────────────┴───┐
     │                    Text Encoder (MentalBERT) d=768     │
     └───────────────────────┬───────────────────────────────┘
                             │
                    ┌────────┴────────┐
                    │  Pre-processed   │
                    │    Utterance     │
                    │  (A+V+T aligned) │
                    └─────────────────┘

TRAINING PIPELINE:
┌─────────────────────────────────────────────────────────────────────┐
│ Stage 1: Large-Scale Pre-training (50 epochs)                       │
│   Data: CMU-MOSEI + MELD + IEMOCAP (39k videos, multimodal)       │
│   Frozen: WavLM, DINOv2, MentalBERT (use pretrained checkpoints)   │
│   Train: Projection MLPs, Quality Estimators, CM-GEF               │
│   Loss: emotion cls + modality dropout reconstruction               │
│   Simulated dropout: 15% random + 10% correlated                   │
│                                                                     │
│ Stage 2: Clinical Adaptation (30 epochs, LoRA)                     │
│   Data: DAIC-WOZ / E-DAIC (~189 sessions, ~80k utterances)        │
│   Frozen: WavLM, DINOv2, MentalBERT (with LoRA adapters)           │
│   Train: LoRA adapters + CM-GEF + SLHT + CCB (fully)              │
│   Loss: PHQ-8 MAE + item MSE + binary BCE + concept reg            │
│   Augmentation: clinically-correlated dropout patterns             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Step 5 — Inductive Bias Justifications

### Modality Quality Estimators
> **"We predict modality reliability from encoder features using a lightweight head trained on synthetic corruptions, because clinically-correlated modality quality (e.g., face occlusion during distress) must be distinguished from random noise for robust gating."**
> — Reliability is input-dependent (not learned static weights), enabling adaptive substitution per utterance.
> **Status:** `grounded` — clinician-validated dropout patterns can be simulated; no known prior clinical MER work uses per-utterance quality estimation.

### Per-Dimension Gated Substitution
> **"We use per-feature-dimension gates (not scalar modality gates) to allow partial feature substitution, because a modality may be unreliable in some feature subspaces (e.g., audio pitch is reliable but timbre is distorted) while remaining useful in others."**
> — Finer granularity than AMTE's scalar exchange weights.
> **Status:** `hypothesis` — intuition is sound but ablation must verify that per-dimension gates outperform scalar modality gates.

### Multi-Head Exchange Attention
> **"We use multi-head cross-modal attention to compute context for substitution, because different attention heads can specialize on different cross-modal patterns (e.g., audio attending to lip movements in video vs. sentiment in text)."**
> — Standard multi-head design, analogous to MulT (2019).
> **Status:** `grounded` — well-established in multimodal Transformers.

### Bidirectional Session Transformer (Not Causal)
> **"We use bidirectional self-attention over the full session (all utterances), because the entire clinical interview is available for post-session analysis and early emotional cues benefit from later context (e.g., a flat start may be clinically significant only when contrasted with later emotional reactivity)."**
> — For streaming use, switch to masked/autoregressive variant with KV-cache.
> **Status:** `grounded` — clinical interviews are offline-analyzed; streaming is secondary.

### Therapeutic Phase Embeddings
> **"We encode therapeutic phases as learned embeddings added to utterance features, because depression markers manifest differently across session phases (e.g., flat affect during rapport-building vs. emotional reactivity during intervention) and the model must contextualize utterances within their therapeutic stage."**
> — Novel temporal prior absent from DSIN, DSTC, and all prior clinical MER work.
> **Status:** `grounded` — clinical literature confirms phase-dependent emotional expression; no prior MER architecture models it.

### Clinical Concept Bottleneck
> **"We predict PHQ-8 scores through an interpretable intermediate concept layer aligned to DSM-5 depression criteria, because clinicians need to verify model reasoning against clinical constructs rather than trusting a black-box score."**
> — Concept bottleneck models (Koh et al., 2020) are established for interpretability; novel application in clinical MER.
> **Status:** `hypothesis` — concept→PHQ-8 mapping is intuitive but unvalidated; requires clinician Likert study for verification.

### Two-Stage Training (SSL Pre-train → LoRA Adapt)
> **"We pre-train the fusion module on large multimodal datasets (CMU-MOSEI) then adapt to clinical data with LoRA, because clinical labeled data is scarce (~189 sessions) and full fine-tuning risks overfitting or losing in-the-wild robustness."**
> — LoRA effectiveness in NLP/CV is established; application to multimodal clinical MER is novel.
> **Status:** `hypothesis` — requires controlled experiment against full fine-tuning with same compute budget.

### Reliability-Weighted Final Aggregation
> **"We weight the three fused modality representations by softmax-normalized reliability scores, because when all three modalities are reliable, the model should ensemble them; when one is unreliable, it should be suppressed at the final aggregation stage as a second line of defense after gated substitution."**
> — Dual mechanism (gate + weight) provides robustness redundancy.
> **Status:** `grounded` — sensible design; ablation will reveal if either mechanism alone suffices.

---

## Step 6 — Research-to-Architecture Traceability

| # | Research contract item | Architecture decision | Evidence status | Validation hook |
|---|---|---|---|---|
| 1 | **Claim:** Cross-modal gated exchange not validated on clinical data with clinically-correlated dropout | **CM-GEF block** with quality estimator + per-dimension gated substitution + reliability-weighted aggregation | `grounded` (gap from AMTE, Mi-CGA, Sadeghi) | Modality dropout stress test: evaluate under 6 random + 3 clinical-correlated patterns; measure MAE degradation vs. full-modality |
| 2 | **Claim:** Session-level therapeutic phase modeling absent from MER | **SLHT block** with phase embeddings (4 phases) + phase-level pooling | `grounded` (gap from DSIN, DSTC) | Sustained affect F1: compare SLHT vs. utterance-level aggregation; emotional reactivity F1 on therapist-probe utterances |
| 3 | **Hypothesis:** Two-stage SSL + LoRA can match full fine-tuning | **Training pipeline** Stage 1 (MOSEI/MELD pre-train) + Stage 2 (LoRA on DAIC-WOZ) | `hypothesis` (SSL in lab MER; LoRA in NLP/CV) | Controlled experiment: full fine-tune vs. LoRA with same compute; sweep n_labeled (5, 10, 25, 50, 100% of DAIC-WOZ) |
| 4 | **Claim:** No demographic fairness reporting in clinical MER | **Evaluation protocol** requiring disaggregated metrics by gender, ethnicity, age | `grounded` (confirming survey findings) | Fairness audit: report MAE/F1 per subgroup; flag groups with >5 pp gap from mean |
| 5 | **Claim:** Generated explanations not grounded in clinical taxonomies | **CCB block** with 8 PHQ-8-aligned concepts + concept→item mapping | `grounded` (gap from ECMC, AEFNet) | Clinician Likert study: rate concept→PHQ-8 mapping interpretability (≥3 clinicians) |
| 6 | **Baseline:** Compare against unimodal, late fusion, early fusion, MDD-MARF | **Evaluation suite** with 5 mandatory baselines; modular design allows dropping CM-GEF, SLHT, CCB independently | `grounded` (standard ML practice) | Benchmark runner: reproduce all baselines on same train/val/test splits |
| 7 | **Evaluation:** MAE/RMSE on DAIC-WOZ/E-DAIC | **Regression head** predicting PHQ-8 total + per-item scores | `grounded` | Standard regression metrics on held-out test set |
| 8 | **Evaluation:** Efficiency (latency, memory, FLOPs) | **Config parameter `streaming_mode`** + instrumentation for per-block profiling | `grounded` | Profiler hook: ms/utterance, peak GPU memory, FLOPs per sample; compare to MDD-MARF |
| 9 | **Blocking unknown:** Clinically-correlated dropout ground truth | **Dropout simulation protocol** — 3 clinical patterns (vision-drop-in-distress, audio-drop-in-noisy, text-drop-in-nonverbal) | `TODO: unverified` | Release dropout simulation code as benchmark contribution |
| 10 | **Blocking unknown:** DAIC-WOZ demographic diversity | **Fairness evaluation** using available demographic labels; if insufficient → report as limitation + evaluate on CMU-MOSEI splits | `TODO: unverified` | Demographic n-count table; confidence intervals per subgroup |

---

## Step 7 — Domain-Specific Considerations

### Speech/Audio

- **Input representation:** Raw waveform → WavLM Base Plus (self-supervised speech model).
  WavLM is chosen over HuBERT/Wav2Vec 2.0 because its denoising and speaker-aware pre-training
  provides better emotion-relevant representations (confirmed by 2024–2025 paralinguistic benchmarks).
- **Causality:** Offline (full utterance available). For streaming, WavLM can operate on
  incremental frames but quality degrades. Streaming mode is a **config switch**, not the
  primary target.
- **Local vs. global acoustic structure:** WavLM captures phoneme-level (~50ms) and word-level
  patterns via Transformer layers. Utterance pooling (mean over time) captures global prosody.
  The quality estimator detects acoustic degradation (noise, codec compression).
- **Utterance pooling:** We use attention-weighted mean pooling over the WavLM output frames
  to produce a fixed-size utterance representation. The attention weights are learned per
  modality task (emotion-relevant frames get higher weight).

### Computer Vision

- **Spatial handling:** DINOv2 processes individual frames; temporal coherence comes from
  the utterance-level pooling (not frame-level temporal model, keeping it lightweight).
  Facial crops are pre-processed with a 2D landmark detector for pose normalization.
- **Occlusion/pose robustness:** DINOv2 features are partially robust to occlusion (due to
  self-supervised training on ImageNet with natural occlusions). The quality estimator
  specifically detects face absence or extreme pose (profile, turned away).
- **Scale invariance:** Fixed 224×224 facial crops after landmark-based alignment. This is
  appropriate for clinical interviews where face distance from camera is roughly constant.
- **Frame sampling:** 30 uniformly sampled frames per utterance (≥2 fps for a 10s utterance).
  This balances temporal coverage with compute cost.

### Language Models

- **Architecture:** MentalBERT (BERT-base fine-tuned on mental health Reddit + clinical text).
  Chosen over DeBERTa-v3 for domain specialization; ClinicalBERT is too small (BioBERT base).
  DistilBERT variant is an option for edge deployment.
- **Short utterance handling:** Clinical utterances are often short ("yes", "no", "hmm").
  The text quality estimator predicts whether the utterance is informative enough; short
  utterances get gated down in CM-GEF.
- **Pooling:** [CLS] token representation, followed by a lightweight adaption layer.

### Multimodal Fusion (Central Design)

- **Missing modality handling:** Triple redundancy — (1) quality estimators detect and gate
  down unreliable modalities, (2) exchange attention provides cross-modal context for
  substitution, (3) reliability-weighted aggregation provides final-level suppression.
- **Asynchronous modalities:** Audio and video are naturally aligned (same time window);
  text corresponds to the transcribed utterance. Utterance-level alignment is sufficient for
  session-level depression prediction; frame-level alignment is not needed.
- **Modality dropout simulation during training:** We simulate 6 conditions — each modality
  independently dropped at 0/20/40/60%, plus 3 clinically-correlated patterns. This teaches
  the model to handle both random and systematic dropout.
- **Modality-specific encoders are frozen during stage 1** (projection MLPs + fusion only)
  and use LoRA in stage 2. This preserves pretrained knowledge while adapting to clinical
  distribution with minimal parameters.

### Time Series (Session-Level)

- **Temporal ordering:** Utterances are ordered by timestamp; position encoding preserves
  order. No future leakage since the session Transformer is bidirectional only in post-hoc
  analysis mode (streaming mode uses causal masking).
- **Session structure:** Clinical sessions have a well-defined structure (phases). The
  phase embedding + phase-level pooling is a form of **explicit temporal decomposition**,
  similar in spirit to time series decomposition (trend + seasonality) but for therapeutic
  stage.
- **Prediction head:** Direct multi-step (predict PHQ-8 from full session). This is
  appropriate for depression screening where the full session is available.

### Scientific ML (Clinical Constructs)

- **Clinical grounding:** The concept bottleneck maps to DSM-5 depression criteria (PHQ-8
  items). This is analogous to physics-informed ML, where the intermediate representation
  is constrained to clinically meaningful variables.
- **Interpretability vs. accuracy trade-off:** The concept bottleneck may reduce predictive
  accuracy compared to a fully black-box model. This is an acceptable trade-off for clinical
  deployment where trust and verifiability are paramount.
- **No equivariance constraints:** Depression severity is not equivariant under any natural
  transformation (unlike molecule rotation in 3D). No group-equivariance is needed.

---

## Step 8 — Implementation Risk Flags

### Risk 1: Quality Estimator Training Collapse
**What:** The quality estimator is pre-trained on synthetic corruptions, but during clinical
fine-tuning, the encoder features may shift, causing the quality scores to become uninformative
or collapse to a constant (always predicting "clean").
**Mitigation:**
- Run quality estimator in inference mode during stage 2 (no gradient through it)
- Or, if fine-tuning, add auxiliary corruption detection loss (keep a held-out corruption
  classification head) to prevent feature collapse
- Monitor quality score distribution on held-out validation set; alert if std. dev. < 0.05

### Risk 2: Phase Label Ambiguity
**What:** Therapeutic phase boundaries are subjective — two clinicians may segment the same
session differently. This label noise propagates through the phase embeddings.
**Mitigation:**
- Use soft phase labels (probability distribution over phases per utterance) instead of
  hard assignments — allow the model to learn phase transitions
- Ablate phase embeddings entirely and rely on learned position/timing alone; compare
  phase-aware vs. phase-agnostic version
- Evaluate inter-annotator agreement on a subset of sessions before committing to
  the phase-labeled dataset

### Risk 3: Per-Dimension Gate Optimization Instability
**What:** Per-dimension gates have d_model × n_modalities = 768 parameters per utterance,
which provides high capacity but may be prone to overfitting on small clinical datasets.
Gates may saturate at 0 or 1 (dead gate) and stop providing graded substitution.
**Mitigation:**
- Add gate entropy regularization (encourage gate values to be diverse, not all 0 or 1)
- Start with scalar modality gates and only enable per-dimension after convergence
- Use dropout on gate MLP activations

### Risk 4: Memory Bottleneck — Full Session Transformer
**What:** Processing 512 utterances with 8-head self-attention produces O(T²) = O(262k)
attention scores per layer. For 4 layers and batch size 16, this requires ~16M attention
scores per forward pass. On a single A100 (80GB), this may be acceptable, but edge
deployment or larger batch sizes may struggle.
**Mitigation:**
- Use linear attention (Performer, Linformer) for the session Transformer if memory is
  an issue
- Chunk long sessions into overlapping windows (e.g., 256 utterances with 50% overlap)
  and aggregate predictions
- Enable gradient checkpointing to trade compute for memory

### Risk 5: Concept Bottleneck Accuracy Ceiling
**What:** The 8-dimensional concept space may be insufficient to represent all clinical
information in the session features, creating an information bottleneck that limits
PHQ-8 prediction accuracy compared to an unconstrained model.
**Mitigation:**
- Compare concept bottleneck vs. direct prediction (no bottleneck) — if accuracy gap >
  5% MAE, increase concept dimension or add residual skip-connection bypassing the
  bottleneck
- Analyze which concept dimensions are most predictive; prune uninformative ones

---

## Step 9 — Suggested Ablations

Each ablation is a single `ModelConfig` field change that tests a specific hypothesis.
Ordered by "turn this off first if it doesn't work":

| # | Ablation | Config field | Baseline value | Ablated value | Hypothesis tested | Expected metric movement | Failure interpretation | Owning stage |
|---|---|---|---|---|---|---|---|---|
| 1 | **Drop CM-GEF → simple late fusion** | `n_exchange_layers` | 2 | 0 (use concat + MLP) | Gated exchange outperforms static late fusion under dropout | MAE ↑ by >2 points under 40% dropout | If late fusion matches or beats CM-GEF under dropout, the gating mechanism adds no value — revisit gate architecture | `ml-architect` |
| 2 | **Scalar gates → per-dim gates** | `exchange_hidden_dim` | 128 | 0 (use scalar gate: gate = broadcast(reliability)) | Per-dimension gates outperform scalar modality gates | MAE ↑ by ≤0.5 points; minimal effect suggests per-dimension is unnecessary complexity | Per-dimension gating is not justified — revert to scalar | `ml-architect` |
| 3 | **Drop quality estimators** | `n_audio_quality_classes` | 3 | 0 (skip quality; gate = sigmoid(self_attended)) | Supervised quality estimation improves over self-attended gating | MAE ↑ under noise/occlusion | Quality estimator pre-training not beneficial — use self-supervised gating only | `ml-architect` |
| 4 | **Drop phase embeddings** | `n_therapeutic_phases` | 4 | 1 (single phase; no phase embedding) | Therapeutic phase modeling improves sustained affect detection | Sustained affect F1 ↓ by ≥10% | Clinical phase priors do not help — revert to position-only temporal model | `ml-architect` |
| 5 | **Drop session Transformer → mean pool** | `n_session_layers` | 4 | 0 (mean pool utterance feats) | Session-level temporal modeling improves over utterance averaging | Session-level F1 (sustained affect) ↓ by ≥10% | Temporal structure not needed — revisit feature design | `ml-architect` |
| 6 | **LoRA vs. full fine-tune** | `lora_r` | 8 | 0 (= full fine-tune) | LoRA adaptation matches or exceeds full fine-tune on small clinical data | Both MAE same; LoRA trains faster; if LoRA MAE > full fine-tune MAE + 0.5 | LoRA insufficient for clinical domain shift — increase rank or use full fine-tune | `ml-research` |
| 7 | **Drop concept bottleneck** | `n_clinical_concepts` | 8 | 0 (direct PHQ-8 prediction) | Concept bottleneck maintains accuracy vs. direct prediction | MAE ↑ by ≤0.5 point (acceptable for interpretability) | If MAE ↑ > 1 point, bottleneck is too lossy — add residual or increase concept dim | `ml-architect` |
| 8 | **Drop modality dropout simulation** | `random_dropout_prob` | 0.15 | 0.0 | Dropout simulation during training improves test-time robustness | MAE ↑ on dropout conditions; minimal effect on clean data | Model already robust from architecture alone — relax training augmentation | `ml-coder` |
| 9 | **WavLM Base+ vs. Small** | `audio_encoder_name` | `wavlm-base-plus` | `wavlm-small` | Smaller encoder maintains target accuracy | MAE ↑ ≤ 0.5; latency ↓ 40% | If MAE ↑ > 1.0, Small is insufficient for clinical prosody | `ml-architect` |
| 10 | **Bidirectional vs. causal session** | `streaming_mode` | false | true | Bidirectional outperforms causal for offline analysis | Sustained affect F1 ↓; enables streaming use-case | Acceptable trade-off if streaming is a requirement; otherwise use bidirectional | `ml-architect` |

---

## Step 10 — Training Pipeline Detail

### Stage 1: Fusion Pre-training (Large-Scale In-the-Wild)

```
Data: CMU-MOSEI (23k YouTube videos, 1000 speakers, 6 emotions)
      + MELD (14k multi-party dialogues, 7 emotions)
      + IEMOCAP (10k utterances, 10 speakers, 4 emotion categories)
Total: ~47k multimodal utterances (or ~10k sessions for MELD)

Frozen modules: WavLM Base Plus, DINOv2 Small, MentalBERT
Trainable modules: Projection MLPs (3 × ~200k params)
                   Quality Estimators (3 × ~50k params)
                   CM-GEF (~2M params)
                   SLHT (optional, if conversation-level data available)
Total trainable: ~3M params

Loss:
  L_total = L_emotion_cls + λ_recon * L_dropout_reconstruction + λ_qual * L_quality_estimation

  L_emotion_cls: Cross-entropy on emotion class labels
  L_dropout_reconstruction: MSE between masked modality features and reconstructed features
  L_quality_estimation: Cross-entropy on synthetic corruption labels

Optimizer: AdamW, lr=1e-4, cosine schedule, 1000-step warmup
Duration: 50 epochs, ~4 hours on 4×A100
```

### Stage 2: Clinical Adaptation (Target Data)

```
Data: DAIC-WOZ (189 sessions, PHQ-8 labels)
      + E-DAIC (275 sessions, extended set)
      + AVEC 2013/2014 (depression datasets, optional)
Augmentation: Clinically-correlated dropout patterns (3 patterns, each with 20% probability)

Frozen modules: WavLM, DINOv2, MentalBERT (base weights)
LoRA modules: Query/Value/Key projections in all encoders (r=8, α=16)
Trainable modules: LoRA adapters (~5M params)
                   CM-GEF (~2M params)
                   SLHT (~3M params)
                   CCB (~0.5M params)
Total trainable: ~10.5M params

Loss:
  L_total = w_mae * L_mae(PHQ-8_pred, PHQ-8_true)
          + w_item * L_mse(item_pred, item_true) [if item labels available]
          + w_bce * L_bce(depression_pred, depression_true)
          + w_concept * L_concept_reg(concept_scores)  [encourage sparsity/diversity]

  Note: PHQ-8 item-level labels are not standard in DAIC-WOZ.
  If unavailable, the item prediction head is trained via:
    - Distillation: total score → items via a fixed allocation matrix
    - Self-supervised: reconstruction of utterance-level emotion labels → item-level projection

Optimizer: AdamW, lr=5e-5 (lower for LoRA), cosine schedule
          No weight decay on LoRA parameters (standard practice)
Duration: 30 epochs, ~2 hours on 1×A100
Batch size: 16
```

---

## Step 11 — Implementation Roadmap

### Phase 1: Data Pipeline (ml-coder)
- [ ] DAIC-WOZ/E-DAIC data loader with utterance segmentation
- [ ] Audio processing: 16kHz waveform → WavLM-compatible features
- [ ] Video processing: face detection → landmark alignment → DINOv2-compatible crops
- [ ] Text processing: tokenization → MentalBERT-compatible inputs
- [ ] Modality alignment: utterance-level (all three modalities share utterance boundaries)
- [ ] Synthetic corruption pipeline: audio noise injection, face occlusion, text truncation
- [ ] Phase label assignment (heuristic based on DAIC-WOZ interview protocol)

### Phase 2: Modular Implementation (ml-coder)
- [ ] Unimodal encoder wrappers (WavLM, DINOv2, MentalBERT) with LoRA support
- [ ] Projection MLPs + Quality Estimators
- [ ] CM-GEF: exchange attention + gated substitution + reliability aggregation
- [ ] SLHT: phase embeddings + session transformer + phase pooling
- [ ] CCB: concept predictor + item predictor + total score head
- [ ] Two-stage training script with checkpointing

### Phase 3: Baselines and Evaluation (ml-validator)
- [ ] Unimodal baselines (text-only, audio-only, video-only)
- [ ] Late fusion baselines
- [ ] Early fusion baselines
- [ ] MDD-MARF re-implementation (or code adaptation)
- [ ] Dropout stress test harness
- [ ] Fairness evaluation module

### Phase 4: Ablation Sweeps (ml-validator)
- [ ] Run all 10 ablations from §9
- [ ] Compute efficiency profile (latency, memory, FLOPs)
- [ ] Clinician interpretability study (Likert-scale)

---

## Output Checklist
- [x] Domain identified (Speech/Audio + CV + LM + Multimodal Fusion)
- [x] Upstream research lifecycle contract read and preserved
- [x] ModelConfig dataclass with all hyperparameters
- [x] Pseudocode for all four novel blocks (CM-GEF, SLHT, CCB, Quality Estimator)
- [x] ASCII architecture diagram
- [x] Inductive bias justification (one sentence per decision, with status labels)
- [x] Research-to-architecture traceability table (10 rows)
- [x] Claims labeled as `grounded`, `hypothesis`, or `TODO: unverified`
- [x] Domain-specific considerations addressed (all 4 domains + time series + scientific ML)
- [x] Implementation risk flags (5 risks with mitigations)
- [x] Baseline and evaluation requirements carried forward
- [x] Suggested ablations (10 ablations, each = single ModelConfig field change, tied to hypothesis)
- [x] Training pipeline detail (Stage 1 + Stage 2)
- [x] Implementation roadmap (4 phases)
