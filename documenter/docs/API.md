# API Reference

## config.py

### `class ModelConfig`
Comprehensive dataclass for all hyperparameters in the MER Mental Health architecture. Every architectural variant, training configuration, and evaluation setting is controlled through this single config object.

**Fields:**

| Field | Type | Default | Rationale |
|---|---|---|---|
| **Task specification** | | | |
| `n_phq8_items` | int | 8 | PHQ-8 has 8 items |
| `phq8_threshold` | int | 10 | PHQ-8 >= 10 indicates depression |
| `n_clinical_concepts` | int | 8 | One concept per PHQ-8 item |
| `task_type` | str | "regression" | "regression" \| "classification" \| "both" |
| **Audio encoder (WavLM)** | | | |
| `audio_encoder_name` | str | "wavlm-base-plus" | "wavlm-base-plus" \| "wavlm-small" |
| `audio_feat_dim` | int | 768 | WavLM Base+ output dimension |
| `audio_sample_rate` | int | 16000 | Standard for speech |
| `audio_max_seconds` | float | 10.0 | Max utterance length |
| `audio_freeze_encoder` | bool | True | Frozen in stage 1 |
| **Video encoder (DINOv2)** | | | |
| `video_encoder_name` | str | "dinov2-small" | "dinov2-small" \| "dinov2-base" |
| `video_feat_dim` | int | 384 | DINOv2 Small output dimension |
| `video_n_frames` | int | 30 | Frames per utterance |
| `video_frame_size` | int | 224 | DINOv2 input resolution |
| `video_freeze_encoder` | bool | True | Frozen in stage 1 |
| **Text encoder (MentalBERT)** | | | |
| `text_encoder_name` | str | "mentalbert" | "mentalbert" \| "deberta-v3-base" |
| `text_feat_dim` | int | 768 | BERT-base output dimension |
| `text_max_length` | int | 128 | Max tokenized utterance length |
| `text_freeze_encoder` | bool | True | Frozen in stage 1 |
| **Common projection** | | | |
| `proj_dim` | int | 256 | Common latent dimension |
| `proj_n_layers` | int | 2 | MLP depth for projection |
| `proj_dropout` | float | 0.1 | — |
| `proj_activation` | str | "gelu" | Activation for projection MLP |
| **CM-GEF (fusion)** | | | |
| `n_exchange_heads` | int | 4 | Multi-head exchange attention |
| `exchange_hidden_dim` | int | 128 | Gate MLP hidden size |
| `exchange_dropout` | float | 0.1 | — |
| `n_exchange_layers` | int | 2 | Stacked exchange blocks |
| `exchange_residual` | bool | True | Residual in stacked layers |
| **Quality estimators** | | | |
| `n_audio_quality_classes` | int | 3 | 0=clean, 1=noisy, 2=very_noisy |
| `quality_estimator_hidden` | int | 64 | MLP hidden dimension |
| `quality_estimator_dropout` | float | 0.1 | — |
| **SLHT (session)** | | | |
| `n_therapeutic_phases` | int | 4 | rapport, exploration, intervention, closure |
| `phase_embed_dim` | int | 64 | Phase embedding dimension |
| `d_session_model` | int | 256 | Session transformer hidden dim |
| `n_session_layers` | int | 4 | Transformer depth |
| `n_session_heads` | int | 8 | Multi-head attention heads |
| `session_d_model_ff` | int | 1024 | FFN expansion |
| `session_dropout` | float | 0.1 | — |
| `max_utterances` | int | 512 | Max utterances per session |
| `session_pos_encoding` | str | "learned" | "learned" \| "sinusoidal" |
| **CCB (concept bottleneck)** | | | |
| `concept_names` | Tuple[str,...] | (8 PHQ-8 names) | Clinical construct names |
| `concept_hidden_dim` | int | 128 | Concept predictor hidden dim |
| `concept_dropout` | float | 0.15 | — |
| **Prediction heads** | | | |
| `regression_head` | bool | True | Predict PHQ-8 total |
| `classification_head` | bool | True | Binary depression |
| `item_prediction_head` | bool | True | Per-item scores |
| **LoRA** | | | |
| `lora_r` | int | 8 | LoRA rank |
| `lora_alpha` | int | 16 | LoRA scaling |
| `lora_dropout` | float | 0.05 | — |
| `lora_target_modules` | Tuple[str,...] | Q,V,K,O,I | Target linear layers |
| **Training** | | | |
| `stage_1_lr` | float | 1e-4 | — |
| `stage_1_epochs` | int | 50 | — |
| `stage_2_lr` | float | 5e-5 | — |
| `stage_2_epochs` | int | 30 | — |
| `warmup_steps` | int | 1000 | Linear warmup |
| `weight_decay` | float | 0.01 | Excluded from bias/norm/LoRA |
| **Loss weights** | | | |
| `loss_mae_weight` | float | 1.0 | PHQ-8 total MAE |
| `loss_bce_weight` | float | 0.5 | Binary depression BCE |
| `loss_item_weight` | float | 0.3 | Item MSE |
| `loss_concept_supervision_weight` | float | 0.2 | Concept BCE (optional) |
| `loss_dropout_reconstruction_weight` | float | 0.1 | Auxiliary recon loss |
| **Dropout simulation** | | | |
| `random_dropout_prob` | float | 0.15 | Per-modality random |
| `clinical_correlated_dropout_prob` | float | 0.1 | Clinical patterns |
| **Inference** | | | |
| `streaming_mode` | bool | False | Per-utterance no session ctx |
| `dtype` | str | "bfloat16" | — |
| `seed` | int | 42 | Random seed |
| `pretrained` | bool | False | Load HF weights (test mode = False) |

---

## model.py

### `class MEROutput`
Structured output dataclass for model predictions.

**Fields:**
- `phq8_total: Tensor` — `(B,)` predicted PHQ-8 total score [0, 24]
- `phq8_items: Tensor` — `(B, 8)` per-item scores [0, 3]
- `concept_scores: Tensor` — `(B, n_concepts)` concept activations [0, 1]
- `depression_prob: Tensor` — `(B,)` probability PHQ-8 >= 10
- `gate_weights: Optional[Tensor]` — `(B, 3)` mean gate per modality
- `modality_weights: Optional[Tensor]` — `(B, 3)` reliability weights
- `utterance_feats: Optional[Tensor]` — `(B, T, proj_dim)` per-utterance features
- `session_feat: Optional[Tensor]` — `(B, d_session_model)`
- `phase_repr: Optional[Tensor]` — `(B, n_phases, d_session_model)`
- `reliability_scores: Optional[Dict[str, Tensor]]` — per-modality reliability dict
- `loss: Optional[Tensor]` — scalar loss

**Methods:**
- `detach()` — Returns a new `MEROutput` with all tensors detached from the graph.

### `class MERMentalHealthModel(nn.Module)`
Full multimodal emotion recognition model for mental health diagnostics.

**Constructor:** `MERMentalHealthModel(config: ModelConfig)`

**Methods:**

#### `forward(waveforms, frames, input_ids, ...) -> MEROutput`
Full forward pass: encode utterances, fuse modalities, model session, predict PHQ-8 through concept bottleneck.

**Args:**

| Arg | Shape | Description |
|---|---|---|
| `waveforms` | `(B, T, T_samples)` | Raw audio per utterance |
| `frames` | `(B, T, n_frames, C, H, W)` | Video frames per utterance |
| `input_ids` | `(B, T, T_tokens)` | Text token IDs per utterance |
| `phase_labels` | `(B, T)` or None | Therapeutic phase index (default: all phase 0) |
| `timestamps` | `(B, T)` or None | Seconds from session start |
| `utterance_mask` | `(B, T)` bool or None | True = valid utterance |
| `audio_mask` | `(B, T)` bool or None | True = audio present |
| `video_mask` | `(B, T)` bool or None | True = video present |
| `text_mask` | `(B, T)` bool or None | True = text present |
| `phq8_labels` | `(B,)` or None | Ground truth for loss |
| `concept_labels` | `(B, n_concepts)` or None | Ground truth concept scores |
| `return_all` | bool | Return all intermediate representations |

**Shape invariants:**
- B ≥ 1; T ≤ config.max_utterances
- dtype in {float32, bfloat16}; float16 not recommended
- If `streaming_mode=True`, session transformer is bypassed

#### `forward_utterance(waveform, frames, input_ids, ...) -> Tensor`
Single utterance encoding without session context. Returns `(B, proj_dim)` fused features.

#### `forward_streaming(fused_feat) -> MEROutput`
Streaming inference from a single utterance's fused features. Uses a lightweight projection instead of the full session transformer.

#### `encode_utterance(waveform, frames, input_ids, ...) -> Tuple[Tensor, Dict, Dict]`
Encode a single utterance through all three unimodal encoders, quality estimators, projections, and cross-modal fusion.

**Returns:**
- `fused_feat: (B, proj_dim)` — fused utterance representation
- `reliability: dict` — `{"audio": (B,), "video": (B,), "text": (B,), ...}`
- `gate_info: dict` — `{"gate_weights": (B, 3), "modality_weights": (B, 3)}`

#### `set_trainable_params(stage: int = 1)`
Configure which parameters are trainable:
- `stage=0`: All frozen (inference mode)
- `stage=1`: Encoders frozen; projection MLPs, quality estimators, fusion, SLHT, CCB trainable
- `stage=2`: Encoders frozen (LoRA adapters trainable); fusion, SLHT, CCB, quality estimators, projections trainable

#### `apply_lora(config: Optional[ModelConfig] = None)`
Apply LoRA adapters to target modules (wraps Linear layers with LoRALinear). Idempotent.

#### `freeze_encoders()` / `unfreeze_encoders()`
Freeze/unfreeze all three unimodal encoder backbones.

#### `compute_loss(phq8_total, phq8_items, depression_prob, concept_scores, phq8_labels, concept_labels) -> Tensor`
Multi-task loss: MAE + BCE for binary depression + MSE for items + BCE for concepts (optional).

### `create_optimizer(model, config, stage) -> AdamW`
Create AdamW optimizer with separate param groups (weight decay for most params, no decay for bias/LoRA/norm).

### `create_scheduler(optimizer, config, steps_per_epoch, stage) -> LambdaLR`
Cosine learning rate scheduler with linear warmup.

### `modality_dropout_augmentation(audio_mask, video_mask, text_mask, ...) -> Tuple[Tensor, Tensor, Tensor]`
Apply random and clinically-correlated modality dropout augmentation during training.

---

## encoders.py

### `class BaseEncoder(ABC, nn.Module)`
Abstract base for all unimodal encoders. Produces fixed-size utterance-level feature vectors.

**Methods:**
- `forward(*args, **kwargs) -> Tensor` — `(B, feature_dim)`
- `freeze()` / `unfreeze()` — Toggle requires_grad

### `class WavLMEncoder(BaseEncoder)`
Audio encoder using WavLM Base Plus. Raw waveform `(B, T_samples)` → `(B, audio_feat_dim=768)`.

**Constructor:** `WavLMEncoder(config: ModelConfig)`

**Forward:** `forward(waveform: (B, T_samples), mask: optional (B, T_frames) bool) -> (B, audio_feat_dim)`

### `class DINOv2Encoder(BaseEncoder)`
Video encoder using DINOv2 Small. Frames `(B, n_frames, C, H, W)` → `(B, video_feat_dim=384)`.

**Constructor:** `DINOv2Encoder(config: ModelConfig)`

**Forward:** `forward(frames: (B, n_frames, C, H, W), mask: optional (B, n_frames) bool) -> (B, video_feat_dim)`

### `class MentalBERTEncoder(BaseEncoder)`
Text encoder using MentalBERT. Token IDs `(B, T_tokens)` → `(B, text_feat_dim=768)`.

**Constructor:** `MentalBERTEncoder(config: ModelConfig)`

**Forward:** `forward(input_ids: (B, T_tokens), attention_mask: optional (B, T_tokens)) -> (B, text_feat_dim)`

### `class ModalityQualityEstimator(nn.Module)`
Predict per-utterance modality reliability from encoder features.

**Constructor:** `ModalityQualityEstimator(d_feat: int, n_classes: int, hidden_dim: int=64, dropout: float=0.1)`

**Methods:**
- `forward(features: (B, d_feat)) -> Tuple[(B,), (B, n_classes)]` — reliability score and class logits
- `estimate(features: (B, d_feat)) -> (B,)` — convenience: returns reliability only (no grad)

---

## fusion.py

### `class BaseFusionOperator(ABC, nn.Module)`
Abstract base for the fusion operator.

**Methods:**
- `forward(audio_feat, video_feat, text_feat)` — each `(B, proj_dim)` — returns `(fused, gates, weights)`

### `class GatedSubstitution(nn.Module)`
Per-feature-dimension gated interpolation between self and cross-modal context.

**Constructor:** `GatedSubstitution(d_model: int, hidden_dim: int)`

**Forward:** `forward(self_feat: (B, d), ctx_feat: (B, d), reliability: (B,)) -> Tuple[(B, d), (B, d)]`
- Output: `(fused, gate)` where `fused = gate * self + (1-gate) * ctx`
- Gate values in [0, 1] per feature dimension

### `class CrossModalGatedExchangeFusion(BaseFusionOperator)`
Full CM-GEF block: exchange attention + gated substitution + reliability-weighted aggregation + optional stacked layers.

**Constructor:** `CrossModalGatedExchangeFusion(config: ModelConfig)`

**Forward:** `forward(audio_feat: (B, d), video_feat: (B, d), text_feat: (B, d), r_audio/r_video/r_text: (B,), audio/video/text_mask: (B,) bool) -> Tuple[(B, d), (B, 3), (B, 3)]`

**Methods:**
- `forward_with_checkpoint(...)` — gradient checkpointing wrapper

---

## session.py

### `class BaseSessionOperator(ABC, nn.Module)`
Abstract base for session-level temporal modeling.

**Methods:**
- `forward(utterance_features, phase_labels, timestamps, utterance_mask)` — returns `(session_feat, phase_repr)`

### `class PhaseEmbedding(nn.Module)`
Learned embedding for therapeutic phases.

**Constructor:** `PhaseEmbedding(n_phases: int, embed_dim: int)`

**Forward:** `forward(phase_labels: (B, T)) -> (B, T, embed_dim)`

### `class SessionTransformerBlock(nn.Module)`
Single transformer block: MHA → residual → LayerNorm → FFN → residual → LayerNorm.

**Constructor:** `SessionTransformerBlock(d_model, n_heads, d_ff, dropout)`

### `class PhaseLevelPooling(nn.Module)`
Aggregate utterances within each therapeutic phase via masked mean + cross-attention refinement.

**Constructor:** `PhaseLevelPooling(d_model, n_phases, dropout)`

**Forward:** `forward(x: (B, T, d_model), phase_labels: (B, T), utterance_mask: (B, T) bool) -> (B, n_phases, d_model)`

### `class SessionLevelTransformer(BaseSessionOperator)`
Full SLHT: phase embedding + position encoding → projection → bidirectional transformer blocks → phase pooling → phase-weighted session aggregation.

**Constructor:** `SessionLevelTransformer(config: ModelConfig)`

**Forward:** `forward(utterance_features: (B, T, proj_dim), phase_labels: (B, T), timestamps: (B, T), utterance_mask: (B, T) bool, use_checkpoint: bool) -> Tuple[(B, d_session_model), (B, n_phases, d_session_model)]`

**Methods:**
- `forward_with_checkpoint(...)` — gradient checkpointing wrapper

---

## concept.py

### `class BaseConceptBottleneck(ABC, nn.Module)`
Abstract base for the clinical concept bottleneck.

### `class ClinicalConceptBottleneck(BaseConceptBottleneck)`
Maps session features through 8 clinical concepts → PHQ-8 items → total score → binary depression.

**Constructor:** `ClinicalConceptBottleneck(config: ModelConfig)`

**Forward:** `forward(session_feat: (B, d_session_model), concept_labels: optional (B, n_concepts)) -> Tuple[(B,), (B, 8), (B, n_concepts), (B,)]`

**Returns:**
- `phq8_total: (B,)` — PHQ-8 total score [0, 24]
- `phq8_items: (B, 8)` — per-item severity [0, 3]
- `concept_scores: (B, n_concepts)` — concept activations [0, 1]
- `depression_prob: (B,)` — depression probability

**Methods:**
- `get_concept_importance(session_feat: (B, d_session)) -> (B, n_concepts)` — concept scores for interpretability
- `get_concept_to_item_weights() -> (n_items, n_concepts)` — weight matrix

---

## layers.py

### `class LayerNorm(nn.Module)`
LayerNorm that always runs in float32 for numerical stability (avoids bf16 precision issues).

### `class FeedForward(nn.Module)`
Position-wise FFN: Linear → activation → Dropout → Linear.

### `class ProjectionMLP(nn.Module)`
Shallow MLP for modality projection: d_in → d_out with configurable depth, dropout, and activation.

### `class MultiHeadSelfAttention(nn.Module)`
Standard multi-head scaled dot-product self-attention. Q/K/V projections, attention masking, optional weight return.

### `class CrossModalAttention(nn.Module)`
Multi-head cross-modal attention where a single query attends to multiple key/value sources (used in CM-GEF exchange attention).

### `class SinusoidalPositionEncoding(nn.Module)`
Fixed sinusoidal position encoding (used as alternative to learned in SLHT).

### `class AttentionPooling(nn.Module)`
Attention-weighted mean pooling over sequence dimension (used in encoders for utterance-level aggregation).

### `class LoRALinear(nn.Module)`
Linear layer with Low-Rank Adaptation. During training: `y = Wx + (x @ A @ B) * (α/r)`. Supports weight merging for inference.

**Methods:**
- `merge_weights()` — Fold LoRA into base weights for zero-overhead inference
- `unmerge_weights()` — Reverse merge for fine-tuning

### `apply_lora_to_linear(module, config, prefix, ...)`
Recursively replace Linear layers matching `config.lora_target_modules` with `LoRALinear`.

### `count_params(model)`
Print total and trainable parameter counts.
