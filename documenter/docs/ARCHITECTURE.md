# Architecture

## 1. Motivation

Depression severity assessment in clinical settings relies on multimodal cues — speech prosody (tone, pitch, hesitations), facial expressions (action units, gaze), and semantic content — that unfold over 30–60 minute therapeutic interviews. Existing multimodal emotion recognition (MER) systems fall short on five fronts identified by the research synthesis (see [research/mer-mental-health-research.md](../../research/mer-mental-health-research.md)):

1. **Missing modality robustness is tested on lab data, not clinical.** Frameworks like AMTE (2025) and Mi-CGA (2025) evaluate gated exchange and graph attention on clean lab benchmarks (CMU-MOSI, IEMOCAP) where missing modalities are random. In clinical settings, dropout is systematically correlated with emotional state — patients turn away during distress (vision lost), cry (speech unintelligible), or fall silent (text absent).

2. **Session-level temporal structure is ignored.** DSIN (2025) and DSTC (2025) model short clips or windowed utterances. No existing architecture captures the therapeutic phase structure (rapport-building, exploration, intervention, closure) that shapes how depression markers manifest across a full session.

3. **Clinical labeled data is extremely scarce.** DAIC-WOZ has only ~189 sessions. Full fine-tuning of large multimodal models risks overfitting. Parameter-efficient adaptation (LoRA) is established in NLP and vision but unvalidated for clinical MER.

4. **No demographic fairness reporting exists.** Every surveyed 2025–2026 clinical MER paper omits disaggregated performance metrics, creating risk of systematic misdiagnosis for underrepresented groups.

5. **Interpretability is not clinically grounded.** ECMC (AAAI 2026) generates free-text emotion captions, but these are not mapped to DSM-5 criteria or actionable clinical constructs (PHQ-8 items).

**Hypothesis this architecture tests:** A lightweight cross-modal gated fusion architecture with session-level hierarchical temporal modeling and a clinical concept bottleneck can achieve clinically viable depression severity estimation (MAE < 3.0 on E-DAIC) while maintaining robustness to clinically-correlated modality dropout and demographic fairness.

## 2. At a glance

```
                            ┌─────────────────┐
                            │   PHQ-8 Total    │
                            │  (B,)            │
                            └────────┬────────┘
                                     ▲
                            ┌────────┴────────┐
                            │  PHQ-8 Items     │
                            │  (B, 8)          │
                            └────────┬────────┘
                                     ▲
                            ┌────────┴────────┐
                            │  CCB             │
                            │  8 concepts      │
                            │  (B, 8)          │
                            └────────┬────────┘
                                     ▲
                   ┌─────────────────┼─────────────────┐
                   │     Session Representation        │
                   │     (B, d_session_model)          │
                   └─────────────────┬─────────────────┘
                                     ▲
        ┌────────────────────────────┼────────────────────────────┐
        │                    SLHT                                 │
        │     ┌──────────────────────────────────┐               │
        │     │  Position Encoding + Phase Emb   │               │
        │     └────────────┬─────────────────────┘               │
        │     ┌────────────┴─────────────────────┐               │
        │     │  Bidirectional MHSA + FFN × 4    │               │
        │     └────────────┬─────────────────────┘               │
        │     ┌────────────┴─────────────────────┐               │
        │     │  Phase-Level Pooling (4 phases)   │               │
        │     └────────────┬─────────────────────┘               │
        └──────────────────┼────────────────────────────────────┘
                           ▲
     ┌─────────────────────┼─────────────────────────────────────┐
     │                    CM-GEF                                  │
     │  ┌──────────┐  ┌──────────┐  ┌──────────┐                │
     │  │Quality   │  │Quality   │  │Quality   │                │
     │  │Estimator │  │Estimator │  │Estimator │                │
     │  │(audio)   │  │(video)   │  │(text)    │                │
     │  └────┬─────┘  └────┬─────┘  └────┬─────┘                │
     │  ┌────┴────┐  ┌────┴────┐  ┌────┴────┐                   │
     │  │Proj MLP │  │Proj MLP │  │Proj MLP │                   │
     │  │d→256    │  │d→256    │  │d→256    │                   │
     │  └────┬────┘  └────┬────┘  └────┬────┘                   │
     │  ┌────┴────────────┴────────────┴────┐                   │
     │  │  Multi-Head Exchange Attention     │                   │
     │  └──────────────────┬────────────────┘                   │
     │  ┌──────────────────┴────────────────┐                   │
     │  │  Per-Dim Gated Substitution        │                   │
     │  └──────────────────┬────────────────┘                   │
     │  ┌──────────────────┴────────────────┐                   │
     │  │  Reliability-Weighted Aggregation  │                   │
     │  └──────────────────┬────────────────┘                   │
     └─────────────────────┼────────────────────────────────────┘
                           ▼
                   Utterance Feature (B, 256)

          ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
          │  WavLM Base+  │  │  DINOv2 Small │  │  MentalBERT   │
          │  (audio) d=768│  │  (video) d=384│  │  (text) d=768 │
          └───────┬───────┘  └───────┬───────┘  └───────┬───────┘
                  │                  │                  │
                  └──────────────────┼──────────────────┘
                                     │
                            Utterance (A+V+T aligned)
```

| Property | Value |
|---|---|
| Parameter count (default config, random init) | ~23M total (est.); ~3M trainable in Stage 1 |
| Time complexity (per utterance) | O(d_model^2 + n_heads * d_head) per exchange layer |
| Time complexity (per session) | O(T^2 * d_session) for SLHT (T = utterances, max 512) |
| Space complexity | O(B * T * d_session) for session attention |
| Hardware requirements | 1× A100 (80 GB) sufficient for training; edge deployment possible with streaming mode |

## 3. Core components

### 3.1 Modality Quality Estimator

#### Intuition

Each modality's reliability varies per utterance — the patient may turn away from the camera (video unreliable), speak through background noise (audio degraded), or produce short non-lexical utterances (text uninformative). Rather than learning static fusion weights, we predict per-utterance reliability from the encoder features using a lightweight MLP head pre-trained on synthetic corruptions. A modality with low reliability is gated down in the fusion step.

#### Equations

Let `f_m ∈ R^d_m` be the pooled encoder features for modality `m` (audio, video, or text). Quality estimator:

```
h = LayerNorm(f_m)
h = GELU(W₁ h + b₁)            // d_m → hidden_dim
s = W₂ h                        // hidden_dim → n_classes
p = softmax(s)                  // class probabilities
r = p₀                          // reliability = probability of "clean" class (index 0)
```

Where `n_classes = 3` for each modality: {clean, noisy, very_noisy} for audio, {visible, occluded, absent} for video, {informative, short, empty} for text.

#### Reference implementation walk-through

```python
# From encoders.py, ModalityQualityEstimator.forward()
def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    h = self.norm(features)        # (B, d_feat)  float32-safe layernorm
    h = self.net(h)                # (B, hidden_dim)  Linear + GELU + Dropout
    logits = self.classifier(h)    # (B, n_classes)
    probs = F.softmax(logits.float(), dim=-1)
    reliability = probs[:, 0]      # (B,)  class 0 = highest quality
    return reliability, logits
```

**Shape evolution:** `(B, d_feat) → (B, hidden_dim) → (B, n_classes) → (B,)`

### 3.2 Cross-Modal Gated Exchange Fusion (CM-GEF)

#### Intuition

CM-GEF is the central novelty. For each utterance, each modality attends to the other two via multi-head exchange attention, producing a cross-modal context vector. A per-dimension gate then interpolates between the modality's own features and the cross-modal context: when the modality is reliable, the gate stays high and the model trusts its own features; when unreliable, the gate drops and cross-modal context substitutes for the degraded signal. Finally, the three gated representations are aggregated via softmax weights proportional to each modality's estimated reliability.

Per-dimension gating (rather than a scalar modality-level gate) is intentional: audio pitch might be reliable while timbre is distorted, so the gate should suppress only the unreliable dimensions.

#### Equations

**Exchange attention** for modality `m` querying the other two modalities `{j, k}`:

```
Q_m = W_q f_m                    # query from current modality
K_{jk} = W_k [f_j; f_k]          # keys from other modalities
V_{jk} = W_v [f_j; f_k]          # values from other modalities

ctx_m = softmax(Q_m K_{jk}^T / √d_head) V_{jk}   # multi-head, (B, d_proj)
```

**Per-dimension gated substitution:**

```
gate_m = σ(W_g [f_m; ctx_m; r_m])     # (B, d_proj), per-feature-dimension gate
fused_m = gate_m ⊙ f_m + (1 - gate_m) ⊙ ctx_m
```

**Reliability-weighted aggregation:**

```
w = softmax([r_a, r_v, r_t])          # (B, 3)
fused = w_a · fused_a + w_v · fused_v + w_t · fused_t    # (B, d_proj)
```

#### Reference implementation walk-through

```python
# From fusion.py, CrossModalGatedExchangeFusion.forward()
# Stage 1: Exchange attention (audio → video+text)
ctx_audio, _ = self.exchange_attn_a(
    query=audio_feat.unsqueeze(1),                          # (B, 1, d)
    keys=[video_feat.unsqueeze(1), text_feat.unsqueeze(1)], # list of (B, 1, d)
    values=[video_feat.unsqueeze(1), text_feat.unsqueeze(1)],
)                                                            # (B, d)

# Stage 2: Per-dimension gated substitution
audio_fused, gate_a = self.gate_a(audio_feat, ctx_audio, r_audio)  # (B, d), (B, d)
video_fused, gate_v = self.gate_v(video_feat, ctx_video, r_video)  # (B, d), (B, d)
text_fused,  gate_t = self.gate_t(text_feat,  ctx_text,  r_text)   # (B, d), (B, d)

# Stage 3: Reliability-weighted aggregation
modality_weights = F.softmax(r_stack, dim=-1)                       # (B, 3)
fused = (modality_weights[:, 0:1] * audio_fused +                   # (B, d)
         modality_weights[:, 1:2] * video_fused +
         modality_weights[:, 2:3] * text_fused)

# Stage 4: Optional stacked exchange layers (residual + FFN)
for _ in range(1, config.n_exchange_layers):
    residual = fused
    fused = self.post_norm(fused)
    fused = fused + self.post_ffn(fused)
    if config.exchange_residual:
        fused = residual + fused
```

**Shape evolution:** `(B, d_proj)` per modality → exchange attention → `(B, d_proj)` context per modality → gated substitution → `(B, d_proj)` per modality → aggregation → `(B, d_proj)` fused utterance representation.

### 3.3 Session-Level Hierarchical Transformer (SLHT)

#### Intuition

Clinical interviews have a well-defined structure: rapport-building (first ~5 min), exploration (open-ended probes), intervention (targeted clinical questions), and closure (wrap-up). Depression markers manifest differently across phases — flat affect during rapport may indicate sustained depression, while emotional reactivity during intervention probes is a positive prognostic sign. SLHT captures this by adding learned phase embeddings to each utterance, processing the full utterance sequence with a bidirectional Transformer, then pooling utterances within each phase via learned cross-attention to produce per-phase representations. A learned importance weight over phases produces the final session representation.

#### Equations

**Phase embedding + position encoding:**

```
x_t = W_proj [utt_t; emb(phase_t)] + pos_enc(t)    # (B, T, d_session)
```

**Bidirectional self-attention over utterances (4 layers):**

```
h^l = MSA(LayerNorm(h^{l-1})) + h^{l-1}
h^{l} = FFN(LayerNorm(h^l)) + h^l                    # (B, T, d_session)
```

**Phase-level pooling** via cross-attention for each phase `p`:

```
μ_p = masked_mean(h, phase_mask_p)                   # (B, d_session)
ctx_p = cross_attn(query=μ_p, keys=h, values=h)      # (B, d_session)
ψ_p = μ_p + ctx_p                                     # residual blend
```

**Session representation:**

```
α_p = softmax(W_imp ψ_p)                              # (B, n_phases)
session = Σ_p α_p · ψ_p + global_mean(h)              # (B, d_session)
```

#### Reference implementation walk-through

```python
# From session.py, SessionLevelTransformer.forward()
# 1. Phase embedding
phase_emb = self.phase_embed(phase_labels)         # (B, T, phase_embed_dim)

# 2. Position encoding (learned)
pos_enc = self.pos_embed(positions)                 # (B, T, d_model)

# 3. Project + combine
x = torch.cat([utterance_features, phase_emb], dim=-1)  # (B, T, d_in + phase_dim)
x = self.input_proj(x) + pos_enc                    # (B, T, d_model)

# 4. Bidirectional transformer blocks
for block in self.blocks:
    x = block(x, attn_mask=attn_mask)               # (B, T, d_model)

# 5. Phase-level pooling
phase_repr = self.phase_pooling(x, phase_labels)    # (B, n_phases, d_model)

# 6. Phase-weighted session aggregation
phase_importance = self.phase_importance(phase_repr)  # (B, n_phases, 1)
alpha = F.softmax(phase_importance, dim=1)
session_feat = (alpha * phase_repr).sum(dim=1)      # (B, d_model)
session_feat = session_feat + x.mean(dim=1)         # residual global context
```

**Shape evolution:** `(B, T, d_proj)` → `(B, T, d_proj + phase_dim)` → `(B, T, d_session)` → `(B, T, d_session)` (4 layers) → `(B, n_phases, d_session)` → `(B, d_session)`.

### 3.4 Clinical Concept Bottleneck (CCB)

#### Intuition

Rather than predicting PHQ-8 scores directly from session features (a black-box mapping), CCB first predicts 8 interpretable concept scores corresponding to DSM-5 depression criteria (anhedonia, depressed mood, sleep disturbance, fatigue, appetite change, guilt/worthlessness, concentration problems, psychomotor change). PHQ-8 item scores are then computed from these concepts via a diagonal-dominant weight matrix (each concept primarily maps to its corresponding item). The total score is the sum of items, and binary depression classification uses a sigmoid over concept scores.

This makes the model's reasoning transparent: clinicians can inspect which concepts were activated and verify that the concept→item mapping is clinically sensible.

#### Equations

**Concept prediction:**

```
c = σ(W_c2 GELU(W_c1 LayerNorm(session)))    # (B, n_concepts) ∈ [0, 1]
```

**Item prediction (blend of per-item heads + matrix):**

```
i = 0.5 · head_item(c) + 0.5 · W_item c      # (B, n_items)
phq8_items = 3 · σ(i)                          # (B, n_items) ∈ [0, 3]
```

**Total score and classification:**

```
phq8_total = Σ_i phq8_items_i                  # (B,) ∈ [0, 24]
d = σ(w_d^T c)                                  # (B,) depression probability
```

**Diagonal-dominant initialization** (from `_init_item_weights`):

```python
W = self.concept_to_item.weight       # (n_items, n_concepts)
nn.init.normal_(W, mean=0.0, std=0.01)
W[diag_idx, diag_idx] = 1.0           # diagonal set to 1.0
```

## 4. Tensor shape evolution

Default config: `proj_dim=256`, `d_session_model=256`, `B=4`, `T=32`, `n_phases=4`, `n_concepts=8`, `n_items=8`.

| Stage | Shape | Notes |
|---|---|---|
| Audio waveform | `(B*T, 16000)` | 1 sec @ 16kHz per utterance |
| Video frames | `(B*T, 30, 3, 224, 224)` | 30 frames/utterance, RGB |
| Text token IDs | `(B*T, 128)` | MentalBERT tokenized |
| WavLM output | `(B*T, 768)` | Attention-pooled; float32/bf16 |
| DINOv2 output | `(B*T, 384)` | [CLS] token + sequence pooling |
| MentalBERT output | `(B*T, 768)` | CLS + attention-pooled sequence |
| Quality scores | `(B*T,)` each ×3 | Scalar ∈ [0,1] per modality |
| Projected features | `(B*T, 256)` each ×3 | MLP: d_feat → proj_dim |
| Exchange context | `(B*T, 256)` each ×3 | Cross-modal attention |
| Gated fused | `(B*T, 256)` each ×3 | gate·self + (1-gate)·context |
| Aggregated utterance | `(B*T, 256)` | Reliability-weighted sum |
| Reshaped session | `(B, 32, 256)` | T utterances per session |
| Phase embedding | `(B, 32, 64)` | Learned embedding × 4 phases |
| Session transformer | `(B, 32, 256)` | 4 layers bidirectional MHSA |
| Phase representations | `(B, 4, 256)` | Cross-attended phase pooling |
| Session representation | `(B, 256)` | Phase-weighted + global context |
| Concept scores | `(B, 8)` | Sigmoid, ∈ [0, 1] |
| PHQ-8 items | `(B, 8)` | Scaled sigmoid, ∈ [0, 3] |
| PHQ-8 total | `(B,)` | Sum of items, ∈ [0, 24] |
| Depression prob | `(B,)` | Sigmoid classifier on concepts |

## 5. Design decisions

| Decision | Alternative considered | Why we chose this | Trade-off accepted |
|---|---|---|---|
| Per-dimension gating | Scalar modality-level gates | Partial feature subspaces may be reliable when others aren't (e.g., pitch reliable but timbre distorted) | Higher capacity → potential overfitting on small clinical data; requires entropy regularization |
| Multi-head exchange attention | Concat + linear fusion | Heads can specialize on different cross-modal patterns (audio→lip movements, audio→sentiment) | Linear in n_heads; standard cross-attention cost |
| Bidirectional session Transformer | Causal/autoregressive | Full session is available for post-hoc analysis; early cues benefit from later context | Not suitable for streaming without masking; O(T²) memory |
| Therapeutic phase embeddings | Timestamp-only positional encoding | Depression markers manifest differently across therapeutic phases; phase is a known clinical prior | Requires phase labels (manual or heuristic); inter-annotator agreement is a risk |
| Clinical concept bottleneck | Direct black-box regression | Clinicians need to verify model reasoning against clinical constructs | May reduce accuracy vs. unconstrained model; 8-dimensional concept space is a potential information bottleneck |
| Two-stage training (pretrain → LoRA) | Full fine-tune from scratch | Clinical labeled data is extremely scarce (~189 sessions); LoRA preserves pretrained knowledge | Additional engineering complexity; LoRA effectiveness for multimodal clinical domain shift is unverified |
| Reliability-weighted aggregation (3rd line of defense) | Gate-only fusion | Triple redundancy: quality estimator → gated substitution → reliability weight | Marginal compute cost for softmax; ablation should verify if both gate and weight are needed |
| WavLM Base+ over HuBERT | HuBERT, Whisper encoder | WavLM's denoising and speaker-aware pretraining provides better emotion-relevant representations per 2024–2025 paralinguistic benchmarks | Slightly larger than wavlm-small; ablation #9 tests trade-off |
| DINOv2 Small over ViT/ResNet | ViT-B, ResNet-50, VGG-Face | DINOv2 features show strong zero-shot facial affect transfer; Self-supervised pretraining handles occlusion better | DINOv2's frame-level processing misses temporal dynamics (trade-off for lightweight design) |
| MentalBERT over DeBERTa/ClinicalBERT | DeBERTa-v3, ClinicalBERT, BioBERT | Domain-adapted for mental health text (Reddit + clinical notes); Balances domain specificity with model capacity | ClinicalBERT is too small; DeBERTa lacks mental health domain adaptation |

## 6. Domain-specific considerations

### Speech/Audio

- **Input representation:** Raw waveform → WavLM Base Plus (self-supervised speech model with denoising and speaker-aware pretraining). Operates at 16 kHz with max 10-second utterances.
- **Causality:** Offline mode (full utterance available) is the primary target. Streaming mode is configured via `streaming_mode` flag with a lightweight projection bypassing the session transformer.
- **Utterance pooling:** Attention-weighted mean pooling over WavLM output frames. The quality estimator detects acoustic degradation (noise, codec compression, silence).

### Computer Vision

- **Spatial handling:** DINOv2 processes 30 uniformly sampled frames per utterance independently. Temporal coherence is captured by utterance-level attention pooling (not a frame-level temporal model, keeping it lightweight).
- **Occlusion/pose robustness:** DINOv2 features are partially robust due to self-supervised training. The quality estimator specifically detects face absence or extreme pose. Facial crops require landmark-based pre-processing.
- **Frame sampling:** 224×224 crops; 30 frames over a 10s utterance (≥2 fps) balances coverage with compute.

### Language Models

- **Architecture:** MentalBERT (BERT-base fine-tuned on mental health Reddit + clinical text). Uses [CLS] token + attention-weighted sequence pooling.
- **Short utterance handling:** The quality estimator predicts whether the utterance is informative; uninformative utterances (e.g., "yes", "no", "hmm") are gated down in CM-GEF.

### Multimodal Fusion (Central Design)

- **Missing modality handling:** Triple redundancy — (1) quality estimators detect unreliability, (2) gated substitution provides cross-modal context, (3) reliability-weighted aggregation provides final suppression.
- **Asynchronous modalities:** Audio and video are naturally aligned (same time window); text corresponds to the transcribed utterance. Utterance-level alignment is sufficient for session-level depression prediction.
- **Dropout simulation during training:** 6 conditions (each modality independently dropped at 0/20/40/60%) + 3 clinically-correlated patterns (vision-drop-during-distress, audio-drop-in-noisy, text-drop-in-nonverbal).

### Time Series (Session-Level)

- **Temporal ordering:** Utterances ordered by timestamp; position encoding preserves order. No future leakage in post-hoc analysis (bidirectional). Streaming mode uses causal masking.
- **Session structure:** Clinical sessions have 4 phases (rapport → exploration → intervention → closure). Phase embedding + phase pooling is an explicit temporal decomposition, analogous to trend + seasonality decomposition in time series.
- **Prediction head:** Direct multi-step — predicts PHQ-8 from the full session (appropriate for depression screening where the full session is available).

### Scientific ML (Clinical Constructs)

- **Clinical grounding:** The concept bottleneck maps to DSM-5 depression criteria (PHQ-8 items). Intermediate representations are constrained to clinically meaningful variables, analogous to physics-informed ML.
- **Interpretability vs. accuracy trade-off:** Acceptable for clinical deployment where trust and verifiability are paramount.
- **No equivariance constraints:** Depression severity is not equivariant under any natural transformation.

## 7. Known limitations

- **No trained model exists.** All current metrics are from randomly initialized models on synthetic data. Quantitative targets (MAE < 3.0 on E-DAIC) are research hypotheses, not demonstrated results.
- **No real clinical data integration.** DAIC-WOZ/E-DAIC data loaders are not implemented.
- **MDD-MARF baseline not implemented.** The current SOTA on DAIC-WOZ (MAE 3.13, RMSE 3.59) cannot be compared against.
- **Early fusion baseline not implemented.** Concatenation+MLP baseline is missing from the benchmark suite.
- **Clinician interpretability study deferred.** Verifying concept→PHQ-8 mapping requires ≥3 clinicians for a Likert-scale study.
- **Fairness evaluation blocked.** Requires demographic labels; DAIC-WOZ/E-DAIC may lack sufficient diversity for powered subgroup analysis.
- **Phase labeling strategy unvalidated.** Heuristic phase assignment (time windows) may have low inter-annotator agreement with clinical experts.
- **Clinically-correlated dropout simulation is heuristic.** The 3 dropout patterns are constructed, not validated against real clinical dropout distributions.
- **No formal theoretical bounds.** Generalization, approximation, and sample complexity bounds are not established.
- **Two-stage training script not implemented.** Stage configuration methods exist in `model.py` but no end-to-end `train.py` script.
