# Training & Reproduction

> **Note:** An end-to-end training script (`train.py`) has not yet been implemented. The two-stage training pipeline is configured via `ModelConfig` fields and stage management methods in `model.py` (`set_trainable_params`, `apply_lora`, `create_optimizer`, `create_scheduler`). This document specifies the intended recipe based on the architecture design. All numerical targets are `TODO: unverified`.

## Environment

- Python: 3.10+
- PyTorch: 2.1+ (tested with 2.5)
- CUDA: 12.1+ (tested on NVIDIA A100 80 GB)
- Other: transformers (HuggingFace — for WavLM, DINOv2, MentalBERT)

```bash
# Environment setup
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers

# Verify
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

> Dependencies note: A `requirements.txt` has not yet been generated. The current codebase uses `transformers` for pretrained model loading; for smoke testing, dummy models are used (set `config.pretrained=False`).

## Default hyperparameters

All hyperparameters are defined in `ModelConfig` (see [config.py](../../coder/config.py) and [docs/API.md](API.md#configpy)). Key fields and their rationales:

| Field | Default | Rationale |
|---|---|---|
| `proj_dim` | 256 | Common latent dimension balancing expressiveness vs. parameter count |
| `d_session_model` | 256 | Session transformer hidden dimension; matches proj_dim for residual flow |
| `n_exchange_layers` | 2 | Stacked exchange blocks for iterative cross-modal refinement |
| `n_session_layers` | 4 | Depth for bidirectional session encoding |
| `n_exchange_heads` | 4 | Multi-head cross-modal attention heads (d_head = 64) |
| `n_session_heads` | 8 | Multi-head self-attention heads for session transformer |
| `phase_embed_dim` | 64 | Dimension of learned therapeutic phase embeddings |
| `concept_hidden_dim` | 128 | Hidden dimension for concept predictor MLP |
| `concept_dropout` | 0.15 | Dropout for concept bottleneck (higher than fusion due to small concept space) |
| `lora_r` | 8 | LoRA rank; standard rank for parameter-efficient fine-tuning |
| `lora_alpha` | 16 | LoRA scaling factor (alpha/r = 2) |
| `random_dropout_prob` | 0.15 | Per-modality random dropout during training |
| `clinical_correlated_dropout_prob` | 0.1 | Clinically-correlated dropout pattern probability |
| `audio_freeze_encoder` | True | Freeze WavLM backbone during stage 1 |
| `video_freeze_encoder` | True | Freeze DINOv2 backbone during stage 1 |
| `text_freeze_encoder` | True | Freeze MentalBERT backbone during stage 1 |

## Two-stage training pipeline

### Stage 1: Fusion Pre-training (In-the-Wild)

**Objective:** Train the fusion mechanism (projection MLPs, quality estimators, CM-GEF, SLHT) on large-scale in-the-wild multimodal emotion data.

**Intended data:**

| Dataset | Size | Modalities | Emotion Labels |
|---|---|---|---|
| CMU-MOSEI | ~23K videos, 1K speakers | A+V+T | 6 emotions + sentiment |
| MELD | ~14K dialogues | A+V+T | 7 emotions |
| IEMOCAP | ~10K utterances, 10 speakers | A+V+T | 4 emotion categories |

**Trainable modules:** Projection MLPs (~0.6M params), Quality Estimators (~0.15M params), CM-GEF (~2M params), SLHT (~3M params), CCB (~0.5M params)

**Frozen modules:** WavLM Base Plus, DINOv2 Small, MentalBERT (pretrained checkpoints loaded from HuggingFace)

**Recommended recipe:**

| Setting | Value | Notes |
|---|---|---|
| Optimizer | AdamW | β₁=0.9, β₂=0.999, ε=1e-8 |
| Peak LR | 1e-4 | Linear warmup over 1000 steps |
| Batch size | 32 | Per GPU; gradient accumulation as needed |
| Weight decay | 0.01 | Excluded from bias/norm/LoRA parameters |
| Grad clip | 1.0 | Global norm |
| Precision | bf16 mixed | FP32 master weights; LayerNorm in float32 |
| Epochs | 50 | ~4 hours on 4×A100 (estimated) |
| Scheduler | Cosine decay | From peak LR to 1e-6 |

**Loss:**

```
L = L_emotion_cls + 0.1 · L_dropout_recon + 0.1 · L_quality_est
```

- `L_emotion_cls`: Cross-entropy on utterance-level emotion labels
- `L_dropout_recon`: MSE between masked modality features and reconstructed features (auxiliary, encourages cross-modal context quality)
- `L_quality_est`: Cross-entropy on synthetic corruption labels for quality estimator pre-training

**Modality dropout simulation during training:**
- Random per-modality dropout: 15% probability per modality per utterance
- Clinically-correlated patterns: 10% probability for 3 patterns (vision drop, audio drop, text drop)
- Implemented in `model.py` → `modality_dropout_augmentation()`

### Stage 2: Clinical Adaptation (Target Data)

**Objective:** Adapt the pretrained fusion mechanism to clinical depression assessment using scarce labeled data.

**Intended data:**

| Dataset | Size | Modalities | Labels |
|---|---|---|---|
| DAIC-WOZ | ~189 sessions | A+V+T (text from transcripts) | PHQ-8 total + binary depression |
| E-DAIC | ~275 sessions | A+V+T (extended set) | PHQ-8 total + binary depression |
| AVEC 2013/2014 | Optional | A+V | Depression severity |

**Trainable modules:** LoRA adapters on encoders (~5M params), CM-GEF (~2M params), SLHT (~3M params), CCB (~0.5M params)

**LoRA configuration:** r=8, α=16, applied to query/value/key/output/intermediate projections in WavLM, DINOv2, and MentalBERT encoders, plus fusion and session transformer.

**Recommended recipe:**

| Setting | Value | Notes |
|---|---|---|
| Optimizer | AdamW | β₁=0.9, β₂=0.999, ε=1e-8 |
| Peak LR | 5e-5 | Lower LR for LoRA adaptation |
| Batch size | 16 | Smaller due to session-level data |
| Weight decay | 0.01 | No weight decay on LoRA parameters (standard practice) |
| Grad clip | 1.0 | Global norm |
| Precision | bf16 mixed | N/A |
| Epochs | 30 | ~2 hours on 1×A100 (estimated) |
| Scheduler | Cosine decay | Linear warmup over 1000 steps |

**Loss:**

```
L = 1.0 · L_mae(PHQ-8_pred, PHQ-8_true)
  + 0.5 · L_bce(depression_pred, depression_true)
  + 0.3 · L_mse(item_pred, item_true)     [if item-level labels available]
  + 0.2 · L_bce(concept_scores, concept_labels)  [if concept labels available]
```

**Note on PHQ-8 item-level labels:** DAIC-WOZ does not natively provide per-item PHQ-8 scores. If unavailable, the item prediction head can be trained via self-supervised distillation from the total score or through the diagonal-dominant prior initialization.

## Expected behavior

> TODO: unverified — no reference training run has been conducted. Targets from the research contract:

| Metric | Target | Status |
|---|---|---|
| MAE on E-DAIC | < 3.0 | `TODO: unverified` |
| RMSE on E-DAIC | < 3.5 | `TODO: unverified` |
| Binary depression F1 (PHQ-8 ≥ 10) | > 0.80 | `TODO: unverified` |
| Degradation under 40% dropout | < 15% MAE increase | `TODO: unverified` |
| Sustained affect F1 gain vs. utterance-pooling | ≥ 10% | `TODO: unverified` |
| Cross-cultural fairness gap | ≤ 5 pp F1 | `TODO: unverified` |

After a training run, fill this section with actual values. Run `python evaluate.py --checkpoint <path>` (not yet implemented) to produce validation metrics.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Loss NaN in first steps | bf16 sensitive operation | Verify LayerNorm uses float32 (current implementation in `layers.py` does this) |
| Quality estimator scores collapse to constant | Feature distribution shift during stage 2 | Run quality estimator in inference mode during stage 2 (no gradient); add held-out corruption detection head |
| Gates saturate at 0 or 1 | Per-dimension gate overfitting on small data | Add gate entropy regularization; start with scalar gates and enable per-dimension after convergence |
| SLHT OOM with long sessions | O(T²) attention memory | Enable gradient checkpointing (`use_checkpoint=True`); reduce `max_utterances`; use linear attention variant |
| Phase labels unavailable | No annotation in dataset | Use heuristic time-window assignment (first 5 min = rapport, last 5 min = closure); compare phase-aware vs. phase-agnostic |
| LoRA doesn't improve over full fine-tune | Domain shift too large for low-rank adaptation | Increase `lora_r` (e.g., 16 → 32); or train full encoder adapters (adapter layers instead of LoRA) |
| Concept bottleneck increases MAE > 1 point | 8-dimensional concept space is too lossy | Increase `n_clinical_concepts`; add residual skip connection bypassing bottleneck |
| CM-GEF no better than late fusion under dropout | Gating mechanism not learning effective substitution | Verify quality estimator training; check gate statistics; increase `n_exchange_layers` |
| Model performs worse than last-value baseline | Temporal data leakage or bad normalization | Audit utterance ordering; verify phase labels; enforce strict chronological train/val/test splits |
| Missing modalities produce NaN outputs | Zero-features propagate through attention | Verify masks are applied correctly throughout all fusion stages (masked_fill, zero-out) |
