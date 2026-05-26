# Benchmarks

All numbers are reproducible with the commands shown. Numbers marked `TODO: unverified` have not been measured on real clinical data or trained models — do not cite them as validated results.

**Important context:** All benchmarks currently run on synthetic random data with a randomly initialized (untrained) model. No model has been trained on clinical data. Benchmarks verify that the *infrastructure* produces meaningful outputs and distinguishes between architectural variants, not that the model achieves clinically useful performance.

## Missing modality stress test

Evaluates the model under 6 dropout conditions (0%, 20%, 40%, 60% per modality). Measures MAE degradation relative to the full-modality baseline.

Research contract target: **< 15% degradation under 40% dropout** (vs. > 30% expected for non-adaptive fusion).

> `TODO: unverified` — requires a trained model on clinical data. Current results are from random initialization on synthetic data.

```bash
python validator/benchmarks.py --benchmark missing_modality
```

## Clinically-correlated dropout patterns

Three clinical scenario patterns:
1. **Vision drop during distress** (40% video missing) — patient turns away
2. **Audio drop in noisy environment** (30% audio missing) — poor room acoustics
3. **Text drop when non-verbal** (40% text + 10% A/V) — patient silent/crying

> `TODO: unverified` — requires trained model. These 3 patterns are implemented but not validated against real clinical dropout distributions.

```bash
python validator/benchmarks.py --benchmark clinical_dropout
```

## Late fusion baseline comparison

Compares CM-GEF against a simple late fusion baseline (mean of available modality predictions) under 0% and 40% random dropout.

Research contract target: **CM-GEF MAE >2 points better than late fusion under 40% dropout**.

> `TODO: unverified` — requires trained model.

```bash
python validator/benchmarks.py --benchmark late_fusion
```

## Quality estimator sensitivity

Measures whether the quality estimator produces different reliability scores for clean vs. synthetically corrupted features. With random initialization, sensitivity is expected to be low.

```bash
python validator/benchmarks.py --benchmark quality_estimator
```

## Session-level sustained affect detection (proxy)

Proxy benchmark comparing session-level representation separation (SLHT) vs. utterance-level mean pooling for sessions with artificially induced "flat affect" segments (low-amplitude waveform).

Research contract target: **≥10% F1 improvement over utterance-level pooling**.

> `TODO: unverified` — replaces synthetic proxy with real clinical data and sustained affect labels.

```bash
python validator/benchmarks.py --benchmark sustained_affect
```

## Concept mapping fidelity

Verifies the concept bottleneck's diagonal-dominant initialization: each concept k should predominantly influence PHQ-8 item k. Measured as diagonal/off-diagonal weight ratio.

| Metric | Measured | Pass criteria | Status |
|---|---|---|---|
| Diagonal mean | ~1.0 (initialized) | — | ✅ (architectural property) |
| Off-diagonal mean | ~0.01 (initialized) | — | ✅ (architectural property) |
| Diagonal/off-diagonal ratio | ~100 | > 2.0 | ✅ (architectural property) |

```bash
python validator/benchmarks.py --benchmark concept_mapping
```

## Throughput and latency

Measured on random-init model; real throughput depends on pretrained encoder inference cost.

```bash
python validator/benchmarks.py --benchmark throughput
python validator/profiling.py --device cuda
```

## Ablation study

Each ablation is a single `ModelConfig` field change that tests a specific hypothesis (from architect §9). Results on synthetic random-init data show the *infrastructure* works — effect sizes are not meaningful without training.

| # | Ablation | Config delta | Primary metric | Expected movement | Status |
|---|---|---|---|---|---|
| 1 | Drop CM-GEF → late fusion | `n_exchange_layers: 2 → 0` | MAE (40% dropout) | MAE ↑ by >2 points under dropout | Implemented; requires training |
| 2 | Scalar gates → per-dim gates | `exchange_hidden_dim: 128 → 0` | MAE (clean) | MAE ↑ ≤ 0.5 point | Implemented; requires training |
| 3 | Drop quality estimators | `n_audio_quality_classes: 3 → 0` | MAE (noise) | MAE ↑ under noise/occlusion | Implemented; requires training |
| 4 | Drop phase embeddings | `n_therapeutic_phases: 4 → 1` | Sustained affect F1 | F1 ↓ ≥ 10% | Implemented; requires training |
| 5 | Drop session Transformer | `n_session_layers: 4 → 0` | Sustained affect F1 | F1 ↓ ≥ 10% | Implemented; requires training |
| 6 | LoRA vs. full fine-tune | `lora_r: 8 → 0` | MAE | Both MAE same; LoRA faster | Implemented; requires training |
| 7 | Drop concept bottleneck | `n_clinical_concepts: 8 → 0` | MAE | MAE ↑ ≤ 0.5 point | Implemented; requires training |
| 8 | Drop dropout simulation | `random_dropout_prob: 0.15 → 0.0` | MAE (dropout) | MAE ↑ on dropout | Implemented; requires training |
| 9 | WavLM Base+ vs. Small | `audio_encoder_name: wavlm-base-plus → wavlm-small` | MAE | MAE ↑ ≤ 0.5; latency ↓ 40% | Implemented; requires training |
| 10 | Bidirectional vs. causal | `streaming_mode: false → true` | Sustained affect F1 | F1 ↓; enables streaming | Implemented; requires training |

Reproduce all:
```bash
python validator/ablation.py --output ablation_results.json
```

## Profiling

GPU: A100 (80 GB), bf16 (estimated — not yet measured on GPU with full-size model):

| Phase | Time | Notes |
|---|---|---|
| Per-utterance forward | `TODO: unverified` | — |
| Per-session forward (32 utt) | `TODO: unverified` | — |
| Training step (fwd + bwd) | `TODO: unverified` | Stage 1 config |
| Peak memory (inference) | `TODO: unverified` | — |
| Peak memory (training) | `TODO: unverified` | — |

Estimated FLOPs: ~6 × total_params × T_utterances per sample (Kaplan et al. scaling estimate for transformer-heavy models). For the default config (~23M params, T=32): ~4.4 GFLOP/sample.

```bash
python validator/profiling.py --device cpu          # CPU profiling
python validator/profiling.py --device cuda         # GPU (requires CUDA)
python validator/profiling.py --device cuda --mode train  # Training profile
python validator/profiling.py --profile             # torch.profiler breakdown
```

## Baseline completeness

| Baseline | Status | Implementation |
|---|---|---|
| Unimodal text-only | ⚠️ Partial (via mask manipulation) | `benchmarks.py` LateFusionWrapper |
| Unimodal audio-only | ⚠️ Partial (via mask manipulation) | `benchmarks.py` LateFusionWrapper |
| Unimodal video-only | ⚠️ Partial (via mask manipulation) | `benchmarks.py` LateFusionWrapper |
| Simple late fusion (ensemble) | ✅ Implemented | `benchmarks.py` → `benchmark_late_fusion_comparison()` |
| Naive early fusion (concat + MLP) | ❌ Missing | Not yet implemented |
| MDD-MARF (current SOTA: MAE 3.13) | ❌ Missing | Requires re-implementation from paper |
| Ablated gating (n_exchange_layers=0) | ✅ Implemented | `ablation.py` → `drop_cmgef` |

## Research-quality evaluation

| Dimension | Score | Evidence | Gaps |
|---|---|---|---|
| Novelty | 3/5 | 5 documented gaps in clinical MER literature; 4 novel blocks (CM-GEF, SLHT, CCB, Quality Estimators) | No external paper validation (Level 1 task); components individually established |
| Experimental coverage | 4/5 | 35+ unit tests (shapes, gradients, numerics, domain); 7 domain benchmarks; 10 ablations; profiling (inference + train) | No real clinical data integration; early fusion baseline missing; MDD-MARF missing |
| Theoretical foundation | 4/5 | Per-component inductive bias justifications with grounded/hypothesis labels; complexity analysis across 7 architecture families; 5 risk flags with mitigations | No formal bounds (generalization, approximation, sample complexity) |
| Result analysis | 2/5 | All metrics computable via scripts | Results on synthetic random data only; no trained model; no SOTA comparison |
| Implementation reproducibility | 4/5 | Complete PyTorch code with ModelConfig; pytest + smoke test + benchmarks all runnable; seed=42 | No requirements.txt; no training script; no data loaders |
| Writing readiness | 3/5 | Comprehensive architecture documentation, tests, benchmarks | No trained model results; interpretability study deferred; fairness evaluation blocked |
| **Average** | **3.3/5** | | |

**Required next experiments** (from research scorecard):

1. Integrate DAIC-WOZ/E-DAIC data loaders and train the model
2. Run all 10 ablations with a trained model to measure real effect sizes
3. Implement MDD-MARF baseline and compare against CM-GEF on clinical test splits
4. Run missing modality stress test with 6 dropout conditions on clinical data
5. Compute demographic fairness metrics on E-DAIC (which has gender labels)
6. Measure sustained affect F1 by training a probe classifier on session features
7. Run label-efficiency sweep (5/10/25/50/100% of DAIC-WOZ) to validate LoRA hypothesis
8. Implement streaming-mode latency benchmark for real-time deployment viability
