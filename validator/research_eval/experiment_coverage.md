# Experiment Coverage — MER Mental Health Validation

## Legend
- ✅ **Implemented** — artifact exists and is runnable
- ⚠️ **Partial** — exists but has gaps (e.g., synthetic data only)
- ❌ **Missing** — not yet implemented
- 🔄 **Automated** — script produces numbers
- 📋 **Manual** — requires human evaluation

---

## 1. Baselines Required by Research Contract

| # | Baseline | Status | Implementation | Notes |
|---|---|---|---|---|
| 1 | Unimodal text-only (MentalBERT) | ⚠️ Partial | `benchmarks.py` LateFusionWrapper can run with text-only by dropping A/V | Requires setting masks programmatically |
| 2 | Unimodal audio-only (WavLM) | ⚠️ Partial | Same mechanism as above | |
| 3 | Unimodal video-only (DINOv2) | ⚠️ Partial | Same mechanism as above | |
| 4 | Simple late fusion (ensemble) | ✅ Implemented | `benchmarks.py` → `benchmark_late_fusion_comparison()` | LateFusionWrapper aggregates by mean |
| 5 | Naive early fusion (concatenate) | ❌ Missing | Not yet implemented | Need an early fusion model class |
| 6 | MDD-MARF (current SOTA) | ❌ Missing | Requires re-implementation from paper | Significant engineering effort |
| 7 | Ablated gating (static weighted) | ✅ Implemented | `ablation.py` → `drop_cmgef` | n_exchange_layers=0 |

## 2. Evaluation Requirements

| # | Requirement | Status | Implementation | Notes |
|---|---|---|---|---|
| 1 | MAE on DAIC-WOZ/E-DAIC | ❌ Missing | Requires real data loaders | Framework in benchmarks.py |
| 2 | RMSE on DAIC-WOZ/E-DAIC | ❌ Missing | Same as above | |
| 3 | Binary depression accuracy/F1/AUC (PHQ-8 >= 10) | ❌ Missing | Requires trained model + test labels | CCB outputs depression_prob |
| 4 | 6 dropout conditions (0/20/40/60% per modality) | ✅ Implemented | `benchmarks.py` → `benchmark_missing_modality_stress()` | Runs on synthetic data |
| 5 | 3 clinically-correlated dropout patterns | ✅ Implemented | `benchmarks.py` → `benchmark_clinical_dropout_patterns()` | 3 patterns defined |
| 6 | Session-level sustained affect F1 | ⚠️ Partial | `benchmarks.py` → `benchmark_sustained_affect_detection()` | Proxy benchmark (synthetic low-variance affect) |
| 7 | Fairness by demographic group | ❌ Missing | Requires demographic labels in data | DAIC-WOZ/E-DAIC have gender labels |
| 8 | Clinician interpretability study | ❌ Missing | Requires ≥3 clinicians + Likert survey | Deferred |
| 9 | Inference latency, memory, FLOPs | ✅ Implemented | `profiling.py` | Supports inference + train modes, torch.profiler |

## 3. Single-Field Ablations (from Architect §9)

| # | Ablation | Config Field | Status | Implementation | Notes |
|---|---|---|---|---|---|
| 1 | Drop CM-GEF → late fusion | `n_exchange_layers: 2 → 0` | ✅ Implemented | `ablation.py` → `drop_cmgef` | |
| 2 | Scalar gates → per-dim gates | `exchange_hidden_dim: 128 → 0` | ✅ Implemented | `ablation.py` → `scalar_gates` | |
| 3 | Drop quality estimators | `n_audio_quality_classes: 3 → 0` | ✅ Implemented | `ablation.py` → `drop_quality_estimators` | Also drops video/text quality |
| 4 | Drop phase embeddings | `n_therapeutic_phases: 4 → 1` | ✅ Implemented | `ablation.py` → `drop_phase_embeddings` | |
| 5 | Drop session Transformer → mean pool | `n_session_layers: 4 → 0` | ✅ Implemented | `ablation.py` → `drop_session_transformer` | |
| 6 | LoRA vs. full fine-tune | `lora_r: 8 → 0` | ✅ Implemented | `ablation.py` → `lora_vs_full_finetune` | |
| 7 | Drop concept bottleneck | `n_clinical_concepts: 8 → 0` | ✅ Implemented | `ablation.py` → `drop_concept_bottleneck` | |
| 8 | Drop modality dropout simulation | `random_dropout_prob: 0.15 → 0.0` | ✅ Implemented | `ablation.py` → `drop_dropout_simulation` | |
| 9 | WavLM Base+ vs. Small | `audio_encoder_name: wavlm-base-plus → wavlm-small` | ✅ Implemented | `ablation.py` → `small_audio_encoder` | |
| 10 | Bidirectional vs. causal session | `streaming_mode: false → true` | ✅ Implemented | `ablation.py` → `streaming_mode` | |

## 4. Synthetic Benchmarks Implemented

| # | Benchmark | File | Function | Type |
|---|---|---|---|---|
| 1 | Missing modality stress test | `benchmarks.py` | `benchmark_missing_modality_stress` | 🔄 Automated |
| 2 | Clinical dropout patterns | `benchmarks.py` | `benchmark_clinical_dropout_patterns` | 🔄 Automated |
| 3 | Late fusion comparison | `benchmarks.py` | `benchmark_late_fusion_comparison` | 🔄 Automated |
| 4 | Quality estimator sensitivity | `benchmarks.py` | `benchmark_quality_estimator` | 🔄 Automated |
| 5 | Sustained affect detection (proxy) | `benchmarks.py` | `benchmark_sustained_affect_detection` | 🔄 Automated |
| 6 | Concept mapping fidelity | `benchmarks.py` | `benchmark_concept_mapping` | 🔄 Automated |
| 7 | Throughput and latency | `benchmarks.py` | `benchmark_throughput` | 🔄 Automated |

## 5. Unit Tests Implemented

| Test Class | File | # Tests | Scope |
|---|---|---|---|
| `TestShapes` | `test_model.py` | 9 | Output shapes for all components, variable sizes |
| `TestGradients` | `test_model.py` | 7 | Gradient flow to all modules, no NaN/Inf, LoRA grads |
| `TestNumerics` | `test_model.py` | 7 | BF16/FP16 stability, extreme inputs, silent audio, all-missing, gate saturation |
| `TestMissingModalityRobustness` | `test_model.py` | 3 | Single modality, dropout augmentation, weight sums |
| `TestClinicalConceptBottleneck` | `test_model.py` | 6 | Score ranges, diagonal initialization, importance methods |
| `TestQualityEstimator` | `test_model.py` | 2 | Reliability range, noisy vs. clean differentiation |
| `TestSessionTransformer` | `test_model.py` | 4 | Phase pooling shape, phase embedding, variable length, mask handling |
| `TestFusion` | `test_model.py` | 3 | Missing modalities, gated substitution, cross-modal attention shape |
| `TestTrainingPipeline` | `test_model.py` | 6 | Stage 1/2 trainable config, param counts, optimizer, loss, detach |

## 6. Metrics Reported

| Metric | Script | Status | Notes |
|---|---|---|---|
| MAE (clean and dropout) | `ablation.py`, `benchmarks.py` | ✅ Synthetic | Uses L1 loss on PHQ-8 |
| MAE degradation % | `benchmarks.py` | ✅ Synthetic | % increase from dropout |
| Throughput (utt/s) | `profiling.py`, `benchmarks.py` | ✅ Realistic | Hardware-dependent |
| Latency (ms/utterance) | `profiling.py` | ✅ Realistic | Per-utterance and per-session |
| Peak memory (GB) | `profiling.py` | ✅ CUDA only | Per-component breakdown |
| FLOP estimate (GFLOP) | `profiling.py` | ✅ Estimated | Kaplan 6× params estimate |
| Parameter count | `profiling.py`, `smoke_test.py` | ✅ Static | Total and trainable |
| Gate statistics | `test_model.py` | ✅ Static | Saturation check |

## 7. Results Still `TODO: unverified`

- Actual MAE/RMSE on DAIC-WOZ and E-DAIC (requires data + training)
- Whether CM-GEF outperforms late fusion under real dropout on clinical data
- Whether SLHT improves sustained affect detection on real clinical interviews
- Whether LoRA matches full fine-tune on clinical data with limited labels
- Whether concept bottleneck accuracy gap is acceptable (< 5% MAE increase)
- Demographic fairness metrics
- Clinician-rated interpretability scores

## 8. Can the Suite Distinguish the Architecture from a Trivial Baseline?

**Partially.** The benchmark suite includes:

✅ **Yes:** CM-GEF vs. late fusion comparison (`benchmark_late_fusion_comparison`)
✅ **Yes:** CM-GEF vs. ablated gating (ablation #1: `drop_cmgef`)
✅ **Yes:** SLHT vs. utterance mean pooling (ablation #5: `drop_session_transformer`)
✅ **Yes:** CCB vs. direct regression (ablation #7: `drop_concept_bottleneck`)
⚠️ **Partial:** Missing modality stress vs. static weighted fusion
⚠️ **Partial:** Unimodal baselines exist but require manual orchestration

❌ **Cannot yet distinguish:** MDD-MARF (not implemented), early fusion (not implemented)
