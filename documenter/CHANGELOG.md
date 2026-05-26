# Changelog

## [0.1.0] — 2026-05-26
### Added
- Initial implementation of MERMentalHealthModel — Multimodal Emotion Recognition for Mental Health Diagnostics.
- Architecture blocks: Cross-Modal Gated Exchange Fusion (CM-GEF), Session-Level Hierarchical Transformer (SLHT) with therapeutic phase embeddings, Clinical Concept Bottleneck (CCB) aligned to PHQ-8 items, Modality Quality Estimators.
- Unimodal encoder wrappers: WavLM Base Plus (audio), DINOv2 Small (video), MentalBERT (text) with LoRA support.
- Shared layers: safe LayerNorm (float32), multi-head self/cross-attention, gated substitution, attention pooling, LoRALinear with merge/unmerge.
- Two-stage training configuration (Stage 1: fusion pre-training; Stage 2: clinical adaptation with LoRA).
- Modality dropout augmentation (6 random + 3 clinically-correlated patterns).
- Unit test suite (35+ tests): shape correctness, gradient flow, numerical stability (BF16/FP16, extreme inputs, all-missing), domain-specific tests (missing modality robustness, concept bottleneck ranges, quality estimators, session transformer, fusion, training pipeline).
- Domain benchmarks (7): missing modality stress test, clinical dropout patterns, late fusion comparison, quality estimator sensitivity, sustained affect proxy, concept mapping fidelity, throughput/latency profiling.
- Ablation runner (10 single-field ablations): drop CM-GEF, scalar gates, drop quality estimators, drop phase embeddings, drop session transformer, LoRA vs. full fine-tune, drop concept bottleneck, drop dropout simulation, small audio encoder, streaming mode.
- Profiling script: inference/train modes, torch.profiler integration, per-component memory breakdown.
- Research evaluation: scorecard (3.3/5 average), claim grounding traceability, experiment coverage audit, rubric with domain-specific questions.
- Documentation: README, ARCHITECTURE (motivation, design decisions, shape evolution, domain considerations), TRAINING (two-stage recipe, troubleshooting), BENCHMARKS (results, ablations, profiling, research evaluation), API reference.
