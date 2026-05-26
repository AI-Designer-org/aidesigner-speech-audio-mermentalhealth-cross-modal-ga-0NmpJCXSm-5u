> **Project layout** — this bundle contains five stage directories from the
> AI-Designer pipeline:
> `research/` (literature survey), `architect/` (blueprint + `ModelConfig`),
> `coder/` (PyTorch implementation), `validator/` (tests + benchmarks), and
> `documenter/` (this README plus `docs/` and `CHANGELOG.md`).
> An optional `paper/` directory holds the NeurIPS-format writeup when the
> paper-generation step was triggered.
>
> The original research request that produced this bundle is preserved
> verbatim in [`prompt.md`](prompt.md) — if any URLs in the prompt were
> fetched server-side for additional context, their cleaned contents are
> appended there too.

---

# MERMentalHealth: Cross-Modal Gated Fusion for Clinical Emotion Recognition

A lightweight multimodal fusion architecture for depression severity estimation from clinical interview recordings. Combines audio (WavLM), video (DINOv2), and text (MentalBERT) with adaptive gated exchange that substitutes unreliable modalities, session-level hierarchical temporal modeling with therapeutic phase awareness, and an interpretable clinical concept bottleneck aligned to PHQ-8 depression criteria.

This project addresses five gaps in clinical multimodal emotion recognition (MER): (1) missing-modality robustness under clinically-correlated dropout patterns, (2) session-level temporal modeling of 30–60 minute therapeutic interviews, (3) label-efficient clinical adaptation via two-stage pre-training and LoRA, (4) demographic fairness auditing, and (5) clinically-grounded interpretability via concept bottlenecks.

> **Note:** All experimental results currently reflect randomly initialized models on synthetic data. No model has been trained on clinical data. Quantitative targets (e.g., MAE < 3.0 on E-DAIC) are research hypotheses awaiting empirical validation. See [docs/BENCHMARKS.md](documenter/docs/BENCHMARKS.md) for status.

## Highlights

- **Missing-modality-robust gated fusion** — Per-dimension gated substitution with reliability-weighted aggregation provides triple redundancy against modality dropout; see [ARCHITECTURE.md#32-cross-modal-gated-exchange-fusion-cm-gef](documenter/docs/ARCHITECTURE.md#32-cross-modal-gated-exchange-fusion-cm-gef)
- **Therapeutic-phase-aware session modeling** — 4-phase hierarchical transformer (rapport, exploration, intervention, closure) captures session-level emotion dynamics absent from prior windowed approaches; see [ARCHITECTURE.md#33-session-level-hierarchical-transformer-slht](documenter/docs/ARCHITECTURE.md#33-session-level-hierarchical-transformer-slht)
- **PHQ-8-grounded concept bottleneck** — 8 interpretable clinical concepts (anhedonia, depressed mood, etc.) map to DSM-5 depression criteria with diagonal-dominant initialization; see [ARCHITECTURE.md#34-clinical-concept-bottleneck-ccb](documenter/docs/ARCHITECTURE.md#34-clinical-concept-bottleneck-ccb)
- **Two-stage training pipeline** — Fusion pre-training on in-the-wild multimodal data followed by LoRA-parameter-efficient adaptation to clinical data (~189 sessions); see [docs/TRAINING.md](documenter/docs/TRAINING.md#two-stage-training-pipeline)
- **Comprehensive validation infrastructure** — 35+ unit tests, 7 domain benchmarks, 10 ablations, and profiling across all computational dimensions; see [docs/BENCHMARKS.md](documenter/docs/BENCHMARKS.md)

## Quick start

```bash
# Clone and install (pre-trained weights require HuggingFace transformers)
pip install torch transformers

# Smoke test with randomly initialized model
python coder/smoke_test.py

# Run full test suite
python -m pytest validator/test_model.py -v

# Run benchmarks (synthetic data)
python validator/benchmarks.py --fast

# Run all ablations
python validator/ablation.py --fast

# Profile latency and memory
python validator/profiling.py
```

## Repository layout

```
coder/                          # Core implementation
  config.py                     # ModelConfig dataclass (all hyperparameters)
  model.py                      # MERMentalHealthModel (full pipeline)
  encoders.py                   # WavLM, DINOv2, MentalBERT wrappers + quality estimators
  fusion.py                     # CrossModalGatedExchangeFusion (CM-GEF)
  session.py                    # SessionLevelTransformer (SLHT)
  concept.py                    # ClinicalConceptBottleneck (CCB)
  layers.py                     # Shared layers (attention, LoRA, normalization, pooling)
  smoke_test.py                 # End-to-end forward pass verification
validator/                      # Validation infrastructure
  test_model.py                 # 35+ pytest tests (shapes, gradients, numerics, domain)
  benchmarks.py                 # 7 domain benchmarks (dropout, fusion, affect, etc.)
  ablation.py                   # 10 single-field ablation experiments
  profiling.py                  # Latency, memory, FLOP profiling
  research_eval/                # Research quality evaluation
    scorecard.json              # Scored evaluation (3.3/5 average)
    claim_grounding.md          # Claim-to-source traceability
    experiment_coverage.md      # Coverage audit
    rubric.md                   # Scoring rubric
docs/                           # Documentation
  ARCHITECTURE.md               # Design rationale and component details
  TRAINING.md                   # Training recipes and environment
  BENCHMARKS.md                 # Results, ablations, profiling
  API.md                        # API reference
```

## Documentation

- [docs/ARCHITECTURE.md](documenter/docs/ARCHITECTURE.md) — Design rationale, inductive biases, shape evolution, and traceability
- [docs/TRAINING.md](documenter/docs/TRAINING.md) — Two-stage training pipeline, environment setup, troubleshooting
- [docs/BENCHMARKS.md](documenter/docs/BENCHMARKS.md) — Domain benchmarks, ablation results, profiling, research-evaluation gaps
- [docs/API.md](documenter/docs/API.md) — Module-level API reference with shape contracts

## Citation

```bibtex
@misc{mer-mental-health-2026,
  title  = {MERMentalHealth: Cross-Modal Gated Fusion for Clinical Emotion Recognition},
  author = {ML-Designer Pipeline Contributors},
  year   = {2026},
  note   = {Generated via ml-designer pipeline. Unvalidated research artifact.}
}
```
