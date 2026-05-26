# Claim Grounding — MER Mental Health Model

Every research claim is mapped to a source file, function, or `TODO: unverified`.
Claims without grounding are flagged as unverified.

---

## Architectural Claims

### Claim 1: "Cross-modal gated exchange (CM-GEF) with per-dimension feature substitution"
- **Status:** `grounded`
- **Source files:**
  - `coder/fusion.py` → `CrossModalGatedExchangeFusion.forward()` — full implementation of exchange attention + gated substitution + reliability aggregation
  - `coder/fusion.py` → `GatedSubstitution.forward()` — per-dimension gated interpolation
  - `coder/layers.py` → `CrossModalAttention.forward()` — multi-head cross-modal attention
  - `architect/mer-mental-health-architecture.md` §3 Block 2 — pseudocode and design rationale
- **Test coverage:**
  - `validator/test_model.py` → `TestFusion.test_fusion_with_missing_modalities()` — verifies gating under missing modalities
  - `validator/test_model.py` → `TestFusion.test_gated_substitution_bounds()` — gates in [0,1]
  - `validator/test_model.py` → `TestFusion.test_cross_modal_attention_shape()` — attention output shape
  - `validator/test_model.py` → `TestNumerics.test_gate_saturation_monitor()` — gates not saturated
- **Benchmark coverage:**
  - `validator/benchmarks.py` → `benchmark_late_fusion_comparison()` — CM-GEF vs. late fusion
  - `validator/benchmarks.py` → `benchmark_missing_modality_stress()` — CM-GEF under dropout
- **Ablation:**
  - `validator/ablation.py` → `drop_cmgef` — removes CM-GEF, measures effect

### Claim 2: "Session-Level Hierarchical Transformer (SLHT) with therapeutic phase embeddings"
- **Status:** `grounded`
- **Source files:**
  - `coder/session.py` → `SessionLevelTransformer.forward()` — full implementation
  - `coder/session.py` → `PhaseEmbedding.forward()` — learned phase embeddings
  - `coder/session.py` → `PhaseLevelPooling.forward()` — phase-conditional utterance aggregation
  - `architect/mer-mental-health-architecture.md` §3 Block 3 — pseudocode and phase assignment strategy
- **Test coverage:**
  - `validator/test_model.py` → `TestSessionTransformer.test_phase_pooling_shape()` — phase repr shape
  - `validator/test_model.py` → `TestSessionTransformer.test_phase_embedding()` — embedding shape
  - `validator/test_model.py` → `TestSessionTransformer.test_variable_length_sessions()` — handles T∈{4,8,16}
  - `validator/test_model.py` → `TestSessionTransformer.test_utterance_mask_handling()` — masked utterances ignored
  - `validator/test_model.py` → `TestShapes.test_session_transformer_output()` — (B, d_session), (B, n_phases, d_session)
- **Benchmark coverage:**
  - `validator/benchmarks.py` → `benchmark_sustained_affect_detection()` — session vs. utterance-level separation
- **Ablation:**
  - `validator/ablation.py` → `drop_session_transformer` — removes SLHT, measures effect
  - `validator/ablation.py` → `drop_phase_embeddings` — removes phase embeddings

### Claim 3: "Clinical Concept Bottleneck (CCB) with PHQ-8-aligned concepts"
- **Status:** `grounded`
- **Source files:**
  - `coder/concept.py` → `ClinicalConceptBottleneck.forward()` — full implementation
  - `coder/concept.py` → `_init_item_weights()` — diagonal-dominant initialization
  - `coder/concept.py` → `get_concept_importance()` — interpretability accessor
  - `coder/concept.py` → `get_concept_to_item_weights()` — weight matrix accessor
  - `architect/mer-mental-health-architecture.md` §3 Block 4 — pseudocode and regularization strategy
- **Test coverage:**
  - `validator/test_model.py` → `TestClinicalConceptBottleneck.test_concept_scores_in_range()` — scores in [0,1]
  - `validator/test_model.py` → `TestClinicalConceptBottleneck.test_phq8_items_in_range()` — items in [0,3]
  - `validator/test_model.py` → `TestClinicalConceptBottleneck.test_phq8_total_in_range()` — total in [0,24]
  - `validator/test_model.py` → `TestClinicalConceptBottleneck.test_diagonal_dominant_initialization()` — diag >> off-diag
  - `validator/test_model.py` → `TestClinicalConceptBottleneck.test_concept_importance_method()` — accessor works
  - `validator/test_model.py` → `TestClinicalConceptBottleneck.test_concept_to_item_weights_return()` — weight shape
  - `validator/test_model.py` → `TestShapes.test_concept_bottleneck_shape()` — all output shapes correct
- **Benchmark coverage:**
  - `validator/benchmarks.py` → `benchmark_concept_mapping()` — diagonal dominance ratio
- **Ablation:**
  - `validator/ablation.py` → `drop_concept_bottleneck` — removes CCB, measures accuracy impact

### Claim 4: "Modality Quality Estimators for per-utterance reliability scoring"
- **Status:** `grounded`
- **Source files:**
  - `coder/encoders.py` → `ModalityQualityEstimator.forward()` — MLP head predicting quality class
  - `architect/mer-mental-health-architecture.md` §3 Block 1 — pseudocode and training strategy
- **Test coverage:**
  - `validator/test_model.py` → `TestShapes.test_modality_quality_estimator_shapes()` — reliability + logits shapes
  - `validator/test_model.py` → `TestQualityEstimator.test_clean_input_high_reliability()` — valid range
  - `validator/test_model.py` → `TestQualityEstimator.test_noisy_features_lower_reliability()` — stochastic check
- **Benchmark coverage:**
  - `validator/benchmarks.py` → `benchmark_quality_estimator()` — sensitivity to synthetic noise
- **Ablation:**
  - `validator/ablation.py` → `drop_quality_estimators` — removes quality estimators

---

## Research Hypothesis Claims

### Claim 5: "Two-stage SSL pre-train + LoRA adaptation matches full fine-tuning"
- **Status:** `hypothesis`
- **Source files:**
  - `coder/model.py` → `set_trainable_params(stage=1)` and `set_trainable_params(stage=2)` — stage configuration
  - `coder/model.py` → `apply_lora()` — LoRA application to encoders/fusion/session
  - `coder/layers.py` → `LoRALinear` — LoRA linear layer implementation
  - `coder/layers.py` → `apply_lora_to_linear()` — recursive LoRA replacement
  - `architect/mer-mental-health-architecture.md` §10 — two-stage training pipeline detail
- **Ablation:**
  - `validator/ablation.py` → `lora_vs_full_finetune` — LoRA (r=8) vs. full fine-tune (r=0)
- **TODO:** Requires training on real clinical data with controlled compute budget

### Claim 6: "Per-dimension gating outperforms scalar modality gates"
- **Status:** `hypothesis`
- **Ablation:**
  - `validator/ablation.py` → `scalar_gates` — exchange_hidden_dim=0 = scalar gate
- **TODO:** Requires training to measure real effect size

### Claim 7: "Modality dropout simulation during training improves test-time robustness"
- **Status:** `hypothesis`
- **Source files:**
  - `coder/model.py` → `modality_dropout_augmentation()` — random + correlated dropout
  - `config.py` → `random_dropout_prob` and `clinical_correlated_dropout_prob`
- **Test coverage:**
  - `validator/test_model.py` → `TestMissingModalityRobustness.test_modality_dropout_augmentation_function()`
- **Ablation:**
  - `validator/ablation.py` → `drop_dropout_simulation` — removes all dropout augmentation
- **TODO:** Requires training with and without dropout, then evaluating on held-out dropout conditions

---

## Literature-Grounded Claims

### Claim 8: "No prior clinical MER work validates gated exchange on clinical data with correlated dropout"
- **Status:** `grounded`
- **Evidence in research artifact:**
  - `research/mer-mental-health-research.md` §3 (Gap 1) — literature analysis showing AMTE, Mi-CGA, and Sadeghi evaluate on lab data only
  - `research/mer-mental-health-research.md` §5 — known_related_work table documenting what each work leaves open
  - `architect/mer-mental-health-architecture.md` §6 row 1 — traceability: claim → CM-GEF block → validation hook
- **Validation:** `validator/benchmarks.py` provides the validation framework if run on clinical data

### Claim 9: "Session-level therapeutic phase modeling is absent from current MER"
- **Status:** `grounded`
- **Evidence in research artifact:**
  - `research/mer-mental-health-research.md` §3 (Gap 2) — DSIN, DSTC, and prior work use short clips or windowed aggregation
  - `research/mer-mental-health-research.md` §5 — known_related_work: DSIN, DSTC
  - `architect/mer-mental-health-architecture.md` §6 row 2 — traceability: claim → SLHT block
- **Validation:** `validator/benchmarks.py` → `benchmark_sustained_affect_detection()` (proxy)

### Claim 10: "No clinical MER reports disaggregated demographic fairness metrics"
- **Status:** `grounded`
- **Evidence in research artifact:**
  - `research/mer-mental-health-research.md` §3 (Gap 4) — surveys identify cross-cultural bias gap
  - `research/mer-mental-health-research.md` §5 — blocking_unknowns: DAIC-WOZ demographic diversity concern
  - `architect/mer-mental-health-architecture.md` §6 row 4 — traceability: claim → evaluation protocol
- **TODO:** Requires demographic labels and sufficiently diverse dataset to evaluate

---

## Performance Target Claims (Expected Observable Behavior)

These are the research contract's expected targets. **None have been validated** as no model training has been conducted.

| Target | Claimed Value | Validation Required | Current Status |
|---|---|---|---|
| MAE on E-DAIC | < 3.0 | Train on DAIC-WOZ, evaluate on E-DAIC | `TODO: unverified` |
| RMSE on E-DAIC | < 3.5 | Same as above | `TODO: unverified` |
| Degradation under 40% dropout | < 15% | Run dropout stress test on trained model | `TODO: unverified` |
| Sustained affect F1 gain | ≥ 10% over windowed baseline | Compare SLHT vs. utterance-pooling on clinical labels | `TODO: unverified` |
| Cross-cultural fairness gap | ≤ 5 pp F1 | Disaggregated evaluation on diverse test set | `TODO: unverified` |

---

## Implementation Claims

### Claim 11: "Full pipeline: raw inputs → PHQ-8 prediction"
- **Status:** `grounded`
- **Source files:**
  - `coder/model.py` → `MERMentalHealthModel.forward()` — end-to-end forward pass
  - `coder/model.py` → `MERMentalHealthModel.encode_utterance()` — per-utterance encoding
  - `coder/model.py` → `MERMentalHealthModel.forward_streaming()` — streaming mode
  - `coder/smoke_test.py` → verifies all forward modes produce correct shapes
- **Test coverage:** All tests in `test_model.py` exercise the pipeline end-to-end

### Claim 12: "Gradient checkpointing support"
- **Status:** `grounded`
- **Source files:**
  - `coder/fusion.py` → `forward_with_checkpoint()`
  - `coder/session.py` → `forward_with_checkpoint()`
- **Test coverage:**
  - `validator/test_model.py` → `TestGradients` — verifies gradient flow (including with checkerpoint-compatible API)

### Claim 13: "Modular design — each component independently testable"
- **Status:** `grounded`
- **Evidence:** Each of the 4 novel blocks has:
  - Abstract base class (`BaseFusionOperator`, `BaseSessionOperator`, `BaseConceptBottleneck`, `BaseEncoder`)
  - Independent instantiation in tests (e.g., `ClinicalConceptBottleneck(cfg)` tested without full model)
  - Config-driven enable/disable via `ModelConfig` field changes

---

## Summary of Ungrounded Claims

| Claim | Reason | What's Needed |
|---|---|---|
| Clinical performance targets (MAE < 3.0) | No training conducted | Data loaders + training + evaluation on E-DAIC |
| Demograpic fairness gap ≤ 5 pp | No demographic evaluation | Demographic labels + disaggregated metrics |
| Clinician interpretability score | No clinician study | IRB approval + clinician recruitment + Likert survey |
| Clinically-correlated dropout is harder than random | No empirical comparison | Controlled dropout experiment on clinical data |
| LoRA effectiveness for clinical MER | No comparison to full fine-tune | Label-efficiency sweep with real training |
| Phase embeddings improve sustained affect | No phase-labeled clinical data | Phase annotation + sustained affect evaluation |
