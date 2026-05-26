# MER Mental Health — Research Quality Evaluation Rubric

**Model:** Cross-Modal Gated Exchange Fusion (CM-GEF)
+ Session-Level Hierarchical Transformer (SLHT)
+ Clinical Concept Bottleneck (CCB)

**Domain:** Multimodal Emotion Recognition for Mental Health Diagnostics (Speech/Audio + CV + LM)

**Date:** 2026-05-26

---

## Scoring Scale

| Score | Meaning |
|---|---|
| 0 | Not addressed or no artifact exists |
| 1 | Mentioned but unsupported |
| 2 | Partially supported with major gaps |
| 3 | Plausible and minimally supported |
| 4 | Strong, with clear evidence and reproducible checks |
| 5 | Publication-ready for this scaffold's scope |

---

## Dimension 1: Novelty

Evaluates whether the proposed architecture represents a genuine contribution beyond existing work.

### Criteria for Clinical MER:
- Does the architecture address a gap that is clearly documented in the literature?
- Are the novel components (CM-GEF gated exchange, SLHT phase modeling, CCB) clearly distinguished from prior art (AMTE, Mi-CGA, MDD-MARF, DSTC)?
- Is there a falsification condition that would disprove the novelty claim?
- Are known related works properly cited with what they leave open?

### Key questions:
- **Missing modality robustness:** Does the gated exchange mechanism actually improve over static fusion under clinically-correlated dropout, or is the gain solely from architectural capacity?
- **Session-level modeling:** Is the benefit from therapeutic phase embeddings specifically, or from having any longer-context mechanism?
- **Clinical interpretability:** Does the concept bottleneck provide clinically actionable insights beyond standard attention visualization?

### Current score: 3/5
- 5 gaps identified and well-documented against known related work
- Level 1 task — novelty claims are internally consistent but unvalidated against external papers
- Components are established individually; novelty is in combination for clinical MER

---

## Dimension 2: Experimental Comprehensiveness

Evaluates whether the validation infrastructure would distinguish the proposed architecture from trivial baselines.

### Criteria for Clinical MER:
- **Missing modality stress test:** 6+ dropout conditions (each modality 0/20/40/60% missing + 3 clinically-correlated patterns)
- **Baseline comparison:** Unimodal text/audio/video + late fusion + early fusion + MDD-MARF + ablated gating
- **Session-level evaluation:** Sustained flat affect F1, emotional reactivity following therapist probe
- **Fairness:** Disaggregated metrics by gender, ethnicity, age (where data permits)
- **Efficiency:** Latency, memory, FLOPs, streaming capability
- **Interpretability:** Concept→PHQ-8 mapping inspectability

### Key questions:
- Are there at least 5 baseline comparisons? (Partially — 4 of 5 planned, MDD-MARF missing)
- Does the suite include at least 6 dropout conditions? (Implemented in benchmarks.py)
- Is there an ablation for every novel component? (Yes — 10 ablations in ablation.py)
- Are synthetic benchmarks supplemented with real-data evaluation? (No — major gap)

### Current score: 4/5
- Extensive unit tests (35+), domain benchmarks (7), ablations (10), and profiling
- Synthetic data benchmarks exist for all required conditions
- Major gap: no real clinical data integration or trained model results
- Early fusion baseline and MDD-MARF not yet implemented

---

## Dimension 3: Theoretical Foundation

Evaluates the depth of reasoning behind architectural choices.

### Criteria for Clinical MER:
- Are inductive biases explicitly stated and justified (e.g., "per-dimension gating because unreliable modalities may have reliable subspaces")?
- Is the missing-modality handling strategy motivated by clinical deployment constraints?
- Is there a complexity analysis (time, space, parallelism)?
- Are there risk flags with mitigations?
- Is the training pipeline (2-stage) justified against alternatives?

### Key questions:
- Does the quality estimator have a principled training objective? (Yes — synthetic corruption labels)
- Is the phase labeling strategy robust to annotation variability? (Discussed with mitigations)
- Is there a formal understanding of when the concept bottleneck degrades accuracy? (Discussed but not quantified)

### Current score: 4/5
- All design decisions have one-sentence justifications with grounded/hypothesis labels
- Complexity table comparing 7 architecture families
- 5 risk flags with mitigations
- Missing: formal theoretical bounds, Bayesian treatment of uncertainty

---

## Dimension 4: Result Analysis

Evaluates whether the results convincingly support the claims.

### Criteria for Clinical MER:
- MAE and RMSE on DAIC-WOZ and E-DAIC test sets
- Degradation under dropout conditions (quantified as % MAE increase)
- Session-level sustained affect F1
- Demographic subgroup metrics
- Latency/memory profiling
- Comparison to published SOTA (MDD-MARF: MAE 3.13, RMSE 3.59)

### Key questions:
- Are results reported with confidence intervals? (No — synthetic data only)
- Does the model meet the target MAE < 3.0 on E-DAIC? (Cannot assess — no training)
- Is the degradation under 40% dropout < 15%? (Cannot assess — random init)
- Is there a statistical comparison to baselines? (Framework exists but not populated)

### Current score: 2/5
- All metrics are computable via scripts but no actual model has been trained
- Results are on synthetic random data — not meaningful for publication
- SOTA comparison framework exists but MDD-MARF not implemented

---

## Dimension 5: Implementation Reproducibility

Evaluates whether another researcher can reproduce the results.

### Criteria for Clinical MER:
- Is the full model code available with config? (Yes — 7 Python files + config)
- Are all test/benchmark commands documented? (Yes — in file headers)
- Is the dataset pipeline described? (Partially — no actual data loaders)
- Is the random seed fixed? (Yes — seed=42)
- Are dependencies specified? (No requirements.txt)

### Key questions:
- Can someone run the smoke test without downloading datasets? (Yes — all synthetic)
- Can someone run all benchmarks with one command? (Yes)
- Are the hyperparameters fully specified? (Yes — ModelConfig dataclass)
- Is there a training script? (Partially — stage config methods exist, no train.py)

### Current score: 4/5
- Clean modular code with docstrings and shape conventions
- pytest suite + smoke test + benchmarks all runnable
- Missing: requirements.txt, data loaders, training script, trained checkpoints

---

## Dimension 6: Writing Readiness

Evaluates how close this artifact is to supporting a research paper.

### Domain-specific questions:

**Speech/Audio:**
- Does the benchmark test log-mel stability? (Not applicable — uses raw waveform → WavLM)
- Does it test causality/streaming? (Yes — streaming_mode ablation + test)
- Does it compare against acoustic-only baseline? (Planned as unimodal baseline)

**Computer Vision:**
- Does it test occlusion handling? (Quality estimator + missing modality test)
- Does it compare against a simple CNN baseline? (Not yet — DINOv2 is the only video encoder)

**Language Models:**
- Does it handle short utterances? (Quality estimator detects short/informative)
- Does it use a domain-adapted model? (Yes — MentalBERT)

**Multimodal Fusion (central design):**
- Does it test missing modality robustness? (Yes — 7 conditions)
- Does it test gated exchange vs. static fusion? (Yes — ablation + benchmark)
- Does it report efficiency? (Yes — profiling script)

**Time Series (session-level):**
- Does it respect chronological order? (Yes — position encoding + timestamps)
- Does it test against utterance-level pooling? (Yes — ablation #5)

**Scientific ML (clinical concepts):**
- Does it test concept bottleneck fidelity? (Yes — diagonal dominance)
- Does it provide interpretability tools? (Yes — get_concept_importance, get_concept_to_item_weights)

### Current score: 3/5
- Strong architecture documentation, tests, and benchmarks
- No trained model or real-data results for a paper
- Interpretability study requires clinical partners
- Fairness analysis requires demographic data

---

## Overall Assessment

| Dimension | Score | Key Strength | Key Gap |
|---|---|---|---|
| Novelty | 3/5 | 5 documented gaps in clinical MER | No external paper validation (Level 1) |
| Experimental Coverage | 4/5 | 35+ tests, 7 benchmarks, 10 ablations | No real clinical data |
| Theoretical Foundation | 4/5 | Per-component justifications + complexity analysis | No formal bounds |
| Result Analysis | 2/5 | All metrics computable via scripts | Only synthetic random data |
| Implementation | 4/5 | Complete modular PyTorch code | No requirements.txt, no train.py |
| Writing Readiness | 3/5 | Comprehensive docs + validation | No trained model results |
| **Average** | **3.3/5** | | |

**To reach 4.5+ (publication-ready):**
1. Integrate DAIC-WOZ/E-DAIC data loaders
2. Train the model and report real metrics
3. Implement MDD-MARF baseline
4. Run fairness evaluation on E-DAIC demographic splits
5. Conduct clinician interpretability study (or substitute with automated probe)
6. Add requirements.txt and training script
