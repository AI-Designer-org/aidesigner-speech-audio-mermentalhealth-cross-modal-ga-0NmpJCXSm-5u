"""
Ablation runner for the MER Mental Health Model.

Each ablation is a single ModelConfig field change that tests a specific
hypothesis about the architecture.

Ablations (10 total, from architect §9):
  1. Drop CM-GEF → simple late fusion (n_exchange_layers: 2 → 0)
  2. Scalar gates → per-dim gates (exchange_hidden_dim: 128 → 0)
  3. Drop quality estimators (n_audio_quality_classes: 3 → 0)
  4. Drop phase embeddings (n_therapeutic_phases: 4 → 1)
  5. Drop session Transformer → mean pool (n_session_layers: 4 → 0)
  6. LoRA vs. full fine-tune (lora_r: 8 → 0)
  7. Drop concept bottleneck (n_clinical_concepts: 8 → 0)
  8. Drop modality dropout simulation (random_dropout_prob: 0.15 → 0.0)
  9. WavLM Base+ vs. Small (audio_encoder_name: wavlm-base-plus → wavlm-small)
  10. Bidirectional vs. causal session (streaming_mode: false → true)

Usage:
    python ablation.py                               # Run all ablations
    python ablation.py --ablation drop_cmgef         # Run single ablation
    python ablation.py --device cuda --fast          # Quick run on GPU
    python ablation.py --output ablation_results.json  # Save results
"""

import sys
import json
import time
import argparse
from dataclasses import replace
from typing import Dict, List, Optional, Callable, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/artifacts/j_0NmpJCXSm-5u/work/coder")

from config import ModelConfig
from model import MERMentalHealthModel, create_optimizer


# ---------------------------------------------------------------------------
# Shared evaluation function
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_ablation_model(model: nn.Module, cfg: ModelConfig,
                            device: torch.device, dtype: torch.dtype,
                            n_batches: int = 4, n_utterances: int = 24,
                            eval_dropout: float = 0.2) -> Dict[str, float]:
    """Evaluate a model on synthetic data and return metrics.

    Metrics:
      - mae: mean absolute error on PHQ-8 total score
      - loss: total combined loss
      - mae_clean: MAE with all modalities available
      - mae_dropout: MAE under dropout conditions

    All metrics are computed on synthetic data. Real clinical evaluation
    requires DAIC-WOZ/E-DAIC data loaders.
    """
    B = 2
    T = n_utterances
    model.eval()

    metrics = {}

    # ── Clean evaluation (all modalities) ──
    mae_clean_list = []
    for _ in range(n_batches):
        waveforms = torch.randn(B, T, 16000, device=device, dtype=dtype)
        frames = torch.randn(B, T, cfg.video_n_frames, 3, cfg.video_frame_size,
                             cfg.video_frame_size, device=device, dtype=dtype)
        input_ids = torch.randint(0, 1000, (B, T, cfg.text_max_length), device=device)
        phq8_labels = torch.rand(B, device=device) * 24
        phase_labels = torch.randint(0, max(cfg.n_therapeutic_phases, 1), (B, T), device=device)
        utterance_mask = torch.ones(B, T, device=device, dtype=torch.bool)

        out = model(
            waveforms=waveforms, frames=frames, input_ids=input_ids,
            phase_labels=phase_labels, utterance_mask=utterance_mask,
            phq8_labels=phq8_labels,
        )
        mae = F.l1_loss(out.phq8_total, phq8_labels).item()
        mae_clean_list.append(mae)

    metrics["mae_clean"] = sum(mae_clean_list) / max(len(mae_clean_list), 1)

    # ── Dropout evaluation ──
    mae_drop_list = []
    for _ in range(n_batches):
        waveforms = torch.randn(B, T, 16000, device=device, dtype=dtype)
        frames = torch.randn(B, T, cfg.video_n_frames, 3, cfg.video_frame_size,
                             cfg.video_frame_size, device=device, dtype=dtype)
        input_ids = torch.randint(0, 1000, (B, T, cfg.text_max_length), device=device)
        phq8_labels = torch.rand(B, device=device) * 24
        audio_mask = torch.rand(B, T, device=device) > eval_dropout
        video_mask = torch.rand(B, T, device=device) > eval_dropout
        text_mask = torch.rand(B, T, device=device) > eval_dropout
        phase_labels = torch.randint(0, max(cfg.n_therapeutic_phases, 1), (B, T), device=device)
        utterance_mask = torch.ones(B, T, device=device, dtype=torch.bool)

        out = model(
            waveforms=waveforms, frames=frames, input_ids=input_ids,
            audio_mask=audio_mask, video_mask=video_mask, text_mask=text_mask,
            phase_labels=phase_labels, utterance_mask=utterance_mask,
            phq8_labels=phq8_labels,
        )
        mae = F.l1_loss(out.phq8_total, phq8_labels).item()
        mae_drop_list.append(mae)

    metrics["mae_dropout"] = sum(mae_drop_list) / max(len(mae_drop_list), 1)
    metrics["dropout_penalty"] = metrics["mae_dropout"] - metrics["mae_clean"]

    # ── Param count ──
    metrics["total_params"] = sum(p.numel() for p in model.parameters())
    metrics["trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return metrics


# ---------------------------------------------------------------------------
# Ablation definitions
# ---------------------------------------------------------------------------

class Ablation:
    """Single ablation experiment."""

    def __init__(self, name: str, description: str, cfg_modifier: Callable[[ModelConfig], ModelConfig],
                 hypothesis: str, expected_movement: str, owning_stage: str):
        self.name = name
        self.description = description
        self.cfg_modifier = cfg_modifier
        self.hypothesis = hypothesis
        self.expected_movement = expected_movement
        self.owning_stage = owning_stage

    def run(self, baseline_cfg: ModelConfig, device: torch.device,
            dtype: torch.dtype, n_batches: int) -> Dict:
        """Run the ablation and return comparison metrics."""
        ablated_cfg = self.cfg_modifier(baseline_cfg)

        # Baseline model
        baseline_model = MERMentalHealthModel(baseline_cfg).to(device, dtype=dtype).eval()
        baseline_metrics = evaluate_ablation_model(
            baseline_model, baseline_cfg, device, dtype, n_batches=n_batches,
        )

        # Ablated model
        try:
            ablated_model = MERMentalHealthModel(ablated_cfg).to(device, dtype=dtype).eval()
            ablated_metrics = evaluate_ablation_model(
                ablated_model, ablated_cfg, device, dtype, n_batches=n_batches,
            )
        except Exception as e:
            return {
                "name": self.name,
                "status": "failed",
                "error": str(e),
                "baseline_metrics": baseline_metrics,
                "ablated_metrics": None,
            }

        return {
            "name": self.name,
            "description": self.description,
            "hypothesis": self.hypothesis,
            "expected_movement": self.expected_movement,
            "owning_stage": self.owning_stage,
            "status": "completed",
            "baseline_metrics": baseline_metrics,
            "ablated_metrics": ablated_metrics,
            "diff": {
                k: ablated_metrics.get(k, 0) - baseline_metrics.get(k, 0)
                for k in ["mae_clean", "mae_dropout"]
            },
        }


# ---------------------------------------------------------------------------
# Build ablation registry
# ---------------------------------------------------------------------------

def _build_ablations() -> List[Ablation]:
    """Construct all 10 ablations from the architect's spec."""

    baseline_cfg = ModelConfig(
        pretrained=False,
        proj_dim=96,
        d_session_model=96,
        audio_feat_dim=96,
        video_feat_dim=96,
        text_feat_dim=96,
        n_exchange_layers=2,
        n_session_layers=2,
        exchange_hidden_dim=48,
        concept_hidden_dim=48,
        quality_estimator_hidden=24,
        head_hidden_dim=24,
        max_utterances=128,
        n_session_heads=6,
        n_exchange_heads=4,
        n_audio_quality_classes=3,
        n_video_quality_classes=3,
        n_text_quality_classes=3,
        n_therapeutic_phases=4,
        phase_embed_dim=16,
        lora_r=8,
        random_dropout_prob=0.15,
        clinical_correlated_dropout_prob=0.1,
        audio_encoder_name="wavlm-base-plus",
        streaming_mode=False,
        n_clinical_concepts=8,
        n_phq8_items=8,
    )

    ablations = [
        Ablation(
            name="drop_cmgef",
            description="Drop CM-GEF → simple late fusion: set n_exchange_layers=0, use concat+MLP",
            cfg_modifier=lambda cfg: replace(cfg, n_exchange_layers=0),
            hypothesis="Gated exchange outperforms static late fusion under dropout",
            expected_movement="MAE ↑ by >2 points under 40% dropout",
            owning_stage="ml-architect",
        ),
        Ablation(
            name="scalar_gates",
            description="Scalar gates instead of per-dim gates: set exchange_hidden_dim=0",
            cfg_modifier=lambda cfg: replace(cfg, exchange_hidden_dim=0),
            hypothesis="Per-dimension gates outperform scalar modality gates",
            expected_movement="MAE ↑ by ≤0.5 points",
            owning_stage="ml-architect",
        ),
        Ablation(
            name="drop_quality_estimators",
            description="Drop quality estimators: set n_audio_quality_classes=0",
            cfg_modifier=lambda cfg: replace(cfg, n_audio_quality_classes=0,
                                             n_video_quality_classes=0, n_text_quality_classes=0),
            hypothesis="Supervised quality estimation improves over self-attended gating",
            expected_movement="MAE ↑ under noise/occlusion",
            owning_stage="ml-architect",
        ),
        Ablation(
            name="drop_phase_embeddings",
            description="Drop phase embeddings: set n_therapeutic_phases=1",
            cfg_modifier=lambda cfg: replace(cfg, n_therapeutic_phases=1, phase_embed_dim=0),
            hypothesis="Therapeutic phase modeling improves sustained affect detection",
            expected_movement="Sustained affect F1 ↓ by ≥10%",
            owning_stage="ml-architect",
        ),
        Ablation(
            name="drop_session_transformer",
            description="Drop session transformer → mean pool: set n_session_layers=0",
            cfg_modifier=lambda cfg: replace(cfg, n_session_layers=0),
            hypothesis="Session-level temporal modeling improves over utterance averaging",
            expected_movement="Session-level F1 (sustained affect) ↓ by ≥10%",
            owning_stage="ml-architect",
        ),
        Ablation(
            name="lora_vs_full_finetune",
            description="LoRA vs full fine-tune: set lora_r=0 (= full fine-tune)",
            cfg_modifier=lambda cfg: replace(cfg, lora_r=0),
            hypothesis="LoRA adaptation matches or exceeds full fine-tune on small clinical data",
            expected_movement="Both MAE same; LoRA trains faster",
            owning_stage="ml-research",
        ),
        Ablation(
            name="drop_concept_bottleneck",
            description="Drop concept bottleneck: set n_clinical_concepts=0 (direct PHQ-8 prediction)",
            cfg_modifier=lambda cfg: replace(cfg, n_clinical_concepts=0, n_phq8_items=8),
            hypothesis="Concept bottleneck maintains accuracy vs. direct prediction",
            expected_movement="MAE ↑ by ≤0.5 point (acceptable for interpretability)",
            owning_stage="ml-architect",
        ),
        Ablation(
            name="drop_dropout_simulation",
            description="Drop modality dropout simulation: set random_dropout_prob=0.0",
            cfg_modifier=lambda cfg: replace(cfg, random_dropout_prob=0.0,
                                             clinical_correlated_dropout_prob=0.0),
            hypothesis="Dropout simulation during training improves test-time robustness",
            expected_movement="MAE ↑ on dropout conditions; minimal effect on clean data",
            owning_stage="ml-coder",
        ),
        Ablation(
            name="small_audio_encoder",
            description="WavLM Small instead of Base+: audio_encoder_name='wavlm-small'",
            cfg_modifier=lambda cfg: replace(cfg, audio_encoder_name="wavlm-small",
                                             audio_feat_dim=cfg.audio_feat_dim // 2),
            hypothesis="Smaller encoder maintains target accuracy",
            expected_movement="MAE ↑ ≤ 0.5; latency ↓ 40%",
            owning_stage="ml-architect",
        ),
        Ablation(
            name="streaming_mode",
            description="Bidirectional vs causal session: set streaming_mode=True",
            cfg_modifier=lambda cfg: replace(cfg, streaming_mode=True),
            hypothesis="Bidirectional outperforms causal for offline analysis",
            expected_movement="Sustained affect F1 ↓; enables streaming use-case",
            owning_stage="ml-architect",
        ),
    ]

    return ablations


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_ablations(device: torch.device, dtype: torch.dtype,
                  ablation_names: Optional[List[str]] = None,
                  n_batches: int = 4, verbose: bool = True) -> List[Dict]:
    """Run specified ablations (or all if none specified)."""
    ablations = _build_ablations()
    baseline_cfg = ModelConfig(
        pretrained=False,
        proj_dim=96,
        d_session_model=96,
        audio_feat_dim=96,
        video_feat_dim=96,
        text_feat_dim=96,
        n_exchange_layers=2,
        n_session_layers=2,
        exchange_hidden_dim=48,
        concept_hidden_dim=48,
        quality_estimator_hidden=24,
        head_hidden_dim=24,
        max_utterances=128,
        n_session_heads=6,
        n_exchange_heads=4,
        n_audio_quality_classes=3,
        n_video_quality_classes=3,
        n_text_quality_classes=3,
        n_therapeutic_phases=4,
        phase_embed_dim=16,
        lora_r=8,
        random_dropout_prob=0.15,
        clinical_correlated_dropout_prob=0.1,
        audio_encoder_name="wavlm-base-plus",
        streaming_mode=False,
        n_clinical_concepts=8,
        n_phq8_items=8,
    )

    results = []

    # Compute baseline once
    if verbose:
        print(f"Computing baseline...")
    baseline_model = MERMentalHealthModel(baseline_cfg).to(device, dtype=dtype).eval()
    baseline_metrics = evaluate_ablation_model(
        baseline_model, baseline_cfg, device, dtype, n_batches=n_batches,
    )
    if verbose:
        print(f"  Baseline MAE (clean): {baseline_metrics['mae_clean']:.4f}")
        print(f"  Baseline MAE (dropout): {baseline_metrics['mae_dropout']:.4f}")

    for ablation in ablations:
        if ablation_names and ablation.name not in ablation_names:
            continue

        if verbose:
            print(f"\n{'─'*60}")
            print(f"  Ablation: {ablation.name}")
            print(f"  {ablation.description}")
            print(f"  Hypothesis: {ablation.hypothesis}")

        try:
            ablated_cfg = ablation.cfg_modifier(baseline_cfg)
            ablated_model = MERMentalHealthModel(ablated_cfg).to(device, dtype=dtype).eval()
            ablated_metrics = evaluate_ablation_model(
                ablated_model, ablated_cfg, device, dtype, n_batches=n_batches,
                eval_dropout=0.2,
            )

            result = {
                "name": ablation.name,
                "description": ablation.description,
                "hypothesis": ablation.hypothesis,
                "expected_movement": ablation.expected_movement,
                "owning_stage": ablation.owning_stage,
                "status": "completed",
                "baseline_mae_clean": baseline_metrics["mae_clean"],
                "baseline_mae_dropout": baseline_metrics["mae_dropout"],
                "ablated_mae_clean": ablated_metrics["mae_clean"],
                "ablated_mae_dropout": ablated_metrics["mae_dropout"],
                "diff_clean": ablated_metrics["mae_clean"] - baseline_metrics["mae_clean"],
                "diff_dropout": ablated_metrics["mae_dropout"] - baseline_metrics["mae_dropout"],
                "baseline_params": baseline_metrics["total_params"],
                "ablated_params": ablated_metrics["total_params"],
            }

            if verbose:
                print(f"  Baseline -> Ablated:")
                print(f"    MAE clean:   {baseline_metrics['mae_clean']:.4f} -> {ablated_metrics['mae_clean']:.4f}  (Δ={result['diff_clean']:+.4f})")
                print(f"    MAE dropout: {baseline_metrics['mae_dropout']:.4f} -> {ablated_metrics['mae_dropout']:.4f}  (Δ={result['diff_dropout']:+.4f})")
                print(f"    Params: {baseline_metrics['total_params']:,} -> {ablated_metrics['total_params']:,}")

            results.append(result)

        except Exception as e:
            import traceback
            error_msg = f"{type(e).__name__}: {e}"
            if verbose:
                print(f"  [FAIL] {error_msg}")
                traceback.print_exc()

            results.append({
                "name": ablation.name,
                "description": ablation.description,
                "hypothesis": ablation.hypothesis,
                "status": "failed",
                "error": error_msg,
                "baseline_mae_clean": baseline_metrics["mae_clean"],
                "baseline_mae_dropout": baseline_metrics["mae_dropout"],
            })

    return results


def main():
    parser = argparse.ArgumentParser(description="MER Ablation Runner")
    parser.add_argument("--device", default="cpu", help="Device (cpu, cuda)")
    parser.add_argument("--fast", action="store_true", help="Fewer batches for CI")
    parser.add_argument("--ablation", nargs="+", default=None,
                        help="Specific ablations to run (default: all)")
    parser.add_argument("--output", default=None, help="Save results to JSON file")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    n_batches = 2 if args.fast else 4

    print("=" * 60)
    print("  MER Mental Health — Ablation Runner")
    print("=" * 60)
    print(f"  Device: {device}  |  dtype: {dtype}")
    print(f"  Batches per evaluation: {n_batches}")

    results = run_ablations(device, dtype, args.ablation, n_batches=n_batches)

    # Summary
    print(f"\n{'='*60}")
    print(f"  Ablation Summary")
    print(f"{'='*60}")
    print(f"  {'Name':30s} {'Status':12s} {'Δ Clean':>10s} {'Δ Dropout':>10s}")
    print(f"  {'─'*30} {'─'*12} {'─'*10} {'─'*10}")

    for r in results:
        name = r["name"]
        status = r["status"]
        if status == "completed":
            d_clean = f"{r['diff_clean']:+.4f}"
            d_drop = f"{r['diff_dropout']:+.4f}"
        else:
            d_clean = "FAIL"
            d_drop = ""
        print(f"  {name:30s} {status:12s} {d_clean:>10s} {d_drop:>10s}")

    if args.output:
        def sanitize(obj):
            if isinstance(obj, dict):
                return {k: sanitize(v) for k, v in obj.items()}
            if isinstance(obj, float):
                return round(obj, 6)
            if isinstance(obj, (torch.Tensor,)):
                return obj.item() if obj.numel() == 1 else obj.tolist()
            if isinstance(obj, (int, str, bool)):
                return obj
            if obj is None:
                return None
            return str(obj)

        clean_results = sanitize(results)
        with open(args.output, "w") as f:
            json.dump(clean_results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
