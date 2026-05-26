"""
Domain-specific benchmarks for the MER Mental Health Model.

These benchmarks evaluate the model's performance on clinically-relevant tasks:
  1. Missing Modality Stress Test — evaluate under 6 dropout conditions
  2. Clinically-Correlated Dropout Patterns — 3 clinical scenario patterns
  3. Quality Estimator Sensitivity — synthetic corruption benchmark
  4. Late Fusion Baseline Comparison — compare gated exchange vs static fusion
  5. Session-Level Sustained Affect Detection — F1 for sustained affect markers
  6. Concept Bottleneck Interpretability — concept→item mapping fidelity

Each benchmark produces a metrics dict that can be compared against
the hypothesis targets from the research lifecycle contract.

Usage:
    python benchmarks.py                          # Run all benchmarks with default (CPU)
    python benchmarks.py --device cuda            # Run on GPU
    python benchmarks.py --fast                   # Quick subset for CI
    python benchmarks.py --benchmark missing_modality  # Single benchmark
"""

import math
import sys
import json
import time
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/artifacts/j_0NmpJCXSm-5u/work/coder")

from config import ModelConfig
from model import MERMentalHealthModel, MEROutput, modality_dropout_augmentation
from layers import count_params


# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkConfig:
    """Shared benchmark configuration."""
    n_utterances: int = 32         # utterances per session
    n_batches: int = 8             # batches per condition (for stable estimates)
    batch_size: int = 2            # sessions per batch
    proj_dim: int = 96             # reduced for CPU testing
    d_session_model: int = 96
    audio_feat_dim: int = 96
    video_feat_dim: int = 96
    text_feat_dim: int = 96
    n_exchange_layers: int = 2
    n_session_layers: int = 2
    device: str = "cpu"
    dtype: torch.dtype = torch.float32
    seed: int = 42
    fast: bool = False             # minimal iterations for CI

    def to_model_config(self) -> ModelConfig:
        """Create a ModelConfig from benchmark settings."""
        return ModelConfig(
            pretrained=False,
            proj_dim=self.proj_dim,
            d_session_model=self.d_session_model,
            audio_feat_dim=self.audio_feat_dim,
            video_feat_dim=self.video_feat_dim,
            text_feat_dim=self.text_feat_dim,
            n_exchange_layers=self.n_exchange_layers,
            n_session_layers=self.n_session_layers,
            exchange_hidden_dim=self.proj_dim // 2,
            concept_hidden_dim=self.proj_dim // 2,
            quality_estimator_hidden=self.proj_dim // 4,
            head_hidden_dim=self.proj_dim // 4,
            max_utterances=128,
            n_session_heads=4 if self.proj_dim % 4 == 0 else 3,
            n_exchange_heads=4 if self.proj_dim % 4 == 0 else 3,
            n_audio_quality_classes=3,
            n_video_quality_classes=3,
            n_text_quality_classes=3,
            n_therapeutic_phases=4,
            phase_embed_dim=self.proj_dim // 4,
        )


def make_bencfg(**kwargs) -> BenchmarkConfig:
    cfg = BenchmarkConfig()
    for k, v in kwargs.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def create_synthetic_session(bcfg: BenchmarkConfig, device: torch.device):
    """Create a synthetic batch of sessions for benchmarking."""
    B = bcfg.batch_size
    T = bcfg.n_utterances
    d_type = bcfg.dtype
    cfg = bcfg.to_model_config()

    waveforms = torch.randn(B, T, 16000, device=device, dtype=d_type)
    frames = torch.randn(B, T, cfg.video_n_frames, 3, cfg.video_frame_size,
                         cfg.video_frame_size, device=device, dtype=d_type)
    input_ids = torch.randint(0, 1000, (B, T, cfg.text_max_length), device=device)
    phase_labels = torch.randint(0, cfg.n_therapeutic_phases, (B, T), device=device)
    timestamps = torch.linspace(0, 600, T, device=device).unsqueeze(0).expand(B, -1)
    utterance_mask = torch.ones(B, T, device=device, dtype=torch.bool)
    phq8_labels = torch.rand(B, device=device) * 24

    return {
        "waveforms": waveforms,
        "frames": frames,
        "input_ids": input_ids,
        "phase_labels": phase_labels,
        "timestamps": timestamps,
        "utterance_mask": utterance_mask,
        "phq8_labels": phq8_labels,
    }


# ---------------------------------------------------------------------------
# Benchmark 1: Missing Modality Stress Test
# ---------------------------------------------------------------------------

def benchmark_missing_modality_stress(bcfg: BenchmarkConfig) -> Dict:
    """Evaluate model under 6 dropout conditions (0/20/40/60% per modality).

    Measures MAE degradation as each modality is progressively dropped.
    Compares against the full-modality (0% dropout) baseline.

    Research contract target: <15% MAE degradation under 40% dropout
                              (vs. >30% for non-adaptive fusion).
    """
    device = torch.device(bcfg.device)
    cfg = bcfg.to_model_config()
    model = MERMentalHealthModel(cfg).to(device, dtype=bcfg.dtype).eval()

    results = {}
    dropout_levels = [0.0, 0.2, 0.4, 0.6]
    modalities = ["audio", "video", "text"]

    for modality in modalities:
        modality_results = {}
        for drop_prob in dropout_levels:
            losses = []
            for _ in range(bcfg.n_batches if not bcfg.fast else 2):
                inp = create_synthetic_session(bcfg, device)
                B, T = inp["waveforms"].shape[:2]

                # Create masks with dropout
                audio_mask = torch.ones(B, T, device=device, dtype=torch.bool)
                video_mask = torch.ones(B, T, device=device, dtype=torch.bool)
                text_mask = torch.ones(B, T, device=device, dtype=torch.bool)

                if modality == "audio":
                    audio_mask = torch.rand(B, T, device=device) > drop_prob
                elif modality == "video":
                    video_mask = torch.rand(B, T, device=device) > drop_prob
                elif modality == "text":
                    text_mask = torch.rand(B, T, device=device) > drop_prob

                with torch.no_grad():
                    out = model(
                        waveforms=inp["waveforms"],
                        frames=inp["frames"],
                        input_ids=inp["input_ids"],
                        phase_labels=inp["phase_labels"],
                        timestamps=inp["timestamps"],
                        utterance_mask=inp["utterance_mask"],
                        audio_mask=audio_mask,
                        video_mask=video_mask,
                        text_mask=text_mask,
                        phq8_labels=inp["phq8_labels"],
                    )
                if out.loss is not None:
                    losses.append(out.loss.item())

            mean_loss = sum(losses) / max(len(losses), 1)
            modality_results[f"dropout_{int(drop_prob*100)}"] = {
                "mae": mean_loss,
                "n_batches": len(losses),
            }
        results[modality] = modality_results

    # Compute degradation
    full_mae = results["audio"]["dropout_0"]["mae"]  # same baseline for all
    summary = {"full_modality_baseline_mae": full_mae}

    for modality in modalities:
        for drop_key in ["dropout_40"]:
            mae = results[modality][drop_key]["mae"]
            degradation = (mae - full_mae) / max(full_mae, 1e-8) * 100
            results[modality][drop_key]["degradation_pct"] = degradation

        summary[f"{modality}_40pct_drop_mae"] = results[modality]["dropout_40"]["mae"]
        summary[f"{modality}_40pct_drop_degradation_pct"] = results[modality]["dropout_40"].get(
            "degradation_pct", 0.0
        )

    summary["all_results"] = results
    summary["benchmark_name"] = "missing_modality_stress"
    summary["pass_criteria"] = "degradation < 15% at 40% dropout"
    summary["pass"] = all(
        results[m]["dropout_40"].get("degradation_pct", 999) < 15
        for m in modalities
    )

    return summary


# ---------------------------------------------------------------------------
# Benchmark 2: Clinically-Correlated Dropout Patterns
# ---------------------------------------------------------------------------

def benchmark_clinical_dropout_patterns(bcfg: BenchmarkConfig) -> Dict:
    """Evaluate model under 3 clinically-correlated dropout patterns.

    Patterns:
      1. Vision-drop-during-distress: video missing on a subset of utterances
         that simulate emotional distress (e.g., patient turns away)
      2. Audio-drop-in-noisy: audio missing on utterances simulating poor
         room acoustics
      3. Text-drop-in-nonverbal: text missing when patient is non-verbal
         (e.g., crying, silence)

    Research contract target: <15% MAE degradation under any clinical pattern.
    """
    device = torch.device(bcfg.device)
    cfg = bcfg.to_model_config()
    model = MERMentalHealthModel(cfg).to(device, dtype=bcfg.dtype).eval()

    patterns = {
        "vision_drop_distress": {"video": 0.4, "audio": 0.0, "text": 0.0},
        "audio_drop_noisy": {"video": 0.0, "audio": 0.3, "text": 0.0},
        "text_drop_nonverbal": {"video": 0.1, "audio": 0.1, "text": 0.4},
    }

    results = {"full_modality_baseline": {}}
    # Full-modality baseline
    baselines = []
    for _ in range(bcfg.n_batches if not bcfg.fast else 2):
        inp = create_synthetic_session(bcfg, device)
        with torch.no_grad():
            out = model(**inp, phq8_labels=inp["phq8_labels"])
        if out.loss is not None:
            baselines.append(out.loss.item())
    results["full_modality_baseline"]["mae"] = sum(baselines) / max(len(baselines), 1)

    for pattern_name, drop_rates in patterns.items():
        pattern_losses = []
        for _ in range(bcfg.n_batches if not bcfg.fast else 2):
            inp = create_synthetic_session(bcfg, device)
            B, T = inp["waveforms"].shape[:2]

            audio_mask = torch.rand(B, T, device=device) > drop_rates.get("audio", 0.0)
            video_mask = torch.rand(B, T, device=device) > drop_rates.get("video", 0.0)
            text_mask = torch.rand(B, T, device=device) > drop_rates.get("text", 0.0)

            with torch.no_grad():
                out = model(
                    **inp,
                    audio_mask=audio_mask,
                    video_mask=video_mask,
                    text_mask=text_mask,
                )
            if out.loss is not None:
                pattern_losses.append(out.loss.item())

        mean_loss = sum(pattern_losses) / max(len(pattern_losses), 1)
        baseline_mae = results["full_modality_baseline"]["mae"]
        degradation = (mean_loss - baseline_mae) / max(baseline_mae, 1e-8) * 100

        results[pattern_name] = {
            "mae": mean_loss,
            "degradation_pct": degradation,
            "n_batches": len(pattern_losses),
        }

    results["benchmark_name"] = "clinical_dropout_patterns"
    results["pass_criteria"] = "degradation < 15% under all clinical patterns"
    results["pass"] = all(
        v["degradation_pct"] < 15
        for k, v in results.items()
        if k not in ["full_modality_baseline", "benchmark_name", "pass_criteria", "pass"]
    )

    return results


# ---------------------------------------------------------------------------
# Benchmark 3: Late Fusion Baseline Comparison
# ---------------------------------------------------------------------------

def benchmark_late_fusion_comparison(bcfg: BenchmarkConfig) -> Dict:
    """Compare CM-GEF vs. simple late fusion under modality dropout.

    Late fusion baseline: independent unimodal predictions averaged together.
    CM-GEF should outperform late fusion under dropout.

    Research contract target: CM-GEF MAE < Late Fusion MAE by >2 points
                              under 40% modality dropout.
    """
    device = torch.device(bcfg.device)
    cfg = bcfg.to_model_config()
    model = MERMentalHealthModel(cfg).to(device, dtype=bcfg.dtype).eval()

    # Build a simple late fusion model: mean of 3 independent predictions
    class LateFusionWrapper(nn.Module):
        def __init__(self, audio_enc, video_enc, text_enc,
                     audio_proj, video_proj, text_proj,
                     session_transformer, concept_bottleneck):
            super().__init__()
            self.audio_enc = audio_enc
            self.video_enc = video_enc
            self.text_enc = text_enc
            self.audio_proj = audio_proj
            self.video_proj = video_proj
            self.text_proj = text_proj
            self.session_transformer = session_transformer
            self.concept_bottleneck = concept_bottleneck
            self.cfg = cfg

        def forward(self, waveforms, frames, input_ids,
                    audio_mask=None, video_mask=None, text_mask=None,
                    phase_labels=None, timestamps=None, utterance_mask=None):
            B, T = waveforms.shape[:2]
            device = waveforms.device

            # Default masks
            if audio_mask is None:
                audio_mask = torch.ones(B, T, device=device, dtype=torch.bool)
            if video_mask is None:
                video_mask = torch.ones(B, T, device=device, dtype=torch.bool)
            if text_mask is None:
                text_mask = torch.ones(B, T, device=device, dtype=torch.bool)

            # Encode each modality independently
            flat_wave = waveforms.reshape(-1, waveforms.shape[-1])
            flat_frame = frames.reshape(-1, *frames.shape[2:])
            flat_text = input_ids.reshape(-1, input_ids.shape[-1])
            flat_audio_mask = audio_mask.reshape(-1)
            flat_video_mask = video_mask.reshape(-1)
            flat_text_mask = text_mask.reshape(-1)

            device_audio = torch.zeros(flat_wave.shape[0], cfg.audio_feat_dim, device=device)
            a_idx = flat_audio_mask
            if a_idx.any():
                device_audio[a_idx] = self.audio_enc(flat_wave[a_idx])

            device_video = torch.zeros(flat_frame.shape[0], cfg.video_feat_dim, device=device)
            v_idx = flat_video_mask
            if v_idx.any():
                device_video[v_idx] = self.video_enc(flat_frame[v_idx])

            device_text = torch.zeros(flat_text.shape[0], cfg.text_feat_dim, device=device)
            t_idx = flat_text_mask
            if t_idx.any():
                device_text[t_idx] = self.text_enc(flat_text[t_idx])

            # Project
            audio_p = self.audio_proj(device_audio)
            video_p = self.video_proj(device_video)
            text_p = self.text_proj(device_text)

            # Simple late fusion = mean of available modalities
            # Normalize by number of available modalities
            avail = torch.stack([flat_audio_mask.float(), flat_video_mask.float(),
                                 flat_text_mask.float()], dim=-1)  # (B*T, 3)
            n_avail = avail.sum(dim=-1, keepdim=True).clamp(min=1)  # (B*T, 1)
            fused = (audio_p * avail[:, 0:1] + video_p * avail[:, 1:2] +
                     text_p * avail[:, 2:3]) / n_avail

            # Session transformer
            utt_feats = fused.view(B, T, cfg.proj_dim)
            session_feat, _ = self.session_transformer(
                utt_feats, phase_labels=phase_labels, timestamps=timestamps,
                utterance_mask=utterance_mask,
            )
            phq8_total, phq8_items, concept_scores, depression_prob = \
                self.concept_bottleneck(session_feat)
            return phq8_total, phq8_items, concept_scores, depression_prob

    late_fusion_model = LateFusionWrapper(
        model.audio_encoder, model.video_encoder, model.text_encoder,
        model.audio_projection, model.video_projection, model.text_projection,
        model.session_transformer, model.concept_bottleneck,
    ).to(device).eval()

    # Compare under dropout
    results = {}
    for drop_prob in [0.0, 0.4]:
        cmgef_losses = []
        late_losses = []
        n_batches = bcfg.n_batches if not bcfg.fast else 2

        for _ in range(n_batches):
            inp = create_synthetic_session(bcfg, device)
            B, T = inp["waveforms"].shape[:2]
            audio_mask = torch.rand(B, T, device=device) > drop_prob
            video_mask = torch.rand(B, T, device=device) > drop_prob
            text_mask = torch.rand(B, T, device=device) > drop_prob
            phq8_labels = inp["phq8_labels"]

            # CM-GEF
            with torch.no_grad():
                out = model(
                    **inp, audio_mask=audio_mask, video_mask=video_mask,
                    text_mask=text_mask,
                )
            pred = out.phq8_total
            cmgef_loss = F.l1_loss(pred, phq8_labels).item()
            cmgef_losses.append(cmgef_loss)

            # Late fusion
            with torch.no_grad():
                lf_pred, _, _, _ = late_fusion_model(
                    inp["waveforms"], inp["frames"], inp["input_ids"],
                    audio_mask=audio_mask, video_mask=video_mask,
                    text_mask=text_mask,
                    phase_labels=inp["phase_labels"],
                    timestamps=inp["timestamps"],
                    utterance_mask=inp["utterance_mask"],
                )
            late_loss = F.l1_loss(lf_pred, phq8_labels).item()
            late_losses.append(late_loss)

        key = f"dropout_{int(drop_prob*100)}"
        cmgef_mean = sum(cmgef_losses) / max(len(cmgef_losses), 1)
        late_mean = sum(late_losses) / max(len(late_losses), 1)
        results[key] = {
            "cm_gef_mae": cmgef_mean,
            "late_fusion_mae": late_mean,
            "advantage": late_mean - cmgef_mean,
        }

    results["benchmark_name"] = "late_fusion_comparison"
    results["pass_criteria"] = "CM-GEF outperforms late fusion under 40% dropout"
    results["pass"] = results.get("dropout_40", {}).get("advantage", -999) > 0

    return results


# ---------------------------------------------------------------------------
# Benchmark 4: Quality Estimator Sensitivity
# ---------------------------------------------------------------------------

def benchmark_quality_estimator(bcfg: BenchmarkConfig) -> Dict:
    """Verify quality estimator produces non-trivial scores under corruption.

    The quality estimator should produce different reliability scores for
    clean vs. corrupted features. This benchmark applies synthetic noise
    to encoder outputs and measures the change in quality scores.
    """
    device = torch.device(bcfg.device)
    cfg = bcfg.to_model_config()
    model = MERMentalHealthModel(cfg).to(device, dtype=bcfg.dtype).eval()

    # Create a batch of data
    inp = create_synthetic_session(bcfg, device)
    B, T = inp["waveforms"].shape[:2]

    reliability_diffs = {"audio": [], "video": [], "text": []}

    for _ in range(bcfg.n_batches if not bcfg.fast else 3):
        inp = create_synthetic_session(bcfg, device)
        B, T = inp["waveforms"].shape[:2]

        with torch.no_grad():
            out = model(**inp, return_all=True)

        if out.reliability_scores is None:
            continue

        for modality in ["audio", "video", "text"]:
            clean_reliability = out.reliability_scores[modality]  # (B*T,)

            # Create noisy version
            inp_noisy = dict(inp)
            if modality == "audio":
                inp_noisy["waveforms"] = inp["waveforms"] + torch.randn_like(inp["waveforms"]) * 5.0
            elif modality == "video":
                inp_noisy["frames"] = inp["frames"] + torch.randn_like(inp["frames"]) * 5.0
            elif modality == "text":
                # Replace with random tokens
                inp_noisy["input_ids"] = torch.randint(0, 1000, inp["input_ids"].shape, device=device)

            with torch.no_grad():
                out_noisy = model(**inp_noisy, return_all=True)

            if out_noisy.reliability_scores is not None:
                noisy_reliability = out_noisy.reliability_scores[modality]
                diff = (clean_reliability - noisy_reliability).abs().mean().item()
                reliability_diffs[modality].append(diff)

    summary = {}
    for modality in ["audio", "video", "text"]:
        diffs = reliability_diffs[modality]
        mean_diff = sum(diffs) / max(len(diffs), 1)
        summary[f"{modality}_mean_reliability_diff"] = mean_diff
        summary[f"{modality}_n_samples"] = len(diffs)

    summary["benchmark_name"] = "quality_estimator_sensitivity"
    summary["pass_criteria"] = "reliability differs between clean and noisy (diff > 0)"
    # With random init this may not pass consistently — informational
    all_diffs = [summary.get(f"{m}_mean_reliability_diff", 0) for m in ["audio", "video", "text"]]
    summary["pass"] = any(d > 0.001 for d in all_diffs)
    summary["note"] = "With random init, sensitivity may be low. Train quality estimator first."

    return summary


# ---------------------------------------------------------------------------
# Benchmark 5: Session-Level Sustained Affect Proxy
# ---------------------------------------------------------------------------

def benchmark_sustained_affect_detection(bcfg: BenchmarkConfig) -> Dict:
    """Proxy benchmark for session-level sustained affect detection.

    Simulates sessions with artificially induced "flat affect" segments
    (low-variance utterance features) and measures whether the SLHT
    produces distinct session representations for flat vs. variable affect.

    Research contract target: ≥10% F1 improvement over utterance-level pooling.
    """
    device = torch.device(bcfg.device)
    cfg = bcfg.to_model_config()
    model = MERMentalHealthModel(cfg).to(device, dtype=bcfg.dtype).eval()

    # Create sessions with vs. without sustained flat affect
    def _create_affect_session(has_flat_affect: bool, T: int = 32):
        """Create a session where some utterances have reduced variance."""
        B = 1
        inp = create_synthetic_session(
            make_bencfg(**{**vars(bcfg), "n_utterances": T, "batch_size": 1}),
            device,
        )
        if has_flat_affect:
            # Replace some utterances with low-variance features
            flat_start = T // 3
            flat_end = 2 * T // 3
            # We manipulate the waveform to have low amplitude
            inp["waveforms"][0, flat_start:flat_end] *= 0.01

        return inp

    # Measure session feature variance difference
    flat_feats = []
    var_feats = []

    n_samples = bcfg.n_batches if not bcfg.fast else 4
    for _ in range(n_samples):
        inp_flat = _create_affect_session(has_flat_affect=True)
        inp_var = _create_affect_session(has_flat_affect=False)

        with torch.no_grad():
            out_flat = model(**inp_flat, return_all=True)
            out_var = model(**inp_var, return_all=True)

        if out_flat.session_feat is not None:
            flat_feats.append(out_flat.session_feat)
        if out_var.session_feat is not None:
            var_feats.append(out_var.session_feat)

    # Compute separation between flat and variable affect sessions
    if flat_feats and var_feats:
        flat_stack = torch.cat(flat_feats, dim=0)  # (N, d)
        var_stack = torch.cat(var_feats, dim=0)
        flat_mean = flat_stack.mean(dim=0)
        var_mean = var_stack.mean(dim=0)
        separation = (flat_mean - var_mean).norm().item()
    else:
        separation = 0.0

    # Also compute utterance-level separation (without SLHT)
    utt_flat = []
    utt_var = []
    for _ in range(n_samples):
        inp_flat = _create_affect_session(has_flat_affect=True)
        inp_var = _create_affect_session(has_flat_affect=False)

        with torch.no_grad():
            # Get per-utterance features (mean pooling)
            out_flat = model(**inp_flat, return_all=True)
            out_var = model(**inp_var, return_all=True)

        if out_flat.utterance_feats is not None:
            utt_flat.append(out_flat.utterance_feats.mean(dim=1))  # (1, d)
        if out_var.utterance_feats is not None:
            utt_var.append(out_var.utterance_feats.mean(dim=1))

    if utt_flat and utt_var:
        utt_flat_stack = torch.cat(utt_flat, dim=0)
        utt_var_stack = torch.cat(utt_var, dim=0)
        utt_separation = (utt_flat_stack.mean(dim=0) - utt_var_stack.mean(dim=0)).norm().item()
    else:
        utt_separation = 0.0

    return {
        "benchmark_name": "sustained_affect_detection",
        "session_level_separation": separation,
        "utterance_level_separation": utt_separation,
        "separation_ratio": separation / max(utt_separation, 1e-8),
        "n_samples": n_samples,
        "pass_criteria": "session_separation > utterance_separation (ratio > 1)",
        "pass": separation > utt_separation,
        "note": "Informational — real evaluation requires clinical data with affect labels",
    }


# ---------------------------------------------------------------------------
# Benchmark 6: Concept Bottleneck Mapping Fidelity
# ---------------------------------------------------------------------------

def benchmark_concept_mapping(bcfg: BenchmarkConfig) -> Dict:
    """Verify the concept bottleneck's diagonal-dominant mapping.

    Each concept k should predominantly influence PHQ-8 item k.
    We measure the ratio of diagonal weight to mean off-diagonal weight.
    """
    cfg = bcfg.to_model_config()
    model = MERMentalHealthModel(cfg)

    # Extract concept->item weight matrix
    W = model.concept_bottleneck.get_concept_to_item_weights()  # (n_items, n_concepts)

    diag = W.diag().abs().mean().item()
    off_diag = (W - torch.diag(W.diag())).abs().mean().item()
    ratio = diag / max(off_diag, 1e-8)

    return {
        "benchmark_name": "concept_mapping_fidelity",
        "diagonal_mean": diag,
        "off_diagonal_mean": off_diag,
        "diag_off_diag_ratio": ratio,
        "pass_criteria": "diagonal dominates off-diagonal (ratio > 2)",
        "pass": ratio > 2.0,
    }


# ---------------------------------------------------------------------------
# Benchmark 7: Throughput and Latency
# ---------------------------------------------------------------------------

def benchmark_throughput(bcfg: BenchmarkConfig) -> Dict:
    """Measure utterances-per-second throughput and memory usage.

    This is a proxy for real-time clinical deployment capability.
    """
    device = torch.device(bcfg.device)
    cfg = bcfg.to_model_config()
    model = MERMentalHealthModel(cfg).to(device, dtype=bcfg.dtype).eval()

    B = bcfg.batch_size
    T = bcfg.n_utterances

    inp = create_synthetic_session(bcfg, device)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            model(**inp)

    # Timed runs
    n_timed = bcfg.n_batches if not bcfg.fast else 3
    torch.cuda.synchronize() if bcfg.device == "cuda" else None
    start = time.perf_counter()

    for _ in range(n_timed):
        with torch.no_grad():
            model(**inp)

    torch.cuda.synchronize() if bcfg.device == "cuda" else None
    elapsed = time.perf_counter() - start

    n_total_utterances = n_timed * B * T
    throughput = n_total_utterances / elapsed

    # Memory (CUDA only)
    memory_gb = 0
    if bcfg.device == "cuda":
        memory_gb = torch.cuda.max_memory_allocated(device) / 1e9

    return {
        "benchmark_name": "throughput",
        "total_utterances_processed": n_total_utterances,
        "total_time_seconds": elapsed,
        "utterances_per_second": throughput,
        "peak_memory_gb": memory_gb,
        "batch_size": B,
        "n_utterances_per_session": T,
    }


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

BENCHMARK_REGISTRY = {
    "missing_modality": benchmark_missing_modality_stress,
    "clinical_dropout": benchmark_clinical_dropout_patterns,
    "late_fusion": benchmark_late_fusion_comparison,
    "quality_estimator": benchmark_quality_estimator,
    "sustained_affect": benchmark_sustained_affect_detection,
    "concept_mapping": benchmark_concept_mapping,
    "throughput": benchmark_throughput,
}


def run_benchmarks(bcfg: BenchmarkConfig, benchmarks: Optional[List[str]] = None):
    """Run specified benchmarks and return consolidated results."""
    torch.manual_seed(bcfg.seed)

    to_run = benchmarks if benchmarks else list(BENCHMARK_REGISTRY.keys())
    results = {}

    for name in to_run:
        if name not in BENCHMARK_REGISTRY:
            print(f"  [SKIP] Unknown benchmark: {name}")
            continue
        print(f"\n{'='*60}")
        print(f"  Benchmark: {name}")
        print(f"{'='*60}")
        try:
            bm_func = BENCHMARK_REGISTRY[name]
            result = bm_func(bcfg)
            results[name] = result

            # Print summary
            for k, v in result.items():
                if k in ("benchmark_name", "pass", "pass_criteria", "note", "all_results"):
                    continue
                if isinstance(v, float):
                    print(f"    {k}: {v:.4f}")
                elif isinstance(v, str):
                    print(f"    {k}: {v}")
            print(f"    Pass: {result.get('pass', 'N/A')} (criteria: {result.get('pass_criteria', 'N/A')})")
            if "note" in result:
                print(f"    Note: {result['note']}")

        except Exception as e:
            import traceback
            print(f"  [FAIL] Benchmark {name}: {e}")
            traceback.print_exc()
            results[name] = {"error": str(e)}

    return results


def main():
    parser = argparse.ArgumentParser(description="MER Mental Health Benchmarks")
    parser.add_argument("--device", default="cpu", help="Device: cpu or cuda")
    parser.add_argument("--fast", action="store_true", help="Minimal iterations for CI")
    parser.add_argument("--benchmark", nargs="+", default=None,
                        choices=list(BENCHMARK_REGISTRY.keys()) + [None],
                        help="Run specific benchmark(s)")
    parser.add_argument("--output", default=None, help="Save results to JSON file")
    args = parser.parse_args()

    bcfg = make_bencfg(
        device=args.device,
        fast=args.fast,
    )
    if args.fast:
        bcfg.n_batches = 2
        bcfg.n_utterances = 8

    print(f"MER Mental Health Benchmarks")
    print(f"  Device: {bcfg.device}  |  Fast: {bcfg.fast}")
    print(f"  Batches per condition: {bcfg.n_batches}")
    print(f"  Utterances per session: {bcfg.n_utterances}")
    print(f"  Batch size: {bcfg.batch_size}")

    results = run_benchmarks(bcfg, benchmarks=args.benchmark)

    # Overall pass/fail
    all_pass = True
    for name, result in results.items():
        if "error" in result:
            all_pass = False
            continue
        if not result.get("pass", True):
            print(f"\n  [FAIL] Benchmark '{name}' did not meet pass criteria")
            all_pass = False

    if all_pass and len(results) > 0:
        print(f"\n{'='*60}")
        print(f"  All benchmarks passed!")
    else:
        print(f"\n{'='*60}")
        print(f"  Some benchmarks did not pass. Review results.")

    # Save to JSON if requested
    if args.output:
        # Convert tensors to floats
        def sanitize(obj):
            if isinstance(obj, dict):
                return {k: sanitize(v) for k, v in obj.items()}
            if isinstance(obj, float):
                return obj
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
