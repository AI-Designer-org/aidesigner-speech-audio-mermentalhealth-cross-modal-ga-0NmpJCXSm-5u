"""
Profiling script for the MER Mental Health Model.

Measures:
  - Inference latency (ms per utterance, ms per session)
  - Peak GPU memory (GB)
  - Estimated FLOPs
  - Per-component breakdown (encoders, fusion, session, concept bottleneck)

Usage:
    python profiling.py                              # CPU profiling
    python profiling.py --device cuda                # GPU profiling (with torch.cuda)
    python profiling.py --device cuda --mode train   # Train mode (fwd+bwd)
    python profiling.py --profile                    # Use torch.profiler
    python profiling.py --output profile_results.json
"""

import sys
import json
import math
import time
import argparse
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/artifacts/j_0NmpJCXSm-5u/work/coder")

from config import ModelConfig
from model import MERMentalHealthModel
from layers import count_params


def profile_inference(model: nn.Module, cfg: ModelConfig,
                      device: torch.device, dtype: torch.dtype,
                      n_warmup: int = 5, n_timed: int = 20,
                      n_utterances: int = 32, batch_size: int = 4) -> Dict:
    """Profile inference latency and throughput.

    Returns:
        dict with latency, throughput, and memory metrics
    """
    B = batch_size
    T = n_utterances

    # Create synthetic session
    waveforms = torch.randn(B, T, 16000, device=device, dtype=dtype)
    frames = torch.randn(B, T, cfg.video_n_frames, 3, cfg.video_frame_size,
                         cfg.video_frame_size, device=device, dtype=dtype)
    input_ids = torch.randint(0, 1000, (B, T, cfg.text_max_length), device=device)
    phase_labels = torch.randint(0, cfg.n_therapeutic_phases, (B, T), device=device)
    timestamps = torch.linspace(0, 600, T, device=device).unsqueeze(0).expand(B, -1)
    utterance_mask = torch.ones(B, T, device=device, dtype=torch.bool)

    model.eval()

    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = model(
                waveforms=waveforms, frames=frames, input_ids=input_ids,
                phase_labels=phase_labels, timestamps=timestamps,
                utterance_mask=utterance_mask,
            )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    # Timed runs
    start = time.perf_counter()
    for _ in range(n_timed):
        with torch.no_grad():
            _ = model(
                waveforms=waveforms, frames=frames, input_ids=input_ids,
                phase_labels=phase_labels, timestamps=timestamps,
                utterance_mask=utterance_mask,
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    total_utterances = n_timed * B * T
    total_sessions = n_timed * B

    # Per-utterance timing
    # Also measure single utterance forward
    wave_utt = torch.randn(B, 16000, device=device, dtype=dtype)
    frame_utt = torch.randn(B, cfg.video_n_frames, 3, cfg.video_frame_size,
                            cfg.video_frame_size, device=device, dtype=dtype)
    input_utt = torch.randint(0, 1000, (B, cfg.text_max_length), device=device)

    warmup = 3
    for _ in range(warmup):
        with torch.no_grad():
            _ = model.forward_utterance(wave_utt, frame_utt, input_utt)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    utt_start = time.perf_counter()
    n_utt_timed = 20
    for _ in range(n_utt_timed):
        with torch.no_grad():
            _ = model.forward_utterance(wave_utt, frame_utt, input_utt)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    utt_elapsed = time.perf_counter() - utt_start

    results = {
        "device": str(device),
        "dtype": str(dtype),
        "batch_size": B,
        "n_utterances_per_session": T,
        "n_timed_runs": n_timed,
        "total_sessions_processed": total_sessions,
        "total_utterances_processed": total_utterances,
        "inference_mode": "forward_no_grad",
        "latency": {
            "total_time_seconds": round(elapsed, 4),
            "ms_per_session": round(elapsed / total_sessions * 1000, 2),
            "ms_per_utterance_session": round(elapsed / total_utterances * 1000, 4),
            "ms_per_utterance_forward": round(utt_elapsed / (n_utt_timed * B) * 1000, 4),
        },
        "throughput": {
            "sessions_per_second": round(total_sessions / elapsed, 2),
            "utterances_per_second": round(total_utterances / elapsed, 2),
        },
    }

    # Memory (CUDA only)
    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated(device) / 1e9
        results["peak_memory_gb"] = round(peak_mem, 3)

        # Memory per component
        def _component_mem(module: nn.Module) -> float:
            return sum(p.numel() * p.element_size() for p in module.parameters()) / 1e6

        results["memory_mb_by_component"] = {
            "audio_encoder": round(_component_mem(model.audio_encoder), 2),
            "video_encoder": round(_component_mem(model.video_encoder), 2),
            "text_encoder": round(_component_mem(model.text_encoder), 2),
            "projection_mlps": round(
                _component_mem(model.audio_projection) +
                _component_mem(model.video_projection) +
                _component_mem(model.text_projection), 2
            ),
            "quality_estimators": round(
                _component_mem(model.audio_quality_estimator) +
                _component_mem(model.video_quality_estimator) +
                _component_mem(model.text_quality_estimator), 2
            ),
            "fusion_cmgef": round(_component_mem(model.fusion), 2),
            "session_transformer": round(_component_mem(model.session_transformer), 2),
            "concept_bottleneck": round(_component_mem(model.concept_bottleneck), 2),
        }

    # Parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    results["parameters"] = {
        "total": total_params,
        "trainable": trainable_params,
        "trainable_pct": round(100 * trainable_params / max(total_params, 1), 1),
    }

    # FLOP estimate (rough: 2× params per forward pass for attention-heavy models)
    # For transformers: ~2 × n_params × n_tokens per forward for self-attention
    # More accurate estimate: count attention multiplications
    # For a single forward pass on a (B, T)-length session:
    #   - Each MHSA: 4 * B * T * d_model^2 (Q,K,V,O projections) + 2 * B * T^2 * n_heads * d_head
    #   - Each FFN: 2 * B * T * d_model * d_ff
    # Rough upper bound: 6 × total_params × T (Kaplan scaling for transformers)
    flop_estimate = 6 * total_params * T  # approximate per-sample
    results["flop_estimate"] = {
        "formula": "6 × total_params × utterances_per_session (Kaplan et al. estimate)",
        "gflops_per_sample": round(flop_estimate / 1e9, 2),
        "gflops_per_batch": round(flop_estimate * B / 1e9, 2),
    }

    return results


def profile_training(model: nn.Module, cfg: ModelConfig,
                     device: torch.device, dtype: torch.dtype,
                     n_warmup: int = 3, n_timed: int = 10,
                     n_utterances: int = 16, batch_size: int = 2) -> Dict:
    """Profile training (forward + backward + optimizer step)."""
    B = batch_size
    T = n_utterances

    model.train()
    model.set_trainable_params(stage=1)

    waveforms = torch.randn(B, T, 16000, device=device, dtype=dtype)
    frames = torch.randn(B, T, cfg.video_n_frames, 3, cfg.video_frame_size,
                         cfg.video_frame_size, device=device, dtype=dtype)
    input_ids = torch.randint(0, 1000, (B, T, cfg.text_max_length), device=device)
    phase_labels = torch.randint(0, cfg.n_therapeutic_phases, (B, T), device=device)
    timestamps = torch.linspace(0, 600, T, device=device).unsqueeze(0).expand(B, -1)
    utterance_mask = torch.ones(B, T, device=device, dtype=torch.bool)
    phq8_labels = torch.rand(B, device=device) * 24

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Warmup
    for _ in range(n_warmup):
        optimizer.zero_grad()
        out = model(
            waveforms=waveforms, frames=frames, input_ids=input_ids,
            phase_labels=phase_labels, timestamps=timestamps,
            utterance_mask=utterance_mask, phq8_labels=phq8_labels,
        )
        out.loss.backward()
        optimizer.step()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    # Timed training steps
    start = time.perf_counter()
    for _ in range(n_timed):
        optimizer.zero_grad()
        out = model(
            waveforms=waveforms, frames=frames, input_ids=input_ids,
            phase_labels=phase_labels, timestamps=timestamps,
            utterance_mask=utterance_mask, phq8_labels=phq8_labels,
        )
        out.loss.backward()
        optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    results = {
        "device": str(device),
        "batch_size": B,
        "n_utterances": T,
        "n_steps": n_timed,
        "training_mode": "forward + backward + optimizer",
        "ms_per_step": round(elapsed / n_timed * 1000, 2),
        "steps_per_second": round(n_timed / elapsed, 2),
    }

    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated(device) / 1e9
        results["peak_memory_gb"] = round(peak_mem, 3)

    # FLOP estimate (training: ~6× params per token)
    total_params = sum(p.numel() for p in model.parameters())
    results["flop_estimate_training"] = {
        "formula": "6 × total_params × utterances (fwd+bwd, Kaplan et al.)",
        "gflops_per_step": round(6 * total_params * T / 1e9, 2),
    }

    return results


def profile_with_torch_profiler(model: nn.Module, cfg: ModelConfig,
                                 device: torch.device, dtype: torch.dtype,
                                 n_steps: int = 5) -> Dict:
    """Run torch.profiler and print/return profile statistics."""
    from torch.profiler import profile, record_function, ProfilerActivity

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    model.eval()
    B, T = 2, 8
    waveforms = torch.randn(B, T, 16000, device=device, dtype=dtype)
    frames = torch.randn(B, T, cfg.video_n_frames, 3, cfg.video_frame_size,
                         cfg.video_frame_size, device=device, dtype=dtype)
    input_ids = torch.randint(0, 1000, (B, T, cfg.text_max_length), device=device)
    phase_labels = torch.randint(0, cfg.n_therapeutic_phases, (B, T), device=device)
    timestamps = torch.linspace(0, 300, T, device=device).unsqueeze(0).expand(B, -1)
    utterance_mask = torch.ones(B, T, device=device, dtype=torch.bool)

    sort_key = "cuda_memory_usage" if device.type == "cuda" else "self_cpu_time_total"

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for _ in range(n_steps):
            with record_function("full_forward"):
                with torch.no_grad():
                    out = model(
                        waveforms=waveforms, frames=frames, input_ids=input_ids,
                        phase_labels=phase_labels, timestamps=timestamps,
                        utterance_mask=utterance_mask,
                    )

    # Get sorted results
    events = prof.key_averages(group_by_input_shape=True)
    table = events.table(sort_by=sort_key, row_limit=20)

    # Top ops by time and memory
    top_time = []
    top_memory = []
    for evt in events:
        name = evt.key
        cpu_time = evt.cpu_time_total
        cuda_time = evt.cuda_time_total if device.type == "cuda" else 0
        mem = evt.cuda_memory_usage if device.type == "cuda" else 0
        top_time.append({"name": name, "cpu_time_us": cpu_time, "cuda_time_us": cuda_time})
        if mem:
            top_memory.append({"name": name, "memory_bytes": mem})

    top_time.sort(key=lambda x: x["cpu_time_us"], reverse=True)
    top_memory.sort(key=lambda x: x.get("memory_bytes", 0), reverse=True)

    return {
        "profiler_table": table,
        "top_ops_by_time": top_time[:10],
        "top_ops_by_memory": top_memory[:10],
        "n_steps": n_steps,
    }


def main():
    parser = argparse.ArgumentParser(description="MER Profiling Script")
    parser.add_argument("--device", default="cpu", help="Device (cpu, cuda)")
    parser.add_argument("--mode", default="inference", choices=["inference", "train"],
                        help="Profiling mode: inference (fwd only) or train (fwd+bwd)")
    parser.add_argument("--profile", action="store_true",
                        help="Use torch.profiler for detailed op breakdown")
    parser.add_argument("--output", default=None, help="Save results to JSON file")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    cfg = ModelConfig(
        pretrained=False,
        proj_dim=256,
        d_session_model=256,
        audio_feat_dim=768,
        video_feat_dim=384,
        text_feat_dim=768,
        n_exchange_layers=2,
        n_session_layers=4,
        exchange_hidden_dim=128,
        concept_hidden_dim=128,
        quality_estimator_hidden=64,
        head_hidden_dim=64,
        max_utterances=512,
        n_session_heads=8,
        n_exchange_heads=4,
        n_audio_quality_classes=3,
        n_video_quality_classes=3,
        n_text_quality_classes=3,
        n_therapeutic_phases=4,
        phase_embed_dim=64,
    )

    print("=" * 60)
    print("  MER Mental Health — Profiling")
    print("=" * 60)
    print(f"  Device: {device}")
    print(f"  Mode: {args.mode}")
    print(f"  Config: proj_dim={cfg.proj_dim}, d_session={cfg.d_session_model}")

    model = MERMentalHealthModel(cfg).to(device, dtype=dtype)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {total_params:,} total, {trainable:,} trainable")

    results = {
        "config": {
            "proj_dim": cfg.proj_dim,
            "d_session_model": cfg.d_session_model,
            "n_exchange_layers": cfg.n_exchange_layers,
            "n_session_layers": cfg.n_session_layers,
            "n_session_heads": cfg.n_session_heads,
        },
        "parameters": {
            "total": total_params,
            "trainable": trainable,
        },
    }

    # ── Inference profile ──
    print(f"\n{'─'*60}")
    print(f"  Inference Profile")
    print(f"{'─'*60}")

    inference_results = profile_inference(
        model, cfg, device, dtype,
        n_utterances=16 if device.type == "cpu" else 32,
        batch_size=4,
    )
    results["inference"] = inference_results

    print(f"  Latency (per utterance, forward):   {inference_results['latency']['ms_per_utterance_forward']:.2f} ms")
    print(f"  Latency (per session, {cfg.max_utterances} utterances): {inference_results['latency']['ms_per_session']:.2f} ms")
    print(f"  Throughput: {inference_results['throughput']['utterances_per_second']:.0f} utt/s")
    if "peak_memory_gb" in inference_results:
        print(f"  Peak memory: {inference_results['peak_memory_gb']:.3f} GB")
    print(f"  Estimated FLOPs: {inference_results['flop_estimate']['gflops_per_sample']:.2f} GFLOP/sample")

    # ── Training profile (if requested) ──
    if args.mode == "train":
        print(f"\n{'─'*60}")
        print(f"  Training Profile (fwd + bwd)")
        print(f"{'─'*60}")

        train_results = profile_training(
            model, cfg, device, dtype,
            n_utterances=8 if device.type == "cpu" else 16,
            batch_size=2,
        )
        results["training"] = train_results
        print(f"  Step time: {train_results['ms_per_step']:.1f} ms")
        print(f"  Steps/s: {train_results['steps_per_second']:.2f}")
        if "peak_memory_gb" in train_results:
            print(f"  Peak memory (train): {train_results['peak_memory_gb']:.3f} GB")

    # ── Torch profiler (if requested) ──
    if args.profile:
        print(f"\n{'─'*60}")
        print(f"  Torch Profiler")
        print(f"{'─'*60}")

        prof_results = profile_with_torch_profiler(model, cfg, device, dtype)
        results["torch_profiler"] = {
            "top_ops_by_time": prof_results["top_ops_by_time"],
            "top_ops_by_memory": prof_results.get("top_ops_by_memory", []),
        }
        print(prof_results["profiler_table"])

    # ── Save ──
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

        with open(args.output, "w") as f:
            json.dump(sanitize(results), f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
