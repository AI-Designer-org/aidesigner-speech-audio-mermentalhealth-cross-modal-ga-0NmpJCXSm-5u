"""
Smoke test for the MER Mental Health Model.

Instantiates the full model with random initialization, runs forward passes
on synthetic data, and verifies all output shapes.

Usage:
    python smoke_test.py                  # Full test suite
    python smoke_test.py --fast           # Quick test (smaller dimensions, fewer utterances)
    python smoke_test.py --device cpu     # Run on CPU explicitly

Expected output:
    [OK] Model instantiated: X params | Trainable: Y
    [OK] Per-utterance forward: fused_feat (B, proj_dim)
    [OK] Full session forward: phq8_total (B,), phq8_items (B, 8), ...
    [OK] Streaming forward: phq8_total (B,)
    [OK] Loss computation: loss = X.XXX
    [OK] Stage 2 LoRA setup: trainable params after LoRA
    [OK] Stage 1 trainable params configured
    All smoke tests passed!
"""

import sys
import argparse
import torch
import torch.nn as nn

# Ensure we can import from the current directory
sys.path.insert(0, ".")

from config import ModelConfig
from model import MERMentalHealthModel, MEROutput, create_optimizer
from layers import count_params


def count_params_from_model(model: nn.Module):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def test_per_utterance(model: MERMentalHealthModel, cfg: ModelConfig,
                       device: torch.device, dtype: torch.dtype):
    """Test single-utterance encoding."""
    B = 4
    T_samples = 16000  # 1 second at 16kHz
    n_frames = cfg.video_n_frames
    C, H, W = 3, cfg.video_frame_size, cfg.video_frame_size
    T_tokens = cfg.text_max_length

    # Create synthetic data
    waveform = torch.randn(B, T_samples, device=device, dtype=dtype)
    frames = torch.randn(B, n_frames, C, H, W, device=device, dtype=dtype)
    input_ids = torch.randint(0, 1000, (B, T_tokens), device=device)

    with torch.no_grad():
        fused_feat = model.forward_utterance(waveform, frames, input_ids)

    expected = (B, cfg.proj_dim)
    assert fused_feat.shape == expected, \
        f"forward_utterance: expected {expected}, got {fused_feat.shape}"
    print(f"  [OK] forward_utterance: fused_feat {fused_feat.shape}")

    # Test with missing modality masks
    audio_mask = torch.ones(B, device=device, dtype=torch.bool)
    video_mask = torch.ones(B, device=device, dtype=torch.bool)
    text_mask = torch.ones(B, device=device, dtype=torch.bool)
    audio_mask[0] = False  # Audio missing for first sample

    with torch.no_grad():
        fused_feat2 = model.forward_utterance(
            waveform, frames, input_ids,
            audio_mask=audio_mask, video_mask=video_mask, text_mask=text_mask,
        )

    assert fused_feat2.shape == expected, \
        f"forward_utterance (missing modality): expected {expected}, got {fused_feat2.shape}"
    print(f"  [OK] forward_utterance (missing modality): fused_feat {fused_feat2.shape}")

    return True


def test_full_session(model: MERMentalHealthModel, cfg: ModelConfig,
                      device: torch.device, dtype: torch.dtype):
    """Test full session-level forward pass."""
    B = 2
    T_utt = 16  # utterances per session
    T_samples = 16000  # 1 sec audio per utterance
    n_frames = cfg.video_n_frames
    C, H, W = 3, cfg.video_frame_size, cfg.video_frame_size
    T_tokens = cfg.text_max_length

    # Session-level inputs
    waveforms = torch.randn(B, T_utt, T_samples, device=device, dtype=dtype)
    frames = torch.randn(B, T_utt, n_frames, C, H, W, device=device, dtype=dtype)
    input_ids = torch.randint(0, 1000, (B, T_utt, T_tokens), device=device)
    phase_labels = torch.randint(0, cfg.n_therapeutic_phases, (B, T_utt), device=device)
    timestamps = torch.linspace(0, 600, T_utt, device=device).unsqueeze(0).expand(B, -1)
    utterance_mask = torch.ones(B, T_utt, device=device, dtype=torch.bool)
    phq8_labels = torch.rand(B, device=device) * 24  # random PHQ-8 scores [0, 24]

    with torch.no_grad():
        outputs = model(
            waveforms=waveforms,
            frames=frames,
            input_ids=input_ids,
            phase_labels=phase_labels,
            timestamps=timestamps,
            utterance_mask=utterance_mask,
            phq8_labels=phq8_labels,
            return_all=True,
        )

    # Check shapes
    assert isinstance(outputs, MEROutput), f"Expected MEROutput, got {type(outputs)}"
    assert outputs.phq8_total.shape == (B,), \
        f"phq8_total: expected ({B},), got {outputs.phq8_total.shape}"
    assert outputs.phq8_items.shape == (B, cfg.n_phq8_items), \
        f"phq8_items: expected ({B}, {cfg.n_phq8_items}), got {outputs.phq8_items.shape}"
    assert outputs.concept_scores.shape == (B, cfg.n_clinical_concepts), \
        f"concept_scores: expected ({B}, {cfg.n_clinical_concepts}), got {outputs.concept_scores.shape}"
    assert outputs.depression_prob.shape == (B,), \
        f"depression_prob: expected ({B},), got {outputs.depression_prob.shape}"
    assert outputs.loss is not None, "Expected loss tensor when phq8_labels provided"

    # Check intermediate representations
    assert outputs.utterance_feats is not None
    assert outputs.utterance_feats.shape == (B, T_utt, cfg.proj_dim), \
        f"utterance_feats: expected ({B}, {T_utt}, {cfg.proj_dim}), got {outputs.utterance_feats.shape}"

    print(f"  [OK] full session phq8_total:     {outputs.phq8_total.shape}")
    print(f"  [OK] full session phq8_items:     {outputs.phq8_items.shape}")
    print(f"  [OK] full session concept_scores: {outputs.concept_scores.shape}")
    print(f"  [OK] full session depression_prob: {outputs.depression_prob.shape}")
    print(f"  [OK] full session loss:           {outputs.loss.item():.4f}")

    # Test with a subset of modalities missing
    audio_mask = torch.ones(B, T_utt, device=device, dtype=torch.bool)
    video_mask = torch.ones(B, T_utt, device=device, dtype=torch.bool)
    text_mask = torch.ones(B, T_utt, device=device, dtype=torch.bool)
    audio_mask[:, 4:8] = False  # utterances 4-7 have no audio

    with torch.no_grad():
        outputs2 = model(
            waveforms=waveforms,
            frames=frames,
            input_ids=input_ids,
            phase_labels=phase_labels,
            timestamps=timestamps,
            utterance_mask=utterance_mask,
            audio_mask=audio_mask,
            video_mask=video_mask,
            text_mask=text_mask,
        )

    assert outputs2.phq8_total.shape == (B,), \
        f"Missing modality: phq8_total expected ({B},), got {outputs2.phq8_total.shape}"
    print(f"  [OK] full session (missing modalities): phq8_total {outputs2.phq8_total.shape}")

    return True


def test_streaming(model: MERMentalHealthModel, cfg: ModelConfig,
                   device: torch.device, dtype: torch.dtype):
    """Test streaming (per-utterance) forward pass."""
    B = 4
    proj_dim = cfg.proj_dim

    fused_feat = torch.randn(B, proj_dim, device=device, dtype=dtype)

    with torch.no_grad():
        outputs = model.forward_streaming(fused_feat)

    assert outputs.phq8_total.shape == (B,), \
        f"Streaming phq8_total: expected ({B},), got {outputs.phq8_total.shape}"
    assert outputs.phq8_items.shape == (B, cfg.n_phq8_items), \
        f"Streaming phq8_items: expected ({B}, {cfg.n_phq8_items}), got {outputs.phq8_items.shape}"

    print(f"  [OK] streaming phq8_total:     {outputs.phq8_total.shape}")
    print(f"  [OK] streaming phq8_items:     {outputs.phq8_items.shape}")

    return True


def test_loss_computation(model: MERMentalHealthModel, cfg: ModelConfig,
                          device: torch.device, dtype: torch.dtype):
    """Verify loss computation produces finite values."""
    B = 4

    phq8_total = torch.rand(B, device=device) * 24
    phq8_items = torch.rand(B, cfg.n_phq8_items, device=device) * 3
    depression_prob = torch.rand(B, device=device)
    concept_scores = torch.rand(B, cfg.n_clinical_concepts, device=device)
    phq8_labels = torch.rand(B, device=device) * 24

    loss = model.compute_loss(
        phq8_total, phq8_items, depression_prob, concept_scores,
        phq8_labels,
    )

    assert torch.isfinite(loss), f"Loss is not finite: {loss}"
    assert loss.ndim == 0, f"Loss should be scalar, got shape {loss.shape}"
    print(f"  [OK] loss computation: {loss.item():.4f}")

    # Test with concept labels
    concept_labels = torch.rand(B, cfg.n_clinical_concepts, device=device)
    loss2 = model.compute_loss(
        phq8_total, phq8_items, depression_prob, concept_scores,
        phq8_labels, concept_labels=concept_labels,
    )
    assert torch.isfinite(loss2), f"Loss with concepts is not finite: {loss2}"
    print(f"  [OK] loss computation (w/ concept supervision): {loss2.item():.4f}")

    return True


def test_training_setup(model: MERMentalHealthModel, cfg: ModelConfig,
                        device: torch.device, dtype: torch.dtype):
    """Verify training stage configuration."""
    # Stage 1
    model.set_trainable_params(stage=1)
    total, trainable = count_params_from_model(model)
    assert trainable > 0, "Stage 1: expected some trainable params"
    assert trainable < total, "Stage 1: expected some frozen params (encoders)"
    print(f"  [OK] Stage 1: {trainable:,}/{total:,} params trainable")

    # Stage 2 (with LoRA)
    model.set_trainable_params(stage=2)
    total2, trainable2 = count_params_from_model(model)
    assert trainable2 > 0, "Stage 2: expected some trainable params (LoRA)"
    print(f"  [OK] Stage 2 (LoRA): {trainable2:,}/{total2:,} params trainable")

    # Optimizer creation
    optimizer = create_optimizer(model, cfg, stage=2)
    assert len(optimizer.param_groups) == 2, \
        f"Expected 2 param groups (decay + no_decay), got {len(optimizer.param_groups)}"
    print(f"  [OK] Optimizer: {len(optimizer.param_groups)} param groups")

    return True


def test_gradient_flow(model: MERMentalHealthModel, cfg: ModelConfig,
                       device: torch.device, dtype: torch.dtype):
    """Verify gradients flow through the full model."""
    model.set_trainable_params(stage=1)
    model.train()

    B = 2
    T_utt = 8
    T_samples = 8000
    n_frames = cfg.video_n_frames
    C, H, W = 3, cfg.video_frame_size, cfg.video_frame_size
    T_tokens = cfg.text_max_length

    waveforms = torch.randn(B, T_utt, T_samples, device=device, dtype=dtype)
    frames = torch.randn(B, T_utt, n_frames, C, H, W, device=device, dtype=dtype)
    input_ids = torch.randint(0, 1000, (B, T_utt, T_tokens), device=device)
    phase_labels = torch.randint(0, cfg.n_therapeutic_phases, (B, T_utt), device=device)
    timestamps = torch.linspace(0, 300, T_utt, device=device).unsqueeze(0).expand(B, -1)
    utterance_mask = torch.ones(B, T_utt, device=device, dtype=torch.bool)
    phq8_labels = torch.rand(B, device=device) * 24

    outputs = model(
        waveforms=waveforms,
        frames=frames,
        input_ids=input_ids,
        phase_labels=phase_labels,
        timestamps=timestamps,
        utterance_mask=utterance_mask,
        phq8_labels=phq8_labels,
    )

    loss = outputs.loss
    loss.backward()

    # Check gradients exist on trainable params
    grad_count = 0
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            grad_count += 1
            assert torch.isfinite(p.grad).all(), f"Non-finite gradient in {name}"

    assert grad_count > 0, "No gradients flowed to any trainable parameters"
    print(f"  [OK] Gradient flow: {grad_count} params received gradients")

    model.zero_grad()
    return True


def test_modality_dropout_augmentation(device: torch.device):
    """Test the modality dropout augmentation function."""
    from model import modality_dropout_augmentation

    B, T = 4, 32
    audio_mask = torch.ones(B, T, device=device, dtype=torch.bool)
    video_mask = torch.ones(B, T, device=device, dtype=torch.bool)
    text_mask = torch.ones(B, T, device=device, dtype=torch.bool)

    a, v, t = modality_dropout_augmentation(
        audio_mask, video_mask, text_mask,
        random_dropout_prob=0.3,
        clinical_correlated_dropout_prob=0.2,
    )

    assert a.shape == audio_mask.shape, f"augmented audio mask shape mismatch"
    assert v.shape == video_mask.shape, f"augmented video mask shape mismatch"
    assert t.shape == text_mask.shape, f"augmented text mask shape mismatch"

    # Some entries should be dropped
    n_dropped = int((~a).sum().item() + (~v).sum().item() + (~t).sum().item())
    assert n_dropped > 0, "Expected some modality dropout"
    print(f"  [OK] Modality dropout augmentation: {n_dropped} entries dropped")

    return True


def main():
    parser = argparse.ArgumentParser(description="MER Mental Health Model Smoke Test")
    parser.add_argument("--fast", action="store_true", help="Quick test with smaller config")
    parser.add_argument("--device", default=None, help="Device (cpu, cuda)")
    args = parser.parse_args()

    # Device setup
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"Device: {device} | dtype: {dtype}")

    # Config
    if args.fast:
        cfg = ModelConfig(
            pretrained=False,
            proj_dim=48,
            d_session_model=48,
            audio_feat_dim=48,
            video_feat_dim=48,
            text_feat_dim=48,
            n_exchange_layers=1,
            n_session_layers=2,
            exchange_hidden_dim=24,
            concept_hidden_dim=24,
            quality_estimator_hidden=12,
            max_utterances=64,
            n_session_heads=4,
            n_exchange_heads=4,
        )
    else:
        cfg = ModelConfig(
            pretrained=False,
            # Moderate dimensions for CPU testing
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
            max_utterances=128,
            n_session_heads=6,
            n_exchange_heads=4,
        )

    print(f"Mode: {'fast' if args.fast else 'standard'}")

    print(f"\n--- Config ---")
    print(f"  proj_dim={cfg.proj_dim}, d_session_model={cfg.d_session_model}")
    print(f"  n_exchange_layers={cfg.n_exchange_layers}, n_session_layers={cfg.n_session_layers}")
    print(f"  n_exchange_heads={cfg.n_exchange_heads}, n_session_heads={cfg.n_session_heads}")

    # Instantiate model
    print(f"\n--- Instantiating Model ---")
    model = MERMentalHealthModel(cfg).to(device=device, dtype=dtype)
    total, trainable = count_params_from_model(model)
    print(f"  Total params: {total:,} | Trainable: {trainable:,}")
    assert total > 0, "Model has zero parameters!"
    print(f"  [OK] Model instantiated")

    # Run tests
    print(f"\n--- Running Smoke Tests ---")

    tests = [
        ("Per-utterance forward", test_per_utterance, (model, cfg, device, dtype)),
        ("Full session forward", test_full_session, (model, cfg, device, dtype)),
        ("Streaming forward", test_streaming, (model, cfg, device, dtype)),
        ("Loss computation", test_loss_computation, (model, cfg, device, dtype)),
        ("Training setup", test_training_setup, (model, cfg, device, dtype)),
        ("Gradient flow", test_gradient_flow, (model, cfg, device, dtype)),
        ("Modality dropout augmentation", test_modality_dropout_augmentation, (device,)),
    ]

    passed = 0
    failed = 0
    for name, test_fn, args_tuple in tests:
        try:
            print(f"\n  >>> {name} <<<")
            test_fn(*args_tuple)
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    # Summary
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    if failed > 0:
        print(f"[FAILED] Some smoke tests did not pass!")
        sys.exit(1)
    else:
        print(f"[PASSED] All smoke tests passed!")

    count_params(model)


if __name__ == "__main__":
    main()
