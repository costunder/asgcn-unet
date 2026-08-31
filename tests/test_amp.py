"""Same-sample AMP overflow recovery without skipped optimizer updates."""

from __future__ import annotations

import copy
import random
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn

from asgcn_unet import engine


class _StatefulLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scalar = nn.Parameter(torch.tensor(0.25))
        self.bn = nn.BatchNorm1d(3)
        self.dropout = nn.Dropout(0.4)
        self.linear = nn.Linear(3, 1)
        self.register_buffer("forward_count", torch.zeros((), dtype=torch.long))

    def forward(self, values: torch.Tensor, multiplier: float = 1.0):
        self.forward_count.add_(1)
        hidden = self.dropout(self.bn(values))
        # Cast-backward overflows at a 65536 scale despite a finite loss and
        # finite unscaled derivative. This uses real fp16 arithmetic/GradScaler.
        loss = self.scalar.to(torch.float16).float() * multiplier
        loss = loss + self.linear(hidden).square().mean() * 0.001
        return loss, hidden


def _scaler(scale: float = 65536.0, *, enabled: bool = True):
    return torch.amp.GradScaler("cpu", init_scale=scale, enabled=enabled)


def _step(model, optimizer, scaler, closure, **overrides):
    return engine._training_step(
        model,
        optimizer,
        scaler,
        closure,
        optimizer_mode=overrides.pop("optimizer_mode", "adamw"),
        max_norm=1.0,
        epoch=3,
        step=7,
        sample_id="test-sequence/image000000000",
        **overrides,
    )


def _rng_equal(actual, expected) -> None:
    assert actual["python"] == expected["python"]
    assert actual["numpy"][0] == expected["numpy"][0]
    np.testing.assert_array_equal(actual["numpy"][1], expected["numpy"][1])
    assert actual["numpy"][2:] == expected["numpy"][2:]
    torch.testing.assert_close(actual["torch"], expected["torch"], rtol=0, atol=0)
    if "cuda" in expected:
        torch.testing.assert_close(actual["cuda"], expected["cuda"], rtol=0, atol=0)


@pytest.mark.parametrize("optimizer_mode", ["adamw", "adam_gc"])
def test_retry_matches_one_safe_scale_update_and_restores_buffers_rng(optimizer_mode) -> None:
    torch.manual_seed(461)
    model = _StatefulLoss().train()
    reference = copy.deepcopy(model)
    optimizer_class = torch.optim.Adam if optimizer_mode == "adam_gc" else torch.optim.AdamW
    optimizer = optimizer_class(model.parameters(), lr=0.001)
    reference_optimizer = optimizer_class(reference.parameters(), lr=0.001)
    scaler = _scaler()
    reference_scaler = _scaler(32768.0)
    values = torch.arange(24, dtype=torch.float32).reshape(8, 3) / 24
    incoming_state = torch.tensor([0.1, 0.2])
    initial_parameters = {name: value.detach().clone() for name, value in model.named_parameters()}
    initial_rng = engine._capture_rng_state()
    attempts = []

    def forward_loss():
        # A failed attempt has not initialized Adam state or changed weights.
        assert not optimizer.state
        for name, value in model.named_parameters():
            torch.testing.assert_close(value, initial_parameters[name], rtol=0, atol=0)
        loss, hidden = model(values)
        payload = {
            "hidden": hidden.detach().clone(),
            "state": incoming_state + 1,
            "python": random.random(),
            "numpy": np.random.random(),
        }
        attempts.append(payload)
        return loss, {"reconstruction": loss.detach()}, payload

    with patch.object(optimizer, "step", wraps=optimizer.step) as optimizer_step:
        payload, loss_values, gradient_norm, info = _step(
            model, optimizer, scaler, forward_loss, optimizer_mode=optimizer_mode
        )
    final_rng = engine._capture_rng_state()
    assert optimizer_step.call_count == 1
    assert len(attempts) == 2
    assert payload is attempts[-1]
    torch.testing.assert_close(attempts[0], attempts[1], rtol=0, atol=0)
    assert info == {"scale_before": 65536.0, "scale_after": 32768.0, "retries": 1}
    assert model.forward_count.item() == model.bn.num_batches_tracked.item() == 1
    assert gradient_norm > 0
    assert loss_values["total"] == loss_values["reconstruction"]
    torch.testing.assert_close(incoming_state, torch.tensor([0.1, 0.2]))

    engine._restore_rng_state(initial_rng)

    def reference_forward():
        loss, hidden = reference(values)
        result = {
            "hidden": hidden.detach().clone(),
            "state": incoming_state + 1,
            "python": random.random(),
            "numpy": np.random.random(),
        }
        return loss, {"reconstruction": loss.detach()}, result

    expected_payload, _, _, expected_info = _step(
        reference,
        reference_optimizer,
        reference_scaler,
        reference_forward,
        optimizer_mode=optimizer_mode,
    )
    assert expected_info["retries"] == 0
    torch.testing.assert_close(payload, expected_payload, rtol=0, atol=0)
    torch.testing.assert_close(model.state_dict(), reference.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(optimizer.state_dict(), reference_optimizer.state_dict(), rtol=0, atol=0)
    assert scaler.state_dict() == reference_scaler.state_dict()
    _rng_equal(engine._capture_rng_state(), final_rng)


def test_default_bound_can_retry_all_the_way_to_scale_one() -> None:
    model = nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.Adam(model.parameters())
    attempts = 0

    def closure():
        nonlocal attempts
        attempts += 1
        loss = model.weight.to(torch.float16).float().sum() * 32768.0
        return loss, {}, "same-sample"

    payload, _, _, info = _step(model, optimizer, _scaler(), closure)
    assert attempts == 17
    assert payload == "same-sample"
    assert info == {"scale_before": 65536.0, "scale_after": 1.0, "retries": 16}
    assert optimizer.state[model.weight]["step"].item() == 1


class _BadBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.clone()

    @staticmethod
    def backward(ctx, gradient):
        return torch.full_like(gradient, float("nan"))


@pytest.mark.parametrize("enabled", [False, True])
def test_persistent_nonfinite_is_fatal_without_weight_or_buffer_updates(enabled) -> None:
    model = _StatefulLoss()
    optimizer = torch.optim.Adam(model.parameters())
    before = copy.deepcopy(model.state_dict())
    rng = engine._capture_rng_state()
    attempts = 0

    def closure():
        nonlocal attempts
        attempts += 1
        loss, _ = model(torch.arange(24, dtype=torch.float32).reshape(8, 3))
        loss = _BadBackward.apply(loss)
        random.random()
        np.random.random()
        return loss, {}, None

    with (
        patch.object(optimizer, "step", wraps=optimizer.step) as optimizer_step,
        pytest.raises(FloatingPointError) as error,
    ):
        _step(model, optimizer, _scaler(enabled=enabled), closure, max_amp_retries=2)
    assert "epoch=3, step=7, sample=test-sequence/image000000000" in str(error.value)
    assert "scale=" in str(error.value)
    assert "parameters=scalar" in str(error.value)
    assert attempts == (3 if enabled else 1)
    assert optimizer_step.call_count == 0
    assert not optimizer.state
    assert all(parameter.grad is None for parameter in model.parameters())
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
    if enabled:
        torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)
        _rng_equal(engine._capture_rng_state(), rng)


@pytest.mark.parametrize("component", [False, True])
def test_nonfinite_forward_loss_is_never_retried(component) -> None:
    model = _StatefulLoss()
    optimizer = torch.optim.Adam(model.parameters())
    scaler = _scaler()
    attempts = 0
    before = copy.deepcopy(model.state_dict())

    def closure():
        nonlocal attempts
        attempts += 1
        loss, _ = model(torch.arange(24, dtype=torch.float32).reshape(8, 3))
        if component:
            parts = {"bad-component": torch.tensor(float("inf"))}
        else:
            loss = loss * float("nan")
            parts = {}
        return loss, parts, None

    with pytest.raises(FloatingPointError, match="bad-component" if component else "total loss"):
        _step(model, optimizer, scaler, closure)
    assert attempts == 1
    assert scaler.get_scale() == 65536.0
    assert not optimizer.state
    torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)


def test_unrelated_clipping_error_keeps_original_exception(monkeypatch) -> None:
    model = _StatefulLoss()
    optimizer = torch.optim.Adam(model.parameters())
    scaler = _scaler(32768.0)
    before = copy.deepcopy(model.state_dict())
    failure = RuntimeError("diagnostic backend launch failure")
    calls = 0

    def closure():
        nonlocal calls
        calls += 1
        loss, _ = model(torch.arange(24, dtype=torch.float32).reshape(8, 3))
        return loss, {}, None

    def broken_clip(*args, **kwargs):
        raise failure

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", broken_clip)
    with pytest.raises(RuntimeError) as caught:
        _step(model, optimizer, scaler, closure)
    assert caught.value is failure
    assert calls == 1
    assert not optimizer.state
    assert scaler.get_scale() == 32768.0
    torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)


def test_post_centralization_nonfinite_does_not_trigger_amp_backoff(monkeypatch) -> None:
    model = nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters())
    scaler = _scaler(32768.0)

    def corrupt_gradients(_model):
        _model.weight.grad.fill_(float("nan"))

    def closure():
        loss = model(torch.ones(2)).sum()
        return loss, {}, None

    monkeypatch.setattr(engine, "_centralize_gradients", corrupt_gradients)
    with pytest.raises(FloatingPointError, match="before clipping"):
        _step(model, optimizer, scaler, closure, optimizer_mode="adam_gc")
    assert scaler.get_scale() == 32768.0
    assert not optimizer.state


def test_overflow_outside_optimizer_does_not_accidentally_step_weights() -> None:
    model = nn.ParameterDict(
        {"included": nn.Parameter(torch.tensor(0.2)), "excluded": nn.Parameter(torch.tensor(0.3))}
    )
    optimizer = torch.optim.Adam([model["included"]])

    def closure():
        loss = model["included"] * 0 + model["excluded"].half().float()
        return loss, {}, None

    with pytest.raises(FloatingPointError, match="did not safely back off"):
        _step(model, optimizer, _scaler(), closure)
    assert not optimizer.state
    assert model["included"].item() == pytest.approx(0.2)


@pytest.mark.parametrize("limit", [-1, True, 1.5])
def test_invalid_retry_limits_fail_before_forward(limit) -> None:
    model = nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters())
    with pytest.raises(ValueError, match="nonnegative integer"):
        _step(model, optimizer, _scaler(), lambda: pytest.fail("must not run"), max_amp_retries=limit)


def test_disabled_amp_matches_direct_non_amp_optimizer_step() -> None:
    torch.manual_seed(641)
    model = nn.Linear(3, 2)
    reference = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.001)
    values = torch.randn(4, 3)

    def closure():
        loss = model(values).square().mean()
        return loss, {}, "payload"

    _, _, _, info = _step(model, optimizer, _scaler(enabled=False), closure)
    reference_optimizer.zero_grad(set_to_none=True)
    reference(values).square().mean().backward()
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0, error_if_nonfinite=True)
    reference_optimizer.step()
    torch.testing.assert_close(model.state_dict(), reference.state_dict(), rtol=0, atol=0)
    assert info == {"scale_before": 1.0, "scale_after": 1.0, "retries": 0}


def test_protocol_binds_retry_policy_and_rejects_tampering() -> None:
    config = {"train": {"amp": True}}
    protocol = engine._training_protocol(config, torch.device("cpu"))
    assert protocol["version"] == 5
    assert engine._valid_training_protocol_contract(protocol)
    assert protocol["mixed_precision"]["overflow_policy"] is None
    policy = engine._amp_retry_policy(True)
    assert policy["max_retries"] == 16
    assert policy["skip_samples"] is False
    assert policy["restore_model_buffers"] is True
    protocol["mixed_precision"]["effective"] = True
    protocol["mixed_precision"]["overflow_policy"] = policy
    assert engine._valid_training_protocol_contract(protocol)
    protocol["mixed_precision"]["overflow_policy"]["skip_samples"] = True
    assert not engine._valid_training_protocol_contract(protocol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device not available")
def test_cuda_fp16_overflow_retries_same_sample_once() -> None:
    model = nn.Linear(1, 1, bias=False).cuda()
    optimizer = torch.optim.Adam(model.parameters())
    scaler = torch.amp.GradScaler("cuda")
    calls = 0

    def closure():
        nonlocal calls
        calls += 1
        with torch.autocast("cuda", dtype=torch.float16):
            output = model(torch.ones((1, 1), device="cuda"))
            loss = output.float().sum()
        return loss, {}, output

    output, _, _, info = _step(model, optimizer, scaler, closure)
    assert calls == 2
    assert info["retries"] == 1
    assert info["scale_after"] == 32768.0
    assert torch.isfinite(output).all()
    assert optimizer.state[model.weight]["step"].item() == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device not available")
def test_cuda_full_model_empty_frame_retry_preserves_sample_buffers_and_rng() -> None:
    """Full-size synthetic regression input, never experiment/dataset evidence."""
    from asgcn_unet.losses import ReconstructionLoss
    from asgcn_unet.model import ASGCNUNet

    torch.manual_seed(2026)
    model = ASGCNUNet().cuda().train()
    model.register_buffer("retry_test_counter", torch.zeros((), dtype=torch.long, device="cuda"))
    optimizer = engine._build_optimizer(model, {"optimizer": "adam_gc"})
    scaler = torch.amp.GradScaler("cuda")
    criterion = ReconstructionLoss()
    sample = {
        "events": torch.empty((0, 4), device="cuda"),
        "target": torch.linspace(0.1, 0.95, 240 * 320, device="cuda").reshape(1, 240, 320),
        "sensor_size": (240, 320),
        "sample_id": "synthetic-regression/empty-frame",
        "metadata": {},
    }
    before_buffers = {name: value.clone() for name, value in model.named_buffers()}
    before_bias = model.decoder.head.bias.detach().clone()
    before_rng = engine._capture_rng_state()
    draws = []
    frame_ids = []
    attempts = 0

    def controlled_overflow(gradient):
        # Ensure this regression exercises retry even on hardware where the
        # initial scale happens to be safe. Subsequent backward passes are real.
        return torch.full_like(gradient, float("inf")) if attempts == 1 else gradient

    overflow_hook = model.decoder.head.bias.register_hook(controlled_overflow)

    def forward_loss():
        nonlocal attempts
        attempts += 1
        assert not optimizer.state
        torch.testing.assert_close(model.decoder.head.bias, before_bias, rtol=0, atol=0)
        model.retry_test_counter.add_(1)
        frame_ids.append(sample["sample_id"])
        draws.append((torch.rand(8, device="cuda").cpu(), random.random(), np.random.random()))
        with torch.autocast("cuda", dtype=torch.float16):
            prediction, diagnostics = model.forward_sample(sample, recurrent_state=None)
            loss, parts = criterion(prediction, sample["target"].unsqueeze(0))
        return loss, parts, (prediction, diagnostics)

    try:
        with patch.object(optimizer, "step", wraps=optimizer.step) as optimizer_step:
            payload, loss_values, gradient_norm, info = _step(
                model, optimizer, scaler, forward_loss, optimizer_mode="adam_gc"
            )
    finally:
        overflow_hook.remove()

    prediction, diagnostics = payload
    assert optimizer_step.call_count == 1
    assert attempts == info["retries"] + 1
    assert 1 <= info["retries"] <= 16
    assert frame_ids == [sample["sample_id"]] * attempts
    for draw in draws[1:]:
        torch.testing.assert_close(draw, draws[0], rtol=0, atol=0)
    assert diagnostics["nodes"] == diagnostics["edges"] == 0
    assert torch.isfinite(prediction).all()
    assert np.isfinite(loss_values["total"])
    assert np.isfinite(gradient_norm)
    assert not engine._nonfinite_gradient_names(model)
    assert model.retry_test_counter.item() == 1
    for name, value in model.named_buffers():
        if name != "retry_test_counter":
            torch.testing.assert_close(value, before_buffers[name], rtol=0, atol=0)
    assert all(state["step"].item() == 1 for state in optimizer.state.values())

    after_rng = engine._capture_rng_state()
    engine._restore_rng_state(before_rng)
    torch.rand(8, device="cuda")
    random.random()
    np.random.random()
    expected_rng = engine._capture_rng_state()
    engine._restore_rng_state(after_rng)
    _rng_equal(after_rng, expected_rng)
