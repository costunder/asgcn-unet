"""Synthetic CPU unit tests only; these are not trained quality measurements."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from asgcn_unet.transformer_decoder import (
    PatchExpansion,
    PatchMerging,
    RecurrentTransformerDecoder,
    WindowAttention,
    WindowTransformerBlock,
    _partition,
    _unpartition,
)
from asgcn_unet.unet import ConvGRUCell, RecurrentUNetDecoder, ResidualBlock


@pytest.fixture(autouse=True)
def synthetic_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _decoder(**kwargs):
    # Channel reduction is confined to a named synthetic unit-test fixture.
    return RecurrentTransformerDecoder(8, 12, 1, **kwargs)


@pytest.mark.parametrize("batch,height,width", [(1, 1, 1), (2, 5, 7), (3, 16, 16), (2, 17, 19)])
@pytest.mark.parametrize("recurrent", [False, True])
def test_all_pixels_odd_shapes_and_baseline_state_contract(batch, height, width, recurrent):
    torch.manual_seed(78)
    model = _decoder(recurrent=recurrent).eval()
    x = torch.randn(batch, 8, height, width)
    output_size = (height * 4 - 1, width * 4 + 1)
    with torch.no_grad():
        output, state = model(x, output_size)
        _, reference_state = RecurrentUNetDecoder(8, 12, 1, recurrent)(x, output_size)
    assert output.shape == (batch, 1, *output_size)
    assert torch.isfinite(output).all()
    assert (output >= 0).all() and (output <= 1).all()
    if recurrent:
        assert (
            state.shape
            == reference_state.shape
            == (batch, 48, math.ceil(height / 4), math.ceil(width / 4))
        )
    else:
        assert state is reference_state is None


def test_default_architecture_matches_existing_pyramid_and_mixing_depth():
    model = RecurrentTransformerDecoder(64, 48, 3)
    baseline = RecurrentUNetDecoder(64, 48, 3)
    blocks = [block for block in model.modules() if isinstance(block, WindowTransformerBlock)]
    assert len(blocks) == sum(isinstance(block, ResidualBlock) for block in baseline.modules()) == 6
    assert [block.attention.channels for block in blocks] == [48, 96, 192, 192, 96, 48]
    assert [block.attention.heads for block in blocks] == [3, 6, 12, 12, 6, 3]
    assert [block.attention.shift_size for block in blocks] == [0, 4, 0, 4, 0, 4]
    assert isinstance(model.recurrent, ConvGRUCell)
    assert model.recurrent.hidden_channels == baseline.recurrent.hidden_channels == 192
    # Spatial paths are real attention/linear patch operators. Only the
    # intentionally shared temporal ConvGRU keeps its convolutional gates.
    conv_ids = {id(layer) for layer in model.recurrent.modules() if isinstance(layer, nn.Conv2d)}
    assert {id(layer) for layer in model.modules() if isinstance(layer, nn.Conv2d)} == conv_ids
    assert not any(isinstance(layer, ResidualBlock) for layer in model.modules())


def test_production_channel_contract_forward_backward_and_deterministic_eval():
    # Only the synthetic image dimensions are small; all production channels,
    # heads, blocks and recurrence are exercised without changing any config.
    torch.manual_seed(8)
    model = RecurrentTransformerDecoder(64, 48, 3)
    x = torch.randn(2, 64, 17, 19, requires_grad=True)
    output, state = model(x, (69, 73))
    assert output.shape == (2, 3, 69, 73)
    assert state.shape == (2, 192, 5, 5)
    output.square().mean().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    model.eval()
    with torch.no_grad():
        evaluation, _ = model(x, (69, 73))
    torch.testing.assert_close(evaluation, output.detach())


def test_cpu_bfloat16_autocast_has_finite_prediction_state_and_gradients():
    torch.manual_seed(9)
    model = _decoder()
    x = torch.randn(2, 8, 9, 11, requires_grad=True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        output, state = model(x, (34, 42))
        loss = output.float().square().mean()
    loss.backward()
    assert output.dtype == state.dtype == torch.bfloat16
    assert torch.isfinite(output).all() and torch.isfinite(state).all()
    assert torch.isfinite(x.grad).all()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name


@pytest.mark.parametrize(
    "height,width,window,shift",
    [(8, 8, 4, 0), (8, 8, 4, 2), (5, 7, 4, 2), (1, 7, 4, 2), (3, 3, 4, 2), (9, 11, 4, 2)],
)
def test_shifted_attention_matches_explicit_nonwrapping_nonpadding_average(
    height, width, window, shift
):
    attention = WindowAttention(1, 1, window, shift).double()
    with torch.no_grad():
        attention.qkv.weight.zero_()
        attention.qkv.bias.zero_()
        attention.qkv.weight[2, 0] = 1.0
        attention.projection.weight.fill_(1.0)
        attention.projection.bias.zero_()
        attention.relative_position_bias.zero_()
    values = torch.arange(1, height * width + 1, dtype=torch.float64).reshape(1, height, width, 1)
    actual = attention(values)
    expected = torch.empty_like(values)
    shift_h = shift if height > window else 0
    shift_w = shift if width > window else 0
    # Deliberately slow independent CPU reference; not a model implementation.
    for row in range(height):
        for column in range(width):
            group_y = (row - shift_h) // window
            group_x = (column - shift_w) // window
            source = [
                values[0, yy, xx, 0]
                for yy in range(height)
                for xx in range(width)
                if (yy - shift_h) // window == group_y and (xx - shift_w) // window == group_x
            ]
            expected[0, row, column, 0] = torch.stack(source).mean()
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_sdpa_is_one_batched_call_with_only_window_length(monkeypatch):
    from asgcn_unet import transformer_decoder

    original = transformer_decoder.F.scaled_dot_product_attention
    calls = []

    def inspected(q, k, v, **kwargs):
        calls.append((q.shape, kwargs["attn_mask"].shape, kwargs["dropout_p"]))
        return original(q, k, v, **kwargs)

    monkeypatch.setattr(transformer_decoder.F, "scaled_dot_product_attention", inspected)
    model = WindowAttention(12, 3, 4, 2)
    output = model(torch.randn(3, 17, 19, 12))
    assert output.shape == (3, 17, 19, 12)
    assert calls == [(torch.Size([3 * 25, 3, 16, 4]), torch.Size([3 * 25, 3, 16, 16]), 0.0)]


@pytest.mark.parametrize("recurrent", [False, True])
def test_batched_output_and_state_equal_independent_samples(recurrent):
    torch.manual_seed(4)
    model = _decoder(recurrent=recurrent).eval()
    x = torch.randn(3, 8, 13, 15)
    state = torch.randn(3, 48, 4, 4) if recurrent else None
    with torch.no_grad():
        output, next_state = model(x, (49, 58), state)
        separate = [
            model(x[i : i + 1], (49, 58), None if state is None else state[i : i + 1])
            for i in range(3)
        ]
    torch.testing.assert_close(
        output, torch.cat([part[0] for part in separate]), atol=2e-6, rtol=2e-5
    )
    if recurrent:
        torch.testing.assert_close(
            next_state, torch.cat([part[1] for part in separate]), atol=2e-6, rtol=2e-5
        )


def test_every_parameter_receives_gradient_and_optimizer_update():
    torch.manual_seed(84)
    model = _decoder()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0)
    x = torch.randn(2, 8, 17, 19, requires_grad=True)
    state = torch.randn(2, 48, 5, 5, requires_grad=True)
    expected = torch.rand(2, 1, 41, 45)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    output, new_state = model(x, expected.shape[-2:], state)
    loss = (output - expected).square().mean()
    loss.backward()
    parameters = dict(model.named_parameters())
    assert set(map(id, parameters.values())) == {
        id(p) for group in optimizer.param_groups for p in group["params"]
    }
    assert new_state.grad_fn is not None
    assert x.grad.abs().sum() > 0 and state.grad.abs().sum() > 0
    for name, parameter in parameters.items():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    optimizer.step()
    for name, parameter in parameters.items():
        assert not torch.equal(parameter.detach(), before[name]), name


def test_recurrent_context_changes_predictions_and_wrong_shape_resets_like_baseline():
    torch.manual_seed(7)
    model = _decoder().eval()
    x = torch.randn(2, 8, 9, 11)
    with torch.no_grad():
        initial, state = model(x, (32, 41))
        continued, _ = model(x, (32, 41), state)
        reset, reset_state = model(x, (32, 41), torch.ones(1, 48, 3, 3))
    assert not torch.equal(initial, continued)
    torch.testing.assert_close(initial, reset)
    torch.testing.assert_close(state, reset_state)


def test_mask_cache_reused_and_bounded_to_latest_shape_and_cleared_by_dtype_change():
    attention = WindowAttention(12, 3, 4, 2)
    attention(torch.randn(2, 7, 9, 12))
    original = attention._mask_value
    attention(torch.randn(1, 7, 9, 12))
    assert attention._mask_value is original
    attention(torch.randn(1, 8, 9, 12))
    assert attention._mask_value is not original
    assert not any("mask" in key for key in attention.state_dict())
    attention.double()
    assert attention._mask_value is None


def test_partition_inverse_preserves_every_token_and_batch():
    values = torch.arange(2 * 12 * 20 * 3).reshape(2, 12, 20, 3)
    torch.testing.assert_close(_unpartition(_partition(values, 4), 12, 20, 4), values)


@pytest.mark.parametrize("height,width", [(1, 1), (5, 6), (6, 5), (8, 8)])
def test_merging_and_expansion_preserve_odd_border_gradient(height, width):
    torch.manual_seed(7)
    x = torch.randn(2, height, width, 12, requires_grad=True)
    merged = PatchMerging(12)(x)
    assert merged.shape == (2, math.ceil(height / 2), math.ceil(width / 2), 24)
    restored = PatchExpansion(24, 12)(merged, (height, width))
    assert restored.shape == x.shape
    restored.square().mean().backward()
    assert (x.grad.abs().sum(dim=-1) > 0).all()


def test_state_dict_roundtrip_without_static_geometry_caches():
    torch.manual_seed(1)
    first = _decoder().eval()
    second = _decoder().eval()
    x = torch.randn(2, 8, 11, 13)
    with torch.no_grad():
        expected = first(x, (42, 49))
        second.load_state_dict(first.state_dict(), strict=True)
        actual = second(x, (42, 49))
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"depths": [1]},
        {"depths": [1, 1, 0, 1, 1]},
        {"heads": [3, 6, 12, 6]},
        {"heads": [5, 6, 12, 6, 3]},
        {"window_size": 0},
        {"window_size": True},
        {"mlp_ratio": 0},
        {"mlp_ratio": float("nan")},
        {"mlp_ratio": True},
        {"recurrent": "yes"},
    ],
)
def test_invalid_contract_is_rejected(kwargs):
    with pytest.raises((ValueError, TypeError)):
        _decoder(**kwargs)


def test_invalid_expansion_or_input_shape_rejected():
    with pytest.raises(ValueError, match="undo"):
        PatchExpansion(24, 12)(torch.randn(1, 3, 3, 24), (4, 5))
    with pytest.raises(ValueError, match="non-empty"):
        _decoder()(torch.randn(2, 7, 5, 5), (20, 20))
    with pytest.raises(ValueError, match="positive"):
        _decoder()(torch.randn(2, 8, 5, 5), (0, 20))
