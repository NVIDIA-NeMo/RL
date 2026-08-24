# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys

import pytest
import torch

from nemo_rl.models.generation.vllm.quantization.fp8_train_utils import (
    MXFP8_BLOCK_SIZE,
    _mxfp8_e4m3_quantize_torch,
    mxfp8_e4m3_quantize_for_refit,
)

pytestmark = pytest.mark.vllm


def _dequantize(x_fp8: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    num_blocks = x_fp8.shape[-1] // MXFP8_BLOCK_SIZE
    x_blocked = x_fp8.to(torch.float32).view(
        *x_fp8.shape[:-1], num_blocks, MXFP8_BLOCK_SIZE
    )
    descale = torch.exp2(scales.to(torch.float32) - 127.0)
    return (x_blocked * descale.unsqueeze(-1)).view(*x_fp8.shape)


@pytest.mark.parametrize("shape", [(64, 128), (7, 96), (4, 16, 64)])
def test_torch_reference_shapes_and_roundtrip(shape):
    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=torch.bfloat16)

    x_fp8, scales = _mxfp8_e4m3_quantize_torch(x)

    assert x_fp8.shape == x.shape
    assert x_fp8.dtype == torch.float8_e4m3fn
    assert scales.dtype == torch.uint8
    expected_scale_shape = (*shape[:-1], shape[-1] // MXFP8_BLOCK_SIZE)
    assert tuple(scales.shape) == expected_scale_shape

    x_dq = _dequantize(x_fp8, scales)
    x32 = x.to(torch.float32)
    abs_err = (x_dq - x32).abs()
    block_amax = (
        x32.abs()
        .reshape(*shape[:-1], shape[-1] // MXFP8_BLOCK_SIZE, MXFP8_BLOCK_SIZE)
        .amax(dim=-1, keepdim=True)
        .expand(*shape[:-1], shape[-1] // MXFP8_BLOCK_SIZE, MXFP8_BLOCK_SIZE)
        .reshape(shape)
    )
    # e4m3 has 3 mantissa bits: elements within the block's representable range
    # must round-trip to ~12.5% relative error; elements far below the block
    # amax may legitimately quantize to zero, so bound them by absolute error
    # instead of a ratio (keeps the test deterministic across torch versions).
    representable = x32.abs() >= block_amax / 64
    rel_err = (abs_err / x32.abs().clamp(min=1e-6))[representable]
    assert rel_err.median() < 0.05
    assert rel_err.max() < 0.25
    assert (abs_err[~representable] <= block_amax[~representable] / 32).all()


def test_last_dim_not_divisible_raises():
    x = torch.randn(8, MXFP8_BLOCK_SIZE + 1, dtype=torch.bfloat16)
    with pytest.raises(AssertionError):
        _mxfp8_e4m3_quantize_torch(x)


def test_refit_quantize_preserves_single_scale_block_dimension():
    x = torch.randn(8, MXFP8_BLOCK_SIZE, dtype=torch.bfloat16)

    _, scales = mxfp8_e4m3_quantize_for_refit(x)

    assert scales.shape == (8, 1)


def test_blackwell_refit_prequantization_requires_flashinfer(monkeypatch):
    class FakeBlackwellTensor:
        is_cuda = True
        device = "cuda"

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (10, 0))
    monkeypatch.setitem(sys.modules, "flashinfer", None)

    with pytest.raises(RuntimeError, match=r"sm100\+ requires FlashInfer"):
        mxfp8_e4m3_quantize_for_refit(FakeBlackwellTensor())


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0),
    reason=(
        "requires sm100+; below it both sides fall back to the shared torch "
        "reference and the comparison is vacuous"
    ),
)
def test_refit_quantize_matches_receiver_path():
    """Bitwise parity with the vLLM receiver path (mxfp8_e4m3_quantize + squeeze)."""
    vllm_mxfp8 = pytest.importorskip(
        "vllm.model_executor.layers.quantization.utils.mxfp8_utils"
    )

    torch.manual_seed(0)
    x = torch.randn(256, 512, dtype=torch.bfloat16, device="cuda")
    x[0].zero_()

    ref_lp, ref_scale = vllm_mxfp8.mxfp8_e4m3_quantize(x)
    ref_scale = torch.squeeze(ref_scale, dim=-1)
    assert torch.any(ref_scale == 0)
    ref_scale = torch.where(ref_scale == 0, torch.ones_like(ref_scale), ref_scale)

    got_lp, got_scale = mxfp8_e4m3_quantize_for_refit(x)

    assert got_lp.dtype == ref_lp.dtype
    assert torch.equal(got_lp.view(torch.uint8), ref_lp.view(torch.uint8))
    assert got_scale.dtype == ref_scale.dtype
    assert got_scale.shape == ref_scale.shape
    assert torch.equal(got_scale.reshape(-1), ref_scale.reshape(-1))


def test_batched_expert_prequantization_preserves_wire_entries_and_reuses_scratch():
    from nemo_rl.models.generation.vllm.quantization import fp8_train_utils

    calls = []

    def quantize(tensor):
        calls.append(tuple(tensor.shape))
        scales = torch.ones(
            (*tensor.shape[:-1], tensor.shape[-1] // MXFP8_BLOCK_SIZE),
            dtype=torch.uint8,
        )
        return tensor.detach().clone(), scales

    def expert_name(expert_id, projection):
        return f"model.layers.0.mlp.experts.{expert_id}.{projection}_proj.weight"

    params = [("model.layers.0.input_layernorm.weight", torch.ones(64))]
    expected = {}
    for expert_id in range(2):
        for projection in ("gate", "up", "down"):
            name = expert_name(expert_id, projection)
            if projection == "down":
                shape = (4, 32)
                fill_value = expert_id + 5
            else:
                shape = (2, 64)
                fill_value = expert_id + (1 if projection == "gate" else 3)
            tensor = torch.full(shape, fill_value, requires_grad=True)
            params.append((name, tensor))
            expected[name] = tensor

    selected_names = set(expected)
    scratch_cache = {}
    output = dict(
        fp8_train_utils.iter_mxfp8_prequantized_params(
            iter(params),
            selected_names,
            quantize_fn=quantize,
            scratch_cache=scratch_cache,
        )
    )

    assert calls == [(4, 64), (4, 64), (8, 32)]
    assert output[params[0][0]] is params[0][1]
    for name, tensor in expected.items():
        torch.testing.assert_close(output[name], tensor)
        scale_name = name + "_scale_from_checkpoint"
        scale_columns = tensor.shape[-1] // MXFP8_BLOCK_SIZE
        expected_scale_shape = (
            tensor.shape[:-1]
            if scale_columns == 1
            else (*tensor.shape[:-1], scale_columns)
        )
        assert output[scale_name].shape == expected_scale_shape
        assert torch.all(output[scale_name] == 1)

    scratch = next(iter(scratch_cache.values()))
    first_scratch_ptr = scratch.data_ptr()
    calls.clear()
    list(
        fp8_train_utils.iter_mxfp8_prequantized_params(
            iter(params),
            selected_names,
            quantize_fn=quantize,
            scratch_cache=scratch_cache,
        )
    )
    assert calls == [(4, 64), (4, 64), (8, 32)]
    assert next(iter(scratch_cache.values())).data_ptr() == first_scratch_ptr


def test_batched_expert_prequantization_bounds_batch_and_has_stable_order():
    from nemo_rl.models.generation.vllm.quantization import fp8_train_utils

    calls = []

    def quantize(tensor):
        calls.append(tuple(tensor.shape))
        scales = torch.ones(
            (*tensor.shape[:-1], tensor.shape[-1] // MXFP8_BLOCK_SIZE),
            dtype=torch.uint8,
        )
        return tensor.clone(), scales

    def expert_name(expert_id, projection):
        return f"model.layers.0.mlp.experts.{expert_id}.{projection}_proj.weight"

    params = []
    for expert_id in range(5):
        for projection in ("gate", "up", "down"):
            shape = (4, 32) if projection == "down" else (2, 64)
            params.append((expert_name(expert_id, projection), torch.ones(*shape)))

    output = list(
        fp8_train_utils.iter_mxfp8_prequantized_params(
            iter(params),
            {name for name, _tensor in params},
            quantize_fn=quantize,
            max_experts_per_batch=2,
        )
    )

    expected_names = []
    for expert_ids, projection in (
        ((0, 1), "gate"),
        ((0, 1), "up"),
        ((0, 1), "down"),
        ((2, 3), "gate"),
        ((2, 3), "up"),
        ((2, 3), "down"),
        ((4,), "gate"),
        ((4,), "up"),
        ((4,), "down"),
    ):
        for expert_id in expert_ids:
            name = expert_name(expert_id, projection)
            expected_names.extend((name, name + "_scale_from_checkpoint"))

    assert [name for name, _tensor in output] == expected_names
    assert calls == [
        (4, 64),
        (4, 64),
        (8, 32),
        (4, 64),
        (4, 64),
        (8, 32),
        (2, 64),
        (2, 64),
        (4, 32),
    ]


def test_batched_expert_prequantization_matches_per_tensor_quantization():
    from nemo_rl.models.generation.vllm.quantization import fp8_train_utils

    torch.manual_seed(0)

    def expert_name(expert_id, projection):
        return f"model.layers.0.mlp.experts.{expert_id}.{projection}_proj.weight"

    params = []
    for expert_id in range(3):
        params.extend(
            [
                (expert_name(expert_id, "gate"), torch.randn(2, 64)),
                (expert_name(expert_id, "up"), torch.randn(2, 64)),
                (expert_name(expert_id, "down"), torch.randn(4, 32)),
            ]
        )

    output = dict(
        fp8_train_utils.iter_mxfp8_prequantized_params(
            params,
            {name for name, _tensor in params},
        )
    )

    for name, tensor in params:
        expected_value, expected_scale = fp8_train_utils.mxfp8_e4m3_quantize_for_refit(
            tensor
        )
        assert torch.equal(
            output[name].view(torch.uint8), expected_value.view(torch.uint8)
        )
        assert torch.equal(output[name + "_scale_from_checkpoint"], expected_scale)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_batched_expert_prequantization_uses_stream_local_scratch():
    from nemo_rl.models.generation.vllm.quantization import fp8_train_utils

    def expert_name(expert_id):
        return f"model.layers.0.mlp.experts.{expert_id}.gate_proj.weight"

    params = [(expert_name(i), torch.ones(2, 64, device="cuda")) for i in range(4)]
    scratch_cache = {}
    output = fp8_train_utils.iter_mxfp8_prequantized_params(
        params,
        {name for name, _tensor in params},
        quantize_fn=lambda tensor: (
            tensor.clone(),
            torch.ones((*tensor.shape[:-1], 2), dtype=torch.uint8, device="cuda"),
        ),
        scratch_cache=scratch_cache,
        max_experts_per_batch=2,
    )
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]

    with torch.cuda.stream(streams[0]):
        first_batch = [next(output) for _ in range(4)]
    with torch.cuda.stream(streams[1]):
        second_batch = [next(output) for _ in range(4)]

    assert len(first_batch) == len(second_batch) == 4
    assert len(scratch_cache) == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_refit_quantize_matches_receiver_quantize_mxfp8_weight():
    """Sender prequantization and the receiver helper must agree bit-for-bit.

    The trainer streams E4M3 data + *_scale_from_checkpoint produced by
    mxfp8_e4m3_quantize_for_refit; weights the receiver quantizes itself go
    through quantize_mxfp8_weight. Refit correctness relies on the two
    implementations producing identical bits for the same input.
    """
    from nemo_rl.models.generation.vllm.quantization.fp8 import quantize_mxfp8_weight

    torch.manual_seed(0)
    x = torch.randn(256, 512, dtype=torch.bfloat16, device="cuda")
    x[0].zero_()

    recv_lp, recv_scale = quantize_mxfp8_weight(x)
    sent_lp, sent_scale = mxfp8_e4m3_quantize_for_refit(x)

    assert sent_lp.dtype == recv_lp.dtype
    assert torch.equal(sent_lp.view(torch.uint8), recv_lp.view(torch.uint8))
    assert sent_scale.dtype == recv_scale.dtype
    assert sent_scale.shape == recv_scale.shape
    assert torch.equal(sent_scale, recv_scale)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_batched_expert_prequantization_waits_when_consumer_stream_changes():
    from nemo_rl.models.generation.vllm.quantization import fp8_train_utils

    def expert_name(expert_id):
        return f"model.layers.0.mlp.experts.{expert_id}.gate_proj.weight"

    params = [
        (expert_name(i), torch.full((2, 64), i + 1, device="cuda")) for i in range(2)
    ]

    def delayed_quantize(tensor):
        value = torch.empty_like(tensor)
        torch.cuda._sleep(5_000_000)
        value.copy_(tensor)
        scale = torch.ones((*tensor.shape[:-1], 2), dtype=torch.uint8, device="cuda")
        return value, scale

    output = fp8_train_utils.iter_mxfp8_prequantized_params(
        params,
        {name for name, _tensor in params},
        quantize_fn=delayed_quantize,
        max_experts_per_batch=2,
    )
    producer_stream = torch.cuda.Stream()
    consumer_stream = torch.cuda.Stream()

    with torch.cuda.stream(producer_stream):
        next(output)
        next(output)
    with torch.cuda.stream(consumer_stream):
        second_expert, _second_scale = next(output), next(output)
        observed = second_expert[1].clone()
    consumer_stream.synchronize()

    torch.testing.assert_close(observed, params[1][1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_prequantization_fallback_waits_when_consumer_stream_changes():
    from nemo_rl.models.generation.vllm.quantization import fp8_train_utils

    name = "model.layers.0.self_attn.q_proj.weight"
    tensor = torch.full((2, 64), 7, device="cuda")

    def delayed_quantize(input_tensor):
        value = torch.empty_like(input_tensor)
        torch.cuda._sleep(5_000_000)
        value.copy_(input_tensor)
        scale = torch.full(
            (*input_tensor.shape[:-1], 2), 3, dtype=torch.uint8, device="cuda"
        )
        return value, scale

    output = fp8_train_utils.iter_mxfp8_prequantized_params(
        [(name, tensor)],
        {name},
        quantize_fn=delayed_quantize,
    )
    producer_stream = torch.cuda.Stream()
    consumer_stream = torch.cuda.Stream()

    with torch.cuda.stream(producer_stream):
        next(output)
    with torch.cuda.stream(consumer_stream):
        scale_name, scale = next(output)
        observed = scale.clone()
    consumer_stream.synchronize()

    assert scale_name == name + "_scale_from_checkpoint"
    assert torch.all(observed == 3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("second_up_shape", [(2, 64), (3, 64)])
def test_batched_expert_prequantization_waits_for_pending_input_stream(
    second_up_shape,
):
    from nemo_rl.models.generation.vllm.quantization import fp8_train_utils

    def expert_name(expert_id, projection):
        return f"model.layers.0.mlp.experts.{expert_id}.{projection}_proj.weight"

    def params():
        yield expert_name(0, "gate"), torch.ones(2, 64, device="cuda")
        delayed_up = torch.empty(2, 64, device="cuda")
        torch.cuda._sleep(5_000_000)
        delayed_up.fill_(7)
        yield expert_name(0, "up"), delayed_up
        yield expert_name(0, "down"), torch.ones(4, 32, device="cuda")
        yield expert_name(1, "gate"), torch.full((2, 64), 2, device="cuda")
        yield expert_name(1, "up"), torch.full(second_up_shape, 3, device="cuda")
        yield expert_name(1, "down"), torch.full((4, 32), 4, device="cuda")

    selected_names = {
        expert_name(expert_id, projection)
        for expert_id in range(2)
        for projection in ("gate", "up", "down")
    }
    output = fp8_train_utils.iter_mxfp8_prequantized_params(
        params(),
        selected_names,
        quantize_fn=lambda input_tensor: (
            input_tensor.clone(),
            torch.ones(
                (*input_tensor.shape[:-1], input_tensor.shape[-1] // 32),
                dtype=torch.uint8,
                device="cuda",
            ),
        ),
        max_experts_per_batch=2,
    )
    producer_stream = torch.cuda.Stream()
    consumer_stream = torch.cuda.Stream()

    with torch.cuda.stream(producer_stream):
        gate_entries = [next(output) for _ in range(4)]
    with torch.cuda.stream(consumer_stream):
        up_name, up_tensor = next(output)
        observed = up_tensor.clone()
    consumer_stream.synchronize()

    assert len(gate_entries) == 4
    assert up_name == expert_name(0, "up")
    torch.testing.assert_close(observed, torch.full_like(observed, 7))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "is_gated,intermediate_size,hidden_size",
    [
        # Aligned: both scale K dims (hidden/32=8, intermediate/32=4) are %4.
        (True, 128, 256),
        # w2 scale K = 192/32 = 6, so pad_flashinfer_scale_k pads it to 8.
        (True, 192, 128),
        # Non-gated (single w13 shard), aligned.
        (False, 128, 256),
    ],
)
def test_batched_moe_shuffle_matches_per_expert(
    is_gated, intermediate_size, hidden_size
):
    """Bitwise parity of the batched TRTLLM MoE shuffle with the per-expert loop."""
    pytest.importorskip("flashinfer")
    fp8 = pytest.importorskip("nemo_rl.models.generation.vllm.quantization.fp8")

    from types import SimpleNamespace

    torch.manual_seed(0)
    num_experts = 4
    w13_rows = (2 if is_gated else 1) * intermediate_size

    def rand_bytes(*shape):
        return torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda")

    w13_weight = rand_bytes(num_experts, w13_rows, hidden_size).view(
        torch.float8_e4m3fn
    )
    w2_weight = rand_bytes(num_experts, hidden_size, intermediate_size).view(
        torch.float8_e4m3fn
    )
    w13_scale = rand_bytes(num_experts, w13_rows, hidden_size // MXFP8_BLOCK_SIZE)
    w2_scale = rand_bytes(
        num_experts, hidden_size, intermediate_size // MXFP8_BLOCK_SIZE
    )

    layer = SimpleNamespace()  # holds the cached row permutations
    epilogue_tile_m = 128
    batched = fp8._shuffle_mxfp8_moe_batched(
        layer, w13_weight, w2_weight, w13_scale, w2_scale, is_gated, epilogue_tile_m
    )
    reference = fp8._shuffle_mxfp8_moe_per_expert(
        w13_weight, w2_weight, w13_scale, w2_scale, is_gated, epilogue_tile_m
    )

    for got, want, name in zip(
        batched, reference, ("w13_weight", "w2_weight", "w13_scale", "w2_scale")
    ):
        assert got.shape == want.shape, name
        assert got.dtype == want.dtype, name
        assert torch.equal(got.view(torch.uint8), want.view(torch.uint8)), name
