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

"""Dense-oracle tests for DFlash structured block attention."""

from __future__ import annotations

import ast
import importlib
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch
from torch import Tensor


pytestmark = [
    pytest.mark.mcore,
    pytest.mark.filterwarnings(
        "error:flex_attention called without torch\\.compile",
    ),
]

_PLAN_MODULE = "nemo_rl.models.megatron.draft.block_plan"
_ATTENTION_MODULE = "nemo_rl.models.megatron.draft.block_attention"
_BENCHMARK_PATH = (
    Path(__file__).parents[4] / "tools/benchmark_dflash_block_attention.py"
)


@pytest.fixture
def _isolated_flex_compile_cache() -> Iterator[None]:
    torch.compiler.reset()
    yield
    torch.compiler.reset()


def _load_module(module_name: str) -> ModuleType:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        pytest.fail(
            f"DFlash production contract is missing: {error}",
            pytrace=False,
        )


def _load_benchmark() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "benchmark_dflash_block_attention",
        _BENCHMARK_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _small_benchmark_plan(benchmark: ModuleType) -> Any:
    return benchmark._build_plan(
        batch_size=1,
        sequence_length=32,
        anchors_per_sample=1,
        block_size=16,
        device=torch.device("cpu"),
    )


def test_benchmark_inputs_use_public_dflash_attention_geometry() -> None:
    """Catches benchmark-only head geometry that production kernels never see."""
    benchmark = _load_benchmark()
    inputs = benchmark._make_inputs(
        _small_benchmark_plan(benchmark),
        device=torch.device("cpu"),
    )

    assert [tuple(tensor.shape) for tensor in inputs] == [
        (1, 32, 8, 128),
        (1, 32, 8, 128),
        (1, 16, 32, 128),
        (1, 16, 8, 128),
        (1, 16, 8, 128),
    ]


def test_default_train_step_calls_production_block_only_attention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches default timing of unused trunk-query attention."""
    benchmark = _load_benchmark()
    plan = _small_benchmark_plan(benchmark)
    inputs = benchmark._make_inputs(plan, device=torch.device("cpu"))
    calls = 0

    def record_block_only_attention(**kwargs: Any) -> Tensor:
        nonlocal calls
        calls += 1
        return _load_module(_ATTENTION_MODULE).dflash_block_only_attention(**kwargs)

    def reject_full_attention(**_kwargs: Any) -> tuple[Tensor, Tensor]:
        raise AssertionError("the default benchmark must not call full attention")

    monkeypatch.setattr(
        benchmark,
        "dflash_block_only_attention",
        record_block_only_attention,
        raising=False,
    )
    monkeypatch.setattr(
        benchmark,
        "dflash_block_attention",
        reject_full_attention,
    )

    benchmark._train_step(plan, inputs)

    assert calls == 1
    assert all(tensor.grad is not None for tensor in inputs)


def test_default_correctness_calls_production_block_only_attention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches correctness evidence for a path the DFlash body does not execute."""
    benchmark = _load_benchmark()
    plan = _small_benchmark_plan(benchmark)
    calls = 0

    def record_block_only_attention(**kwargs: Any) -> Tensor:
        nonlocal calls
        calls += 1
        return _load_module(_ATTENTION_MODULE).dflash_block_only_attention(**kwargs)

    def reject_full_attention(**_kwargs: Any) -> tuple[Tensor, Tensor]:
        raise AssertionError("default correctness must not call full attention")

    def run_without_cuda_timing(_device: torch.device, operation: Any) -> float:
        operation()
        return 0.0

    monkeypatch.setattr(benchmark, "_build_plan", lambda **_kwargs: plan)
    monkeypatch.setattr(benchmark, "_time_call", run_without_cuda_timing)
    monkeypatch.setattr(
        benchmark,
        "dflash_block_only_attention",
        record_block_only_attention,
        raising=False,
    )
    monkeypatch.setattr(
        benchmark,
        "dflash_block_attention",
        reject_full_attention,
    )

    record = benchmark._correctness_comparison(
        device=torch.device("cpu"),
        iterations=1,
    )

    assert calls == 2
    assert record["kind"] == "block_only_forward_correctness_comparison"
    assert record["attention_path"] == "dflash_block_only_attention"


def _load_attention_contract() -> tuple[type[Any], Any]:
    plan_module = _load_module(_PLAN_MODULE)
    attention_module = _load_module(_ATTENTION_MODULE)
    return plan_module.DFlashBatchPlan, attention_module.dflash_block_attention


def _load_block_only_attention_contract() -> tuple[type[Any], Any]:
    plan_module = _load_module(_PLAN_MODULE)
    attention_module = _load_module(_ATTENTION_MODULE)
    return (
        plan_module.DFlashBatchPlan,
        attention_module.dflash_block_only_attention,
    )


def _make_plan(
    plan_type: type[Any],
    *,
    token_valid_mask: Tensor,
    sample_rows: list[int],
    anchor_positions: list[int],
    slot_valid: Tensor,
) -> Any:
    batch_size, sequence_length = token_valid_mask.shape
    num_blocks, block_size = slot_valid.shape
    assert num_blocks == len(sample_rows) == len(anchor_positions)
    assert num_blocks % batch_size == 0
    device = token_valid_mask.device
    sample_rows_tensor = torch.tensor(
        sample_rows,
        dtype=torch.int64,
        device=device,
    )
    anchors_tensor = torch.tensor(
        anchor_positions,
        dtype=torch.int64,
        device=device,
    )
    sequence_positions = torch.arange(sequence_length, device=device)
    trunk_lengths = (
        token_valid_mask[sample_rows_tensor]
        & (sequence_positions.unsqueeze(0) < anchors_tensor.unsqueeze(1))
    ).sum(dim=1)
    safe_positions = torch.clamp(
        anchors_tensor.unsqueeze(1) + torch.arange(block_size, device=device),
        min=0,
        max=sequence_length - 1,
    )
    loss_mask = slot_valid.clone()
    loss_mask[:, 0] = False
    return plan_type(
        token_valid_mask=token_valid_mask,
        sample_rows=sample_rows_tensor,
        anchor_ids=torch.arange(num_blocks, dtype=torch.int64, device=device),
        anchor_positions=anchors_tensor,
        trunk_lengths=trunk_lengths,
        query_positions=safe_positions,
        label_positions=safe_positions.clone(),
        block_valid=slot_valid.any(dim=1),
        slot_valid=slot_valid,
        loss_mask=loss_mask,
        batch_size=batch_size,
        sequence_length=sequence_length,
        anchors_per_sample=num_blocks // batch_size,
        gamma=block_size - 1,
        block_size=block_size,
    )


def _dense_attention_oracle(
    *,
    plan: Any,
    trunk_q: Tensor,
    trunk_k: Tensor,
    trunk_v: Tensor,
    block_q: Tensor,
    block_k: Tensor,
    block_v: Tensor,
    scale: float | None = None,
) -> tuple[Tensor, Tensor]:
    """Independent scalar-loop implementation of the written visibility rules."""
    batch_size, sequence_length, num_query_heads, head_dim = trunk_q.shape
    num_kv_heads = trunk_k.shape[2]
    heads_per_group = num_query_heads // num_kv_heads
    effective_scale = head_dim**-0.5 if scale is None else scale
    trunk_output = torch.zeros_like(trunk_q)
    block_output = torch.zeros_like(block_q)

    for batch_index in range(batch_size):
        for query_position in range(sequence_length):
            if not bool(plan.token_valid_mask[batch_index, query_position]):
                continue
            visible_positions = [
                key_position
                for key_position in range(query_position + 1)
                if bool(plan.token_valid_mask[batch_index, key_position])
            ]
            for query_head in range(num_query_heads):
                kv_head = query_head // heads_per_group
                query = trunk_q[batch_index, query_position, query_head]
                keys = torch.stack(
                    [
                        trunk_k[batch_index, key_position, kv_head]
                        for key_position in visible_positions
                    ]
                )
                values = torch.stack(
                    [
                        trunk_v[batch_index, key_position, kv_head]
                        for key_position in visible_positions
                    ]
                )
                probabilities = torch.softmax(
                    torch.mv(keys, query) * effective_scale,
                    dim=0,
                )
                trunk_output[batch_index, query_position, query_head] = (
                    probabilities.unsqueeze(0) @ values
                ).squeeze(0)

    num_blocks, block_size = plan.slot_valid.shape
    for block_index in range(num_blocks):
        sample_row = int(plan.sample_rows[block_index])
        anchor_position = int(plan.anchor_positions[block_index])
        visible_trunk_positions = [
            position
            for position in range(sequence_length)
            if position < anchor_position
            and bool(plan.token_valid_mask[sample_row, position])
        ]
        visible_block_positions = [
            position
            for position in range(block_size)
            if bool(plan.slot_valid[block_index, position])
        ]
        for query_position in range(block_size):
            if not bool(plan.slot_valid[block_index, query_position]):
                continue
            for query_head in range(num_query_heads):
                kv_head = query_head // heads_per_group
                query = block_q[block_index, query_position, query_head]
                keys = [
                    trunk_k[sample_row, position, kv_head]
                    for position in visible_trunk_positions
                ] + [
                    block_k[block_index, position, kv_head]
                    for position in visible_block_positions
                ]
                values = [
                    trunk_v[sample_row, position, kv_head]
                    for position in visible_trunk_positions
                ] + [
                    block_v[block_index, position, kv_head]
                    for position in visible_block_positions
                ]
                stacked_keys = torch.stack(keys)
                stacked_values = torch.stack(values)
                probabilities = torch.softmax(
                    torch.mv(stacked_keys, query) * effective_scale,
                    dim=0,
                )
                block_output[block_index, query_position, query_head] = (
                    probabilities.unsqueeze(0) @ stacked_values
                ).squeeze(0)

    return trunk_output, block_output


def _clone_with_grad(tensors: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
    return tuple(tensor.detach().clone().requires_grad_(True) for tensor in tensors)


def _random_attention_inputs(
    *,
    batch_size: int,
    sequence_length: int,
    num_blocks: int,
    block_size: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> tuple[Tensor, ...]:
    generator = torch.Generator(device=device).manual_seed(seed)
    return (
        torch.randn(
            (batch_size, sequence_length, num_query_heads, head_dim),
            generator=generator,
            device=device,
            dtype=dtype,
        ),
        torch.randn(
            (batch_size, sequence_length, num_kv_heads, head_dim),
            generator=generator,
            device=device,
            dtype=dtype,
        ),
        torch.randn(
            (batch_size, sequence_length, num_kv_heads, head_dim),
            generator=generator,
            device=device,
            dtype=dtype,
        ),
        torch.randn(
            (num_blocks, block_size, num_query_heads, head_dim),
            generator=generator,
            device=device,
            dtype=dtype,
        ),
        torch.randn(
            (num_blocks, block_size, num_kv_heads, head_dim),
            generator=generator,
            device=device,
            dtype=dtype,
        ),
        torch.randn(
            (num_blocks, block_size, num_kv_heads, head_dim),
            generator=generator,
            device=device,
            dtype=dtype,
        ),
    )


@pytest.mark.parametrize(
    "num_query_heads,num_kv_heads",
    [
        pytest.param(2, 2, id="mha"),
        pytest.param(4, 2, id="gqa"),
    ],
)
def test_dense_fp32_forward_and_qkv_gradient_parity(
    num_query_heads: int,
    num_kv_heads: int,
) -> None:
    """Catches wrong visibility, GQA head mapping, scaling, or backward math."""
    plan_type, attention = _load_attention_contract()
    token_valid_mask = torch.tensor(
        [
            [True, True, True, True],
            [True, True, True, False],
        ]
    )
    plan = _make_plan(
        plan_type,
        token_valid_mask=token_valid_mask,
        sample_rows=[0, 0, 0, 1],
        anchor_positions=[0, 3, 4, 2],
        slot_valid=torch.tensor(
            [
                [True, True, True],
                [True, True, True],
                [True, True, True],
                [True, True, False],
            ]
        ),
    )
    generator = torch.Generator().manual_seed(1234)
    head_dim = 3
    tensors = (
        torch.randn((2, 4, num_query_heads, head_dim), generator=generator),
        torch.randn((2, 4, num_kv_heads, head_dim), generator=generator),
        torch.randn((2, 4, num_kv_heads, head_dim), generator=generator),
        torch.randn((4, 3, num_query_heads, head_dim), generator=generator),
        torch.randn((4, 3, num_kv_heads, head_dim), generator=generator),
        torch.randn((4, 3, num_kv_heads, head_dim), generator=generator),
    )
    production_inputs = _clone_with_grad(tensors)
    oracle_inputs = _clone_with_grad(tensors)

    production_outputs = attention(
        plan=plan,
        trunk_q=production_inputs[0],
        trunk_k=production_inputs[1],
        trunk_v=production_inputs[2],
        block_q=production_inputs[3],
        block_k=production_inputs[4],
        block_v=production_inputs[5],
    )
    oracle_outputs = _dense_attention_oracle(
        plan=plan,
        trunk_q=oracle_inputs[0],
        trunk_k=oracle_inputs[1],
        trunk_v=oracle_inputs[2],
        block_q=oracle_inputs[3],
        block_k=oracle_inputs[4],
        block_v=oracle_inputs[5],
    )

    assert len(production_outputs) == 2
    torch.testing.assert_close(production_outputs[0], oracle_outputs[0])
    torch.testing.assert_close(production_outputs[1], oracle_outputs[1])

    trunk_weight = torch.randn(production_outputs[0].shape, generator=generator)
    block_weight = torch.randn(production_outputs[1].shape, generator=generator)
    production_loss = (production_outputs[0] * trunk_weight).sum() + (
        production_outputs[1] * block_weight
    ).sum()
    oracle_loss = (oracle_outputs[0] * trunk_weight).sum() + (
        oracle_outputs[1] * block_weight
    ).sum()
    production_gradients = torch.autograd.grad(production_loss, production_inputs)
    oracle_gradients = torch.autograd.grad(oracle_loss, oracle_inputs)

    for production_gradient, oracle_gradient in zip(
        production_gradients,
        oracle_gradients,
        strict=True,
    ):
        torch.testing.assert_close(
            production_gradient,
            oracle_gradient,
            atol=2e-5,
            rtol=2e-5,
        )


@pytest.mark.parametrize(
    "num_query_heads,num_kv_heads",
    [
        pytest.param(2, 2, id="mha"),
        pytest.param(4, 2, id="gqa"),
    ],
)
def test_block_only_fp32_forward_and_gradient_parity(
    num_query_heads: int,
    num_kv_heads: int,
) -> None:
    plan_type, attention = _load_block_only_attention_contract()
    plan = _make_plan(
        plan_type,
        token_valid_mask=torch.tensor(
            [[True, True, True, True], [True, True, True, False]]
        ),
        sample_rows=[0, 0, 1, 1],
        anchor_positions=[0, 3, 2, 3],
        slot_valid=torch.tensor(
            [
                [True, True, True],
                [True, True, True],
                [True, True, False],
                [True, False, False],
            ]
        ),
    )
    tensors = _random_attention_inputs(
        batch_size=2,
        sequence_length=4,
        num_blocks=4,
        block_size=3,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
        seed=909,
    )
    production_inputs = _clone_with_grad(tensors[1:])
    oracle_inputs = _clone_with_grad(tensors)

    actual = attention(
        plan=plan,
        trunk_k=production_inputs[0],
        trunk_v=production_inputs[1],
        block_q=production_inputs[2],
        block_k=production_inputs[3],
        block_v=production_inputs[4],
    )
    expected = _dense_attention_oracle(
        plan=plan,
        trunk_q=oracle_inputs[0],
        trunk_k=oracle_inputs[1],
        trunk_v=oracle_inputs[2],
        block_q=oracle_inputs[3],
        block_k=oracle_inputs[4],
        block_v=oracle_inputs[5],
    )[1]
    torch.testing.assert_close(actual, expected)

    weight = torch.randn_like(actual)
    actual_gradients = torch.autograd.grad((actual * weight).sum(), production_inputs)
    expected_gradients = torch.autograd.grad(
        (expected * weight).sum(),
        oracle_inputs[1:],
    )
    for actual_gradient, expected_gradient in zip(
        actual_gradients,
        expected_gradients,
        strict=True,
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient)


def test_block_queries_cover_empty_remainder_and_full_trunk_boundaries() -> None:
    """Catches inclusion of the anchor or exclusion of valid prefix boundaries."""
    plan_type, attention = _load_attention_contract()
    plan = _make_plan(
        plan_type,
        token_valid_mask=torch.ones((1, 4), dtype=torch.bool),
        sample_rows=[0, 0, 0],
        anchor_positions=[0, 2, 4],
        slot_valid=torch.ones((3, 2), dtype=torch.bool),
    )
    trunk_q = torch.zeros((1, 4, 1, 1))
    trunk_k = torch.zeros((1, 4, 1, 1))
    trunk_v = torch.tensor([1.0, 2.0, 3.0, 4.0]).reshape(1, 4, 1, 1)
    block_q = torch.zeros((3, 2, 1, 1))
    block_k = torch.zeros((3, 2, 1, 1))
    block_v = torch.tensor(
        [
            [[[10.0]], [[20.0]]],
            [[[10.0]], [[20.0]]],
            [[[10.0]], [[20.0]]],
        ]
    )

    _, block_output = attention(
        plan=plan,
        trunk_q=trunk_q,
        trunk_k=trunk_k,
        trunk_v=trunk_v,
        block_q=block_q,
        block_k=block_k,
        block_v=block_v,
    )

    expected = torch.tensor([15.0, 8.25, 40.0 / 6.0]).reshape(3, 1, 1, 1)
    torch.testing.assert_close(block_output, expected.expand(-1, 2, -1, -1))


def test_duplicate_anchors_and_multiple_rows_remain_block_local() -> None:
    """Catches cross-block and cross-sample K/V leakage."""
    plan_type, attention = _load_attention_contract()
    plan = _make_plan(
        plan_type,
        token_valid_mask=torch.ones((2, 2), dtype=torch.bool),
        sample_rows=[0, 0, 1, 1],
        anchor_positions=[1, 1, 2, 2],
        slot_valid=torch.ones((4, 2), dtype=torch.bool),
    )
    trunk_q = torch.zeros((2, 2, 1, 1))
    trunk_k = torch.zeros((2, 2, 1, 1))
    trunk_v = torch.tensor([1.0, 3.0, 100.0, 300.0]).reshape(2, 2, 1, 1)
    block_q = torch.zeros((4, 2, 1, 1))
    block_k = torch.zeros((4, 2, 1, 1))
    block_v = torch.tensor(
        [
            [[[5.0]], [[7.0]]],
            [[[50.0]], [[70.0]]],
            [[[500.0]], [[700.0]]],
            [[[5000.0]], [[7000.0]]],
        ]
    )

    _, baseline = attention(
        plan=plan,
        trunk_q=trunk_q,
        trunk_k=trunk_k,
        trunk_v=trunk_v,
        block_q=block_q,
        block_k=block_k,
        block_v=block_v,
    )
    changed_block_v = block_v.clone()
    changed_block_v[1:] = 1_000_000.0
    _, changed = attention(
        plan=plan,
        trunk_q=trunk_q,
        trunk_k=trunk_k,
        trunk_v=trunk_v,
        block_q=block_q,
        block_k=block_k,
        block_v=changed_block_v,
    )

    torch.testing.assert_close(baseline[0], torch.full_like(baseline[0], 13.0 / 3.0))
    torch.testing.assert_close(changed[0], baseline[0])
    assert not torch.equal(baseline[1], baseline[0])
    assert not torch.equal(baseline[2], baseline[0])


def test_invalid_block_queries_are_zero_with_finite_isolated_gradients() -> None:
    """Catches NaNs or gradient flow through masked block queries and K/V slots."""
    plan_type, attention = _load_attention_contract()
    plan = _make_plan(
        plan_type,
        token_valid_mask=torch.ones((1, 2), dtype=torch.bool),
        sample_rows=[0],
        anchor_positions=[1],
        slot_valid=torch.tensor([[True, False, True]]),
    )
    generator = torch.Generator().manual_seed(77)
    inputs = _clone_with_grad(
        (
            torch.randn((1, 2, 2, 3), generator=generator),
            torch.randn((1, 2, 1, 3), generator=generator),
            torch.randn((1, 2, 1, 3), generator=generator),
            torch.randn((1, 3, 2, 3), generator=generator),
            torch.randn((1, 3, 1, 3), generator=generator),
            torch.randn((1, 3, 1, 3), generator=generator),
        )
    )

    trunk_output, block_output = attention(
        plan=plan,
        trunk_q=inputs[0],
        trunk_k=inputs[1],
        trunk_v=inputs[2],
        block_q=inputs[3],
        block_k=inputs[4],
        block_v=inputs[5],
    )
    loss = trunk_output.square().sum() + block_output.square().sum()
    gradients = torch.autograd.grad(loss, inputs)

    assert torch.equal(block_output[:, 1], torch.zeros_like(block_output[:, 1]))
    assert torch.isfinite(trunk_output).all()
    assert torch.isfinite(block_output).all()
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert torch.equal(gradients[3][:, 1], torch.zeros_like(gradients[3][:, 1]))
    assert torch.equal(gradients[4][:, 1], torch.zeros_like(gradients[4][:, 1]))
    assert torch.equal(gradients[5][:, 1], torch.zeros_like(gradients[5][:, 1]))


def test_cuda_implementation_uses_only_public_flex_attention_apis() -> None:
    """Catches private FlashAttention or private PyTorch dependencies."""
    source_path = (
        Path(__file__).parents[4] / "nemo_rl/models/megatron/draft/block_attention.py"
    )
    source = source_path.read_text()
    tree = ast.parse(source)

    imported_symbols = {
        (node.module, alias.name)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert (
        "torch.nn.attention.flex_attention",
        "BlockMask",
    ) in imported_symbols
    assert (
        "torch.nn.attention.flex_attention",
        "flex_attention",
    ) in imported_symbols
    assert (
        "torch.nn.attention.flex_attention",
        "create_block_mask",
    ) not in imported_symbols
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "BlockMask"
        and node.func.attr == "from_kv_blocks"
        for node in ast.walk(tree)
    )

    forbidden_import_prefixes = (
        "flash_attn",
        "transformer_engine.pytorch.attention",
        "torch._",
    )
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported_modules.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert not any(
        module_name.startswith(forbidden_import_prefixes)
        for module_name in imported_modules
    )
    assert not any(
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "torch"
        and node.attr.startswith("_")
        for node in ast.walk(tree)
    )


def test_invalid_attention_shapes_fail_before_computation() -> None:
    """Catches silent broadcasting of malformed Q/K/V layouts."""
    plan_type, attention = _load_attention_contract()
    plan = _make_plan(
        plan_type,
        token_valid_mask=torch.ones((1, 4), dtype=torch.bool),
        sample_rows=[0],
        anchor_positions=[2],
        slot_valid=torch.ones((1, 3), dtype=torch.bool),
    )
    inputs = _random_attention_inputs(
        batch_size=1,
        sequence_length=4,
        num_blocks=1,
        block_size=3,
        num_query_heads=4,
        num_kv_heads=2,
        head_dim=8,
        device=torch.device("cpu"),
        dtype=torch.float32,
        seed=91,
    )

    with pytest.raises(ValueError, match="block_q shape"):
        attention(
            plan=plan,
            trunk_q=inputs[0],
            trunk_k=inputs[1],
            trunk_v=inputs[2],
            block_q=inputs[3][:, :-1],
            block_k=inputs[4],
            block_v=inputs[5],
        )

    with pytest.raises(ValueError, match="divisible"):
        attention(
            plan=plan,
            trunk_q=inputs[0][:, :, :3],
            trunk_k=inputs[1],
            trunk_v=inputs[2],
            block_q=inputs[3][:, :, :3],
            block_k=inputs[4],
            block_v=inputs[5],
        )

    with pytest.raises(ValueError, match="block value shapes"):
        attention(
            plan=plan,
            trunk_q=inputs[0],
            trunk_k=inputs[1],
            trunk_v=inputs[2],
            block_q=inputs[3],
            block_k=inputs[4],
            block_v=inputs[5][..., :-1],
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.usefixtures("_isolated_flex_compile_cache")
@pytest.mark.parametrize(
    "dtype,num_query_heads,num_kv_heads,scale,tolerance",
    [
        pytest.param(torch.float32, 2, 2, None, 2e-3, id="fp32-mha-default-scale"),
        pytest.param(torch.float32, 4, 2, 0.37, 2.75e-3, id="fp32-gqa-custom-scale"),
        pytest.param(torch.bfloat16, 2, 2, 0.37, 5e-2, id="bf16-mha-custom-scale"),
        pytest.param(torch.bfloat16, 4, 2, None, 5e-2, id="bf16-gqa-default-scale"),
    ],
)
def test_cuda_forward_and_all_qkv_gradients_match_dense_oracle(
    dtype: torch.dtype,
    num_query_heads: int,
    num_kv_heads: int,
    scale: float | None,
    tolerance: float,
) -> None:
    """Catches CUDA FlexAttention visibility, scaling, GQA, or backward drift."""
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support bfloat16")

    plan_type, attention = _load_attention_contract()
    device = torch.device("cuda")
    plan = _make_plan(
        plan_type,
        token_valid_mask=torch.tensor(
            [[True, True, True, True], [True, True, True, False]],
            device=device,
        ),
        sample_rows=[0, 0, 0, 1],
        anchor_positions=[0, 3, 4, 2],
        slot_valid=torch.tensor(
            [
                [True, True, True],
                [True, True, True],
                [True, True, True],
                [True, True, False],
            ],
            device=device,
        ),
    )
    tensors = _random_attention_inputs(
        batch_size=2,
        sequence_length=4,
        num_blocks=4,
        block_size=3,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=16,
        device=device,
        dtype=dtype,
        seed=2026,
    )
    production_inputs = _clone_with_grad(tensors)
    oracle_inputs = _clone_with_grad(tensors)

    production_outputs = attention(
        plan=plan,
        trunk_q=production_inputs[0],
        trunk_k=production_inputs[1],
        trunk_v=production_inputs[2],
        block_q=production_inputs[3],
        block_k=production_inputs[4],
        block_v=production_inputs[5],
        scale=scale,
    )
    oracle_outputs = _dense_attention_oracle(
        plan=plan,
        trunk_q=oracle_inputs[0],
        trunk_k=oracle_inputs[1],
        trunk_v=oracle_inputs[2],
        block_q=oracle_inputs[3],
        block_k=oracle_inputs[4],
        block_v=oracle_inputs[5],
        scale=scale,
    )
    for production_output, oracle_output in zip(
        production_outputs,
        oracle_outputs,
        strict=True,
    ):
        torch.testing.assert_close(
            production_output,
            oracle_output,
            atol=tolerance,
            rtol=tolerance,
        )

    generator = torch.Generator(device=device).manual_seed(8128)
    trunk_weight = torch.randn(
        production_outputs[0].shape,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    block_weight = torch.randn(
        production_outputs[1].shape,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    production_loss = (production_outputs[0] * trunk_weight).sum() + (
        production_outputs[1] * block_weight
    ).sum()
    oracle_loss = (oracle_outputs[0] * trunk_weight).sum() + (
        oracle_outputs[1] * block_weight
    ).sum()
    production_gradients = torch.autograd.grad(production_loss, production_inputs)
    oracle_gradients = torch.autograd.grad(oracle_loss, oracle_inputs)
    for production_gradient, oracle_gradient in zip(
        production_gradients,
        oracle_gradients,
        strict=True,
    ):
        torch.testing.assert_close(
            production_gradient,
            oracle_gradient,
            atol=tolerance,
            rtol=tolerance,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.usefixtures("_isolated_flex_compile_cache")
def test_cuda_public_geometry_block_only_forward_and_gradients_match_dense() -> None:
    """Catches BF16 drift at the public 32Q/8KV/128D/block-16 geometry."""
    if not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support bfloat16")

    plan_type, attention = _load_block_only_attention_contract()
    device = torch.device("cuda")
    plan = _make_plan(
        plan_type,
        token_valid_mask=torch.ones((1, 32), dtype=torch.bool, device=device),
        sample_rows=[0],
        anchor_positions=[24],
        slot_valid=torch.ones((1, 16), dtype=torch.bool, device=device),
    )
    tensors = _random_attention_inputs(
        batch_size=1,
        sequence_length=32,
        num_blocks=1,
        block_size=16,
        num_query_heads=32,
        num_kv_heads=8,
        head_dim=128,
        device=device,
        dtype=torch.bfloat16,
        seed=32128,
    )
    production_inputs = _clone_with_grad(tensors[1:])
    oracle_inputs = _clone_with_grad(tensors)

    actual = attention(
        plan=plan,
        trunk_k=production_inputs[0],
        trunk_v=production_inputs[1],
        block_q=production_inputs[2],
        block_k=production_inputs[3],
        block_v=production_inputs[4],
    )
    expected = _dense_attention_oracle(
        plan=plan,
        trunk_q=oracle_inputs[0],
        trunk_k=oracle_inputs[1],
        trunk_v=oracle_inputs[2],
        block_q=oracle_inputs[3],
        block_k=oracle_inputs[4],
        block_v=oracle_inputs[5],
    )[1]
    torch.testing.assert_close(actual, expected, atol=5e-2, rtol=5e-2)

    generator = torch.Generator(device=device).manual_seed(16128)
    weight = torch.randn(
        actual.shape,
        dtype=actual.dtype,
        device=device,
        generator=generator,
    )
    actual_gradients = torch.autograd.grad((actual * weight).sum(), production_inputs)
    expected_gradients = torch.autograd.grad(
        (expected * weight).sum(),
        oracle_inputs[1:],
    )
    for actual_gradient, expected_gradient in zip(
        actual_gradients,
        expected_gradients,
        strict=True,
    ):
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            atol=5e-2,
            rtol=5e-2,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.usefixtures("_isolated_flex_compile_cache")
def test_cuda_holes_and_all_invalid_rows_are_finite_and_gradient_isolated() -> None:
    """Catches fully masked-row NaNs and gradients through invalid Q/K/V entries."""
    plan_type, attention = _load_attention_contract()
    device = torch.device("cuda")
    plan = _make_plan(
        plan_type,
        token_valid_mask=torch.tensor(
            [[True, False, True, True], [False, False, False, False]],
            device=device,
        ),
        sample_rows=[0, 0, 1, 1],
        anchor_positions=[0, 3, 0, 0],
        slot_valid=torch.tensor(
            [
                [True, False, True],
                [True, True, False],
                [False, False, False],
                [False, False, False],
            ],
            device=device,
        ),
    )
    inputs = _clone_with_grad(
        _random_attention_inputs(
            batch_size=2,
            sequence_length=4,
            num_blocks=4,
            block_size=3,
            num_query_heads=4,
            num_kv_heads=2,
            head_dim=16,
            device=device,
            dtype=torch.float32,
            seed=314,
        )
    )

    trunk_output, block_output = attention(
        plan=plan,
        trunk_q=inputs[0],
        trunk_k=inputs[1],
        trunk_v=inputs[2],
        block_q=inputs[3],
        block_k=inputs[4],
        block_v=inputs[5],
    )
    gradients = torch.autograd.grad(
        trunk_output.square().sum() + block_output.square().sum(),
        inputs,
    )

    assert torch.isfinite(trunk_output).all()
    assert torch.isfinite(block_output).all()
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert torch.equal(
        trunk_output[~plan.token_valid_mask],
        torch.zeros_like(trunk_output[~plan.token_valid_mask]),
    )
    assert torch.equal(
        block_output[~plan.slot_valid],
        torch.zeros_like(block_output[~plan.slot_valid]),
    )
    for gradient in gradients[:3]:
        assert torch.equal(
            gradient[~plan.token_valid_mask],
            torch.zeros_like(gradient[~plan.token_valid_mask]),
        )
    for gradient in gradients[3:]:
        assert torch.equal(
            gradient[~plan.slot_valid],
            torch.zeros_like(gradient[~plan.slot_valid]),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.usefixtures("_isolated_flex_compile_cache")
def test_cuda_duplicate_anchors_do_not_share_block_values() -> None:
    """Catches flattened global K/V masks that leak between duplicate anchors."""
    plan_type, attention = _load_attention_contract()
    device = torch.device("cuda")
    plan = _make_plan(
        plan_type,
        token_valid_mask=torch.ones((1, 3), dtype=torch.bool, device=device),
        sample_rows=[0, 0],
        anchor_positions=[2, 2],
        slot_valid=torch.ones((2, 2), dtype=torch.bool, device=device),
    )
    trunk_q = torch.zeros((1, 3, 1, 16), device=device)
    trunk_k = torch.zeros_like(trunk_q)
    trunk_v = torch.arange(3.0, device=device).reshape(1, 3, 1, 1).expand_as(trunk_q)
    block_q = torch.zeros((2, 2, 1, 16), device=device)
    block_k = torch.zeros_like(block_q)
    block_v = (
        torch.tensor([10.0, 20.0, 100.0, 200.0], device=device)
        .reshape(2, 2, 1, 1)
        .expand_as(block_q)
    )

    _, baseline = attention(
        plan=plan,
        trunk_q=trunk_q,
        trunk_k=trunk_k,
        trunk_v=trunk_v,
        block_q=block_q,
        block_k=block_k,
        block_v=block_v,
    )
    changed_values = block_v.clone()
    changed_values[1] = 1_000_000.0
    _, changed = attention(
        plan=plan,
        trunk_q=trunk_q,
        trunk_k=trunk_k,
        trunk_v=trunk_v,
        block_q=block_q,
        block_k=block_k,
        block_v=changed_values,
    )

    torch.testing.assert_close(changed[0], baseline[0])
    assert not torch.equal(changed[1], baseline[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_flex_calls_keep_one_global_trunk_kv_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches per-anchor trunk K/V or token-dense mask metadata."""
    plan_type, _ = _load_attention_contract()
    attention_module = _load_module(_ATTENTION_MODULE)
    device = torch.device("cuda")
    sequence_length = 2048
    plan = _make_plan(
        plan_type,
        token_valid_mask=torch.ones(
            (2, sequence_length),
            dtype=torch.bool,
            device=device,
        ),
        sample_rows=[0, 0, 0, 1, 1, 1],
        anchor_positions=[1, 512, 1536, 2, 768, 1792],
        slot_valid=torch.ones((6, 3), dtype=torch.bool, device=device),
    )
    inputs = _random_attention_inputs(
        batch_size=2,
        sequence_length=sequence_length,
        num_blocks=6,
        block_size=3,
        num_query_heads=4,
        num_kv_heads=2,
        head_dim=16,
        device=device,
        dtype=torch.float32,
        seed=2718,
    )
    calls: list[tuple[Tensor, Tensor, Tensor, Any]] = []

    def record_flex_call(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        **kwargs: Any,
    ) -> Tensor:
        calls.append((query, key, value, kwargs["block_mask"]))
        return torch.zeros_like(query)

    monkeypatch.setattr(
        attention_module,
        "_COMPILED_FLEX_ATTENTION",
        record_flex_call,
        raising=False,
    )
    attention_module.dflash_block_attention(
        plan=plan,
        trunk_q=inputs[0],
        trunk_k=inputs[1],
        trunk_v=inputs[2],
        block_q=inputs[3],
        block_k=inputs[4],
        block_v=inputs[5],
    )

    assert len(calls) == 2
    global_kv_length = 2 * sequence_length + 6 * 3
    assert calls[0][1].shape == (2, 2, sequence_length, 16)
    assert calls[1][1].shape == (1, 2, global_kv_length, 16)
    assert calls[1][2].shape == (1, 2, global_kv_length, 16)
    expected_storage_bytes = (inputs[1].numel() + inputs[4].numel()) * inputs[
        1
    ].element_size()
    assert calls[1][1].untyped_storage().nbytes() == expected_storage_bytes
    assert calls[1][2].untyped_storage().nbytes() == expected_storage_bytes

    trunk_mask = calls[0][3]
    block_mask = calls[1][3]
    assert trunk_mask.BLOCK_SIZE == (128, 128)
    assert block_mask.BLOCK_SIZE == (128, 128)
    assert trunk_mask.kv_indices.shape == (2, 1, 16, 16)
    assert block_mask.kv_indices.shape == (6, 1, 1, 19)
    sparse_metadata_elements = sum(
        tensor.numel()
        for mask in (trunk_mask, block_mask)
        for tensor in (
            mask.kv_num_blocks,
            mask.kv_indices,
            mask.q_num_blocks,
            mask.q_indices,
        )
    )
    token_dense_elements = (
        2 * sequence_length * sequence_length + 6 * 3 * global_kv_length
    )
    assert sparse_metadata_elements * 100 < token_dense_elements


def test_global_block_mask_candidates_are_prefix_bounded() -> None:
    plan_type, _ = _load_block_only_attention_contract()
    attention_module = _load_module(_ATTENTION_MODULE)
    batch_size = 8
    sequence_length = 512
    block_size = 3
    sample_rows = [row for row in range(batch_size) for _ in range(2)]
    anchor_positions = [anchor for _ in range(batch_size) for anchor in (1, 129)]
    plan = _make_plan(
        plan_type,
        token_valid_mask=torch.ones(
            (batch_size, sequence_length),
            dtype=torch.bool,
        ),
        sample_rows=sample_rows,
        anchor_positions=anchor_positions,
        slot_valid=torch.ones((len(sample_rows), block_size), dtype=torch.bool),
    )

    block_mask = attention_module._create_global_block_mask(plan)

    max_sample_prefix_blocks = (
        sequence_length + 2 * block_mask.BLOCK_SIZE[1] - 2
    ) // block_mask.BLOCK_SIZE[1]
    max_own_block_blocks = (
        block_size + 2 * block_mask.BLOCK_SIZE[1] - 2
    ) // block_mask.BLOCK_SIZE[1]
    assert block_mask.kv_indices.shape[-1] <= (
        max_sample_prefix_blocks + max_own_block_blocks
    )

    first_count = int(block_mask.kv_num_blocks[0, 0, 0])
    first_candidates = set(block_mask.kv_indices[0, 0, 0, :first_count].tolist())
    assert first_candidates == {0, 32}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.usefixtures("_isolated_flex_compile_cache")
@pytest.mark.parametrize(
    "sequence_length,peak_limit_bytes",
    [
        pytest.param(8_192, 128 * 1024**2, id="8k"),
        pytest.param(32_768, 256 * 1024**2, id="32k"),
    ],
)
def test_cuda_block_only_attention_has_bounded_long_context_memory(
    sequence_length: int,
    peak_limit_bytes: int,
) -> None:
    plan_type, attention = _load_block_only_attention_contract()
    device = torch.device("cuda")
    plan = _make_plan(
        plan_type,
        token_valid_mask=torch.ones(
            (1, sequence_length),
            dtype=torch.bool,
            device=device,
        ),
        sample_rows=[0],
        anchor_positions=[sequence_length],
        slot_valid=torch.ones((1, 3), dtype=torch.bool, device=device),
    )
    generator = torch.Generator(device=device).manual_seed(1776)
    trunk_k = torch.randn(
        (1, sequence_length, 1, 16),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    trunk_v = torch.randn_like(trunk_k)
    block_q = torch.randn(
        (1, 3, 2, 16),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    block_k = torch.randn(
        (1, 3, 1, 16),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    block_v = torch.randn_like(block_k)

    torch.cuda.reset_peak_memory_stats()
    output = attention(
        plan=plan,
        trunk_k=trunk_k,
        trunk_v=trunk_v,
        block_q=block_q,
        block_k=block_k,
        block_v=block_v,
    )
    torch.cuda.synchronize()

    assert output.shape == block_q.shape
    assert torch.isfinite(output).all()
    assert torch.cuda.max_memory_allocated() < peak_limit_bytes


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_block_only_256k_storage_and_metadata_are_linear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_type, _ = _load_block_only_attention_contract()
    attention_module = _load_module(_ATTENTION_MODULE)
    device = torch.device("cuda")
    sequence_length = 262_144
    plan = _make_plan(
        plan_type,
        token_valid_mask=torch.ones(
            (1, sequence_length),
            dtype=torch.bool,
            device=device,
        ),
        sample_rows=[0],
        anchor_positions=[sequence_length],
        slot_valid=torch.ones((1, 3), dtype=torch.bool, device=device),
    )
    trunk_k = torch.zeros(
        (1, sequence_length, 1, 16),
        dtype=torch.bfloat16,
        device=device,
    )
    trunk_v = torch.zeros_like(trunk_k)
    block_q = torch.zeros((1, 3, 2, 16), dtype=torch.bfloat16, device=device)
    block_k = torch.zeros((1, 3, 1, 16), dtype=torch.bfloat16, device=device)
    block_v = torch.zeros_like(block_k)
    calls: list[tuple[Tensor, Tensor, Tensor, Any]] = []

    def record_flex_call(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        **kwargs: Any,
    ) -> Tensor:
        calls.append((query, key, value, kwargs["block_mask"]))
        return torch.zeros_like(query)

    monkeypatch.setattr(
        attention_module,
        "_COMPILED_FLEX_ATTENTION",
        record_flex_call,
        raising=False,
    )
    output = attention_module.dflash_block_only_attention(
        plan=plan,
        trunk_k=trunk_k,
        trunk_v=trunk_v,
        block_q=block_q,
        block_k=block_k,
        block_v=block_v,
    )

    assert output.shape == block_q.shape
    assert len(calls) == 1
    query, key, value, block_mask = calls[0]
    assert query.shape == (1, 2, 3, 16)
    assert key.shape == value.shape == (1, 1, sequence_length + 3, 16)
    assert (
        key.untyped_storage().nbytes()
        == (trunk_k.numel() + block_k.numel()) * trunk_k.element_size()
    )
    metadata_elements = sum(
        tensor.numel()
        for tensor in (
            block_mask.kv_num_blocks,
            block_mask.kv_indices,
            block_mask.q_num_blocks,
            block_mask.q_indices,
        )
    )
    assert metadata_elements < sequence_length // 32
