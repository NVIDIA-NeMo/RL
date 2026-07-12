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

"""Unit tests for the additive_weight_load_context mechanism.

End-to-end bit-identity vs. a real vLLM ColumnParallelLinear is deferred to
the integration test that also exercises the receiver wiring (M2). At the
unit level we only need to prove the torch-level monkey-patch is scoped
correctly and cleaned up on exception.
"""

import pytest
import torch

from nemo_rl.models.generation.vllm.vllm_sparse_delta_additive import (
    additive_weight_load_context,
    apply_sparse_delta_via_additive_load,
)


@pytest.mark.vllm
def test_additive_context_matches_narrowed_view_of_target() -> None:
    """vLLM packed loaders write to `param.data.narrow(...)`, which has a
    different data_ptr from the base tensor but shares storage. The context
    manager must redirect narrowed-view copies too, otherwise QKV/gate_up
    stay overwrite-semantics and corrupt the base slice."""
    device = torch.device("cuda", 0)
    target = torch.arange(8, dtype=torch.float32, device=device)
    view = target.narrow(0, 2, 4)
    src = torch.full((4,), 10.0, device=device)

    assert view.data_ptr() != target.data_ptr()
    assert view.untyped_storage().data_ptr() == target.untyped_storage().data_ptr()

    with additive_weight_load_context(target):
        view.copy_(src)

    expected = torch.tensor(
        [0.0, 1.0, 12.0, 13.0, 14.0, 15.0, 6.0, 7.0],
        dtype=torch.float32,
        device=device,
    )
    assert torch.equal(target, expected)


@pytest.mark.vllm
def test_additive_weight_load_context_scopes_to_targets() -> None:
    device = torch.device("cuda", 0)
    target = torch.ones(4, device=device)
    bystander = torch.ones(4, device=device)
    src = torch.full((4,), 2.0, device=device)

    with additive_weight_load_context(target):
        target.copy_(src)
        bystander.copy_(src)

    assert torch.equal(target, torch.full((4,), 3.0, device=device))
    assert torch.equal(bystander, torch.full((4,), 2.0, device=device))


@pytest.mark.vllm
def test_additive_context_restored_on_exception() -> None:
    device = torch.device("cuda", 0)
    target = torch.ones(4, device=device)
    original_copy_ = torch.Tensor.copy_

    with pytest.raises(RuntimeError, match="boom"):
        with additive_weight_load_context(target):
            raise RuntimeError("boom")

    assert torch.Tensor.copy_ is original_copy_


@pytest.mark.vllm
def test_additive_bit_identity_over_weight_loader_pattern() -> None:
    """Emulate vLLM's plain-linear weight_loader op pattern without the class.

    vLLM 0.20's ColumnParallelLinear.__init__ requires a full engine config
    bootstrap that a unit test should not fake. The op the context has to
    intercept is a single `param.data.copy_(loaded_weight)` after TP-narrow.
    Reproduce that call site directly and assert additive semantics.
    """
    device = torch.device("cuda", 0)
    dtype = torch.float32

    torch.manual_seed(42)
    param = torch.nn.Parameter(torch.randn(4, 8, dtype=dtype, device=device))
    original = param.data.clone()

    indices = torch.tensor([0, 15, 31], dtype=torch.int64, device=device)
    values = torch.tensor([1.0, 2.0, 3.0], dtype=dtype, device=device)

    reference = original.clone()
    reference.view(-1).index_add_(0, indices, values)

    dense = torch.zeros(4, 8, dtype=dtype, device=device)
    dense.view(-1).index_copy_(0, indices, values)

    with additive_weight_load_context(param.data):
        param.data.copy_(dense)

    assert torch.equal(param.data, reference), (
        f"Additive apply mismatch. "
        f"max abs diff: {(param.data - reference).abs().max().item()}"
    )


@pytest.mark.vllm
def test_apply_sparse_delta_via_additive_load_end_to_end() -> None:
    """Full entrypoint exercised with a SimpleNamespace stand-in for the vLLM model.

    Mirrors the mock pattern used by test_vllm_sparse_delta.py for the shadow
    dispatch — the only real code under test is our entrypoint + context
    manager; the model shim just has to expose named_parameters and
    load_weights the way vLLM's model API does.
    """
    device = torch.device("cuda", 0)
    dtype = torch.float32
    name = "linear.weight"

    torch.manual_seed(0)
    param = torch.nn.Parameter(torch.randn(4, 8, dtype=dtype, device=device))
    original = param.data.clone()

    indices = torch.tensor([1, 7, 30], dtype=torch.int64)
    values = torch.tensor([0.5, -1.0, 2.5], dtype=dtype)

    reference = original.clone()
    reference.view(-1).index_add_(0, indices.to(device), values.to(device))

    class _ShimModel:
        def named_parameters(self):
            yield name, param

        def load_weights(self, weights):
            for wname, tensor in weights:
                assert wname == name
                param.data.copy_(tensor)

    apply_sparse_delta_via_additive_load(
        name=name,
        sparse_indices=indices,
        sparse_values=values,
        target_shape=(4, 8),
        target_dtype=dtype,
        model=_ShimModel(),
        device=device,
    )

    assert torch.equal(param.data, reference), (
        f"Entrypoint mismatch. "
        f"max abs diff: {(param.data - reference).abs().max().item()}"
    )
