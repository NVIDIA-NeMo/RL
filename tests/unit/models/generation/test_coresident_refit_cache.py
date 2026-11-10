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
"""Parameter-derived Mamba decode caches must follow an in-place IPC refit."""

from types import SimpleNamespace

import pytest
import torch

from nemo_rl.models.megatron import sm_partition


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("shared", [False, True])
def test_coresident_refit_refreshes_captured_mamba_cache(shared: bool) -> None:
    # The standalone CPU suite does not need the optional Megatron inference dependencies.
    from megatron.core.resharding.utils import named_refit_tensors
    from megatron.core.ssm.mamba_mixer import MambaMixer
    from megatron.core.transformer.module import MegatronModule

    from nemo_rl.models.generation.megatron.megatron_worker import (
        MegatronGenerationRefitMixin,
    )

    torch.cuda.set_device(0)
    mixer = object.__new__(MambaMixer)
    MegatronModule.__init__(mixer, SimpleNamespace())
    mixer.A_log = torch.nn.Parameter(torch.tensor([0.2, 0.7], device="cuda"))
    mixer._A_neg_exp_cache = torch.empty_like(mixer.A_log)
    mixer._A_neg_exp_cache_stale = True
    mixer.headdim, mixer.d_state = 2, 3
    source = {"A_log": mixer.A_log.detach().clone()}
    if shared:
        sm_partition.bind_shared(source, named_refit_tensors(mixer), {id(mixer.A_log)})
    mixer.eval()
    cache_ptr = mixer._A_neg_exp_cache.data_ptr()
    weight_ptr = mixer.A_log.data_ptr()
    x = torch.ones(2, 2, 3, device="cuda")
    out = torch.empty_like(x)
    with torch.inference_mode():
        out.copy_(x * mixer._get_decode_A_neg_exp())
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(graph):
        out.copy_(x * mixer._get_decode_A_neg_exp())

    source["A_log"].copy_(torch.tensor([0.9, 1.1], device="cuda"))
    sm_partition.copy_refit(source, named_refit_tensors(mixer))
    graph.replay()
    torch.cuda.synchronize()
    expected = -source["A_log"].exp().view(2, 1, 1).expand_as(x)
    assert not torch.equal(out, expected), (
        "negative control: parameter copy alone must leave the captured cache stale"
    )

    worker = SimpleNamespace(
        _coresident_src=source,
        _gen_model=lambda: mixer,
        _refresh_flashinfer_mxfp8_weights=lambda: None,
    )
    for value in (1.2, 1.6):
        source["A_log"].fill_(value)
        MegatronGenerationRefitMixin._coresident_copy_weights(worker)
        graph.replay()
        torch.cuda.synchronize()
        expected = -source["A_log"].exp().view(2, 1, 1).expand_as(x)
        assert torch.equal(out, expected)
        assert mixer._A_neg_exp_cache.data_ptr() == cache_ptr
        assert mixer.A_log.data_ptr() == weight_ptr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("packed", [False, True])
def test_shared_expert_stacks_keep_graph_addresses_and_refit_values(packed: bool) -> None:
    # Actual inference modules are optional in the standalone CPU suite.
    from megatron.core.resharding.utils import named_refit_tensors
    from megatron.core.transformer.module import MegatronModule
    from megatron.core.transformer.moe.experts import InferenceGroupedMLP

    experts = object.__new__(InferenceGroupedMLP)
    MegatronModule.__init__(experts, SimpleNamespace())
    experts.num_local_experts = 3
    experts._concatenated_weights_built = False
    source = {}
    for linear_name in ("linear_fc1", "linear_fc2"):
        linear = torch.nn.Module()
        experts.add_module(linear_name, linear)
        backing = torch.randn(3, 16, 16, device="cuda", dtype=torch.bfloat16)
        for index in range(3):
            tensor = backing[index] if packed else backing[2 - index]
            linear.register_parameter(f"weight{index}", torch.nn.Parameter(torch.empty_like(tensor)))
            source[f"{linear_name}.weight{index}"] = tensor
    sm_partition.bind_shared(source, named_refit_tensors(experts), {id(p) for p in experts.parameters()})
    stats = sm_partition.bind_shared_expert_stacks(experts, source)
    assert stats["shared_expert_layers"] == int(packed)
    if not packed:
        assert not experts._concatenated_weights_built
        experts._build_concatenated_weights()
    assert not {"_fc1_weight", "_fc2_weight"}.intersection(dict(named_refit_tensors(experts)))
    stacks = (experts._fc1_weight, experts._fc2_weight)
    pointers = [stack.data_ptr() for stack in stacks]
    x = torch.randn(3, 8, 16, device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(x)
    with torch.inference_mode():
        out.copy_(torch.bmm(torch.bmm(x, stacks[0]), stacks[1]))
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(graph):
        out.copy_(torch.bmm(torch.bmm(x, stacks[0]), stacks[1]))
    for value in (0.25, 0.5):
        for tensor in source.values():
            tensor.fill_(value)
        stats = sm_partition.copy_refit(source, named_refit_tensors(experts))
        graph.replay()
        torch.cuda.synchronize()
        expected_stacks = [torch.stack([source[f"{name}.weight{i}"] for i in range(3)])
                           for name in ("linear_fc1", "linear_fc2")]
        expected = torch.bmm(torch.bmm(x, expected_stacks[0]), expected_stacks[1])
        assert torch.equal(out, expected)
        assert [stack.data_ptr() for stack in stacks] == pointers
        assert (stats["bytes"] == 0) == packed



@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_shared_expert_order_uses_native_optimizer_storage_layout() -> None:
    # Validate the pinned optimizer's actual packing, not a simulated reversal.
    from megatron.core.distributed import DistributedDataParallelConfig
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
    from megatron.core.transformer.module import MegatronModule
    from megatron.core.transformer.moe.experts import TEGroupedMLP

    experts = object.__new__(TEGroupedMLP)
    MegatronModule.__init__(experts, SimpleNamespace())
    experts.num_local_experts = 3
    originals = {}
    for linear_name in ("linear_fc1", "linear_fc2"):
        linear = torch.nn.Module()
        experts.add_module(linear_name, linear)
        for index in range(3):
            value = torch.randn(16, 16, device="cuda", dtype=torch.bfloat16)
            linear.register_parameter(f"weight{index}", torch.nn.Parameter(value))
            originals[f"{linear_name}.weight{index}"] = value.clone()
    models = [experts]
    assert sm_partition.prepare_shared_expert_layout(models) is models
    layout = DistributedOptimizer._compute_per_buffer_param_layout(
        list(experts.parameters()), bucket_size=512, data_parallel_world_size=1,
        ddp_config=DistributedDataParallelConfig(pad_buckets_for_high_nccl_busbw=False),
    )
    backing = torch.empty(max(end for _, end, _ in layout.param_index_map.values()), device="cuda", dtype=torch.bfloat16)
    for name, parameter in experts.named_parameters():
        start, end, _ = layout.param_index_map[parameter]
        parameter.data = backing[start:end].view_as(parameter)
        parameter.data.copy_(originals[name])
    for linear_name in ("linear_fc1", "linear_fc2"):
        linear = getattr(experts, linear_name)
        weights = [getattr(linear, f"weight{i}") for i in range(3)]
        stack = sm_partition._contiguous_expert_stack(weights)
        assert stack is not None
        assert torch.equal(stack, torch.stack([originals[f"{linear_name}.weight{i}"] for i in range(3)]))
        assert all(stack[i].data_ptr() == weights[i].data_ptr() for i in range(3))
