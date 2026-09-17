# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Numerical checks for NRL-owned MTP and resident gradient accumulation.

Run with torch.distributed.run inside the MCore environment. No checkpoints,
Ray actors, or sandbox services are needed.
"""

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel
from megatron.core.distributed.distributed_data_parallel_config import (
    DistributedDataParallelConfig,
)
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.megatron.setup import _freeze_disabled_mtp
from nemo_rl.models.megatron.train import model_forward
from nemo_rl.models.policy.workers.megatron_policy_worker import (
    MegatronPolicyWorkerImpl,
)

pytestmark = pytest.mark.mcore


@pytest.fixture
def model_parallel():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    parallel_state.initialize_model_parallel()
    model_parallel_cuda_manual_seed(123)
    torch.manual_seed(123)
    yield
    parallel_state.destroy_model_parallel()


def _hybrid(pattern, dtype):
    return (
        HybridModel(
            config=TransformerConfig(
                num_layers=2,
                hidden_size=256,
                num_attention_heads=4,
                use_cpu_initialization=True,
                bf16=dtype == torch.bfloat16,
                params_dtype=dtype,
                attention_dropout=0.0,
                hidden_dropout=0.0,
                mtp_num_layers=1,
                mamba_state_dim=16,
                mamba_head_dim=64,
                mamba_num_groups=8,
            ),
            hybrid_stack_spec=hybrid_stack_spec,
            hybrid_layer_pattern=pattern,
            vocab_size=256,
            max_sequence_length=128,
        )
        .cuda()
        .train()
    )


@pytest.mark.parametrize("pattern", ["**/*", "M*/*"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mtp_bypass_logits_gradients_and_checkpoint_keys(
    model_parallel, pattern, dtype
):
    reference = _hybrid(pattern, dtype)
    actual = _hybrid(pattern, dtype)
    _freeze_disabled_mtp([actual])
    actual.load_state_dict(reference.state_dict(), strict=True)
    assert actual.state_dict().keys() == reference.state_dict().keys()
    assert all(not parameter.requires_grad for parameter in actual.mtp.parameters())
    input_ids = torch.arange(128, device="cuda").unsqueeze(0)
    data = BatchedDataDict({"input_ids": input_ids})
    with (
        patch.object(
            actual.mtp, "forward", side_effect=AssertionError("MTP forward ran")
        ),
        patch(
            "megatron.core.models.hybrid.hybrid_model.process_mtp_loss",
            side_effect=AssertionError("MTP loss ran"),
        ),
    ):
        expected = reference(input_ids, input_ids.clone(), None, compute_mtp_loss=False)
        observed = model_forward(
            model=actual,
            data_dict=data,
            input_ids_cp_sharded=input_ids,
            position_ids=input_ids.clone(),
            attention_mask=None,
            compute_mtp_loss=False,
        )
        torch.testing.assert_close(observed, expected, rtol=0, atol=0)
        weights = torch.randn_like(expected.float())
        (expected.float() * weights).sum().backward()
        (observed.float() * weights).sum().backward()
    reference_params = dict(reference.named_parameters())
    nonzero = 0
    for name, parameter in actual.named_parameters():
        expected_grad = reference_params[name].grad
        if expected_grad is None:
            assert parameter.grad is None, name
            continue
        assert torch.isfinite(parameter.grad).all(), name
        torch.testing.assert_close(
            parameter.grad, expected_grad, rtol=0, atol=0, msg=name
        )
        nonzero += int(torch.count_nonzero(parameter.grad))
    assert nonzero > 0
    print(
        f"MTP bypass {pattern} {dtype}: bit-exact logits/gradients; checkpoint keys retained"
    )


def test_mtp_bypass_rejects_frozen_backbone_and_unsupported_model(model_parallel):
    model = _hybrid("**/*", torch.float32)
    model.config.freeze_base_model_for_mtp = True
    with pytest.raises(ValueError, match="cannot both be enabled"):
        _freeze_disabled_mtp([model])
    unsupported = SimpleNamespace(config=SimpleNamespace(mtp_num_layers=1))
    with pytest.raises(ValueError, match="requires HybridModel"):
        _freeze_disabled_mtp([unsupported])
    with pytest.raises(ValueError, match="requires HybridModel"):
        model_forward(
            model=unsupported,
            data_dict=BatchedDataDict({"input_ids": torch.ones(1, 2, device="cuda")}),
            input_ids_cp_sharded=torch.ones(1, 2, device="cuda"),
            position_ids=None,
            attention_mask=None,
            compute_mtp_loss=False,
        )


def test_accumulated_ddp_gradients_survive_logprob_phase(model_parallel):
    config = TransformerConfig(num_layers=1, hidden_size=16, num_attention_heads=4)
    actual = torch.nn.Linear(16, 8, device="cuda")
    reference = torch.nn.Linear(16, 8, device="cuda")
    reference.load_state_dict(actual.state_dict())
    wrapped = DistributedDataParallel(
        config, DistributedDataParallelConfig(overlap_grad_reduce=False), actual
    )
    worker = object.__new__(MegatronPolicyWorkerImpl)
    worker.model = wrapped
    worker.optimizer = None
    worker.optimizer_cpu_offload = False
    worker.offload_optimizer_for_logprob = False
    worker._train_step_state = {}
    worker._opd_full_teacher_lm_head = None
    worker._opd_full_enabled = False
    worker.rank = dist.get_rank()
    worker.cfg = {"megatron_cfg": {"empty_unused_memory_level": 0}}
    worker.megatron_cfg = SimpleNamespace(
        optimizer=SimpleNamespace(), ddp=SimpleNamespace()
    )
    wrapped.zero_grad_buffer()
    inputs = [torch.randn(4, 16, device="cuda") for _ in range(2)]
    for index, values in enumerate(inputs):
        with wrapped.no_sync():
            wrapped(values).square().sum().backward()
        reference(values).square().sum().backward()
        if index == 0:
            saved = [buffer.grad_data.clone() for buffer in wrapped.buffers]
            worker.prepare_for_lp_inference(keep_train_buffers=True)
            with torch.no_grad():
                wrapped(values)
            worker.prepare_for_training()
            for buffer, expected in zip(wrapped.buffers, saved):
                torch.testing.assert_close(buffer.grad_data, expected, rtol=0, atol=0)
    for actual_parameter, reference_parameter in zip(
        actual.parameters(), reference.parameters()
    ):
        torch.testing.assert_close(
            actual_parameter.main_grad, reference_parameter.grad, rtol=0, atol=0
        )
    print(
        "Two-chunk DDP accumulation with intervening logprob phase: bit-exact gradients"
    )
