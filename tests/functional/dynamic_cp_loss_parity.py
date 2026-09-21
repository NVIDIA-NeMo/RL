# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Four-GPU parity for real NRL packing, CP logprob collectives and gradients.

Run with the pinned Megatron environment:
    torchrun --standalone --nproc-per-node=4 tests/functional/dynamic_cp_loss_parity.py
This uses a tiny trainable token-to-logit table so the assertion isolates the
data/loss integration from transformer numerical differences. Qwen GRPO tests
the actual attention model separately.
"""

import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core import parallel_state

from nemo_rl.algorithms.loss.loss_functions import NLLLossFn
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.named_sharding import NamedSharding
from nemo_rl.models.megatron.dynamic_cp import planned_microbatches
from nemo_rl.models.megatron.train import LossPostProcessor
from nemo_rl.models.policy.dynamic_cp import build_cp_dispatch


def main() -> None:
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 4:
        raise ValueError("Run this parity test with four ranks")
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        hybrid_context_parallel=True,
    )
    # Leave enough short work after filling long packs to exercise CP1 too.
    lengths = torch.tensor([7, 45, 101, 11, 55, 9, 29, 29, 29, 29])
    ids = (torch.arange(len(lengths) * 112).reshape(len(lengths), 112) % 31 + 1).long()
    data = BatchedDataDict(
        input_ids=ids,
        input_lengths=lengths,
        token_mask=(torch.arange(112)[None, :] < lengths[:, None]).long(),
        sample_mask=torch.tensor([1, 1, 1, 0, 1, 1, 1, 1, 1, 1]),
    )
    mesh = NamedSharding(
        np.arange(world).reshape(1, world, 1, 1),
        ["pipeline_parallel", "data_parallel", "context_parallel", "tensor_parallel"],
    )
    cfg = {
        "make_sequence_length_divisible_by": 1,
        "sequence_packing": {"enabled": True},
        "megatron_cfg": {
            "tensor_model_parallel_size": 1,
            "expert_model_parallel_size": 1,
            "sequence_parallel": False,
            "dynamic_context_parallel": {
                "enabled": True,
                "tokens_per_rank": 32,
                "max_size": 4,
            },
        },
    }
    initial = torch.sin(torch.arange(32 * 32, device="cuda").float()).reshape(32, 32)
    for base_cp in (1, 2):
        # Reinitialize static groups to test active CP both below and above base CP.
        if base_cp != 1:
            parallel_state.destroy_model_parallel()
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                context_parallel_size=base_cp,
                hybrid_context_parallel=True,
            )
            mesh = NamedSharding(
                np.arange(world).reshape(1, world // base_cp, base_cp, 1), mesh.names
            )
        dispatch = build_cp_dispatch(
            data, cfg, mesh, batch_size=data.size, training=True
        )
        active_sizes = {
            task.cp_size
            for dp_plans in dispatch.plans
            for rank_plan in dp_plans
            for rank_step in rank_plan.steps
            for task in rank_step.assignments
            if task.sample_indices
        }
        assert active_sizes == {1, 2, 4}, active_sizes
        plan = dispatch.plans[rank // base_cp][rank % base_cp]
        payload = dispatch.data[rank // base_cp][rank % base_cp]
        step = plan.steps[0]
        reference = initial.clone().requires_grad_()
        labels = ids[:, 1:].cuda()
        mask = (data["token_mask"][:, 1:] * data["sample_mask"][:, None]).cuda()
        reference_loss = (
            F.cross_entropy(
                F.embedding(ids[:, :-1].cuda(), reference).flatten(0, 1),
                labels.flatten(),
                reduction="none",
            ).reshape_as(mask)
            * mask
        ).sum() / mask.sum()
        reference_loss.backward()
        weight = initial.clone().requires_grad_()
        processor = LossPostProcessor(
            NLLLossFn(), cfg, num_microbatches=len(step.assignments)
        )
        reported = torch.zeros((), device="cuda")
        sizes = []
        group_boundaries = 0
        for task, batch in zip(
            step.assignments, planned_microbatches(payload, plan, step, None)
        ):
            if batch.dynamic_cp_group_start:
                group_boundaries += 1
                dist.barrier()
            sizes.append(task.cp_size)
            callback = processor(
                batch.data_dict,
                batch.packed_seq_params,
                torch.tensor(step.valid_sequences, device="cuda"),
                torch.tensor(step.valid_tokens, device="cuda"),
            )
            loss, metrics = callback(F.embedding(batch.input_ids_cp_sharded, weight))
            (loss * base_cp / len(step.assignments)).backward()
            if task.sample_indices and rank == task.lane_start:
                reported += metrics["loss"]
        assert group_boundaries == len(step.groups)
        dist.all_reduce(weight.grad)
        dist.all_reduce(reported)
        torch.testing.assert_close(reported, reference_loss, rtol=3e-5, atol=3e-6)
        torch.testing.assert_close(weight.grad, reference.grad, rtol=3e-5, atol=3e-6)
        torch.testing.assert_close(
            initial - 0.1 * weight.grad,
            initial - 0.1 * reference.grad,
            rtol=3e-5,
            atol=3e-6,
        )
        print(
            f"dynamic CP parity passed: rank={rank} base_cp={base_cp} active_sizes={sizes}",
            flush=True,
        )
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
