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
"""Four-GPU fused-RoPE/TE transformer parity across TP and runtime CP groups."""

import io
import os
import pickle
from types import SimpleNamespace

import numpy as np
import ray.cloudpickle
import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from ray.util.serialization import StandaloneSerializationContext

from nemo_rl.algorithms.loss.loss_functions import NLLLossFn
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.named_sharding import NamedSharding
from nemo_rl.distributed.tensor_serialization import (
    tensor_from_payload,
    tensor_to_payload,
)
from nemo_rl.models.megatron.data import process_microbatch
from nemo_rl.models.megatron.dynamic_cp import (
    RuntimeCPContext,
    initialize_dynamic_cp_runtime,
    planned_microbatches,
    preserve_attention_cp_groups,
)
from nemo_rl.models.megatron.train import (
    LogprobsPostProcessor,
    LossPostProcessor,
    model_forward,
)
from nemo_rl.models.policy.dynamic_cp import build_cp_dispatch, collect_cp_outputs
from nemo_rl.models.policy.workers.megatron_policy_worker import (
    MegatronPolicyWorkerImpl,
)
from nemo_rl.utils.timer import Timer


class DriverPayloadUnpickler(pickle.Unpickler):
    """Check that a worker result can load in a driver without MCore."""

    def find_class(self, module, name):
        if module == "megatron" or module.startswith(
            (
                "megatron.",
                "nemo_rl.models.megatron.",
                "nemo_rl.models.policy.workers.megatron",
            )
        ):
            raise AssertionError(
                f"Worker result requires backend class {module}.{name}"
            )
        return super().find_class(module, name)


def main() -> None:
    # Same reducer as the real worker, without starting another Ray cluster.
    StandaloneSerializationContext()._register_cloudpickle_serializer(
        torch.Tensor, tensor_to_payload, tensor_from_payload
    )
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 4
    lengths = torch.tensor([7, 45, 101, 11, 55, 9])
    ids = (torch.arange(6 * 112).reshape(6, 112) % 31 + 1).long()
    data = BatchedDataDict(
        input_ids=ids,
        input_lengths=lengths,
        token_mask=(torch.arange(112)[None, :] < lengths[:, None]).long(),
        sample_mask=torch.tensor([1, 1, 1, 0, 1, 1]),
    )
    for tp, base_cp in ((1, 1), (1, 2), (2, 1), (2, 2)):
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=tp,
            context_parallel_size=base_cp,
            hybrid_context_parallel=True,
        )
        initialize_dynamic_cp_runtime()
        torch.manual_seed(123)
        model_parallel_cuda_manual_seed(123)
        config = TransformerConfig(
            num_layers=2,
            hidden_size=256,
            num_attention_heads=4,
            num_query_groups=2,
            ffn_hidden_size=512,
            tensor_model_parallel_size=tp,
            context_parallel_size=base_cp,
            sequence_parallel=tp > 1,
            bf16=True,
            params_dtype=torch.bfloat16,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            apply_rope_fusion=True,
            gradient_accumulation_fusion=False,
        )
        # Exercise the same fused RoPE and TE attention path as the Qwen recipe.
        model = (
            GPTModel(
                config,
                get_gpt_layer_with_transformer_engine_spec(),
                vocab_size=128,
                max_sequence_length=128,
                position_embedding_type="rope",
                parallel_output=True,
            )
            .cuda()
            .bfloat16()
        )
        mesh = NamedSharding(
            np.arange(world).reshape(1, world // tp // base_cp, base_cp, tp),
            [
                "pipeline_parallel",
                "data_parallel",
                "context_parallel",
                "tensor_parallel",
            ],
        )
        cfg = {
            "logprob_batch_size": 1,
            "make_sequence_length_divisible_by": 1,
            "sequence_packing": {"enabled": True},
            "megatron_cfg": {
                "tensor_model_parallel_size": tp,
                "expert_model_parallel_size": 1,
                "sequence_parallel": tp > 1,
                "dynamic_context_parallel": {
                    "enabled": True,
                    "tokens_per_rank": 32 * tp,
                    "max_size": world // tp,
                },
            },
        }
        dispatch = build_cp_dispatch(data, cfg, mesh, batch_size=6, training=True)
        lane = rank // tp
        plan = dispatch.plans[lane // base_cp][lane % base_cp]
        payload = dispatch.data[lane // base_cp][lane % base_cp]
        step = plan.steps[0]
        worker = object.__new__(MegatronPolicyWorkerImpl)
        worker.model = model
        worker.cfg = cfg
        worker.timer = Timer()
        worker.mcore_state = SimpleNamespace(straggler_timer=None)
        worker.sampling_params = None
        worker.defer_fp32_logits = False
        worker._router_replay_enabled = False
        worker.media_placeholder_token_id = None
        worker.delegate_pack_to_model = False
        worker.delegate_mtp_loss_mask_to_model = False
        worker.model_slices_context_parallel_inputs = False
        scores = worker.get_logprobs(data=payload, cp_plan=plan)
        DriverPayloadUnpickler(io.BytesIO(ray.cloudpickle.dumps(scores))).load()
        gathered_scores = [None] * world
        dist.all_gather_object(gathered_scores, scores["logprobs"].numpy())
        restored_scores = collect_cp_outputs(
            [
                BatchedDataDict(logprobs=torch.from_numpy(value))
                for value in gathered_scores[::tp]
            ],
            dispatch,
            data.size,
        )["logprobs"].cuda()
        normalizers = (
            torch.tensor(step.valid_sequences, device="cuda"),
            torch.tensor(step.valid_tokens, device="cuda"),
        )
        reference_data = data.to("cuda")
        reference_batch = process_microbatch(
            reference_data,
            seq_length_key="input_lengths",
            pack_sequences=True,
            pad_individual_seqs_to_multiple_of=tp,
            cp_context=RuntimeCPContext(1, 0, None),
        )

        def forward(batch, batch_data):
            return model_forward(
                model=model,
                data_dict=batch_data,
                input_ids_cp_sharded=batch.input_ids_cp_sharded,
                position_ids=batch.position_ids,
                attention_mask=batch.attention_mask,
                packed_seq_params=batch.packed_seq_params,
            )

        with preserve_attention_cp_groups(model):
            with torch.no_grad():
                score_callback = LogprobsPostProcessor(cfg)(
                    reference_data,
                    reference_batch.input_ids,
                    reference_batch.packed_seq_params.cu_seqlens_q_padded,
                    ids.shape[1],
                    cp_context=RuntimeCPContext(1, 0, None),
                )
                _, reference_scores = score_callback(
                    forward(reference_batch, reference_data)
                )
                valid = reference_data["token_mask"].bool()
                torch.testing.assert_close(
                    restored_scores[valid],
                    reference_scores["logprobs"][valid],
                    rtol=0.01,
                    atol=0.03,
                    msg=lambda msg: f"Reassembled worker scoring: {msg}",
                )
            model.train()
            reference_callback = LossPostProcessor(NLLLossFn(), cfg)(
                reference_data, reference_batch.packed_seq_params, *normalizers
            )
            reference_loss, reference_metrics = reference_callback(
                forward(reference_batch, reference_data)
            )
            (reference_loss * base_cp).backward()
            for parameter in model.parameters():
                if parameter.grad is not None and getattr(
                    parameter, "sequence_parallel", False
                ):
                    dist.all_reduce(
                        parameter.grad,
                        group=parallel_state.get_tensor_model_parallel_group(),
                    )
            reference_grads = {
                name: p.grad.float().clone()
                for name, p in model.named_parameters()
                if p.grad is not None
            }
            model.zero_grad(set_to_none=True)
            reported = torch.zeros((), device="cuda")
            processor = LossPostProcessor(NLLLossFn(), cfg, len(step.assignments))
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
                    batch.data_dict, batch.packed_seq_params, *normalizers
                )
                loss, metrics = callback(forward(batch, batch.data_dict))
                (loss * base_cp / len(step.assignments)).backward()
                if task.sample_indices and lane == task.lane_start:
                    reported += metrics["loss"]
            assert group_boundaries == len(step.groups)
            domain = parallel_state.get_data_parallel_group(with_context_parallel=True)
            dist.all_reduce(reported, group=domain)
            torch.testing.assert_close(
                reported.item(), reference_metrics["loss"], rtol=0.01, atol=0.01
            )
            squared_error = torch.zeros((), device="cuda")
            squared_reference = torch.zeros((), device="cuda")
            for name, parameter in model.named_parameters():
                if name in reference_grads:
                    gradient = parameter.grad.float()
                    if getattr(parameter, "sequence_parallel", False):
                        dist.all_reduce(
                            gradient,
                            group=parallel_state.get_tensor_model_parallel_group(),
                        )
                    dist.all_reduce(gradient, group=domain)
                    squared_error += (gradient - reference_grads[name]).square().sum()
                    squared_reference += reference_grads[name].square().sum()
                    torch.testing.assert_close(
                        gradient,
                        reference_grads[name],
                        rtol=0.08,
                        atol=0.003,
                        msg=lambda msg: f"tp={tp}, base_cp={base_cp}, {name}: {msg}",
                    )
            relative_error = (squared_error / squared_reference).sqrt().item()
            assert relative_error < 0.03, (tp, base_cp, relative_error)
        print(
            f"attention CP parity passed: rank={rank} tp={tp} base_cp={base_cp} "
            f"active_sizes={sizes} gradient_relative_error={relative_error:.6f}",
            flush=True,
        )
        parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
