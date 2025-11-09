# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from typing import cast

from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    ParallelStyle,
    RowwiseParallel,
    SequenceParallel,
)
from torch.distributed.tensor.placement_types import Replicate, Shard


def get_custom_parallel_plan():
    # Reuse llama default parallel plan
    base_model_tp_plan: dict[str, ParallelStyle] = {
        "model.embed_tokens": RowwiseParallel(input_layouts=Replicate()),
        "model.layers.*.self_attn.q_proj": ColwiseParallel(),
        "model.layers.*.self_attn.k_proj": ColwiseParallel(),
        "model.layers.*.self_attn.v_proj": ColwiseParallel(),
        "model.layers.*.self_attn.o_proj": RowwiseParallel(),
        "model.layers.*.mlp.up_proj": ColwiseParallel(),
        "model.layers.*.mlp.gate_proj": ColwiseParallel(),
        "model.layers.*.mlp.down_proj": RowwiseParallel(),
        "lm_head": ColwiseParallel(output_layouts=Shard(-1), use_local_output=False),
    }

    base_model_sp_plan = {
        "model.embed_tokens": RowwiseParallel(
            input_layouts=Replicate(), output_layouts=Shard(1)
        ),
        "model.norm": SequenceParallel(),
        "model.layers.*.input_layernorm": SequenceParallel(),
        "model.layers.*.self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1)),
        "model.layers.*.post_attention_layernorm": SequenceParallel(),
        "model.layers.*.mlp.down_proj": RowwiseParallel(output_layouts=Shard(1)),
        "lm_head": ColwiseParallel(
            input_layouts=Shard(1), output_layouts=Shard(-1), use_local_output=False
        ),
    }

    if False:
        # Enable sequence parallelism only if TP size > 1
        base_model_tp_plan.update(cast(dict[str, ParallelStyle], base_model_sp_plan))

    return base_model_tp_plan


custom_parallel_plan: dict[str, ParallelStyle] = get_custom_parallel_plan()
# {

# "model.embed_tokens": RowwiseParallel(
#     input_layouts=Replicate(), output_layouts=Replicate(), use_local_output=True
# ),
# "model.layers.*.self_attn.q_proj": ColwiseParallel(use_local_output=False),
# "model.layers.*.self_attn.k_proj": ColwiseParallel(use_local_output=False),
# "model.layers.*.self_attn.v_proj": ColwiseParallel(use_local_output=False),
# "model.layers.*.self_attn.o_proj": RowwiseParallel(
#     output_layouts=Replicate(), use_local_output=True
# ),
# "model.layers.*.self_attn.rotary_emb": PrepareModuleOutput(
#     output_layouts=(Replicate(), Replicate()),
#     desired_output_layouts=(Replicate(), Replicate()),
#     use_local_output=False,
# ),
# "model.layers.*.mlp.up_proj": ColwiseParallel(),
# "model.layers.*.mlp.gate_proj": ColwiseParallel(),
# "model.layers.*.mlp.down_proj": RowwiseParallel(
#     output_layouts=Replicate(), use_local_output=True
# ),
# "lm_head": ColwiseParallel(output_layouts=Shard(-1), use_local_output=False),
# }
