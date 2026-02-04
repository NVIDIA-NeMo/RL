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

from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel
from torch.distributed.tensor.placement_types import Replicate, Shard

custom_parallel_plan = {
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

"""
Note on numerical stability:

- Default plans that keep attention output proj and mlp downproj RowwiseParallel are numerically
  unstable and tend to increase with larger TP (e.g., TP >= 4).

Enable this custom plan via:

- policy.dtensor_cfg.custom_parallel_plan=examples.custom_parallel.qwen_model_tp_plan_stable

Based on https://github.com/NVIDIA-NeMo/Automodel/blob/d79ccb94b0eca94a4c479313db2f9eee80db0139/nemo_automodel/components/distributed/optimized_tp_plans.py#L205-L217
"""
qwen_model_tp_plan_stable = {
    "lm_head": ColwiseParallel(output_layouts=Shard(-1), use_local_output=False),
    "model.embed_tokens": RowwiseParallel(
        input_layouts=Replicate(),
    ),
    "model.layers.*.self_attn.q_proj": ColwiseParallel(),
    "model.layers.*.self_attn.k_proj": ColwiseParallel(),
    "model.layers.*.self_attn.v_proj": ColwiseParallel(),
    "model.layers.*.self_attn.o_proj": ColwiseParallel(
        input_layouts=Shard(-1),
        output_layouts=Replicate(),
        use_local_output=True,
    ),
    "model.layers.*.mlp.up_proj": ColwiseParallel(),
    "model.layers.*.mlp.gate_proj": ColwiseParallel(),
    "model.layers.*.mlp.down_proj": ColwiseParallel(
        input_layouts=Shard(-1),
        output_layouts=Replicate(),
        use_local_output=True,
    ),
}
