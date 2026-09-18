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
"""Native validation; never rewrites data or resolves judge credentials into receipts."""

import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from omegaconf import OmegaConf
from nemo_rl.algorithms.grpo import MasterConfig
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

root = Path(os.environ["SUPER_RL_ROOT"])
data = Path(os.environ["SUPER_RL_DATA"])
routes = {
    "math_with_judge_simple_agent",
    "equivalence_llm_judge_simple_agent",
    "scicode_agent",
}
counts = Counter()
digest = hashlib.sha256()
with data.open("rb") as stream:
    for raw in stream:
        digest.update(raw)
        route = json.loads(raw)["agent_ref"]["name"]
        if route not in routes:
            raise ValueError(
                f"Unsupported data route: {route}; this recipe does not drop or rewrite rows"
            )
        counts[route] += 1
if (
    sum(counts.values()) < 25600
    or digest.hexdigest() != os.environ["SUPER_RL_DATA_SHA256"]
):
    raise ValueError("Data digest mismatch or insufficient rows for 100 ordered steps")
register_omegaconf_resolvers()
config_path = Path(os.environ["SUPER_RL_CONFIG"])
cfg = load_config(str(config_path))
native = MasterConfig(**OmegaConf.to_container(cfg, resolve=True))
assert native.grpo.max_num_steps == 100
assert cfg.grpo.num_prompts_per_step * cfg.grpo.num_generations_per_prompt == 4096
assert cfg.grpo.async_grpo.max_trajectory_age_steps == 2
assert cfg.policy.train_global_batch_size == 4096
assert (
    cfg.cluster.num_nodes == 64
    and cfg.policy.generation.colocated.resources.num_nodes == 48
)
assert (
    cfg.policy.megatron_cfg.context_parallel_size == 4
    and cfg.policy.megatron_cfg.expert_model_parallel_size == 16
)
assert (
    cfg.policy.generation.max_new_tokens == 102400
    and cfg.policy.max_total_sequence_length == 131072
)
assert cfg.env.nemo_gym.policy_model.responses_api_models.vllm_model.chat_template_kwargs.enable_thinking
assert cfg.policy.router_replay.enabled and cfg.policy.router_replay.transport == "ray"
assert not cfg.checkpointing.load_replay_buffer
model = Path(os.environ["SUPER_RL_MODEL"])
index = json.loads((model / "model.safetensors.index.json").read_text())
shards = set(index["weight_map"].values())
if len(shards) != 64 or not all((model / name).stat().st_size > 0 for name in shards):
    raise ValueError("This reproduction expects the s120 64-shard model")
receipt = {
    "complete": True,
    "source_commit": (root / "source.commit").read_text().strip(),
    "total_gpus": 256,
    "num_steps": 100,
    "rows": sum(counts.values()),
    "routes": dict(counts),
    "data_sha256": digest.hexdigest(),
    "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    "reasoning_enabled": True,
    "load_replay_buffer": False,
    "scope": "Native preflight, not GPU smoke or model-quality certification",
}
with (root / "manifests/preflight.json").open("x") as stream:
    json.dump(receipt, stream, indent=2)
print(json.dumps(receipt))
