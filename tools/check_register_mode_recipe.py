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
"""Resolve a recipe and report what its data plane and mesh will actually do.

Cheap pre-flight for a multi-node submission: catches an unresolvable
`defaults:` chain, a MasterConfig validation error, a mesh that does not divide
the allocation, and a data_plane block that silently resolves to the wrong
backend — all without burning an allocation to find out.

    python tools/check_register_mode_recipe.py <config.yaml>
"""

import sys

from omegaconf import OmegaConf

from nemo_rl.data_plane.interfaces import backend_config
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers


def main(path: str) -> int:
    """Print the resolved mesh / data-plane facts; return non-zero on mismatch."""
    # Exemplar configs use NeMo-RL's own resolvers (``${mul:...}``); resolving
    # without registering them first fails the way the entrypoint never would.
    register_omegaconf_resolvers()
    cfg = load_config(path)
    resolved = OmegaConf.to_container(cfg, resolve=True)

    from nemo_rl.algorithms.grpo import MasterConfig

    MasterConfig(**resolved)
    print("MasterConfig: OK")

    cluster = resolved["cluster"]
    gen = resolved["policy"]["generation"]["colocated"]
    mcore = resolved["policy"]["megatron_cfg"]
    gen_nodes = 0 if gen["enabled"] else gen["resources"]["num_nodes"]
    train_nodes = cluster["num_nodes"] - gen_nodes
    train_gpus = train_nodes * cluster["gpus_per_node"]
    mesh = (
        mcore["tensor_model_parallel_size"]
        * mcore["pipeline_model_parallel_size"]
        * mcore["context_parallel_size"]
    )
    print(
        f"cluster: {cluster['num_nodes']}n x {cluster['gpus_per_node']}g "
        f"-> train {train_nodes}n ({train_gpus} GPU), gen {gen_nodes}n"
    )
    print(
        f"mesh: TP{mcore['tensor_model_parallel_size']} x "
        f"PP{mcore['pipeline_model_parallel_size']} x "
        f"CP{mcore['context_parallel_size']} = {mesh} -> DP{train_gpus // mesh}"
    )

    dp = resolved["data_plane"]
    block = backend_config(dp)
    print(f"data_plane: enabled={dp['enabled']} backend={dp['backend']}")
    print(f"  resolved block: {block}")

    ok = True
    if train_gpus % mesh:
        print(f"FAIL: mesh {mesh} does not divide {train_gpus} train GPUs")
        ok = False
    gbs = resolved["policy"]["train_global_batch_size"]
    prompts = resolved["grpo"]["num_prompts_per_step"]
    gens = resolved["grpo"]["num_generations_per_prompt"]
    if prompts * gens != gbs:
        print(
            f"FAIL: SingleController needs num_prompts_per_step x "
            f"num_generations_per_prompt ({prompts} x {gens} = {prompts * gens}) "
            f"== train_global_batch_size ({gbs})"
        )
        ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
