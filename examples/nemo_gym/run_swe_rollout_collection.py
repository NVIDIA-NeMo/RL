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
"""Generation-only rollout collection with NeMo-Gym (stage A, decoupled PPO).

Reuses the production PPO config so shards match what the coupled run's critic
warmup would consume, but spins up ONLY the vLLM engines + NeMo-Gym — no
policy/value workers, no refit NCCL group, no replay buffer. Job shape and
sharding knobs come from a ``collection:`` config block (see
``nemo_rl.algorithms.rollout_collection.resolve_collection_config``); launch
via scripts/swe/ppo/collect_rollouts.sh (1-node SLURM array tasks).
"""

import argparse
import os
import pprint
import time
from concurrent.futures import ThreadPoolExecutor

from omegaconf import OmegaConf

from nemo_rl.algorithms.grpo import _should_use_nemo_gym
from nemo_rl.algorithms.ppo import MasterConfig
from nemo_rl.algorithms.rollout_collection import (
    collect_rollouts,
    resolve_collection_config,
    spinup_nemo_gym,
)
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.utils import setup_response_data
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster, init_ray
from nemo_rl.environments.nemo_gym import setup_nemo_gym_config
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.vllm import VllmGeneration
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generation-only SWE rollout collection with NeMo-Gym"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    args, overrides = parser.parse_known_args()
    return args, overrides


def main() -> None:
    """Main entry point."""
    register_omegaconf_resolvers()
    args, overrides = parse_args()

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")
    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)
    config = OmegaConf.to_container(config, resolve=True)
    config = MasterConfig(**config)

    collection = resolve_collection_config(
        getattr(config, "collection", None), config.ppo
    )

    tokenizer = get_tokenizer(config.policy["tokenizer"])
    assert config.policy["generation"] is not None, (
        "A generation config is required for rollout collection"
    )
    # is_eval=True => vllm_cfg.load_format="auto": the engine must load REAL
    # pi_0 weights from policy.model_name. There is no policy worker and no
    # refit in this job; the training default ("dummy") would generate from
    # random weights.
    config.policy["generation"] = configure_generation_config(
        config.policy["generation"], tokenizer, is_eval=True
    )
    setup_nemo_gym_config(config, tokenizer)
    assert _should_use_nemo_gym(config), (
        "Rollout collection requires the NeMo-Gym path "
        "(env.should_use_nemo_gym=true with an async, HTTP-exposed vLLM engine)."
    )

    print("\n▶ Setting up data...")
    train_dataset, _ = setup_response_data(tokenizer, config.data, env_configs=None)
    assert not isinstance(train_dataset, dict), (
        "use_multiple_dataloader is not supported for rollout collection"
    )

    print("Final config:")
    pprint.pprint(config)
    print(f"Collection config: {collection}")

    init_ray()

    # ------------------------------------------------------------------
    # Inference cluster over ALL nodes of this job (generation-only shape).
    # ------------------------------------------------------------------
    cluster_config = config.cluster
    num_nodes = cluster_config["num_nodes"]
    gpus_per_node = cluster_config["gpus_per_node"]
    cluster = RayVirtualCluster(
        name="rollout_collection_cluster",
        bundle_ct_per_node_list=[gpus_per_node] * num_nodes,
        use_gpus=True,
        num_gpus_per_node=gpus_per_node,
        max_colocated_worker_groups=1,
        port_range_low=cluster_config.get("master_port_range_low"),
        port_range_high=cluster_config.get("master_port_range_high"),
    )
    print(f"  ✓ Ray cluster initialized: {num_nodes} nodes x {gpus_per_node} GPUs")

    # ------------------------------------------------------------------
    # Deferred vLLM init -> reserve server ports -> spin up NeMo-Gym while the
    # model loads (same overlap pattern as ppo.setup()).
    # ------------------------------------------------------------------
    generation_config = config.policy["generation"]
    generation_config["model_name"] = config.policy["model_name"]
    generation_config["vllm_kwargs"]["hf_overrides"] = config.policy.get(
        "hf_config_overrides", {}
    )

    setup_start = time.perf_counter()
    print("  ⚡ Deferred vLLM load: reserving ports for overlapped NeMo-Gym init")
    policy_generation = VllmGeneration(
        cluster=cluster, config=generation_config, defer_model_load=True
    )
    print(
        f"  ✓ Reserved {len(policy_generation.dp_openai_server_base_urls)} vLLM "
        f"server URLs: {policy_generation.dp_openai_server_base_urls}"
    )

    def _load_vllm():
        policy_generation.load_and_start()
        policy_generation.finish_generation()

    def _init_gym():
        return spinup_nemo_gym(
            config,
            policy_generation.dp_openai_server_base_urls,
            generation_config["model_name"],
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        vllm_future = executor.submit(_load_vllm)
        gym_future = executor.submit(_init_gym)
        vllm_future.result()
        nemo_gym = gym_future.result()
    print(f"  ✓ vLLM + NeMo-Gym ready in {time.perf_counter() - setup_start:.1f}s")

    # ------------------------------------------------------------------
    # Collect.
    # ------------------------------------------------------------------
    task_to_env = {"nemo_gym": nemo_gym}
    summary = collect_rollouts(
        policy_generation=policy_generation,
        tokenizer=tokenizer,
        task_to_env=task_to_env,
        master_config=config,
        dataset=train_dataset,
        collection=collection,
    )

    print("🛑 Shutting down generation workers...")
    try:
        policy_generation.shutdown()
    except Exception as e:
        print(f"⚠️ vLLM shutdown failed (non-fatal): {e}")

    if summary.get("aborted"):
        raise SystemExit(
            "Collection aborted after repeated consecutive failures; "
            "see logs above."
        )
    if os.environ.get("SLURM_ARRAY_TASK_ID") is not None and summary["remaining"] > 0:
        print(
            f"ℹ️ {summary['remaining']} assigned groups still missing "
            "(walltime/failures) — resubmit the same array to resume."
        )


if __name__ == "__main__":
    main()
