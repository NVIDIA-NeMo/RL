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
"""Standalone vLLM generation server for disaggregated RL training.

Entry point for the generation K8s RayCluster. Analogous to nemo_gym.standalone_server.

Creates a VllmGeneration instance and wraps it in a single GenerationRouter
that exposes both the data plane (``/v1/{completions,chat/completions}``,
proxied to per-shard vLLM endpoints with cordon + replay) and the control
plane (weight sync, lifecycle, refit gate, metrics) on one port (8089).

Usage:
    python examples/run_standalone_generation_server.py --config <config.yaml>

The server blocks forever. The training cluster drives all lifecycle
(init_collective, weight sync, generation requests) via HTTP.
"""

import argparse
import os
import pprint
import signal
import sys
import time

import ray
from omegaconf import OmegaConf

from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.distributed.virtual_cluster import (
    RayVirtualCluster,
    _get_node_ip_local,
    init_ray,
)
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.fault_inject import maybe_launch_fault_injector
from nemo_rl.models.generation.generation_router import GenerationRouter
from nemo_rl.models.generation.vllm import VllmGeneration
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone vLLM generation server")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument(
        "--port",
        type=int,
        default=8089,
        help="Unified router port (data plane + control plane).",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (default: all available)",
    )
    args, overrides = parser.parse_known_args()
    return args, overrides


def main():
    register_omegaconf_resolvers()
    args, overrides = parse_args()

    # Load config
    config = load_config(args.config)
    if overrides:
        config = parse_hydra_overrides(config, overrides)
    config = OmegaConf.to_container(config, resolve=True)

    print("Standalone Generation Server — config:")
    pprint.pprint(config)

    # Extract generation config
    policy_config = config["policy"]
    generation_config = policy_config["generation"]

    # Force async engine and HTTP server exposure
    generation_config["vllm_cfg"]["async_engine"] = True
    generation_config["vllm_cfg"]["expose_http_server"] = True
    generation_config["model_name"] = policy_config["model_name"]

    # Non-colocated since this IS the standalone inference server
    generation_config.setdefault("colocated", {})
    generation_config["colocated"]["enabled"] = False

    # Initialize Ray
    init_ray()

    # Setup tokenizer and configure generation
    tokenizer = get_tokenizer(policy_config["tokenizer"])
    generation_config = configure_generation_config(generation_config, tokenizer)

    # Detect available GPUs from the Ray cluster. Count only GPU-capable Ray nodes,
    # since KubeRay head pods are often CPU-only and would otherwise skew the division.
    cluster_resources = ray.cluster_resources()
    num_gpus = int(cluster_resources.get("GPU", 0))
    alive_nodes = [node for node in ray.nodes() if node.get("Alive", False)]
    gpu_node_gpu_counts = [
        int(node.get("Resources", {}).get("GPU", 0))
        for node in alive_nodes
        if int(node.get("Resources", {}).get("GPU", 0)) > 0
    ]

    colocated_resources = generation_config.get("colocated", {}).get("resources", {})
    configured_num_nodes = colocated_resources.get("num_nodes")
    configured_gpus_per_node = colocated_resources.get("gpus_per_node")

    # When autoscaler v2 + minReplicas=0 is in play, no GPU pods are up at
    # boot — placement_group(...) will trigger provisioning. Trust the
    # config (and --num-gpus) instead of inspecting live cluster resources;
    # pg.wait() blocks downstream until the autoscaler delivers.
    if num_gpus > 0:
        num_nodes = configured_num_nodes or len(gpu_node_gpu_counts) or len(alive_nodes)
        gpus_per_node = configured_gpus_per_node or (
            num_gpus // max(len(gpu_node_gpu_counts) or num_nodes, 1)
        )
    else:
        # Pre-autoscaler bring-up. Require explicit shape from config or CLI.
        if not (configured_num_nodes and configured_gpus_per_node):
            raise RuntimeError(
                "No GPUs visible yet (autoscaler hasn't provisioned). "
                "Set policy.generation.colocated.resources.{num_nodes,gpus_per_node} "
                "in the standalone gen config so we can request the right "
                "placement group up front."
            )
        num_nodes = configured_num_nodes
        gpus_per_node = configured_gpus_per_node
        num_gpus = num_nodes * gpus_per_node

    if args.num_gpus is not None:
        num_gpus = min(args.num_gpus, num_gpus)
        gpus_per_node = min(gpus_per_node, num_gpus)
    print(f"Using {num_gpus} GPUs across {num_nodes} nodes ({gpus_per_node} per node)")

    # Create virtual cluster for inference
    cluster = RayVirtualCluster(
        name="generation_server_cluster",
        bundle_ct_per_node_list=[gpus_per_node] * num_nodes,
        use_gpus=True,
        num_gpus_per_node=gpus_per_node,
        max_colocated_worker_groups=1,
    )

    # Create VllmGeneration (spawns Ray actor workers on GPU nodes)
    print("Initializing VllmGeneration...")
    t0 = time.perf_counter()
    generation = VllmGeneration(cluster=cluster, config=generation_config)
    generation.finish_generation()  # Reset prefix cache, matches grpo.py init_vllm() pattern
    print(f"VllmGeneration initialized in {time.perf_counter() - t0:.1f}s")

    # `policy.generation.router` lets the recipe override health-poll +
    # cordon thresholds without code changes. Defaults live in
    # GenerationRouter.__init__; only overrides take effect.
    router_cfg = (generation_config or {}).get("router") or {}
    router_kwargs = {
        k: router_cfg[k]
        for k in (
            "health_poll_interval_s",
            "health_timeout_s",
            "failure_threshold",
            "join_success_threshold",
            "proxy_timeout_s",
            "proxy_failure_threshold",
            "proxy_pool_limit_total",
            "proxy_pool_limit_per_host",
            "uvicorn_backlog",
            "uvicorn_keep_alive_s",
            "auto_backfill",
            "backfill_target",
            "backfill_max_concurrent",
            "pg_ready_timeout_s",
        )
        if k in router_cfg
    }
    if router_kwargs:
        print(f"GenerationRouter overrides: {router_kwargs}")
    router = GenerationRouter(port=args.port, generation=generation, **router_kwargs)
    # Per-shard world size = TP * PP for one DP group. Used by
    # /current_gen_world_size when the train driver re-inits the
    # weight-sync NCCL group on shard add/remove.
    per_shard_ws = (
        generation_config["vllm_cfg"]["tensor_parallel_size"]
        * generation_config["vllm_cfg"]["pipeline_parallel_size"]
    )
    # Group worker handles + placement groups by DP shard so the router
    # owns the lifecycle (kill actors / free PG on remove_shard). The
    # bundle indices in worker_metadata map onto the cluster's PG list
    # via the same `node_idx` the worker group used during init.
    worker_metadata = generation.worker_group.worker_metadata
    actors = generation.worker_group.workers
    cluster_pgs = cluster.get_placement_groups()
    actor_handles_by_shard: dict[str, list] = {}
    pg_by_shard: dict[str, object] = {}
    worker_indices_by_shard: dict[str, list[int]] = {}
    for worker_idx, (actor, meta) in enumerate(zip(actors, worker_metadata)):
        shard_id = f"dp-{meta['dp_shard_idx']}"
        actor_handles_by_shard.setdefault(shard_id, []).append(actor)
        worker_indices_by_shard.setdefault(shard_id, []).append(worker_idx)
        # node_idx is the index into the cluster's per-node PG list when
        # multiple PGs were created (one per node). When the cluster
        # uses a single unified PG (cross-node parallelism), node_idx
        # still resolves to the same PG.
        node_idx = meta["node_idx"]
        if node_idx < len(cluster_pgs):
            pg_by_shard.setdefault(shard_id, cluster_pgs[node_idx])
    shard_entries = [
        (f"dp-{i}", url)
        for i, url in enumerate(generation.dp_openai_server_base_urls)
        if url is not None
    ]
    router.register_shards(
        shard_entries,
        per_shard_world_size=per_shard_ws,
        actor_handles_by_shard=actor_handles_by_shard,
        pg_by_shard=pg_by_shard,
        worker_indices_by_shard=worker_indices_by_shard,
        generation=generation,
    )
    router.start()
    router_url = f"http://{_get_node_ip_local()}:{router.port}"

    # Register in K8s endpoint registry if disagg_job_id is set
    disagg_job_id = os.environ.get("DISAGG_JOB_ID") or generation_config.get(
        "disagg_job_id"
    )
    if disagg_job_id:
        import json

        from nemo_rl.distributed.k8s_endpoint_registry import K8sEndpointRegistry

        node_ip = _get_node_ip_local()
        registry = K8sEndpointRegistry(job_id=disagg_job_id)
        registry.create(owner_raycluster_name=os.environ.get("RAY_CLUSTER_NAME"))
        registry.set("generation_server_url", f"http://{node_ip}:{args.port}")
        registry.set("generation_world_size", str(cluster.world_size()))
        registry.set(
            "dp_openai_server_base_urls",
            json.dumps(generation.dp_openai_server_base_urls),
        )
        # Backward-compat alias: NemoGym's standalone_server reads
        # `vllm_base_urls`. Keep publishing both keys until gym consumers
        # migrate to the new name.
        registry.set(
            "vllm_base_urls",
            json.dumps(generation.dp_openai_server_base_urls),
        )
        print(f"Registered in K8sEndpointRegistry (job_id={disagg_job_id})")

    print(
        f"\nGeneration server ready.\n"
        f"  Router:  {router_url}/v1/completions  (status: /shards, control: /init_collective ...)\n"
        f"  DP shards: {generation.dp_openai_server_base_urls}\n"
        f"\nWaiting for training cluster..."
    )

    # Optionally fire the RL-412 fault injector (one-shot Ray actor).
    maybe_launch_fault_injector(config, router_url)

    # SIGTERM-driven graceful shutdown so `nrl-k8s --replace`'s stop_job
    # leaves no orphan placement groups or worker actors. Ray's stop_job
    # only sends SIGTERM and updates the dashboard status — actor / PG
    # GC waits for the driver to officially exit AND for GCS heartbeat
    # timeouts (~30s), which races against the next daemon's
    # _init_placement_groups. By explicitly ray.kill'ing every worker
    # in worker_group + remove_placement_group'ing the cluster's PGs
    # before sys.exit, this driver leaves nothing for the next run to
    # trip over. SIGINT (Ctrl-C) goes through the same path.
    def _shutdown(signo, _frame):
        sig_name = signal.Signals(signo).name if signo else "exit"
        print(f"[gen-daemon] received {sig_name}; reaping worker actors + PGs", flush=True)
        try:
            wg = getattr(generation, "worker_group", None)
            if wg is not None:
                for idx, actor in enumerate(getattr(wg, "_workers", []) or []):
                    try:
                        ray.kill(actor, no_restart=True)
                        print(f"[gen-daemon]   killed worker {idx}", flush=True)
                    except Exception as e:  # noqa: BLE001
                        print(f"[gen-daemon]   ray.kill worker {idx} raised {e}", flush=True)
            from ray.util.placement_group import remove_placement_group

            for pg in cluster.get_placement_groups() or []:
                try:
                    remove_placement_group(pg)
                except Exception as e:  # noqa: BLE001
                    print(f"[gen-daemon]   remove_placement_group raised {e}", flush=True)
            try:
                generation.shutdown()
            except Exception as e:  # noqa: BLE001
                print(f"[gen-daemon]   generation.shutdown raised {e}", flush=True)
        finally:
            sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    # Block forever — training cluster drives lifecycle via HTTP. The
    # signal handlers above tear everything down on SIGTERM / Ctrl-C so
    # the next `nrl-k8s --replace` lands on a clean Ray state.
    while True:
        time.sleep(60)


if __name__ == "__main__":
    main()
