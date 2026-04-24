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

Creates a VllmGeneration instance, wraps it in a GenerationControlServer
(control plane + DP shard router), and optionally registers in K8sEndpointRegistry.

Usage:
    python examples/run_standalone_generation_server.py --config <config.yaml>

The server blocks forever. The training cluster drives all lifecycle
(init_collective, weight sync, generation requests) via HTTP.
"""

import argparse
import os
import pprint
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
from nemo_rl.models.generation.generation_control_server import GenerationControlServer
from nemo_rl.models.generation.vllm import VllmGeneration
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone vLLM generation server")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--port", type=int, default=8089, help="Control server port")
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

    num_nodes = configured_num_nodes or len(gpu_node_gpu_counts) or len(alive_nodes)
    gpus_per_node = configured_gpus_per_node or (
        num_gpus // max(len(gpu_node_gpu_counts) or num_nodes, 1)
    )

    assert num_gpus > 0, (
        f"No GPUs available in Ray cluster. Resources: {cluster_resources}"
    )
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

    # Start control-plane server (no DP router: clients talk to shards directly).
    server = GenerationControlServer(
        generation=generation,
        port=args.port,
    )
    server.start()

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
        f"\nGeneration server ready on port {args.port}.\n"
        f"  Router:  http://0.0.0.0:{args.port}/v1/completions\n"
        f"  Control: http://0.0.0.0:{args.port}/health\n"
        f"  DP shards: {generation.dp_openai_server_base_urls}\n"
        f"\nWaiting for training cluster..."
    )

    # Block forever — training cluster drives lifecycle via HTTP
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("Shutting down generation server...")
        generation.shutdown()


if __name__ == "__main__":
    main()
