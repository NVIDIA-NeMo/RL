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
"""Standalone NeMo Gym server for disaggregated RL-Gym deployments.

Starts the Gym servers (head, agent, model, resource) as a long-running
HTTP service that can be deployed on a separate Kubernetes cluster/pod.

Service discovery uses a shared K8s ConfigMap (endpoint registry). The Gym
server registers its address and waits for the RL cluster to register vLLM URLs.

The --config-yaml should contain the env.nemo_gym section from the GRPO config,
which includes config_paths and server overrides. Paths in config_paths are
resolved relative to the Gym repo root (PARENT_DIR).

Usage:
    uv run --extra nemo_gym python -m nemo_rl.distributed.standalone_gym_server \
        --job-id my-job \
        --port 9090 \
        --model-name Qwen/Qwen3-0.6B \
        --config-yaml /path/to/gym_config.yaml

    # Or with static vLLM URLs (skips registry for vLLM discovery):
    uv run --extra nemo_gym python -m nemo_rl.distributed.standalone_gym_server \
        --port 9090 \
        --model-name Qwen/Qwen3-0.6B \
        --vllm-base-urls http://10.0.0.1:8000/v1 \
        --config-yaml /path/to/gym_config.yaml
"""

import argparse
import json
import signal
import sys
from pathlib import Path

from nemo_gym.cli import GlobalConfigDictParserConfig, RunHelper
from nemo_gym.server_utils import HEAD_SERVER_KEY_NAME
from omegaconf import DictConfig, OmegaConf


def _get_node_ip() -> str:
    import socket

    return socket.gethostbyname(socket.gethostname())


def main():
    parser = argparse.ArgumentParser(description="Standalone NeMo Gym server")
    parser.add_argument("--port", type=int, default=9090, help="Head server port")
    parser.add_argument(
        "--job-id",
        type=str,
        default=None,
        help="Job ID for K8s ConfigMap endpoint registry. "
        "If set, registers this server's address and discovers vLLM URLs via the registry.",
    )
    parser.add_argument(
        "--vllm-base-urls",
        type=str,
        nargs="+",
        default=None,
        help="vLLM HTTP server base URLs. If not set and --job-id is provided, "
        "URLs are discovered via the K8s endpoint registry.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Policy model name (e.g., Qwen/Qwen3-0.6B)",
    )
    parser.add_argument(
        "--config-yaml",
        type=str,
        default=None,
        help="Path to a YAML with the env.nemo_gym config section (config_paths, server overrides).",
    )
    parser.add_argument(
        "--dotenv-path",
        type=str,
        default=None,
        help="Path to the env.yaml dotenv file",
    )
    args = parser.parse_args()

    # Resolve vLLM URLs — either from CLI or from K8s endpoint registry.
    vllm_base_urls = args.vllm_base_urls
    if vllm_base_urls is None and args.job_id:
        from nemo_rl.distributed.k8s_endpoint_registry import K8sEndpointRegistry

        registry = K8sEndpointRegistry(job_id=args.job_id)

        # Register our address first so RL can find us.
        node_ip = _get_node_ip()
        gym_url = f"http://{node_ip}:{args.port}"
        registry.set("gym_head_server", gym_url)
        print(f"Registered gym_head_server = {gym_url}")

        # Wait for RL to register vLLM URLs.
        print("Waiting for vLLM base URLs from RL cluster...")
        vllm_base_urls = json.loads(registry.get("vllm_base_urls"))
        print(f"Discovered vLLM base URLs: {vllm_base_urls}")
    elif vllm_base_urls is None:
        print("Error: --vllm-base-urls or --job-id is required", file=sys.stderr)
        sys.exit(1)

    # Build the global config dict from the config YAML.
    # We use resolve=False to avoid failing on unresolvable interpolations
    # (e.g., ${cluster.num_nodes}) that are only valid in the full GRPO config.
    initial_global_config_dict = {}
    if args.config_yaml:
        loaded = OmegaConf.load(args.config_yaml)
        initial_global_config_dict = OmegaConf.to_container(loaded, resolve=False)

        def _strip_interpolations(d):
            if isinstance(d, dict):
                return {
                    k: _strip_interpolations(v)
                    for k, v in d.items()
                    if not (isinstance(v, str) and "${" in v)
                }
            if isinstance(d, list):
                return [_strip_interpolations(item) for item in d]
            return d

        initial_global_config_dict = _strip_interpolations(initial_global_config_dict)

    # Remove RL-specific keys that don't apply to standalone mode.
    initial_global_config_dict.pop("is_trajectory_collection", None)
    initial_global_config_dict.pop("rollout_max_attempts_to_avoid_lp_nan", None)

    # Bind to actual pod IP instead of localhost so remote RL cluster can reach us.
    initial_global_config_dict["use_absolute_ip"] = True

    # Set policy/connection config.
    initial_global_config_dict["policy_model_name"] = args.model_name
    initial_global_config_dict["policy_api_key"] = "dummy_key"
    initial_global_config_dict["policy_base_url"] = vllm_base_urls
    initial_global_config_dict.setdefault(
        "global_aiohttp_connector_limit_per_host", 16_384
    )
    initial_global_config_dict.setdefault("global_aiohttp_connector_limit", 65_536)

    initial_global_config_dict[HEAD_SERVER_KEY_NAME] = {
        "host": "0.0.0.0",
        "port": args.port,
    }

    # Determine dotenv path.
    dotenv_path = Path(args.dotenv_path) if args.dotenv_path else None

    print(f"Starting standalone NeMo Gym server on port {args.port}")
    print(f"vLLM base URLs: {vllm_base_urls}")
    print(f"Model: {args.model_name}")
    if "config_paths" in initial_global_config_dict:
        print(f"Config paths: {initial_global_config_dict['config_paths']}")

    rh = RunHelper()
    rh.start(
        global_config_dict_parser_config=GlobalConfigDictParserConfig(
            dotenv_path=dotenv_path,
            initial_global_config_dict=DictConfig(initial_global_config_dict),
            skip_load_from_cli=True,
        )
    )

    rh.display_server_instance_info()
    print(f"\nStandalone Gym server is running on port {args.port}")
    print("Press Ctrl+C to stop.")

    def handle_signal(signum, frame):
        print(f"\nReceived signal {signum}, shutting down...")
        rh.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)
    signal.pause()


if __name__ == "__main__":
    main()
