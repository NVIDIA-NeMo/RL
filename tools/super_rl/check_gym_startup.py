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


"""CPU-only Gym startup check; start required sandbox backends before running."""

import argparse
from contextlib import suppress
import hashlib
import json
import os
from pathlib import Path
import signal

from omegaconf import DictConfig, OmegaConf


def startup_config(config: Path, runtime: Path) -> DictConfig:
    recipe = OmegaConf.load(config)
    graph = OmegaConf.create(OmegaConf.to_container(recipe.env.nemo_gym, resolve=False))
    graph.policy_model_name = recipe.policy.model_name
    graph.policy_api_key = "unused-cpu-startup-check"
    graph.policy_base_url = ["http://127.0.0.1:1/v1"]
    graph.default_host = "127.0.0.1"
    # This test starts real Gym apps but deliberately has no GPU model backend.
    # Keep the training recipe unchanged; inference readiness is a separate gate.
    graph.model_endpoint_readiness_timeout_seconds = 0
    graph.nemo_gym_log_dir = str(runtime / "logs")
    graph.cache_dir = str(runtime / "cache")
    graph.results_dir = str(runtime / "results")
    return graph


def check(config: Path, gym: Path, runtime: Path, cpus: int, timeout: int) -> dict:
    import ray
    from nemo_gym.cli.env import RunHelper
    from nemo_gym.global_config import GlobalConfigDictParserConfig

    runtime.mkdir(parents=True, exist_ok=False)
    graph = startup_config(config, runtime)
    original_cwd = Path.cwd()
    os.chdir(gym)
    helper = RunHelper()

    def expired(signum, frame):
        raise TimeoutError("Gym CPU service startup exceeded its deadline")

    previous_handler = signal.signal(signal.SIGALRM, expired)
    signal.alarm(timeout)
    try:
        ray.init(address="local", num_cpus=cpus, num_gpus=0, include_dashboard=False)
        graph.ray_head_node_address = ray.get_runtime_context().gcs_address
        try:
            helper.start(
                GlobalConfigDictParserConfig(
                    initial_global_config_dict=graph,
                    skip_load_from_cli=True,
                    skip_load_from_dotenv=True,
                )
            )
            helper.poll()
        except BaseException:
            with suppress(Exception):
                helper.shutdown()
            raise
        else:
            helper.shutdown()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
        ray.shutdown()
        os.chdir(original_cwd)
    return {
        "complete": True,
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "all_configured_gym_services_started": True,
        "gpus": 0,
        "scope": "Gym constructors, local assets, service startup and dependency readiness probes; no policy/judge inference, task verification, rewards or optimizer updates",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gym", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--cpus", type=int, required=True)
    parser.add_argument("--timeout", type=int, required=True)
    args = parser.parse_args()
    if args.cpus < 1 or args.timeout < 1:
        parser.error("--cpus and --timeout must be positive")
    config, gym, runtime = (
        args.config.resolve(),
        args.gym.resolve(),
        args.runtime.resolve(),
    )
    receipt = check(config, gym, runtime, args.cpus, args.timeout)
    with (runtime / "result.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
