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

"""Prefetch NeMo Gym internal venvs by doing a dry run of NemoGym initialization.

This complements nemo_rl/utils/prefetch_venvs.py (which prefetches Ray actor venvs)
by also triggering NeMo Gym's own internal venv creation for its servers (code_gen,
math, etc.). It reuses the real code path (create_env -> NemoGym.__init__) with
dry_run=True so no actual policy model is needed.
"""

import argparse
import sys

import ray
from omegaconf import OmegaConf

from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.nemo_gym import (
    NemoGymConfig,
    get_nemo_gym_uv_cache_dir,
    get_nemo_gym_venv_dir,
)
from nemo_rl.environments.utils import create_env
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers


def prefetch_nemo_gym_venvs(config_paths: list[str]) -> None:
    """Prefetch NeMo Gym venvs for each config by doing a dry-run initialization.

    Args:
        config_paths: List of paths to NeMo RL config files that contain
            an env.nemo_gym section.
    """
    register_omegaconf_resolvers()
    init_ray()

    succeeded = []
    failed = []

    for config_path in config_paths:
        print(f"\n{'=' * 60}")
        print(f"Processing config: {config_path}")
        print("=" * 60)

        try:
            config = load_config(config_path)
            config = OmegaConf.to_container(config, resolve=True)

            nemo_gym_dict = dict(config["env"]["nemo_gym"])
            nemo_gym_dict["dry_run"] = True
            uv_cache_dir = get_nemo_gym_uv_cache_dir()
            if uv_cache_dir is not None:
                nemo_gym_dict.setdefault("uv_cache_dir", uv_cache_dir)
            uv_venv_dir = get_nemo_gym_venv_dir()
            if uv_venv_dir is not None:
                nemo_gym_dict.setdefault("uv_venv_dir", uv_venv_dir)

            nemo_gym_config = NemoGymConfig(
                model_name="dummy-model",
                base_urls=["http://localhost:8000"],
                initial_global_config_dict=nemo_gym_dict,
            )

            print("Creating NeMo Gym environment (dry_run=True)...")
            nemo_gym = create_env(env_name="nemo_gym", env_config=nemo_gym_config)

            print("Waiting for NeMo Gym to finish initialization...")
            ray.get(nemo_gym.health_check.remote())
            print("NeMo Gym initialized successfully.")

            # TODO: Hangs... (DONT MERGE UNTIL FIXED - but kill may be fine)
            # print("Shutting down NeMo Gym environment...")
            # ray.get(nemo_gym.shutdown.remote())
            print("Killing NeMo Gym actor...")
            ray.kill(nemo_gym)

            succeeded.append(config_path)
            print(f"Done with config: {config_path}")

        except Exception as e:
            print(f"Error processing {config_path}: {e}")
            failed.append((config_path, str(e)))

    print(f"\n{'=' * 60}")
    print("NeMo Gym venv prefetch summary")
    print("=" * 60)
    print(f"  Succeeded: {len(succeeded)}")
    for path in succeeded:
        print(f"    - {path}")
    if failed:
        print(f"  Failed: {len(failed)}")
        for path, err in failed:
            print(f"    - {path}: {err}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prefetch NeMo Gym internal venvs via dry-run initialization.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Prefetch venvs for a single config
  uv run python examples/nemo_gym/prefetch_venvs.py \\
    examples/nemo_gym/grpo_workplace_assistant_nemotron_nano_v2_9b.yaml

  # Prefetch venvs for multiple configs sequentially
  uv run python examples/nemo_gym/prefetch_venvs.py \\
    examples/nemo_gym/grpo_workplace_assistant_nemotron_nano_v2_9b.yaml \\
    examples/nemo_gym/grpo_qwen3_30ba3b_instruct.yaml
""",
    )
    parser.add_argument(
        "configs",
        nargs="+",
        help="One or more NeMo RL config file paths containing an env.nemo_gym section.",
    )
    args = parser.parse_args()

    prefetch_nemo_gym_venvs(args.configs)
