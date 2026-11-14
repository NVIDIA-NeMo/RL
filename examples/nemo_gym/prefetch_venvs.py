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

"""Prefetch NeMo Gym internal venvs with a NeMo Gym dry run.

This complements nemo_rl/utils/prefetch_venvs.py (which prefetches Ray actor venvs)
by also triggering NeMo Gym's own internal venv creation for its servers (code_gen,
math, etc.). A dry run builds each server's venv and exits, so no policy model is needed.

Run it with an interpreter that has the nemo_gym extra installed. It imports only NeMo
Gym and nemo_rl/utils/config.py, so docker/Dockerfile can run it before the rest of the
NeMo RL source is copied into the image.
"""

import argparse
import multiprocessing
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

# Keys under env.nemo_gym that NeMo RL reads itself and does not forward to NeMo Gym.
_NEMO_RL_ONLY_KEYS = ("effort_levels", "pad_dynamic_image_shapes")


def build_dry_run_config(
    nemo_gym_section: dict[str, Any],
    uv_cache_dir: str | None,
    uv_venv_dir: str | None,
) -> dict[str, Any]:
    """Turn a recipe's env.nemo_gym section into a NeMo Gym dry-run config.

    Args:
        nemo_gym_section: The resolved env.nemo_gym section of a NeMo RL config.
        uv_cache_dir: uv cache directory for NeMo Gym, or None for NeMo Gym's default.
        uv_venv_dir: Directory for the server venvs, or None for NeMo Gym's default.

    Returns:
        The initial global config for a NeMo Gym dry run.
    """
    config = dict(nemo_gym_section)
    for key in _NEMO_RL_ONLY_KEYS:
        config.pop(key, None)
    config["dry_run"] = True
    # A dry run talks to no model. NeMo Gym still needs these to resolve the config.
    config["policy_model_name"] = "dummy-model"
    config["policy_api_key"] = "dummy_key"
    config["policy_base_url"] = ["http://localhost:8000"]
    if uv_cache_dir is not None:
        config.setdefault("uv_cache_dir", uv_cache_dir)
    if uv_venv_dir is not None:
        config.setdefault("uv_venv_dir", uv_venv_dir)
    return config


def _uv_cache_dir() -> str | None:
    """Return uv's cache directory inside the container, or None outside one."""
    if not os.environ.get("NRL_CONTAINER"):
        return None
    return subprocess.check_output(["uv", "cache", "dir"]).decode().strip()


def _prefetch_one(config_path: str) -> None:
    """Build the NeMo Gym server venvs for one config."""
    from nemo_gym.cli import GlobalConfigDictParserConfig, RunHelper

    register_omegaconf_resolvers()
    config = OmegaConf.to_container(load_config(config_path), resolve=True)
    dry_run_config = build_dry_run_config(
        config["env"]["nemo_gym"],
        uv_cache_dir=_uv_cache_dir(),
        uv_venv_dir=os.environ.get("NEMO_GYM_VENV_DIR"),
    )

    # NeMo Gym's server setup scripts inherit this environment. They shell out to
    # `uv pip install --no-cache` (e.g. swe_agents' r2e_gym.sh), which uv refuses to
    # combine with symlink installs, so pin hardlink whatever the caller set.
    os.environ["UV_LINK_MODE"] = "hardlink"
    os.environ["VIRTUAL_ENV"] = sys.prefix
    os.environ["UV_PROJECT_ENVIRONMENT"] = sys.prefix

    run_helper = RunHelper()
    # Returns once every server's venv is built, and raises if any setup failed.
    run_helper.start(
        global_config_dict_parser_config=GlobalConfigDictParserConfig(
            dotenv_path=Path(__file__).absolute().parents[2] / "nemo_gym_env.yaml",
            initial_global_config_dict=DictConfig(dry_run_config),
            skip_load_from_cli=True,
        )
    )
    run_helper.shutdown()


def prefetch_nemo_gym_venvs(config_paths: list[str]) -> None:
    """Prefetch NeMo Gym venvs for each config with a NeMo Gym dry run.

    Args:
        config_paths: List of paths to NeMo RL config files that contain
            an env.nemo_gym section.
    """
    # One process per config: NeMo Gym keeps its global config and head server per process.
    spawn = multiprocessing.get_context("spawn")
    succeeded = []
    failed = []

    for config_path in config_paths:
        print(f"\n{'=' * 60}")
        print(f"Processing config: {config_path}")
        print("=" * 60)

        process = spawn.Process(target=_prefetch_one, args=(config_path,))
        process.start()
        process.join()
        if process.exitcode == 0:
            succeeded.append(config_path)
            print(f"Done with config: {config_path}")
        else:
            print(f"Error processing {config_path}: exit code {process.exitcode}")
            failed.append((config_path, f"exit code {process.exitcode}"))

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
        description="Prefetch NeMo Gym internal venvs with a NeMo Gym dry run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Prefetch venvs for a single config
  uv run --extra nemo_gym python examples/nemo_gym/prefetch_venvs.py \\
    examples/nemo_gym/grpo_workplace_assistant_nemotron_nano_v2_9b.yaml

  # Prefetch venvs for multiple configs sequentially
  uv run --extra nemo_gym python examples/nemo_gym/prefetch_venvs.py \\
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
