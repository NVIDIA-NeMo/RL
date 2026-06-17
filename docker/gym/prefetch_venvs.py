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

"""Pre-bake NeMo-Gym virtual environments from a NeMo-RL training config.

Called during Docker image build so that the server venvs exist in the
read-only container image at runtime. With skip_venv_if_present=true in the
training YAML, runtime code finds the pre-baked venvs and skips re-creation.
"""

import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from platform import python_version

import yaml
from omegaconf import DictConfig, OmegaConf


def _deep_merge(base: dict, override: dict) -> None:
    for key, val in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(val, dict):
            _deep_merge(base[key], val)
        else:
            base[key] = val


def load_yaml_raw(config_path: Path) -> dict:
    """Load a NeMo-RL YAML, resolving 'defaults' inheritance without interpolation."""
    with open(config_path) as f:
        config = yaml.safe_load(f) or {}

    defaults = config.pop("defaults", None)
    if defaults is None:
        return config
    if isinstance(defaults, str):
        defaults = [defaults]

    merged: dict = {}
    for default in defaults:
        parent = load_yaml_raw(config_path.parent / default)
        _deep_merge(merged, parent)
    _deep_merge(merged, config)
    return merged


def find_servers_with_entrypoints(config: DictConfig) -> list[tuple[str, str]]:
    """Return (server_type, server_name) pairs for all servers with entrypoints."""
    servers = []
    for _instance_key, instance_val in config.items():
        if not isinstance(instance_val, DictConfig):
            continue
        for server_type, server_type_val in instance_val.items():
            if not isinstance(server_type_val, DictConfig):
                continue
            for server_name, server_val in server_type_val.items():
                if isinstance(server_val, DictConfig) and "entrypoint" in server_val:
                    servers.append((server_type, server_name))
    return servers


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <nemo_rl_config.yaml>", file=sys.stderr)
        sys.exit(1)

    config_path = Path(sys.argv[1])
    nemo_rl_config = load_yaml_raw(config_path)

    nemo_gym_section = nemo_rl_config.get("env", {}).get("nemo_gym", {})
    config_paths = nemo_gym_section.get("config_paths", [])

    if not config_paths:
        print("[prefetch_venvs] No config_paths found in env.nemo_gym, nothing to do.")
        return

    from nemo_gym import PARENT_DIR
    from nemo_gym.cli_setup_command import setup_env_command
    from nemo_gym.global_config import (
        HEAD_SERVER_DEPS_KEY_NAME,
        PIP_INSTALL_VERBOSE_KEY_NAME,
        PYTHON_VERSION_KEY_NAME,
        SKIP_VENV_IF_PRESENT_KEY_NAME,
        UV_PIP_SET_PYTHON_KEY_NAME,
        UV_VENV_DIR_KEY_NAME,
    )

    uv_venv_dir = nemo_gym_section.get("uv_venv_dir", str(PARENT_DIR))

    global_cfg: DictConfig = OmegaConf.create(
        {
            HEAD_SERVER_DEPS_KEY_NAME: [
                f"ray[default]=={version('ray')}",
                f"openai=={version('openai')}",
            ],
            UV_VENV_DIR_KEY_NAME: uv_venv_dir,
            PYTHON_VERSION_KEY_NAME: python_version(),
            SKIP_VENV_IF_PRESENT_KEY_NAME: False,
            UV_PIP_SET_PYTHON_KEY_NAME: False,
            PIP_INSTALL_VERBOSE_KEY_NAME: False,
        }
    )

    seen: set[tuple[str, str]] = set()
    for cfg_path_str in config_paths:
        gym_cfg_path = PARENT_DIR / cfg_path_str
        if not gym_cfg_path.exists():
            print(
                f"[prefetch_venvs] WARNING: gym config not found: {gym_cfg_path}",
                file=sys.stderr,
            )
            continue

        gym_config = OmegaConf.load(gym_cfg_path)
        for server_type, server_name in find_servers_with_entrypoints(gym_config):
            if (server_type, server_name) in seen:
                continue
            seen.add((server_type, server_name))

            dir_path = PARENT_DIR / Path(server_type, server_name)
            if not dir_path.exists():
                print(
                    f"[prefetch_venvs] WARNING: server dir not found: {dir_path}",
                    file=sys.stderr,
                )
                continue

            print(f"[prefetch_venvs] Creating venv for {server_type}:{server_name} ...")
            setup_cmd = setup_env_command(
                dir_path, global_cfg, f"{server_type}:{server_name}"
            )
            print(f"[prefetch_venvs] Setup command: {setup_cmd}")
            result = subprocess.run(["bash", "-c", setup_cmd])
            if result.returncode != 0:
                print(
                    f"[prefetch_venvs] ERROR: venv setup failed for {server_type}:{server_name}",
                    file=sys.stderr,
                )
                sys.exit(result.returncode)

    print("[prefetch_venvs] All venvs created successfully.")


if __name__ == "__main__":
    main()
