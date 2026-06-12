#!/usr/bin/env python3
"""Prefetch NeMo Gym server venvs from a NeMo-RL config.

Run this inside the same container image used for the training job. It parses
the configured NeMo Gym servers and runs only their setup commands, creating the
per-server uv virtualenvs without launching server entrypoints or touching Ray.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

repo_root = Path(os.environ.get("NEMO_RL_REPO", Path.cwd())).resolve()
gym_root = repo_root / "3rdparty" / "Gym-workspace" / "Gym"
if gym_root.exists():
    sys.path.insert(0, str(gym_root))

from nemo_gym import PARENT_DIR as GYM_PARENT_DIR
from nemo_gym.cli_setup_command import setup_env_command
from nemo_gym.global_config import (
    HEAD_SERVER_KEY_NAME,
    NEMO_GYM_RESERVED_TOP_LEVEL_KEYS,
    GlobalConfigDictParserConfig,
    get_global_config_dict,
)
import nemo_rl.environments.nemo_gym as nemo_gym_env_module
from nemo_rl.utils.config import load_config

OmegaConf.register_new_resolver("mul", lambda a, b: a * b, replace=True)


def _uv_cache_dir() -> str | None:
    if not os.environ.get("NRL_CONTAINER"):
        return None
    try:
        return subprocess.check_output(["uv", "cache", "dir"], text=True).strip()
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="NeMo-RL config with env.nemo_gym")
    parser.add_argument(
        "--venv-dir",
        required=True,
        help="Shared directory for Gym server venvs, visible at the same path on all nodes",
    )
    parser.add_argument("--log-dir", default=None, help="Unused; kept for wrapper compatibility")
    args = parser.parse_args()

    venv_dir = Path(args.venv_dir).absolute()
    venv_dir.mkdir(parents=True, exist_ok=True)

    os.environ["NEMO_GYM_VENV_DIR"] = str(venv_dir)
    os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")

    config = load_config(args.config)
    config_dict = OmegaConf.to_container(config, resolve=True)
    nemo_gym_dict = dict(config_dict["env"]["nemo_gym"])
    nemo_gym_dict["uv_venv_dir"] = str(venv_dir)
    nemo_gym_dict.setdefault("skip_venv_if_present", True)
    nemo_gym_dict.setdefault("policy_model_name", "dummy-model")
    nemo_gym_dict.setdefault("policy_api_key", "dummy_key")
    nemo_gym_dict.setdefault("policy_base_url", ["http://127.0.0.1:1/v1"])
    nemo_gym_dict.setdefault("default_host", "127.0.0.1")
    nemo_gym_dict.setdefault("port_range_low", 15001)
    nemo_gym_dict.setdefault("port_range_high", 20000)
    nemo_gym_dict.setdefault("ray_head_node_address", "127.0.0.1:6379")
    nemo_gym_dict.setdefault(
        HEAD_SERVER_KEY_NAME,
        {"host": "127.0.0.1", "port": nemo_gym_dict["port_range_low"]},
    )
    uv_cache_dir = _uv_cache_dir()
    if uv_cache_dir is not None:
        nemo_gym_dict.setdefault("uv_cache_dir", uv_cache_dir)

    global_config_dict = get_global_config_dict(
        global_config_dict_parser_config=GlobalConfigDictParserConfig(
            dotenv_path=Path(nemo_gym_env_module.__file__).with_name(
                "nemo_gym_env.yaml"
            ),
            initial_global_config_dict=DictConfig(nemo_gym_dict),
            skip_load_from_cli=True,
        )
    )

    top_level_paths = [
        k for k in global_config_dict.keys() if k not in NEMO_GYM_RESERVED_TOP_LEVEL_KEYS
    ]
    installed = 0
    skipped = 0

    for top_level_path in top_level_paths:
        server_config_dict = global_config_dict[top_level_path]
        if not isinstance(server_config_dict, DictConfig):
            continue

        first_key = list(server_config_dict)[0]
        server_config_dict = server_config_dict[first_key]
        if not isinstance(server_config_dict, DictConfig):
            continue

        second_key = list(server_config_dict)[0]
        server_config_dict = server_config_dict[second_key]
        if not isinstance(server_config_dict, DictConfig):
            continue
        if "entrypoint" not in server_config_dict:
            continue

        dir_path = GYM_PARENT_DIR / Path(first_key, second_key)
        venv_python = venv_dir / first_key / second_key / ".venv" / "bin" / "python"
        if venv_python.exists():
            print(f"[skip] {top_level_path}: {venv_python}", flush=True)
            skipped += 1
            continue

        command = setup_env_command(dir_path, global_config_dict, top_level_path)
        env = os.environ.copy()
        env["UV_CACHE_DIR"] = str(global_config_dict["uv_cache_dir"])
        print(f"[install] {top_level_path}: {venv_python}", flush=True)
        subprocess.run(["bash", "-lc", command], cwd=str(dir_path), env=env, check=True)
        installed += 1

    print(
        f"Gym venv prefetch complete: {venv_dir} "
        f"(installed={installed}, already_present={skipped})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
