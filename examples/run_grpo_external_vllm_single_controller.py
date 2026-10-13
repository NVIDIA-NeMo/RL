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

"""Async Single Controller GRPO against an externally managed stock vLLM."""

import argparse
import pprint
import runpy
import sys
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from nemo_rl.models.generation.remote_vllm import (
    RemoteVllmServiceConfig,
    preflight_remote_vllm_service,
)
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run Single Controller GRPO against an external vLLM service"
    )
    parser.add_argument("--config", type=str)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate the remote service contract and exit before Ray startup",
    )
    return parser.parse_known_args()


def _load_remote_config(
    config_path: str, overrides: list[str]
) -> RemoteVllmServiceConfig:
    raw_config = load_config(config_path)
    if overrides:
        raw_config = parse_hydra_overrides(raw_config, overrides)
    resolved = OmegaConf.to_container(raw_config, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError("Top-level configuration must be a mapping")

    policy = resolved.get("policy")
    if not isinstance(policy, dict):
        raise ValueError("Configuration is missing policy")
    generation = policy.get("generation")
    if not isinstance(generation, dict):
        raise ValueError("Configuration is missing policy.generation")
    if generation.get("backend") != "remote_vllm":
        raise ValueError(
            "This entrypoint requires policy.generation.backend=remote_vllm"
        )
    remote_config: Any = generation.get("remote_vllm_cfg")
    if not isinstance(remote_config, dict):
        raise ValueError("This entrypoint requires policy.generation.remote_vllm_cfg")
    return RemoteVllmServiceConfig.model_validate(remote_config)


def main() -> None:
    register_omegaconf_resolvers()
    args, overrides = _parse_args()
    config_path = args.config or str(
        Path(__file__).parent
        / "nemo_gym"
        / "nemotron-3.5-nano"
        / "rlvr_sc_smoke_small_external_vllm.yaml"
    )

    remote_config = _load_remote_config(config_path, overrides)
    info = preflight_remote_vllm_service(remote_config)
    print("External rollout service preflight succeeded:")
    pprint.pprint(info.model_dump())

    if args.preflight_only:
        return

    # Keep the canonical entrypoint as the owner of Ray startup, actor
    # construction, training, checkpointing, and teardown.
    canonical_entrypoint = Path(__file__).with_name("run_grpo_single_controller.py")
    sys.argv = [str(canonical_entrypoint), "--config", config_path, *overrides]
    runpy.run_path(str(canonical_entrypoint), run_name="__main__")


if __name__ == "__main__":
    main()
