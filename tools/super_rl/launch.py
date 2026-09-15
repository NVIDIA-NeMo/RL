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

"""Read-only Super RL configuration/source check; submission is not yet exposed.

This first stage of the unified entrypoint deliberately does not import the GPU
stack, contact a scheduler, resolve secrets, hash large assets, or write files.
Native preflight and smoke/submit will be added after the documented release
blockers are addressed. A clean static check is not permission to allocate GPUs.
"""

import argparse
from collections.abc import Mapping
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Annotated, Literal

from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException
from pydantic import AfterValidator, BaseModel, Field, ValidationError
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "training_configs/super_rl"
# Immutable Git object IDs, not credentials.
PR3941_BASE = "ca06137460b7e2edcaf6f1fd67ddbda5ddd6b8e2"  # pragma: allowlist secret
PINNED_SUBMODULES = {
    "3rdparty/Automodel-workspace/Automodel": "24b47e856263d313b942f0ed666c63fff83306b4",  # pragma: allowlist secret
    "3rdparty/Gym-workspace/Gym": "749432dc5de23b8eeb3d044c80350a7c0ae9a03f",  # pragma: allowlist secret
    "3rdparty/Megatron-Bridge-workspace/Megatron-Bridge": "3961f399ef181bca689de8e984110b85c4df00fe",  # pragma: allowlist secret
    "3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM": "f2f0f7bfd88fcb1243df55275988d6af52daea35",  # pragma: allowlist secret
}
NATIVE_GATES = (
    "Kimi reward shaping and cumulative budgets across CoT/TIR/SciCode",
    "Judge missing-verdict/transport failures propagate; no reward-zero fallback",
    "Exact-container imports, image digests, helper ABI and read-only mounts",
    "Model/data integrity, all agent routes, and effective reasoning-on requests",
    "Live scheduler association, topology, CPU affinity and resource availability",
    "Age-2 compact replay: >=3 targets, update/refit, checkpoint and distributed resume",
    "W&B scalar delivery and acceptable steady-state throughput",
)


def _absolute_path(value: Path) -> Path:
    if not value.is_absolute() or ".." in value.parts:
        raise ValueError("Expected an absolute path without parent traversal")
    return value


AbsolutePath = Annotated[Path, AfterValidator(_absolute_path)]
PositiveInt = Annotated[int, Field(strict=True, gt=0)]
NonnegativeInt = Annotated[int, Field(strict=True, ge=0)]
Name = Annotated[str, Field(pattern=r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")]
Architecture = Literal["aarch64", "x86_64"]


class ClusterProfile(BaseModel, extra="forbid"):
    """Site/hardware facts only; no private paths or experiment hyperparameters."""

    name: Literal["aws-cmh", "oci-hsg", "h100"]
    architecture: Architecture
    gpu_model: Literal["GB300", "GB200", "H100"]
    gpus_per_node: PositiveInt
    container_workdir: AbsolutePath
    training_image_hint: str | None
    filesystem_roots_to_check: list[AbsolutePath]
    certified: bool


class ImageConfig(BaseModel, extra="forbid"):
    """User's artifact location and expected identity; native stage checks bytes."""

    path: AbsolutePath
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    architecture: Architecture


class MountConfig(BaseModel, extra="forbid"):
    source: AbsolutePath
    target: AbsolutePath
    read_only: bool


class UserConfig(BaseModel, extra="forbid"):
    """Private launch inputs. Unknown keys fail instead of silently being ignored."""

    site: Name
    account: Name
    partition: Name
    qos: Name
    reservation: Name | None = None
    walltime_minutes: PositiveInt
    cpus_per_task: PositiveInt
    train_nodes: PositiveInt
    generation_nodes: PositiveInt
    gym_nodes: NonnegativeInt
    judge_nodes: NonnegativeInt
    work_root: AbsolutePath
    model: AbsolutePath
    train_data: AbsolutePath
    ccc_metadata: AbsolutePath
    scicode_hdf5: AbsolutePath
    scicode_prompts: AbsolutePath
    training_image: ImageConfig
    sandbox_image: ImageConfig
    worker_python: AbsolutePath
    judge_mode: Literal["self_hosted", "hosted"] = "self_hosted"
    secret_env: list[Annotated[str, Field(pattern=r"^[A-Z_][A-Z0-9_]*$")]] = Field(
        default_factory=list
    )
    mounts: list[MountConfig] = Field(default_factory=list)


def read_yaml(path: Path) -> DictConfig:
    """Load literal, non-secret configuration without evaluating interpolations."""
    try:
        config = OmegaConf.load(path)
    except (OSError, yaml.YAMLError, OmegaConfBaseException):
        # Parser exceptions can include source lines containing accidentally pasted keys.
        raise ValueError("Cannot read YAML; inspect the file locally") from None
    if not isinstance(config, DictConfig):
        raise ValueError("Expected a YAML mapping")
    literal = OmegaConf.to_yaml(config, resolve=False)
    if "${" in literal:
        raise ValueError("Interpolations are not supported here; use secret_env names")
    missing = sorted(OmegaConf.missing_keys(config))
    if missing:
        raise ValueError("Unfilled configuration fields: " + ", ".join(missing))
    return config


def experiment_errors(config: DictConfig) -> list[str]:
    """Check the Kimi delta's scalar consistency, not GRPO/Gym integration."""
    errors: list[str] = []
    positive_integers = (
        "grpo.max_num_steps",
        "grpo.num_prompts_per_step",
        "grpo.num_generations_per_prompt",
        "grpo.reasoning_effort.max_output_tokens",
        "policy.generation.max_new_tokens",
        "policy.train_global_batch_size",
    )
    for key in positive_integers:
        value = OmegaConf.select(config, key)
        if type(value) is not int or value <= 0:
            errors.append(f"Expected positive integer experiment field: {key}")
    if errors:
        return errors
    if (
        config.grpo.reasoning_effort.max_output_tokens
        != config.policy.generation.max_new_tokens
    ):
        errors.append("Reasoning budget cap and policy max_new_tokens disagree")
    if (
        config.policy.train_global_batch_size
        != config.grpo.num_prompts_per_step * config.grpo.num_generations_per_prompt
    ):
        errors.append("Global batch size must match prompts times generations")
    required = {
        "grpo.reasoning_effort.enabled": True,
        "grpo.reasoning_effort.method": "kimi",
        "grpo.reasoning_effort.missing_metadata": "error",
        "grpo.async_grpo.enabled": True,
        "grpo.async_grpo.max_trajectory_age_steps": 2,
        "grpo.async_grpo.in_flight_weight_updates": True,
        "policy.router_replay.enabled": True,
        "policy.router_replay.transport": "ray",
        "policy.sequence_packing.enabled": True,
        "env.nemo_gym.policy_model.responses_api_models.vllm_model.chat_template_kwargs.enable_thinking": True,
        "policy.megatron_cfg.env_vars.NRL_R3_TRACE": "0",
        "policy.megatron_cfg.env_vars.NRL_R3_TRACE_VERIFY_FORWARD": "0",
        "checkpointing.load_replay_buffer": False,
        "data.shuffle": False,
    }
    for key, expected in required.items():
        actual = OmegaConf.select(config, key)
        if type(actual) is not type(expected) or actual != expected:
            errors.append(f"Unsupported Kimi/age-2 experiment field: {key}")
    for key in (
        "grpo.reasoning_effort.kimi.budget_multipliers.low",
        "grpo.reasoning_effort.kimi.budget_multipliers.high",
        "grpo.reasoning_effort.kimi.over_budget_reward",
    ):
        value = OmegaConf.select(config, key)
        if type(value) not in (float, int) or not math.isfinite(value):
            errors.append(f"Expected finite numeric experiment field: {key}")
        elif "budget_multipliers" in key and value <= 0:
            errors.append(f"Budget multiplier must be positive: {key}")
    return errors


def source_errors(repo: Path) -> list[str]:
    """Check immutable ancestry and recursively pinned checkouts without fetching."""
    errors: list[str] = []

    def git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "--no-optional-locks", "-C", str(cwd), *args],
            capture_output=True,
            text=True,
            check=False,
        )

    if git(repo, "merge-base", "--is-ancestor", PR3941_BASE, "HEAD").returncode:
        errors.append("Source is not a verified descendant of the fixed PR3941 base")
    status = git(repo, "status", "--porcelain", "--untracked-files=normal")
    if status.returncode or status.stdout:
        errors.append(
            "Source checkout is dirty or unreadable; commit a reviewed snapshot"
        )
    for relative, expected in PINNED_SUBMODULES.items():
        checkout = repo / relative
        # Without this check git -C an empty submodule walks into the parent repo.
        if not (checkout / ".git").exists():
            errors.append(f"Uninitialized pinned dependency: {relative}")
            continue
        actual = git(checkout, "rev-parse", "HEAD")
        if actual.returncode or actual.stdout.strip() != expected:
            errors.append(f"Dependency does not match pin {expected}: {relative}")
        status = git(checkout, "status", "--porcelain", "--untracked-files=normal")
        if status.returncode or status.stdout:
            errors.append(f"Dirty or unreadable dependency: {relative}")
    return errors


def configuration_errors(
    profile: ClusterProfile, user: UserConfig, *, environment: Mapping[str, str]
) -> list[str]:
    """Lightweight host-path and configuration checks, never native certification."""
    errors: list[str] = []
    if profile.name != "h100" and user.site != profile.name:
        errors.append("User site does not match selected cluster profile")
    if profile.name == "h100" and user.site == "h100":
        errors.append("H100 is hardware; specify the actual Slurm site in user.site")
    for name, image_config in (
        ("training_image", user.training_image),
        ("sandbox_image", user.sandbox_image),
    ):
        if image_config.architecture != profile.architecture:
            errors.append(f"{name} declared architecture does not match profile")
        if not image_config.path.is_file() or not os.access(image_config.path, os.R_OK):
            errors.append(f"{name} is not a readable local file")
    for name, path in (
        ("train_data", user.train_data),
        ("ccc_metadata", user.ccc_metadata),
        ("scicode_hdf5", user.scicode_hdf5),
    ):
        if not path.is_file() or not os.access(path, os.R_OK):
            errors.append(f"{name} is not a readable local file")
    for name, path in (
        ("model", user.model),
        ("scicode_prompts", user.scicode_prompts),
        ("work_root", user.work_root),
    ):
        if not path.is_dir() or not os.access(path, os.R_OK | os.X_OK):
            errors.append(f"{name} is not a readable/traversable local directory")
    if not os.access(user.work_root, os.W_OK):
        errors.append("work_root is not writable by the current user")
    for name in user.secret_env:
        if not environment.get(name):
            errors.append(f"Required secret environment variable is unset: {name}")
    if user.judge_mode == "hosted" and "NVIDIA_API_KEY" not in user.secret_env:
        errors.append(
            "Hosted NVIDIA judge requires explicit NVIDIA_API_KEY in secret_env"
        )
    targets: set[Path] = set()
    for mount in user.mounts:
        if not mount.source.exists():
            errors.append("A mount source is absent on this host")
        if mount.target in targets:
            errors.append("Duplicate container mount target")
        targets.add(mount.target)
        if mount.target == profile.container_workdir and (
            mount.source.resolve() != REPO_ROOT or not mount.read_only
        ):
            errors.append("Checkout mount must bind this audited repository read-only")
    if profile.container_workdir not in targets:
        errors.append("Explicit checkout mount at container_workdir is required")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", choices=["aws-cmh", "oci-hsg", "h100"], required=True
    )
    parser.add_argument("--user", type=Path, required=True)
    parser.add_argument(
        "--experiment",
        type=Path,
        default=CONFIG_ROOT / "experiments/kimi_s25.yaml",
        help="Kimi YAML delta, not a complete training config",
    )
    parser.add_argument(
        "--check", action="store_true", help="Read-only check (the default)"
    )
    args = parser.parse_args()
    errors = source_errors(REPO_ROOT)
    try:
        errors.extend(experiment_errors(read_yaml(args.experiment)))
    except ValueError as error:
        errors.append(str(error))
    try:
        profile = ClusterProfile.model_validate(
            OmegaConf.to_container(
                read_yaml(CONFIG_ROOT / f"profiles/{args.profile}.yaml")
            )
        )
        user = UserConfig.model_validate(OmegaConf.to_container(read_yaml(args.user)))
    except ValidationError as error:
        # Do not serialize input values, even for unknown or malformed secret fields.
        errors.extend(
            "Invalid config field: " + ".".join(map(str, item["loc"]))
            for item in error.errors(include_input=False, include_context=False)
        )
        user = None
    except ValueError as error:
        errors.append(str(error))
        user = None
    summary: dict[str, str | int] = {}
    if user is not None:
        errors.extend(configuration_errors(profile, user, environment=os.environ))
        total_nodes = (
            user.train_nodes + user.generation_nodes + user.gym_nodes + user.judge_nodes
        )
        summary = {
            "profile": profile.name,
            "site": user.site,
            "nodes": total_nodes,
            "gpus": total_nodes * profile.gpus_per_node,
            "judge_mode": user.judge_mode,
        }
    print(
        json.dumps(
            {
                "mode": "check_only",
                "static_checks_passed": not errors,
                "submission_supported": False,
                "pr3941_base": PR3941_BASE,
                "summary": summary,
                "errors": errors,
                "unverified_release_gates": NATIVE_GATES,
                "guide": "docs/guides/super-rl-launch.md",
            },
            indent=2,
        )
    )
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
