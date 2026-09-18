# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small pre-allocation and online-logging guards for standalone image-tool runs."""

import argparse
import hashlib
import json
from pathlib import Path


def source_fingerprint(root: Path) -> str:
    """Bind qualification to the port, launch scripts, and model implementations."""
    paths = [
        root / "uv.lock",
        root / "ray.sub",
        root / "nemo_rl/environments/nemo_gym.py",
        root / "nemo_rl/models/megatron/setup.py",
    ]
    paths.extend(sorted((root / "tools").glob("*image_tools*.py")))
    paths.extend(sorted((root / "tools").glob("*image_tools*.sh")))
    paths.extend(sorted((root / "tools/runtime_source_override").glob("*.py")))
    paths.extend(sorted((root / "tools/image_tools_build_bin").glob("*")))
    paths.extend(
        sorted((root / "examples/configs/recipes/vlm").glob("*image-tools*.yaml"))
    )
    gym = root / "3rdparty/Gym-workspace/Gym"
    paths.extend(sorted((gym / "responses_api_agents/image_tools_agent").glob("*.py")))
    paths.extend(sorted((gym / "environments/image_tools_grpo").glob("*.yaml")))
    bridge = root / "3rdparty/Megatron-Bridge-workspace/Megatron-Bridge"
    paths.extend(
        sorted((bridge / "src/megatron/bridge/models/nemotron_omni").glob("*.py"))
    )
    paths.append(bridge / "src/megatron/bridge/models/conversion/auto_bridge.py")
    paths.append(bridge / "src/megatron/bridge/models/hybrid/hybrid_provider.py")
    paths.append(
        bridge / "3rdparty/Megatron-LM/megatron/core/models/hybrid/hybrid_model.py"
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode() + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def require_online_wandb(environment):
    """Reject missing credentials and silent offline/disabled logging without exposing keys."""
    if not environment.get("WANDB_API_KEY", "").strip():
        raise ValueError("The synced project credentials must supply WANDB_API_KEY")
    if environment.get("WANDB_MODE") != "online":
        raise ValueError("Image-tool training requires WANDB_MODE=online")
    if environment.get("WANDB_DISABLED", "false").lower() not in {"false", "0"}:
        raise ValueError("Image-tool training must not disable W&B")
    if not environment.get("WANDB_RUN_ID", "").strip():
        raise ValueError("Image-tool training requires a stable WANDB_RUN_ID")


def validate_qualification(
    log, overlay, source_lock, *, project_root=None, model_checkpoint=None
):
    """Require a complete successful probe tied to this overlay and source lock."""
    records = []
    for line in log.read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    expected = {
        "driver",
        "learner",
        "generation",
        "collector",
        "gym_actor",
        "responses_api_agents/image_tools_agent",
        "responses_api_agents/simple_agent",
        "resources_servers/string_match",
        "resources_servers/math_with_judge",
        "resources_servers/mcqa",
        "responses_api_models/vllm_model",
    }
    checks = [r for r in records if "check" in r]
    if (
        len(checks) != len(expected)
        or {r["check"] for r in checks} != expected
        or not all(r.get("ok") is True for r in checks)
    ):
        raise ValueError("Runtime qualification has not passed every required check")
    receipt = json.loads((overlay / "READY.json").read_text())
    if (
        receipt["source_lock_sha256"]
        != hashlib.sha256(source_lock.read_bytes()).hexdigest()
    ):
        raise ValueError("Qualified overlay does not match the source lock")
    if receipt not in records:
        raise ValueError("Qualification log does not identify this exact overlay build")
    if project_root is not None:
        identity = {
            "qualification_source_sha256": source_fingerprint(project_root),
            "model_checkpoint": model_checkpoint,
        }
        if identity not in records:
            raise ValueError("Qualification does not match this source and checkpoint")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--model-checkpoint", required=True)
    args = parser.parse_args()
    validate_qualification(
        args.log,
        args.overlay,
        args.lock,
        project_root=args.project_root,
        model_checkpoint=args.model_checkpoint,
    )
    print("IMAGE_TOOLS_RUNTIME_QUALIFICATION_VERIFIED")
