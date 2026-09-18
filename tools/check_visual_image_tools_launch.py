# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check the explicit qualification boundary before allocating mixed training.

Reuse real standalone model-training evidence only for unchanged model code.
Check new game/metric/server code against the exact component-test snapshot.
Neither check claims a successful mixed model rollout or optimizer update.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re

from tools.callback_fix_contract import (
    BRIDGE_SETUP,
    CALLBACK_EDITS,
    PR_HEADS,
    expected_callback_source,
    require_callback_sources,
)

from tools.check_visual_image_tools_mix import (
    GYM_V_TRAIN,
    GYM_V_VALIDATION,
    VISGYM_TRAIN,
)


def require_same_files(
    *, current: Path, qualified: Path, paths: list[str], callback_fixes: bool = False
) -> str:
    """Compare a bounded set of runtime files and return their content identity."""
    digest = hashlib.sha256()
    for relative in sorted(set(paths)):
        content = (current / relative).read_bytes()
        expected = (qualified / relative).read_bytes()
        if callback_fixes and relative in CALLBACK_EDITS:
            expected = expected_callback_source(relative, expected)
        if content != expected:
            raise ValueError(f"Qualification source mismatch: {relative}")
        digest.update(relative.encode() + b"\0" + content)
    return digest.hexdigest()


def runtime_files(root: Path, directories: list[str]) -> list[str]:
    """Enumerate only the selected small runtime components, excluding tests."""
    return [
        str(path.relative_to(root))
        for directory in directories
        for path in sorted((root / directory).rglob("*"))
        if path.is_file()
        and path.suffix in {".py", ".yaml", ".whl"}
        and not {"tests", "__pycache__", ".venv"}.intersection(path.parts)
    ]


def require_pytest_pass(log: Path, *, minimum: int) -> None:
    """Require a completed passing pytest summary, not a test-start banner."""
    text = log.read_text()
    summaries = re.findall(r"(?m)^=*[ \t]*(\d+) passed([^\n]*)$", text)
    if not summaries or int(summaries[-1][0]) < minimum:
        raise ValueError(f"Required tests did not complete: {log}")
    if re.search(r"\b(?:failed|errors?|xfailed)\b", summaries[-1][1]):
        raise ValueError(f"Required tests failed: {log}")


def qualify(
    *,
    project_root: Path,
    image_source: Path,
    image_run: Path,
    component_source: Path,
    component_run: Path,
    games_venv_root: Path,
    overlay: Path,
    model_checkpoint: str,
) -> dict:
    """Require model training, current component coverage, and unchanged runtimes."""
    status = json.loads(
        (image_run / "checkpoints/latest_checkpoint_status.json").read_text()
    )
    if (
        status["last_checkpoint_step"] < 20
        or not status["last_successful_ckpt_save_completion"]
    ):
        raise ValueError("Standalone image-tool training has no completed checkpoint")
    provenance = json.loads((image_run / "tokenizer/provenance.json").read_text())
    if provenance["model"] != model_checkpoint or not provenance["fix_mistral_regex"]:
        raise ValueError("Standalone model/tokenizer qualification does not match")
    receipt = json.loads((overlay / "READY.json").read_text())
    if (
        receipt["source_lock_sha256"]
        != hashlib.sha256((project_root / "uv.lock").read_bytes()).hexdigest()
    ):
        raise ValueError("Overlay/source lock mismatch")
    require_callback_sources(project_root)
    model_paths = [
        BRIDGE_SETUP,
        "uv.lock",
        "ray.sub",
        "nemo_rl/models/megatron/setup.py",
        "nemo_rl/environments/nemo_gym.py",
        "nemo_rl/environments/nemo_gym_tool_calls.py",
        "tools/image_tools_dataset_helpers.sh",
        "tools/image_tools_build_bin/python3-config",
        "tools/image_tools_processor_assets.py",
    ]
    bridge = "3rdparty/Megatron-Bridge-workspace/Megatron-Bridge"
    model_dirs = [
        "tools/runtime_source_override",
        "tools/image_tools_build_bin",
        f"{bridge}/src/megatron/bridge/models/nemotron_omni",
    ]
    model_paths += runtime_files(project_root, model_dirs)
    model_paths += runtime_files(image_source, model_dirs)
    model_paths += [
        f"{bridge}/{path}"
        for path in (
            "src/megatron/bridge/models/conversion/auto_bridge.py",
            "src/megatron/bridge/models/hybrid/hybrid_provider.py",
            "3rdparty/Megatron-LM/megatron/core/models/hybrid/hybrid_model.py",
        )
    ]
    model_identity = require_same_files(
        current=project_root,
        qualified=image_source,
        paths=model_paths,
        callback_fixes=True,
    )
    gym = "3rdparty/Gym-workspace/Gym"
    component_dirs = [
        f"{gym}/{path}"
        for path in (
            "resources_servers/gym_v",
            "resources_servers/visgym",
            "responses_api_agents/gymv_agent",
            "responses_api_agents/visgym_agent",
            "responses_api_agents/image_tools_agent",
            "environments/image_tools_grpo",
            "environments/visual_games_image_tools",
        )
    ]
    component_paths = [
        f"{gym}/responses_api_models/vllm_model/app.py",
        f"{gym}/nemo_gym/server_utils.py",
        "nemo_rl/experience/rollouts.py",
        "nemo_rl/experience/nemo_gym_metrics.py",
        "tools/check_visual_image_tools_components.py",
    ]
    # The diagnostic runner is not imported by training. Its corrected pytest
    # working directory does not invalidate unchanged game code plus the stored
    # successful component/guard/metric results, which are checked below.
    # Include both inventories so additions/removals cannot evade the comparison.
    component_paths += runtime_files(project_root, component_dirs)
    component_paths += runtime_files(component_source, component_dirs)
    component_identity = require_same_files(
        current=project_root, qualified=component_source, paths=component_paths
    )
    report = json.loads((component_run / "component-summary.json").read_text())
    results = report["results"]
    expected = GYM_V_TRAIN | GYM_V_VALIDATION | VISGYM_TRAIN
    if (
        len(results) != len(expected)
        or {r["env_id"] for r in results} != expected
        or not all(r["passed"] for r in results)
    ):
        raise ValueError("Not all 42 game components passed")
    certificate = games_venv_root / ".visual-games-suite-prefetch-complete"
    if (
        not report["certificate_unchanged"]
        or hashlib.sha256(certificate.read_bytes()).hexdigest()
        != report["certificate_sha256"]
    ):
        raise ValueError("Qualified game runtime certificate changed")
    require_pytest_pass(component_run / "agent-guard-tests.log", minimum=115)
    require_pytest_pass(component_run / "rl-metric-test.log", minimum=1)
    return {
        "approved_callback_pr_heads": PR_HEADS,
        "callback_runtime_tests_required_before_training": True,
        "model_source_sha256": model_identity,
        "component_source_sha256": component_identity,
        "standalone_checkpoint_step": status["last_checkpoint_step"],
        "model_checkpoint": model_checkpoint,
        "component_run": str(component_run),
        "mixed_training_verified": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "project-root",
        "image-source",
        "image-run",
        "component-source",
        "component-run",
        "games-venv-root",
        "overlay",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model-checkpoint", required=True)
    print(json.dumps(qualify(**vars(parser.parse_args())), indent=2))
