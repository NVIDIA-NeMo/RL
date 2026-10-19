# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mixed recipes must wire both OpenCode roles without assembly-only prompts."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo_rl.utils.config import load_config, parse_hydra_overrides

REPO = Path(__file__).resolve().parents[2]
RECIPES = (
    "8n8g-megatron-claweval-agentic-vbench-sc.v1",
    "16n8g-megatron-claweval-agentic-vbench-async-sc.v1",
)
AGENTS = (
    ("agentic_vbench_agent", "AGENTIC_VBENCH"),
    ("claweval_agent", "CLAW_EVAL"),
)
AVB_RECIPES = RECIPES + (
    "8n8g-megatron-agentic-vbench-sc.v1",
    "16n8g-megatron-agentic-vbench-async-sc.v1",
)


@pytest.mark.parametrize("recipe", RECIPES)
@pytest.mark.parametrize("agent,prefix", AGENTS)
def test_mixed_recipe_resolves_task_independent_opencode_roles(
    recipe: str, agent: str, prefix: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    for _, env_prefix in AGENTS:
        monkeypatch.delenv(f"{env_prefix}_DELEGATION_PROMPT", raising=False)
        monkeypatch.delenv(f"{env_prefix}_SUBAGENT_PROMPT", raising=False)
    config = load_config(
        REPO
        / "examples/configs/recipes/vlm"
        / f"vlm_grpo-nemotron-3.5-super-vl-120ba12b-{recipe}.yaml"
    )
    gym = config.env.nemo_gym
    assert gym.training_mode == "subagent" and gym.agent_name is None
    settings = gym[agent].responses_api_agents[agent]
    assert settings.opencode_training_mode == "subagent"
    assert settings.opencode_subagents_enabled is True
    assert settings.opencode_compaction_auto is False
    assert settings.opencode_compaction_prune is False
    for field in ("opencode_delegation_prompt", "opencode_subagent_prompt"):
        assert "task" in settings[field]
        assert "solution.mp4" not in settings[field]
        assert "storyboard" not in settings[field]

    # Resolve the same inherited recipe again with independently configured
    # role instructions, including punctuation that previously appeared in runs.
    parent = "Delegate one bounded task, then verify the original deliverables."
    child = "Complete the assigned subtask, then report evidence to the parent."
    monkeypatch.setenv(f"{prefix}_DELEGATION_PROMPT", parent)
    monkeypatch.setenv(f"{prefix}_SUBAGENT_PROMPT", child)
    resolved = OmegaConf.to_container(settings, resolve=True)
    assert resolved["opencode_delegation_prompt"] == parent
    assert resolved["opencode_subagent_prompt"] == child


@pytest.mark.parametrize("recipe", AVB_RECIPES)
def test_mixed_recipe_admits_the_complete_avb_model_context(recipe: str) -> None:
    config = load_config(
        REPO
        / "examples/configs/recipes/vlm"
        / f"vlm_grpo-nemotron-3.5-super-vl-120ba12b-{recipe}.yaml"
    )
    assert config.policy.max_total_sequence_length == 262144
    assert (
        config.token_capture.max_train_row_tokens
        == config.policy.max_total_sequence_length
    )
    # The admission limit must follow the model window rather than retaining
    # an independent limit inherited from the shorter-context benchmark.
    config.policy.max_total_sequence_length = 131072
    assert config.token_capture.max_train_row_tokens == 131072


def _capture_avb_launcher_arguments(
    tmp_path: Path, *, row_limit: str | None = None
) -> subprocess.CompletedProcess[bytes]:
    script = tmp_path / "smoke_subagents_fast_step.sh"
    shutil.copyfile(REPO / "scripts/agentic_vbench" / script.name, script)
    # Replace only the submission boundary. Exercise the real preceding shell
    # exports and pass its exact argv to the native Hydra parser below.
    (tmp_path / "smoke_off_32turns_256k.sh").write_text(
        "#!/usr/bin/env bash\nprintf '%s\\0' \"$@\"\n"
    )
    env = dict(os.environ, MAX_STEPS="1000000", NUM_PROMPTS="4")
    env.pop("SUBAGENT_MAX_ROW_TOKENS", None)
    env.pop("AGENTIC_VBENCH_CONTEXT_TOKENS", None)
    if row_limit is not None:
        env["SUBAGENT_MAX_ROW_TOKENS"] = row_limit
    return subprocess.run(
        ["bash", str(script)], env=env, capture_output=True, timeout=20
    )


@pytest.mark.parametrize("recipe", AVB_RECIPES)
def test_real_avb_launcher_keeps_full_context_with_existing_recipe_keys(
    recipe: str, tmp_path: Path
) -> None:
    launched = _capture_avb_launcher_arguments(tmp_path)
    assert launched.returncode == 0, launched.stderr.decode(errors="replace")
    overrides = launched.stdout.decode().rstrip("\0").split("\0")
    config = load_config(
        REPO
        / "examples/configs/recipes/vlm"
        / f"vlm_grpo-nemotron-3.5-super-vl-120ba12b-{recipe}.yaml"
    )
    configured = parse_hydra_overrides(config, overrides)
    assert configured.token_capture.max_train_row_tokens == 262144
    assert configured.env.nemo_gym.training_mode == "subagent"
    assert configured.grpo.max_num_steps == 1000000
    assert configured.grpo.max_num_epochs is None


def test_avb_launcher_rejects_a_hidden_shorter_training_window(tmp_path: Path) -> None:
    launched = _capture_avb_launcher_arguments(tmp_path, row_limit="65536")
    assert launched.returncode == 2
    assert b"must cover the complete" in launched.stderr
    assert not launched.stdout
