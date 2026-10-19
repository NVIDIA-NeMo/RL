# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ClawEval launcher overrides must compose with existing recipe fields."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from nemo_rl.utils.config import load_config, parse_hydra_overrides

REPO = Path(__file__).resolve().parents[2]
RECIPES = (
    "vlm_grpo-nemotron-3.5-super-vl-120ba12b-8n8g-megatron-claweval-sc.v1.yaml",
    "vlm_grpo-nemotron-super-omni-120ba12b-8n8g-megatron-claweval-sc.v2.yaml",
)


@pytest.mark.parametrize("recipe", RECIPES)
@pytest.mark.parametrize("asynchronous", (False, True))
def test_claweval_shell_overrides_compose_with_native_recipe(
    recipe: str,
    asynchronous: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = 'Delegate one bounded task, then verify the "original" result.\nKeep [evidence] in /workspace.'
    child = "Complete the subtask, report evidence, and preserve the parent's files."
    monkeypatch.setenv("CLAW_EVAL_DELEGATION_PROMPT", parent)
    monkeypatch.setenv("CLAW_EVAL_SUBAGENT_PROMPT", child)
    script_dir = tmp_path / "scripts/claweval"
    script_dir.mkdir(parents=True)
    launcher = (
        "train_subagents_async_16n.sh" if asynchronous else "train_subagents_sync_8n.sh"
    )
    source = REPO / "scripts/claweval" / launcher
    script = script_dir / launcher
    if asynchronous:
        # Exercise the actual async submission arguments only. Donor/auditor
        # checks above this block require a qualified training checkpoint and
        # are deliberately not claimed by this parsing regression.
        arguments = source.read_text().split("agent=env.nemo_gym.", 1)[1]
        script.write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            'script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)\n'
            "agent=env.nemo_gym." + arguments
        )
    else:
        shutil.copyfile(source, script)
    # Stop at the submission boundary, before backend access or Slurm calls.
    (script_dir / "run_sc.sh").write_text(
        "#!/usr/bin/env bash\nprintf '%s\\0' \"$@\"\n"
    )
    env = dict(
        os.environ,
        MAX_STEPS="1000000",
        GENERATION_TENSOR_PARALLEL_SIZE="4",
        NUM_PROMPTS="8",
        inflight_groups="24",
        CLAW_EVAL_MAX_ROW_TOKENS="65536",
        CLAW_EVAL_MAX_TURNS="40",
        CLAW_EVAL_CONTEXT_TOKENS="262144",
        CLAW_EVAL_OUTPUT_TOKENS="8192",
        SYNC_POLICY_CHECKPOINT="/tmp/qualified-sync/step_20/policy/weights",
    )
    launched = subprocess.run(
        ["bash", str(script)], env=env, capture_output=True, timeout=20, check=True
    )
    overrides = launched.stdout.decode().rstrip("\0").split("\0")
    config = parse_hydra_overrides(
        load_config(REPO / "examples/configs/recipes/vlm" / recipe), overrides
    )
    agent = config.env.nemo_gym.claweval_agent.responses_api_agents.claweval_agent
    assert config.env.nemo_gym.training_mode == "subagent"
    assert agent.opencode_training_mode == "subagent"
    assert agent.opencode_subagents_enabled is True
    assert agent.opencode_compaction_auto is False
    assert agent.opencode_compaction_prune is False
    assert agent.opencode_delegation_prompt == parent
    assert agent.opencode_subagent_prompt == child
    assert config.grpo.max_num_steps == 1000000
    assert config.grpo.max_num_epochs is None
    assert config.token_capture.max_train_row_tokens == 65536
    if asynchronous:
        assert config.async_rl.sampler.name == "ready_first"
        assert (
            config.checkpointing.pretrained_checkpoint.path
            == env["SYNC_POLICY_CHECKPOINT"]
        )
    else:
        assert config.async_rl.sampler.name == "in_order"
