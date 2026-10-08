# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve the actual shell launch arguments without downloading data or training."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo_rl.algorithms.single_controller_utils.config import (
    MasterConfig,
    validate_single_controller_config,
)
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)


@pytest.fixture(params=[False, True])
def functional_recipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
):
    cc = request.param
    if shutil.which("jq") is None:
        pytest.skip("functional data preparation requires jq")
    source = Path(__file__).resolve().parents[3]
    scripts = tmp_path / "tests/functional"
    scripts.mkdir(parents=True)
    for name in (
        "grpo_async_gym_single_controller.sh",
        "grpo_async_gym_single_controller_context_compaction.sh",
    ):
        shutil.copy(source / "tests/functional" / name, scripts / name)
    gym = tmp_path / "3rdparty/Gym-workspace/Gym"
    data = gym / "data/workplace_assistant"
    data.mkdir(parents=True)
    (gym / "env.yaml").write_text("{}\n")
    original = {
        "task_source": "workplace_assistant",
        "responses_create_params": {"tools": [{"name": "first"}, {"name": "second"}]},
    }
    for split in ("train", "validation"):
        (data / f"{split}.jsonl").write_text(json.dumps(original) + "\n")
    binaries = tmp_path / "bin"
    binaries.mkdir()
    calls = tmp_path / "commands.jsonl"
    # Replace only external processes; execute the real shell and jq transforms.
    for name in ("git", "uv"):
        executable = binaries / name
        executable.write_text(
            f"#!{sys.executable}\nimport json, sys\n"
            f"with open({str(calls)!r}, 'a') as f: f.write(json.dumps(sys.argv) + '\\n')\n"
        )
        executable.chmod(0o755)
    monkeypatch.setenv("PATH", f"{binaries}:{os.environ['PATH']}")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("RUN_CONVERGENCE_CHECKS", "1")
    monkeypatch.delenv("SC_TEST_CONTEXT_COMPACTION", raising=False)
    monkeypatch.delenv("SC_TEST_EXP_NAME", raising=False)
    name = "grpo_async_gym_single_controller" + ("_context_compaction" if cc else "")
    subprocess.run(
        ["bash", str(scripts / f"{name}.sh")],
        check=True,
        capture_output=True,
        text=True,
    )
    commands = [json.loads(line) for line in calls.read_text().splitlines()]
    train = next(args for args in commands if "--config" in args)
    register_omegaconf_resolvers()
    index = train.index("--config")
    config = load_config(source / "examples/nemo_gym/grpo_qwen3_30ba3b_instruct.yaml")
    config = parse_hydra_overrides(config, train[index + 2 :])
    resolved = OmegaConf.to_container(config, resolve=True)
    master = MasterConfig.model_validate(resolved)
    validate_single_controller_config(master)
    transformed = json.loads(
        (scripts / name / "data/workplace_assistant_train.jsonl").read_text()
    )
    checks = next(args for args in commands if "tests/check_metrics.py" in args)
    return cc, master, transformed, checks


def test_gym_functional_launch_config(functional_recipe) -> None:
    cc, master, transformed, checks = functional_recipe
    assert master.token_capture.context_compaction is cc
    if cc:
        assert transformed["responses_create_params"]["tools"] == [
            {"name": "first"},
            {"name": "second"},
        ]
        assert master.policy["max_total_sequence_length"] == 16384
        assert transformed["agent_ref"]["name"] == "simple_agent_with_compaction"
        assert "task_source" not in transformed
        assert master.token_capture.enabled
        assert master.loss_fn.token_level_loss
        assert not master.loss_fn.sequence_level_importance_ratios
        assert not master.grpo.calculate_advantages_on_gpu
        history = master.env["nemo_gym"]["simple_agent_with_compaction"][
            "responses_api_agents"
        ]["simple_agent_with_compaction"]["context_history"]
        assert history["schedule"]["actions_per_chunk"] == 1
        assert history["policy"]["config"]["reasoning"]["keep_last_blocks"] == 0
        assert 'max(data["train/global_valid_seqs"]) > 8' in checks
        assert 'max(data["train/token_mult_prob_error"]) < 1.05' in checks
        assert 'len(data["train/token_mult_prob_error"]) == 10' in checks
    else:
        assert transformed["responses_create_params"]["tools"] == [{"name": "first"}]
        assert transformed["task_source"] == "workplace_assistant"
        assert not any("token_mult_prob_error" in arg for arg in checks)


@pytest.mark.nemo_gym
def test_gym_functional_config_routes(functional_recipe) -> None:
    pytest.importorskip("nemo_gym", reason="requires the paired Gym checkout")
    from nemo_gym.global_config import (
        GlobalConfigDictParser,
        GlobalConfigDictParserConfig,
    )
    from nemo_gym.rollout_collection import (
        _environment_server_for_agent,
        _environment_servers_by_agent,
    )

    cc, master, _, _ = functional_recipe
    initial = dict(master.env["nemo_gym"])
    initial.update(
        policy_model_name="preflight",
        policy_base_url="http://127.0.0.1:1/v1",
        policy_api_key="preflight",
    )
    resolved = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.create(initial),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    agent = "simple_agent_with_compaction" if cc else "workplace_assistant_simple_agent"
    assert _environment_server_for_agent(agent, _environment_servers_by_agent(resolved))
