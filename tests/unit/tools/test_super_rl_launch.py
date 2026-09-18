# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
from unittest.mock import patch

from omegaconf import OmegaConf
from pydantic import ValidationError
import pytest

from tools.super_rl.launch import (
    CONFIG_ROOT,
    REPO_ROOT,
    ClusterProfile,
    UserConfig,
    configuration_errors,
    experiment_errors,
    main,
    read_yaml,
    source_errors,
)


@pytest.fixture
def user_dict(tmp_path):
    for name in (
        "train.jsonl",
        "ccc.jsonl",
        "scicode.h5",
        "training.sqsh",
        "sandbox.sqsh",
    ):
        (tmp_path / name).touch()
    image = {"sha256": "a" * 64, "architecture": "aarch64"}
    return {
        "site": "aws-cmh",
        "account": "my_account",
        "partition": "my_partition",
        "qos": "my_qos",
        "walltime_minutes": 120,
        "cpus_per_task": 140,
        "train_nodes": 32,
        "generation_nodes": 96,
        "gym_nodes": 0,
        "judge_nodes": 0,
        "work_root": str(tmp_path),
        "model": str(tmp_path),
        "train_data": str(tmp_path / "train.jsonl"),
        "ccc_metadata": str(tmp_path / "ccc.jsonl"),
        "scicode_hdf5": str(tmp_path / "scicode.h5"),
        "scicode_prompts": str(tmp_path),
        "training_image": {**image, "path": str(tmp_path / "training.sqsh")},
        "sandbox_image": {**image, "path": str(tmp_path / "sandbox.sqsh")},
        "worker_python": "/opt/worker/bin/python",
        "mounts": [
            {"source": str(REPO_ROOT), "target": "/opt/nemo-rl", "read_only": True}
        ],
    }


def profile(name):
    return ClusterProfile.model_validate(
        OmegaConf.to_container(read_yaml(CONFIG_ROOT / f"profiles/{name}.yaml"))
    )


@pytest.mark.parametrize(
    "name,architecture,gpus",
    [
        ("aws-cmh", "aarch64", 4),
        ("oci-hsg", "aarch64", 4),
        ("h100", "x86_64", 8),
    ],
)
def test_profiles_do_not_claim_certification(name, architecture, gpus):
    selected = profile(name)
    assert selected.architecture == architecture
    assert selected.gpus_per_node == gpus
    assert not selected.certified


def test_valid_host_metadata_does_not_need_secret_values(user_dict):
    user_dict["secret_env"] = ["WANDB_API_KEY"]
    user = UserConfig.model_validate(user_dict)
    assert (
        configuration_errors(
            profile("aws-cmh"), user, environment={"WANDB_API_KEY": "private"}
        )
        == []
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("cpus_per_task", True),
        ("train_nodes", 0),
        ("account", "-bad"),
        ("model", "relative/model"),
        ("work_root", "/data/../work"),
        ("api_key", "secret"),
        ("secret_env", ["KEY=value"]),
    ],
)
def test_invalid_user_config_fails(user_dict, key, value):
    user_dict[key] = value
    with pytest.raises(ValidationError):
        UserConfig.model_validate(user_dict)


def test_hosted_is_explicit_and_requires_secret_name(user_dict):
    user = UserConfig.model_validate(user_dict)
    assert user.judge_mode == "self_hosted"
    user_dict["judge_mode"] = "hosted"
    errors = configuration_errors(
        profile("aws-cmh"), UserConfig.model_validate(user_dict), environment={}
    )
    assert any("explicit NVIDIA_API_KEY" in item for item in errors)


def test_architecture_missing_assets_and_mount_collisions(user_dict):
    user_dict["training_image"]["architecture"] = "x86_64"
    user_dict["ccc_metadata"] += ".absent"
    user_dict["mounts"] *= 2
    errors = configuration_errors(
        profile("aws-cmh"), UserConfig.model_validate(user_dict), environment={}
    )
    assert len(errors) == 3
    assert "architecture" in errors[0]
    assert "ccc_metadata" in errors[1]
    assert "Duplicate" in errors[2]


def test_example_requires_user_input():
    with pytest.raises(ValueError, match="Unfilled"):
        read_yaml(CONFIG_ROOT / "user.example.yaml")


def test_cannot_mount_another_checkout_or_write_source(user_dict):
    user_dict["mounts"][0]["source"] = user_dict["work_root"]
    user_dict["mounts"][0]["read_only"] = False
    errors = configuration_errors(
        profile("aws-cmh"), UserConfig.model_validate(user_dict), environment={}
    )
    assert any("audited repository read-only" in item for item in errors)


def test_h100_requires_a_real_site(user_dict):
    user_dict["site"] = "h100"
    user_dict["training_image"]["architecture"] = "x86_64"
    user_dict["sandbox_image"]["architecture"] = "x86_64"
    errors = configuration_errors(
        profile("h100"), UserConfig.model_validate(user_dict), environment={}
    )
    assert any("actual Slurm site" in item for item in errors)


def test_never_resolves_secret_interpolation(tmp_path, monkeypatch):
    monkeypatch.setenv("PRIVATE_KEY", "do-not-print-me")
    path = tmp_path / "input.yaml"
    path.write_text("api_key: ${oc.env:PRIVATE_KEY}\n")
    with pytest.raises(ValueError, match="Interpolations") as error:
        read_yaml(path)
    assert "do-not-print-me" not in str(error.value)


def test_parser_error_does_not_repeat_source(tmp_path):
    path = tmp_path / "input.yaml"
    path.write_text("key: [never-print-this-secret\n")
    with pytest.raises(ValueError) as error:
        read_yaml(path)
    assert "never-print" not in str(error.value)


@pytest.mark.parametrize(
    "field,value",
    [
        ("wandb_enabled", False),
        ("tensorboard_enabled", True),
        ("mlflow_enabled", True),
        ("swanlab_enabled", True),
    ],
)
def test_kimi_delta_rejects_non_wandb_logging(field: str, value: bool) -> None:
    config = read_yaml(CONFIG_ROOT / "experiments/kimi_s25.yaml")
    config.logger[field] = value
    assert any(f"logger.{field}" in error for error in experiment_errors(config))


def test_kimi_delta_has_agreed_values_and_preserves_route_catalog():
    config = read_yaml(CONFIG_ROOT / "experiments/kimi_s25.yaml")
    assert experiment_errors(config) == []
    assert config.grpo.reasoning_effort.kimi.budget_multipliers == {
        "low": 1.25,
        "high": 2.0,
    }
    assert config.grpo.reasoning_effort.kimi.over_budget_reward == -0.5
    assert config.policy.generation.max_new_tokens == 102400
    assert config.grpo.max_num_steps * config.grpo.num_prompts_per_step == 38400
    assert "config_paths" not in config.env.nemo_gym


@pytest.mark.parametrize(
    "key,value",
    [
        ("policy.generation.max_new_tokens", 81920),
        ("grpo.num_prompts_per_step", True),
        ("grpo.reasoning_effort.kimi.budget_multipliers.low", float("nan")),
        ("grpo.reasoning_effort.kimi.budget_multipliers.high", -1),
        ("grpo.async_grpo.max_trajectory_age_steps", 1),
        ("checkpointing.load_replay_buffer", True),
        ("data.shuffle", True),
        ("policy.megatron_cfg.env_vars.NRL_R3_TRACE", "1"),
        (
            "env.nemo_gym.policy_model.responses_api_models.vllm_model.chat_template_kwargs.enable_thinking",
            False,
        ),
    ],
)
def test_inconsistent_experiment_fails(key, value):
    config = read_yaml(CONFIG_ROOT / "experiments/kimi_s25.yaml")
    OmegaConf.update(config, key, value)
    assert experiment_errors(config)


def test_source_check_does_not_fetch_or_accept_parent_git_as_submodule(tmp_path):
    with patch("tools.super_rl.launch.subprocess.run") as run:
        run.return_value = subprocess.CompletedProcess([], 0, "", "")
        errors = source_errors(tmp_path)
    assert len(errors) == 4
    assert all("Uninitialized" in item for item in errors)
    assert all(call.args[0][0] == "git" for call in run.call_args_list)
    assert not any("fetch" in call.args[0] for call in run.call_args_list)


@pytest.mark.parametrize("invalid", [False, True])
def test_cli_never_submits_or_displays_values(tmp_path, user_dict, capsys, invalid):
    user_dict["secret_env"] = ["WANDB_API_KEY"]
    if invalid:
        user_dict["api_key"] = "do-not-print-me"
    path = tmp_path / "user.yaml"
    OmegaConf.save(OmegaConf.create(user_dict), path)
    with (
        patch.object(
            sys, "argv", ["launch.py", "--profile", "aws-cmh", "--user", str(path)]
        ),
        patch("tools.super_rl.launch.source_errors", return_value=[]),
        patch.dict("os.environ", {"WANDB_API_KEY": "do-not-print-me"}),
        patch("tools.super_rl.launch.subprocess.run") as run,
        pytest.raises(SystemExit) as result,
    ):
        main()
    run.assert_not_called()
    assert result.value.code == int(invalid)
    output = capsys.readouterr().out
    assert "do-not-print-me" not in output
    report = json.loads(output)
    assert report["submission_supported"] is False
    assert report["unverified_release_gates"]
    if not invalid:
        assert report["summary"]["gpus"] == 512


def test_submit_is_not_exposed(capsys):
    with (
        patch.object(
            sys,
            "argv",
            ["launch.py", "--profile", "aws-cmh", "--user", "unused", "--submit"],
        ),
        pytest.raises(SystemExit),
    ):
        main()
    assert "unrecognized arguments: --submit" in capsys.readouterr().err
