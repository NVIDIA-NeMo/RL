# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline config contracts; no Gym servers, provider requests, Ray, or GPUs.

These tests intentionally stop at the NeMo RL/OmegaConf boundary. They are
not a substitute for validation with the pinned Gym parser and a native canary.
"""

import json
from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import InterpolationResolutionError

from nemo_rl.utils.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[3]
JUDGE_DIR = REPO_ROOT / "training_configs" / "judges"
PROVIDER_PATH = JUDGE_DIR / "nvidia_deepseek_v4_flash.yaml"
ROUTES_PATH = JUDGE_DIR / "nvidia_deepseek_v4_flash_routes.yaml"
PROVIDER_REF = (
    "${oc.env:NRL_REPO_ROOT}/training_configs/judges/nvidia_deepseek_v4_flash.yaml"
)
PROVIDER_NAME = "deepseek_v4_flash_judge_model"
TEST_KEY = "offline-test-key-not-a-credential"


@pytest.fixture
def base_config() -> DictConfig:
    """Represent an existing recipe with unrelated routes to preserve."""
    return OmegaConf.create(
        {
            "data": {"train_data_path": "/data/unchanged.jsonl"},
            "policy": {"generation": {"max_new_tokens": 102400}},
            "env": {
                "nemo_gym": {
                    "config_paths": [
                        "responses_api_models/vllm_model/configs/vllm_model_for_training.yaml",
                        "resources_servers/math_with_judge/configs/math_with_judge.yaml",
                        "resources_servers/equivalence_llm_judge/configs/equivalence_llm_judge.yaml",
                        "resources_servers/code_gen/configs/code_gen.yaml",
                    ],
                    "policy_model": {
                        "responses_api_models": {
                            "vllm_model": {
                                "chat_template_kwargs": {"enable_thinking": True}
                            }
                        }
                    },
                    "math_with_judge": {
                        "resources_servers": {
                            "math_with_judge": {
                                "judge_model_server": {"name": "old_judge"},
                                "judge_responses_create_params": {
                                    "max_output_tokens": 4096
                                },
                                "should_use_judge": False,
                            }
                        }
                    },
                    "code_gen": {"resources_servers": {"code_gen": {"debug": False}}},
                }
            },
        }
    )


def compose_child(tmp_path: Path, base_config: DictConfig) -> DictConfig:
    """Exercise the documented two-parent, explicit full-list composition."""
    base_path = tmp_path / "existing_recipe.yaml"
    child_path = tmp_path / "child.yaml"
    OmegaConf.save(base_config, base_path)
    child = OmegaConf.create(
        {
            "defaults": [base_path.name, str(ROUTES_PATH)],
            "env": {
                "nemo_gym": {
                    "config_paths": [
                        *base_config.env.nemo_gym.config_paths,
                        PROVIDER_REF,
                    ]
                }
            },
        }
    )
    OmegaConf.save(child, child_path)
    return load_config(child_path)


def test_provider_contract() -> None:
    provider = OmegaConf.to_container(OmegaConf.load(PROVIDER_PATH), resolve=False)
    assert provider == {
        PROVIDER_NAME: {
            "responses_api_models": {
                "inference_provider": {
                    "entrypoint": "app.py",
                    "base_url": "https://inference-api.nvidia.com/v1",
                    "model": "nvidia/deepseek-ai/deepseek-v4-flash",
                    "api_key": "${oc.env:NVIDIA_API_KEY}",
                    "uses_reasoning_parser": True,
                    "num_workers": 1,
                    "num_concurrent_requests": 256,
                    "extra_body": {"reasoning_effort": "medium"},
                }
            }
        }
    }


@pytest.mark.parametrize("resource", ["math_with_judge", "equivalence_llm_judge"])
def test_judge_routes(resource: str) -> None:
    config = load_config(ROUTES_PATH)
    block = config.env.nemo_gym[resource].resources_servers[resource]
    assert block.judge_model_server == {
        "type": "responses_api_models",
        "name": PROVIDER_NAME,
    }
    assert block.judge_responses_create_params == {
        "input": [],
        "max_output_tokens": 8192,
        "temperature": 0,
    }
    if resource == "math_with_judge":
        assert block.should_use_judge is True
    else:
        assert block.judge_endpoint_max_concurrency == 256


def test_routes_fragment_does_not_replace_environment_paths() -> None:
    gym = load_config(ROUTES_PATH).env.nemo_gym
    assert set(gym) == {"math_with_judge", "equivalence_llm_judge"}


def test_routes_change_only_named_resources_in_pr3941_super_recipe() -> None:
    base = load_config(
        REPO_ROOT / "examples/nemo_gym/nemotron-3-super/stage1_rlvr.yaml"
    )
    merged = OmegaConf.merge(base, load_config(ROUTES_PATH))
    before = OmegaConf.to_container(base, resolve=False)
    after = OmegaConf.to_container(merged, resolve=False)
    for resource in ("math_with_judge", "equivalence_llm_judge"):
        before["env"]["nemo_gym"].pop(resource)
        block = after["env"]["nemo_gym"].pop(resource)
        assert block["resources_servers"][resource]["judge_model_server"] == {
            "type": "responses_api_models",
            "name": PROVIDER_NAME,
        }
    assert after == before


def test_composition_preserves_recipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, base_config: DictConfig
) -> None:
    monkeypatch.setenv("NRL_REPO_ROOT", str(REPO_ROOT))
    merged = compose_child(tmp_path, base_config)
    gym = merged.env.nemo_gym
    assert list(gym.config_paths) == [
        *base_config.env.nemo_gym.config_paths,
        str(PROVIDER_PATH),
    ]
    assert merged.policy == base_config.policy
    assert merged.data == base_config.data
    assert gym.policy_model == base_config.env.nemo_gym.policy_model
    assert gym.code_gen == base_config.env.nemo_gym.code_gen
    math = gym.math_with_judge.resources_servers.math_with_judge
    assert math.judge_model_server.name == PROVIDER_NAME
    assert math.judge_responses_create_params.max_output_tokens == 8192
    assert math.should_use_judge is True


@pytest.mark.parametrize("key_present", [False, True])
def test_resolved_training_config_excludes_judge_secret(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    base_config: DictConfig,
    key_present: bool,
) -> None:
    monkeypatch.setenv("NRL_REPO_ROOT", str(REPO_ROOT))
    if key_present:
        monkeypatch.setenv("NVIDIA_API_KEY", TEST_KEY)
    else:
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    config = compose_child(tmp_path, base_config)
    serialized = json.dumps(OmegaConf.to_container(config, resolve=True))
    assert TEST_KEY not in serialized
    assert "NVIDIA_API_KEY" not in serialized
    assert "api_key" not in serialized
    assert PROVIDER_NAME not in config.env.nemo_gym


def test_provider_resolves_key_only_when_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", TEST_KEY)
    resolved = OmegaConf.to_container(OmegaConf.load(PROVIDER_PATH), resolve=True)
    assert (
        resolved[PROVIDER_NAME]["responses_api_models"]["inference_provider"]["api_key"]
        == TEST_KEY
    )


def test_missing_provider_key_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    with pytest.raises(InterpolationResolutionError, match="NVIDIA_API_KEY"):
        OmegaConf.to_container(OmegaConf.load(PROVIDER_PATH), resolve=True)


def test_missing_checkout_root_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, base_config: DictConfig
) -> None:
    monkeypatch.delenv("NRL_REPO_ROOT", raising=False)
    config = compose_child(tmp_path, base_config)
    with pytest.raises(InterpolationResolutionError, match="NRL_REPO_ROOT"):
        OmegaConf.to_container(config, resolve=True)


def test_absolute_provider_path_works_outside_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, base_config: DictConfig
) -> None:
    checkout_alias = tmp_path / "checkout with spaces"
    checkout_alias.symlink_to(REPO_ROOT, target_is_directory=True)
    monkeypatch.setenv("NRL_REPO_ROOT", str(checkout_alias))
    monkeypatch.chdir(tmp_path)
    config = compose_child(tmp_path, base_config)
    path = Path(config.env.nemo_gym.config_paths[-1])
    assert path.is_absolute()
    assert path.resolve() == PROVIDER_PATH
    assert PROVIDER_NAME in OmegaConf.load(path)
