# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the image-tool launcher config against stale container metadata."""

import ast
import importlib.metadata
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_rl.utils.config import load_config


@pytest.mark.parametrize(
    "suite", ["image-tools-8n4g", "visual-games-image-tools-40n4g"]
)
def test_actual_driver_gym_preflight_with_stale_metadata(monkeypatch, suite):
    root = Path(__file__).resolve().parents[3]
    for key in (
        "MODEL_CHECKPOINT",
        "VLLM_TOKENIZER",
        "TRAIN_MANIFEST",
        "EVAL_MANIFEST",
        "CHECKPOINT_DIR",
        "RUN_LOG_DIR",
        "WANDB_RUN_NAME",
        "IMAGE_TOOLS_OUTPUT_DIR",
    ):
        monkeypatch.setenv(key, "/fixture/" + key)
    monkeypatch.setenv("PROJECT_ROOT", str(root))
    monkeypatch.setenv("NEMO_GYM_EXTRA_ROOTS", str(root / "3rdparty/Gym-workspace/Gym"))
    monkeypatch.setattr(importlib.metadata, "requires", lambda name: ["openai<=2.7.2"])
    monkeypatch.setattr("nemo_gym.global_config.openai_version", "2.6.1")
    recipe = (
        root
        / f"examples/configs/recipes/vlm/vlm_grpo-nemotron-super-omni-120ba12b-{suite}-megatron-tp8ep16cp2-async.v1.yaml"
    )
    driver = (root / "tools/image_tools_train_hsg.sh").read_text()
    # Execute the exact CPU-only config preflight block from the driver.
    block = driver.split("gym_config = ", 1)[1].split("# Exercise the entrypoint", 1)[0]
    namespace = {
        "OmegaConf": OmegaConf,
        "GlobalConfigDictParser": GlobalConfigDictParser,
        "GlobalConfigDictParserConfig": GlobalConfigDictParserConfig,
        "cfg": load_config(str(recipe)),
        "model": "/fixture/MODEL_CHECKPOINT",
    }
    exec(
        compile(
            ast.parse("gym_config = " + block),
            str(root / "tools/image_tools_train_hsg.sh"),
            "exec",
        ),
        namespace,
    )
    resolved = namespace["resolved_gym"]
    assert resolved.allow_openai_version_skew is False
    assert "openai==2.6.1" in resolved.head_server_deps
    assert (
        resolved.image_tools_simple_agent.responses_api_agents.image_tools_agent.max_output_tokens
        == 512
    )
    if suite.startswith("visual-games"):
        assert resolved.gym_v_agent.responses_api_agents.gymv_agent.max_steps == 8
        assert resolved.visgym_agent.responses_api_agents.visgym_agent.max_steps == 35
