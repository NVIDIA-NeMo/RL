# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from omegaconf import OmegaConf

from tools.super_rl.check_gym_startup import startup_config


def test_cpu_service_check_keeps_science_and_isolates_outputs(tmp_path, monkeypatch):
    monkeypatch.setenv("SUPER_RL_MODEL", "/models/s120")
    root = Path(__file__).resolve().parents[3]
    path = root / "training_configs/super_rl/experiments/regular_s120_smoke.yaml"
    original = path.read_bytes()
    graph = startup_config(path, tmp_path)
    assert graph.policy_base_url == ["http://127.0.0.1:1/v1"]
    assert graph.model_endpoint_readiness_timeout_seconds == 0
    assert graph.policy_model.responses_api_models.vllm_model.chat_template_kwargs.enable_thinking
    assert (
        graph.ns_tools_simple_agent.responses_api_agents.simple_agent.max_total_output_tokens
        == 102400
    )
    assert (
        graph.scicode_agent.responses_api_agents.scicode_agent.max_total_output_tokens
        == 102400
    )
    assert graph.cache_dir == str(tmp_path / "cache")
    assert graph.results_dir == str(tmp_path / "results")
    assert path.read_bytes() == original
    assert (
        "model_endpoint_readiness_timeout_seconds"
        not in OmegaConf.load(path).env.nemo_gym
    )
