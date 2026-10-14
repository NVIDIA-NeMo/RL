"""Credential transport regressions, runnable without Docker or a model."""

import json
from pathlib import Path

import pytest

pytest.importorskip("harbor")

from harbor.agents.factory import AgentFactory
from harbor.models.job.config import JobConfig

from tools.http_profile.harbor_key_smoke import MODES, Mode, agent_config


@pytest.mark.parametrize("mode", MODES)
def test_internal_job_handoff_preserves_independent_credentials(
    mode: Mode, tmp_path: Path
) -> None:
    agents = [
        agent_config(
            mode=mode, key=f"smoke-only-key-{index}", base_url="http://backend:8000/v1"
        )
        for index in range(2)
    ]
    original = JobConfig(agents=agents)
    wire = original.model_dump(mode="json", context={"redact_sensitive_env": False})
    received = JobConfig.model_validate(json.loads(json.dumps(wire)))
    for index, config in enumerate(received.agents):
        assert config.env == agents[index].env
        agent = AgentFactory.create_agent_from_config(
            config, logs_dir=tmp_path / str(index)
        )
        assert agent.extra_env == config.env
        if mode == "provider-env":
            assert agent.extra_env["OPENAI_API_KEY"] == f"smoke-only-key-{index}"
            provider = config.kwargs["opencode_config"]["provider"]["nrl"]
            assert provider["env"] == ["OPENAI_API_KEY"]
            assert provider["options"]["baseURL"] == "http://backend:8000/v1"
        elif mode == "config-env":
            assert agent.extra_env["NRL_ROLLOUT_API_KEY"] == f"smoke-only-key-{index}"
        else:
            command = agent._build_register_config_command()
            assert f"smoke-only-key-{index}" in command
            assert f"smoke-only-key-{1 - index}" not in command


def test_default_dump_redacts_key_so_cannot_be_used_for_worker_transport() -> None:
    key = "smoke-only-not-a-real-secret"
    original = JobConfig(
        agents=[agent_config(mode="config-env", key=key, base_url="http://backend/v1")]
    )
    # Reproduces harbor_agent_general.run()'s existing model_dump(mode="json").
    persisted = original.model_dump(mode="json")
    assert persisted["agents"][0]["env"]["NRL_ROLLOUT_API_KEY"] != key
    assert "****" in persisted["agents"][0]["env"]["NRL_ROLLOUT_API_KEY"]
    assert original.agents[0].env["NRL_ROLLOUT_API_KEY"] == key
