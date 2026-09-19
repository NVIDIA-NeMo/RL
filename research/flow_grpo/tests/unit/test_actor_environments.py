import shlex
from pathlib import Path

from flow_grpo.actor_environments import (
    POLICY_WORKER,
    REWARD_WORKER,
    register_actor_environments,
)

from nemo_rl.distributed.ray_actor_environment_registry import (
    ACTOR_ENVIRONMENT_REGISTRY,
)
from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES


def test_actors_select_research_project(monkeypatch):
    monkeypatch.delenv("NEMO_RL_PY_EXECUTABLES_SYSTEM", raising=False)
    for worker in (POLICY_WORKER, REWARD_WORKER):
        monkeypatch.setitem(ACTOR_ENVIRONMENT_REGISTRY, worker, "previous")
    register_actor_environments()
    project = Path(__file__).resolve().parents[2]
    for worker in (POLICY_WORKER, REWARD_WORKER):
        assert shlex.split(ACTOR_ENVIRONMENT_REGISTRY[worker]) == [
            "uv",
            "run",
            "--locked",
            "--directory",
            str(project),
        ]


def test_actors_respect_system_interpreter(monkeypatch):
    monkeypatch.setenv("NEMO_RL_PY_EXECUTABLES_SYSTEM", "1")
    for worker in (POLICY_WORKER, REWARD_WORKER):
        monkeypatch.setitem(ACTOR_ENVIRONMENT_REGISTRY, worker, "previous")
    register_actor_environments()
    for worker in (POLICY_WORKER, REWARD_WORKER):
        assert ACTOR_ENVIRONMENT_REGISTRY[worker] == PY_EXECUTABLES.SYSTEM
