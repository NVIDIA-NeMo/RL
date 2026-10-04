import os

import pytest
import ray

from nemo_rl.distributed.ray_actor_environment_registry import (
    ACTOR_ENVIRONMENT_REGISTRY,
    PY_EXECUTABLES,
)
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster, init_ray
from nemo_rl.distributed.worker_groups import RayWorkerBuilder, RayWorkerGroup


@ray.remote
class EnvCaptureActor:
    def __init__(self):
        self.env_vars = dict(os.environ)

    def get_env_var(self, var_name):
        return self.env_vars.get(var_name)


ENV_CAPTURE_ACTOR_FQN = f"{EnvCaptureActor.__module__}.EnvCaptureActor"


@pytest.fixture(scope="module", autouse=True)
def ray_cluster():
    init_ray()
    yield
    ray.shutdown()


@pytest.fixture
def register_env_capture_actor():
    original_registry_value = ACTOR_ENVIRONMENT_REGISTRY.get(ENV_CAPTURE_ACTOR_FQN)
    ACTOR_ENVIRONMENT_REGISTRY[ENV_CAPTURE_ACTOR_FQN] = PY_EXECUTABLES.SYSTEM

    yield ENV_CAPTURE_ACTOR_FQN

    if ENV_CAPTURE_ACTOR_FQN in ACTOR_ENVIRONMENT_REGISTRY:
        if original_registry_value is None:
            del ACTOR_ENVIRONMENT_REGISTRY[ENV_CAPTURE_ACTOR_FQN]
        else:
            ACTOR_ENVIRONMENT_REGISTRY[ENV_CAPTURE_ACTOR_FQN] = original_registry_value


@pytest.fixture
def virtual_cluster():
    cluster = RayVirtualCluster(bundle_ct_per_node_list=[2], use_gpus=False)
    yield cluster
    cluster.shutdown()


def test_default_environment_variables_do_not_leak_between_worker_groups(
    register_env_capture_actor, virtual_cluster, monkeypatch
):
    builder = RayWorkerBuilder(register_env_capture_actor)
    leak_var = "NRL_TEST_LEAKED_ENV_VAR"

    monkeypatch.setenv(leak_var, "first-launch-only")
    first_worker_group = RayWorkerGroup(
        cluster=virtual_cluster,
        remote_worker_builder=builder,
        workers_per_node=1,
    )

    assert (
        ray.get(first_worker_group.workers[0].get_env_var.remote(leak_var))
        == "first-launch-only"
    )
    first_worker_group.shutdown(force=True)

    monkeypatch.delenv(leak_var, raising=False)
    second_worker_group = RayWorkerGroup(
        cluster=virtual_cluster,
        remote_worker_builder=builder,
        workers_per_node=1,
    )

    assert ray.get(second_worker_group.workers[0].get_env_var.remote(leak_var)) is None
    second_worker_group.shutdown(force=True)


def test_custom_env_vars_input_is_not_mutated(
    register_env_capture_actor, virtual_cluster, monkeypatch
):
    builder = RayWorkerBuilder(register_env_capture_actor)

    monkeypatch.setenv("NRL_TEST_SYSTEM_ENV_ONLY", "system-value")
    custom_env_vars = {"NRL_TEST_CUSTOM_ENV": "custom-value"}

    worker_group = RayWorkerGroup(
        cluster=virtual_cluster,
        remote_worker_builder=builder,
        workers_per_node=1,
        env_vars=custom_env_vars,
    )

    worker = worker_group.workers[0]
    assert ray.get(worker.get_env_var.remote("NRL_TEST_CUSTOM_ENV")) == "custom-value"
    assert ray.get(worker.get_env_var.remote("NRL_TEST_SYSTEM_ENV_ONLY")) == "system-value"
    assert custom_env_vars == {"NRL_TEST_CUSTOM_ENV": "custom-value"}

    worker_group.shutdown(force=True)
