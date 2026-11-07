# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MegatronDynamoGeneration: the prefill/decode engines behind the frontend and their refit side."""

import asyncio
import copy
import pickle
from types import SimpleNamespace
from typing import Any

import pytest
import ray
import torch
from ray.exceptions import GetTimeoutError

import nemo_rl.models.generation.megatron.dynamo_generation as generation_module
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation import resolve_generation_class
from nemo_rl.models.generation.megatron.dynamo_config import (
    dynamo_enabled,
    engine_width,
    megatron_dynamo_cfg,
)
from nemo_rl.models.generation.megatron.dynamo_generation import (
    MegatronDynamoEngine,
    MegatronDynamoGeneration,
    PinnedHandoffTimeout,
)
from nemo_rl.models.generation.megatron.megatron_generation import MegatronGeneration
from nemo_rl.weight_sync.membership import desired_membership
from tests.unit.models.generation.test_megatron_dynamo_config import (
    MEGATRON_POLICY,
    _megatron_config,
)

pytestmark = pytest.mark.mcore

# The default config: one prefill and two decode engines per replica, TP 2 each.
_ROLES = ["prefill", "decode", "decode"]
_NODES = ["10.0.0.1", "10.0.0.1", "10.0.0.2"]
_IDS = ["prefill-0", "decode-0", "decode-1"]
# Short names for the refit log: instance id -> entry prefix.
_SHORT = {"prefill-0": "p0", "decode-0": "d0", "decode-1": "d1"}
_NAMES = list(_SHORT.values())
_CONTROL_TIMEOUT_S = 600.0

# The fakes the module under test is patched with; set by the ``env`` fixture.
# Workers reach it by name so they stay picklable.
_STATE: "_Env"


class _Ref:
    """Stands in for a Ray object ref; the fake ``ray.get`` unwraps it."""

    def __init__(self, value=None, error=None, hang=False):
        self.value = value
        self.error = error
        self.hang = hang


class _Remote:
    """A remote method: ``.remote(...)`` runs the function and wraps the result.

    Like Ray, a failure surfaces at ``ray.get``, not at ``.remote()``.
    """

    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        try:
            return _Ref(self._fn(*args, **kwargs))
        except Exception as error:
            return _Ref(error=error)


class _FakeRay:
    exceptions = ray.exceptions

    def __init__(self):
        # Every timeout a ray.get was given; the refit tests check the bounds.
        self.get_timeouts: list = []

    def get(self, item, timeout=None):
        self.get_timeouts.append(timeout)
        # Ray raises at timeout 0 unless the value is already there, which a
        # call just issued never is.
        if timeout is not None and timeout <= 0:
            raise GetTimeoutError(f"no time left (timeout={timeout})")
        return self._resolve(item)

    def _resolve(self, item):
        if isinstance(item, list):
            return [self._resolve(entry) for entry in item]
        if not isinstance(item, _Ref):
            return item
        if item.hang:
            raise GetTimeoutError("still running")
        if item.error is not None:
            raise item.error
        return item.value

    @staticmethod
    def put(value):
        return value


class _FakePlacementGroup:
    def __init__(self, bundle_nodes):
        self.bundle_nodes = bundle_nodes
        self.bundle_count = len(bundle_nodes)


class _FakeCluster:
    """One engine's ``RayVirtualCluster``."""

    def __init__(self, env: "_Env", kwargs):
        self.kwargs = kwargs
        self.log = env.log
        self.name = kwargs["name"].rsplit("_", 1)[-1]
        index = len(env.clusters)
        (width,) = kwargs["bundle_ct_per_node_list"]
        bundle_nodes = env.bundle_nodes.get(index, [env.nodes[index]] * width)
        self.placement_group = _FakePlacementGroup(bundle_nodes)
        self.placement_policy_config = None
        self.log.append(f"cluster {self.name} up")

    def get_placement_groups(self):
        return [self.placement_group]

    def shutdown(self):
        self.log.append(f"cluster {self.name} down")
        return True


class _FakeWorker:
    """One Ray worker of an engine: ``worker.method.remote(**kwargs)``.

    Holds only its engine's short name and its rank, so a pickled generation
    object (the SingleController actor's copy) carries it.
    """

    def __init__(self, short: str, rank: int) -> None:
        self.short = short
        self.rank = rank

    def __getattr__(self, method: str):
        if method.startswith("__"):
            raise AttributeError(method)
        short, rank = self.short, self.rank
        return SimpleNamespace(
            remote=lambda *args, **kwargs: _STATE.engines_by_short[short].call(
                rank, method, kwargs
            )
        )


class _FakeEngine:
    """Stand-in for one inner ``MegatronGeneration``: a worker group, leader first."""

    def __init__(self, env: "_Env", kwargs):
        self.kwargs = kwargs
        self.log = env.log
        self.name = kwargs["name_prefix"].removeprefix("dynamo_")
        self.short = _SHORT.get(self.name, self.name)
        self.env = env
        self.alive = True
        # The probe raises RayActorError, as Ray does for a dead actor.
        self.actor_dead = False
        self.coordinator = None
        self.fail_shutdown = False
        self.unclean_shutdown = False
        self.fail_calls = False
        self.reject_attestation = False
        self.shutdown_calls = 0
        # (rank, parent_event_address) per attach_dynamo_events call.
        self.attached: list[tuple[int, str]] = []
        self.metadata = None
        self.endpoint_kwargs = None
        # What rank zero's dynamo_endpoint_error reports; None while serving.
        self.endpoint_error = None
        # Refit fakes: rank zero's calls land in the log. ``pinned`` is a
        # scripted sequence of pinned-handoff reads (the last value repeats);
        # ``faults`` maps a worker method to an exception, or "hang", or (for
        # an endpoint method) "disconnected", which raises from ``.remote()``.
        self.pinned = list(env.pinned.get(self.name, [0]))
        self.faults: dict[str, object] = {}
        # Every method rank zero was called with, in call order.
        self.calls: list[str] = []
        (width,) = kwargs["cluster"].kwargs["bundle_ct_per_node_list"]
        workers = [_FakeWorker(self.short, rank) for rank in range(width)]
        self.worker_group = SimpleNamespace(
            workers=workers,
            get_dp_leader_worker_idx=lambda shard: 0,
            py_executable="/venvs/mcore/bin/python",
        )
        self.log.append(f"engine {self.name} up")

    def call(self, rank: int, method: str, kwargs: dict) -> _Ref:
        if rank == 0:
            self.calls.append(method)
        if method == "is_alive":
            if self.actor_dead:
                return _Ref(error=ray.exceptions.RayActorError())
            return _Ref(self.alive)
        if method in (
            "start_dynamo_endpoint",
            "attach_dynamo_events",
            "dynamo_endpoint_metadata",
            "stop_dynamo_endpoint",
            "confirm_decode_terminated",
            "dynamo_coordinator_address",
            "dynamo_endpoint_error",
        ):
            fault = self.faults.get(method)
            if fault == "disconnected":
                # Ray gone: issuing the call raises, as after ray.shutdown().
                raise AttributeError("'Worker' object has no attribute 'core_worker'")
            if fault == "hang":
                return _Ref(hang=True)
            if fault is not None:
                return _Ref(error=fault)
            try:
                return _Ref(getattr(self, f"_{method}")(rank, **kwargs))
            except Exception as error:
                return _Ref(error=error)
        if rank == 0:
            self.log.append(self._entry(method, kwargs))
        fault = self.faults.get(method)
        if fault == "hang":
            return _Ref(hang=True)
        if fault is not None:
            return _Ref(error=fault)
        if method == "pinned_handoff_block_count":
            return _Ref(self.pinned.pop(0) if len(self.pinned) > 1 else self.pinned[0])
        return _Ref(True)

    def _entry(self, method: str, kwargs: dict) -> str:
        short = {
            "pause_for_refit": "pause",
            "suspend_for_refit": "suspend",
            "resume_after_refit": "resume",
            "pinned_handoff_block_count": "pinned",
            "prepare_refit_info": "prepare",
            "update_weights_from_collective": "update",
        }
        if method == "init_collective":
            return (
                f"{self.short}.init(ws={kwargs['world_size']},"
                f"train={kwargs['train_world_size']},off={kwargs['rank_offset']})"
            )
        if method in ("prepare_for_generation", "set_rollout_weight_version"):
            (value,) = kwargs.values()
            return f"{self.short}.{method}({value})"
        return f"{self.short}.{short[method]}"

    @property
    def role(self) -> str:
        return self.kwargs["config"]["generation"]["_dynamo_role"]

    def _start_dynamo_endpoint(self, rank, **kwargs):
        """Rank zero's endpoint start: node-local ports, an owner for decode."""
        assert rank == 0
        if self.metadata is not None:
            return self.metadata
        index = len(self.env.endpoints)
        if self.env.fail_endpoint_at == index:
            raise RuntimeError(f"endpoint {self.name} failed to start")
        self.endpoint_kwargs = kwargs
        node_ip = self.kwargs["cluster"].placement_group.bundle_nodes[0]
        self.event_port = 7000 + 2 * index
        self.system_port = self.event_port + 1
        self.owner = f"owner-{self.name}" if self.role == "decode" else None
        self.env.endpoints.append(self)
        self.log.append(f"endpoint {self.name} {self.role}")
        self.metadata = {
            "node_ip": node_ip,
            "event_port": self.event_port,
            "system_port": self.system_port,
            "parent_event_address": f"tcp://{node_ip}:{self.event_port}",
            "handoff_owner": self.owner,
        }
        return self.metadata

    def _attach_dynamo_events(self, rank, *, parent_event_address):
        self.attached.append((rank, parent_event_address))
        if rank == 0:
            self.log.append(f"attach {self.name}")
        return True

    def _dynamo_endpoint_metadata(self, rank):
        if self.fail_calls:
            raise ray.exceptions.RayActorError()
        return self.metadata

    def _stop_dynamo_endpoint(self, rank):
        assert rank == 0
        if self.fail_calls:
            raise RuntimeError(f"engine {self.name} is gone")
        self.log.append(f"stop-endpoint {self.name}")

    def _dynamo_endpoint_error(self, rank):
        assert rank == 0
        return self.endpoint_error

    def _confirm_decode_terminated(
        self, rank, *, coordinator_addresses, owner, timeout_s
    ):
        assert rank == 0
        if self.fail_calls:
            raise RuntimeError(f"engine {self.name} is gone")
        self.log.append(
            f"confirm {self.name} {owner} {coordinator_addresses} {timeout_s}"
        )
        return not self.reject_attestation

    def _dynamo_coordinator_address(self, rank):
        assert rank == 0
        return self.coordinator

    def shutdown(self):
        self.shutdown_calls += 1
        if self.fail_shutdown:
            raise RuntimeError(f"engine {self.name} shutdown failed")
        self.log.append(f"engine {self.name} down")
        return not self.unclean_shutdown


class _FakeRuntime:
    """Stand-in for ``ManagedDynamoRuntime``: etcd, NATS and the frontend."""

    def __init__(self, env: "_Env", *, dynamo_cfg, model_name, frontend_python):
        self.dynamo_cfg = dynamo_cfg
        self.model_name = model_name
        self.frontend_python = frontend_python
        self.log = env.log
        self.waits: list[Any] = []
        self.namespace = "ns"
        self.engine_env = {"ETCD_ENDPOINTS": "http://etcd:2379"}
        self.frontend_url = "http://10.0.0.1:3000/v1"
        env.runtimes.append(self)

    def start(self):
        self.log.append("runtime up")

    def wait_for_engines(self, expected_components, check_engines):
        check_engines()
        self.waits.append(dict(expected_components))

    def shutdown(self):
        self.log.append("runtime down")


class _NodeIpTask:
    def options(self, *, scheduling_strategy):
        placement_group = scheduling_strategy.placement_group
        index = scheduling_strategy.placement_group_bundle_index
        return _Remote(lambda: placement_group.bundle_nodes[index])


class _Env:
    """Everything the engine build touches, patched into the generation module."""

    def __init__(self, monkeypatch, nodes):
        global _STATE
        _STATE = self
        self.nodes = nodes
        self.bundle_nodes: dict[int, list[str]] = {}
        self.log: list[str] = []
        self.clusters: list[_FakeCluster] = []
        self.engines: list[_FakeEngine] = []
        self.engines_by_short: dict[str, _FakeEngine] = {}
        # Engines whose rank zero started its Dynamo endpoint, in start order.
        self.endpoints: list[_FakeEngine] = []
        self.runtimes: list[_FakeRuntime] = []
        self.fail_engine_at = None
        self.fail_endpoint_at = None
        # instance id -> scripted pinned-handoff reads for that engine.
        self.pinned: dict[str, list[int]] = {}
        self.ray = _FakeRay()
        # Every generation built here; torn down by close().
        self.generations: list[MegatronDynamoGeneration] = []

        def make_cluster(**kwargs):
            cluster = _FakeCluster(self, kwargs)
            self.clusters.append(cluster)
            return cluster

        def make_engine(**kwargs):
            if self.fail_engine_at == len(self.engines):
                raise RuntimeError("engine failed to start")
            engine = _FakeEngine(self, kwargs)
            self.engines.append(engine)
            self.engines_by_short[engine.short] = engine
            return engine

        def init_cluster_placement_groups(cluster, config):
            cluster.placement_policy_config = config
            self.log.append(f"placement {cluster.name}")

        make_engine.init_cluster_placement_groups = init_cluster_placement_groups

        # The per-engine MegatronGeneration is looked up by name at build time.
        monkeypatch.setattr(generation_module, "MegatronGeneration", make_engine)
        monkeypatch.setattr(generation_module, "RayVirtualCluster", make_cluster)
        monkeypatch.setattr(generation_module, "cluster_node_ip", _NodeIpTask())
        monkeypatch.setattr(generation_module, "ray", self.ray)
        monkeypatch.setattr(
            generation_module,
            "ManagedDynamoRuntime",
            lambda **kwargs: _FakeRuntime(self, **kwargs),
        )

    def generation(
        self,
        generation_config: dict | None = None,
        *,
        world_size: int = 6,
        policy: dict | None = None,
        **cluster_overrides,
    ) -> MegatronDynamoGeneration:
        policy = policy or {
            **MEGATRON_POLICY,
            "generation": generation_config or _megatron_config(),
        }
        cluster = dict(
            world_size=lambda: world_size,
            num_gpus_per_node=8,
            port_range_low=None,
            port_range_high=None,
            node_resource_constraints=None,
        )
        cluster.update(cluster_overrides)
        generation = MegatronDynamoGeneration(
            config=policy,  # type: ignore[arg-type]
            tokenizer="tok",  # type: ignore[arg-type]
            cluster=SimpleNamespace(**cluster),  # type: ignore[arg-type]
        )
        # The pin-drain poll must not sleep in tests.
        generation._idle_poll_interval_s = 0.0
        self.generations.append(generation)
        return generation

    def started(
        self, pinned: list[int] | None = None, **kwargs
    ) -> MegatronDynamoGeneration:
        """A built generation object with the start-up log cleared.

        ``pinned`` scripts the prefill engine's pinned-handoff reads.
        """
        self.pinned = {"prefill-0": pinned} if pinned is not None else {}
        generation = self.generation(**kwargs)
        self.log.clear()
        self.ray.get_timeouts.clear()
        return generation

    def engine(self, short: str) -> _FakeEngine:
        return self.engines_by_short[short]

    @property
    def runtime(self) -> _FakeRuntime:
        (runtime,) = self.runtimes
        return runtime

    def refit_timeouts(self) -> list:
        """Timeouts of every bounded get since they were last cleared.

        Liveness probes are included: the pin wait probes every engine on each
        poll that finds pins, and the first resume's readiness wait probes them
        too.
        """
        return [t for t in self.ray.get_timeouts if t is not None]

    def close(self) -> None:
        """Tear every generation down while this env's fakes are the ones it reaches.

        One left to the GC would run ``__del__`` -> ``shutdown()`` during a
        later test, against that test's engines and log.
        """
        for generation in self.generations:
            generation.shutdown()


@pytest.fixture
def env(monkeypatch):
    env = _Env(monkeypatch, _NODES)
    yield env
    env.close()


# One replica of two prefill and two decode engines, TP 2 each.
_TWO_PREFILL_NODES = ["10.0.0.1", "10.0.0.1", "10.0.0.2", "10.0.0.2"]


def _two_prefill_config() -> dict:
    config = _megatron_config()
    config["disaggregation"]["num_prefill_engines"] = 2
    return config


@pytest.fixture
def env2(monkeypatch):
    env = _Env(monkeypatch, _TWO_PREFILL_NODES)
    yield env
    env.close()


def _with_role(policy_config: dict, role: str) -> dict:
    """``policy_config`` as an engine of ``role`` is built from it."""
    return {
        **policy_config,
        "generation": {
            **policy_config["generation"],
            "dynamo": False,
            "_dynamo_role": role,
        },
    }


def _widths(width: int) -> dict[str, int]:
    """One width for both roles."""
    return {"prefill": width, "decode": width}


def _with_layout(prefill: dict | None = None, decode: dict | None = None) -> dict:
    generation = _megatron_config()
    generation["dynamo_cfg"].update(
        prefill_engine=prefill or {}, decode_engine=decode or {}
    )
    return generation


# ---------------------------------------------------------------------------
# Building the engines
# ---------------------------------------------------------------------------


def test_builds_one_cluster_and_engine_per_role(env):
    generation = env.generation()

    records = generation.engine_records
    assert [r.instance_id for r in records] == _IDS
    assert [r.role for r in records] == _ROLES
    assert [r.node_ip for r in records] == _NODES
    assert all(r.world_size == 2 for r in records)
    assert generation.engine_records is not generation.engine_records
    assert generation.shard_widths == (2, 2, 2)
    assert generation.inference_world_size == 6
    assert generation.get_inference_world_size() == 6
    assert generation.num_engines == 3
    assert generation.expected_components() == {"prefill": 1, "backend": 2}
    # Engines first, then the runtime from their environment, as for vLLM. No
    # endpoint yet: the engines have no weights until the first refit.
    expected_log = []
    for name in _IDS:
        expected_log += [f"cluster {name} up", f"placement {name}", f"engine {name} up"]
    assert env.log == expected_log + ["runtime up"] and env.endpoints == []
    assert env.runtime.frontend_python == "/venvs/mcore/bin/python"
    assert env.runtime.model_name == "Qwen/Qwen3-0.6B"
    assert generation._dynamo_namespace == "ns"
    assert generation.dp_openai_server_base_urls == ["http://10.0.0.1:3000/v1"]

    cluster = env.clusters[0]
    assert cluster.kwargs == {
        "name": f"dynamo_{generation._fleet_id}_prefill-0",
        "bundle_ct_per_node_list": [2],
        "use_gpus": True,
        "num_gpus_per_node": 8,
        "max_colocated_worker_groups": 1,
        "port_range_low": None,
        "port_range_high": None,
        "node_resource_constraints": None,
    }
    # No role overrides: each engine builds from a copy of the shared policy
    # that names its role.
    for engine, role in zip(env.engines, _ROLES):
        assert engine.kwargs["config"] == _with_role(generation._policy_config, role)
    assert cluster.placement_policy_config == env.engines[0].kwargs["config"]
    engine = env.engines[0]
    assert engine.kwargs["cluster"] is cluster
    assert engine.kwargs["skip_weight_load"] is True
    assert engine.kwargs["tokenizer"] == "tok"
    # This object owns no policy or worker group of its own.
    with pytest.raises(AttributeError, match="no single worker group"):
        generation.worker_group
    # The collective synchronizer probes with getattr and must see no group.
    assert getattr(generation, "worker_group", None) is None
    assert generation.invalidate_kv_cache() is False
    assert generation.blocks_training() is False
    assert generation.prepare_for_generation() is True
    assert generation.finish_generation() is True
    assert generation.shard_liveness_ref(1).value is True


def _layout(engine_config: dict) -> tuple[int, int]:
    """(TP, PP) the Megatron workers of a policy config are built with."""
    from nemo_rl.models.generation.megatron.config import (
        merged_inference_megatron_cfg,
    )

    merged = merged_inference_megatron_cfg(engine_config)
    return merged["tensor_model_parallel_size"], merged["pipeline_model_parallel_size"]


@pytest.mark.parametrize(
    ("prefill", "decode", "widths", "layouts"),
    [
        # 1 prefill at TP 4 next to 2 decode at the shared TP 2.
        (
            {"tensor_model_parallel_size": 4},
            {},
            {"prefill": 4, "decode": 2},
            [(4, 1), (2, 1), (2, 1)],
        ),
        # Decode moves to PP 2 instead; prefill keeps the shared layout.
        (
            {},
            {"pipeline_model_parallel_size": 2},
            {"prefill": 2, "decode": 4},
            [(2, 1), (2, 2), (2, 2)],
        ),
    ],
    ids=["prefill_tp4", "decode_pp2"],
)
def test_each_role_builds_with_its_own_layout(env, prefill, decode, widths, layouts):
    """Cluster width, record width and engine config follow the role."""
    generation_config = _with_layout(prefill, decode)
    shared = {**MEGATRON_POLICY, "generation": generation_config}
    replica = widths["prefill"] + 2 * widths["decode"]

    generation = env.generation(generation_config, world_size=replica)

    records = generation.engine_records
    assert [r.world_size for r in records] == [widths[role] for role in _ROLES]
    assert [c.kwargs["bundle_ct_per_node_list"] for c in env.clusters] == [
        [widths[role]] for role in _ROLES
    ]
    assert [_layout(e.kwargs["config"]) for e in env.engines] == layouts
    assert [_layout(c.placement_policy_config) for c in env.clusters] == layouts
    # The shared policy config is untouched; each engine got its own copy.
    assert generation._policy_config == shared
    for role in ("prefill", "decode"):
        assert generation_module.role_policy_config(shared, role, {}) == _with_role(
            shared, role
        )


@pytest.mark.parametrize(
    ("mcore_generation_config", "overrides", "layout"),
    [
        # A null shared block is the policy layout plus the role's overrides.
        (None, {"tensor_model_parallel_size": 4}, (4, 1)),
        # Overrides win over the shared block's layout keys.
        (
            {"pipeline_model_parallel_size": 2},
            {"tensor_model_parallel_size": 4},
            (4, 2),
        ),
    ],
    ids=["null_shared_block", "shared_block_present"],
)
def test_role_policy_config_merges_overrides_into_a_copy(
    mcore_generation_config, overrides, layout
):
    generation = {
        **_megatron_config(),
        "mcore_generation_config": mcore_generation_config,
    }
    shared = {**MEGATRON_POLICY, "generation": generation}
    before = copy.deepcopy(shared)

    merged = generation_module.role_policy_config(shared, "decode", overrides)

    assert _layout(merged) == layout
    assert merged["generation"]["_dynamo_role"] == "decode"
    assert shared == before
    assert merged["generation"] is not shared["generation"]


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_engine_config_builds_a_plain_engine_of_its_own_width(role):
    """An engine's config has Dynamo off, so it is sized as one engine.

    With ``dynamo`` left on, the engine's NVLink span would be a whole
    prefill/decode replica instead of its own width.
    """
    generation = _megatron_config()
    generation["dynamo_cfg"]["prefill_engine"] = {"tensor_model_parallel_size": 4}
    shared = {**MEGATRON_POLICY, "generation": generation}

    engine_config = generation_module.role_policy_config(
        shared, role, megatron_dynamo_cfg(generation).role_overrides(role)
    )

    assert not dynamo_enabled(engine_config["generation"])
    assert dynamo_enabled(shared["generation"])
    assert resolve_generation_class(engine_config["generation"]) is MegatronGeneration
    assert MegatronGeneration.nvlink_domain_span(engine_config) == engine_width(
        shared, role
    )


@pytest.mark.parametrize(
    ("prefill", "decode", "world_size", "match"),
    [
        pytest.param(
            {"tensor_model_parallel_size": 16},
            {"tensor_model_parallel_size": 16},
            48,
            "multi-node engines",
            id="multi_node",
        ),
        pytest.param(
            {},
            {"tensor_model_parallel_size": 16},
            34,
            r"decode engine spans 16 GPUs.*decode_engine",
            id="named_role",
        ),
    ],
)
def test_rejects_engines_wider_than_a_node(env, prefill, decode, world_size, match):
    with pytest.raises(ValueError, match=match):
        env.generation(_with_layout(prefill, decode), world_size=world_size)
    assert env.log == []


_PINS = [{"nvl-a": 0.001}, {"nvl-b": 0.001}]


@pytest.mark.parametrize(
    ("prefill_tp", "decode_tp", "pin_indices"),
    [
        # Two 4-GPU engines per 8-GPU node: the first two take node 0's pin.
        pytest.param(4, 4, [0, 0, 1], id="uniform_4"),
        # 4 + 2 + 2 fills node 0.
        pytest.param(4, 2, [0, 0, 0], id="mixed_fits_one_node"),
        # 8 + 4 + 4 needs node 1 for both decode engines.
        pytest.param(8, 4, [0, 1, 1], id="mixed_spills"),
    ],
)
def test_engine_clusters_tile_pinned_nodes_in_launch_order(
    env, prefill_tp, decode_tp, pin_indices
):
    """Each engine sits on one pinned node; a full node moves to the next pin.

    Engine clusters also inherit the inference cluster's port range.
    """
    generation_config = _with_layout(
        {"tensor_model_parallel_size": prefill_tp},
        {"tensor_model_parallel_size": decode_tp},
    )
    env.generation(
        generation_config,
        world_size=prefill_tp + 2 * decode_tp,
        port_range_low=2100,
        port_range_high=2200,
        node_resource_constraints=_PINS,
    )

    assert [c.kwargs["node_resource_constraints"] for c in env.clusters] == [
        [_PINS[index]] for index in pin_indices
    ]
    assert {
        (c.kwargs["port_range_low"], c.kwargs["port_range_high"]) for c in env.clusters
    } == {(2100, 2200)}


@pytest.mark.parametrize(
    ("prefill_tp", "decode_tp", "match"),
    [
        pytest.param(4, 4, "more than the 1 pinned inference nodes", id="count"),
        pytest.param(
            8, 4, r"widths \{'prefill': 8, 'decode': 4\}.*tile 2 nodes", id="tiling"
        ),
    ],
)
def test_too_few_pinned_nodes_for_the_engines_is_rejected(
    env, prefill_tp, decode_tp, match
):
    generation_config = _with_layout(
        {"tensor_model_parallel_size": prefill_tp},
        {"tensor_model_parallel_size": decode_tp},
    )
    with pytest.raises(ValueError, match=match):
        env.generation(
            generation_config,
            world_size=prefill_tp + 2 * decode_tp,
            node_resource_constraints=_PINS[:1],
        )
    assert env.log == []


def test_start_failure_tears_down_what_exists(env, caplog):
    # The third engine fails to come up.
    env.fail_engine_at = 2
    with caplog.at_level("ERROR"), pytest.raises(RuntimeError, match="failed to start"):
        env.generation()
    down = [line for line in env.log if line.endswith(" down")]
    assert set(down) == {
        "engine prefill-0 down",
        "engine decode-0 down",
        "cluster prefill-0 down",
        "cluster decode-0 down",
        "cluster decode-1 down",
    }
    # Decode first, then prefill.
    assert [line for line in env.log if line.startswith("stop-endpoint")] == [
        "stop-endpoint decode-0",
        "stop-endpoint prefill-0",
    ]
    # Engines without a record are not mistaken for a broken set.
    assert "prefill coordinator addresses" not in caplog.text
    # No service was started for engines that never came up.
    assert env.runtimes == []


def test_engine_split_across_nodes_is_rejected(env):
    env.bundle_nodes[1] = ["10.0.0.1", "10.0.0.2"]
    with pytest.raises(RuntimeError, match=r"decode-0 spans multiple nodes"):
        env.generation()
    # Rejected before the engine's workers exist.
    assert [engine.name for engine in env.engines] == ["prefill-0"]
    assert "cluster decode-0 down" in env.log


def test_rejects_colocation_and_a_cluster_that_does_not_fit_the_replica(env):
    with pytest.raises(ValueError, match="colocated.enabled=false"):
        MegatronDynamoGeneration(
            config={**MEGATRON_POLICY, "generation": _megatron_config()},  # type: ignore[arg-type]
            tokenizer=object(),  # type: ignore[arg-type]
            policy=object(),  # type: ignore[arg-type]
        )
    with pytest.raises(NotImplementedError, match="text prompts only"):
        MegatronDynamoGeneration(
            config={**MEGATRON_POLICY, "generation": _megatron_config()},  # type: ignore[arg-type]
            tokenizer=object(),  # type: ignore[arg-type]
            cluster=object(),  # type: ignore[arg-type]
            processor=object(),  # type: ignore[arg-type]
        )
    generation_config = _megatron_config()
    generation_config["disaggregation"]["num_prefill_engines"] = 2
    with pytest.raises(ValueError, match="not a multiple of one replica"):
        env.generation(generation_config, world_size=4)
    assert env.log == [] and env.runtimes == []


_COORDINATOR = "tcp://10.0.0.1:5555"
_DECODE_STOPS = [
    "stop-endpoint decode-0",
    "engine decode-0 down",
    "stop-endpoint decode-1",
    "engine decode-1 down",
]
_PREFILL_STOP = ["stop-endpoint prefill-0", "engine prefill-0 down"]
_CLUSTERS_DOWN = [
    "cluster prefill-0 down",
    "cluster decode-0 down",
    "cluster decode-1 down",
]


def _confirm(owner: str) -> str:
    return f"confirm prefill-0 {owner} ['{_COORDINATOR}'] {_CONTROL_TIMEOUT_S}"


@pytest.mark.parametrize(
    ("coordinator", "faults", "expected_log", "warnings"),
    [
        # Decode engines stop first; the prefill engine's rank zero attests the
        # stopped decode owners before it stops itself; the services stop last.
        pytest.param(
            True,
            {},
            _DECODE_STOPS
            + [_confirm("owner-decode-0"), _confirm("owner-decode-1")]
            + _PREFILL_STOP
            + _CLUSTERS_DOWN,
            [],
            id="clean",
        ),
        pytest.param(
            False,
            {},
            _DECODE_STOPS + _PREFILL_STOP + _CLUSTERS_DOWN,
            [
                "No prefill coordinator address could be read; decode owners "
                "['owner-decode-0', 'owner-decode-1'] are not attested as terminated"
            ],
            id="no_prefill_coordinator",
        ),
        # Ray is gone, so issuing the owner or address reads raises outright;
        # every engine, cluster and the runtime still stop, unattested.
        pytest.param(
            True,
            {"d0": {"faults": {"dynamo_endpoint_metadata": "disconnected"}}},
            _DECODE_STOPS + _PREFILL_STOP + _CLUSTERS_DOWN,
            ["Could not read the decode engines' handoff owners"],
            id="owner_reads_cannot_be_issued",
        ),
        pytest.param(
            True,
            {"p0": {"faults": {"dynamo_coordinator_address": "disconnected"}}},
            _DECODE_STOPS + _PREFILL_STOP + _CLUSTERS_DOWN,
            ["Could not read prefill coordinator addresses"],
            id="address_reads_cannot_be_issued",
        ),
        # The attester's endpoint calls raise and decode-0 fails to stop at all.
        pytest.param(
            True,
            {
                "p0": {"fail_calls": True},
                "d0": {"fail_calls": True, "fail_shutdown": True},
            },
            ["stop-endpoint decode-1", "engine decode-1 down", "engine prefill-0 down"]
            + _CLUSTERS_DOWN,
            [
                "Could not stop a Megatron engine's Dynamo endpoint",
                "Could not stop a Megatron engine",
                "Could not read the handoff owner of decode-0",
                "Could not attest decode owner owner-decode-1 termination",
            ],
            id="continues_past_failures",
        ),
        # Only cleanly stopped decode engines are attested; a refused
        # attestation is logged, not raised.
        pytest.param(
            True,
            {"d0": {"unclean_shutdown": True}, "p0": {"reject_attestation": True}},
            _DECODE_STOPS
            + [_confirm("owner-decode-1")]
            + _PREFILL_STOP
            + _CLUSTERS_DOWN,
            [
                "did not shut down cleanly",
                "Decode engine decode-0 did not stop cleanly",
                "did not accept decode owner owner-decode-1",
            ],
            id="attests_only_clean_stops",
        ),
    ],
)
def test_shutdown_stops_decode_attests_owners_then_prefill_then_the_runtime(
    env, caplog, coordinator, faults, expected_log, warnings
):
    generation = env.started()
    _start_endpoints(generation)
    if coordinator:
        env.engine("p0").coordinator = _COORDINATOR
    assert generation._prefill_coordinator_addresses() == (
        {0: _COORDINATOR} if coordinator else {}
    )
    for short, flags in faults.items():
        vars(env.engine(short)).update(flags)
    env.log.clear()

    with caplog.at_level("WARNING"):
        assert generation.shutdown() is True

    assert env.log == expected_log + ["runtime down"]
    for warning in warnings:
        assert warning in caplog.text
    assert generation.engine_records == [] and generation.num_engines == 0
    assert generation.expected_components() == {}
    env.log.clear()
    assert generation.shutdown() is True
    assert env.log == []


def test_shutdown_bounds_every_endpoint_call_and_still_stops_the_engines(env, caplog):
    """A hung endpoint call is cut off at the control timeout; the engine still stops."""
    generation = env.started()
    _start_endpoints(generation)
    env.engine("p0").coordinator = _COORDINATOR
    env.engine("d1").faults["dynamo_endpoint_metadata"] = "hang"
    env.engine("d1").faults["stop_dynamo_endpoint"] = "hang"
    env.engine("p0").faults["confirm_decode_terminated"] = "hang"
    env.log.clear()
    env.ray.get_timeouts.clear()

    with caplog.at_level("WARNING"):
        assert generation.shutdown() is True

    assert [line for line in env.log if line.startswith("engine ")] == [
        "engine decode-0 down",
        "engine decode-1 down",
        "engine prefill-0 down",
    ]
    assert "Decode engine decode-1 did not stop cleanly" in caplog.text
    assert "Could not read the handoff owner of decode-1" in caplog.text
    # decode-0 stopped cleanly; the attestation of its owner hung and was cut
    # off at the bound.
    assert "Could not attest decode owner owner-decode-0 termination" in caplog.text
    assert all(0 < t <= _CONTROL_TIMEOUT_S + 10.0 for t in env.ray.get_timeouts)
    assert _CONTROL_TIMEOUT_S + 10.0 in env.ray.get_timeouts


@pytest.mark.parametrize(
    "fault",
    ["hang", ray.exceptions.RayActorError()],
    ids=["address_read_times_out", "address_read_fails"],
)
def test_shutdown_attests_through_the_prefill_engines_that_answered(
    env2, caplog, fault
):
    """One prefill engine's unreadable address keeps no other from releasing pins."""
    generation = env2.generation(_two_prefill_config(), world_size=8)
    _start_endpoints(generation)
    second = "tcp://10.0.0.2:5555"
    env2.engine("p0").coordinator = _COORDINATOR
    env2.engine("prefill-1").coordinator = second
    env2.engine("p0").faults["dynamo_coordinator_address"] = fault
    env2.log.clear()

    with caplog.at_level("WARNING"):
        assert generation.shutdown() is True

    # prefill-1 answered, so its rank zero attests both stopped decode owners
    # to its own coordinator; prefill-0 still stops normally.
    assert env2.log == [
        *_DECODE_STOPS,
        f"confirm prefill-1 owner-decode-0 ['{second}'] {_CONTROL_TIMEOUT_S}",
        f"confirm prefill-1 owner-decode-1 ['{second}'] {_CONTROL_TIMEOUT_S}",
        "stop-endpoint prefill-0",
        "engine prefill-0 down",
        "stop-endpoint prefill-1",
        "engine prefill-1 down",
        "cluster prefill-0 down",
        "cluster prefill-1 down",
        "cluster decode-0 down",
        "cluster decode-1 down",
        "runtime down",
    ]
    assert "Could not read the coordinator address of prefill-0" in caplog.text


def test_shutdown_asks_no_prefill_engine_without_a_decode_owner(env):
    """Before the first refit starts the endpoints there is no owner to attest."""
    generation = env.started()
    env.engine("p0").coordinator = _COORDINATOR

    assert generation.shutdown() is True

    assert "dynamo_coordinator_address" not in env.engine("p0").calls
    assert env.log == _DECODE_STOPS + _PREFILL_STOP + _CLUSTERS_DOWN + ["runtime down"]


# ---------------------------------------------------------------------------
# Serving through the frontend: endpoints and the readiness wait
# ---------------------------------------------------------------------------


def _start_endpoints(generation) -> None:
    generation.start_endpoints(
        namespace="ns",
        engine_env={"ETCD_ENDPOINTS": "http://etcd:2379"},
        model_name="Qwen/Qwen3-0.6B",
        timeout_s=_CONTROL_TIMEOUT_S,
    )


def test_start_endpoints_starts_rank_zero_then_attaches_every_rank(env):
    generation = env.started()

    _start_endpoints(generation)

    # Engine by engine: rank zero's endpoint, then every rank's events.
    assert env.log == [
        line
        for name, role in zip(_IDS, _ROLES)
        for line in (f"endpoint {name} {role}", f"attach {name}")
    ]
    for engine in env.engines:
        address = engine.metadata["parent_event_address"]
        assert engine.attached == [(0, address), (1, address)]
    assert [e.owner for e in env.endpoints] == [
        None,
        "owner-decode-0",
        "owner-decode-1",
    ]
    # The second engine on 10.0.0.1 avoids the first one's ports; the engine on
    # 10.0.0.2 starts fresh.
    first = env.endpoints[0]
    assert [e.endpoint_kwargs["excluded_ports"] for e in env.endpoints] == [
        [],
        [first.event_port, first.system_port],
        [],
    ]
    assert first.endpoint_kwargs == {
        "namespace": "ns",
        "manager_env": {"ETCD_ENDPOINTS": "http://etcd:2379"},
        "model_name": "Qwen/Qwen3-0.6B",
        "served_model_name": "Qwen/Qwen3-0.6B",
        "endpoint_types": ["completions"],
        "engine_start_timeout_s": 600,
        "excluded_ports": [],
    }
    # Later refits do not start them again.
    env.log.clear()
    _start_endpoints(generation)
    assert env.log == []


def test_start_endpoints_names_the_engine_that_failed(env):
    env.fail_endpoint_at = 1
    generation = env.started()
    with pytest.raises(RuntimeError, match="failed to start") as raised:
        _start_endpoints(generation)
    assert "decode-0" in str(raised.value.__notes__)
    # The next attempt resumes where the workers left off.
    env.fail_endpoint_at = None
    _start_endpoints(generation)
    assert [e.name for e in env.endpoints] == _IDS


def test_check_alive_names_the_dead_engines(env):
    generation = env.started()
    generation.check_alive()
    env.engine("d0").alive = False
    with pytest.raises(RuntimeError, match="died: decode-0"):
        generation.check_alive()


def test_check_alive_names_the_engine_whose_endpoint_stopped(env):
    """Once the endpoints started, a failed or exited endpoint task fails the check."""
    generation = env.started()
    env.engine("d1").endpoint_error = "RuntimeError('adapter failed')"
    # Before the endpoints start there is nothing to report.
    generation.check_alive()
    _start_endpoints(generation)
    env.ray.get_timeouts.clear()
    with pytest.raises(
        RuntimeError,
        match=r"stopped serving: decode-1: RuntimeError\('adapter failed'\)",
    ):
        generation.check_alive()
    timeouts = env.ray.get_timeouts
    assert len(timeouts) == 4
    # One deadline spans the three probes and the endpoint check, so every
    # later call gets strictly less of it.
    assert all(0 < t < _CONTROL_TIMEOUT_S for t in timeouts)
    assert timeouts == sorted(timeouts, reverse=True)


def test_check_alive_counts_a_dead_actor_as_dead(env):
    """A dead actor raises on the probe rather than answering False."""
    generation = env.started()
    env.engine("d0").actor_dead = True
    env.engine("d1").actor_dead = True
    with pytest.raises(RuntimeError, match="died: decode-0, decode-1") as raised:
        generation.check_alive()
    assert isinstance(raised.value.__cause__, ray.exceptions.RayActorError)


def test_first_resume_starts_the_endpoints_and_awaits_the_frontend(env):
    generation = env.started()

    generation.suspend_for_refit()
    generation.prepare_for_generation(tags=["weights"])
    generation.prepare_for_generation(tags=["kv_cache"])
    assert env.runtime.waits == [] and env.endpoints == []
    env.log.clear()
    generation.resume_after_refit()
    # The first resume is the engines' startup: the refit built them, so their
    # endpoints start now, and the frontend must list both components. Every
    # engine resumes in one wave, in record order.
    assert env.log == _RESUMED + [
        line
        for name, role in zip(_IDS, _ROLES)
        for line in (f"endpoint {name} {role}", f"attach {name}")
    ]
    assert env.runtime.waits == [{"prefill": 1, "backend": 2}]
    env.log.clear()
    generation.resume_after_refit()
    assert env.log == _RESUMED
    assert env.runtime.waits == [{"prefill": 1, "backend": 2}], "awaited once"


def test_pickled_copy_awaits_the_frontend_over_http(env, monkeypatch):
    generation = env.started()
    clone = pickle.loads(pickle.dumps(generation))
    assert clone._dynamo_runtime is None
    assert clone.dp_openai_server_base_urls == ["http://10.0.0.1:3000/v1"]
    with pytest.raises(RuntimeError, match="copy .* owns no engines"):
        clone.engine_python()
    # The copy owns nothing: shutdown stops no engine and no service.
    assert clone.shutdown() is True
    assert env.log == []

    waits: list[dict[str, Any]] = []

    def fake_wait(**kwargs):
        kwargs.pop("check_alive")()  # the wait probes the engines each poll
        waits.append(kwargs)

    monkeypatch.setattr(generation_module, "wait_for_engine_registrations", fake_wait)
    clone = pickle.loads(pickle.dumps(generation))
    clone.resume_after_refit()
    # The copy starts the endpoints itself, through the workers' handles.
    assert [e.name for e in env.endpoints] == _IDS
    assert waits == [
        {
            "frontend_url": "http://10.0.0.1:3000/v1",
            "namespace": "ns",
            "expected_components": {"prefill": 1, "backend": 2},
            "model_name": "Qwen/Qwen3-0.6B",
            "timeout_s": 600.0,
        }
    ]
    # A dead engine fails the readiness wait instead of running out the timeout.
    clone._dynamo_fleet_ready = False
    env.engine("d0").alive = False
    with pytest.raises(RuntimeError, match="Megatron engines died: decode-0"):
        clone.resume_after_refit()
    # So does an engine whose endpoint stopped serving.
    env.engine("d0").alive = True
    env.engine("p0").endpoint_error = "exited"
    with pytest.raises(RuntimeError, match="stopped serving: prefill-0: exited"):
        clone.resume_after_refit()


# ---------------------------------------------------------------------------
# The refit side: fan-out, rank placement, ordered pause, one-wave resume
# ---------------------------------------------------------------------------


# Every engine resumes in one wave, in record order: the kv_cache refit phase
# already woke them all at once, and persist-mode engines need no role order.
_RESUMED = [f"{name}.resume" for name in _NAMES]


def test_records_hold_only_worker_handles_never_the_generation(env):
    # A copy holding the MegatronGeneration would shut the engine down when it
    # is garbage collected; the refit path keeps only the workers' handles.
    generation = env.started()
    for record, engine in zip(generation.engine_records, env.engines, strict=True):
        assert isinstance(record, MegatronDynamoEngine)
        workers = engine.worker_group.workers
        assert record.workers == tuple(workers)
        assert record.leader is workers[0]
    assert set(MegatronDynamoEngine.__dataclass_fields__) == {
        "role",
        "world_size",
        "instance_id",
        "node_ip",
        "workers",
        "leader",
    }


def test_pickled_copy_drives_the_engines_but_owns_none(env):
    generation = env.started()

    clone = pickle.loads(pickle.dumps(generation))

    assert clone.is_client_copy and not generation.is_client_copy
    # The records travel: plain fields as they were, the workers as handles.
    assert [(r.role, r.instance_id, r.node_ip) for r in clone.engine_records] == [
        (r.role, r.instance_id, r.node_ip) for r in generation.engine_records
    ]
    assert [[(w.short, w.rank) for w in r.workers] for r in clone.engine_records] == [
        [(w.short, w.rank) for w in r.workers] for r in generation.engine_records
    ]
    assert clone.shard_widths == (2, 2, 2)
    assert clone._engines == [] and clone._clusters == [] and clone._tokenizer is None
    # The copy owns nothing: shutdown stops no engine.
    assert clone.shutdown() is True
    assert all(engine.shutdown_calls == 0 for engine in env.engines)
    assert env.log == []
    # The copy still drives the engines through the handles it kept.
    clone = pickle.loads(pickle.dumps(generation))
    clone._dynamo_fleet_ready = True  # the readiness wait is covered below
    clone.suspend_for_refit()
    clone.resume_after_refit()
    assert (
        env.log
        == [
            "p0.pause",
            "p0.pinned",
            "p0.suspend",
            "d0.suspend",
            "d1.suspend",
        ]
        + _RESUMED
    )
    # The leader's liveness is read through the same handles.
    assert clone.shard_liveness_ref(1).value is True


@pytest.mark.parametrize(
    ("prefill_tp", "world_size", "prefixes"),
    [
        pytest.param(2, 14, (0, 2, 4), id="uniform"),
        # A prefill engine at TP 4 next to two decode engines at TP 2: each
        # engine is offset by the widths of the engines before it.
        pytest.param(4, 16, (0, 4, 6), id="mixed"),
    ],
)
def test_collective_fans_out_in_record_order_with_rank_prefixes(
    env, prefill_tp, world_size, prefixes
):
    generation = env.started(
        generation_config=_with_layout({"tensor_model_parallel_size": prefill_tp}),
        world_size=prefill_tp + 4,
    )
    shard_widths = (prefill_tp, 2, 2)
    assert generation.shard_widths == shard_widths
    assert generation.inference_world_size == sum(shard_widths)
    full = generation.full_fleet_membership(train_world_size=8)
    assert full.shard_prefixes == dict(enumerate(prefixes))
    assert full.workers_per_shard == shard_widths
    generation.check_full_fleet(full)
    # The full fleet is the one membership set_refit_membership accepts.
    generation.set_refit_membership(full)

    assert generation.prepare_refit_info({"w": ((1,), "float32")}) is None
    futures = generation.init_collective(
        "10.0.0.9", 1234, world_size, train_world_size=8
    )
    futures += generation.update_weights_from_collective(5.0)
    generation.set_rollout_weight_version(3)
    # One future per rank of every engine, for the init and for the update.
    assert len(futures) == 2 * sum(shard_widths)
    # Each engine is offset to the trainers plus the widths of the engines before it.
    assert env.log == (
        [f"{name}.prepare" for name in _NAMES]
        + [
            f"{name}.init(ws={world_size},train=8,off={8 + prefix})"
            for name, prefix in zip(_NAMES, prefixes, strict=True)
        ]
        + [f"{name}.update" for name in _NAMES]
        + [f"{name}.set_rollout_weight_version(3)" for name in _NAMES]
    )
    # prepare_refit_info is bounded by the control timeout.
    timeouts = env.refit_timeouts()
    assert timeouts and all(t <= _CONTROL_TIMEOUT_S for t in timeouts)
    with pytest.raises(ValueError, match=f"does not match expected {world_size}"):
        generation.init_collective("10.0.0.9", 1234, world_size - 1, train_world_size=8)
    with pytest.raises(ValueError, match="must not be None"):
        generation.prepare_refit_info(None)


def test_refit_sequence_suspends_prefill_pins_around_decode(env):
    """The Megatron weight synchronizer's sequence, phase by phase."""
    generation = env.started(pinned=[3, 1, 0])
    log = env.log
    generation.suspend_for_refit()
    assert log == [
        "p0.pause",
        # Prefill is suspended only after its pins drained, which takes three
        # reads; decode keeps serving meanwhile and only answers the liveness
        # probe after each nonzero read, which the log does not record.
        "p0.pinned",
        "p0.pinned",
        "p0.pinned",
        "p0.suspend",
        "d0.suspend",
        "d1.suspend",
    ]
    assert env.engine("d0").calls.count("is_alive") == 2
    log.clear()
    generation.prepare_for_generation(tags=["weights"])
    generation.prepare_for_generation(tags=["kv_cache"])
    assert log == [
        f"{name}.prepare_for_generation({tags!r})"
        for tags in (["weights"], ["kv_cache"])
        for name in _NAMES
    ]
    log.clear()
    generation.resume_after_refit()
    assert log[: len(_NAMES)] == _RESUMED
    # Every wait on an engine is bounded by the control timeout.
    assert all(t <= _CONTROL_TIMEOUT_S for t in env.refit_timeouts())


class _Clock:
    """A fake ``time`` for the module under test: sleeping advances it."""

    def __init__(self):
        self.now = 1000.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def test_stuck_pins_time_out_after_bounded_polls_and_unpause_prefill(
    env, monkeypatch, capsys
):
    generation = env.started(pinned=[2])
    generation._idle_poll_interval_s = 0.2
    generation._pin_report_interval_s = 0.3
    clock = _Clock()
    monkeypatch.setattr(generation_module, "time", clock)

    with pytest.raises(PinnedHandoffTimeout, match="'prefill-0': 2"):
        generation.suspend_for_refit(timeout_s=0.5)

    # Polls at t=0, 0.2 and 0.4; the last sleep is cut to what is left, and the
    # spent deadline is reported before a fourth read is issued.
    assert clock.sleeps == [0.2, 0.2, pytest.approx(0.1)]
    # Prefill was paused, then handed back to serving; nothing was suspended.
    assert env.log == ["p0.pause"] + ["p0.pinned"] * 3 + ["p0.resume"]
    # The counts are printed on the first nonzero read (t=0), then once the
    # report interval has passed (t=0.4), not on every poll.
    assert capsys.readouterr().out.count("{'prefill-0': 2}") == 2


def test_a_dead_decode_engine_fails_the_pin_wait_as_itself(env, monkeypatch):
    """Only decode releases a pin it claimed, so every poll probes the fleet."""
    generation = env.started(pinned=[2])
    generation._idle_poll_interval_s = 0.2
    monkeypatch.setattr(generation_module, "time", _Clock())
    env.engine("d0").actor_dead = True

    with pytest.raises(RuntimeError, match="died: decode-0") as raised:
        generation.suspend_for_refit(timeout_s=0.5)

    assert isinstance(raised.value.__cause__, ray.exceptions.RayActorError)
    # Found on the first poll, not when the deadline ran out.
    assert env.log == ["p0.pause", "p0.pinned", "p0.resume"]


def test_a_pin_read_cut_off_after_another_engine_reported_pins_times_out(env2):
    """The deadline ending inside a read still reports the pins known so far."""
    env2.pinned = {"prefill-0": [2]}
    generation = env2.generation(_two_prefill_config(), world_size=8)
    env2.engine("prefill-1").faults["pinned_handoff_block_count"] = "hang"
    env2.log.clear()

    with pytest.raises(PinnedHandoffTimeout, match="'prefill-0': 2") as raised:
        generation.suspend_for_refit()

    assert isinstance(raised.value.__cause__, GetTimeoutError)
    assert env2.log == [
        "p0.pause",
        "prefill-1.pause",
        "p0.pinned",
        "prefill-1.pinned",
        "p0.resume",
        "prefill-1.resume",
    ]


@pytest.mark.parametrize(
    ("short", "flags", "error", "match"),
    [
        pytest.param(
            "d0",
            {"faults": {"dynamo_endpoint_error": "hang"}},
            PinnedHandoffTimeout,
            "'prefill-0': 2",
            id="probe_cut_off",
        ),
        pytest.param(
            "d1",
            {"endpoint_error": "RuntimeError('adapter failed')"},
            RuntimeError,
            "stopped serving: decode-1",
            id="endpoint_stopped",
        ),
    ],
)
def test_the_pin_wait_probe_checks_the_endpoints_once_they_serve(
    env, short, flags, error, match
):
    """A probe the deadline cuts off is the wait running out; a stopped endpoint is itself."""
    generation = env.started(pinned=[2])
    _start_endpoints(generation)
    vars(env.engine(short)).update(flags)
    env.log.clear()

    with pytest.raises(error, match=match) as raised:
        generation.suspend_for_refit()

    if error is PinnedHandoffTimeout:
        assert isinstance(raised.value.__cause__, GetTimeoutError)
    assert env.log == ["p0.pause", "p0.pinned", "p0.resume"]


@pytest.mark.parametrize(
    ("engine", "method", "restored"),
    [
        ("p0", "pause_for_refit", ["p0.resume"]),
        ("p0", "pinned_handoff_block_count", ["p0.resume"]),
        ("p0", "suspend_for_refit", ["p0.resume"]),
        ("d1", "suspend_for_refit", _RESUMED),
    ],
    ids=["prefill_pause", "pin_read", "prefill_suspend", "decode_suspend"],
)
def test_any_suspend_failure_resumes_every_engine_it_reached(
    env, engine, method, restored
):
    generation = env.started()
    env.engine(engine).faults[method] = ConnectionError("engine gone")
    with pytest.raises(ConnectionError, match="engine gone") as caught:
        generation.suspend_for_refit()
    instance_id = env.engine(engine).name
    assert f"failed on engine {instance_id}" in "".join(caught.value.__notes__)
    # Every engine the suspend reached is resumed, in record order, as after a
    # refit.
    assert [entry for entry in env.log if entry.endswith(".resume")] == restored


def test_a_hung_engine_call_times_out_and_the_engines_are_resumed(env):
    generation = env.started()
    env.engine("d0").faults["suspend_for_refit"] = "hang"
    with pytest.raises(GetTimeoutError) as caught:
        generation.suspend_for_refit(timeout_s=0.5)
    assert "suspend_for_refit failed on engine decode-0" in "".join(
        caught.value.__notes__
    )
    assert [entry for entry in env.log if entry.endswith(".resume")] == _RESUMED
    # The suspend waits share one deadline no longer than timeout_s; the
    # restore that follows gets a fresh control-timeout deadline.
    timeouts = env.refit_timeouts()
    suspend_waits = timeouts[: -len(_NAMES)]
    assert suspend_waits and all(0 < t < 0.5 for t in suspend_waits)


def test_failed_restore_is_reported_on_the_original_error(env):
    generation = env.started()
    env.engine("d0").faults["suspend_for_refit"] = ConnectionError("engine gone")
    env.engine("d0").faults["resume_after_refit"] = RuntimeError("still gone")
    with pytest.raises(ConnectionError) as caught:
        generation.suspend_for_refit()
    notes = "".join(caught.value.__notes__)
    assert "resuming engines after the failed suspend also failed" in notes
    assert "decode-0: resume_after_refit" in notes


def test_resume_attempts_every_engine_and_reports_all_failures(env):
    generation = env.started()
    env.engine("d0").faults["resume_after_refit"] = RuntimeError("stuck")
    with pytest.raises(RuntimeError, match="decode-0: resume_after_refit") as caught:
        generation.resume_after_refit()
    assert "decode-1" not in str(caught.value) and "prefill-0" not in str(caught.value)
    # The failure does not stop the other engines from resuming.
    assert [entry for entry in env.log if entry.endswith(".resume")] == _RESUMED


@pytest.mark.parametrize(
    ("prefill_tp", "membership", "match"),
    [
        pytest.param(
            2,
            dict(absent_shards=[1], dp_size=3, total_gen_workers=6),
            "fixed fleet",
            id="partial",
        ),
        # An evenly sharded membership would start the decode engines at the
        # wrong ranks next to a wider prefill engine.
        pytest.param(
            4,
            dict(absent_shards=[], dp_size=3, total_gen_workers=6),
            r"ranks \[4, 2, 2\]",
            id="uniform_prefixes_against_mixed_widths",
        ),
        # Same prefixes but a width that is not the engine's is refused too.
        pytest.param(
            4,
            dict(
                absent_shards=[], dp_size=3, total_gen_workers=8, shard_widths=(4, 3, 1)
            ),
            "fixed fleet",
            id="mismatched_widths",
        ),
    ],
)
def test_refit_membership_must_be_every_engine_in_order(
    env, prefill_tp, membership, match
):
    generation = env.started(
        generation_config=_with_layout({"tensor_model_parallel_size": prefill_tp}),
        world_size=prefill_tp + 4,
    )
    generation.check_full_fleet(generation.full_fleet_membership(train_world_size=8))
    with pytest.raises(NotImplementedError, match=match):
        generation.set_refit_membership(
            desired_membership(train_world_size=8, **membership)
        )


# ---------------------------------------------------------------------------
# Rollouts
# ---------------------------------------------------------------------------


def test_rollouts_are_not_served_yet(env):
    generation = env.started()
    data = BatchedDataDict(
        {"input_ids": torch.tensor([[1, 2, 3, 0]]), "input_lengths": torch.tensor([3])}
    )
    with pytest.raises(NotImplementedError, match="do not serve rollouts yet"):
        generation.generate(data)
    assert generation.get_inference_world_size() == 6

    async def collect():
        return [item async for item in generation.generate_async(data)]

    with pytest.raises(NotImplementedError, match="do not serve rollouts yet"):
        asyncio.run(collect())
    with pytest.raises(NotImplementedError, match="NeMo-Gym"):
        generation.setup_token_capture({}, "staging")  # type: ignore[arg-type]
