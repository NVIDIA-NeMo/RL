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

"""MegatronDynamoGeneration: the prefill/decode engines and their refit side."""

import asyncio
import copy
import pickle
from types import SimpleNamespace

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
        self.alive = True
        self.fail_shutdown = False
        self.unclean_shutdown = False
        self.shutdown_calls = 0
        # Refit fakes: rank zero's calls land in the log. ``faults`` maps a
        # worker method to an exception, or "hang".
        self.faults: dict[str, object] = {}
        (width,) = kwargs["cluster"].kwargs["bundle_ct_per_node_list"]
        workers = [_FakeWorker(self.short, rank) for rank in range(width)]
        self.worker_group = SimpleNamespace(
            workers=workers, get_dp_leader_worker_idx=lambda shard: 0
        )
        self.log.append(f"engine {self.name} up")

    def call(self, rank: int, method: str, kwargs: dict) -> _Ref:
        if method == "is_alive":
            return _Ref(self.alive)
        if rank == 0:
            self.log.append(self._entry(method, kwargs))
        fault = self.faults.get(method)
        if fault == "hang":
            return _Ref(hang=True)
        if fault is not None:
            return _Ref(error=fault)
        return _Ref(True)

    def _entry(self, method: str, kwargs: dict) -> str:
        short = {
            "suspend_for_refit": "suspend",
            "resume_after_refit": "resume",
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

    def shutdown(self):
        self.shutdown_calls += 1
        if self.fail_shutdown:
            raise RuntimeError(f"engine {self.name} shutdown failed")
        self.log.append(f"engine {self.name} down")
        return not self.unclean_shutdown


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
        self.fail_engine_at = None
        self.ray = _FakeRay()

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
        return MegatronDynamoGeneration(
            config=policy,  # type: ignore[arg-type]
            tokenizer="tok",  # type: ignore[arg-type]
            cluster=SimpleNamespace(**cluster),  # type: ignore[arg-type]
        )

    def started(self, **kwargs) -> MegatronDynamoGeneration:
        """A built generation object with the start-up log cleared."""
        generation = self.generation(**kwargs)
        self.log.clear()
        self.ray.get_timeouts.clear()
        return generation

    def engine(self, short: str) -> _FakeEngine:
        return self.engines_by_short[short]

    def refit_timeouts(self) -> list:
        """Timeouts of the gets the refit path made (liveness probes excluded)."""
        return [t for t in self.ray.get_timeouts if t is not None]


@pytest.fixture
def env(monkeypatch):
    return _Env(monkeypatch, _NODES)


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
    expected_log = []
    for name in _IDS:
        expected_log += [f"cluster {name} up", f"placement {name}", f"engine {name} up"]
    assert env.log == expected_log

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


def test_start_failure_tears_down_what_exists(env):
    # The third engine fails to come up.
    env.fail_engine_at = 2
    with pytest.raises(RuntimeError, match="failed to start"):
        env.generation()
    down = [line for line in env.log if line.endswith(" down")]
    assert set(down) == {
        "engine prefill-0 down",
        "engine decode-0 down",
        "cluster prefill-0 down",
        "cluster decode-0 down",
        "cluster decode-1 down",
    }


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
    assert env.log == []


_ENGINES_DOWN = [
    "engine prefill-0 down",
    "engine decode-0 down",
    "engine decode-1 down",
]
_CLUSTERS_DOWN = [
    "cluster prefill-0 down",
    "cluster decode-0 down",
    "cluster decode-1 down",
]


@pytest.mark.parametrize(
    ("faults", "expected_log", "errors"),
    [
        pytest.param({}, _ENGINES_DOWN + _CLUSTERS_DOWN, [], id="clean"),
        # A failed or unclean engine stop is logged; the teardown continues.
        pytest.param(
            {"d0": {"fail_shutdown": True}, "d1": {"unclean_shutdown": True}},
            ["engine prefill-0 down", "engine decode-1 down"] + _CLUSTERS_DOWN,
            ["Could not stop a Megatron engine", "did not shut down cleanly"],
            id="continues_past_failures",
        ),
    ],
)
def test_shutdown_stops_every_engine_then_its_cluster(
    env, caplog, faults, expected_log, errors
):
    generation = env.started()
    for short, flags in faults.items():
        vars(env.engine(short)).update(flags)

    with caplog.at_level("ERROR"):
        assert generation.shutdown() is True

    assert env.log == expected_log
    for error in errors:
        assert error in caplog.text
    assert generation.engine_records == [] and generation.num_engines == 0
    env.log.clear()
    assert generation.shutdown() is True
    assert env.log == []


# ---------------------------------------------------------------------------
# The refit side: fan-out, rank placement, suspend and resume
# ---------------------------------------------------------------------------


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
    clone.suspend_for_refit()
    clone.resume_after_refit()
    assert env.log == [f"{name}.suspend" for name in _NAMES] + [
        f"{name}.resume" for name in _NAMES
    ]
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


def test_refit_sequence_runs_every_phase_on_every_engine(env):
    """The Megatron weight synchronizer's sequence, phase by phase."""
    generation = env.started()
    log = env.log
    generation.suspend_for_refit()
    assert log == [f"{name}.suspend" for name in _NAMES]
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
    assert log == [f"{name}.resume" for name in _NAMES]
    # Every wait on an engine is bounded by the control timeout.
    assert all(t <= _CONTROL_TIMEOUT_S for t in env.refit_timeouts())


@pytest.mark.parametrize("engine", ["p0", "d1"], ids=["prefill", "decode"])
def test_a_failed_suspend_resumes_every_engine(env, engine):
    generation = env.started()
    env.engine(engine).faults["suspend_for_refit"] = ConnectionError("engine gone")
    with pytest.raises(ConnectionError, match="engine gone") as caught:
        generation.suspend_for_refit()
    instance_id = env.engine(engine).name
    assert f"failed on engine {instance_id}" in "".join(caught.value.__notes__)
    # The suspend was sent to every engine, so every engine is put back.
    assert [entry for entry in env.log if entry.endswith(".resume")] == [
        f"{name}.resume" for name in _NAMES
    ]


def test_a_hung_engine_call_times_out_and_the_engines_are_resumed(env):
    generation = env.started()
    env.engine("d0").faults["suspend_for_refit"] = "hang"
    with pytest.raises(GetTimeoutError) as caught:
        generation.suspend_for_refit(timeout_s=0.5)
    assert "suspend_for_refit failed on engine decode-0" in "".join(
        caught.value.__notes__
    )
    assert [entry for entry in env.log if entry.endswith(".resume")] == [
        f"{name}.resume" for name in _NAMES
    ]
    # The suspend waits share one deadline no longer than timeout_s; the
    # restore that follows gets a fresh control-timeout deadline.
    timeouts = env.refit_timeouts()
    suspend_waits = timeouts[: -len(_NAMES)]
    assert suspend_waits and all(t <= 0.5 for t in suspend_waits)


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
    assert [entry for entry in env.log if entry.endswith(".resume")] == [
        f"{name}.resume" for name in _NAMES
    ]


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

    async def collect():
        return [item async for item in generation.generate_async(data)]

    with pytest.raises(NotImplementedError, match="do not serve rollouts yet"):
        asyncio.run(collect())
    with pytest.raises(NotImplementedError, match="NeMo-Gym"):
        generation.setup_token_capture({}, "staging")  # type: ignore[arg-type]
