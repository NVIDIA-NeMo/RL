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
"""Storage-unit placement: cluster names -> one Ray node ID per unit."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from nemo_rl.data_plane import mooncake_storage_unit as msu

# Ray validates node IDs as 28-byte hex strings.
N0, N1, N2, N3, N4 = (str(i) * 56 for i in range(5))


def _cluster(*pg_nodes: list[str]) -> SimpleNamespace:
    """A cluster whose placement groups put their bundles on ``pg_nodes``."""
    pgs = [
        SimpleNamespace(id=SimpleNamespace(hex=lambda n=nodes: "pg-" + "-".join(n)))
        for nodes in pg_nodes
    ]
    return SimpleNamespace(get_placement_groups=lambda: pgs)


def _use_clusters(monkeypatch, by_name: dict) -> dict:
    """Serve ``by_name``'s placement groups from a fake placement-group table."""
    table = {
        pg.id.hex(): {"bundles_to_node_id": dict(enumerate(pg.id.hex()[3:].split("-")))}
        for cluster in by_name.values()
        for pg in cluster.get_placement_groups()
    }
    monkeypatch.setattr(msu, "placement_group_table", lambda: table)
    return by_name


@pytest.fixture
def clusters_by_name(monkeypatch):
    return _use_clusters(
        monkeypatch,
        {
            "train": _cluster([N0, N1]),
            "inference": _cluster([N2], [N3]),
            "teacher:math": _cluster([N4]),
        },
    )


@pytest.fixture
def started(monkeypatch) -> list[str]:
    """Fake MooncakeStorageUnit: records the node each unit is hard-pinned to."""
    nodes: list[str] = []

    def options(*, runtime_env, scheduling_strategy):
        assert scheduling_strategy.soft is False
        unit = SimpleNamespace(__ray_ready__=SimpleNamespace(remote=lambda: None))
        return SimpleNamespace(
            remote=lambda cfg: nodes.append(scheduling_strategy.node_id) or unit
        )

    monkeypatch.setattr(msu, "MooncakeStorageUnit", SimpleNamespace(options=options))
    monkeypatch.setattr(msu, "make_actor_runtime_env", lambda _cls: {})
    return nodes


@pytest.mark.parametrize(
    ("placement", "count", "expected"),
    [
        (["inference"], None, [N2, N3, N2, N3]),  # None: 2 per selected node
        (["train"], 3, [N0, N1, N0]),  # uneven count: round-robin
        (["inference", "teacher:math"], 3, [N2, N3, N4]),
        ("all", None, [N0, N1, N2, N3, N4] * 2),  # all includes teachers
    ],
)
def test_units_round_robin_over_the_named_clusters(
    clusters_by_name, placement, count, expected
) -> None:
    clusters = msu.select_clusters(placement, clusters_by_name)
    assert msu.storage_node_ids(clusters, count) == expected


def test_unknown_cluster_name_fails_with_the_valid_names(clusters_by_name) -> None:
    with pytest.raises(ValueError, match=r"\['trian'\] not in \['inference'"):
        msu.select_clusters(["trian"], clusters_by_name)


def test_colocated_names_share_nodes_once_and_warn(monkeypatch, caplog) -> None:
    """Colocated: one cluster under two names; excluding one excludes nothing."""
    shared = _cluster([N0, N1])
    by_name = _use_clusters(monkeypatch, {"train": shared, "inference": shared})

    assert msu.storage_node_ids(msu.select_clusters("all", by_name), None) == [
        N0,
        N1,
        N0,
        N1,
    ]
    with caplog.at_level("WARNING"):
        msu.select_clusters(["inference"], by_name)
    assert "also covers ['train']" in caplog.text


@pytest.mark.parametrize(
    "dp_config",
    [
        {"backend": "simple"},  # no simple block
        {"backend": "simple", "simple": {"num_storage_units": 4}},  # placement None
        {"backend": "mooncake_cpu"},  # absent block: segment size defaults to 0
        {"backend": "mooncake_cpu", "mooncake_cpu": {"storage_unit_segment_size": 0}},
    ],
)
def test_storage_units_are_off_by_default(dp_config) -> None:
    def boom():
        raise AssertionError("no cluster lookup when storage units are off")

    off = {"train": SimpleNamespace(get_placement_groups=boom)}
    assert msu.plan_storage_unit_nodes(dp_config, off) is None
    assert msu.start_storage_units(dp_config, None) == ()


def test_plan_uses_each_backends_count_and_placement(clusters_by_name) -> None:
    simple = {
        "backend": "simple",
        "simple": {"num_storage_units": 3, "storage_unit_placement": ["inference"]},
    }
    mooncake = {
        "backend": "mooncake_cpu",
        "mooncake_cpu": {
            "storage_unit_segment_size": 1 << 30,
            "storage_unit_placement": ["train"],
        },
    }
    assert msu.plan_storage_unit_nodes(simple, clusters_by_name) == [N2, N3, N2]
    assert msu.plan_storage_unit_nodes(mooncake, clusters_by_name) == [
        N0,
        N1,
        N0,
        N1,
    ]


def test_start_storage_units_hard_pins_one_unit_per_planned_node(
    monkeypatch, started
) -> None:
    monkeypatch.setattr(msu.ray, "get", lambda refs, timeout: list(refs))

    units = msu.start_storage_units({"backend": "mooncake_cpu"}, [N2, N3, N2])

    assert started == [N2, N3, N2]
    assert len(units) == 3


def test_unit_startup_failure_kills_started_units(monkeypatch, started) -> None:
    """A node with no free CPU must fail setup, not hang it."""
    import ray

    def timed_out(refs, timeout):
        raise ray.exceptions.GetTimeoutError("not scheduled")

    killed = []
    monkeypatch.setattr(msu.ray, "get", timed_out)
    monkeypatch.setattr(msu.ray, "kill", killed.append)

    with pytest.raises(RuntimeError, match="Each unit needs 1 free CPU"):
        msu.start_storage_units({"backend": "mooncake_cpu"}, [N2, N3])
    assert len(killed) == 2
