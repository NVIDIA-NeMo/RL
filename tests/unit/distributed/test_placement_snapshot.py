# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nemo_rl.distributed.placement_snapshot import build_placement_snapshot


def node(node_id, capacity, labels=None):
    return {
        "Alive": True,
        "NodeID": node_id,
        "NodeManagerHostname": f"host-{node_id}",
        "Resources": {"GPU": capacity},
        "Labels": labels or {},
    }


def actor(node_id, name, resources):
    return {
        "node_id": node_id,
        "name": name,
        "class_name": "Worker",
        "required_resources": resources,
    }


def test_head_keeps_physical_domain_and_cpu_actors_with_no_schedulable_gpus():
    rows = build_placement_snapshot(
        [
            node(
                "head",
                0,
                {
                    "nrl.nvidia.com/nvlink-domain": "fabric-A",
                    "nrl.nvidia.com/gpu-ids": "0.1.2.3",
                },
            )
        ],
        [actor("head", "Gym", {"CPU": 1})],
        {},
    )
    assert rows[0].domain == "fabric-A"
    assert rows[0].gpu_ids == ()
    assert rows[0].cpu_actors == ("Gym",)


def test_explicit_roles_disambiguate_same_class_and_noncontiguous_devices():
    rows = build_placement_snapshot(
        [node("1", 2, {"nrl.nvidia.com/gpu-ids": "0.2"})],
        [
            actor("1", "", {"GPU": 0.5}),
            actor("1", "", {"GPU_group_bundle": 0.5}),
            actor("1", "CPU-worker", {}),
        ],
        {"P": {"1": (0,)}, "R": {"1": (2,)}, "G": {"1": (0,)}},
    )
    assert rows[0].gpu_roles == {0: ("P", "G"), 2: ("R",)}
    assert rows[0].cpu_actors == ("CPU-worker",)


def test_missing_inventory_never_invents_unused_gpu_ids():
    rows = build_placement_snapshot([node("1", 8)], [], {"P": {"1": (3,)}})
    assert rows[0].gpu_ids == (3,)
    assert rows[0].advertised_gpus == 8


def test_stale_device_inventory_and_missing_hosts_fail_explicitly():
    with pytest.raises(ValueError, match="disagrees"):
        build_placement_snapshot(
            [node("1", 1, {"nrl.nvidia.com/gpu-ids": "0"})], [], {"P": {"1": (2,)}}
        )
    with pytest.raises(ValueError, match="missing Ray nodes"):
        build_placement_snapshot([], [actor("missing", "Gym", {})], {})


@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.parametrize("foreign", [False, True])
def test_cpu_inventory_exceeds_dashboard_limit_or_reports_failure(
    monkeypatch, capsys, failure, foreign
):
    from nemo_rl.distributed.placement_snapshot import print_actor_placement

    query = AsyncMock()
    if failure:
        query.side_effect = RuntimeError("GCS unavailable")
    else:
        query.return_value = {
            i: SimpleNamespace(
                node_id=b"\x01",
                job_id=b"\x02" if foreign else b"\x01",
                name="",
                class_name="Gym",
                required_resources={},
            )
            for i in range(10_001)
        }
    fake_ray = SimpleNamespace(
        _private=SimpleNamespace(
            worker=SimpleNamespace(
                global_worker=SimpleNamespace(
                    gcs_client=SimpleNamespace(async_get_all_actor_info=query)
                )
            )
        ),
        nodes=lambda: [node("01", 1, {"nrl.nvidia.com/gpu-ids": "0"})],
        get_runtime_context=lambda: SimpleNamespace(get_job_id=lambda: "01"),
    )
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    cluster = SimpleNamespace(gpu_placement={"01": (0,)})
    args = SimpleNamespace(
        train_cluster=cluster,
        inference_cluster=cluster,
        reference_handle=None,
        value_handle=None,
        teacher_worker_groups=None,
    )
    config = SimpleNamespace(loss_fn=SimpleNamespace(reference_policy_kl_penalty=0))
    print_actor_placement(args, config)
    output = capsys.readouterr().out
    assert "P+G" in output
    expected = "Gym [job=02] x10001" if foreign else "Gym x10001"
    assert ("inventory unavailable" if failure else expected) in output
    query.assert_awaited_once_with(actor_state_name="ALIVE", timeout=30)
