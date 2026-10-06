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
"""Socket detection from sysfs and socket-first placement (nemo_rl/data_plane/nixl/numa.py)."""

from __future__ import annotations

from pathlib import Path

from nemo_rl.data_plane.nixl import numa
from nemo_rl.data_plane.nixl.placement import make_placement


def _fake_nodes(root: Path, layout: dict[int, str]) -> str:
    for node, cpulist in layout.items():
        d = root / f"node{node}"
        d.mkdir(parents=True)
        (d / "cpulist").write_text(cpulist + "\n")
    return str(root)


def _fake_ib(root: Path, devs: dict[str, tuple[int, str]]) -> str:
    for name, (node, state) in devs.items():
        (root / name / "device").mkdir(parents=True)
        (root / name / "device" / "numa_node").write_text(f"{node}\n")
        (root / name / "ports" / "1").mkdir(parents=True)
        (root / name / "ports" / "1" / "state").write_text(state + "\n")
    return str(root)


def test_gb300_tray_two_sockets_gpu_memory_nodes_ignored(tmp_path):
    layout = {0: "0-71", 1: "72-143", **{n: "" for n in range(2, 34)}}
    nodes = numa.cpu_nodes(_fake_nodes(tmp_path / "node", layout))
    assert sorted(nodes) == [0, 1]
    assert len(nodes[0]) == 72 and max(nodes[1]) == 143


def test_single_socket_is_one_node(tmp_path):
    assert list(numa.cpu_nodes(_fake_nodes(tmp_path / "node", {0: "0-71", 1: ""}))) == [
        0
    ]


def test_cpulist_ranges_and_singles():
    assert numa._parse_cpulist("0-2,5,7-8") == {0, 1, 2, 5, 7, 8}


def test_rdma_nics_by_socket_active_and_allowed(tmp_path):
    ib = _fake_ib(
        tmp_path / "ib",
        {
            "mlx5_8": (0, "4: ACTIVE"),  # active but not allowed by UCX_NET_DEVICES
            "rdma_rail0": (0, "1: DOWN"),
            "rdma_vf_rail0": (0, "4: ACTIVE"),
            "rdma_vf_rail1": (0, "4: ACTIVE"),
            "rdma_vf_rail2": (1, "4: ACTIVE"),
            "rdma_vf_rail3": (1, "4: ACTIVE"),
        },
    )
    allowed = numa.allowed_from_env(
        {
            "UCX_NET_DEVICES": "rdma_vf_rail0:1,rdma_vf_rail1:1,rdma_vf_rail2:1,rdma_vf_rail3:1"
        }
    )
    assert numa.rdma_nics(0, allowed, ib) == ["rdma_vf_rail0", "rdma_vf_rail1"]
    assert numa.rdma_nics(1, allowed, ib) == ["rdma_vf_rail2", "rdma_vf_rail3"]
    assert "mlx5_8" in numa.rdma_nics(0, None, ib)
    assert numa.device_list_param(["a", "b"]) == "a, b"


def test_local_first_prefers_same_socket():
    units = [
        {"unit_id": 0, "node_id": "A", "numa": 0},
        {"unit_id": 1, "node_id": "A", "numa": 1},
        {"unit_id": 2, "node_id": "B", "numa": 0},
    ]
    for socket, first in ((0, 0), (1, 1)):
        order = make_placement("local_first", numa=socket).order(
            units, node_id="A", nbytes=1
        )
        assert [u["unit_id"] for u in order][:2] == [first, 1 - first]
        assert order[-1]["unit_id"] == 2
    order = make_placement("local_first", numa=None).order(units, node_id="A", nbytes=1)
    assert {u["unit_id"] for u in order[:2]} == {0, 1} and order[-1]["unit_id"] == 2
