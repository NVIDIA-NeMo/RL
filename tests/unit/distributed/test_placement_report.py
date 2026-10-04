# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


def test_cpu_actors_split_otherwise_identical_rows():
    from nemo_rl.distributed.placement_report import NodePlacement, render_placement

    rows = [
        NodePlacement(
            "a", "gb-01", "fabric-A", 1, (0, 1, 2, 3), {0: ("P", "G")}, ("Gym",)
        ),
        NodePlacement("b", "gb-02", "fabric-A", 2, (0, 1, 2, 3), {0: ("P", "G")}, ()),
    ]
    output = render_placement(rows)
    assert "gb-01" in output and "gb-02" in output
    assert "gb-[01-02]" not in output
    assert "P+G" in output and "Gym" in output


def test_zero_advertised_gpus_does_not_erase_head_domain():
    from nemo_rl.distributed.placement_report import NodePlacement, render_placement

    output = render_placement(
        [NodePlacement("head", "gb-head", "fabric-A", 1, (), {}, ("Controller",))]
    )
    assert "fabric-A" in output and "Controller" in output
    assert "GPU0" not in output
    head_row = next(line for line in output.splitlines() if "gb-head" in line)
    assert "fabric-A" in head_row and "unknown" not in head_row


def test_thousand_regular_hosts_compress_without_omitting_members():
    from nemo_rl.distributed.placement_report import NodePlacement, render_placement

    rows = [
        NodePlacement(
            str(i),
            f"h100-{i:04d}",
            "host-local",
            i,
            tuple(range(8)),
            {gpu: ("P", "R") for gpu in range(8)},
            (),
        )
        for i in range(1000)
    ]
    output = render_placement(rows)
    assert "h100-[0000-0999]" in output
    assert len(output.splitlines()) < 20
    assert "1000 hosts" in output and "8000 assigned GPUs" in output


def test_full_and_filtered_views_do_not_hide_irregular_hosts():
    from nemo_rl.distributed.placement_report import NodePlacement, render_placement

    rows = [
        NodePlacement("a", "worker-alice", "A", 2, (0, 2), {2: ("T1",)}, ()),
        NodePlacement("b", "worker-bob", "B", 1, (0, 1), {0: ("R",)}, ()),
    ]
    output = render_placement(rows, host_filter="worker-alice", full=True)
    assert "worker-alice" in output and "worker-bob" not in output
    assert "GPU2" in output and "T1" in output
    assert "1 assigned GPUs" in output


def test_teacher_legend_uses_only_supplied_metadata():
    from nemo_rl.distributed.placement_report import NodePlacement, render_placement

    output = render_placement(
        [NodePlacement("a", "worker-1", "A", 1, (0,), {0: ("T1",)}, ())],
        teachers={"T1": (("default_teacher",), "Qwen/Qwen3-1.7B")},
    )
    assert "default_teacher" in output and "Qwen/Qwen3-1.7B" in output
    assert "math" not in output


def test_irregular_fleet_has_bounded_default_and_explicit_full_view():
    from nemo_rl.distributed.placement_report import NodePlacement, render_placement

    rows = [
        NodePlacement(
            str(i), f"host-{i}", "host-local", i, (0,), {0: ("P",)}, (f"cpu-{i}",)
        )
        for i in range(1000)
    ]
    compact = render_placement(rows)
    assert len(compact.splitlines()) < 40
    assert "980 more layouts" in compact
    assert "1000 hosts" in compact
    assert len(render_placement(rows, full=True).splitlines()) > 1000
