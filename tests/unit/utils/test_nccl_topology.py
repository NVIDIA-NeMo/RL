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

import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from nemo_rl.utils.nccl_topology import (
    RdmaDevice,
    _write_text_atomically,
    diagnose_topology,
    main,
    read_rdma_inventory,
)


def test_repair_uses_validated_ucx_devices_as_exact_nccl_allowlist() -> None:
    inventory = {
        "data0": RdmaDevice(name="data0", ports=(1,), active_ports=(1,)),
        "data1": RdmaDevice(name="data1", ports=(1,), active_ports=(1,)),
        "service0": RdmaDevice(name="service0", ports=(1,), active_ports=(1,)),
    }

    result = diagnose_topology(
        action="repair",
        nccl_ib_hca="=data0:1,service0:1",
        ucx_net_devices="data0:1,data1:1",
        expected_hcas=None,
        repair_source="ucx",
        inventory=inventory,
    )

    assert result.status == "repaired"
    assert result.replacement == "=data0:1,data1:1"
    assert result.nccl_only == ("service0:1",)
    assert result.candidate_only == ("data1:1",)


def test_warn_reports_mismatch_without_replacing_nccl_selection() -> None:
    inventory = {
        "data0": RdmaDevice(name="data0", ports=(1,), active_ports=(1,)),
        "data1": RdmaDevice(name="data1", ports=(1,), active_ports=(1,)),
        "service0": RdmaDevice(name="service0", ports=(1,), active_ports=(1,)),
    }

    result = diagnose_topology(
        action="warn",
        nccl_ib_hca="=data0:1,service0:1",
        ucx_net_devices="data0:1,data1:1",
        expected_hcas=None,
        repair_source=None,
        inventory=inventory,
    )

    assert result.status == "warning"
    assert result.replacement is None
    assert result.nccl_only == ("service0:1",)
    assert result.candidate_only == ("data1:1",)
    assert result.warnings
    assert "NCCL_IB_HCA differs" in result.warnings[0]


def test_repair_refuses_inactive_candidate_port() -> None:
    inventory = {
        "data0": RdmaDevice(name="data0", ports=(1,), active_ports=(1,)),
        "data1": RdmaDevice(name="data1", ports=(1,), active_ports=()),
    }

    result = diagnose_topology(
        action="repair",
        nccl_ib_hca="=data0:1",
        ucx_net_devices="data0:1,data1:1",
        expected_hcas=None,
        repair_source="ucx",
        inventory=inventory,
    )

    assert result.status == "error"
    assert result.replacement is None
    assert result.errors == ("HCA 'data1' port 1 is not active",)


def test_repair_rejects_more_than_nccl_hca_limit() -> None:
    inventory = {
        f"data{index}": RdmaDevice(name=f"data{index}", ports=(1,), active_ports=(1,))
        for index in range(33)
    }
    expected_hcas = "=" + ",".join(f"data{index}:1" for index in range(33))

    result = diagnose_topology(
        action="auto",
        nccl_ib_hca="=data0:1",
        ucx_net_devices=None,
        expected_hcas=expected_hcas,
        repair_source=None,
        inventory=inventory,
    )

    assert result.status == "error"
    assert result.replacement is None
    assert result.errors == ("NCCL supports at most 32 HCA devices; got 33",)


def test_repair_prefers_explicit_expected_hcas_over_ucx_devices() -> None:
    inventory = {
        "data0": RdmaDevice(name="data0", ports=(1,), active_ports=(1,)),
        "data1": RdmaDevice(name="data1", ports=(1,), active_ports=(1,)),
        "service0": RdmaDevice(name="service0", ports=(1,), active_ports=(1,)),
    }

    result = diagnose_topology(
        action="repair",
        nccl_ib_hca="=data0:1,service0:1",
        ucx_net_devices="data0:1,service0:1",
        expected_hcas="=data0:1,data1:1",
        repair_source=None,
        inventory=inventory,
    )

    assert result.status == "repaired"
    assert result.replacement == "=data0:1,data1:1"
    assert result.candidate_source == "expected"


def test_auto_repairs_when_cluster_provides_expected_hcas() -> None:
    inventory = {
        "data0": RdmaDevice(name="data0", ports=(1,), active_ports=(1,)),
        "data1": RdmaDevice(name="data1", ports=(1,), active_ports=(1,)),
        "service0": RdmaDevice(name="service0", ports=(1,), active_ports=(1,)),
    }

    result = diagnose_topology(
        action="auto",
        nccl_ib_hca="=data0:1,service0:1",
        ucx_net_devices="data0:1,service0:1",
        expected_hcas="=data0:1,data1:1",
        repair_source=None,
        inventory=inventory,
    )

    assert result.status == "repaired"
    assert result.replacement == "=data0:1,data1:1"


def test_auto_repairs_trusted_expected_rail_and_plane_mapping() -> None:
    inventory = {
        "data0": RdmaDevice(name="data0", ports=(1,), active_ports=(1,)),
        "data1": RdmaDevice(name="data1", ports=(1,), active_ports=(1,)),
    }

    result = diagnose_topology(
        action="auto",
        nccl_ib_hca="=data0:1:0:1,data1:1:0:0",
        ucx_net_devices=None,
        expected_hcas="=data0:1:0:0,data1:1:0:1",
        repair_source=None,
        inventory=inventory,
    )

    assert result.status == "repaired"
    assert result.replacement == "=data0:1:0:0,data1:1:0:1"


def test_auto_rejects_plane_id_with_nccl_reserved_bit() -> None:
    inventory = {
        "data0": RdmaDevice(name="data0", ports=(1,), active_ports=(1,)),
    }

    result = diagnose_topology(
        action="auto",
        nccl_ib_hca="=data0:1",
        ucx_net_devices=None,
        expected_hcas="=data0:1:0:16384",
        repair_source=None,
        inventory=inventory,
    )

    assert result.status == "error"
    assert result.replacement is None
    assert result.errors == ("NCCL cannot use plane ID 16384",)


def test_auto_rejects_invalid_current_plane_when_trusted_ucx_devices_match() -> None:
    inventory = {
        "data0": RdmaDevice(name="data0", ports=(1,), active_ports=(1,)),
    }

    result = diagnose_topology(
        action="auto",
        nccl_ib_hca="=data0:1:0:16384",
        ucx_net_devices="data0:1",
        expected_hcas=None,
        repair_source="ucx",
        inventory=inventory,
    )

    assert result.status == "error"
    assert result.replacement is None
    assert result.errors == ("NCCL cannot use plane ID 16384",)


def test_auto_rejects_more_than_twelve_user_defined_planes() -> None:
    inventory = {
        f"data{index}": RdmaDevice(name=f"data{index}", ports=(1,), active_ports=(1,))
        for index in range(13)
    }
    expected_hcas = "=" + ",".join(f"data{index}:1:0:{index}" for index in range(13))

    result = diagnose_topology(
        action="auto",
        nccl_ib_hca="=data0:1",
        ucx_net_devices=None,
        expected_hcas=expected_hcas,
        repair_source=None,
        inventory=inventory,
    )

    assert result.status == "error"
    assert result.replacement is None
    assert result.errors == ("NCCL supports at most 12 user-defined plane IDs; got 13",)


def test_auto_keeps_matching_exact_selection_unchanged() -> None:
    inventory = {
        "data0": RdmaDevice(name="data0", ports=(1,), active_ports=(1,)),
        "data1": RdmaDevice(name="data1", ports=(1,), active_ports=(1,)),
    }

    result = diagnose_topology(
        action="auto",
        nccl_ib_hca="=data0:1,data1:1",
        ucx_net_devices="data0:1,data1:1",
        expected_hcas="=data0:1,data1:1",
        repair_source=None,
        inventory=inventory,
    )

    assert result.status == "ok"
    assert result.replacement is None


def test_trusted_expected_comparison_normalizes_omitted_topology_fields() -> None:
    inventory = {
        "data0": RdmaDevice(name="data0", ports=(1,), active_ports=(1,)),
    }

    omitted_rail_and_plane = diagnose_topology(
        action="auto",
        nccl_ib_hca="=data0:1",
        ucx_net_devices=None,
        expected_hcas="=data0:1:-1:-1",
        repair_source=None,
        inventory=inventory,
    )
    omitted_plane = diagnose_topology(
        action="auto",
        nccl_ib_hca="=data0:1:0",
        ucx_net_devices=None,
        expected_hcas="=data0:1:0:-1",
        repair_source=None,
        inventory=inventory,
    )

    assert omitted_rail_and_plane.status == "ok"
    assert omitted_rail_and_plane.replacement is None
    assert omitted_plane.status == "ok"
    assert omitted_plane.replacement is None


def test_off_skips_without_requiring_network_configuration() -> None:
    result = diagnose_topology(
        action="off",
        nccl_ib_hca=None,
        ucx_net_devices=None,
        expected_hcas=None,
        repair_source=None,
        inventory={},
    )

    assert result.status == "skipped"
    assert result.replacement is None


def test_auto_skips_when_no_network_selection_is_configured() -> None:
    result = diagnose_topology(
        action="auto",
        nccl_ib_hca=None,
        ucx_net_devices=None,
        expected_hcas=None,
        repair_source=None,
        inventory={},
    )

    assert result.status == "skipped"
    assert result.replacement is None


def test_auto_skips_untrusted_ucx_list_when_nccl_selection_is_unset() -> None:
    inventory = {
        "data0": RdmaDevice(name="data0", ports=(1,), active_ports=(1,)),
    }

    result = diagnose_topology(
        action="auto",
        nccl_ib_hca=None,
        ucx_net_devices="data0:1",
        expected_hcas=None,
        repair_source=None,
        inventory=inventory,
    )

    assert result.status == "skipped"
    assert result.replacement is None


def test_auto_warns_when_current_hca_is_missing_and_no_repair_source_exists() -> None:
    result = diagnose_topology(
        action="auto",
        nccl_ib_hca="=missing0:1",
        ucx_net_devices=None,
        expected_hcas=None,
        repair_source=None,
        inventory={},
    )

    assert result.status == "warning"
    assert result.replacement is None
    assert result.warnings == ("HCA 'missing0' does not exist",)


def test_warn_reports_missing_current_hca_when_ucx_candidate_matches() -> None:
    result = diagnose_topology(
        action="warn",
        nccl_ib_hca="=missing0:1",
        ucx_net_devices="missing0:1",
        expected_hcas=None,
        repair_source=None,
        inventory={},
    )

    assert result.status == "warning"
    assert result.replacement is None
    assert result.warnings == ("HCA 'missing0' does not exist",)


def test_auto_reports_inactive_current_port_when_ucx_candidate_matches() -> None:
    inventory = {
        "data0": RdmaDevice(name="data0", ports=(1,), active_ports=()),
    }

    result = diagnose_topology(
        action="auto",
        nccl_ib_hca="=data0:1",
        ucx_net_devices="data0:1",
        expected_hcas=None,
        repair_source=None,
        inventory=inventory,
    )

    assert result.status == "warning"
    assert result.replacement is None
    assert result.warnings == ("HCA 'data0' port 1 is not active",)


def test_strict_rejects_missing_hca_in_current_exact_selection() -> None:
    inventory = {
        "data0": RdmaDevice(name="data0", ports=(1,), active_ports=(1,)),
    }

    result = diagnose_topology(
        action="strict",
        nccl_ib_hca="=data0:1,missing0:1",
        ucx_net_devices=None,
        expected_hcas=None,
        repair_source=None,
        inventory=inventory,
    )

    assert result.status == "error"
    assert result.replacement is None
    assert result.errors == ("HCA 'missing0' does not exist",)


def test_strict_rejects_non_exact_selector_without_comparison_source() -> None:
    inventory = {
        "data0": RdmaDevice(name="data0", ports=(1,), active_ports=(1,)),
    }

    result = diagnose_topology(
        action="strict",
        nccl_ib_hca="data",
        ucx_net_devices=None,
        expected_hcas=None,
        repair_source=None,
        inventory=inventory,
    )

    assert result.status == "error"
    assert result.replacement is None
    assert result.errors == ("NCCL_IB_HCA must be an exact include selector",)


@pytest.mark.parametrize(
    ("repair_source", "expected_error"),
    [
        ("ucx", "NRL_NCCL_HCA_REPAIR_SOURCE=ucx requires UCX_NET_DEVICES"),
        ("ucxx", "Unsupported HCA repair source: 'ucxx'"),
    ],
)
def test_auto_rejects_invalid_or_missing_ucx_repair_source_configuration(
    repair_source: str,
    expected_error: str,
) -> None:
    inventory = {
        "data0": RdmaDevice(name="data0", ports=(1,), active_ports=(1,)),
    }

    with pytest.raises(ValueError) as error:
        diagnose_topology(
            action="auto",
            nccl_ib_hca="=data0:1",
            ucx_net_devices=None,
            expected_hcas=None,
            repair_source=repair_source,
            inventory=inventory,
        )
    assert str(error.value) == expected_error


@pytest.mark.parametrize("action", ["auto", "warn", "repair", "strict"])
def test_enabled_actions_require_declared_ucx_source_to_be_present(
    action: str,
) -> None:
    inventory = {
        "data0": RdmaDevice(name="data0", ports=(1,), active_ports=(1,)),
    }

    with pytest.raises(
        ValueError,
        match="NRL_NCCL_HCA_REPAIR_SOURCE=ucx requires UCX_NET_DEVICES",
    ):
        diagnose_topology(
            action=action,
            nccl_ib_hca="=data0:1",
            ucx_net_devices=None,
            expected_hcas="=data0:1",
            repair_source="ucx",
            inventory=inventory,
        )


def test_strict_rejects_mismatch_with_trusted_expected_hcas() -> None:
    inventory = {
        "data0": RdmaDevice(name="data0", ports=(1,), active_ports=(1,)),
        "data1": RdmaDevice(name="data1", ports=(1,), active_ports=(1,)),
        "service0": RdmaDevice(name="service0", ports=(1,), active_ports=(1,)),
    }

    result = diagnose_topology(
        action="strict",
        nccl_ib_hca="=data0:1,service0:1",
        ucx_net_devices=None,
        expected_hcas="=data0:1,data1:1",
        repair_source=None,
        inventory=inventory,
    )

    assert result.status == "error"
    assert result.replacement is None
    assert result.errors == ("NCCL_IB_HCA differs from the trusted expected HCA list",)


def test_strict_rejects_trusted_expected_rail_and_plane_mismatch() -> None:
    inventory = {
        "data0": RdmaDevice(name="data0", ports=(1,), active_ports=(1,)),
        "data1": RdmaDevice(name="data1", ports=(1,), active_ports=(1,)),
    }

    result = diagnose_topology(
        action="strict",
        nccl_ib_hca="=data0:1:0:1,data1:1:0:0",
        ucx_net_devices=None,
        expected_hcas="=data0:1:0:0,data1:1:0:1",
        repair_source=None,
        inventory=inventory,
    )

    assert result.status == "error"
    assert result.replacement is None
    assert result.errors == ("NCCL_IB_HCA differs from the trusted expected HCA list",)


def test_warn_does_not_treat_prefix_selector_as_exact_device_set() -> None:
    inventory = {
        "data0": RdmaDevice(name="data0", ports=(1,), active_ports=(1,)),
        "data1": RdmaDevice(name="data1", ports=(1,), active_ports=(1,)),
    }

    result = diagnose_topology(
        action="warn",
        nccl_ib_hca="data",
        ucx_net_devices="data0:1,data1:1",
        expected_hcas=None,
        repair_source=None,
        inventory=inventory,
    )

    assert result.status == "warning"
    assert result.nccl_only == ()
    assert result.candidate_only == ()
    assert result.warnings == (
        "NCCL_IB_HCA is not an exact include selector; device sets were not compared",
    )


def test_warn_accepts_exact_nccl_rail_and_plane_assignments() -> None:
    inventory = {
        "data0": RdmaDevice(name="data0", ports=(1,), active_ports=(1,)),
        "data1": RdmaDevice(name="data1", ports=(1,), active_ports=(1,)),
    }

    result = diagnose_topology(
        action="warn",
        nccl_ib_hca="=data0:1:0:0,data1::0:1",
        ucx_net_devices="data0:1,data1",
        expected_hcas=None,
        repair_source=None,
        inventory=inventory,
    )

    assert result.status == "ok"
    assert result.nccl_only == ()
    assert result.candidate_only == ()


def test_read_rdma_inventory_collects_ports_netdevs_and_pci_bdf(
    tmp_path: Path,
) -> None:
    device_target = tmp_path / "devices" / "0000-af-00.0"
    (device_target / "net" / "rdma0").mkdir(parents=True)
    hca_root = tmp_path / "class" / "infiniband" / "data0"
    (hca_root / "ports" / "1").mkdir(parents=True)
    (hca_root / "ports" / "2").mkdir(parents=True)
    (hca_root / "ports" / "1" / "state").write_text("4: ACTIVE\n")
    (hca_root / "ports" / "2" / "state").write_text("1: INACTIVE\n")
    (hca_root / "device").symlink_to(device_target, target_is_directory=True)

    inventory = read_rdma_inventory(sysfs_root=tmp_path)

    assert inventory == {
        "data0": RdmaDevice(
            name="data0",
            ports=(1, 2),
            active_ports=(1,),
            net_devices=("rdma0",),
            pci_bdf="0000-af-00.0",
        )
    }


def test_atomic_writes_do_not_share_temporary_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "report.json"
    barrier = threading.Barrier(2)
    original_replace = Path.replace

    def synchronized_replace(source: Path, target: Path) -> Path:
        barrier.wait(timeout=5)
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", synchronized_replace)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(_write_text_atomically, destination, content)
            for content in ("first", "second")
        ]
        for future in futures:
            future.result()

    assert destination.read_text() in {"first", "second"}


def test_cli_default_auto_writes_sourceable_repair_and_json_report(
    tmp_path: Path,
) -> None:
    for name in ("data0", "data1", "service0"):
        port_root = tmp_path / "sys" / "class" / "infiniband" / name / "ports" / "1"
        port_root.mkdir(parents=True)
        (port_root / "state").write_text("4: ACTIVE\n")
    report_path = tmp_path / "report.json"
    env_path = tmp_path / "repair.env"
    environment = os.environ.copy()
    for variable in (
        "NRL_NCCL_TOPOLOGY_ACTION",
        "NRL_NCCL_EXPECTED_HCAS",
        "NRL_NCCL_HCA_REPAIR_SOURCE",
    ):
        environment.pop(variable, None)
    environment.update(
        {
            "NCCL_IB_HCA": "=data0:1,service0:1",
            "UCX_NET_DEVICES": "data0:1,service0:1",
            "NRL_NCCL_EXPECTED_HCAS": "=data0:1,data1:1",
        }
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "nemo_rl.utils.nccl_topology",
            "--sysfs-root",
            str(tmp_path / "sys"),
            "--node-rank",
            "2",
            "--report",
            str(report_path),
            "--env-file",
            str(env_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(report_path.read_text())
    assert report["node_rank"] == "2"
    assert report["status"] == "repaired"
    assert report["replacement"] == "=data0:1,data1:1"
    sourced = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; printf "%s" "$NCCL_IB_HCA"',
            "bash",
            str(env_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        env={**environment, "NCCL_IB_HCA": "=old0:1"},
    )
    assert sourced.stdout == "=data0:1,data1:1"


def test_cli_writes_error_report_for_invalid_expected_hca_selector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report_path = tmp_path / "report.json"
    env_path = tmp_path / "repair.env"
    monkeypatch.delenv("NRL_NCCL_TOPOLOGY_ACTION", raising=False)
    monkeypatch.delenv("UCX_NET_DEVICES", raising=False)
    monkeypatch.delenv("NRL_NCCL_HCA_REPAIR_SOURCE", raising=False)
    monkeypatch.setenv("NCCL_IB_HCA", "=data0:1")
    monkeypatch.setenv("NRL_NCCL_EXPECTED_HCAS", "=invalid device:1")

    exit_code = main(
        [
            "--sysfs-root",
            str(tmp_path / "sys"),
            "--node-rank",
            "3",
            "--report",
            str(report_path),
            "--env-file",
            str(env_path),
        ]
    )

    assert exit_code == 2
    assert (
        "[NRL NCCL topology][ERROR] Invalid HCA device list" in capsys.readouterr().out
    )
    report = json.loads(report_path.read_text())
    assert report["status"] == "error"
    assert report["errors"] == ["Invalid HCA device list: '=invalid device:1'"]
    assert env_path.read_text() == ""


@pytest.mark.parametrize("action", ["off", "auto"])
def test_cli_noop_actions_skip_unreadable_inventory(
    action: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report_path = tmp_path / "report.json"
    env_path = tmp_path / "repair.env"
    for variable in (
        "NCCL_IB_HCA",
        "NRL_NCCL_EXPECTED_HCAS",
        "NRL_NCCL_HCA_REPAIR_SOURCE",
        "UCX_NET_DEVICES",
    ):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("NRL_NCCL_TOPOLOGY_ACTION", action)

    def fail_inventory_read(*, sysfs_root: Path) -> dict[str, RdmaDevice]:
        raise PermissionError("RDMA sysfs is unavailable")

    monkeypatch.setattr(
        "nemo_rl.utils.nccl_topology.read_rdma_inventory", fail_inventory_read
    )

    exit_code = main(
        [
            "--sysfs-root",
            str(tmp_path / "sys"),
            "--node-rank",
            "5",
            "--report",
            str(report_path),
            "--env-file",
            str(env_path),
        ]
    )

    assert exit_code == 0
    report = json.loads(report_path.read_text())
    assert report["status"] == "skipped"
    assert report["rdma_devices"] == []
    assert env_path.read_text() == ""


def test_cli_writes_error_report_when_inventory_read_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report_path = tmp_path / "report.json"
    env_path = tmp_path / "repair.env"
    monkeypatch.setenv("NRL_NCCL_TOPOLOGY_ACTION", "strict")
    monkeypatch.setenv("NCCL_IB_HCA", "=data0:1")
    for variable in (
        "NRL_NCCL_EXPECTED_HCAS",
        "NRL_NCCL_HCA_REPAIR_SOURCE",
        "UCX_NET_DEVICES",
    ):
        monkeypatch.delenv(variable, raising=False)

    def fail_inventory_read(*, sysfs_root: Path) -> dict[str, RdmaDevice]:
        raise PermissionError("RDMA sysfs is unavailable")

    monkeypatch.setattr(
        "nemo_rl.utils.nccl_topology.read_rdma_inventory", fail_inventory_read
    )

    exit_code = main(
        [
            "--sysfs-root",
            str(tmp_path / "sys"),
            "--node-rank",
            "6",
            "--report",
            str(report_path),
            "--env-file",
            str(env_path),
        ]
    )

    assert exit_code == 2
    assert (
        "[NRL NCCL topology][ERROR] Unable to read RDMA inventory"
        in capsys.readouterr().out
    )
    report = json.loads(report_path.read_text())
    assert report["status"] == "error"
    assert report["errors"] == [
        "Unable to read RDMA inventory: RDMA sysfs is unavailable"
    ]
    assert report["rdma_devices"] == []
    assert env_path.read_text() == ""


def test_cli_auto_emits_warning_for_untrusted_ucx_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    for name in ("data0", "data1", "service0"):
        port_root = tmp_path / "sys" / "class" / "infiniband" / name / "ports" / "1"
        port_root.mkdir(parents=True)
        (port_root / "state").write_text("4: ACTIVE\n")
    report_path = tmp_path / "report.json"
    env_path = tmp_path / "repair.env"
    monkeypatch.delenv("NRL_NCCL_EXPECTED_HCAS", raising=False)
    monkeypatch.delenv("NRL_NCCL_HCA_REPAIR_SOURCE", raising=False)
    monkeypatch.setenv("NRL_NCCL_TOPOLOGY_ACTION", "auto")
    monkeypatch.setenv("NCCL_IB_HCA", "=data0:1,service0:1")
    monkeypatch.setenv("UCX_NET_DEVICES", "data0:1,data1:1")

    exit_code = main(
        [
            "--sysfs-root",
            str(tmp_path / "sys"),
            "--node-rank",
            "4",
            "--report",
            str(report_path),
            "--env-file",
            str(env_path),
        ]
    )

    assert exit_code == 0
    assert "[NRL NCCL topology][WARN]" in capsys.readouterr().out
    report = json.loads(report_path.read_text())
    assert report["status"] == "warning"
    assert report["warnings"]
