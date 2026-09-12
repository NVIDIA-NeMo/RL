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

import argparse
import json
import os
import re
import shlex
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Mapping

_DEVICE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
_INTEGER_FIELD_PATTERN = re.compile(r"^-?[0-9]+$")
_NCCL_IB_PLANE_VIRTUAL_BIT = 1 << 14
_NCCL_MAX_USER_DEFINED_PLANES = 12


@dataclass(frozen=True)
class RdmaDevice:
    """RDMA device state relevant to NCCL HCA selection."""

    name: str
    ports: tuple[int, ...]
    active_ports: tuple[int, ...]
    net_devices: tuple[str, ...] = ()
    pci_bdf: str | None = None


@dataclass(frozen=True)
class _HcaEntry:
    raw: str
    name: str
    port: int | None
    rail: int | None
    plane: int | None

    @property
    def selection_key(self) -> str:
        return f"{self.name}:{self.port}" if self.port is not None else self.name

    @property
    def topology_key(self) -> str:
        port = "" if self.port is None else str(self.port)
        rail = -1 if self.rail is None else self.rail
        plane = -1 if self.plane is None else self.plane
        return f"{self.name}:{port}:{rail}:{plane}"


@dataclass(frozen=True)
class TopologyDiagnostic:
    """Result of inspecting and optionally repairing NCCL HCA selection."""

    status: Literal["ok", "warning", "repaired", "error", "skipped"]
    replacement: str | None
    nccl_only: tuple[str, ...]
    candidate_only: tuple[str, ...]
    candidate_source: Literal["expected", "ucx"] | None
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


def _parse_optional_integer(field: str, *, value: str) -> int | None:
    if not field:
        return None
    if not _INTEGER_FIELD_PATTERN.fullmatch(field):
        raise ValueError(f"Invalid HCA device list: {value!r}")
    return int(field)


def _parse_device_list(value: str) -> tuple[_HcaEntry, ...]:
    normalized = value.removeprefix("=")
    tokens = tuple(part.strip() for part in normalized.split(",") if part.strip())
    if not tokens:
        raise ValueError(f"Invalid HCA device list: {value!r}")
    devices: list[_HcaEntry] = []
    for token in tokens:
        fields = token.split(":")
        if len(fields) > 4 or not _DEVICE_NAME_PATTERN.fullmatch(fields[0]):
            raise ValueError(f"Invalid HCA device list: {value!r}")
        optional_fields = fields[1:] + [""] * (4 - len(fields))
        devices.append(
            _HcaEntry(
                raw=token,
                name=fields[0],
                port=_parse_optional_integer(optional_fields[0], value=value),
                rail=_parse_optional_integer(optional_fields[1], value=value),
                plane=_parse_optional_integer(optional_fields[2], value=value),
            )
        )
    return tuple(devices)


def _device_validation_errors(
    devices: tuple[_HcaEntry, ...], inventory: Mapping[str, RdmaDevice]
) -> tuple[str, ...]:
    if len(devices) > 32:
        return (f"NCCL supports at most 32 HCA devices; got {len(devices)}",)
    errors: list[str] = []
    for entry in devices:
        device = inventory.get(entry.name)
        if device is None:
            errors.append(f"HCA {entry.name!r} does not exist")
            continue
        if entry.port is None and not device.active_ports:
            errors.append(f"HCA {entry.name!r} has no active port")
        if entry.port is not None and entry.port not in device.active_ports:
            errors.append(f"HCA {entry.name!r} port {entry.port} is not active")
        if (
            entry.plane is not None
            and entry.plane != -1
            and entry.plane & _NCCL_IB_PLANE_VIRTUAL_BIT
        ):
            errors.append(f"NCCL cannot use plane ID {entry.plane}")
    plane_ids = {
        entry.plane
        for entry in devices
        if entry.plane is not None and entry.plane != -1
    }
    if len(plane_ids) > _NCCL_MAX_USER_DEFINED_PLANES:
        errors.append(
            "NCCL supports at most "
            f"{_NCCL_MAX_USER_DEFINED_PLANES} user-defined plane IDs; "
            f"got {len(plane_ids)}"
        )
    return tuple(errors)


def read_rdma_inventory(*, sysfs_root: Path = Path("/sys")) -> dict[str, RdmaDevice]:
    """Read the node's RDMA device and port state from sysfs.

    Args:
        sysfs_root: Root of the sysfs tree. Tests can provide a fixture root.

    Returns:
        RDMA devices keyed by HCA name.
    """
    infiniband_root = sysfs_root / "class" / "infiniband"
    if not infiniband_root.is_dir():
        return {}

    inventory: dict[str, RdmaDevice] = {}
    for hca_path in sorted(infiniband_root.iterdir()):
        if not hca_path.is_dir():
            continue
        ports: list[int] = []
        active_ports: list[int] = []
        ports_root = hca_path / "ports"
        if ports_root.is_dir():
            for port_path in sorted(ports_root.iterdir(), key=lambda path: path.name):
                if not port_path.name.isdigit():
                    continue
                port = int(port_path.name)
                ports.append(port)
                try:
                    state = (port_path / "state").read_text().strip()
                except OSError:
                    state = ""
                state_code = state.partition(":")[0].strip()
                if state_code == "4":
                    active_ports.append(port)

        device_path = hca_path / "device"
        net_root = device_path / "net"
        net_devices = (
            tuple(sorted(path.name for path in net_root.iterdir()))
            if net_root.is_dir()
            else ()
        )
        pci_bdf = device_path.resolve().name if device_path.exists() else None
        inventory[hca_path.name] = RdmaDevice(
            name=hca_path.name,
            ports=tuple(ports),
            active_ports=tuple(active_ports),
            net_devices=net_devices,
            pci_bdf=pci_bdf,
        )
    return inventory


def diagnose_topology(
    *,
    action: str,
    nccl_ib_hca: str | None,
    ucx_net_devices: str | None,
    expected_hcas: str | None,
    repair_source: str | None,
    inventory: Mapping[str, RdmaDevice],
) -> TopologyDiagnostic:
    """Diagnose and optionally repair an NCCL HCA selection."""
    if action not in {"auto", "warn", "repair", "strict", "off"}:
        raise ValueError(f"Unsupported topology action: {action!r}")
    if action == "off":
        return TopologyDiagnostic(
            status="skipped",
            replacement=None,
            nccl_only=(),
            candidate_only=(),
            candidate_source=None,
        )
    if repair_source not in {None, "ucx"}:
        raise ValueError(f"Unsupported HCA repair source: {repair_source!r}")
    if repair_source == "ucx" and not ucx_net_devices:
        raise ValueError("NRL_NCCL_HCA_REPAIR_SOURCE=ucx requires UCX_NET_DEVICES")
    if (
        action == "auto"
        and not nccl_ib_hca
        and not expected_hcas
        and repair_source != "ucx"
    ):
        return TopologyDiagnostic(
            status="skipped",
            replacement=None,
            nccl_only=(),
            candidate_only=(),
            candidate_source=None,
        )
    current_is_exact = bool(nccl_ib_hca and nccl_ib_hca.startswith("="))
    if current_is_exact:
        assert nccl_ib_hca is not None
        current = _parse_device_list(nccl_ib_hca)
    else:
        current = ()
    if action == "strict":
        if not current_is_exact:
            return TopologyDiagnostic(
                status="error",
                replacement=None,
                nccl_only=(),
                candidate_only=(),
                candidate_source=None,
                errors=("NCCL_IB_HCA must be an exact include selector",),
            )
        current_errors = _device_validation_errors(current, inventory)
        if current_errors:
            return TopologyDiagnostic(
                status="error",
                replacement=None,
                nccl_only=(),
                candidate_only=(),
                candidate_source=None,
                errors=current_errors,
            )
    if expected_hcas:
        candidate_text = expected_hcas
        candidate_source: Literal["expected", "ucx"] = "expected"
    elif ucx_net_devices:
        candidate_text = ucx_net_devices
        candidate_source = "ucx"
    else:
        if action in {"auto", "warn"} and not nccl_ib_hca:
            return TopologyDiagnostic(
                status="skipped",
                replacement=None,
                nccl_only=(),
                candidate_only=(),
                candidate_source=None,
            )
        if action in {"auto", "warn"}:
            current_warnings = (
                _device_validation_errors(current, inventory)
                if current_is_exact
                else (
                    "NCCL_IB_HCA is not an exact include selector; "
                    "the selected device set was not validated",
                )
            )
            return TopologyDiagnostic(
                status="warning" if current_warnings else "ok",
                replacement=None,
                nccl_only=(),
                candidate_only=(),
                candidate_source=None,
                warnings=current_warnings,
            )
        if action == "strict":
            return TopologyDiagnostic(
                status="ok",
                replacement=None,
                nccl_only=(),
                candidate_only=(),
                candidate_source=None,
            )
        raise ValueError("An expected or UCX HCA list is required for comparison")

    candidate = _parse_device_list(candidate_text)
    if candidate_source == "expected":
        current_set = {entry.topology_key for entry in current}
        candidate_set = {entry.topology_key for entry in candidate}
    else:
        current_set = {entry.selection_key for entry in current}
        candidate_set = {entry.selection_key for entry in candidate}
    nccl_only = tuple(sorted(current_set - candidate_set))
    candidate_only = tuple(sorted(candidate_set - current_set))

    if action == "strict":
        candidate_errors = _device_validation_errors(candidate, inventory)
        if candidate_errors:
            return TopologyDiagnostic(
                status="error",
                replacement=None,
                nccl_only=nccl_only,
                candidate_only=candidate_only,
                candidate_source=candidate_source,
                errors=candidate_errors,
            )
        if nccl_only or candidate_only:
            source_description = (
                "trusted expected HCA list"
                if candidate_source == "expected"
                else "UCX HCA list"
            )
            return TopologyDiagnostic(
                status="error",
                replacement=None,
                nccl_only=nccl_only,
                candidate_only=candidate_only,
                candidate_source=candidate_source,
                errors=(f"NCCL_IB_HCA differs from the {source_description}",),
            )
        return TopologyDiagnostic(
            status="ok",
            replacement=None,
            nccl_only=(),
            candidate_only=(),
            candidate_source=candidate_source,
        )

    effective_action = action
    if action == "auto":
        has_trusted_candidate = candidate_source == "expected" or repair_source == "ucx"
        effective_action = "repair" if has_trusted_candidate else "warn"

    if effective_action == "warn":
        if nccl_ib_hca and not current_is_exact:
            return TopologyDiagnostic(
                status="warning",
                replacement=None,
                nccl_only=(),
                candidate_only=(),
                candidate_source=candidate_source,
                warnings=(
                    "NCCL_IB_HCA is not an exact include selector; "
                    "device sets were not compared",
                ),
            )
        mismatch_warnings: tuple[str, ...] = ()
        if nccl_only or candidate_only:
            source_description = (
                "trusted expected HCA list"
                if candidate_source == "expected"
                else "UCX HCA list"
            )
            mismatch_warnings = (f"NCCL_IB_HCA differs from the {source_description}",)
        current_warnings = (
            _device_validation_errors(current, inventory) if current_is_exact else ()
        )
        warnings = current_warnings + mismatch_warnings
        return TopologyDiagnostic(
            status="warning" if warnings else "ok",
            replacement=None,
            nccl_only=nccl_only,
            candidate_only=candidate_only,
            candidate_source=candidate_source,
            warnings=warnings,
        )
    if effective_action != "repair" or (
        candidate_source == "ucx" and repair_source != "ucx"
    ):
        raise ValueError("Repair requires NRL_NCCL_HCA_REPAIR_SOURCE=ucx")

    errors = _device_validation_errors(candidate, inventory)
    if errors:
        return TopologyDiagnostic(
            status="error",
            replacement=None,
            nccl_only=nccl_only,
            candidate_only=candidate_only,
            candidate_source=candidate_source,
            errors=errors,
        )
    if current_is_exact and not nccl_only and not candidate_only:
        current_errors = _device_validation_errors(current, inventory)
        if current_errors:
            return TopologyDiagnostic(
                status="error",
                replacement=None,
                nccl_only=(),
                candidate_only=(),
                candidate_source=candidate_source,
                errors=current_errors,
            )
        return TopologyDiagnostic(
            status="ok",
            replacement=None,
            nccl_only=(),
            candidate_only=(),
            candidate_source=candidate_source,
        )
    return TopologyDiagnostic(
        status="repaired",
        replacement="=" + ",".join(entry.raw for entry in candidate),
        nccl_only=nccl_only,
        candidate_only=candidate_only,
        candidate_source=candidate_source,
    )


def _write_text_atomically(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", prefix=f".{path.name}.", dir=path.parent, delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(content)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _build_report(
    *,
    node_rank: str,
    action: str,
    nccl_ib_hca: str | None,
    diagnostic: TopologyDiagnostic,
    inventory: Mapping[str, RdmaDevice],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "node_rank": node_rank,
        "action": action,
        "status": diagnostic.status,
        "original_nccl_ib_hca": nccl_ib_hca,
        "replacement": diagnostic.replacement,
        "candidate_source": diagnostic.candidate_source,
        "nccl_only": list(diagnostic.nccl_only),
        "candidate_only": list(diagnostic.candidate_only),
        "errors": list(diagnostic.errors),
        "warnings": list(diagnostic.warnings),
        "rdma_devices": [asdict(inventory[name]) for name in sorted(inventory)],
    }


def main(argv: list[str] | None = None) -> int:
    """Run NCCL topology diagnostics and emit a sourceable repair file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sysfs-root", type=Path, default=Path("/sys"))
    parser.add_argument("--node-rank", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    args = parser.parse_args(argv)

    action = os.environ.get("NRL_NCCL_TOPOLOGY_ACTION", "auto").lower()
    nccl_ib_hca = os.environ.get("NCCL_IB_HCA")
    expected_hcas = os.environ.get("NRL_NCCL_EXPECTED_HCAS")
    ucx_net_devices = os.environ.get("UCX_NET_DEVICES")
    repair_source = os.environ.get("NRL_NCCL_HCA_REPAIR_SOURCE")
    candidate_source: Literal["expected", "ucx"] | None = None
    if expected_hcas:
        candidate_source = "expected"
    elif ucx_net_devices:
        candidate_source = "ucx"

    inventory: dict[str, RdmaDevice] = {}
    inventory_required = not (
        action == "off"
        or (
            action == "auto"
            and not nccl_ib_hca
            and not expected_hcas
            and repair_source != "ucx"
        )
        or (repair_source == "ucx" and not ucx_net_devices)
    )
    try:
        if inventory_required:
            inventory = read_rdma_inventory(sysfs_root=args.sysfs_root)
    except OSError as error:
        diagnostic = TopologyDiagnostic(
            status="error",
            replacement=None,
            nccl_only=(),
            candidate_only=(),
            candidate_source=candidate_source,
            errors=(f"Unable to read RDMA inventory: {error}",),
        )
    else:
        try:
            diagnostic = diagnose_topology(
                action=action,
                nccl_ib_hca=nccl_ib_hca,
                ucx_net_devices=ucx_net_devices,
                expected_hcas=expected_hcas,
                repair_source=repair_source,
                inventory=inventory,
            )
        except ValueError as error:
            diagnostic = TopologyDiagnostic(
                status="error",
                replacement=None,
                nccl_only=(),
                candidate_only=(),
                candidate_source=candidate_source,
                errors=(str(error),),
            )
    report = _build_report(
        node_rank=args.node_rank,
        action=action,
        nccl_ib_hca=nccl_ib_hca,
        diagnostic=diagnostic,
        inventory=inventory,
    )
    _write_text_atomically(
        args.report, json.dumps(report, indent=2, sort_keys=True) + "\n"
    )

    if diagnostic.replacement is not None:
        export_line = f"export NCCL_IB_HCA={shlex.quote(diagnostic.replacement)}\n"
        _write_text_atomically(args.env_file, export_line)
        print(
            "[NRL NCCL topology][WARN] Replacing NCCL_IB_HCA with a "
            f"validated {diagnostic.candidate_source} device list."
        )
    else:
        _write_text_atomically(args.env_file, "")

    for warning in diagnostic.warnings:
        print(f"[NRL NCCL topology][WARN] {warning}")
    for error in diagnostic.errors:
        print(f"[NRL NCCL topology][ERROR] {error}")
    return 2 if diagnostic.status == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
