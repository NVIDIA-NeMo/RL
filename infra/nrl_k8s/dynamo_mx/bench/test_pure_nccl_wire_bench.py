from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest


BENCH_DIR = Path(__file__).parent


def _load(name: str):
    path = BENCH_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


wire = _load("pure_nccl_wire_bench")
native = _load("native_nccl_refit_bench")


def test_wire_statistics_and_critical_path_schema() -> None:
    result = wire.build_result(
        byte_count=1_000_000_000,
        warmups=2,
        rank_seconds={0: [1.0, 2.0, 3.0, 4.0], 1: [1.1, 1.9, 3.5, 3.9]},
        init_seconds={0: 5.0, 1: 6.0},
        allocation_seconds={0: 0.5, 1: 0.6},
        preload_seconds={0: 0.7, 1: 0.0},
    )

    assert result["wire_seconds"] == {
        "samples": 4,
        "min": 1.1,
        "median": 2.75,
        "p95": 4.0,
        "max": 4.0,
    }
    assert result["effective_gbps"]["median"] == pytest.approx(
        (8.0 / 2.0 + 8.0 / 3.5) / 2
    )
    assert result["communicator_init_seconds"] == {"0": 5.0, "1": 6.0}
    assert result["source_preload_seconds"] == 0.7
    assert result["raw_iterations"][1]["wire_seconds"] == 2.0


def test_wire_json_and_csv_outputs(tmp_path: Path) -> None:
    result = wire.build_result(
        byte_count=8_000_000_000,
        warmups=1,
        rank_seconds={0: [2.0], 1: [2.1]},
        init_seconds={0: 1.0, 1: 1.1},
        allocation_seconds={0: 0.1, 1: 0.2},
        preload_seconds={0: 0.3, 1: 0.0},
    )
    json_path = tmp_path / "wire.json"
    csv_path = tmp_path / "wire.csv"

    wire.write_outputs(result, str(json_path), str(csv_path))

    assert json.loads(json_path.read_text())["schema_version"] == wire.SCHEMA_VERSION
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["metric"] for row in rows] == ["wire_seconds", "effective_gbps"]
    assert rows[0]["p95"] == "2.1"


@pytest.mark.parametrize(
    "argv",
    [
        ["sender", "--result-json", "out.json"],
        [
            "sender",
            "--master-address",
            "rank0",
            "--bytes",
            "0",
            "--result-json",
            "out.json",
        ],
        [
            "receiver",
            "--master-address",
            "rank0",
            "--warmups",
            "-1",
            "--result-json",
            "out.json",
        ],
        [
            "receiver",
            "--master-address",
            "rank0",
            "--iterations",
            "0",
            "--result-json",
            "out.json",
        ],
    ],
)
def test_wire_argument_validation(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        wire.parse_args(argv)


def test_native_preconsolidated_ep4_to_tp1_is_explicit() -> None:
    args = native.parse_args(
        [
            "sender",
            "--master-address",
            "rank0",
            "--checkpoint",
            "weights.pt",
            "--manifest",
            "manifest.json",
            "--trigger",
            "trigger",
            "--result",
            "result.json",
            "--source-layout",
            "preconsolidated_transport_only",
            "--source-ep-size",
            "4",
            "--destination-tp-size",
            "1",
        ]
    )

    layout = native._source_layout_metadata(args)
    assert layout["declared_source_layout"] == "EP4"
    assert layout["destination_layout"] == "TP1"
    assert layout["actual_source_processes"] == 1
    assert layout["consolidation_included"] is False
    assert layout["true_ep_topology_match"] is False


def test_native_consolidated_e2e_has_unsupported_schema() -> None:
    args = native.parse_args(
        [
            "sender",
            "--master-address",
            "rank0",
            "--manifest",
            "manifest.json",
            "--trigger",
            "trigger",
            "--result",
            "result.json",
            "--source-layout",
            "consolidated_e2e",
            "--destination-tp-size",
            "1",
        ]
    )

    result = native._unsupported_source_layout_result(args)
    assert result["status"] == "unsupported"
    assert result["reason_code"] == "ep_shard_consolidation_not_implemented"
    assert result["source_layout"]["true_ep_topology_match"] is False
    assert result["source_layout"]["consolidation_included"] is False
    assert result["stages"]["source_preparation"]["status"] == "unsupported"
    assert result["stages"]["wire_transfer"]["status"] == "not_run"


def test_native_layout_size_validation() -> None:
    with pytest.raises(SystemExit):
        native.parse_args(
            [
                "controller",
                "--master-address",
                "rank0",
                "--manifest",
                "manifest.json",
                "--trigger",
                "trigger",
                "--result",
                "result.json",
                "--source-layout",
                "consolidated_e2e",
                "--source-ep-size",
                "0",
            ]
        )
