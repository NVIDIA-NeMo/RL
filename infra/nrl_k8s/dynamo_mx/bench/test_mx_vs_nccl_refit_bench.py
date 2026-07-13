from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


MODULE_PATH = Path(__file__).with_name("mx_vs_nccl_refit_bench.py")
SPEC = importlib.util.spec_from_file_location("mx_vs_nccl_refit_bench", MODULE_PATH)
assert SPEC and SPEC.loader
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)

NATIVE_PATH = Path(__file__).with_name("native_nccl_refit_bench.py")
NATIVE_SPEC = importlib.util.spec_from_file_location("native_nccl_refit_bench", NATIVE_PATH)
assert NATIVE_SPEC and NATIVE_SPEC.loader
native = importlib.util.module_from_spec(NATIVE_SPEC)
NATIVE_SPEC.loader.exec_module(native)

PUBLISHER_PATH = Path(__file__).with_name("mx_hf_publisher_bench.py")
PUBLISHER_SPEC = importlib.util.spec_from_file_location(
    "mx_hf_publisher_bench", PUBLISHER_PATH
)
assert PUBLISHER_SPEC and PUBLISHER_SPEC.loader
publisher_bench = importlib.util.module_from_spec(PUBLISHER_SPEC)
PUBLISHER_SPEC.loader.exec_module(publisher_bench)

CANONICAL_STAGES = (
    "control_discovery",
    "source_preparation",
    "setup_registration",
    "transfer_planning",
    "wire_transfer",
    "receive_sync",
    "transformation",
    "installation",
    "post_install",
    "rollout_readiness",
)


def _args(backend: str) -> argparse.Namespace:
    return argparse.Namespace(
        worker_url="http://worker:9090",
        backend=backend,
        init_rpc="init_broadcaster",
        update_rpc="update_weights_from_distributed",
        init_kwargs='{"master_address": "sender"}',
        update_kwargs='{"names": ["weight"]}',
        mx_config='{"mx_server_url": "mx:8001"}',
        timeout=30.0,
        cycles=1,
        warmup_cycles=0,
        start_version=7,
        parse_logs="",
        bytes=1_000_000_000,
        publisher_trigger="",
        publisher_ack="",
        publisher_done="",
        coordination_timeout=30.0,
    )


@pytest.mark.parametrize(
    ("backend", "expected_routes"),
    [
        (
            "mx",
            [
                "pause_generation",
                "update_weights_via_mx",
                "flush_cache",
                "resume_generation",
            ],
        ),
        (
            "nccl",
            [
                "init_weights_update_group",
                "pause_generation",
                "update_weights_from_distributed",
                "resume_generation",
            ],
        ),
    ],
)
def test_http_uses_backend_specific_routes(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    expected_routes: list[str],
) -> None:
    calls: list[tuple[str, dict]] = []

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"status": "ok"}

    def post(url: str, *, json: dict, timeout: float) -> Response:
        assert timeout == 30.0
        calls.append((url.rsplit("/", 1)[-1], json))
        return Response()

    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(post=post))
    result = bench.run_http(_args(backend))

    assert [route for route, _ in calls] == expected_routes
    update_route = (
        "update_weights_via_mx"
        if backend == "mx"
        else "update_weights_from_distributed"
    )
    update_body = next(body for route, body in calls if route == update_route)
    if backend == "mx":
        assert update_body["version"] == 7
        assert update_body["mx_config"] == {"mx_server_url": "mx:8001"}
        assert "weight_version" not in update_body
    else:
        assert update_body["weight_version"] == "7"
        assert update_body["engine_rpc"] == "update_weights_from_distributed"
    assert result["schema_version"] == bench.SCHEMA_VERSION
    assert set(result["stages_s"]) == set(bench.STAGE_NAMES)


def test_parse_structured_mx_timing_and_reduce_ranks(tmp_path: Path) -> None:
    log = tmp_path / "worker.log"
    log.write_text(
        "noise\n"
        'INFO MX_REFIT_TIMING {"version": 9, "rank": 0, "stages": '
        '{"wire_transfer": {"duration_ms": 1000.0, "status": "ok"}, '
        '"installation": {"duration_ms": 2000.0, "status": "ok"}}}\n'
        'INFO MX_REFIT_TIMING {"version": 9, "rank": 1, "stages": '
        '{"wire": {"duration_s": 1.2}, "load": {"seconds": 1.8}}} trailing\n'
        "MX_REFIT_TIMING not-json\n",
        encoding="utf-8",
    )

    records = bench._parse_structured_mx_logs(str(log))
    timing = bench._timing_for_cycle(records, cycle=0, version=9)

    assert len(records) == 2
    assert timing["wire_transfer"]["seconds"] == pytest.approx(1.2)
    assert timing["installation"]["seconds"] == pytest.approx(2.0)


def test_parse_structured_nccl_timing(tmp_path: Path) -> None:
    log = tmp_path / "worker.log"
    log.write_text(
        'INFO NCCL_REFIT_TIMING {"backend":"nccl","stages":'
        '{"wire_transfer":{"duration_ms":5000.0,"status":"combined"},'
        '"installation":{"duration_ms":750.0,"status":"measured"}}}\n',
        encoding="utf-8",
    )

    records = bench._parse_structured_mx_logs(str(log))

    assert len(records) == 1
    assert records[0]["marker"] == "NCCL_REFIT_TIMING"
    assert records[0]["stages"]["wire_transfer"]["seconds"] == pytest.approx(5.0)
    assert records[0]["stages"]["installation"]["seconds"] == pytest.approx(0.75)


def test_stage_aggregation_reports_all_statistics() -> None:
    cycles = []
    for cycle, value in enumerate((1.0, 2.0, 3.0, 4.0)):
        stages = {
            name: {"status": "available", "seconds": value}
            for name in bench.STAGE_NAMES
        }
        cycles.append(
            {
                "cycle": cycle,
                "stages": stages,
                "unattributed_seconds": value / 10,
            }
        )

    summary = bench._summary(
        "mx",
        [10.0, 20.0, 30.0, 40.0],
        {},
        cycles=cycles,
        byte_count=1_000_000_000,
    )

    wire = summary["stages_s"]["wire_transfer"]
    assert wire == {
        "status": "available",
        "statuses": ["available"],
        "samples": 4,
        "min": 1.0,
        "median": 2.5,
        "p95": 4.0,
        "max": 4.0,
    }
    assert summary["unattributed_s"]["median"] == pytest.approx(0.25)
    assert summary["e2e_s"]["p95"] == 40.0


def test_csv_is_google_sheets_ready(tmp_path: Path) -> None:
    stages = {
        name: {
            "status": "available",
            "statuses": ["available"],
            "samples": 1,
            "min": 1.0,
            "median": 1.0,
            "p95": 1.0,
            "max": 1.0,
        }
        for name in bench.STAGE_NAMES
    }
    result = {
        "schema_version": bench.SCHEMA_VERSION,
        "backend": "mx",
        "cycles": 1,
        "bytes": 8_000_000_000,
        "gbps": {"median": 64.0},
        "e2e_s": {
            "samples": 1,
            "min": 10.0,
            "median": 10.0,
            "p95": 10.0,
            "max": 10.0,
        },
        "unattributed_s": {
            "samples": 1,
            "min": 0.5,
            "median": 0.5,
            "p95": 0.5,
            "max": 0.5,
        },
        "stages_s": stages,
    }
    output = tmp_path / "result.csv"

    bench._write_csv(result, output)

    lines = output.read_text(encoding="utf-8").splitlines()
    assert lines[0] == (
        "schema_version,backend,stage_number,stage,status,samples,"
        "min_s,median_s,p95_s,max_s,bytes,median_gbps"
    )
    assert len(lines) == 13  # header + ten stages + e2e + unattributed
    assert json.loads(json.dumps(bench._csv_rows(result)))[0]["stage"] == (
        "control_discovery"
    )


def test_compare_accepts_legacy_json(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    legacy = {
        "backend": "nccl",
        "cycles": 2,
        "e2e_s": {"min": 4.0, "median": 5.0, "p95": 6.0, "max": 6.0},
        "phases_s": {"wire": {"median": 3.0, "max": 3.2}},
    }
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")

    bench.compare([str(path)])

    output = capsys.readouterr().out
    assert "nccl" in output
    assert "5.000" in output
    assert "wire 3.000s" in output


def test_canonical_stage_names_match_across_harnesses() -> None:
    assert bench.STAGE_NAMES == CANONICAL_STAGES
    assert native.STAGE_NAMES == CANONICAL_STAGES
    assert publisher_bench.STAGE_NAMES == CANONICAL_STAGES


def test_native_warmup_records_are_excluded_from_aggregate() -> None:
    records = [
        {"version": 10, "excluded": True},
        {"version": 11, "excluded": False},
        {"version": 12, "excluded": False},
    ]

    assert [record["version"] for record in native._measured_cycles(records)] == [
        11,
        12,
    ]


def test_native_receiver_stage_reads_dynamo_milliseconds() -> None:
    response = {
        "timing": {
            "stages": {
                "wire_transfer": {
                    "status": "combined",
                    "route_phase": "receive_and_incremental_load",
                    "route_phases": ["receive_and_incremental_load"],
                    "duration_ms": 1250.0,
                    "combined_with": ["receive_sync", "installation"],
                    "reason": "not separable at the route boundary",
                }
            }
        }
    }

    stage = native._response_stage(response, "wire_transfer")

    assert stage == {
        "status": "combined",
        "seconds": 1.25,
        "source": "Dynamo vLLM receiver timing",
        "combined_group": "receiver_update",
        "combined_with": ["receive_sync", "installation"],
        "detail": (
            "route phases: receive_and_incremental_load; "
            "not separable at the route boundary"
        ),
    }


def test_native_combined_receiver_duration_is_counted_once() -> None:
    stages = {
        name: native._stage("unavailable") for name in native.STAGE_NAMES
    }
    for name in ("transfer_planning", "wire_transfer", "receive_sync"):
        stages[name] = native._stage(
            "combined",
            2.0,
            combined_group="receiver_update",
            combined_with=["installation"],
        )

    result = native._result_schema(
        role="controller",
        stages=stages,
        total_seconds=3.0,
        byte_count=1,
        rate_seconds=2.0,
    )

    assert result["unattributed_seconds"] == pytest.approx(1.0)


def test_native_receiver_manifest_uses_default_worker_and_nccl() -> None:
    manifest = (
        Path(__file__).parent
        / "configs"
        / "native_nccl_receiver_30b.gb200.yaml"
    ).read_text(encoding="utf-8")

    assert "- auto" in manifest
    assert "DYN_WEIGHT_TRANSFER_BACKEND, value: nccl" in manifest
    assert "--worker-cls" not in manifest
    assert "modelexpress.vllm_worker" not in manifest
    assert "DYN_MX_REFIT_ENABLED" not in manifest


def test_multi_cycle_trigger_names_are_versioned(tmp_path: Path) -> None:
    base = tmp_path / "trigger"
    expected = [tmp_path / "trigger.v20", tmp_path / "trigger.v21"]

    assert [native._version_path(base, version) for version in (20, 21)] == expected
    assert [bench._version_path(base, version) for version in (20, 21)] == expected
    assert [
        publisher_bench._version_path(base, version) for version in (20, 21)
    ] == expected


def test_mx_publisher_ready_coordination(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"status": "ok"}

    def post(url: str, *, json: dict, timeout: float) -> Response:
        calls.append(url.rsplit("/", 1)[-1])
        return Response()

    trigger = tmp_path / "publish"
    ack = tmp_path / "ready"
    for version in (7, 8):
        bench._version_path(ack, version).touch()
    args = _args("mx")
    args.cycles = 2
    args.publisher_trigger = str(trigger)
    args.publisher_ack = str(ack)
    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(post=post))

    result = bench.run_http(args)

    assert bench._version_path(trigger, 7).exists()
    assert bench._version_path(trigger, 8).exists()
    assert calls.count("update_weights_via_mx") == 2
    assert [record["version"] for record in result["raw_cycles"]] == [7, 8]


def test_http_warmup_is_excluded(monkeypatch: pytest.MonkeyPatch) -> None:
    updates: list[int] = []

    class Response:
        def __init__(self, body: dict):
            self.body = body

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"status": "ok"}

    def post(url: str, *, json: dict, timeout: float) -> Response:
        if url.endswith("/update_weights_via_mx"):
            updates.append(json["version"])
        return Response(json)

    args = _args("mx")
    args.warmup_cycles = 1
    args.cycles = 2
    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(post=post))

    result = bench.run_http(args)

    assert updates == [7, 8, 9]
    assert result["cycles"] == 2
    assert [record["version"] for record in result["raw_cycles"]] == [8, 9]
