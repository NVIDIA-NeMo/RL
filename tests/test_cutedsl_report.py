# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import importlib.util
import json
import os
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


EXPERIMENT_DIR = Path(__file__).parents[1] / "experiments/cutedsl_qwen3_30ba3b_oci_1n4g"
RENDERER_PATH = EXPERIMENT_DIR / "render_cutedsl_report.py"
EVENTS_PATH = EXPERIMENT_DIR / "lib/events.sh"
FUNCTIONAL_PATH = EXPERIMENT_DIR / "run_cutedsl_functional.sbatch"
MATRIX_PATH = EXPERIMENT_DIR / "run_cutedsl_matrix.sbatch"
REQUIRED_PHASES = {
    "preflight",
    "image_hash",
    "runtime_bootstrap",
    "config_validation",
    "focused_tests",
    "gpu_smoke",
    "functional_grpo",
    "timing",
    "profile",
    "metrics_export",
    "complete",
}


class LinkCollector(HTMLParser):
    """Collect href values from generated HTML for local-target verification."""

    def __init__(self: "LinkCollector") -> None:
        """Initialize an empty href collection."""
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(
        self: "LinkCollector", tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        """Record each href on an anchor element."""
        if tag != "a":
            return
        for name, value in attrs:
            if name == "href" and value is not None:
                self.hrefs.append(value)


def assert_public_links_exist(public_dir: Path) -> None:
    """Assert every generated public href resolves inside the public tree."""
    public_root = public_dir.resolve()
    for html_path in sorted(public_dir.rglob("*.html")):
        collector = LinkCollector()
        collector.feed(html_path.read_text())
        for href in collector.hrefs:
            target = (html_path.parent / href).resolve()
            assert target.is_relative_to(public_root), (html_path, href)
            assert target.exists(), (html_path, href)


def load_renderer() -> ModuleType:
    """Load the standalone experiment renderer from its filesystem path."""
    spec = importlib.util.spec_from_file_location(
        "render_cutedsl_report", RENDERER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, value: Any) -> None:
    """Write a JSON fixture using the same newline convention as run artifacts."""
    path.write_text(json.dumps(value, indent=2) + "\n")


def render_fixture(
    tmp_path: Path,
    events: list[dict[str, Any]],
    *,
    status: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    slurm_output: str | None = None,
) -> str:
    """Render a run fixture and return the generated HTML."""
    run_dir = tmp_path / "run-123"
    run_dir.mkdir()
    (run_dir / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events)
    )
    write_json(
        run_dir / "status.json",
        status
        or {
            "run_id": "run-123",
            "job_id": "123",
            "exit_code": 1,
            "finished_at_utc": "2026-07-11T18:11:00Z",
        },
    )
    if slurm_output is not None:
        (run_dir / "slurm.out").write_text(slurm_output)
    write_json(
        run_dir / "metadata.json",
        metadata
        or {
            "run": {
                "run_id": "run-123",
                "cluster_profile": "pre_tyche",
                "recipe": "qwen3-30ba3b.yaml",
                "effective_config": {
                    "cluster": {"num_nodes": 1, "gpus_per_node": 4},
                    "policy": {
                        "megatron_cfg": {
                            "env_vars": {"NVTE_CUTEDSL_FUSED_GROUPED_MLP": "1"}
                        }
                    },
                },
            },
            "source": {"branch": "feature", "sha": "a" * 40},
            "image": {"path": "/images/nemo.sqsh", "sha256": "b" * 64},
            "slurm": {
                "account": "coreai_dlalgo_llm",
                "partition": "batch",
                "job_id": "123",
            },
        },
    )
    renderer = load_renderer()
    renderer.render_run(run_dir)
    return (run_dir / "report.html").read_text()


def failure_events() -> list[dict[str, Any]]:
    """Return an incident fixture with a completed diagnostic trail."""
    return [
        {
            "timestamp_utc": "2026-07-11T18:10:00Z",
            "cluster": "pre_tyche",
            "job_id": "123",
            "phase": "runtime_diagnostic",
            "status": "pass",
            "exit_code": 0,
            "message": "both uv environment variables resolve to /runtime/venv",
            "artifact": "runtime_env.log",
        },
        {
            "timestamp_utc": "2026-07-11T18:00:00Z",
            "cluster": "pre_tyche",
            "job_id": "123",
            "phase": "runtime_bootstrap",
            "status": "fail",
            "exit_code": 1,
            "message": "UV_PROJECT_ENVIRONMENT mismatch",
            "artifact": "slurm.out",
        },
        {
            "timestamp_utc": "2026-07-11T18:11:00Z",
            "cluster": "pre_tyche",
            "job_id": "123",
            "phase": "root_cause",
            "status": "resolved",
            "exit_code": 0,
            "message": "runtime environment diagnosis",
            "artifact": "runtime_env.log",
            "symptom": "UV_PROJECT_ENVIRONMENT mismatch",
            "evidence": "runtime_env.log shows divergent prefixes",
            "root_cause": "Pyxis inherited a stale environment path",
            "fix_commit": "abc1234",
            "verification_job": "456",
            "reproduction": "submit with stale inherited environment",
            "hypothesis": "Pyxis retained the submission environment",
            "tested_change": "override both runtime environment variables",
            "verification_evidence": "job 456 runtime_env.log and three updates",
        },
    ]


def test_report_renders_failure_root_cause_and_verification(tmp_path: Path) -> None:
    """A failed report preserves symptom through verification evidence."""
    html = render_fixture(tmp_path, failure_events())

    assert "UV_PROJECT_ENVIRONMENT mismatch" in html
    assert "runtime_env.log" in html
    assert "Root cause" in html
    assert "Pyxis inherited a stale environment path" in html
    assert "abc1234" in html
    assert "456" in html
    assert "submit with stale inherited environment" in html
    assert "Pyxis retained the submission environment" in html
    assert "override both runtime environment variables" in html
    assert "job 456 runtime_env.log and three updates" in html


def test_report_escapes_untrusted_values_and_sorts_events(tmp_path: Path) -> None:
    """Event fields are escaped and ordered by their UTC timestamps."""
    events = failure_events()
    events[0]["message"] = '<script>alert("late")</script>'
    events[0]["artifact"] = "javascript:alert(1)"
    html = render_fixture(tmp_path, events)

    assert "<script>alert" not in html
    assert 'href="javascript:' not in html
    assert "&lt;script&gt;alert(&quot;late&quot;)&lt;/script&gt;" in html
    assert html.index("UV_PROJECT_ENVIRONMENT mismatch") < html.index(
        "&lt;script&gt;alert(&quot;late&quot;)&lt;/script&gt;"
    )


def test_report_handles_success_and_missing_optional_artifacts(tmp_path: Path) -> None:
    """A successful minimal run renders without metric or profile artifacts."""
    html = render_fixture(
        tmp_path,
        [
            {
                "timestamp_utc": "2026-07-11T18:00:00Z",
                "cluster": "aws_dfw",
                "job_id": "123",
                "phase": "complete",
                "status": "pass",
                "exit_code": 0,
                "message": "functional gate complete",
                "artifact": None,
            }
        ],
        status={
            "run_id": "run-123",
            "job_id": "123",
            "exit_code": 0,
            "finished_at_utc": "2026-07-11T18:00:00Z",
        },
    )

    assert "PASS" in html
    assert "No timing summary recorded." in html
    assert "No Nsight evidence recorded." in html
    assert html.count("No Nsight evidence recorded.") == 1
    assert "No root-cause record required." in html


def test_report_bounds_error_excerpt_to_recent_lines(tmp_path: Path) -> None:
    """Only a bounded tail of a run log is included in the report."""
    run_dir = tmp_path / "run-123"
    run_dir.mkdir()
    (run_dir / "events.jsonl").write_text("")
    write_json(run_dir / "status.json", {"run_id": "run-123", "exit_code": 1})
    (run_dir / "slurm.out").write_text(
        "old-secret-like-value\n"
        + "".join(f"recent-{index:03d}\n" for index in range(300))
        + "Authorization: Bearer credential-that-must-not-render\n"
    )

    renderer = load_renderer()
    renderer.render_run(run_dir)
    html = (run_dir / "report.html").read_text()

    assert "old-secret-like-value" not in html
    assert "recent-299" in html
    assert "credential-that-must-not-render" not in html
    assert "Authorization: [REDACTED]" in html
    assert len(html) < 60_000


def test_renderer_outputs_metric_profile_and_reproducibility_sections(
    tmp_path: Path,
) -> None:
    """Known metric and Nsight artifacts populate the evidence tables."""
    run_dir = tmp_path / "benchmark-123"
    (run_dir / "profiles/0-on").mkdir(parents=True)
    (run_dir / "events.jsonl").write_text("")
    write_json(run_dir / "status.json", {"run_id": "benchmark-123", "exit_code": 0})
    write_json(
        run_dir / "timing_summary.json",
        {
            "median_policy_training_seconds": {"on": 4.0, "off": 5.0},
            "median_normalized_throughput": {"on": 12.5, "off": 10.0},
            "primary_on_over_off_speedup": 1.25,
        },
    )
    write_json(
        run_dir / "profiles/0-on/profile_summary.json",
        {
            "arm": "on",
            "nsight_report_count": 1,
            "kernel_evidence": "kernel_evidence.txt",
        },
    )

    renderer = load_renderer()
    renderer.render_run(run_dir)
    html = (run_dir / "report.html").read_text()

    assert "Component timing" in html
    assert "Normalized throughput" in html
    assert "1.25" in html
    assert "Nsight evidence" in html
    assert "profiles/0-on/kernel_evidence.txt" in html
    assert "Reproducibility" in html


def test_event_writer_emits_schema_and_root_cause_record(tmp_path: Path) -> None:
    """The shell writer produces valid JSONL for regular and root-cause events."""
    result_dir = tmp_path / "run"
    command = f"""
set -euo pipefail
source {EVENTS_PATH!s}
export CUTEDSL_EVENT_CLUSTER=pre_tyche
export CUTEDSL_EVENT_JOB_ID=123
cutedsl_events_init {result_dir!s}
cutedsl_write_event gpu_smoke start '' 'four-GPU Transformer Engine smoke' gpu_smoke.log
cutedsl_write_root_cause 'bad <env>' runtime_env.log 'stale path' abc123 456 \
    'submit stale env' 'inherited env' 'override env' 'job 456 passed'
"""
    subprocess.run(["bash", "-c", command], check=True)
    events = [
        json.loads(line)
        for line in (result_dir / "events.jsonl").read_text().splitlines()
    ]

    assert REQUIRED_PHASES.issubset(load_renderer().REQUIRED_PHASES)
    assert events[0]["phase"] == "gpu_smoke"
    assert events[0]["exit_code"] is None
    assert events[0]["artifact"] == "gpu_smoke.log"
    assert events[1]["phase"] == "root_cause"
    assert events[1]["symptom"] == "bad <env>"
    assert events[1]["verification_job"] == "456"
    assert events[1]["reproduction"] == "submit stale env"
    assert events[1]["hypothesis"] == "inherited env"
    assert events[1]["tested_change"] == "override env"
    assert events[1]["verification_evidence"] == "job 456 passed"


def test_payload_exit_handlers_always_render_without_masking_exit_code() -> None:
    """Both payloads render evidence from EXIT and preserve the original status."""
    for payload_path in (FUNCTIONAL_PATH, MATRIX_PATH):
        script = payload_path.read_text()
        assert 'source "${EXPERIMENT_DIR}/lib/events.sh"' in script
        assert "cutedsl_events_init" in script
        assert 'cutedsl_finalize_run "${exit_code}"' in script
        assert '"${EXPERIMENT_DIR}/render_cutedsl_report.py"' in script
        assert 'return "${final_exit_code}"' in script
        assert script.index("trap on_exit EXIT") < script.index(
            'if [[ ! -r "${IMAGE}" ]]'
        )


def test_aggregate_report_uses_local_assets_and_incident_timeline() -> None:
    """The committed index is self-contained and carries the evidence headings."""
    index = (EXPERIMENT_DIR / "report/public/index.html").read_text()
    incidents = json.loads((EXPERIMENT_DIR / "report/incidents.json").read_text())
    run_index = (EXPERIMENT_DIR / "report/run_index.tsv").read_text()

    assert "CuTeDSL experiment report" in index
    assert "Root-cause timeline" in index
    assert "Reproducibility" in index
    assert "https://" not in index
    assert "http://" not in index
    assert "<script src=" not in index
    assert isinstance(incidents, list)
    assert run_index.startswith("run_id\treport_path\tstatus\tcluster\tfeature_cell\n")


def test_report_redacts_known_credentials_from_all_displayed_inputs(
    tmp_path: Path,
) -> None:
    """Known credential fixtures never appear in structured or log HTML."""
    sentinels = {
        "aws": "SENTINEL_AWS_SECRET_91",
        "token": "SENTINEL_BUILD_TOKEN_92",
        "password": "SENTINEL_PASSWORD_93",
        "private_key": "SENTINEL_PRIVATE_KEY_94",
        "cookie": "SENTINEL_COOKIE_95",
        "bearer": "SENTINEL_BEARER_96",
        "basic": "SENTINEL_BASIC_97",
        "userinfo": "SENTINEL_URL_PASSWORD_98",
    }
    events = failure_events()
    credential_text = (
        f"AWS_SECRET_ACCESS_KEY={sentinels['aws']}\n"
        f"BUILD_TOKEN={sentinels['token']}\n"
        f"PASSWORD={sentinels['password']}\n"
        f"Cookie: session={sentinels['cookie']}\n"
        f"Authorization: Bearer {sentinels['bearer']}\n"
        f"Authorization: Basic {sentinels['basic']}\n"
        f"remote=https://user:{sentinels['userinfo']}@example.invalid/repo"
    )
    events[0]["message"] = credential_text
    events[2]["hypothesis"] = f"PRIVATE_KEY={sentinels['private_key']}"
    metadata = {
        "run": {
            "run_id": "run-123",
            "cluster_profile": "pre_tyche",
            "recipe": f"https://user:{sentinels['userinfo']}@example.invalid/config",
            "effective_config": {
                "cluster": {"num_nodes": 1, "gpus_per_node": 4},
                "credentials": {
                    "AWS_SECRET_ACCESS_KEY": sentinels["aws"],
                    "PRIVATE_KEY": sentinels["private_key"],
                },
            },
        },
        "source": {
            "sha": "a" * 40,
            "remote": f"https://user:{sentinels['userinfo']}@example.invalid/repo",
        },
        "image": {"sha256": "b" * 64},
        "slurm": {"account": "account", "partition": "batch"},
    }

    html = render_fixture(
        tmp_path, events, metadata=metadata, slurm_output=credential_text
    )

    for sentinel in sentinels.values():
        assert sentinel not in html
    assert "[REDACTED]" in html


def test_aggregate_redacts_known_incident_credentials(tmp_path: Path) -> None:
    """Aggregate incident fields pass through the same credential redaction."""
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    write_json(
        report_dir / "incidents.json",
        [
            {
                "timestamp_utc": "2026-07-11T18:00:00Z",
                "symptom": "SESSION_TOKEN=SENTINEL_INCIDENT_TOKEN_101",
                "evidence": "Authorization: Bearer SENTINEL_INCIDENT_BEARER_102",
                "root_cause": "cookie=session=SENTINEL_INCIDENT_COOKIE_103",
                "fix_commit": "abc123",
                "verification_job": "456",
            }
        ],
    )
    (report_dir / "run_index.tsv").write_text(
        "run_id\treport_path\tstatus\tcluster\tfeature_cell\n"
    )

    renderer = load_renderer()
    output = renderer.render_aggregate(report_dir)
    html = output.read_text()

    assert "SENTINEL_INCIDENT_TOKEN_101" not in html
    assert "SENTINEL_INCIDENT_BEARER_102" not in html
    assert "SENTINEL_INCIDENT_COOKIE_103" not in html


def test_event_writer_json_escapes_backslashes_exactly(tmp_path: Path) -> None:
    """All Bash controls and literal backslashes round-trip through JSON."""
    result_dir = tmp_path / "run"
    all_controls = "".join(chr(codepoint) for codepoint in range(1, 32))
    values = {
        "CUTEDSL_TEST_MESSAGE": f"controls={all_controls}; path=C:\\runtime\\unknown\\",
        "CUTEDSL_TEST_ARTIFACT": f"artifact={chr(1)}{chr(31)}\\unknown\\",
        "CUTEDSL_TEST_SYMPTOM": f"symptom={chr(8)}{chr(12)}\\q\\",
        "CUTEDSL_TEST_EVIDENCE": f"evidence={chr(27)}{chr(1)}\\z\\",
        "CUTEDSL_TEST_ROOT": f"root={chr(31)}\\cause\\",
        "CUTEDSL_TEST_REPRODUCTION": f"repro={chr(8)}\\unknown\\",
        "CUTEDSL_TEST_HYPOTHESIS": f"hypothesis={chr(12)}\\q\\",
        "CUTEDSL_TEST_CHANGE": f"change={chr(27)}\\z\\",
        "CUTEDSL_TEST_VERIFICATION": f"verify={all_controls}\\end\\",
        "CUTEDSL_TEST_RUN_ID": f"run={chr(1)}{chr(8)}{chr(12)}{chr(27)}{chr(31)}\\",
        "CUTEDSL_TEST_JOB_ID": f"job={chr(31)}{chr(1)}\\",
    }
    command = f"""
set -euo pipefail
source {EVENTS_PATH!s}
RUN_ID="$CUTEDSL_TEST_RUN_ID"
SLURM_JOB_ID="$CUTEDSL_TEST_JOB_ID"
RESULT_DIR={result_dir!s}
export RUN_ID SLURM_JOB_ID RESULT_DIR
export CUTEDSL_EVENT_CLUSTER=pre_tyche CUTEDSL_EVENT_JOB_ID="$SLURM_JOB_ID"
cutedsl_events_init {result_dir!s}
cutedsl_write_event gpu_smoke start '' "$CUTEDSL_TEST_MESSAGE" "$CUTEDSL_TEST_ARTIFACT"
cutedsl_write_root_cause "$CUTEDSL_TEST_SYMPTOM" "$CUTEDSL_TEST_EVIDENCE" \
    "$CUTEDSL_TEST_ROOT" abc123 456 "$CUTEDSL_TEST_REPRODUCTION" \
    "$CUTEDSL_TEST_HYPOTHESIS" "$CUTEDSL_TEST_CHANGE" "$CUTEDSL_TEST_VERIFICATION"
cutedsl_write_status 0
"""
    subprocess.run(["bash", "-c", command], check=True, env={**os.environ, **values})
    parsed = [
        json.loads(line)
        for line in (result_dir / "events.jsonl").read_text().splitlines()
    ]

    assert parsed[0]["message"] == values["CUTEDSL_TEST_MESSAGE"]
    assert parsed[0]["artifact"] == values["CUTEDSL_TEST_ARTIFACT"]
    assert parsed[1]["symptom"] == values["CUTEDSL_TEST_SYMPTOM"]
    assert parsed[1]["verification_evidence"] == values["CUTEDSL_TEST_VERIFICATION"]
    status = json.loads((result_dir / "status.json").read_text())
    assert status["run_id"] == values["CUTEDSL_TEST_RUN_ID"]
    assert status["job_id"] == values["CUTEDSL_TEST_JOB_ID"]


def test_early_failure_finalization_records_every_required_phase(
    tmp_path: Path,
) -> None:
    """A trapped early failure records fail/skip phases and keeps exit status."""
    result_dir = tmp_path / "run"
    command = f"""
set -euo pipefail
RESULT_DIR={result_dir!s}
RUN_ID=early-failure
SLURM_JOB_ID=321
EXPERIMENT_DIR={EXPERIMENT_DIR!s}
export RESULT_DIR RUN_ID SLURM_JOB_ID
source {EVENTS_PATH!s}
export CUTEDSL_EVENT_CLUSTER=pre_tyche CUTEDSL_EVENT_JOB_ID="$SLURM_JOB_ID"
cutedsl_events_init "$RESULT_DIR"
on_exit() {{
    local exit_code=$?
    set +e
    cutedsl_finalize_run "$exit_code" "early failure" \
        "$EXPERIMENT_DIR/render_cutedsl_report.py"
    local final_exit_code=$?
    trap - EXIT
    exit "$final_exit_code"
}}
trap on_exit EXIT
cutedsl_write_event preflight start '' 'early boundary' slurm.out
exit 23
"""
    completed = subprocess.run(["bash", "-c", command], check=False)
    events = [
        json.loads(line)
        for line in (result_dir / "events.jsonl").read_text().splitlines()
    ]
    by_phase: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        by_phase.setdefault(event["phase"], []).append(event)

    assert completed.returncode == 23
    assert REQUIRED_PHASES == REQUIRED_PHASES.intersection(by_phase)
    assert any(event["status"] == "fail" for event in by_phase["preflight"])
    assert all(phase in by_phase for phase in REQUIRED_PHASES)
    assert any(event["status"] == "skip" for event in by_phase["gpu_smoke"])
    automatic_root_cause = by_phase["root_cause"][0]
    for field in (
        "symptom",
        "evidence",
        "root_cause",
        "fix_commit",
        "verification_job",
        "reproduction",
        "hypothesis",
        "tested_change",
        "verification_evidence",
    ):
        assert automatic_root_cause[field]
    assert automatic_root_cause["root_cause"] == "Pending investigation"
    assert automatic_root_cause["verification_evidence"] == (
        "Pending verification evidence"
    )
    assert json.loads((result_dir / "status.json").read_text())["exit_code"] == 23
    html = (result_dir / "report.html").read_text()
    assert "Evidence completeness" in html
    assert "COMPLETE" in html


def test_renderer_surfaces_missing_required_phases(tmp_path: Path) -> None:
    """Legacy or malformed run evidence visibly reports phase incompleteness."""
    html = render_fixture(tmp_path, failure_events())

    assert "Evidence completeness" in html
    assert "INCOMPLETE" in html
    assert "gpu_smoke" in html


def test_success_finalization_records_every_required_phase(tmp_path: Path) -> None:
    """A successful short run records all unreached phases as skipped."""
    result_dir = tmp_path / "successful-run"
    command = f"""
set -u
RESULT_DIR={result_dir!s}
RUN_ID=successful-run
SLURM_JOB_ID=777
export RESULT_DIR RUN_ID SLURM_JOB_ID
source {EVENTS_PATH!s}
export CUTEDSL_EVENT_CLUSTER=aws_dfw CUTEDSL_EVENT_JOB_ID="$SLURM_JOB_ID"
cutedsl_events_init "$RESULT_DIR"
cutedsl_write_event preflight start '' 'short success' slurm.out
cutedsl_write_event preflight pass 0 'short success passed' slurm.out
cutedsl_finalize_run 0 'successful fixture' {RENDERER_PATH!s}
exit $?
"""
    completed = subprocess.run(["bash", "-c", command], check=False)
    events = [
        json.loads(line)
        for line in (result_dir / "events.jsonl").read_text().splitlines()
    ]

    assert completed.returncode == 0
    assert REQUIRED_PHASES.issubset({event["phase"] for event in events})
    assert any(
        event["phase"] == "gpu_smoke" and event["status"] == "skip" for event in events
    )
    assert any(
        event["phase"] == "complete" and event["status"] == "pass" for event in events
    )
    assert "COMPLETE" in (result_dir / "report.html").read_text()


@pytest.mark.parametrize(
    ("original_exit", "expected_exit"),
    [(0, 1), (17, 17)],
)
def test_finalize_run_promotes_render_failure_without_masking_original_failure(
    tmp_path: Path, original_exit: int, expected_exit: int
) -> None:
    """Report failure fails success, while an original payload error still wins."""
    result_dir = tmp_path / f"run-{original_exit}"
    command = f"""
set -u
RESULT_DIR={result_dir!s}
RUN_ID=render-failure
SLURM_JOB_ID=654
export RESULT_DIR RUN_ID SLURM_JOB_ID
source {EVENTS_PATH!s}
export CUTEDSL_EVENT_CLUSTER=pre_tyche CUTEDSL_EVENT_JOB_ID="$SLURM_JOB_ID"
export CUTEDSL_REPORT_PYTHON=false
mkdir -p "$RESULT_DIR"
printf 'stale report' > "$RESULT_DIR/report.html"
cutedsl_events_init "$RESULT_DIR"
cutedsl_write_event preflight start '' 'render failure boundary' slurm.out
cutedsl_finalize_run {original_exit} 'render failure fixture' {RENDERER_PATH!s}
exit $?
"""
    completed = subprocess.run(["bash", "-c", command], check=False)
    status = json.loads((result_dir / "status.json").read_text())
    events = [
        json.loads(line)
        for line in (result_dir / "events.jsonl").read_text().splitlines()
    ]

    assert completed.returncode == expected_exit
    assert status["exit_code"] == expected_exit
    assert not (result_dir / "report.html").exists()
    assert any(
        event["phase"] == "complete"
        and event["status"] == "fail"
        and event["exit_code"] == expected_exit
        for event in events
    )


def test_refresh_aggregate_discovers_completed_runs_and_incidents(
    tmp_path: Path,
) -> None:
    """Explicit refresh deterministically rebuilds indexes from completed runs."""
    experiment_dir = tmp_path / "experiment"
    run_dir = experiment_dir / "results/123"
    run_dir.mkdir(parents=True)
    (experiment_dir / "report").mkdir()
    write_json(
        run_dir / "status.json",
        {"run_id": "123", "job_id": "123", "exit_code": 1},
    )
    write_json(
        run_dir / "metadata.json",
        {
            "run": {"run_id": "123", "cluster_profile": "pre_tyche"},
            "source": {"sha": "a" * 40},
            "image": {"sha256": "b" * 64},
            "slurm": {"account": "account", "partition": "batch"},
        },
    )
    events = failure_events()
    events[2]["hypothesis"] = "BUILD_TOKEN=SENTINEL_COLLECTOR_TOKEN_104"
    events.append(
        {
            "timestamp_utc": "2026-07-11T18:12:00Z",
            "cluster": "pre_tyche",
            "job_id": "123",
            "phase": "runtime_diagnostic",
            "status": "fail",
            "exit_code": 1,
            "message": "COOKIE=SENTINEL_ARBITRARY_COOKIE_105",
            "artifact": "credentials.txt",
        }
    )
    (run_dir / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events)
    )
    write_json(
        run_dir / "metrics_summary.json",
        {
            "median_post_warmup_policy_training_time_s": 1.25,
            "BUILD_TOKEN": "SENTINEL_STRUCTURED_TOKEN_106",
        },
    )
    profile_dir = run_dir / "profiles/0-on"
    (profile_dir / "nsight").mkdir(parents=True)
    write_json(
        profile_dir / "profile_summary.json",
        {
            "arm": "on",
            "nsight_report_count": 1,
            "kernel_evidence": "kernel_evidence.txt",
        },
    )
    (profile_dir / "kernel_evidence.txt").write_text(
        "OVERSIZED_EVIDENCE_SENTINEL_107\n" + "x" * 100_000
    )
    (profile_dir / "nsight/worker.nsys-rep").write_bytes(b"NSYS_BINARY_SENTINEL_108")
    (run_dir / "credentials.txt").write_text("SENTINEL_ARBITRARY_FILE_109\n")
    (run_dir / "symlinked_credentials.raw").write_text(
        "SYMLINKED_CREDENTIAL_SENTINEL_112\n"
    )
    (run_dir / "topology.txt").symlink_to("symlinked_credentials.raw")
    (run_dir / "slurm.out").write_text(
        "RAW_SLURM_PREFIX_SENTINEL_110\n"
        + "y" * 40_000
        + "\nAWS_SECRET_ACCESS_KEY=SENTINEL_SLURM_SECRET_111\n"
    )

    renderer = load_renderer()
    renderer.refresh_aggregate(experiment_dir)
    first_index = (experiment_dir / "report/run_index.tsv").read_text()
    first_incidents = (experiment_dir / "report/incidents.json").read_text()
    public_dir = experiment_dir / "report/public"
    first_public_tree = {
        path.relative_to(public_dir).as_posix(): path.read_bytes()
        for path in sorted(public_dir.rglob("*"))
        if path.is_file()
    }
    renderer.refresh_aggregate(experiment_dir)

    assert "123" in first_index
    assert "runs/results/123/report.html" in first_index
    assert "UV_PROJECT_ENVIRONMENT mismatch" in first_incidents
    assert "SENTINEL_COLLECTOR_TOKEN_104" not in first_incidents
    assert first_index == (experiment_dir / "report/run_index.tsv").read_text()
    assert first_incidents == (experiment_dir / "report/incidents.json").read_text()
    second_public_tree = {
        path.relative_to(public_dir).as_posix(): path.read_bytes()
        for path in sorted(public_dir.rglob("*"))
        if path.is_file()
    }
    assert first_public_tree == second_public_tree
    assert_public_links_exist(public_dir)
    index = (public_dir / "index.html").read_text()
    assert "123" in index
    assert "UV_PROJECT_ENVIRONMENT mismatch" in index
    assert "--refresh-experiment-dir" in index
    public_bytes = b"\n".join(second_public_tree.values())
    for sentinel in (
        b"SENTINEL_COLLECTOR_TOKEN_104",
        b"SENTINEL_ARBITRARY_COOKIE_105",
        b"SENTINEL_STRUCTURED_TOKEN_106",
        b"OVERSIZED_EVIDENCE_SENTINEL_107",
        b"NSYS_BINARY_SENTINEL_108",
        b"SENTINEL_ARBITRARY_FILE_109",
        b"RAW_SLURM_PREFIX_SENTINEL_110",
        b"SENTINEL_SLURM_SECRET_111",
        b"SYMLINKED_CREDENTIAL_SENTINEL_112",
    ):
        assert sentinel not in public_bytes
    staged_run = public_dir / "runs/results/123"
    assert (staged_run / "status.json").is_file()
    assert (staged_run / "events.jsonl").is_file()
    assert (staged_run / "metrics_summary.json").is_file()
    assert (staged_run / "slurm.out").stat().st_size <= renderer.MAX_EXCERPT_BYTES
    assert not (staged_run / "credentials.txt").exists()
    assert not list(staged_run.rglob("*.nsys-rep"))


def test_matrix_report_renders_scheduler_and_complete_parallel_topology(
    tmp_path: Path,
) -> None:
    """Benchmark manifest provenance renders scheduler and TP/PP/CP/ETP/EP."""
    run_dir = tmp_path / "benchmark-123"
    run_dir.mkdir()
    (run_dir / "events.jsonl").write_text("")
    write_json(run_dir / "status.json", {"run_id": "benchmark-123", "exit_code": 0})
    write_json(
        run_dir / "benchmark_manifest.json",
        {
            "run_id": "benchmark-123",
            "cluster_profile": "pre_tyche",
            "source_sha": "a" * 40,
            "image_sha256": "b" * 64,
            "scheduler": {
                "account": "coreai_dlalgo_llm",
                "partition": "batch",
                "gres": "",
                "segment": "1",
            },
            "topology": {
                "num_nodes": 1,
                "gpus_per_node": 4,
                "tensor_model_parallel_size": 1,
                "pipeline_model_parallel_size": 1,
                "context_parallel_size": 1,
                "expert_tensor_parallel_size": 1,
                "expert_model_parallel_size": 4,
            },
        },
    )

    renderer = load_renderer()
    renderer.render_run(run_dir)
    html = (run_dir / "report.html").read_text()

    assert "coreai_dlalgo_llm / batch / segment=1" in html
    assert "TP1 / PP1 / CP1 / ETP1 / EP4" in html
    matrix = MATRIX_PATH.read_text()
    for key in (
        "CUTEDSL_ACCOUNT",
        "CUTEDSL_PARTITION",
        "CUTEDSL_GRES",
        "CUTEDSL_SEGMENT",
        '"tensor_model_parallel_size": 1',
        '"expert_model_parallel_size": 4',
    ):
        assert key in matrix


@pytest.mark.parametrize(
    "path",
    [
        "../secret.log",
        "logs/../secret.log",
        "%2e%2e/secret.log",
        r"logs\secret.log",
        "/absolute/secret.log",
        "https://example.invalid/secret.log",
        "javascript:alert(1)",
    ],
)
def test_artifact_links_reject_non_descendant_paths(path: str) -> None:
    """Only normalized report-relative descendants become clickable links."""
    renderer = load_renderer()

    assert "<a href=" not in renderer.artifact_link(path)
    assert "<a href=" in renderer.artifact_link("logs/safe.log")


def test_payloads_do_not_refresh_tracked_aggregate_during_jobs() -> None:
    """Scheduled payloads never mutate the tracked aggregate report."""
    for payload_path in (FUNCTIONAL_PATH, MATRIX_PATH):
        assert "--refresh-experiment-dir" not in payload_path.read_text()
