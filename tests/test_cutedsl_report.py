# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


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
    write_json(
        run_dir / "metadata.json",
        {
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
cutedsl_write_root_cause 'bad <env>' runtime_env.log 'stale path' abc123 456
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


def test_payload_exit_handlers_always_render_without_masking_exit_code() -> None:
    """Both payloads render evidence from EXIT and preserve the original status."""
    for payload_path in (FUNCTIONAL_PATH, MATRIX_PATH):
        script = payload_path.read_text()
        assert 'source "${EXPERIMENT_DIR}/lib/events.sh"' in script
        assert "cutedsl_events_init" in script
        assert 'cutedsl_write_status "${exit_code}"' in script
        assert 'render_cutedsl_report.py" --run-dir "${RESULT_DIR}"' in script
        assert (
            'cutedsl_write_event complete "${completion_status}" "${exit_code}"'
            in script
        )
        assert 'return "${exit_code}"' in script
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
