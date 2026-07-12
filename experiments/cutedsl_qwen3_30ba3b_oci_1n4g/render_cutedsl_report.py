#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Render self-contained CuTeDSL run and aggregate evidence reports."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import posixpath
import re
import secrets
import shutil
import stat
from pathlib import Path
from typing import Any

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
MAX_EXCERPT_BYTES = 16_384
MAX_EXCERPT_LINES = 120
MAX_PUBLIC_TEXT_BYTES = 65_536
MAX_PUBLIC_STRUCTURED_BYTES = 1_048_576
PUBLIC_JSON_ALLOWLIST = (
    "status.json",
    "metadata.json",
    "benchmark_manifest.json",
    "timing_summary.json",
    "metrics_summary.json",
    "matched_config_diff.json",
    "kernel_attribution.json",
    "nemo_unit_results.json",
)
PUBLIC_TEXT_ALLOWLIST = (
    "image.sha256",
    "kernel_evidence.txt",
    "timing_order.txt",
    "topology.txt",
)
SENSITIVE_NAME_PATTERN = re.compile(
    r"(?i)(?:secret|token|password|passwd|private[_-]?key|api[_-]?key|"
    r"access[_-]?key|cookie|session|authorization|auth)"
)
SECRET_ASSIGNMENT_PATTERN = re.compile(
    r"(?im)\b([a-z0-9_.-]*(?:secret|token|password|passwd|private[_-]?key|"
    r"api[_-]?key|access[_-]?key|cookie|session|authorization|auth)"
    r"[a-z0-9_.-]*)([\"']?\s*[:=]\s*).*$"
)
AUTH_VALUE_PATTERN = re.compile(r"(?i)\b(Bearer|Basic)\s+\S+")
URL_USERINFO_PATTERN = re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://)[^/@\s]+@")
INCIDENT_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
REFRESH_COMMAND = (
    "uv run --no-project python "
    "experiments/cutedsl_qwen3_30ba3b_oci_1n4g/render_cutedsl_report.py "
    "--refresh-experiment-dir experiments/cutedsl_qwen3_30ba3b_oci_1n4g"
)
STYLE = """
:root { color-scheme: light; --ink:#182433; --muted:#607184; --line:#d9e1e8;
  --panel:#f6f8fa; --pass:#147d3f; --fail:#b42318; --accent:#0b5cad; }
* { box-sizing:border-box; } body { margin:0; color:var(--ink); background:#fff;
  font:15px/1.5 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
main { max-width:1180px; margin:0 auto; padding:32px 24px 64px; }
h1 { margin:0 0 6px; font-size:32px; } h2 { margin-top:34px; border-bottom:2px solid var(--line);
  padding-bottom:7px; } h3 { margin:18px 0 8px; } .lede,.muted { color:var(--muted); }
.badge { display:inline-block; border-radius:999px; padding:4px 10px; font-weight:700; }
.pass { color:var(--pass); background:#e8f7ee; } .fail { color:var(--fail); background:#ffebe9; }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:12px; }
.card { border:1px solid var(--line); border-radius:8px; padding:14px; background:var(--panel); }
.label { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.05em; }
.value { overflow-wrap:anywhere; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
table { width:100%; border-collapse:collapse; margin:8px 0 18px; }
th,td { padding:8px 10px; border:1px solid var(--line); text-align:left; vertical-align:top; }
th { background:var(--panel); } code,pre { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
pre { white-space:pre-wrap; overflow-wrap:anywhere; background:#111827; color:#e5edf5; padding:14px;
  border-radius:8px; max-height:430px; overflow:auto; }
a { color:var(--accent); } .timeline { border-left:3px solid var(--line); margin-left:8px; padding-left:20px; }
.event { margin:0 0 18px; } .event time { color:var(--muted); font-family:ui-monospace,monospace; }
.event.fail-event { border-left:4px solid var(--fail); padding-left:10px; }
"""


def redact_text(value: str) -> str:
    """Redact concrete common credential forms from display text."""
    redacted = URL_USERINFO_PATTERN.sub(r"\1[REDACTED]@", value)
    redacted = AUTH_VALUE_PATTERN.sub(r"\1 [REDACTED]", redacted)
    return SECRET_ASSIGNMENT_PATTERN.sub(r"\1\2[REDACTED]", redacted)


def redact_value(value: Any) -> Any:
    """Recursively redact sensitive mapping keys and credential-bearing text."""
    if isinstance(value, dict):
        return {
            key: (
                "[REDACTED]"
                if SENSITIVE_NAME_PATTERN.search(str(key))
                else redact_value(child)
            )
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [redact_value(child) for child in value]
    if isinstance(value, tuple):
        return tuple(redact_value(child) for child in value)
    if isinstance(value, str):
        return redact_text(value)
    return value


def escape(value: Any) -> str:
    """Redact and return an HTML-safe string for an artifact value."""
    value = redact_value(value)
    if value is None:
        return "—"
    if isinstance(value, (dict, list)):
        value = json.dumps(value, sort_keys=True)
    return html.escape(str(value), quote=True)


def read_json(path: Path) -> dict[str, Any]:
    """Read an optional JSON object, failing clearly for malformed content."""
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as error:
        raise ValueError(f"Malformed JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(
            f"Expected a JSON object in {path}, found {type(value).__name__}"
        )
    return value


def read_events(path: Path) -> list[dict[str, Any]]:
    """Read and chronologically order optional JSONL event records."""
    try:
        lines = path.read_text().splitlines()
    except FileNotFoundError:
        return []
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Malformed JSONL in {path}:{line_number}: {error}"
            ) from error
        if not isinstance(event, dict):
            raise ValueError(f"Expected a JSON object in {path}:{line_number}")
        events.append(event)
    return sorted(events, key=lambda event: str(event.get("timestamp_utc", "")))


def nested(value: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Return a nested mapping value or a default when any key is absent."""
    current: Any = value
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def artifact_link(path: str | None, label: str | None = None) -> str:
    """Render a safe local artifact link, or an em dash when absent."""
    if not path:
        return "—"
    redacted_path = redact_text(path)
    normalized = posixpath.normpath(path)
    is_descendant = (
        redacted_path == path
        and "\\" not in path
        and not any(character in path for character in ("%", "?", "#"))
        and not Path(path).is_absolute()
        and ":" not in path.partition("/")[0]
        and normalized == path
        and normalized not in {"", "."}
        and ".." not in Path(path).parts
    )
    if not is_descendant:
        return f"<code>{escape(path)}</code>"
    return f'<a href="{escape(path)}">{escape(label or path)}</a>'


def existing_artifact_link(
    run_dir: Path, path: str | None, label: str | None = None
) -> str:
    """Link only an existing normalized file contained by the run directory."""
    if not path or "<a href=" not in artifact_link(path, label):
        return f"<code>{escape(path)}</code>" if path else "—"
    target = (run_dir / path).resolve()
    if not target.is_relative_to(run_dir.resolve()) or not target.is_file():
        return f"<code>{escape(path)}</code>"
    return artifact_link(path, label)


def read_bounded_excerpt(path: Path) -> str:
    """Read a redacted, bounded tail from a potentially large run log."""
    try:
        with path.open("rb") as stream:
            stream.seek(0, 2)
            size = stream.tell()
            stream.seek(max(0, size - MAX_EXCERPT_BYTES))
            raw = stream.read(MAX_EXCERPT_BYTES)
    except FileNotFoundError:
        return ""
    text = raw.decode("utf-8", errors="replace")
    if size > MAX_EXCERPT_BYTES and "\n" in text:
        text = text.split("\n", 1)[1]
    lines = text.splitlines()[-MAX_EXCERPT_LINES:]
    redacted = [redact_text(line) for line in lines]
    return "\n".join(redacted)


def summary_cards(
    status: dict[str, Any], metadata: dict[str, Any], manifest: dict[str, Any]
) -> str:
    """Render high-level status, scheduler, source, image, and feature cards."""
    exit_code = status.get("exit_code")
    result = "PASS" if exit_code == 0 else "FAIL"
    badge_class = "pass" if exit_code == 0 else "fail"
    run = metadata.get("run", {}) if isinstance(metadata.get("run"), dict) else {}
    source = (
        metadata.get("source", {}) if isinstance(metadata.get("source"), dict) else {}
    )
    image = metadata.get("image", {}) if isinstance(metadata.get("image"), dict) else {}
    slurm = metadata.get("slurm", {}) if isinstance(metadata.get("slurm"), dict) else {}
    scheduler = slurm
    if not scheduler and isinstance(manifest.get("scheduler"), dict):
        scheduler = manifest["scheduler"]
    config = (
        run.get("effective_config", {})
        if isinstance(run.get("effective_config"), dict)
        else {}
    )
    nodes = nested(config, "cluster", "num_nodes", default=1)
    gpus = nested(config, "cluster", "gpus_per_node", default=4)
    manifest_topology = (
        manifest.get("topology", {})
        if isinstance(manifest.get("topology"), dict)
        else {}
    )
    nodes = manifest_topology.get("num_nodes", nodes)
    gpus = manifest_topology.get("gpus_per_node", gpus)
    megatron_config = nested(config, "policy", "megatron_cfg", default={})
    topology = manifest_topology or (
        megatron_config if isinstance(megatron_config, dict) else {}
    )
    topology_label = " / ".join(
        [
            f"TP{topology.get('tensor_model_parallel_size', '?')}",
            f"PP{topology.get('pipeline_model_parallel_size', '?')}",
            f"CP{topology.get('context_parallel_size', '?')}",
            f"ETP{topology.get('expert_tensor_parallel_size', '?')}",
            f"EP{topology.get('expert_model_parallel_size', '?')}",
        ]
    )
    scheduler_parts = [
        str(scheduler.get("account", "unknown")),
        str(scheduler.get("partition", "unknown")),
    ]
    if scheduler.get("gres"):
        scheduler_parts.append(f"gres={scheduler['gres']}")
    if scheduler.get("segment"):
        scheduler_parts.append(f"segment={scheduler['segment']}")
    feature_cell = nested(
        config,
        "policy",
        "megatron_cfg",
        "env_vars",
        "NVTE_CUTEDSL_FUSED_GROUPED_MLP",
        default=manifest.get("timing_order", "unknown"),
    )
    values = [
        (
            "Status",
            f'<span class="badge {badge_class}">{result}</span> (exit {escape(exit_code)})',
        ),
        (
            "Cluster",
            escape(
                run.get("cluster_profile", manifest.get("cluster_profile", "unknown"))
            ),
        ),
        (
            "Scheduler",
            escape(" / ".join(scheduler_parts)),
        ),
        (
            "Job",
            escape(status.get("job_id", slurm.get("job_id", "unknown"))),
        ),
        (
            "Code SHA",
            escape(source.get("sha", manifest.get("source_sha", "not recorded"))),
        ),
        (
            "Image SHA256",
            escape(image.get("sha256", manifest.get("image_sha256", "not recorded"))),
        ),
        (
            "Model / topology",
            f"Qwen3 30B-A3B · {escape(nodes)} node · {escape(gpus)} GPUs · {escape(topology_label)}",
        ),
        ("Feature cell", escape(feature_cell)),
    ]
    return (
        '<div class="grid">'
        + "".join(
            f'<div class="card"><div class="label">{label}</div><div class="value">{value}</div></div>'
            for label, value in values
        )
        + "</div>"
    )


def timing_section(timing: dict[str, Any], metrics: dict[str, Any]) -> str:
    """Render component timing and normalized throughput evidence."""
    timing_values = timing.get("median_policy_training_seconds", {})
    throughput_values = timing.get("median_normalized_throughput", {})
    if isinstance(timing_values, dict) and timing_values:
        rows = "".join(
            f"<tr><td>{escape(arm)}</td><td>{escape(seconds)}</td><td>{escape(throughput_values.get(arm) if isinstance(throughput_values, dict) else None)}</td></tr>"
            for arm, seconds in sorted(timing_values.items())
        )
        speedup = timing.get("primary_on_over_off_speedup", "not recorded")
        return (
            "<h2>Component timing</h2>"
            "<table><thead><tr><th>Feature cell</th><th>Median policy training (s)</th>"
            "<th>Normalized throughput (tokens/s/GPU)</th></tr></thead><tbody>"
            f"{rows}</tbody></table><p>Primary ON/OFF speedup: <strong>{escape(speedup)}</strong></p>"
        )
    if metrics:
        return (
            "<h2>Component timing</h2><table><tbody>"
            f"<tr><th>Median policy training (s)</th><td>{escape(metrics.get('median_post_warmup_policy_training_time_s'))}</td></tr>"
            f"<tr><th>Normalized throughput (tokens/s/GPU)</th><td>{escape(metrics.get('median_post_warmup_policy_training_tokens_per_s_per_gpu'))}</td></tr>"
            "</tbody></table>"
        )
    return '<h2>Component timing</h2><p class="muted">No timing summary recorded.</p>'


def profile_section(run_dir: Path, metrics: dict[str, Any]) -> str:
    """Render Nsight report counts and kernel-evidence links."""
    rows: list[str] = []
    for path in sorted(run_dir.glob("profiles/*/profile_summary.json")):
        summary = read_json(path)
        relative_dir = path.parent.relative_to(run_dir)
        evidence = summary.get("kernel_evidence")
        evidence_path = str(relative_dir / str(evidence)) if evidence else None
        rows.append(
            f"<tr><td>{escape(summary.get('arm'))}</td><td>{escape(summary.get('nsight_report_count'))}</td>"
            f"<td>{existing_artifact_link(run_dir, evidence_path)}</td></tr>"
        )
    reports = metrics.get("nsight_reports", [])
    if isinstance(reports, list) and reports:
        evidence = metrics.get("kernel_evidence_file")
        rows.append(
            f"<tr><td>functional</td><td>{len(reports)}</td>"
            f"<td>{existing_artifact_link(run_dir, str(evidence) if evidence else None)}</td></tr>"
        )
    if not rows:
        return (
            '<h2>Nsight evidence</h2><p class="muted">No Nsight evidence recorded.</p>'
        )
    return (
        "<h2>Nsight evidence</h2><table><thead><tr><th>Cell</th><th>Reports</th>"
        f"<th>Kernel evidence</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


def root_cause_section(events: list[dict[str, Any]], failed: bool) -> str:
    """Render the structured symptom-to-verification trail for a failure."""
    records = [event for event in events if event.get("phase") == "root_cause"]
    if not records:
        message = (
            "Root-cause record pending." if failed else "No root-cause record required."
        )
        return f'<h2>Root cause</h2><p class="muted">{message}</p>'
    rows = []
    for record in records:
        rows.append(
            "<tr>"
            f"<td>{escape(record.get('symptom'))}</td>"
            f"<td>{escape(record.get('evidence'))}</td>"
            f"<td>{escape(record.get('root_cause'))}</td>"
            f"<td>{escape(record.get('fix_commit'))}</td>"
            f"<td>{escape(record.get('verification_job'))}</td>"
            f"<td>{escape(record.get('reproduction'))}</td>"
            f"<td>{escape(record.get('hypothesis'))}</td>"
            f"<td>{escape(record.get('tested_change'))}</td>"
            f"<td>{escape(record.get('verification_evidence'))}</td>"
            "</tr>"
        )
    return (
        "<h2>Root cause</h2><table><thead><tr><th>Symptom</th><th>Boundary evidence</th>"
        "<th>Root cause / one fix</th><th>Fix commit</th><th>Verification job</th>"
        "<th>Reproduction</th><th>Hypothesis</th><th>Tested change</th>"
        "<th>Verification evidence</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def completeness_section(events: list[dict[str, Any]]) -> str:
    """Render whether every required phase has at least one event record."""
    observed = {str(event.get("phase")) for event in events}
    missing = sorted(REQUIRED_PHASES - observed)
    if missing:
        return (
            '<h2>Evidence completeness</h2><p><span class="badge fail">INCOMPLETE</span> '
            f"Missing phases: {escape(', '.join(missing))}</p>"
        )
    return (
        '<h2>Evidence completeness</h2><p><span class="badge pass">COMPLETE</span> '
        "Every required phase is recorded.</p>"
    )


def timeline_section(run_dir: Path, events: list[dict[str, Any]]) -> str:
    """Render the chronological incident and phase history."""
    if not events:
        return '<h2>Chronological incident history</h2><p class="muted">No events recorded.</p>'
    items = []
    for event in events:
        event_class = "event fail-event" if event.get("status") == "fail" else "event"
        artifact = existing_artifact_link(
            run_dir, str(event["artifact"]) if event.get("artifact") else None
        )
        items.append(
            f'<div class="{event_class}"><time>{escape(event.get("timestamp_utc"))}</time> '
            f"<strong>{escape(event.get('phase'))}: {escape(event.get('status'))}</strong>"
            f"<div>{escape(event.get('message'))}</div><div>Evidence: {artifact}</div></div>"
        )
    return f'<h2>Chronological incident history</h2><div class="timeline">{"".join(items)}</div>'


def reproducibility_section(
    run_dir: Path,
    metadata: dict[str, Any],
    manifest: dict[str, Any],
    events: list[dict[str, Any]],
) -> str:
    """Render local links to reproducibility inputs and bounded run evidence."""
    candidates = [
        "status.json",
        "events.jsonl",
        "metadata.json",
        "benchmark_manifest.json",
        "effective_config.yaml",
        "effective_config_on.yaml",
        "effective_config_off.yaml",
        "timing_summary.json",
        "metrics_summary.json",
        "kernel_attribution.json",
        "nemo_unit_results.json",
        "slurm.out",
    ]
    linked = [candidate for candidate in candidates if (run_dir / candidate).exists()]
    event_artifacts = set()
    for event in events:
        artifact = str(event["artifact"]) if event.get("artifact") else None
        if artifact and "<a href=" in existing_artifact_link(run_dir, artifact):
            event_artifacts.add(artifact)
    linked.extend(sorted(event_artifacts - set(linked)))
    links = (
        " · ".join(artifact_link(path) for path in linked)
        or "No local artifacts recorded."
    )
    recipe = nested(
        metadata, "run", "recipe", default=manifest.get("recipe", "not recorded")
    )
    return (
        "<h2>Reproducibility</h2>"
        f"<p><strong>Recipe:</strong> <code>{escape(recipe)}</code></p><p>{links}</p>"
    )


def render_run(run_dir: Path) -> Path:
    """Render one run directory to ``report.html`` and return its path."""
    status = read_json(run_dir / "status.json")
    metadata = read_json(run_dir / "metadata.json")
    manifest = read_json(run_dir / "benchmark_manifest.json")
    metrics = read_json(run_dir / "metrics_summary.json")
    timing = read_json(run_dir / "timing_summary.json")
    events = read_events(run_dir / "events.jsonl")
    failed = status.get("exit_code") != 0
    run_id = status.get(
        "run_id",
        nested(metadata, "run", "run_id", default=manifest.get("run_id", run_dir.name)),
    )
    excerpt = read_bounded_excerpt(run_dir / "slurm.out") if failed else ""
    excerpt_section = (
        f"<h2>Bounded error excerpt</h2><pre>{escape(excerpt)}</pre>" if excerpt else ""
    )
    body = "".join(
        [
            f"<h1>CuTeDSL run {escape(run_id)}</h1>",
            '<p class="lede">Self-contained functional and performance evidence.</p>',
            summary_cards(status, metadata, manifest),
            timing_section(timing, metrics),
            profile_section(run_dir, metrics),
            root_cause_section(events, failed),
            completeness_section(events),
            timeline_section(run_dir, events),
            excerpt_section,
            reproducibility_section(run_dir, metadata, manifest, events),
        ]
    )
    output = run_dir / "report.html"
    output.write_text(
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>CuTeDSL run {escape(run_id)}</title><style>{STYLE}</style></head>"
        f"<body><main>{body}</main></body></html>\n"
    )
    return output


def read_incidents(path: Path) -> list[dict[str, Any]]:
    """Read an optional aggregate incident list with strict shape validation."""
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError:
        return []
    except json.JSONDecodeError as error:
        raise ValueError(f"Malformed JSON in {path}: {error}") from error
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"Expected a JSON array of objects in {path}")
    return value


def read_run_index(path: Path) -> list[dict[str, str]]:
    """Read the optional aggregate run index TSV."""
    try:
        with path.open(newline="") as stream:
            return list(csv.DictReader(stream, delimiter="\t"))
    except FileNotFoundError:
        return []


def is_manual_incident(incident: dict[str, Any]) -> bool:
    """Return whether an incident owns a bounded report-root evidence file."""
    run_id = incident.get("run_id")
    report_path = incident.get("report_path")
    return (
        isinstance(run_id, str)
        and INCIDENT_RUN_ID_PATTERN.fullmatch(run_id) is not None
        and report_path == f"evidence/job-{run_id}.txt"
    )


def feature_cell(metadata: dict[str, Any], manifest: dict[str, Any]) -> str:
    """Return a compact feature-cell label for the aggregate run index."""
    config = nested(metadata, "run", "effective_config", default={})
    value = nested(
        config if isinstance(config, dict) else {},
        "policy",
        "megatron_cfg",
        "env_vars",
        "NVTE_CUTEDSL_FUSED_GROUPED_MLP",
        default=manifest.get("timing_order", "unknown"),
    )
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value)


def is_contained_regular_file(source_run: Path, candidate: Path) -> bool:
    """Return whether a regular non-symlink file resolves beneath a run root."""
    try:
        source_root = source_run.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError:
        return False
    return (
        candidate.is_file()
        and not candidate.is_symlink()
        and resolved.is_file()
        and resolved.is_relative_to(source_root)
    )


def write_public_json(source_run: Path, source: Path, destination: Path) -> bool:
    """Write one size-bounded structured JSON artifact with values redacted."""
    if not is_contained_regular_file(source_run, source):
        return False
    if source.stat().st_size > MAX_PUBLIC_STRUCTURED_BYTES:
        raise ValueError(f"Structured public artifact exceeds size bound: {source}")
    try:
        value = json.loads(source.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"Malformed JSON in {source}: {error}") from error
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(redact_value(value), indent=2, sort_keys=True) + "\n"
    )
    return True


def write_public_events(source_run: Path, source: Path, destination: Path) -> bool:
    """Write size-bounded JSONL events after recursive value redaction."""
    if not is_contained_regular_file(source_run, source):
        return False
    if source.stat().st_size > MAX_PUBLIC_STRUCTURED_BYTES:
        raise ValueError(f"Structured public artifact exceeds size bound: {source}")
    events = read_events(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(
            json.dumps(redact_value(event), sort_keys=True) + "\n" for event in events
        )
    )
    return True


def copy_public_text(source_run: Path, source: Path, destination: Path) -> bool:
    """Copy one allowlisted UTF-8 text artifact when it is within the size bound."""
    if (
        not is_contained_regular_file(source_run, source)
        or source.stat().st_size > MAX_PUBLIC_TEXT_BYTES
    ):
        return False
    try:
        content = source.read_text()
    except UnicodeDecodeError:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(redact_text(content))
    return True


def bounded_utf8_tail(value: str, maximum_bytes: int) -> str:
    """Return a UTF-8-safe tail whose encoded size does not exceed a bound."""
    encoded = value.encode()
    if len(encoded) <= maximum_bytes:
        return value
    return encoded[-maximum_bytes:].decode(errors="ignore")


def stage_public_run(source_run: Path, public_run: Path) -> Path:
    """Build a redacted, allowlisted public run tree and render its report."""
    if public_run.exists():
        shutil.rmtree(public_run)
    public_run.mkdir(parents=True)

    for name in PUBLIC_JSON_ALLOWLIST:
        source = source_run / name
        write_public_json(source_run, source, public_run / name)
    events_source = source_run / "events.jsonl"
    write_public_events(source_run, events_source, public_run / "events.jsonl")
    slurm_source = source_run / "slurm.out"
    if is_contained_regular_file(source_run, slurm_source):
        excerpt = bounded_utf8_tail(
            read_bounded_excerpt(slurm_source), MAX_EXCERPT_BYTES
        )
        (public_run / "slurm.out").write_text(excerpt)
    for name in PUBLIC_TEXT_ALLOWLIST:
        copy_public_text(source_run, source_run / name, public_run / name)

    for summary_source in sorted(source_run.glob("profiles/*/profile_summary.json")):
        relative = summary_source.relative_to(source_run)
        if not write_public_json(source_run, summary_source, public_run / relative):
            continue
        evidence_source = summary_source.parent / "kernel_evidence.txt"
        evidence_destination = public_run / relative.parent / "kernel_evidence.txt"
        copy_public_text(source_run, evidence_source, evidence_destination)

    return render_run(public_run)


def refresh_aggregate(experiment_dir: Path) -> Path:
    """Discover completed runs and deterministically rebuild aggregate evidence."""
    report_dir = experiment_dir / "report"
    public_runs_dir = report_dir / "public/runs"
    if public_runs_dir.exists():
        shutil.rmtree(public_runs_dir)
    rows: list[dict[str, str]] = []
    incidents = [
        incident
        for incident in read_incidents(report_dir / "incidents.json")
        if is_manual_incident(incident)
    ]
    incident_fields = [
        "timestamp_utc",
        "symptom",
        "evidence",
        "root_cause",
        "fix_commit",
        "verification_job",
        "reproduction",
        "hypothesis",
        "tested_change",
        "verification_evidence",
    ]

    for category in ("results", "benchmark_results"):
        category_dir = experiment_dir / category
        if not category_dir.is_dir():
            continue
        for run_dir in sorted(
            path
            for path in category_dir.iterdir()
            if path.is_dir() and not path.is_symlink()
        ):
            if not is_contained_regular_file(run_dir, run_dir / "status.json"):
                continue
            public_relative = Path("runs") / category / run_dir.name / "report.html"
            public_run = (report_dir / "public" / public_relative).parent
            stage_public_run(run_dir, public_run)
            status = read_json(public_run / "status.json")
            metadata = read_json(public_run / "metadata.json")
            manifest = read_json(public_run / "benchmark_manifest.json")
            events = read_events(public_run / "events.jsonl")

            run_id = str(
                status.get(
                    "run_id",
                    nested(
                        metadata,
                        "run",
                        "run_id",
                        default=manifest.get("run_id", run_dir.name),
                    ),
                )
            )
            cluster = str(
                nested(
                    metadata,
                    "run",
                    "cluster_profile",
                    default=manifest.get("cluster_profile", "unknown"),
                )
            )
            row = {
                "run_id": redact_text(run_id),
                "report_path": public_relative.as_posix(),
                "status": "PASS" if status.get("exit_code") == 0 else "FAIL",
                "cluster": redact_text(cluster),
                "feature_cell": redact_text(feature_cell(metadata, manifest)),
            }
            rows.append(row)
            for event in events:
                if event.get("phase") != "root_cause":
                    continue
                incident = {
                    field: redact_value(event.get(field)) for field in incident_fields
                }
                incident.update(
                    {
                        "run_id": row["run_id"],
                        "report_path": row["report_path"],
                        "cluster": row["cluster"],
                    }
                )
                incidents.append(incident)

    rows.sort(key=lambda row: (row["run_id"], row["report_path"]))
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / "run_index.tsv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "run_id",
                "report_path",
                "status",
                "cluster",
                "feature_cell",
            ],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    incidents.sort(
        key=lambda incident: (
            str(incident.get("timestamp_utc", "")),
            str(incident.get("run_id", "")),
        )
    )
    (report_dir / "incidents.json").write_text(
        json.dumps(incidents, indent=2, sort_keys=True) + "\n"
    )
    return render_aggregate(report_dir)


def _open_directory_at(parent_fd: int, name: str, *, create: bool = False) -> int:
    """Open one direct directory child without following symbolic links."""
    if create:
        try:
            os.mkdir(name, mode=0o755, dir_fd=parent_fd)
        except FileExistsError:
            pass
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    return os.open(name, flags, dir_fd=parent_fd)


def _directory_entry_matches(parent_fd: int, name: str, child_fd: int) -> bool:
    """Return whether a direct non-symlink child still names the held directory."""
    try:
        entry = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        held = os.fstat(child_fd)
    except OSError:
        return False
    return (
        stat.S_ISDIR(entry.st_mode)
        and entry.st_dev == held.st_dev
        and entry.st_ino == held.st_ino
    )


def _unlink_at(directory_fd: int, name: str) -> bool:
    """Unlink one direct non-directory child, treating absence as success."""
    try:
        os.unlink(name, dir_fd=directory_fd)
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return True


def _read_bounded_regular_text(directory_fd: int, name: str) -> str | None:
    """Read one bounded UTF-8 regular file without following its leaf."""
    file_descriptor: int | None = None
    try:
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
        file_descriptor = os.open(name, flags, dir_fd=directory_fd)
        file_status = os.fstat(file_descriptor)
        if (
            not stat.S_ISREG(file_status.st_mode)
            or file_status.st_size > MAX_PUBLIC_TEXT_BYTES
        ):
            return None
        content = bytearray()
        while len(content) <= MAX_PUBLIC_TEXT_BYTES:
            chunk = os.read(
                file_descriptor,
                min(65_536, MAX_PUBLIC_TEXT_BYTES + 1 - len(content)),
            )
            if not chunk:
                break
            content.extend(chunk)
        if len(content) > MAX_PUBLIC_TEXT_BYTES:
            return None
        return content.decode()
    except (OSError, UnicodeDecodeError):
        return None
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)


def _write_all(file_descriptor: int, data: bytes) -> bool:
    """Write every byte to an open file descriptor."""
    remaining = memoryview(data)
    while remaining:
        written = os.write(file_descriptor, remaining)
        if written <= 0:
            return False
        remaining = remaining[written:]
    return True


def _atomic_stage_incident_evidence(
    source_directory_fd: int,
    destination_directory_fd: int,
    filename: str,
) -> bool:
    """Redact and atomically stage one bounded incident evidence file."""
    content = _read_bounded_regular_text(source_directory_fd, filename)
    if content is None:
        return False
    payload = redact_text(content).encode()
    temporary_name = f".{filename}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
    temporary_fd: int | None = None
    temporary_exists = False
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
        temporary_fd = os.open(
            temporary_name,
            flags,
            0o644,
            dir_fd=destination_directory_fd,
        )
        temporary_exists = True
        if not _write_all(temporary_fd, payload):
            return False
        os.fsync(temporary_fd)
        os.close(temporary_fd)
        temporary_fd = None
        os.replace(
            temporary_name,
            filename,
            src_dir_fd=destination_directory_fd,
            dst_dir_fd=destination_directory_fd,
        )
        temporary_exists = False
        return True
    except OSError:
        return False
    finally:
        if temporary_fd is not None:
            os.close(temporary_fd)
        if temporary_exists:
            _unlink_at(destination_directory_fd, temporary_name)


def _regular_entry_at(directory_fd: int, name: str) -> bool:
    """Return whether one held-directory child is a regular non-symlink file."""
    try:
        entry = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError:
        return False
    return stat.S_ISREG(entry.st_mode)


def aggregate_incident_evidence(report_dir: Path, incident: dict[str, Any]) -> str:
    """Stage and link one safe report-root incident artifact when it exists."""
    evidence = escape(incident.get("evidence"))
    report_path = incident.get("report_path")
    if not isinstance(report_path, str) or not report_path:
        return evidence
    unavailable = f"{evidence}<br><code>{escape(report_path)}</code>"
    run_id = incident.get("run_id")
    if not isinstance(run_id, str) or INCIDENT_RUN_ID_PATTERN.fullmatch(run_id) is None:
        return unavailable
    expected_path = f"evidence/job-{run_id}.txt"
    if report_path != expected_path:
        return unavailable
    filename = f"job-{run_id}.txt"
    report_fd: int | None = None
    public_fd: int | None = None
    source_evidence_fd: int | None = None
    destination_evidence_fd: int | None = None
    try:
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        report_fd = os.open(report_dir, flags)
        public_fd = _open_directory_at(report_fd, "public", create=True)
        destination_evidence_fd = _open_directory_at(public_fd, "evidence", create=True)
        if not _unlink_at(destination_evidence_fd, filename):
            return unavailable
        source_evidence_fd = _open_directory_at(report_fd, "evidence")
        if not _directory_entry_matches(report_fd, "evidence", source_evidence_fd):
            return unavailable
        if not _atomic_stage_incident_evidence(
            source_evidence_fd,
            destination_evidence_fd,
            filename,
        ):
            _unlink_at(destination_evidence_fd, filename)
            return unavailable
        if not (
            _directory_entry_matches(report_fd, "public", public_fd)
            and _directory_entry_matches(public_fd, "evidence", destination_evidence_fd)
            and _regular_entry_at(destination_evidence_fd, filename)
        ):
            _unlink_at(destination_evidence_fd, filename)
            return unavailable
    except OSError:
        if destination_evidence_fd is not None:
            _unlink_at(destination_evidence_fd, filename)
        return unavailable
    finally:
        for file_descriptor in (
            source_evidence_fd,
            destination_evidence_fd,
            public_fd,
            report_fd,
        ):
            if file_descriptor is not None:
                os.close(file_descriptor)
    link = existing_artifact_link(
        report_dir / "public", report_path, "evidence snapshot"
    )
    return f"{evidence}<br>{link}"


def render_aggregate(report_dir: Path) -> Path:
    """Render the committed aggregate index from incidents and run-index data."""
    incidents = sorted(
        read_incidents(report_dir / "incidents.json"),
        key=lambda incident: str(incident.get("timestamp_utc", "")),
    )
    runs = read_run_index(report_dir / "run_index.tsv")
    run_rows = (
        "".join(
            "<tr>"
            f"<td>{escape(run.get('run_id'))}</td><td>{escape(run.get('status'))}</td>"
            f"<td>{escape(run.get('cluster'))}</td><td>{escape(run.get('feature_cell'))}</td>"
            f"<td>{artifact_link(run.get('report_path'), 'report.html')}</td></tr>"
            for run in runs
        )
        or '<tr><td colspan="5" class="muted">No run reports indexed yet.</td></tr>'
    )
    incident_rows = (
        "".join(
            "<tr>"
            f"<td>{escape(item.get('timestamp_utc'))}</td><td>{escape(item.get('symptom'))}</td>"
            f"<td>{aggregate_incident_evidence(report_dir, item)}</td><td>{escape(item.get('root_cause'))}</td>"
            f"<td>{escape(item.get('fix_commit'))}</td><td>{escape(item.get('verification_job'))}</td>"
            f"<td>{escape(item.get('reproduction'))}</td><td>{escape(item.get('hypothesis'))}</td>"
            f"<td>{escape(item.get('tested_change'))}</td>"
            f"<td>{escape(item.get('verification_evidence'))}</td></tr>"
            for item in incidents
        )
        or '<tr><td colspan="10" class="muted">No incidents recorded yet.</td></tr>'
    )
    body = (
        "<h1>CuTeDSL experiment report</h1>"
        '<p class="lede">Portable Qwen3 30B-A3B functional and factorial evidence.</p>'
        "<h2>Runs and reproducibility</h2><table><thead><tr><th>Run</th><th>Status</th>"
        f"<th>Cluster</th><th>Feature cell</th><th>Reproducibility</th></tr></thead><tbody>{run_rows}</tbody></table>"
        "<h2>Root-cause timeline</h2><table><thead><tr><th>Time</th><th>Symptom</th>"
        "<th>Boundary evidence</th><th>Root cause / one fix</th><th>Fix commit</th>"
        "<th>Verification job</th><th>Reproduction</th><th>Hypothesis</th>"
        "<th>Tested change</th><th>Verification evidence</th></tr></thead>"
        f"<tbody>{incident_rows}</tbody></table>"
        "<h2>Refresh aggregate</h2><p>After completed result directories are collected, "
        "run this explicit deterministic refresh. Scheduled payloads do not mutate tracked "
        f"aggregate files.</p><pre>{escape(REFRESH_COMMAND)}</pre>"
    )
    output = report_dir / "public/index.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>CuTeDSL experiment report</title><style>{STYLE}</style></head>"
        f"<body><main>{body}</main></body></html>\n"
    )
    return output


def parse_args() -> argparse.Namespace:
    """Parse renderer command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-dir", type=Path)
    group.add_argument("--report-dir", type=Path)
    group.add_argument(
        "--refresh-experiment-dir",
        type=Path,
        help="discover completed runs and rebuild TSV, incidents JSON, and aggregate HTML",
    )
    return parser.parse_args()


def main() -> int:
    """Render the selected report and return a process exit code."""
    args = parse_args()
    if args.run_dir is not None:
        render_run(args.run_dir)
    elif args.report_dir is not None:
        render_aggregate(args.report_dir)
    else:
        refresh_aggregate(args.refresh_experiment_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
