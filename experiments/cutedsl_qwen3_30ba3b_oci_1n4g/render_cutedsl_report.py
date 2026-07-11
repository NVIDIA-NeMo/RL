#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Render self-contained CuTeDSL run and aggregate evidence reports."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
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
SECRET_PATTERN = re.compile(
    r"(?i)(authorization|api[_-]?key|password|secret|token)(\s*[:=]\s*).*$"
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


def escape(value: Any) -> str:
    """Return an HTML-safe string for an artifact value."""
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
    if Path(path).is_absolute() or ":" in path.partition("/")[0]:
        return f"<code>{escape(path)}</code>"
    return f'<a href="{escape(path)}">{escape(label or path)}</a>'


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
    redacted = [SECRET_PATTERN.sub(r"\1\2[REDACTED]", line) for line in lines]
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
    config = (
        run.get("effective_config", {})
        if isinstance(run.get("effective_config"), dict)
        else {}
    )
    nodes = nested(config, "cluster", "num_nodes", default=1)
    gpus = nested(config, "cluster", "gpus_per_node", default=4)
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
            f"{escape(slurm.get('account', 'unknown'))} / {escape(slurm.get('partition', 'unknown'))}",
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
            f"Qwen3 30B-A3B · {escape(nodes)} node · {escape(gpus)} GPUs",
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
            f"<td>{artifact_link(evidence_path)}</td></tr>"
        )
    reports = metrics.get("nsight_reports", [])
    if isinstance(reports, list) and reports:
        evidence = metrics.get("kernel_evidence_file")
        rows.append(
            f"<tr><td>functional</td><td>{len(reports)}</td><td>{artifact_link(str(evidence) if evidence else None)}</td></tr>"
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
            "</tr>"
        )
    return (
        "<h2>Root cause</h2><table><thead><tr><th>Symptom</th><th>Boundary evidence</th>"
        "<th>Root cause / one fix</th><th>Fix commit</th><th>Verification job</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def timeline_section(events: list[dict[str, Any]]) -> str:
    """Render the chronological incident and phase history."""
    if not events:
        return '<h2>Chronological incident history</h2><p class="muted">No events recorded.</p>'
    items = []
    for event in events:
        event_class = "event fail-event" if event.get("status") == "fail" else "event"
        artifact = artifact_link(
            str(event["artifact"]) if event.get("artifact") else None
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
        "slurm.out",
    ]
    linked = [candidate for candidate in candidates if (run_dir / candidate).exists()]
    event_artifacts = {
        str(event["artifact"])
        for event in events
        if event.get("artifact") and not Path(str(event["artifact"])).is_absolute()
    }
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
            timeline_section(events),
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
            f"<td>{escape(item.get('evidence'))}</td><td>{escape(item.get('root_cause'))}</td>"
            f"<td>{escape(item.get('fix_commit'))}</td><td>{escape(item.get('verification_job'))}</td></tr>"
            for item in incidents
        )
        or '<tr><td colspan="6" class="muted">No incidents recorded yet.</td></tr>'
    )
    body = (
        "<h1>CuTeDSL experiment report</h1>"
        '<p class="lede">Portable Qwen3 30B-A3B functional and factorial evidence.</p>'
        "<h2>Runs and reproducibility</h2><table><thead><tr><th>Run</th><th>Status</th>"
        f"<th>Cluster</th><th>Feature cell</th><th>Reproducibility</th></tr></thead><tbody>{run_rows}</tbody></table>"
        "<h2>Root-cause timeline</h2><table><thead><tr><th>Time</th><th>Symptom</th>"
        "<th>Boundary evidence</th><th>Root cause / one fix</th><th>Fix commit</th>"
        f"<th>Verification job</th></tr></thead><tbody>{incident_rows}</tbody></table>"
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
    return parser.parse_args()


def main() -> int:
    """Render the selected report and return a process exit code."""
    args = parse_args()
    if args.run_dir is not None:
        render_run(args.run_dir)
    else:
        render_aggregate(args.report_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
