#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generic MX-vs-NCCL weight-refit benchmark harness.

Because MX and NCCL are both selectable via vLLM's native weight-transfer API,
the comparison is a **backend swap** on the same model + step: run the same
refit loop with ``--backend mx`` and ``--backend nccl`` and compare per-phase
timings. This script is deployment-agnostic (no cluster/namespace assumptions);
point it at your own endpoints via flags or env.

Two drive modes:
  * ``http``   — drive a running vLLM-under-Dynamo worker via its RL routes
                 using update_weights_via_mx for MX and the native init/update
                 routes for NCCL. Times pause/update/cache/resume boundaries.
  * ``inproc`` — construct the native WeightTransferEngine in-process and call
                 init/receive directly (run this inside a worker process).

Structured ``MX_REFIT_TIMING`` JSON and legacy ``[TIMING]`` / ``[mx-mdl]`` log
lines can be folded in with ``--parse-logs <file>``.

Example:
  # HTTP mode against two backends, 10 cycles each:
  python mx_vs_nccl_refit_bench.py --mode http \
      --worker-url http://<worker-host>:<port> \
      --backend mx   --model <model> --cycles 10 --out mx.json
  python mx_vs_nccl_refit_bench.py --mode http \
      --worker-url http://<worker-host>:<port> \
      --backend nccl --model <model> --cycles 10 --out nccl.json
  python mx_vs_nccl_refit_bench.py --compare mx.json nccl.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

# ----- optional deps loaded lazily so --compare works without them -----

SCHEMA_VERSION = "refit-stage-v1"
STAGE_NAMES = (
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
STATUS_AVAILABLE = "available"
STATUS_COMBINED = "combined"
STATUS_UNAVAILABLE = "unavailable"
STATUS_NOT_APPLICABLE = "not_applicable"

STAGE_ALIASES = {
    "control": "control_discovery",
    "discovery": "control_discovery",
    "pause": "control_discovery",
    "source_prepare": "source_preparation",
    "source_preparation": "source_preparation",
    "checkpoint_preload": "source_preparation",
    "setup": "setup_registration",
    "registration": "setup_registration",
    "group_init": "setup_registration",
    "planning": "transfer_planning",
    "plan": "transfer_planning",
    "transfer_planning": "transfer_planning",
    "wire": "wire_transfer",
    "transfer": "wire_transfer",
    "wire_transfer": "wire_transfer",
    "receive": "receive_sync",
    "receive_sync": "receive_sync",
    "receive_synchronization": "receive_sync",
    "translate": "transformation",
    "translation": "transformation",
    "transformation": "transformation",
    "load": "installation",
    "install": "installation",
    "installation": "installation",
    "cache": "post_install",
    "cache_reset": "post_install",
    "post_install": "post_install",
    "post_install_sync": "post_install",
    "post_install_sync_cache_reset": "post_install",
    "resume": "rollout_readiness",
    "ready": "rollout_readiness",
    "rollout_readiness": "rollout_readiness",
}


def _pctl(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1)))))
    return xs[k]


def _stats(xs: list[float]) -> dict[str, float | int | None]:
    return {
        "samples": len(xs),
        "min": min(xs) if xs else None,
        "median": statistics.median(xs) if xs else None,
        "p95": _pctl(xs, 95) if xs else None,
        "max": max(xs) if xs else None,
    }


def _empty_stage(
    status: str = STATUS_UNAVAILABLE, detail: str | None = None
) -> dict[str, Any]:
    stage: dict[str, Any] = {"status": status, "seconds": None}
    if detail:
        stage["detail"] = detail
    return stage


def _stage_seconds(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, dict):
        for key in ("seconds", "duration_s", "wall_s", "elapsed_s"):
            seconds = value.get(key)
            if isinstance(seconds, (int, float)) and not isinstance(seconds, bool):
                return float(seconds)
        duration_ms = value.get("duration_ms")
        if isinstance(duration_ms, (int, float)) and not isinstance(
            duration_ms, bool
        ):
            return float(duration_ms) / 1000.0
    return None


def _canonical_stage(name: str) -> str | None:
    normalized = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    normalized = re.sub(r"_(seconds|secs|sec|duration_s|wall_s)$", "", normalized)
    if normalized in STAGE_NAMES:
        return normalized
    return STAGE_ALIASES.get(normalized)


def _normalize_timing_payload(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Extract canonical stage entries from a structured timing object."""
    stages_obj: Any = payload.get(
        "stages",
        payload.get("durations_s", payload.get("stage_seconds", payload)),
    )
    items: list[tuple[str, Any]] = []
    if isinstance(stages_obj, dict):
        items = list(stages_obj.items())
    elif isinstance(stages_obj, list):
        items = [
            (str(item.get("name", item.get("stage", ""))), item)
            for item in stages_obj
            if isinstance(item, dict)
        ]

    normalized: dict[str, dict[str, Any]] = {}
    for raw_name, value in items:
        stage_name = _canonical_stage(raw_name)
        if stage_name is None:
            continue
        seconds = _stage_seconds(value)
        entry = dict(value) if isinstance(value, dict) else {}
        entry["seconds"] = seconds
        entry.setdefault(
            "status",
            STATUS_AVAILABLE if seconds is not None else STATUS_UNAVAILABLE,
        )
        normalized[stage_name] = entry
    return normalized


def _response_timing(response: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Consume Dynamo timing metadata without depending on one response version."""
    candidates: list[dict[str, Any]] = []
    for key in ("refit_timing", "timing", "timings"):
        value = response.get(key)
        if isinstance(value, dict):
            candidates.append(value)
    metadata = response.get("metadata")
    if isinstance(metadata, dict):
        candidates.append(metadata)
        for key in ("refit_timing", "timing", "timings"):
            value = metadata.get(key)
            if isinstance(value, dict):
                candidates.append(value)
    merged: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        merged.update(_normalize_timing_payload(candidate))
    return merged


def _aggregate_stage_records(
    cycles: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    aggregated: dict[str, dict[str, Any]] = {}
    for name in STAGE_NAMES:
        entries = [cycle["stages"][name] for cycle in cycles]
        seconds = [
            float(entry["seconds"])
            for entry in entries
            if entry.get("seconds") is not None
        ]
        statuses = sorted({str(entry.get("status", STATUS_UNAVAILABLE)) for entry in entries})
        aggregated[name] = {
            "status": statuses[0] if len(statuses) == 1 else "mixed",
            "statuses": statuses,
            **_stats(seconds),
        }
    return aggregated


def _summary(
    name: str,
    e2e: list[float],
    phases: dict[str, list[float]],
    *,
    cycles: list[dict[str, Any]] | None = None,
    byte_count: int | None = None,
) -> dict[str, Any]:
    # ``e2e_s`` and ``phases_s`` are retained for old result consumers.
    out: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "backend": name,
        "cycles": len(e2e),
        "e2e_s": _stats(e2e),
    }
    for phase, xs in phases.items():
        if xs:
            out.setdefault("phases_s", {})[phase] = {
                "min": min(xs),
                "median": statistics.median(xs),
                "p95": _pctl(xs, 95),
                "max": max(xs),
            }
    if cycles is not None:
        out["raw_cycles"] = cycles
        out["stages_s"] = _aggregate_stage_records(cycles)
        unattributed = [float(cycle["unattributed_seconds"]) for cycle in cycles]
        out["unattributed_s"] = _stats(unattributed)
        cycle_gbps = [
            float(cycle["gbps"])
            for cycle in cycles
            if cycle.get("gbps") is not None
        ]
        if cycle_gbps:
            out["gbps"] = _stats(cycle_gbps)
    if byte_count is not None:
        out["bytes"] = byte_count
        if "gbps" not in out:
            out["gbps"] = {
                key: (
                    byte_count * 8 / float(value) / 1e9
                    if isinstance(value, (int, float)) and value > 0
                    else None
                )
                for key, value in out["e2e_s"].items()
                if key != "samples"
            }
    return out


# ------------------------- HTTP drive mode -------------------------

def _http_routes(backend: str) -> dict[str, str | None]:
    if backend == "mx":
        return {
            "init": None,
            "update": "update_weights_via_mx",
            "pause": "pause_generation",
            "cache": "flush_cache",
            "resume": "resume_generation",
        }
    if backend == "nccl":
        return {
            "init": "init_weights_update_group",
            "update": "update_weights_from_distributed",
            "pause": "pause_generation",
            # The distributed handler resets prefix cache before returning.
            "cache": None,
            "resume": "resume_generation",
        }
    raise ValueError(f"unsupported backend: {backend}")


def _version_path(base: str | Path, version: int) -> Path:
    """Return one distinct publisher handshake path per weight version."""
    return Path(f"{base}.v{version}")


def _wait_for_path(path: Path, timeout: float) -> float:
    started = time.perf_counter()
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"publisher acknowledgement did not appear: {path}")
        time.sleep(0.05)
    return time.perf_counter() - started


def _cycle_from_boundaries(
    *,
    cycle: int,
    version: int,
    backend: str,
    e2e_seconds: float,
    coordination_seconds: float,
    pause_seconds: float,
    update_seconds: float,
    cache_seconds: float,
    resume_seconds: float,
    detailed: dict[str, dict[str, Any]],
    responses: dict[str, dict[str, Any]],
    byte_count: int | None,
) -> dict[str, Any]:
    stages = {name: _empty_stage() for name in STAGE_NAMES}
    stages["control_discovery"] = {
        "status": STATUS_AVAILABLE,
        "seconds": coordination_seconds + pause_seconds,
        "source": (
            "publisher READY coordination plus pause_generation HTTP boundary"
            if coordination_seconds
            else "pause_generation HTTP boundary"
        ),
        "publisher_ready_wait_seconds": coordination_seconds,
        "pause_seconds": pause_seconds,
    }
    detailed_post_install = detailed.get("post_install")
    if detailed_post_install and detailed_post_install.get("seconds") is not None:
        stages["post_install"] = {
            **detailed_post_install,
            "seconds": float(detailed_post_install["seconds"]) + cache_seconds,
            "source": (
                "structured inner post-install timing plus flush_cache "
                "HTTP boundary"
            ),
            "inner_seconds": float(detailed_post_install["seconds"]),
            "cache_reset_seconds": cache_seconds,
        }
    else:
        stages["post_install"] = {
            "status": STATUS_AVAILABLE,
            "seconds": cache_seconds,
            "source": "flush_cache HTTP boundary",
            "response": responses["cache"],
        }
    stages["rollout_readiness"] = {
        "status": STATUS_AVAILABLE,
        "seconds": resume_seconds,
        "source": "resume_generation HTTP boundary",
    }

    inner_stages = {
        name: entry
        for name, entry in detailed.items()
        if name
        not in (
            "control_discovery",
            "post_install",
            "rollout_readiness",
        )
    }
    if inner_stages:
        for name, entry in inner_stages.items():
            stages[name] = dict(entry)
            stages[name].setdefault("source", "structured MX/Dynamo timing")
        inner_measured = sum(
            float(entry["seconds"])
            for entry in inner_stages.values()
            if entry.get("seconds") is not None
        )
        if detailed_post_install and detailed_post_install.get("seconds") is not None:
            inner_measured += float(detailed_post_install["seconds"])
        update_unattributed = max(0.0, update_seconds - inner_measured)
    else:
        combined = [
            "wire_transfer",
            "receive_sync",
            "transformation",
            "installation",
        ]
        stages["wire_transfer"] = {
            "status": STATUS_COMBINED,
            "seconds": update_seconds,
            "source": f"{_http_routes(backend)['update']} HTTP boundary",
            "combined_group": "receiver_update",
            "combined_with": combined,
            "detail": "owner of the combined receiver update duration",
        }
        for name in combined[1:]:
            stages[name] = {
                "status": STATUS_COMBINED,
                "seconds": None,
                "combined_group": "receiver_update",
                "combined_with": combined,
            }
        update_unattributed = 0.0

    measured_boundaries = (
        coordination_seconds
        + pause_seconds
        + update_seconds
        + cache_seconds
        + resume_seconds
    )
    unattributed = max(0.0, e2e_seconds - measured_boundaries) + update_unattributed
    return {
        "cycle": cycle,
        "version": version,
        "backend": backend,
        "e2e_seconds": e2e_seconds,
        "unattributed_seconds": unattributed,
        "bytes": byte_count,
        "gbps": (
            byte_count * 8 / update_seconds / 1e9
            if byte_count is not None and update_seconds > 0
            else None
        ),
        "boundaries_s": {
            "publisher_ready_wait": coordination_seconds,
            "pause": pause_seconds,
            "update": update_seconds,
            "flush_cache": cache_seconds,
            "resume": resume_seconds,
        },
        "stages": stages,
        "responses": responses,
    }


def run_http(args: argparse.Namespace) -> dict[str, Any]:
    import requests  # lazy

    base = args.worker_url.rstrip("/")
    routes = _http_routes(args.backend)
    init_kwargs = json.loads(args.init_kwargs or "{}")
    update_kwargs = json.loads(args.update_kwargs or "{}")
    mx_config = json.loads(getattr(args, "mx_config", "") or "{}")
    byte_count = getattr(args, "bytes", None)

    def post(route: str, body: dict) -> dict:
        r = requests.post(f"{base}/{route}", json=body, timeout=args.timeout)
        r.raise_for_status()
        return r.json()

    setup_seconds: float | None = None
    if routes["init"]:
        init_body = dict(init_kwargs)
        if args.init_rpc:
            init_body.setdefault("engine_rpc", args.init_rpc)
        setup_start = time.perf_counter()
        init_response = post(str(routes["init"]), init_body)
        setup_seconds = time.perf_counter() - setup_start
        if init_response.get("status") not in ("ok", "success"):
            raise RuntimeError(f"{args.backend} init failed: {init_response}")

    e2e: list[float] = []
    cycle_records: list[dict[str, Any]] = []
    warmup_cycles = int(getattr(args, "warmup_cycles", 0))
    total_cycles = warmup_cycles + args.cycles
    for i in range(total_cycles):
        version = args.start_version + i
        if args.backend == "mx":
            body = {"version": version, "mx_config": mx_config}
            cycle_kwargs = dict(update_kwargs)
            if isinstance(cycle_kwargs.get("mx_config"), dict):
                body["mx_config"] = {
                    **mx_config,
                    **cycle_kwargs.pop("mx_config"),
                }
            body.update(cycle_kwargs)
            body["version"] = version
        else:
            body = dict(update_kwargs)
            if args.update_rpc:
                body.setdefault("engine_rpc", args.update_rpc)
            body["weight_version"] = str(version)

        responses: dict[str, dict[str, Any]] = {}
        cycle_start = time.perf_counter()
        coordination_seconds = 0.0
        publisher_trigger = getattr(args, "publisher_trigger", "")
        publisher_ack = getattr(args, "publisher_ack", "")
        publisher_done = getattr(args, "publisher_done", "")
        if args.backend == "mx" and publisher_trigger:
            ack_base = publisher_ack or f"{publisher_trigger}.ready"
            trigger_path = _version_path(publisher_trigger, version)
            ack_path = _version_path(ack_base, version)
            trigger_path.touch()
            coordination_seconds = _wait_for_path(
                ack_path,
                float(getattr(args, "coordination_timeout", args.timeout)),
            )
            responses["publisher"] = {
                "status": "ready",
                "trigger_path": str(trigger_path),
                "ack_path": str(ack_path),
            }
        pause_start = time.perf_counter()
        responses["pause"] = post(str(routes["pause"]), {})
        pause_seconds = time.perf_counter() - pause_start
        update_seconds = 0.0
        cache_seconds = 0.0
        resume_seconds = 0.0
        try:
            update_start = time.perf_counter()
            responses["update"] = post(str(routes["update"]), body)
            update_seconds = time.perf_counter() - update_start
            if routes["cache"]:
                cache_start = time.perf_counter()
                responses["cache"] = post(str(routes["cache"]), {})
                cache_seconds = time.perf_counter() - cache_start
            else:
                responses["cache"] = {
                    "status": "not_applicable",
                    "reason": "cache reset is included in the update route",
                }
        finally:
            resume_start = time.perf_counter()
            responses["resume"] = post(str(routes["resume"]), {})
            resume_seconds = time.perf_counter() - resume_start
        dt = time.perf_counter() - cycle_start
        if args.backend == "mx" and publisher_done:
            done_path = _version_path(publisher_done, version)
            done_path.touch()
            responses.setdefault("publisher", {})["done_path"] = str(done_path)

        for boundary, response in responses.items():
            if response.get("status") not in ("ok", "success", "ready", None):
                print(
                    f"[warn] cycle {i} {boundary}: {response}",
                    file=sys.stderr,
                )

        detailed = _response_timing(responses["update"])
        log_records = (
            _parse_structured_mx_logs(args.parse_logs) if args.parse_logs else []
        )
        log_timing = _timing_for_cycle(log_records, cycle=i, version=version)
        detailed.update(log_timing)
        measured_cycle = i - warmup_cycles
        record = _cycle_from_boundaries(
            cycle=measured_cycle,
            version=version,
            backend=args.backend,
            e2e_seconds=dt,
            coordination_seconds=coordination_seconds,
            pause_seconds=pause_seconds,
            update_seconds=update_seconds,
            cache_seconds=cache_seconds,
            resume_seconds=resume_seconds,
            detailed=detailed,
            responses=responses,
            byte_count=byte_count,
        )
        if i < warmup_cycles:
            print(f"[{args.backend}] warmup {i}: e2e={dt:.3f}s (excluded)")
            continue
        if setup_seconds is not None and measured_cycle == 0:
            record["stages"]["setup_registration"] = {
                "status": STATUS_AVAILABLE,
                "seconds": setup_seconds,
                "source": "init_weights_update_group HTTP route",
                "excluded_from_cycle_e2e": True,
            }
        e2e.append(dt)
        cycle_records.append(record)
        print(
            f"[{args.backend}] cycle {measured_cycle}: e2e={dt:.3f}s "
            f"unattributed={record['unattributed_seconds']:.3f}s"
        )

    phases = _parse_logs(args.parse_logs) if args.parse_logs else {}
    return _summary(
        args.backend,
        e2e,
        phases,
        cycles=cycle_records,
        byte_count=byte_count,
    )


# ------------------------ in-process drive mode ------------------------

def run_inproc(args: argparse.Namespace) -> dict[str, Any]:
    """Drive the native WeightTransferEngine directly (run inside a worker)."""
    from vllm.config import ParallelConfig, WeightTransferConfig
    from vllm.distributed.weight_transfer import WeightTransferEngineFactory

    cfg = WeightTransferConfig(backend=args.backend)
    engine = WeightTransferEngineFactory.create_engine(cfg, ParallelConfig())
    init_info = json.loads(args.init_kwargs or "{}")
    engine.init_transfer_engine(engine.init_info_cls(**init_info))  # type: ignore[attr-defined]

    # A no-op load callback keeps this a transport benchmark; swap in the real
    # model.load_weights (or MdlLoader(model).load_weights) to include load time.
    def _noop_load(weights):  # noqa: ANN001
        for _ in weights:
            pass

    e2e: list[float] = []
    cycle_records: list[dict[str, Any]] = []
    warmup_cycles = int(getattr(args, "warmup_cycles", 0))
    total_cycles = warmup_cycles + args.cycles
    for i in range(total_cycles):
        upd = json.loads(args.update_kwargs or "{}")
        upd["version"] = args.start_version + i
        t0 = time.perf_counter()
        engine.receive_weights(engine.update_info_cls(**upd), load_weights=_noop_load)  # type: ignore[attr-defined]
        elapsed = time.perf_counter() - t0
        measured_cycle = i - warmup_cycles
        if i < warmup_cycles:
            print(f"[{args.backend}] warmup {i}: e2e={elapsed:.3f}s (excluded)")
            continue
        e2e.append(elapsed)
        cycle_records.append(
            _cycle_from_boundaries(
                cycle=measured_cycle,
                version=args.start_version + i,
                backend=args.backend,
                e2e_seconds=elapsed,
                coordination_seconds=0.0,
                pause_seconds=0.0,
                update_seconds=elapsed,
                cache_seconds=0.0,
                resume_seconds=0.0,
                detailed={},
                responses={"pause": {}, "update": {}, "cache": {}, "resume": {}},
                byte_count=getattr(args, "bytes", None),
            )
        )
        cycle_records[-1]["stages"]["control_discovery"] = _empty_stage(
            STATUS_NOT_APPLICABLE
        )
        cycle_records[-1]["stages"]["post_install"] = _empty_stage(
            STATUS_NOT_APPLICABLE
        )
        cycle_records[-1]["stages"]["rollout_readiness"] = _empty_stage(
            STATUS_NOT_APPLICABLE
        )
        print(f"[{args.backend}] cycle {measured_cycle}: e2e={e2e[-1]:.3f}s")

    phases = _parse_logs(args.parse_logs) if args.parse_logs else {}
    return _summary(
        args.backend,
        e2e,
        phases,
        cycles=cycle_records,
        byte_count=getattr(args, "bytes", None),
    )


# --------------------------- log parsing ---------------------------

def _parse_structured_mx_logs(path: str) -> list[dict[str, Any]]:
    """Parse MX/NCCL structured timing records, ignoring unrelated log text."""
    records: list[dict[str, Any]] = []
    if not path or not os.path.exists(path):
        return records
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            marker = next(
                (
                    candidate
                    for candidate in ("MX_REFIT_TIMING", "NCCL_REFIT_TIMING")
                    if candidate in line
                ),
                None,
            )
            if marker is None:
                continue
            marker_at = line.find(marker)
            json_at = line.find("{", marker_at + len(marker))
            if json_at < 0:
                continue
            try:
                payload, _ = json.JSONDecoder().raw_decode(line[json_at:])
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            records.append(
                {
                    "line": line_number,
                    "marker": marker,
                    "payload": payload,
                    "stages": _normalize_timing_payload(payload),
                }
            )
    return records


def _record_identity(record: dict[str, Any], key: str) -> Any:
    payload = record["payload"]
    if key in payload:
        return payload[key]
    metadata = payload.get("metadata")
    return metadata.get(key) if isinstance(metadata, dict) else None


def _timing_for_cycle(
    records: list[dict[str, Any]], *, cycle: int, version: int
) -> dict[str, dict[str, Any]]:
    """Select one cycle's records and reduce per-rank stages by critical-path max."""
    matching = [
        record for record in records if _record_identity(record, "version") == version
    ]
    if not matching:
        matching = [
            record for record in records if _record_identity(record, "cycle") == cycle
        ]
    if not matching and cycle < len(records):
        matching = [records[cycle]]

    reduced: dict[str, dict[str, Any]] = {}
    for record in matching:
        for name, entry in record["stages"].items():
            current = reduced.get(name)
            seconds = entry.get("seconds")
            if (
                current is None
                or current.get("seconds") is None
                or (seconds is not None and seconds > current["seconds"])
            ):
                reduced[name] = {
                    **entry,
                    "source": "MX_REFIT_TIMING log",
                    "log_line": record["line"],
                }
    return reduced


def _parse_logs(path: str) -> dict[str, list[float]]:
    """Fold structured and legacy MX phase markers into timing lists.

    Recognizes lines like:
      MX_REFIT_TIMING {"version": 2, "stages": {...}}
      [TIMING] register 0.16s | wire 0.92s | translate 0.20s
      [mx-mdl] warm-cycle N: ... in 0.55s
    Best-effort; unknown formats are ignored.
    """
    phases: dict[str, list[float]] = {}
    if not os.path.exists(path):
        return phases
    for record in _parse_structured_mx_logs(path):
        for name, entry in record["stages"].items():
            seconds = entry.get("seconds")
            if seconds is not None:
                phases.setdefault(name, []).append(float(seconds))
    pat = re.compile(r"(register|wire|translate|load)\s+([0-9.]+)s")
    mdl = re.compile(r"\[mx-mdl\].*?in\s+([0-9.]+)s")
    with open(path, encoding="utf-8") as f:
        for line in f:
            for name, val in pat.findall(line):
                phases.setdefault(name, []).append(float(val))
            m = mdl.search(line)
            if m:
                phases.setdefault("load", []).append(float(m.group(1)))
    return phases


# ----------------------------- output ------------------------------

def _csv_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Return stable, flat rows suitable for direct Google Sheets import."""
    rows: list[dict[str, Any]] = []
    stages = result.get("stages_s", {})
    for position, name in enumerate(STAGE_NAMES, 1):
        stats = stages.get(name, {})
        rows.append(
            {
                "schema_version": result.get("schema_version", ""),
                "backend": result.get("backend", ""),
                "stage_number": position,
                "stage": name,
                "status": stats.get("status", STATUS_UNAVAILABLE),
                "samples": stats.get("samples", 0),
                "min_s": stats.get("min"),
                "median_s": stats.get("median"),
                "p95_s": stats.get("p95"),
                "max_s": stats.get("max"),
                "bytes": result.get("bytes"),
                "median_gbps": (
                    result.get("gbps", {}).get("median")
                    if isinstance(result.get("gbps"), dict)
                    else None
                ),
            }
        )
    e2e = result.get("e2e_s", {})
    rows.append(
        {
            "schema_version": result.get("schema_version", ""),
            "backend": result.get("backend", ""),
            "stage_number": 11,
            "stage": "end_to_end",
            "status": STATUS_AVAILABLE,
            "samples": e2e.get("samples", result.get("cycles", 0)),
            "min_s": e2e.get("min"),
            "median_s": e2e.get("median"),
            "p95_s": e2e.get("p95"),
            "max_s": e2e.get("max"),
            "bytes": result.get("bytes"),
            "median_gbps": (
                result.get("gbps", {}).get("median")
                if isinstance(result.get("gbps"), dict)
                else None
            ),
        }
    )
    unattributed = result.get("unattributed_s")
    if isinstance(unattributed, dict):
        rows.append(
            {
                "schema_version": result.get("schema_version", ""),
                "backend": result.get("backend", ""),
                "stage_number": 12,
                "stage": "unattributed",
                "status": STATUS_AVAILABLE,
                "samples": unattributed.get("samples", 0),
                "min_s": unattributed.get("min"),
                "median_s": unattributed.get("median"),
                "p95_s": unattributed.get("p95"),
                "max_s": unattributed.get("max"),
                "bytes": result.get("bytes"),
                "median_gbps": None,
            }
        )
    return rows


def _write_csv(result: dict[str, Any], path: str | Path) -> None:
    rows = _csv_rows(result)
    fieldnames = list(rows[0]) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ----------------------------- compare -----------------------------

def compare(paths: list[str]) -> None:
    runs = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            runs.append(json.load(handle))
    print("\n=== MX vs NCCL refit comparison ===")
    hdr = f"{'backend':<8} {'cycles':>6} {'e2e_med':>9} {'e2e_p95':>9} {'e2e_min':>9}"
    print(hdr)
    print("-" * len(hdr))
    for r in runs:
        e = r["e2e_s"]
        print(f"{r['backend']:<8} {r['cycles']:>6} "
              f"{e['median']:>9.3f} {e['p95']:>9.3f} {e['min']:>9.3f}")
    for r in runs:
        if "phases_s" in r:
            ph = " | ".join(f"{k} {v['median']:.3f}s" for k, v in r["phases_s"].items())
            print(f"  {r['backend']} phases (median): {ph}")
        if "stages_s" in r:
            available = []
            for name in STAGE_NAMES:
                stage = r["stages_s"].get(name, {})
                median = stage.get("median")
                if median is not None:
                    available.append(f"{name} {median:.3f}s [{stage.get('status')}]")
            if available:
                print(f"  {r['backend']} stages (median): " + " | ".join(available))
        unattributed = r.get("unattributed_s", {}).get("median")
        if unattributed is not None:
            print(f"  {r['backend']} unattributed (median): {unattributed:.3f}s")


# ------------------------------- cli -------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["http", "inproc"], default="http")
    ap.add_argument("--backend", choices=["mx", "nccl"], default=os.environ.get("WT_BACKEND", "mx"))
    ap.add_argument("--model", default=os.environ.get("WT_MODEL", ""))
    ap.add_argument("--worker-url", default=os.environ.get("WT_WORKER_URL", ""))
    ap.add_argument("--cycles", type=int, default=int(os.environ.get("WT_CYCLES", "10")))
    ap.add_argument(
        "--warmup-cycles",
        type=int,
        default=int(os.environ.get("WT_WARMUP_CYCLES", "1")),
        help="extra cold/warmup updates to run and exclude from summaries",
    )
    ap.add_argument("--start-version", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--init-rpc", default=os.environ.get("WT_INIT_RPC", "init_broadcaster"))
    ap.add_argument("--update-rpc", default=os.environ.get("WT_UPDATE_RPC", "update_weights_from_distributed"))
    ap.add_argument("--init-kwargs", default=os.environ.get("WT_INIT_KWARGS", ""))
    ap.add_argument("--update-kwargs", default=os.environ.get("WT_UPDATE_KWARGS", ""))
    ap.add_argument(
        "--mx-config",
        default=os.environ.get("WT_MX_CONFIG", ""),
        help="JSON object passed as update_weights_via_mx.mx_config",
    )
    ap.add_argument(
        "--publisher-trigger",
        default=os.environ.get("WT_PUBLISHER_TRIGGER", ""),
        help="optional MX publisher trigger base; a .v<version> suffix is added",
    )
    ap.add_argument(
        "--publisher-ack",
        default=os.environ.get("WT_PUBLISHER_ACK", ""),
        help="optional READY acknowledgement base (defaults to trigger.ready)",
    )
    ap.add_argument(
        "--publisher-done",
        default=os.environ.get("WT_PUBLISHER_DONE", ""),
        help="optional receive-complete acknowledgement base for the publisher",
    )
    ap.add_argument(
        "--coordination-timeout",
        type=float,
        default=float(os.environ.get("WT_COORDINATION_TIMEOUT", "900")),
    )
    ap.add_argument(
        "--bytes",
        type=int,
        default=int(os.environ["WT_BYTES"]) if os.environ.get("WT_BYTES") else None,
        help="bytes transferred per rollout rank, for Gbps reporting",
    )
    ap.add_argument("--parse-logs", default="")
    ap.add_argument("--out", default="")
    ap.add_argument(
        "--csv-out",
        default="",
        help="CSV output path (defaults to --out with a .csv suffix)",
    )
    ap.add_argument("--compare", nargs="+", help="compare N result JSON files and exit")
    args = ap.parse_args()

    if args.compare:
        compare(args.compare)
        return 0

    result = run_http(args) if args.mode == "http" else run_inproc(args)
    print("\n" + json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[saved] {args.out}")
    csv_out = args.csv_out
    if not csv_out and args.out:
        csv_out = str(Path(args.out).with_suffix(".csv"))
    if csv_out:
        _write_csv(result, csv_out)
        print(f"[saved] {csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
