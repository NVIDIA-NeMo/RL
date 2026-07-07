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
                 (init_weights_update_group / update_weights_from_distributed).
                 Times the coarse end-to-end refresh per cycle.
  * ``inproc`` — construct the native WeightTransferEngine in-process and call
                 init/receive directly (run this inside a worker process).

Phase breakdown (register / wire / translate / load) is emitted by the MX engine
as ``[TIMING]`` / ``[mx-mdl]`` log lines; pass ``--parse-logs <file>`` to fold
those into the report when available.

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
import json
import os
import statistics
import sys
import time
from typing import Any

# ----- optional deps loaded lazily so --compare works without them -----


def _pctl(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1)))))
    return xs[k]


def _summary(name: str, e2e: list[float], phases: dict[str, list[float]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "backend": name,
        "cycles": len(e2e),
        "e2e_s": {
            "min": min(e2e) if e2e else None,
            "median": statistics.median(e2e) if e2e else None,
            "p95": _pctl(e2e, 95),
            "max": max(e2e) if e2e else None,
        },
    }
    for phase, xs in phases.items():
        if xs:
            out.setdefault("phases_s", {})[phase] = {
                "median": statistics.median(xs),
                "max": max(xs),
            }
    return out


# ------------------------- HTTP drive mode -------------------------

def run_http(args: argparse.Namespace) -> dict[str, Any]:
    import requests  # lazy

    base = args.worker_url.rstrip("/")
    # RL routes on Dynamo main dispatch to the vLLM worker by engine_rpc name.
    # For MX the backend is auto-registered via the vllm.general_plugins entry
    # point; select it on the worker launch with the weight-transfer config.
    init_body = {"engine_rpc": args.init_rpc, **json.loads(args.init_kwargs or "{}")}
    upd_body = {
        "engine_rpc": args.update_rpc,
        **json.loads(args.update_kwargs or "{}"),
    }

    def post(route: str, body: dict) -> dict:
        r = requests.post(f"{base}/{route}", json=body, timeout=args.timeout)
        r.raise_for_status()
        return r.json()

    # one-time init of the transfer group/engine
    if args.init_rpc:
        post("init_weights_update_group", init_body)

    e2e: list[float] = []
    for i in range(args.cycles):
        post("pause_generation", {})
        t0 = time.perf_counter()
        body = dict(upd_body)
        body["weight_version"] = str(args.start_version + i)
        resp = post("update_weights_from_distributed", body)
        dt = time.perf_counter() - t0
        post("resume_generation", {})
        if resp.get("status") != "ok":
            print(f"[warn] cycle {i}: {resp}", file=sys.stderr)
        e2e.append(dt)
        print(f"[{args.backend}] cycle {i}: e2e={dt:.3f}s")

    phases = _parse_logs(args.parse_logs) if args.parse_logs else {}
    return _summary(args.backend, e2e, phases)


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
    for i in range(args.cycles):
        upd = json.loads(args.update_kwargs or "{}")
        upd["version"] = args.start_version + i
        t0 = time.perf_counter()
        engine.receive_weights(engine.update_info_cls(**upd), load_weights=_noop_load)  # type: ignore[attr-defined]
        e2e.append(time.perf_counter() - t0)
        print(f"[{args.backend}] cycle {i}: e2e={e2e[-1]:.3f}s")

    phases = _parse_logs(args.parse_logs) if args.parse_logs else {}
    return _summary(args.backend, e2e, phases)


# --------------------------- log parsing ---------------------------

def _parse_logs(path: str) -> dict[str, list[float]]:
    """Fold MX [TIMING]/[mx-mdl] phase markers into per-phase timing lists.

    Recognizes lines like:
      [TIMING] register 0.16s | wire 0.92s | translate 0.20s
      [mx-mdl] warm-cycle N: ... in 0.55s
    Best-effort; unknown formats are ignored.
    """
    import re

    phases: dict[str, list[float]] = {}
    if not os.path.exists(path):
        return phases
    pat = re.compile(r"(register|wire|translate|load)\s+([0-9.]+)s")
    mdl = re.compile(r"\[mx-mdl\].*?in\s+([0-9.]+)s")
    with open(path) as f:
        for line in f:
            for name, val in pat.findall(line):
                phases.setdefault(name, []).append(float(val))
            m = mdl.search(line)
            if m:
                phases.setdefault("load", []).append(float(m.group(1)))
    return phases


# ----------------------------- compare -----------------------------

def compare(paths: list[str]) -> None:
    runs = [json.load(open(p)) for p in paths]
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


# ------------------------------- cli -------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["http", "inproc"], default="http")
    ap.add_argument("--backend", choices=["mx", "nccl"], default=os.environ.get("WT_BACKEND", "mx"))
    ap.add_argument("--model", default=os.environ.get("WT_MODEL", ""))
    ap.add_argument("--worker-url", default=os.environ.get("WT_WORKER_URL", ""))
    ap.add_argument("--cycles", type=int, default=int(os.environ.get("WT_CYCLES", "10")))
    ap.add_argument("--start-version", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--init-rpc", default=os.environ.get("WT_INIT_RPC", "init_broadcaster"))
    ap.add_argument("--update-rpc", default=os.environ.get("WT_UPDATE_RPC", "update_weights_from_distributed"))
    ap.add_argument("--init-kwargs", default=os.environ.get("WT_INIT_KWARGS", ""))
    ap.add_argument("--update-kwargs", default=os.environ.get("WT_UPDATE_KWARGS", ""))
    ap.add_argument("--parse-logs", default="")
    ap.add_argument("--out", default="")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
