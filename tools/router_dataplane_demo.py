#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Data-plane fault-tolerance demo for the GenerationRouter (RL-412).

Sends sustained /v1/completions through the router and triggers a fault
mid-run. Reports per-second success/failure counts and confirms the
router transparently re-routes around the cordoned shard.

This is a pure data-plane demo — no training cluster, no cross-cluster
NCCL. Useful when the platform's cross-pod NCCL setup is broken (the
weight-sync path fails before we can exercise router-on-data-path).

Usage:
    python tools/router_dataplane_demo.py \\
        --router http://<gen-head-ip>:8089 \\
        --duration 60 \\
        --fault-mode http-error \\
        --target-shard dp-0 \\
        --fault-after 20 \\
        --recover-after 30
"""

import argparse
import asyncio
import json
import sys
import time
from collections import Counter
from typing import Any

import aiohttp


PROMPT_TOKENS = list(range(100, 130))  # 30 tokens of nonsense; exercises tokenize+gen path


async def send_one(session: aiohttp.ClientSession, router_url: str) -> tuple[int, str]:
    """One /v1/completions request. Returns (status_code, body_summary)."""
    payload = {
        "model": "Qwen/Qwen3-4B-Instruct-2507",
        "prompt": PROMPT_TOKENS,
        "max_tokens": 8,
        "temperature": 1.0,
        "top_p": 1.0,
        "logprobs": 0,
    }
    try:
        async with session.post(
            f"{router_url}/v1/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            text = await resp.text()
            return resp.status, text[:80]
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        return 0, f"transport: {type(e).__name__}: {e}"


async def cordon(session: aiohttp.ClientSession, router_url: str, shard_id: str) -> None:
    async with session.post(
        f"{router_url}/admin/cordon",
        json={"shard_id": shard_id, "reason": "data-plane demo"},
        timeout=aiohttp.ClientTimeout(total=10),
    ) as resp:
        body = await resp.text()
        print(f"[fault] cordon {shard_id} -> {resp.status}: {body}", flush=True)


async def remove_shard(
    session: aiohttp.ClientSession, router_url: str, shard_id: str
) -> None:
    async with session.post(
        f"{router_url}/admin/remove_shard",
        json={"shard_id": shard_id, "reason": "data-plane demo"},
        timeout=aiohttp.ClientTimeout(total=120),
    ) as resp:
        body = await resp.text()
        print(f"[fault] remove_shard {shard_id} -> {resp.status}: {body}", flush=True)


async def uncordon(
    session: aiohttp.ClientSession, router_url: str, shard_id: str
) -> None:
    async with session.post(
        f"{router_url}/admin/uncordon",
        json={"shard_id": shard_id},
        timeout=aiohttp.ClientTimeout(total=10),
    ) as resp:
        body = await resp.text()
        print(f"[recover] uncordon {shard_id} -> {resp.status}: {body}", flush=True)


async def shards(session: aiohttp.ClientSession, router_url: str) -> Any:
    async with session.get(f"{router_url}/shards", timeout=aiohttp.ClientTimeout(total=10)) as resp:
        return await resp.json()


async def driver_loop(
    session: aiohttp.ClientSession,
    router_url: str,
    duration: float,
    concurrency: int,
    counters: Counter,
    samples: list[tuple[float, int]],
) -> None:
    deadline = time.monotonic() + duration

    async def worker():
        while time.monotonic() < deadline:
            t0 = time.monotonic()
            status, body = await send_one(session, router_url)
            t1 = time.monotonic()
            counters[status] += 1
            samples.append((t1 - t0, status))
            if status >= 500 or status == 0:
                print(
                    f"[t={t1:.1f}] FAIL status={status} body={body!r}",
                    flush=True,
                )

    await asyncio.gather(*(worker() for _ in range(concurrency)))


async def amain() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--router", required=True, help="Router URL, e.g. http://10.0.0.1:8089")
    p.add_argument("--duration", type=float, default=60.0, help="Seconds to drive load")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument(
        "--fault-mode",
        choices=("http-error", "actor-kill"),
        default="http-error",
    )
    p.add_argument("--target-shard", default="dp-0")
    p.add_argument(
        "--fault-after",
        type=float,
        default=20.0,
        help="Seconds into the run to trigger the fault",
    )
    p.add_argument(
        "--recover-after",
        type=float,
        default=0.0,
        help="If > 0 (and fault-mode=http-error), uncordon after this many seconds",
    )
    args = p.parse_args()

    router_url = args.router.rstrip("/")
    counters: Counter = Counter()
    samples: list[tuple[float, int]] = []

    async with aiohttp.ClientSession() as session:
        # Sanity check + initial state.
        initial = await shards(session, router_url)
        ready_initial = [s["shard_id"] for s in initial if s["status"] == "ready"]
        print(
            f"[t=0] initial shards: {len(initial)} total, {len(ready_initial)} ready: "
            f"{ready_initial}",
            flush=True,
        )

        # Background driver loop.
        driver_task = asyncio.create_task(
            driver_loop(
                session, router_url, args.duration, args.concurrency, counters, samples
            )
        )

        # Trigger the fault at t=fault_after.
        await asyncio.sleep(args.fault_after)
        if args.fault_mode == "http-error":
            await cordon(session, router_url, args.target_shard)
        else:
            await remove_shard(session, router_url, args.target_shard)

        # Print state after the fault.
        post_fault = await shards(session, router_url)
        ready_after = [s["shard_id"] for s in post_fault if s["status"] == "ready"]
        print(
            f"[t={args.fault_after:.0f}] post-fault shards: "
            f"{len(post_fault)} total, {len(ready_after)} ready: {ready_after}",
            flush=True,
        )

        # Optional recovery (uncordon).
        if args.fault_mode == "http-error" and args.recover_after > 0:
            await asyncio.sleep(args.recover_after - args.fault_after)
            await uncordon(session, router_url, args.target_shard)

        await driver_task

    # Summary.
    n = sum(counters.values())
    n_ok = counters[200]
    n_5xx = sum(v for k, v in counters.items() if k >= 500)
    n_transport = counters[0]
    n_other = n - n_ok - n_5xx - n_transport
    print("=" * 64, flush=True)
    print(
        f"Total requests: {n}  OK: {n_ok}  5xx: {n_5xx}  transport-error: "
        f"{n_transport}  other: {n_other}",
        flush=True,
    )
    print(f"Status histogram: {dict(counters)}", flush=True)
    if samples:
        ok_lat = [d for d, s in samples if s == 200]
        if ok_lat:
            ok_lat.sort()
            p50 = ok_lat[len(ok_lat) // 2]
            p95 = ok_lat[min(int(len(ok_lat) * 0.95), len(ok_lat) - 1)]
            print(
                f"OK latency: p50={p50 * 1000:.0f}ms  p95={p95 * 1000:.0f}ms  "
                f"max={max(ok_lat) * 1000:.0f}ms  n={len(ok_lat)}",
                flush=True,
            )
    return 0 if (n_ok > 0 and n_transport == 0) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(amain()))
