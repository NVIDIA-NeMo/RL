#!/usr/bin/env -S uv run --script -q
# /// script
# dependencies = []
# ///
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Perf-floor guard: MXFP8 generation throughput must beat BF16 at the
concurrency the RL rollouts actually use.

The MXFP8 flashinfer trtllm-gen MoE wins decisively at production concurrency
but loses at very low concurrency (small-M, memory-bound) -- a structural
property of the kernel, not a regression. Measured head-to-head on Nemotron
Ultra (standalone vllm serve A/B, ignore_eos), reproduced from
``auto_harness_results/xover_*``:

    shape       c64     c128    c256    c320
    ab_mid      1.01x   1.38x   1.82x   2.69x   MXFP8/BF16
    ab_decode   0.89x   1.57x   2.52x   2.99x   MXFP8/BF16

So the guarded invariant is: at concurrency >= MIN_CONCURRENCY (default 128),
MXFP8 throughput >= BF16 throughput (within tolerance). A regression that
silently knocks generation off the MXFP8 fast path (dropped env var, kernel
fallback, EP re-enabled) shows up here as the high-concurrency ratio falling
below 1.0.

Consumes the sweep JSONs written by ``serve_bench_in_container.sh``
(``{"sweep": [{"concurrency": N, "decode_tok_s": X}, ...]}``).

Usage:
    python check_moe_speed.py --mxfp8 bench_mxfp8.json --bf16 bench_bf16.json
"""

import argparse
import json
import sys

MIN_CONCURRENCY = 128
TOLERANCE = 0.05  # allow MXFP8 to be within 5% at the crossover knee


def _sweep(path):
    d = json.load(open(path))
    if not d.get("healthy", True):
        raise SystemExit(f"FAIL: {path} reports an unhealthy serve (no data)")
    rate_key = "decode_tok_s" if "decode_tok_s" in d["sweep"][0] else "throughput_tok_s"
    return {int(r["concurrency"]): float(r[rate_key]) for r in d["sweep"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mxfp8", required=True, help="MXFP8 sweep JSON")
    ap.add_argument("--bf16", required=True, help="BF16 sweep JSON")
    ap.add_argument("--min-concurrency", type=int, default=MIN_CONCURRENCY)
    ap.add_argument("--tolerance", type=float, default=TOLERANCE)
    args = ap.parse_args()

    mx, bf = _sweep(args.mxfp8), _sweep(args.bf16)
    shared = sorted(set(mx) & set(bf))
    if not shared:
        print("FAIL: no shared concurrency points between the two sweeps", file=sys.stderr)
        return 2

    failures = []
    print(f"{'conc':>6} {'bf16':>10} {'mxfp8':>10} {'ratio':>7}  verdict")
    for c in shared:
        ratio = mx[c] / bf[c] if bf[c] else 0.0
        guarded = c >= args.min_concurrency
        ok = (not guarded) or ratio >= (1.0 - args.tolerance)
        tag = "" if not guarded else ("OK" if ok else "FAIL <-- regression")
        print(f"{c:>6} {bf[c]:>10.0f} {mx[c]:>10.0f} {ratio:>6.2f}x  {tag}")
        if guarded and not ok:
            failures.append(
                f"concurrency {c}: MXFP8 {mx[c]:.0f} tok/s < BF16 {bf[c]:.0f} tok/s "
                f"(ratio {ratio:.2f}x, floor {1.0 - args.tolerance:.2f}x). The MXFP8 "
                f"fast path regressed at a production-concurrency point."
            )

    if not any(c >= args.min_concurrency for c in shared):
        print(f"FAIL: no measured concurrency >= {args.min_concurrency}", file=sys.stderr)
        return 2

    if failures:
        print("\nFAIL: MXFP8 perf-floor violated:", file=sys.stderr)
        for m in failures:
            print(f"  - {m}", file=sys.stderr)
        return 1
    print(f"\nPASS: MXFP8 >= BF16 at all concurrency >= {args.min_concurrency}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
