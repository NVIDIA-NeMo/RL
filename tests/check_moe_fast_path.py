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

"""Runtime guard: assert a run actually used the MXFP8 flashinfer trtllm-gen MoE.

The static guard (``tests/unit/models/generation/test_moe_fast_path_guard.py``)
checks that the *config + env* request the fast path. This script closes the
loop at runtime: it reads a finished run's log and asserts that vLLM really
dispatched the trtllm-gen MXFP8 MoE kernel and did NOT silently fall back to a
slow path. It is meant to be called from the MXFP8 test-suite scripts after the
run completes, alongside the ``check_metrics.py`` perf-floor check.

Usage:
    python check_moe_fast_path.py <run_log>

Exit code 0 if the fast path was used; non-zero (with a diagnostic) otherwise.
"""

import argparse
import re
import sys

# ALL of these must appear -> proof the MXFP8 flashinfer trtllm-gen MoE kernel
# was actually selected at runtime. Confirmed against a real known-good Ultra
# MXFP8 serve log: it shows `quantization=modelopt_mxfp8`,
# `moe_backend='flashinfer_trtllm'`, and the kernel
# `flashinfer::trtllm_fp8_block_scale_moe`. A BF16 run instead shows
# `trtllm_bf16_moe` and no `modelopt_mxfp8`, so requiring the fp8 kernel name
# cleanly distinguishes the MXFP8 fast path from a bf16 / slow fallback.
REQUIRED_PATTERNS = [
    (r"quantization=modelopt_mxfp8", "MXFP8 quantization (modelopt_mxfp8) engaged"),
    (r"moe_backend='?flashinfer_trtllm'?", "flashinfer_trtllm MoE backend selected"),
    (r"trtllm_fp8_block_scale_moe", "MXFP8 trtllm-gen MoE kernel dispatched"),
]

# NONE of these may appear -> they indicate the optimized MoE path was
# disabled / fell back to a slow kernel, or the autotuner cache-miss storm
# (PR #2594 regression) re-occurred. trtllm_bf16_moe in an MXFP8 run means the
# experts silently ran in bf16 instead of the MXFP8 kernel.
NEGATIVE_PATTERNS = [
    (r"AUTOTUNE_FALLBACK", "flashinfer autotuner cache-miss fallback storm"),
    (
        r"falling back to.*(triton|cutlass|default).*moe",
        "MoE fell back to a slow (triton/cutlass/default) kernel",
    ),
    (
        r"flashinfer.*moe.*(disabled|not available|unavailable)",
        "flashinfer MoE reported unavailable -> slow fallback",
    ),
]

# A fallback storm = the same fallback line repeated many times. A handful of
# warmup-time fallbacks can be benign; a storm is the regression we guard.
FALLBACK_STORM_THRESHOLD = 50


def check_log(text: str) -> list[str]:
    """Return a list of failure messages (empty == fast path confirmed)."""
    failures = []

    # 1) Every required positive marker must be present.
    for pat, what in REQUIRED_PATTERNS:
        if not re.search(pat, text, re.IGNORECASE):
            failures.append(
                f"Missing fast-path marker ({what}): no match for '{pat}'. "
                f"Generation may have silently run on a slow / non-MXFP8 MoE path."
            )

    # 2) Hard negative markers must be absent.
    for pat, why in NEGATIVE_PATTERNS:
        hits = len(re.findall(pat, text, re.IGNORECASE))
        if pat.upper().find("AUTOTUNE_FALLBACK") != -1 or "AUTOTUNE_FALLBACK" in pat:
            # AUTOTUNE_FALLBACK is only a failure if it storms.
            if hits >= FALLBACK_STORM_THRESHOLD:
                failures.append(
                    f"{why}: '{pat}' appeared {hits} times "
                    f"(>= {FALLBACK_STORM_THRESHOLD}). The fast MoE path "
                    f"regressed into per-call autotune fallbacks."
                )
        elif hits > 0:
            failures.append(f"{why}: matched '{pat}' {hits}x.")

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Assert a run used the MXFP8 flashinfer trtllm-gen MoE fast path."
    )
    parser.add_argument("run_log", help="Path to the run.log to inspect")
    args = parser.parse_args()

    try:
        with open(args.run_log, "r", errors="ignore") as f:
            text = f.read()
    except FileNotFoundError:
        print(f"FAIL: run log not found: {args.run_log}", file=sys.stderr)
        return 2

    failures = check_log(text)
    if failures:
        print("FAIL: MXFP8 MoE fast-path runtime check failed:", file=sys.stderr)
        for msg in failures:
            print(f"  - {msg}", file=sys.stderr)
        return 1

    print("PASS: MXFP8 flashinfer trtllm-gen MoE fast path confirmed in run log.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
