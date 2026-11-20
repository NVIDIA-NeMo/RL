# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS
# OF ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.
"""Bitwise generator-vs-trainer log-prob check on a logged GRPO step (train_data_step<k>.jsonl).

At identical weights (step 1, or any step at lag 0) with batch-invariant kernels, generation_logprobs must equal the
trainer's prev_logprobs exactly on every loss-masked token. Usage: logprob_match.py <train_data_stepK.jsonl> [...]
"""

import json
import math
import struct
import sys
from typing import Any


def first(x: list[Any]) -> list[Any]:
    return x[0] if x and isinstance(x[0], list) else x


failed = False
for path in sys.argv[1:]:
    n = nz = seqs = bad = 0
    mx = 0.0
    ex = []
    for line in open(path):
        r = json.loads(line)
        seqs += 1
        g, p, m = (
            first(r[k])
            for k in ("generation_logprobs", "prev_logprobs", "token_loss_mask")
        )
        hit = False
        for i, (a, c, k) in enumerate(zip(g, p, m, strict=True)):
            if not k:
                continue
            n += 1
            if not (math.isfinite(a) and math.isfinite(c)) or struct.pack(
                "!f", a
            ) != struct.pack("!f", c):
                nz += 1
                hit = True
                mx = max(mx, abs(a - c))
                if len(ex) < 3:
                    ex.append((seqs, i, a, c))
        bad += hit
    verdict = "BITWISE MATCH" if nz == 0 and n > 0 else "MISMATCH"
    failed |= verdict == "MISMATCH"
    print(
        f"{path}: {verdict} seqs {seqs} tokens {n} mismatched {nz} ({nz / max(n, 1):.2e}) seqs-with-mismatch {bad} "
        f"max|diff| {mx:.3e} examples {ex}"
    )

if failed:
    raise SystemExit(1)
