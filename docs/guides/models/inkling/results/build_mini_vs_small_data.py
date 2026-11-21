# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Merge the Inkling 2 Mini W&B run dump with the Inkling 2 Small SGLang BF16 reference rows.

    python build_mini_vs_small_data.py <inkling2-small.json> <wandb-dump.json> <out.json> <mini-run-id>...

The W&B dump maps run id to {"rows": [...]}, the history of each run; run ids are given in step order.
"""

import json
import sys

small = json.load(open(sys.argv[1]))
wb = json.load(open(sys.argv[2]))
run_ids = sys.argv[4:]  # Mini runs in step order
TRAIN = [
    "train/gen_kl_error",
    "train/policy_kl_error",
    "train/approx_entropy",
    "train/mean_gen_tokens_per_sample",
    "train/loss",
    "train/reward",
    "train/truncation_rate",
    "timing/train/total_step_time",
]
VAL = [
    "validation/accuracy",
    "validation/avg_length",
    "timing/validation/total_validation_time",
]
train, val = {}, {}
for rid in run_ids:  # later runs win on overlapping steps
    for r in wb[rid]["rows"]:
        s = r.get("nemo_rl/step")
        if s is None:
            continue
        s = int(s)
        t = {k: r[k] for k in TRAIN if k in r}
        v = {k: r[k] for k in VAL if k in r}
        if t:
            train.setdefault(s, {}).update(t)
        if v:
            val.setdefault(s, {}).update(v)
mini = {
    "label": "Inkling 2 Mini · SGLang BF16 · sync · 2 nodes colocated",
    "training": [
        {"nemo_rl/step": s, **train[s]}
        for s in sorted(train)
        if "train/reward" in train[s]
    ],
    "validation": [{"nemo_rl/step": s, **val[s]} for s in sorted(val)],
}
sm = small["experiments"]["sglang-bf16-sync"]
sm = {
    "label": "Inkling 2 Small · SGLang BF16 · sync · 32 nodes colocated",
    "training": sm["training"],
    "validation": sm["validation"],
}
json.dump(
    {
        "source": {
            "small": "docs/guides/models/inkling/results/inkling2-small.json (sglang-bf16-sync)",
            "mini": "W&B nvidia/GRPO Inkling runs " + ", ".join(run_ids),
        },
        "experiments": {
            "inkling2-small-sglang-bf16": sm,
            "inkling2-mini-sglang-bf16": mini,
        },
    },
    open(sys.argv[3], "w"),
)
print(
    "mini steps",
    mini["training"][0]["nemo_rl/step"],
    mini["training"][-1]["nemo_rl/step"],
    "val",
    [v["nemo_rl/step"] for v in mini["validation"]],
)
