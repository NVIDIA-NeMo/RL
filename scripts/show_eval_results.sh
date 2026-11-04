#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
set -euo pipefail

# =============================================================================
# show_eval_results.sh
#
# Run this ON YOUR MACBOOK. SSHes into the login node, finds every
# <variant>/<name>-step_N/*/metrics-summary.json under RESULTS_ROOT (one per
# evaluated checkpoint step and eval setting, as written by
# evaluate_bird.sbatch; <variant> is e.g. maxout-16000-temp-0), and prints one
# table with a row per (step, variant), sorted by step then variant. Shows the
# repeat count k, the pass@1 accuracy averaged over the k repeats
# (pass@1[avg-of-k]/accuracy), its std dev across repeats (k > 1 only), and
# the no_answer rate averaged the same way, and the number of rows whose reference
# SQL failed to execute (gold_execution_error), which are scored 0 regardless of the
# model's answer. A high value means an environment problem, not a model problem.
# If a (variant, step) has more than one job's results (e.g. re-evaluated),
# the most recently modified metrics-summary.json wins.
#
# Usage:
#   ./scripts/show_eval_results.sh oapl-20000seq-lr1e-5-beta-1-1e3-run4
#
#   # Only some variants (shell glob matched against the variant dir name):
#   VARIANT='maxout-24000-*' ./scripts/show_eval_results.sh <name>
#
#   # Different results root:
#   RESULTS_ROOT=/lustre/.../Gym/results ./scripts/show_eval_results.sh <name>
# =============================================================================

LOGIN_HOST="${LOGIN_HOST:-dmisra@oci-hsg-cs-001-login-01}"
RESULTS_ROOT="${RESULTS_ROOT:-/lustre/fsw/portfolios/nemotron/users/dmisra/Gym/results}"
VARIANT="${VARIANT:-*}"
# Lightning MODEL_NAMEs are nemotron-3.5-lightning-<run>-step_N; Dory's are <run>-step_N
# (see evaluate_all_checkpoints.sh). The family is auto-detected from the run name like
# there; export MODEL_FAMILY or MODEL_NAME_PREFIX (may be empty) to override.
if [[ -z "${MODEL_FAMILY:-}" ]]; then
  case "${1:-}" in
    *dory*) MODEL_FAMILY=dory ;;
    *) MODEL_FAMILY=lightning ;;
  esac
fi
if [[ -z "${MODEL_NAME_PREFIX+set}" ]]; then
  if [[ "${MODEL_FAMILY}" == dory ]]; then
    MODEL_NAME_PREFIX=""
  else
    MODEL_NAME_PREFIX="nemotron-3.5-lightning-"
  fi
fi

RUN="${1:?Usage: $0 <run_name> (matches ${MODEL_NAME_PREFIX}<run_name>-step_N dirs under RESULTS_ROOT/<variant>)}"
NAME="${MODEL_NAME_PREFIX}${RUN}"

ssh "${LOGIN_HOST}" RESULTS_ROOT="${RESULTS_ROOT}" VARIANT="${VARIANT}" NAME="${NAME}" python3 <<'PY'
import glob
import json
import os
import re
import sys

results_root = os.environ["RESULTS_ROOT"]
variant_glob = os.environ["VARIANT"]
name = os.environ["NAME"]

pattern = os.path.join(results_root, variant_glob, f"{name}-step_*", "*", "metrics-summary.json")
paths = glob.glob(pattern)

step_re = re.compile(rf"^{re.escape(name)}-step_(\d+)$")
best = {}
for p in paths:
    job_dir = os.path.dirname(p)
    step_dir_path = os.path.dirname(job_dir)
    m = step_re.match(os.path.basename(step_dir_path))
    if not m:
        continue
    key = (int(m.group(1)), os.path.basename(os.path.dirname(step_dir_path)))
    mtime = os.path.getmtime(p)
    if key not in best or mtime > best[key][1]:
        best[key] = (p, mtime)

if not best:
    print(f"No metrics-summary.json found matching: {pattern}", file=sys.stderr)
    sys.exit(1)

def fmt_pct(value):
    # gym's metrics-summary.json already reports these on a 0-100 scale,
    # not a 0-1 fraction -- no further scaling needed.
    if isinstance(value, (int, float)):
        return f"{value:.2f}%"
    return "N/A"

avg_re = re.compile(r"^pass@1\[avg-of-(\d+)\]/accuracy$")

def summarize(data):
    # Highest-k pass@1[avg-of-k] = pass@1 averaged over all repeats. Fall back to
    # plain pass@1 for older summaries that lack it.
    ks = [int(m.group(1)) for key in data if (m := avg_re.match(key))]
    if not ks:
        return "?", fmt_pct(data.get("pass@1/accuracy")), "N/A", fmt_pct(data.get("pass@1/no_answer"))
    k = max(ks)
    std_dev = next(
        (v for key, v in data.items() if key.startswith(f"pass@1[avg-of-{k}]/accuracy/") and "std_dev" in key),
        None,
    )
    na = data.get(f"pass@1[avg-of-{k}]/no_answer", data.get("pass@1/no_answer"))
    return (
        str(k),
        fmt_pct(data.get(f"pass@1[avg-of-{k}]/accuracy")),
        f"{std_dev:.2f}" if isinstance(std_dev, (int, float)) else "-",
        fmt_pct(na),
    )

def gold_errors(job_dir):
    # failure_reason only exists per row in the job's full.jsonl (not in gym's aggregate
    # metrics), so count it from there.
    full = os.path.join(job_dir, "full.jsonl")
    if not os.path.exists(full):
        return "N/A"
    n = total = 0
    with open(full) as f:
        for row_line in f:
            total += 1
            n += json.loads(row_line).get("failure_reason") == "gold_execution_error"
    return f"{n} ({100.0 * n / total:.1f}%)" if total else "N/A"

rows = []
for step, variant in sorted(best):
    path, _ = best[(step, variant)]
    try:
        with open(path) as f:
            data = json.load(f)
        rows.append((step, variant, *summarize(data), gold_errors(os.path.dirname(path)), None))
    except Exception as e:
        rows.append((step, variant, None, None, None, None, None, str(e)))

title = f" Results for '{name}' "
headers = ("Step", "Variant", "Repeats", "avg pass@1/accuracy", "std dev", "no_answer", "gold_exec_err")
widths = [
    max(len(headers[0]), max(len(str(r[0])) for r in rows)) + 2,
    max(len(headers[1]), max(len(r[1]) for r in rows)) + 2,
    *(len(h) + 2 for h in headers[2:]),
]

border = "+" + "+".join("-" * w for w in widths) + "+"

def line(cells):
    return "|" + "|".join(c.center(w) for c, w in zip(cells, widths)) + "|"

print()
print(title.center(len(border), "="))
print(results_root)
print(border)
print(line(headers))
print(border)
for step, variant, k, acc, se, na, ge, err in rows:
    if err:
        print(f"| {str(step)} {variant} ERROR: {err}")
        continue
    print(line((str(step), variant, k, acc, se, na, ge)))
print(border)
print()
PY
