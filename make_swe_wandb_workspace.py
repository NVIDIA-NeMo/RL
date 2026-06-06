#!/usr/bin/env python
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
"""Create a grouped W&B workspace for SWE GRPO runs.

The Layer-4 instrumentation logs ~10 scalars per metric
(`mean/median/min/max/stddev/p50/p90/p95/p99` + `histogram`), which W&B renders
as one panel each — ~150 panels total. This builds a saved workspace view with a
single multi-line panel per metric (mean + p50/p90/p95/p99 + max on one chart),
collapsing those ~150 panels to ~17 while keeping the queryable scalar
percentiles intact (no training-code change).

Run (login node lacks Python 3.13.13 / a writable default uv cache):

    export UV_CACHE_DIR=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/joyang/uv_cache
    SYSPY=$(command -v python3.11 || command -v python3.10 || command -v python3)
    uv run --no-project --python "$SYSPY" --with "wandb<0.19" --with wandb-workspaces \
        python make_swe_wandb_workspace.py [RUN_ID]

`wandb<0.19` is required: newer wandb dropped the top-level `wandb_gql` module
that `wandb-workspaces` imports. RUN_ID (default below) is only used to discover
which metrics exist; the workspace applies to the whole project.
"""

import sys

import wandb
import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws

ENTITY = "nvidia"
PROJECT = "swe-benchmark-harness"
DEFAULT_RUN = "w6rxgn6t"

# Lines drawn on each per-metric panel, in legend order. median is omitted
# because it equals p50; min is omitted because it is usually 0 (just noise).
LINES = ("mean", "p50", "p90", "p95", "p99", "max")


def _percentile_bases(summary_keys: list[str]) -> list[str]:
    """Return metric prefixes that have a `/p50` (i.e. got Layer-4 percentiles)."""
    return sorted(
        {k.rsplit("/", 1)[0] for k in summary_keys if k.rsplit("/", 1)[-1] == "p50"}
    )


def _panel(base: str, present: set[str]) -> wr.LinePlot:
    """One multi-line panel for a single metric base (only lines that exist)."""
    ys = [f"{base}/{s}" for s in LINES if f"{base}/{s}" in present]
    return wr.LinePlot(title=base.split("/")[-1], y=ys)


def main() -> None:
    run_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RUN
    api = wandb.Api(timeout=60)
    keys = list(api.run(f"{ENTITY}/{PROJECT}/{run_id}").summary.keys())
    present = set(keys)

    bases = _percentile_bases(keys)
    if not bases:
        raise SystemExit(f"No '/p50' metrics found in run {run_id}; nothing to group.")

    agent = [b for b in bases if "/swe_agents_train/" in b or "/swe_agents_val/" in b]
    summary = [b for b in bases if b not in agent]

    sections = []
    if summary:
        sections.append(
            ws.Section(
                name="Per-step percentiles — reward & tokens",
                panels=[_panel(b, present) for b in summary],
                is_open=True,
            )
        )
    if agent:
        sections.append(
            ws.Section(
                name="Per-step percentiles — SWE agent",
                panels=[_panel(b, present) for b in agent],
                is_open=True,
            )
        )

    workspace = ws.Workspace(
        entity=ENTITY,
        project=PROJECT,
        name="SWE percentiles (grouped)",
        sections=sections,
        # Only show the grouped panels we define — do not auto-add a panel per metric.
        auto_generate_panels=False,
    )
    saved = workspace.save()
    n_panels = sum(len(s.panels) for s in sections)
    print(
        f"Created workspace with {n_panels} grouped panels (from {len(bases)} metrics)."
    )
    print(f"URL: {getattr(saved, 'url', saved)}")


if __name__ == "__main__":
    main()
