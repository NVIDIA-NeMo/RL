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
"""What each training step trained on, by agent and agent family.

A multi-environment run (SWE + Terminal-Bench + search + GDPval, ...) samples
its prompt groups ready-first, so a step's mix drifts with rollout speed. Each
step logs, over the prompt groups it trained on:

* ``batch_mix/family/<family>/frac`` and ``.../groups``: share and count per
  family (``swe``, ``tb``, ``search``, ``gdpval``, else ``other``);
* ``batch_mix/agent/<agent>/frac``: share per Gym agent (``agent_ref`` name
  without the ``_sandboxed_agent`` / ``_agent`` suffix);
* ``batch_mix/chart``: a stacked bar chart, one bar per step, one segment per
  family (W&B only; see :meth:`BatchMixHistory.figure`).

Groups, not rows: a prompt group is one task's GPP rollouts, so the share of
groups is the share of rollouts; subagent segment rows would weight SWE by its
delegation rate instead.
"""

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

UNKNOWN_AGENT = "unknown"
FAMILY_ORDER = ("swe", "tb", "search", "gdpval", "other")
FAMILY_LABELS = {
    "swe": "SWE",
    "tb": "Terminal-Bench",
    "search": "Search",
    "gdpval": "GDPval",
    "other": "Other",
}
# Tableau 10: distinct under color-vision deficiency and in print.
FAMILY_COLORS = {
    "swe": "#4E79A7",
    "tb": "#F28E2B",
    "search": "#59A14F",
    "gdpval": "#B07AA1",
    "other": "#BAB0AC",
}


def agent_name_from_env_info(extra_env_info: Mapping[str, Any] | None) -> str:
    """The Gym agent a dataset row routes to (``agent_ref.name``), or ``unknown``."""
    ref = (extra_env_info or {}).get("agent_ref")
    name = ref.get("name") if isinstance(ref, Mapping) else ref
    return str(name) if name else UNKNOWN_AGENT


def agent_slices(agent_name: str) -> tuple[str, str]:
    """``(agent, family)`` for an ``agent_ref`` name.

    Same rule as critic stage B's held-out breakdown: ``agent`` drops the
    ``_sandboxed_agent`` / ``_agent`` suffix; ``family`` is ``tb``, ``search``,
    ``gdpval``, ``swe`` (any other named agent) or ``other`` (no agent).
    """
    if agent_name == UNKNOWN_AGENT:
        return UNKNOWN_AGENT, "other"
    name = agent_name
    for suffix in ("_sandboxed_agent", "_agent"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    low = name.lower()
    if "terminal" in low or low.startswith("tb"):
        family = "tb"
    elif "browsecomp" in low or "search" in low:
        family = "search"
    elif "gdpval" in low:
        family = "gdpval"
    else:
        family = "swe"
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name), family


def batch_mix_metrics(agent_names: Sequence[str]) -> dict[str, float]:
    """Per-family and per-agent shares of one step's prompt groups.

    Args:
        agent_names: one ``agent_ref`` name per trained prompt group.

    Returns:
        ``batch_mix/groups`` plus, for every family and agent present,
        ``batch_mix/family/<family>/{groups,frac}`` and
        ``batch_mix/agent/<agent>/frac``. Empty when no group trained.
    """
    total = len(agent_names)
    if total == 0:
        return {}
    slices = [agent_slices(name) for name in agent_names]
    metrics: dict[str, float] = {"batch_mix/groups": float(total)}
    for family, count in Counter(family for _, family in slices).items():
        metrics[f"batch_mix/family/{family}/groups"] = float(count)
        metrics[f"batch_mix/family/{family}/frac"] = count / total
    for agent, count in Counter(agent for agent, _ in slices).items():
        metrics[f"batch_mix/agent/{agent}/frac"] = count / total
    return metrics


@dataclass
class BatchMixHistory:
    """Per-step family counts for the stacked bar chart, last ``max_steps`` steps."""

    max_steps: int = 300
    steps: list[int] = field(default_factory=list)
    family_counts: list[dict[str, int]] = field(default_factory=list)

    def add(self, step: int, agent_names: Sequence[str]) -> None:
        """Record one trained step; steps without trained groups are skipped."""
        if not agent_names:
            return
        self.steps.append(step)
        self.family_counts.append(
            dict(Counter(agent_slices(name)[1] for name in agent_names))
        )
        del self.steps[: -self.max_steps]
        del self.family_counts[: -self.max_steps]

    def figure(self) -> Any:
        """A plotly stacked bar chart: one bar per step, 100% tall, one segment per family.

        Hover shows each segment's share and group count. Returns ``None`` when
        nothing was recorded.
        """
        if not self.steps:
            return None
        import plotly.graph_objects as go  # deferred: only the driver plots

        totals = [sum(counts.values()) for counts in self.family_counts]
        fig = go.Figure()
        for family in FAMILY_ORDER:
            counts = [c.get(family, 0) for c in self.family_counts]
            if not any(counts):
                continue
            fig.add_trace(
                go.Bar(
                    x=self.steps,
                    y=[100.0 * n / t for n, t in zip(counts, totals)],
                    customdata=[[n, t] for n, t in zip(counts, totals)],
                    name=FAMILY_LABELS[family],
                    marker={
                        "color": FAMILY_COLORS[family],
                        "line": {"color": "white", "width": 0.5},
                    },
                    hovertemplate=(
                        f"<b>{FAMILY_LABELS[family]}</b><br>step %{{x}}<br>"
                        "%{y:.1f}% (%{customdata[0]} of %{customdata[1]} groups)"
                        "<extra></extra>"
                    ),
                )
            )
        fig.update_layout(
            template="plotly_white",
            barmode="stack",
            bargap=0.12,
            title={
                "text": "Batch mix by agent family (share of prompt groups per step)",
                "x": 0.01,
                "font": {"size": 15},
            },
            xaxis={"title": "train step", "showgrid": False},
            yaxis={
                "title": "share of groups",
                "range": [0, 100],
                "ticksuffix": "%",
                "gridcolor": "#ECECEC",
            },
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "right",
                "x": 1.0,
                "traceorder": "normal",
            },
            hovermode="closest",
            margin={"l": 60, "r": 20, "t": 70, "b": 50},
            font={"family": "Inter, Helvetica, Arial, sans-serif", "size": 12},
        )
        return fig
