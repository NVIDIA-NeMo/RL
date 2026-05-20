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
"""Build Grafana dashboards for offline Dynamo Prometheus exports."""

import copy
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

DEFAULT_GRAFANA_DASHBOARD_FILENAME = "grafana-dashboard.json"
DEFAULT_GRAFANA_DASHBOARD_TEMPLATE_FILENAME = "dynamo_grafana_dashboard_template.json"

_PROMQL_METRIC_REFERENCE_RE = re.compile(
    r"(?<![A-Za-z0-9_:])([A-Za-z_:][A-Za-z0-9_:]*)(?=\s*(?:\{|\[|\)|,|$))"
)
_PROMQL_METRIC_PREFIXES = (
    "DCGM_",
    "container_",
    "dynamo_",
    "etcd_",
    "go_",
    "nats_",
    "node_",
    "process_",
    "sglang:",
    "trtllm_",
    "vllm:",
)


def build_grafana_dashboard(
    start_time_iso: Optional[str],
    end_time_iso: Optional[str],
    available_metric_names: set[str] | None = None,
) -> dict[str, Any]:
    """Load the bundled Dynamo dashboard and clip it to exported metrics."""
    dashboard = _load_grafana_dashboard_template()
    dashboard["id"] = None
    dashboard["uid"] = "nemo-rl-dynamo-prometheus"
    dashboard["title"] = "NeMo RL Dynamo Prometheus Replay"
    dashboard["refresh"] = ""
    dashboard["time"] = {
        "from": start_time_iso or "now-30m",
        "to": end_time_iso or "now",
    }

    if available_metric_names is not None:
        dashboard["panels"] = _filter_grafana_panels(
            dashboard.get("panels", []), available_metric_names
        )
        _reflow_grafana_panels(dashboard["panels"])

    return dashboard


def _load_grafana_dashboard_template() -> dict[str, Any]:
    template_path = Path(__file__).with_name(
        DEFAULT_GRAFANA_DASHBOARD_TEMPLATE_FILENAME
    )
    with open(template_path, encoding="utf-8") as f:
        return json.load(f)


def _filter_grafana_panels(
    panels: Sequence[Mapping[str, Any]],
    available_metric_names: set[str],
) -> list[dict[str, Any]]:
    filtered = []
    for panel in panels:
        filtered_panel = _filter_grafana_panel_targets(panel, available_metric_names)
        if filtered_panel is not None:
            filtered.append(filtered_panel)
    return filtered


def _filter_grafana_panel_targets(
    panel: Mapping[str, Any],
    available_metric_names: set[str],
) -> Optional[dict[str, Any]]:
    targets = panel.get("targets")
    if not isinstance(targets, Sequence) or isinstance(targets, (str, bytes)):
        return copy.deepcopy(dict(panel))

    kept_targets = []
    for target in targets:
        if not isinstance(target, Mapping):
            continue
        expr = target.get("expr")
        if not isinstance(expr, str):
            continue
        metric_names = _extract_promql_metric_names(expr)
        if metric_names and metric_names.issubset(available_metric_names):
            kept_targets.append(copy.deepcopy(dict(target)))

    if not kept_targets:
        return None

    filtered_panel = copy.deepcopy(dict(panel))
    filtered_panel["targets"] = kept_targets
    return filtered_panel


def _reflow_grafana_panels(panels: Sequence[dict[str, Any]]) -> None:
    x = 0
    y = 0
    row_height = 0
    for panel in panels:
        grid_pos = dict(panel.get("gridPos", {}) or {})
        width = int(grid_pos.get("w", 12))
        height = int(grid_pos.get("h", 8))
        width = max(1, min(width, 24))
        height = max(1, height)
        if x + width > 24:
            x = 0
            y += row_height
            row_height = 0
        panel["gridPos"] = {"h": height, "w": width, "x": x, "y": y}
        x += width
        row_height = max(row_height, height)


def _extract_promql_metric_names(expr: str) -> set[str]:
    names = set()
    for match in _PROMQL_METRIC_REFERENCE_RE.finditer(expr):
        name = match.group(1)
        if name.startswith(_PROMQL_METRIC_PREFIXES):
            names.add(name)
    return names
