# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Print actual host/GPU/actor assignments with exact repeated-row compression."""

import re
from collections import defaultdict
from dataclasses import dataclass
from fnmatch import fnmatchcase
from itertools import groupby


@dataclass
class NodePlacement:
    node_id: str
    hostname: str
    domain: str
    topo_rank: int
    gpu_ids: tuple[int, ...]
    gpu_roles: dict[int, tuple[str, ...]]
    cpu_actors: tuple[str, ...]
    advertised_gpus: int | None = None


def _host_ranges(hostnames: list[str]) -> str:
    numbered: dict[tuple[str, str, int], list[int]] = defaultdict(list)
    names = []
    for host in hostnames:
        match = re.fullmatch(r"(.*?)([0-9]+)([^0-9]*)", host)
        if match is None:
            names.append(host)
        else:
            prefix, number, suffix = match.groups()
            numbered[(prefix, suffix, len(number))].append(int(number))
    for (prefix, suffix, width), numbers in sorted(numbered.items()):
        for _, group in groupby(
            enumerate(sorted(set(numbers))), lambda item: item[1] - item[0]
        ):
            values = [value for _, value in group]
            first, last = values[0], values[-1]
            if first == last:
                names.append(f"{prefix}{first:0{width}d}{suffix}")
            else:
                names.append(f"{prefix}[{first:0{width}d}-{last:0{width}d}]{suffix}")
    return ", ".join(sorted(names))


def render_placement(
    nodes: list[NodePlacement],
    *,
    teachers: dict[str, tuple[tuple[str, ...], str]] | None = None,
    full: bool = False,
    host_filter: str | None = None,
    domain_filter: str | None = None,
) -> str:
    """Render a snapshot; filters use shell patterns and totals cover matching hosts."""
    selected = sorted(
        (
            node
            for node in nodes
            if (host_filter is None or fnmatchcase(node.hostname, host_filter))
            and (domain_filter is None or fnmatchcase(node.domain, domain_filter))
        ),
        key=lambda node: (node.domain, node.topo_rank, node.hostname),
    )
    gpu_ids = sorted({gpu for node in selected for gpu in node.gpu_ids})
    header = ["NVLink domain", "Hosts", "Ray GPUs"]
    header.extend(f"GPU{gpu}" for gpu in gpu_ids)
    header.append("CPU actors")
    groups: dict[tuple[str, ...], list[str]] = {}
    for node in selected:
        cells = tuple(
            "+".join(node.gpu_roles[gpu])
            if gpu in node.gpu_roles
            else "."
            if gpu in node.gpu_ids
            else ""
            for gpu in gpu_ids
        )
        cpu = ", ".join(sorted(node.cpu_actors))
        capacity = (
            node.advertised_gpus
            if node.advertised_gpus is not None
            else len(node.gpu_ids)
        )
        key = (node.domain, str(capacity), *cells, cpu)
        if full:
            key = (*key, node.node_id)
        groups.setdefault(key, []).append(node.hostname)
    rows = [header]
    # Bound the default log view; full/filtered views expose irregular fleets.
    visible_groups = list(groups.items()) if full else list(groups.items())[:20]
    for key, hosts in visible_groups:
        domain, gpu_count, *cells = key[:-1] if full else key
        rows.append([domain, _host_ranges(hosts), gpu_count, *cells])
    widths = [max(len(row[i]) for row in rows) for i in range(len(header))]

    def line(row: list[str]) -> str:
        return (
            "| "
            + " | ".join(cell.ljust(width) for cell, width in zip(row, widths))
            + " |"
        )

    output = [
        "=== Actor placement ===",
        "Roles: P=policy, R=reference, G=generation, V=critic, Tn=teacher",
        "Cells: +=roles share a GPU, .=unused advertised GPU, blank=not advertised",
        "GPU IDs are host-local. Host ranges are inclusive and describe identical rows.",
        "CPU actors include all Ray jobs; [job=ID] marks actors from another job.",
        line(header),
        "| " + " | ".join("-" * width for width in widths) + " |",
        *(line(row) for row in rows[1:]),
    ]
    if teachers:
        output.append("Teachers:")
        for label, (aliases, checkpoint) in teachers.items():
            output.append(
                f"  {label}: aliases=[{', '.join(aliases)}], checkpoint={checkpoint}"
            )
    if len(visible_groups) < len(groups):
        output.append(
            f"{len(groups) - len(visible_groups)} more layouts. Use --placement-full, "
            "--placement-host PATTERN, or --placement-domain PATTERN to inspect them."
        )
    assigned = sum(len(node.gpu_roles) for node in selected)
    unused = (
        sum(
            node.advertised_gpus
            if node.advertised_gpus is not None
            else len(node.gpu_ids)
            for node in selected
        )
        - assigned
    )
    missing_inventory = any(
        node.advertised_gpus is not None and len(node.gpu_ids) != node.advertised_gpus
        for node in selected
    )
    if missing_inventory:
        output.append(
            "GPU inventory incomplete: unassigned device IDs are unavailable on some hosts."
        )
    output.extend(
        [
            f"Total: {len(selected)} hosts, {assigned} assigned GPUs, {unused} unused GPUs",
            "Shared GPUs count once. Totals cover all hosts matching the filters.",
            'Domain "host-local" means each host has its own domain; "unknown" means unavailable.',
            "=== End actor placement ===",
        ]
    )
    return "\n".join(output)
