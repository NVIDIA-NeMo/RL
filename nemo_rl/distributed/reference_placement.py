# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resource budgets for a separate frozen reference policy."""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, PositiveInt


class ReferencePlacementConfig(BaseModel, extra="forbid"):
    """Omit this section to keep the reference inside the policy workers.

    ``num_nodes`` is the reference host count; ``gpus_per_node`` is its GPU
    allocation per host. ``same_node`` splits GPUs on each training host;
    ``separate_nodes`` reserves additional hosts from the total node budget.
    """

    placement: Literal["same_node", "separate_nodes"] = "separate_nodes"
    num_nodes: PositiveInt
    gpus_per_node: PositiveInt


@dataclass(frozen=True)
class ReferencePlacementPlan:
    train_nodes: int
    train_gpus_per_node: int
    generation_nodes: int
    generation_gpus_per_node: int
    reference_nodes: int


def plan_reference_placement(
    *,
    total_nodes: int,
    gpus_per_node: int,
    teacher_nodes: int,
    generation_colocated: bool,
    generation_nodes: int | None,
    generation_gpus_per_node: int | None,
    reference: ReferencePlacementConfig,
) -> ReferencePlacementPlan:
    """Preserve the student budget after reserving separate reference hosts."""
    if reference.gpus_per_node > gpus_per_node:
        raise ValueError("reference.gpus_per_node exceeds cluster.gpus_per_node")
    student_nodes = total_nodes - teacher_nodes
    if reference.placement == "separate_nodes":
        student_nodes -= reference.num_nodes
    if student_nodes <= 0:
        raise ValueError("Reference and teachers leave no nodes for the policy")

    train_nodes = student_nodes
    train_gpus = gpus_per_node
    if generation_colocated:
        inference_nodes = train_nodes
        inference_gpus = train_gpus
    else:
        if generation_gpus_per_node is None or generation_gpus_per_node <= 0:
            raise ValueError("Separate generation requires a positive GPU allocation")
        if generation_gpus_per_node > gpus_per_node:
            raise ValueError("Generation GPUs exceed cluster.gpus_per_node")
        inference_nodes = generation_nodes or 1
        inference_gpus = generation_gpus_per_node
        if student_nodes == 1:
            if inference_nodes != 1:
                raise ValueError("A single student host requires one generation host")
            train_gpus -= inference_gpus
        else:
            train_nodes -= inference_nodes

    if reference.placement == "same_node":
        if reference.num_nodes != train_nodes:
            raise ValueError(
                "same_node reference.num_nodes must match the policy host count"
            )
        train_gpus -= reference.gpus_per_node
        if generation_colocated:
            inference_gpus = train_gpus
    if train_nodes <= 0 or train_gpus <= 0:
        raise ValueError(
            "Reference/generation allocation leaves no GPUs for the policy"
        )

    return ReferencePlacementPlan(
        train_nodes=train_nodes,
        train_gpus_per_node=train_gpus,
        generation_nodes=inference_nodes,
        generation_gpus_per_node=inference_gpus,
        reference_nodes=reference.num_nodes,
    )
