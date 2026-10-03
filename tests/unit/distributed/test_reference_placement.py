# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest


def test_added_reference_node_preserves_single_node_gpu_split():
    from nemo_rl.distributed.reference_placement import (
        ReferencePlacementConfig,
        plan_reference_placement,
    )

    plan = plan_reference_placement(
        total_nodes=2,
        gpus_per_node=8,
        teacher_nodes=0,
        generation_colocated=False,
        generation_nodes=1,
        generation_gpus_per_node=4,
        reference=ReferencePlacementConfig(num_nodes=1, gpus_per_node=4),
    )
    assert (plan.train_nodes, plan.train_gpus_per_node) == (1, 4)
    assert (plan.generation_nodes, plan.generation_gpus_per_node) == (1, 4)
    assert plan.reference_nodes == 1


def test_reference_shares_host_but_not_gpus():
    from nemo_rl.distributed.reference_placement import (
        ReferencePlacementConfig,
        plan_reference_placement,
    )

    plan = plan_reference_placement(
        total_nodes=1,
        gpus_per_node=8,
        teacher_nodes=0,
        generation_colocated=True,
        generation_nodes=None,
        generation_gpus_per_node=None,
        reference=ReferencePlacementConfig(
            placement="same_node", num_nodes=1, gpus_per_node=4
        ),
    )
    assert (plan.train_nodes, plan.train_gpus_per_node) == (1, 4)
    assert plan.generation_gpus_per_node == 4


def test_separate_reference_and_teachers_leave_fixed_student_budget():
    from nemo_rl.distributed.reference_placement import (
        ReferencePlacementConfig,
        plan_reference_placement,
    )

    plan = plan_reference_placement(
        total_nodes=5,
        gpus_per_node=4,
        teacher_nodes=2,
        generation_colocated=False,
        generation_nodes=1,
        generation_gpus_per_node=4,
        reference=ReferencePlacementConfig(num_nodes=1, gpus_per_node=4),
    )
    assert (plan.train_nodes, plan.generation_nodes, plan.reference_nodes) == (1, 1, 1)


@pytest.mark.parametrize("reference_gpus", [4, 8])
def test_same_node_split_requires_gpus_left_for_policy(reference_gpus):
    from nemo_rl.distributed.reference_placement import (
        ReferencePlacementConfig,
        plan_reference_placement,
    )

    with pytest.raises(ValueError, match="policy"):
        plan_reference_placement(
            total_nodes=1,
            gpus_per_node=8,
            teacher_nodes=0,
            generation_colocated=False,
            generation_nodes=1,
            generation_gpus_per_node=4,
            reference=ReferencePlacementConfig(
                placement="same_node", num_nodes=1, gpus_per_node=reference_gpus
            ),
        )


def test_misspelled_reference_setting_is_rejected():
    from pydantic import ValidationError

    from nemo_rl.distributed.reference_placement import ReferencePlacementConfig

    with pytest.raises(ValidationError):
        ReferencePlacementConfig(num_nodes=1, gpus_per_node=4, placment="same_node")
