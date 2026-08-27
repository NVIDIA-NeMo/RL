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

"""Translation from NeMo RL refit metadata into a ModelExpress plan.

The failure this guards is quiet: a mesh translated wrongly still produces a
plan that validates, and the collective still runs. It just moves the wrong
bytes. So the conversions are pinned, and the cases that cannot be expressed
are required to raise rather than approximate.
"""

import pytest
import torch
from torch.distributed._tensor import Shard
from torch.distributed.tensor.placement_types import Replicate

from nemo_rl.weight_sync.mx_collective_plan import (
    MxPlanTranslationError,
    build_mx_plan,
    dtype_name,
    mesh_to_spec,
    placements_to_mx,
    slot_ids,
)
from nemo_rl.weight_sync.nccl_reshard_utils import MeshInfo

modelexpress_rl = pytest.importorskip(
    "modelexpress_rl.collective",
    reason="the ModelExpress collective client is an optional dependency",
)


class TestMeshTranslation:
    def test_a_one_dimensional_mesh_keeps_its_offset(self):
        mesh = MeshInfo(torch.arange(4, 8))
        assert mesh_to_spec(mesh) == ((4,), 4)

    def test_a_two_dimensional_mesh_keeps_its_shape(self):
        # build_mesh_info reshapes arange, so the grid is row-major contiguous.
        mesh = MeshInfo(torch.arange(0, 8).reshape(2, 4))
        assert mesh_to_spec(mesh) == ((2, 4), 0)

    def test_a_generator_mesh_offset_by_the_trainer_ranks(self):
        mesh = MeshInfo(torch.arange(2, 6).reshape(2, 2))
        assert mesh_to_spec(mesh) == ((2, 2), 2)

    def test_a_non_contiguous_mesh_is_rejected_not_flattened(self):
        # MX describes a mesh as shape plus starting rank. A grid that is not
        # contiguous cannot be expressed that way, and silently accepting it
        # would describe a different topology than the one that exists.
        mesh = MeshInfo(torch.tensor([0, 2, 4, 6]))
        with pytest.raises(MxPlanTranslationError, match="contiguous and ascending"):
            mesh_to_spec(mesh)

    def test_a_descending_mesh_is_rejected(self):
        mesh = MeshInfo(torch.tensor([3, 2, 1, 0]))
        with pytest.raises(MxPlanTranslationError, match="contiguous and ascending"):
            mesh_to_spec(mesh)

    def test_an_empty_mesh_is_rejected(self):
        with pytest.raises(MxPlanTranslationError, match="no ranks"):
            mesh_to_spec(MeshInfo(torch.tensor([], dtype=torch.int64)))

    def test_a_mesh_without_a_grid_is_rejected(self):
        class Bare:
            pass

        with pytest.raises(MxPlanTranslationError, match="does not expose"):
            mesh_to_spec(Bare())


class TestPlacementTranslation:
    def test_shard_and_replicate_round_trip(self):
        converted = placements_to_mx([Replicate(), Shard(1)])
        assert converted[0].canonical() == "R"
        assert converted[1].canonical() == "S1"

    def test_msgspec_flattened_placements_are_understood(self):
        # vLLM's collective_rpc serializes Shard(N) to {"dim": N}, so the
        # metadata can arrive in that form on the generation side.
        converted = placements_to_mx([{"dim": 2}, {}])
        assert converted[0].canonical() == "S2"
        assert converted[1].canonical() == "R"


class TestDtypeNames:
    @pytest.mark.parametrize(
        ("dtype", "expected"),
        [
            (torch.bfloat16, "bfloat16"),
            (torch.float16, "float16"),
            (torch.float32, "float32"),
            (torch.float8_e4m3fn, "float8_e4m3fn"),
        ],
    )
    def test_known_dtypes_get_a_canonical_name(self, dtype, expected):
        assert dtype_name(dtype) == expected

    def test_a_string_dtype_is_normalized(self):
        assert dtype_name("torch.bfloat16") == "bfloat16"
        assert dtype_name("bfloat16") == "bfloat16"

    def test_an_unknown_dtype_raises_rather_than_guessing(self):
        # Both sides hash the dtype into the plan digest, so two workers
        # spelling it differently would never form a group.
        with pytest.raises(MxPlanTranslationError, match="no canonical name"):
            dtype_name(torch.int8)


def refit_info(pp_size=1):
    src = MeshInfo(torch.arange(0, 2))
    dst = MeshInfo(torch.arange(2, 6))
    return {
        "layer_names": ["model.layers.0"],
        "per_layer_params": {
            "model.layers.0": [
                {
                    "name": "model.layers.0.mlp.gate_proj.weight",
                    "global_shape": (8, 4),
                    "dtype": torch.bfloat16,
                    "pp_stage": 0,
                    "src_mesh_info": src,
                    "src_placements": [Shard(0)],
                    "dst_mesh_info": dst,
                    "dst_placements": [Shard(0)],
                }
            ]
        },
        "pp_size": pp_size,
    }


class TestPlanConstruction:
    def test_bulk_entries_carry_over_with_their_geometry(self):
        plan = build_mx_plan(refit_info())
        assert len(plan.bulk) == 1
        entry = plan.bulk[0]
        assert entry.name == "model.layers.0.mlp.gate_proj.weight"
        assert entry.global_shape == (8, 4)
        assert entry.dtype == "bfloat16"
        assert entry.src_mesh.rank_offset == 0
        assert entry.dst_mesh.rank_offset == 2
        assert entry.src_placements[0].canonical() == "S0"

    def test_the_pipeline_stage_becomes_the_source_partition(self):
        # MX routes a parameter to its partition's reshard lane, and the
        # pipeline stage is what partitions the trainer here.
        plan = build_mx_plan(refit_info(pp_size=2))
        assert plan.source_partition_count == 2
        assert plan.bulk[0].partition_id == 0

    def test_the_grouped_expert_tag_is_preserved(self):
        info = refit_info()
        info["per_layer_params"]["model.layers.0"][0]["grouped_expert_proj"] = "gate_proj"
        plan = build_mx_plan(info)
        assert plan.bulk[0].group_key == "gate_proj"

    def test_misc_order_is_preserved(self):
        # The misc list order is the broadcast payload layout and MX folds it
        # into the plan digest, so reordering it is a different plan.
        misc = {
            "model.embed_tokens.weight": {"shape": (16, 4), "dtype": torch.bfloat16},
            "model.norm.weight": {"shape": (4,), "dtype": torch.bfloat16},
        }
        plan = build_mx_plan(refit_info(), misc)
        assert [m.name for m in plan.misc] == list(misc)

    def test_a_plan_with_no_misc_parameters_is_valid(self):
        plan = build_mx_plan(refit_info())
        assert plan.misc == []

    def test_the_translated_plan_digests_deterministically(self):
        from modelexpress_rl.collective import plan_digest

        assert plan_digest(build_mx_plan(refit_info())) == plan_digest(
            build_mx_plan(refit_info())
        )


class TestSlotIds:
    def test_slots_are_stable_and_distinct_across_roles(self):
        trainers, generators = slot_ids(2, 3)
        assert trainers == ["train/0", "train/1"]
        assert generators == ["gen/0", "gen/1", "gen/2"]
        assert not set(trainers) & set(generators)
