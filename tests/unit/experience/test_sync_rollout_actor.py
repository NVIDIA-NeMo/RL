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
"""Sync rollout metadata names prompt groups independently of their tokens."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.grouping import group_index_column, row_group_ids
from nemo_rl.data_plane.schema import GROUP_ID_TAG
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.experience import sync_rollout_actor as actor_mod


@pytest.mark.parametrize("multimodal", [False, True])
def test_actor_stamps_distinct_groups_and_preserves_multimodal_tags(
    monkeypatch, multimodal
):
    cls = actor_mod.SyncRolloutActor.__ray_metadata__.modified_class
    actor = object.__new__(cls)
    actor.policy_generation = MagicMock()
    actor.tokenizer = SimpleNamespace(pad_token_id=0)
    actor.task_to_env = {}
    actor._dp_client = MagicMock()
    actor.master_config = SimpleNamespace(
        policy={
            "max_total_sequence_length": 32,
            "make_sequence_length_divisible_by": 1,
            "precision": "float32",
            "generation": {},
        },
        grpo=SimpleNamespace(max_rollout_turns=1, deduplicate_multimodal_data=False),
    )
    batch = BatchedDataDict(
        {
            "message_log": [
                [
                    {
                        "role": "user",
                        "content": "same prompt",
                        "token_ids": torch.tensor([1, 2]),
                    },
                    {
                        "role": "assistant",
                        "content": "answer",
                        "token_ids": torch.tensor([3]),
                        "generation_logprobs": torch.tensor([0.1]),
                    },
                ]
                for _ in range(4)
            ],
            "total_reward": torch.tensor([0.0, 2.0, 10.0, 14.0]),
            "loss_multiplier": torch.ones(4),
            "truncated": torch.zeros(4, dtype=torch.bool),
            "length": torch.full((4,), 2),
        }
    )
    monkeypatch.setattr(
        "nemo_rl.environments.nemo_gym.should_use_nemo_gym", lambda _: False
    )
    monkeypatch.setattr(
        "nemo_rl.models.generation.interfaces.should_use_async_rollouts",
        lambda _: False,
    )
    monkeypatch.setattr(
        actor_mod, "run_multi_turn_rollout", lambda **kwargs: (batch, {})
    )
    geometry = [{"pixel_values_row_shapes": {"shapes": [(2, 3)]}} for _ in range(4)]
    monkeypatch.setattr(
        actor_mod, "multimodal_row_tags", lambda *args: geometry if multimodal else None
    )

    def write(bulk, **kwargs):
        return KVBatchMeta(
            sample_ids=kwargs["sample_ids"],
            partition_id=kwargs["partition_id"],
            task_name=kwargs["task_name"],
            fields=list(bulk),
            tags=kwargs["tags"],
        )

    monkeypatch.setattr(actor_mod, "kv_first_write", write)
    meta, carry, _, _ = actor.rollout_to_tq(batch, partition_id="train", group_size=2)
    groups = row_group_ids(meta)
    assert groups[0] == groups[1]
    assert groups[2] == groups[3]
    assert groups[0] != groups[2]
    assert meta.sample_ids == [
        f"{groups[0]}_g0",
        f"{groups[0]}_g1",
        f"{groups[2]}_g0",
        f"{groups[2]}_g1",
    ]
    assert group_index_column(groups).squeeze(1).tolist() == [0, 0, 1, 1]
    assert "prompt_ids_for_adv" not in carry
    assert "prompt_ids_for_adv" not in meta.fields
    for tag in meta.tags:
        assert GROUP_ID_TAG in tag
        if multimodal:
            assert tag["pixel_values_row_shapes"] == {"shapes": [(2, 3)]}


def test_empty_group_index_column_has_row_key_shape():
    assert group_index_column([]).shape == (0, 1)
    assert group_index_column([]).dtype == torch.long
