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

"""Artificial input provenance is independent of loss filtering and row order."""

from unittest.mock import patch

import torch
from tensordict import TensorDict

from nemo_rl.data_plane.codec import materialize
from nemo_rl.data_plane.preshard import shard_meta_for_dp
from nemo_rl.experience.rollout_reassembler import FinalizedRollout, RolloutReassembler


class CaptureStore:
    def put_samples(self, **kwargs):
        self.published = kwargs


def test_placeholder_provenance_survives_publication_reordering_and_sharding():
    store = CaptureStore()
    publisher = RolloutReassembler(
        store,
        partition_id="train",
        staging_partition="stage",
        pad_token_id=0,
        max_seq_len=16,
    )
    rows = [
        FinalizedRollout(
            "group_g0",
            True,
            None,
            [0, 11, 12],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -0.2],
            2,
            1.0,
            [],
        ),
        FinalizedRollout("group_g1", False, "missing_receipt", [], [], [], 0, 0.0, []),
    ]
    with patch.object(publisher, "finalize_rollout", side_effect=rows):
        result = publisher.finalize_group(
            "group",
            [row.rollout_id for row in rows],
            [None, None],
            [1.0, 0.0],
            mask_sample=[True, False],
            fallback_weight_version=0,
            prompt_idx=0,
        )
    assert result.meta.sequence_lengths == [3, 1]
    assert [tag["is_artificial_input"] for tag in result.meta.tags] == [False, True]
    meta = result.meta.subset([1, 0])
    shards, _ = shard_meta_for_dp(meta, dp_world=2)
    fields = store.published["fields"]
    for shard, original in zip(shards, [1, 0], strict=True):
        # Model the storage fetch by sample identity, preserving the wire layout.
        td = TensorDict(
            {
                key: torch.nested.as_nested_tensor(
                    [value[original]], layout=torch.jagged
                )
                if value.is_nested
                else value[original : original + 1]
                for key, value in fields.items()
            },
            batch_size=[1],
        )
        batch = materialize(td, tags=shard.tags)
        assert batch["is_artificial_input"].tolist() == [original == 1]
        assert batch["input_lengths"].tolist() == [[3, 1][original]]
    # A genuine row flagged for loss filtering, containing pad-ID and context
    # tokens, remains genuine. No inference from either token or loss masks.
    assert fields["mask_sample"].tolist() == [True, False]


def test_legacy_payload_does_not_invent_provenance():
    td = TensorDict({"input_ids": torch.tensor([[0, 1]])}, batch_size=[1])
    assert "is_artificial_input" not in materialize(td)
