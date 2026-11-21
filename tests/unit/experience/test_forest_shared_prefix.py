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
"""Shared-prefix row tags of forest-training groups, on CPU.

Uses real capture, staging and finalization, the driver's shared-prefix planner and
the controller's advantage stage. Generation and storage are doubles.
"""

import asyncio
from collections.abc import Iterator
from pathlib import Path

import pytest

pytest.importorskip("nemo_gym", reason="requires the paired Gym checkout")

from nemo_rl.algorithms.single_controller_utils.utils import tensor_field
from nemo_rl.data.packing.shared_prefix.plan import plan_shared_prefix_shards
from nemo_rl.data.packing.shared_prefix.row_metadata import (
    PROMPT_HASH_TAG,
    PROMPT_LENGTH_TAG,
    UNSHARED_ROW,
    prompt_hash,
)
from nemo_rl.data_plane.adapters.noop import NoOpDataPlaneClient
from nemo_rl.data_plane.grouping import row_group_ids
from nemo_rl.data_plane.tq_token_sink import TQTokenSink, TQTokenSource
from nemo_rl.experience.rollout_reassembler import RolloutReassembler
from nemo_rl.models.generation.capture_context import decide_capture_input
from tests.unit.experience.test_logical_owner_finalization import (
    capture_segment,
    finalize,
    gym_harness,
)
from tests.unit.single_controller.test_logical_advantage import _controller

pytestmark = pytest.mark.nemo_gym


@pytest.fixture
def forest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[tuple]:
    """Capture harness, data plane, finalizer and the root prompt a test may rewrite."""
    plane = NoOpDataPlaneClient()
    plane.register_partition("staged", [], 128, ["finalize"])
    plane.register_partition("train", [], 128, ["train"])
    source = TQTokenSource(plane, staging_partition="staged")
    root_prompt = [10, 11]
    harness = gym_harness.make_capture_harness(
        monkeypatch,
        tmp_path,
        sink=TQTokenSink(plane, staging_partition="staged"),
        fetch_prefix=lambda keys: source.fetch_prefix_chains(keys).expanded,
        root_prompt=root_prompt,
        decide_input=decide_capture_input,
    )
    finalizer = RolloutReassembler(
        plane,
        partition_id="train",
        staging_partition="staged",
        pad_token_id=0,
        max_seq_len=1024,
    )
    yield harness, plane, finalizer, root_prompt
    harness.client.close()
    asyncio.run(harness.ledger.close())
    plane.close()


def test_rows_with_the_group_prompt_share_it_and_each_rollout_keeps_one_advantage(
    forest: tuple,
) -> None:
    """One branch and one compaction: which rows share, and the advantages."""
    harness, plane, finalizer, root_prompt = forest
    # Rollout group_g0: a first call, two continuations of it (a branch), then a new
    # root with a rewritten prompt (a compaction). Rollout group_g1: one call.
    first = gym_harness.assert_clean(harness.post("group_g0", gym_harness.HISTORY))
    branches = [
        gym_harness.assert_clean(
            harness.post(
                "group_g0",
                gym_harness.HISTORY
                + first["output"]
                + [{"role": "user", "content": f"observation {index}"}],
            )
        )
        for index in range(2)
    ]
    root_prompt[:] = [12, 13]
    rewritten = gym_harness.assert_clean(harness.post("group_g0", gym_harness.HISTORY))
    root_prompt[:] = [10, 11]
    manifest = harness.manifest("group_g0")
    first_call = manifest.records[0].model_call_id
    assert [record.parent_call_id for record in manifest.records] == [
        None,
        first_call,
        first_call,
        None,
    ]
    branched = {
        "scope": "group_g0",
        "manifest": manifest,
        "ids": [reply["id"] for reply in (first, *branches, rewritten)],
    }
    sibling = capture_segment(harness, "group_g1")

    meta = finalize(finalizer, [[branched], [sibling]], execution_row_multiple=3).meta

    assert meta.sample_ids == [
        "group_g0_s0",  # first call and the first continuation
        "group_g0_s1",  # first call and the second continuation
        "group_g0_s2",  # the rewritten root
        "group_g1_s0",
        "group_pad0",
        "group_pad1",
    ]
    tags = meta.tags
    assert row_group_ids(meta) == ["group"] * 6
    assert [tag[PROMPT_LENGTH_TAG] for tag in tags] == [2, 2, 2, 2, 0, 0]
    group_prompt, rewritten_prompt, no_prompt = (
        prompt_hash(tokens) for tokens in ([10, 11], [12, 13], [])
    )
    assert [tag[PROMPT_HASH_TAG] for tag in tags] == [
        group_prompt,
        group_prompt,
        rewritten_prompt,
        group_prompt,
        no_prompt,
        no_prompt,
    ]
    data = plane.get_samples(meta.sample_ids, "train", meta.fields)
    token_mask = tensor_field(data, "token_mask").bool()
    # No prompt token that a row may share carries loss.
    assert not token_mask[:4, :2].any()

    [shard] = plan_shared_prefix_shards(
        sequence_lengths=meta.sequence_lengths,
        group_ids=row_group_ids(meta),
        prompt_lengths=[tag[PROMPT_LENGTH_TAG] for tag in tags],
        prompt_hashes=[tag[PROMPT_HASH_TAG] for tag in tags],
        num_shards=1,
        batch_size=None,
        max_tokens_per_microbatch=64,
        sequence_length_pad_multiple=1,
        algorithm="modified_first_fit_decreasing",
        bound_by_full_length=False,
        largest_first=False,
    )
    tree_ids = dict(zip(shard.row_indices, shard.tree_ids, strict=True))
    # The rows that start with the group's prompt form one tree, across both rollouts.
    # The rewritten root and the padding rows run as ordinary packed sequences.
    assert [tree_ids[row] for row in range(6)] == [
        0,
        0,
        UNSHARED_ROW,
        0,
        UNSHARED_ROW,
        UNSHARED_ROW,
    ]

    _, has_training_tokens = asyncio.run(_controller(meta, data)._advantage_stage(meta))
    assert has_training_tokens
    advantages = tensor_field(data, "advantages")
    # Rewards 1 and 2 give the rollouts -0.5 and +0.5. The three rows of group_g0 carry
    # its one value, and the padding rows carry no loss.
    assert [advantages[row][token_mask[row]].unique().tolist() for row in range(6)] == [
        [-0.5],
        [-0.5],
        [-0.5],
        [0.5],
        [],
        [],
    ]
    assert data["sample_mask"].flatten().tolist() == [1, 1, 1, 1, 0, 0]


def test_a_rejected_rollout_and_padding_rows_do_not_share(forest: tuple) -> None:
    """Rows that publish a copy of a verified row's tokens get prompt length 0."""
    harness, plane, finalizer, _ = forest
    verified = [capture_segment(harness, "group_g0") for _ in range(3)]
    rejected = capture_segment(harness, "group_g1", child=True)
    # Corrupt a committed call, so that rollout group_g1 fails verification.
    key = rejected["manifest"].records[-1].staging_key
    plane._partitions["staged"].rows[key]["digest_bytes"][0] ^= 1

    meta = finalize(finalizer, [verified, [rejected]], execution_row_multiple=3).meta

    assert [tag["uses_borrowed_input"] for tag in meta.tags] == [False] * 3 + [True] * 3
    input_ids = tensor_field(
        plane.get_samples(meta.sample_ids, "train", ["input_ids"]), "input_ids"
    )
    # Every published row starts with the group's prompt, borrowed or not.
    assert input_ids[:, :2].tolist() == [[10, 11]] * 6
    # Only the three rows of the verified rollout may share it.
    assert [tag[PROMPT_LENGTH_TAG] for tag in meta.tags] == [2, 2, 2, 0, 0, 0]
    assert {tag[PROMPT_HASH_TAG] for tag in meta.tags[:3]} == {prompt_hash([10, 11])}
    assert {tag[PROMPT_HASH_TAG] for tag in meta.tags[3:]} == {prompt_hash([])}
