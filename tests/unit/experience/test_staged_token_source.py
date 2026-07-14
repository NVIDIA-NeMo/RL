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

import pytest

from nemo_rl.experience.rollout_writer import build_staging_delta
from nemo_rl.experience.staged_token_source import (
    StagedCallSnapshot,
    StagedSnapshotTokenSource,
)

RID = "8f93c10ac6a347a99c9b6b8d5f21e402"


def _snapshot_from_worker_delta(
    call_id: str,
    *,
    prompt: list[int],
    generated: list[int],
    logprobs: list[float],
    prev_len: int,
) -> StagedCallSnapshot:
    """Build the snapshot exactly as the worker's write path stages it."""
    ids, mask, lps = build_staging_delta(
        prompt_token_ids=prompt,
        generated_token_ids=generated,
        generated_logprobs=logprobs,
        prev_len=prev_len,
    )
    return StagedCallSnapshot(
        call_id=call_id,
        prev_len=prev_len,
        token_ids_delta=ids,
        token_mask_delta=mask,
        logprobs_delta=lps,
        weight_version=41,
    )


def test_round_trips_worker_deltas_to_token_entries() -> None:
    """The doc's two-turn worked example: staging deltas reconstruct each
    call's full prompt/generation/logprobs exactly."""
    snapshots = [
        _snapshot_from_worker_delta(
            "c1",
            prompt=[101, 102, 103],
            generated=[201, 202],
            logprobs=[-0.10, -0.20],
            prev_len=0,
        ),
        _snapshot_from_worker_delta(
            "c2",
            prompt=[101, 102, 103, 201, 202, 301, 302],
            generated=[401, 402],
            logprobs=[-0.05, -0.08],
            prev_len=5,
        ),
    ]
    source = StagedSnapshotTokenSource(RID, snapshots, model="policy")
    entries = source.entries(RID)

    assert [e.request_id for e in entries] == ["c1", "c2"]
    assert entries[0].prompt_token_ids == [101, 102, 103]
    assert entries[0].generation_token_ids == [201, 202]
    assert entries[0].generation_log_probs == [-0.10, -0.20]
    assert entries[1].prompt_token_ids == [101, 102, 103, 201, 202, 301, 302]
    assert entries[1].generation_token_ids == [401, 402]
    assert entries[1].generation_log_probs == [-0.05, -0.08]
    assert all(e.weight_version == 41 for e in entries)


def test_rejects_wrong_rollout_and_inconsistent_deltas() -> None:
    snapshot = _snapshot_from_worker_delta(
        "c1", prompt=[1, 2], generated=[3], logprobs=[-0.1], prev_len=0
    )
    source = StagedSnapshotTokenSource(RID, [snapshot])
    with pytest.raises(ValueError, match="asked for"):
        source.entries("other")

    over_extended = StagedCallSnapshot(
        call_id="c2",
        prev_len=99,
        token_ids_delta=[4],
        token_mask_delta=[1.0],
        logprobs_delta=[-0.2],
    )
    with pytest.raises(ValueError, match="prev_len"):
        StagedSnapshotTokenSource(RID, [snapshot, over_extended]).entries(RID)

    interleaved_mask = StagedCallSnapshot(
        call_id="c3",
        prev_len=0,
        token_ids_delta=[1, 2, 3],
        token_mask_delta=[0.0, 1.0, 0.0],
        logprobs_delta=[0.0, -0.1, 0.0],
    )
    with pytest.raises(ValueError, match="prompt-carry prefix"):
        StagedSnapshotTokenSource(RID, [interleaved_mask]).entries(RID)


@pytest.mark.nemo_gym
def test_gym_builder_assembles_staged_snapshot_end_to_end() -> None:
    """The full P2 contract: Gym's pure prefix_merging builder consumes the
    TQ-backed token source, and the projected response satisfies the vendored
    NeMo-RL contiguity contract."""
    from nemo_gym.trajectory.builder import (
        assert_nemo_rl_contiguity,
        build_trajectories,
        project_main_chain_response,
    )
    from nemo_gym.trajectory.registry import get_builder

    snapshots = [
        _snapshot_from_worker_delta(
            "c1",
            prompt=[101, 102, 103],
            generated=[201, 202],
            logprobs=[-0.10, -0.20],
            prev_len=0,
        ),
        _snapshot_from_worker_delta(
            "c2",
            prompt=[101, 102, 103, 201, 202, 301, 302],
            generated=[401, 402],
            logprobs=[-0.05, -0.08],
            prev_len=5,
        ),
    ]
    source = StagedSnapshotTokenSource(RID, snapshots, model="policy")
    trajectories = build_trajectories(
        RID, source, builder="prefix_merging", reward=1.0, policy_model="policy"
    )
    assert len(trajectories) == 1
    main = trajectories[0]
    assert main.token_ids == [101, 102, 103, 201, 202, 301, 302, 401, 402]
    assert main.loss_mask == [0, 0, 0, 1, 1, 0, 0, 1, 1]
    assert main.reward == 1.0
    # PG loss confined to the 4 sampled tokens, staged logprobs intact.
    sampled = [lp for lp, m in zip(main.logprobs, main.loss_mask) if m == 1]
    assert sampled == [-0.10, -0.20, -0.05, -0.08]

    chains = get_builder("prefix_merging")(source.entries(RID)).chains
    projection = project_main_chain_response(RID, chains, {}, model="policy")
    assert_nemo_rl_contiguity(projection)
