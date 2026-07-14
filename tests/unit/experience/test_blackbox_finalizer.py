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
import torch
from tensordict import TensorDict

from nemo_rl.data_plane.adapters.noop import NoOpDataPlaneClient
from nemo_rl.data_plane.schema import ROLLOUT_STAGING_FIELDS
from nemo_rl.experience.blackbox_finalizer import finalize_blackbox_rollout
from nemo_rl.experience.rollout_writer import (
    EMPTY_PREFIX_HASH,
    ForestCursorStateMachine,
    build_staging_delta,
    compute_staging_digest,
    hash_token_ids,
)

pytestmark = pytest.mark.nemo_gym

RID = "8f93c10ac6a347a99c9b6b8d5f21e402"
WV = 41


def _stage_call(machine, client, call_id, *, prompt, generated, logprobs, wv=WV):
    """Mirror the worker's forest write path exactly."""
    parent, prev_len, prev_hash = None, 0, EMPTY_PREFIX_HASH
    for candidate in machine.get_candidates(RID):
        if candidate.length <= len(prompt) and (
            hash_token_ids(prompt[: candidate.length]) == candidate.sequence_hash
        ):
            parent = candidate.call_id
            prev_len, prev_hash = candidate.length, candidate.sequence_hash
            break
    reservation = machine.reserve_call(
        RID, call_id, parent_call_id=parent, prev_len=prev_len, prev_hash=prev_hash
    )
    ids, mask, lps = build_staging_delta(
        prompt_token_ids=prompt,
        generated_token_ids=generated,
        generated_logprobs=logprobs,
        prev_len=prev_len,
    )
    digest = compute_staging_digest(
        rollout_id=RID,
        call_id=call_id,
        prev_len=prev_len,
        token_ids_delta=ids,
        token_mask_delta=mask,
        logprobs_delta=lps,
    )
    key = f"{RID}/{call_id}"
    client.put_samples(
        sample_ids=[key],
        partition_id="staging",
        fields=TensorDict(
            {
                "token_ids_delta": torch.tensor([ids], dtype=torch.int64),
                "token_mask_delta": torch.tensor([mask], dtype=torch.float32),
                "generation_logprobs_delta": torch.tensor([lps], dtype=torch.float32),
            },
            batch_size=[1],
        ),
        tags=[{}],
    )
    full = prompt + generated
    machine.commit_call(
        RID,
        call_id,
        reservation.lease,
        staging_key=key,
        new_len=len(full),
        new_hash=hash_token_ids(full),
        prompt_len=len(prompt),
        gen_len=len(generated),
        digest=digest,
        weight_version=wv,
    )


def _receipt(call_ids, *, reward=1.0, status="completed"):
    from nemo_gym.observability.integration import (
        CallManifestEntry,
        RolloutCallManifest,
        RolloutReceipt,
    )

    entries = [
        CallManifestEntry(
            rollout_id=RID, call_id=c, admission_index=i, stage_disposition="staged"
        )
        for i, c in enumerate(call_ids)
    ]
    return RolloutReceipt(
        rollout_id=RID,
        manifest=RolloutCallManifest(rollout_id=RID, entries=entries),
        reward=reward,
        status=status,
    ).model_dump()


def _fixture():
    machine = ForestCursorStateMachine(lease_ttl_s=10, cursor_ttl_s=100)
    client = NoOpDataPlaneClient()
    client.register_partition(
        "staging", list(ROLLOUT_STAGING_FIELDS), 16, consumer_tasks=[]
    )
    return machine, client


def _finalize(machine, client, receipt, **overrides):
    kwargs = dict(
        dp_client=client,
        staging_partition="staging",
        receipt=receipt,
        forest_manifest=machine.get_forest_manifest(RID),
        expected_weight_version=WV,
    )
    kwargs.update(overrides)
    return finalize_blackbox_rollout(**kwargs)


def test_finalizes_the_worked_example_into_one_canonical_row() -> None:
    machine, client = _fixture()
    _stage_call(
        machine,
        client,
        "c1",
        prompt=[101, 102, 103],
        generated=[201, 202],
        logprobs=[-0.10, -0.20],
    )
    _stage_call(
        machine,
        client,
        "c2",
        prompt=[101, 102, 103, 201, 202, 301, 302],
        generated=[401, 402],
        logprobs=[-0.05, -0.08],
    )

    row = _finalize(machine, client, _receipt(["c1", "c2"]))

    assert row.valid and row.rejection_reason is None
    assert row.input_ids == [101, 102, 103, 201, 202, 301, 302, 401, 402]
    assert row.token_mask == [0, 0, 0, 1, 1, 0, 0, 1, 1]
    # Staged logprobs on sampled positions (float32 storage); 0.0 elsewhere.
    assert row.generation_logprobs[3:5] == pytest.approx([-0.10, -0.20])
    assert row.generation_logprobs[7:9] == pytest.approx([-0.05, -0.08])
    assert row.generation_logprobs[:3] == [0.0, 0.0, 0.0]
    assert row.reward == 1.0
    assert row.staging_keys == (f"{RID}/c1", f"{RID}/c2")


def test_branch_or_compaction_yields_masked_placeholder() -> None:
    machine, client = _fixture()
    _stage_call(machine, client, "c1", prompt=[1, 2], generated=[3], logprobs=[-0.1])
    # Second child of c1's committed prefix -> branch.
    _stage_call(
        machine, client, "c2", prompt=[1, 2, 3, 4], generated=[5], logprobs=[-0.2]
    )
    _stage_call(
        machine, client, "c3", prompt=[1, 2, 3, 9], generated=[8], logprobs=[-0.3]
    )

    row = _finalize(machine, client, _receipt(["c1", "c2", "c3"]))
    assert not row.valid
    assert row.rejection_reason.startswith("ambiguous_forest")


@pytest.mark.parametrize(
    "mutation, reason_prefix",
    [
        ("missing_row", "missing_staging_row"),
        ("orphan_node", "orphan_staged_node"),
        ("tampered_row", "staging_digest_mismatch"),
        ("uncollected", "uncollected_call"),
        ("bad_status", "receipt_status"),
        ("wrong_weight_version", "weight_version_mismatch"),
    ],
)
def test_failure_injection_invalidates_rollout(mutation, reason_prefix) -> None:
    machine, client = _fixture()
    _stage_call(machine, client, "c1", prompt=[1, 2], generated=[3], logprobs=[-0.1])
    _stage_call(
        machine, client, "c2", prompt=[1, 2, 3, 4], generated=[5], logprobs=[-0.2]
    )
    receipt = _receipt(["c1", "c2"])
    expected_weight_version = WV

    if mutation == "missing_row":
        client.clear_samples([f"{RID}/c2"], "staging")
    elif mutation == "orphan_node":
        receipt = _receipt(["c1"])  # c2 committed in TQ but not in the manifest
    elif mutation == "tampered_row":
        client.put_samples(
            sample_ids=[f"{RID}/c2"],
            partition_id="staging",
            fields=TensorDict(
                {
                    "token_ids_delta": torch.tensor([[4, 6]], dtype=torch.int64),
                    "token_mask_delta": torch.tensor([[0.0, 1.0]], dtype=torch.float32),
                    "generation_logprobs_delta": torch.tensor(
                        [[0.0, -0.2]], dtype=torch.float32
                    ),
                },
                batch_size=[1],
            ),
            tags=[{}],
        )
    elif mutation == "uncollected":
        # A third admitted call whose node never commits (harness killed mid-call).
        machine.reserve_call(
            RID,
            "c3",
            parent_call_id="c2",
            prev_len=5,
            prev_hash=hash_token_ids([1, 2, 3, 4, 5]),
        )
        receipt = _receipt(["c1", "c2", "c3"])
    elif mutation == "bad_status":
        receipt = _receipt(["c1", "c2"], status="harness_error")
    elif mutation == "wrong_weight_version":
        expected_weight_version = WV + 1

    row = _finalize(
        machine,
        client,
        receipt,
        expected_weight_version=expected_weight_version,
    )
    assert not row.valid
    assert row.rejection_reason.startswith(reason_prefix), row.rejection_reason
