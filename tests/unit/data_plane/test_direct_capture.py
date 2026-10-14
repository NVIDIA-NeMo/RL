"""Production native codec and row assembly using CPU tensors and a memory DP."""

import torch

from nemo_rl.data_plane.direct_capture import NativeCallStore
from nemo_rl.experience.direct_capture import NativeCall
from nemo_rl.experience.rollout_reassembler import RolloutReassembler


class MemoryDataPlane:
    def __init__(self):
        self.rows = {}
        self.tags = {}

    def put_samples(self, *, sample_ids, partition_id, fields, tags):
        for index, key in enumerate(sample_ids):
            self.rows[partition_id, key] = {
                name: value[index].clone() for name, value in fields.items()
            }
            self.tags[partition_id, key] = tags[index]

    def get_samples(self, *, sample_ids, partition_id, select_fields):
        return {
            field: [self.rows[partition_id, key][field] for key in sample_ids]
            for field in select_fields
        }

    def clear_samples(self, *, sample_ids, partition_id):
        for key in sample_ids:
            self.rows.pop((partition_id, key))


def stage_rollout(dp, *, routed=False):
    store = NativeCallStore(dp, "staging", router_replay=routed)
    calls = [
        NativeCall(
            "r",
            "a",
            0,
            4,
            (1, 2),
            (3,),
            (-0.25,),
            torch.zeros(2, 2, 2, dtype=torch.int16) if routed else None,
        ),
        NativeCall(
            "r",
            "fork",
            1,
            4,
            (1, 2),
            (8,),
            (-0.5,),
            torch.ones(2, 2, 2, dtype=torch.int16) if routed else None,
        ),
        NativeCall(
            "r",
            "b",
            2,
            5,
            (1, 2, 3, 4),
            (5, 6),
            (-0.75, -0.5),
            torch.full((5, 2, 2), 2, dtype=torch.int16) if routed else None,
        ),
    ]
    return {
        "format": "nrl_direct_v1",
        "rollout_id": "r",
        "failure": None,
        "manifest": [store.stage(call) for call in calls],
    }


def test_native_roundtrip_main_branch_masks_and_excluded_calls():
    dp = MemoryDataPlane()
    receipt = stage_rollout(dp, routed=True)
    assembler = RolloutReassembler(
        dp,
        partition_id="train",
        staging_partition="staging",
        pad_token_id=0,
        max_seq_len=32,
        router_replay_enabled=True,
    )
    row = assembler.finalize_rollout("r", receipt, reward=1)
    assert row.valid, row.rejection_reason
    assert row.token_ids == [1, 2, 3, 4, 5, 6]
    assert row.token_mask == [0, 0, 1, 0, 1, 1]
    assert row.logprobs == [0, 0, -0.25, 0, -0.75, -0.5]
    assert row.reward == 1
    assert (row.min_wv, row.max_wv) == (4, 5)
    assert row.reward_metrics["finalize/direct_excluded_calls"] == 1
    assert torch.equal(row.routed_experts[1], torch.zeros(2, 2, dtype=torch.int16))
    assert torch.equal(row.routed_experts[3], torch.full((2, 2), 2, dtype=torch.int16))
    assert row.routed_experts[-1].eq(-1).all()


def test_native_group_publishes_train_rows_and_clears_all_calls_including_excluded():
    dp = MemoryDataPlane()
    receipt = stage_rollout(dp)
    assembler = RolloutReassembler(
        dp,
        partition_id="train",
        staging_partition="staging",
        pad_token_id=0,
        max_seq_len=32,
    )
    group = assembler.finalize_group(
        "group",
        ["r", "missing"],
        [receipt, None],
        [1, 0],
        mask_sample=[False, False],
        fallback_weight_version=0,
        prompt_idx=0,
        canonical_sample_ids=["group_g0", "group_g1"],
    )
    assert group.valid_row_count == 1
    assert group.total_row_count == 2
    assert group.canonical_output_tokens == 3
    assert group.meta.sample_ids == ["group_g0", "group_g1"]
    assert not any(partition == "staging" for partition, _ in dp.rows)
    assert dp.rows["train", "group_g1"]["sample_mask"].item() == 0


def test_native_receipt_rejects_tampered_identity():
    dp = MemoryDataPlane()
    receipt = stage_rollout(dp)
    receipt["manifest"][0]["generation_length"] = 900
    assembler = RolloutReassembler(
        dp,
        partition_id="train",
        staging_partition="staging",
        pad_token_id=0,
        max_seq_len=32,
    )
    row = assembler.finalize_rollout("r", receipt, reward=1)
    assert not row.valid
    assert "identity mismatch" in row.rejection_reason


def test_full_tree_keeps_fork_and_compaction_once_and_rewards_one_rollout():
    dp = MemoryDataPlane()
    receipt = stage_rollout(dp)
    store = NativeCallStore(dp, "staging", router_replay=False)
    receipt["manifest"].append(
        store.stage(NativeCall("r", "compact", 3, 5, (90,), (91,), (-0.25,), None))
    )
    assembler = RolloutReassembler(
        dp,
        partition_id="train",
        staging_partition="staging",
        pad_token_id=0,
        max_seq_len=32,
        training_view="full_tree",
    )
    group = assembler.finalize_group(
        "group",
        ["r", "missing"],
        [receipt, None],
        [1, 0],
        mask_sample=[False, False],
        fallback_weight_version=0,
        prompt_idx=0,
        canonical_sample_ids=["group_g0", "group_g1"],
    )
    assert group.valid_row_count == 1  # roots, independent of calls per root
    assert group.total_row_count == 2
    assert len(group.meta.sample_ids) == 5
    assert group.canonical_output_tokens == 5  # 1 parent + 1 fork + 2 child + 1 compact
    assert len(group.reward_observations) == 1
    assert group.metrics["finalize/direct_excluded_calls"] == 0
    expected_masks = [[0, 0, 1], [0, 0, 1], [0, 0, 0, 0, 1, 1], [0, 1]]
    for i, mask in enumerate(expected_masks):
        sample_id = f"group_g0/call{i}"
        assert dp.rows["train", sample_id]["token_mask"].tolist() == mask
        assert dp.tags["train", sample_id]["rollout_tree_id"] == "group_g0"
        assert dp.tags["train", sample_id]["rollout_tree_rows"] == 4
    assert not any(partition == "staging" for partition, _ in dp.rows)


def test_full_tree_rejects_entire_rollout_when_any_call_cannot_train():
    dp = MemoryDataPlane()
    receipt = stage_rollout(dp)
    assembler = RolloutReassembler(
        dp,
        partition_id="train",
        staging_partition="staging",
        pad_token_id=0,
        max_seq_len=4,
        training_view="full_tree",
    )
    group = assembler.finalize_group(
        "group",
        ["r"],
        [receipt],
        [1],
        mask_sample=[False],
        fallback_weight_version=0,
        prompt_idx=0,
        canonical_sample_ids=["group_g0"],
    )
    assert group.valid_row_count == 0
    assert group.total_row_count == 1
    assert group.canonical_output_tokens == 0
    assert group.meta.sample_ids == ["group_g0/call0"]
    assert not any(partition == "staging" for partition, _ in dp.rows)
