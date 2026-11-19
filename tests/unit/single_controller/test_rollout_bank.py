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
"""The rollout bank (collection-run output) and the controller stage that writes it."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from nemo_rl.algorithms.single_controller import SingleControllerActor
from nemo_rl.algorithms.single_controller_utils import rollout_bank as rb
from nemo_rl.algorithms.single_controller_utils.config import (
    AdvantageConfig,
    RolloutCollectionConfig,
    _validate_rollout_collection,
)
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.schema import ROUTE_PLAN_TAG

IDENTITY = {"format": rb.BANK_FORMAT, "policy_model_name": "m", "data": {"x": 1}}


def _rows(lengths, *, reward=1.0):
    """Per-field row lists for len(lengths) rows of the given token lengths."""
    n = len(lengths)
    return {
        "input_ids": [torch.arange(L, dtype=torch.int64) + 100 for L in lengths],
        "token_mask": [torch.tensor([0] * 2 + [1] * (L - 2)) for L in lengths],
        "generation_logprobs": [torch.full((L,), -0.5) for L in lengths],
        "input_lengths": [torch.tensor(L) for L in lengths],
        "sample_mask": [torch.tensor(1.0) for _ in range(n)],
        "mask_sample": [torch.tensor(False) for _ in range(n)],
        "truncated": [torch.tensor(i == n - 1) for i in range(n)],
        "total_reward": [torch.tensor(reward) for _ in range(n)],
    }


def _write(writer, step, group_id, lengths, *, prompt_idx=7, instance_id="repo__1"):
    tags = [
        {"trace_in_rollout_idx": i, "prompt_idx": prompt_idx, ROUTE_PLAN_TAG: "x"}
        for i in range(len(lengths))
    ]
    writer.write_group(
        step=step,
        group_id=group_id,
        prompt_idx=prompt_idx,
        instance_id=instance_id,
        extra_env_info={"instance_id": instance_id},
        sample_ids=[
            f"{group_id}_g0" if i == 0 else f"{group_id}_g0_t{i}"
            for i in range(len(lengths))
        ],
        tags=tags,
        sequence_lengths=lengths,
        fields=_rows(lengths),
    )


def test_round_trip_restores_rows_dtypes_and_drops_the_route_plan(tmp_path):
    writer = rb.RolloutBankWriter(tmp_path, IDENTITY, restored_step=0)
    _write(writer, 1, "grpA", [5, 3])
    assert writer.commit_step(1, trainer_version=0) == 1

    (ref,) = rb.list_bank_groups([tmp_path])
    assert (ref.step, ref.group_id, ref.rows, ref.rollouts) == (1, "grpA", 2, 1)
    assert ref.task_key == "instance:repo__1"
    group = rb.load_bank_group(ref.path)
    expected = _rows([5, 3])
    for name, rows in expected.items():
        assert [r.dtype for r in group.fields[name]] == [r.dtype for r in rows]
        for got, want in zip(group.fields[name], rows):
            assert torch.equal(got, want)
    assert all(ROUTE_PLAN_TAG not in tag for tag in group.tags)
    assert group.extra_env_info == {"instance_id": "repo__1"}
    assert group.sequence_lengths == [5, 3]
    # int64 token ids are stored as int32 and restored on load.
    raw = torch.load(ref.path, weights_only=False)
    assert raw["fields"]["input_ids"]["rows"][0].dtype == torch.int32


def test_uncommitted_steps_are_invisible(tmp_path):
    writer = rb.RolloutBankWriter(tmp_path, IDENTITY, restored_step=0)
    _write(writer, 1, "a", [4])
    writer.commit_step(1, trainer_version=0)
    _write(writer, 2, "b", [4])  # no commit: the job died here
    assert [ref.group_id for ref in rb.list_bank_groups([tmp_path])] == ["a"]


def test_resume_moves_steps_past_the_checkpoint_aside(tmp_path):
    writer = rb.RolloutBankWriter(tmp_path, IDENTITY, restored_step=0)
    for step in (1, 2, 3):
        _write(writer, step, f"g{step}", [4])
        writer.commit_step(step, trainer_version=step - 1)
    _write(writer, 4, "g4", [4])  # uncommitted
    # The run resumes from checkpoint step_2: steps 3 and 4 are re-collected.
    rb.RolloutBankWriter(tmp_path, IDENTITY, restored_step=2)
    assert [ref.step for ref in rb.list_bank_groups([tmp_path])] == [1, 2]
    moved = sorted(
        p.name.split(".")[0] for p in (tmp_path / rb.ORPHANED_DIRNAME).iterdir()
    )
    assert moved == ["step_000003", "step_000004"]


def test_fresh_run_refuses_a_bank_with_committed_steps(tmp_path):
    writer = rb.RolloutBankWriter(tmp_path, IDENTITY, restored_step=0)
    _write(writer, 1, "a", [4])
    writer.commit_step(1, trainer_version=0)
    with pytest.raises(ValueError, match="starts from scratch"):
        rb.RolloutBankWriter(tmp_path, IDENTITY, restored_step=0)


def test_bank_refuses_a_run_with_different_settings(tmp_path):
    rb.RolloutBankWriter(tmp_path, IDENTITY, restored_step=0)
    with pytest.raises(ValueError, match="different settings"):
        rb.RolloutBankWriter(tmp_path, {**IDENTITY, "data": {"x": 2}}, restored_step=0)
    assert json.loads((tmp_path / rb.BANK_META_FILENAME).read_text())["data"] == {
        "x": 1
    }


def test_row_tensors_cut_padded_and_jagged_columns_to_input_lengths():
    lengths = torch.tensor([3, 5])
    jagged = torch.nested.as_nested_tensor(
        [torch.arange(3.0), torch.arange(5.0)], layout=torch.jagged
    )
    padded = torch.arange(10.0).reshape(2, 5)
    data = TensorDict(
        {
            "token_mask": jagged,
            "generation_logprobs": padded,
            "total_reward": torch.tensor([1.0, 0.0]),
        },
        batch_size=[2],
    )
    assert [r.tolist() for r in rb.row_tensors(data, "token_mask", lengths)] == [
        [0.0, 1.0, 2.0],
        [0.0, 1.0, 2.0, 3.0, 4.0],
    ]
    assert [
        r.tolist() for r in rb.row_tensors(data, "generation_logprobs", lengths)
    ] == [
        [0.0, 1.0, 2.0],
        [5.0, 6.0, 7.0, 8.0, 9.0],
    ]
    assert [float(r) for r in rb.row_tensors(data, "total_reward", lengths)] == [
        1.0,
        0.0,
    ]


def test_batch_fields_rebuilds_jagged_token_columns_and_stacks_row_columns():
    rows = _rows([4, 6])
    batched = rb.batch_fields(rows)
    assert batched["input_ids"].is_nested
    assert [r.numel() for r in batched["input_ids"].unbind()] == [4, 6]
    assert batched["total_reward"].shape == (2,)


# ── config ─────────────────────────────────────────────────────────────────


def test_collection_requires_a_bank_dir():
    with pytest.raises(ValueError, match="bank_dir"):
        RolloutCollectionConfig(enabled=True)
    assert not RolloutCollectionConfig().enabled


def _collect_config(
    *,
    ppo=True,
    colocated=False,
    start_step=0,
    warm_start=None,
    replay=False,
    backend="simple",
    restart_dead_shards=False,
):
    return SimpleNamespace(
        rollout_collection=RolloutCollectionConfig(enabled=True, bank_dir="/tmp/b"),
        ppo=SimpleNamespace(
            policy_training_start_step=start_step,
            warm_start_value_checkpoint=warm_start,
        )
        if ppo
        else None,
        policy={"generation": {"backend": "vllm", "colocated": {"enabled": colocated}}},
        checkpointing={"save_data_plane": False},
        async_rl=SimpleNamespace(
            checkpoint_replay_buffer=replay,
            generation_fleet_health=SimpleNamespace(
                restart_dead_shards=restart_dead_shards
            ),
        ),
        data_plane={"backend": backend},
    )


def test_collection_validation():
    _validate_rollout_collection(_collect_config())
    with pytest.raises(ValueError, match="needs the `ppo` block"):
        _validate_rollout_collection(_collect_config(ppo=False))
    with pytest.raises(ValueError, match="non-colocated vLLM"):
        _validate_rollout_collection(_collect_config(colocated=True))
    with pytest.raises(ValueError, match="save_data_plane=false"):
        _validate_rollout_collection(_collect_config(replay=True))
    with pytest.raises(ValueError, match="simple"):
        _validate_rollout_collection(_collect_config(backend="mooncake_cpu"))
    with pytest.raises(ValueError, match="restart_dead_shards"):
        _validate_rollout_collection(_collect_config(restart_dead_shards=True))
    with pytest.raises(ValueError, match="policy_training_start_step"):
        _validate_rollout_collection(_collect_config(start_step=2))
    with pytest.warns(UserWarning, match="warm_start_value_checkpoint is ignored"):
        _validate_rollout_collection(_collect_config(warm_start="/x/step_1"))


# ── controller stage ───────────────────────────────────────────────────────


class _Plane:
    def __init__(self, data: TensorDict) -> None:
        self.data = data
        self.selected: list[str] | None = None

    def get_samples(self, *, sample_ids, select_fields, **kwargs):
        del kwargs
        self.selected = list(select_fields)
        assert len(sample_ids) == self.data.batch_size[0]
        return self.data


def _stage_batch():
    """Group A: canonical + one subagent row; group B: one row. Then one DP pad row."""
    lengths = [5, 3, 4]
    rows = _rows(lengths)
    data = TensorDict(rb.batch_fields(rows), batch_size=[3])
    sample_ids = ["A_g0", "A_g0_t1", "B_g0"]
    tags = [
        {"prompt_idx": 7, "trace_in_rollout_idx": 0, "weight_version": 0},
        {"prompt_idx": 7, "trace_in_rollout_idx": 1, "weight_version": 0},
        {"prompt_idx": 9, "trace_in_rollout_idx": 0, "weight_version": 0},
    ]
    meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=sample_ids + ["A_g0_pad0"],
        fields=list(data.keys()),
        sequence_lengths=lengths + [5],
        tags=tags + [dict(tags[0])],
    )
    return data, meta, rows


def test_bank_stage_writes_one_file_per_group_and_skips_pad_rows(tmp_path):
    data, meta, rows = _stage_batch()
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    ctrl._dp_client = _Plane(data)
    ctrl._advantage_cfg = AdvantageConfig()
    ctrl._train_steps = 4
    ctrl._bank = rb.RolloutBankWriter(tmp_path, IDENTITY, restored_step=4)

    async def _dataset_prompt(idx):
        md = {"instance_id": f"repo__{idx}", "patch": "diff"}
        return {"extra_env_info": {"responses_create_params": {"metadata": md}}}

    ctrl._dataset_prompt = _dataset_prompt
    # _Plane asserts only the three real rows are fetched.
    asyncio.run(ctrl._bank_stage(meta))
    assert ctrl._bank.commit_step(5, trainer_version=4) == 2
    assert set(ctrl._dp_client.selected) == set(rb.BANK_FIELDS)

    refs = {ref.group_id: ref for ref in rb.list_bank_groups([tmp_path])}
    assert set(refs) == {"A", "B"}
    assert (refs["A"].rows, refs["A"].rollouts, refs["A"].instance_id) == (
        2,
        1,
        "repo__7",
    )
    assert refs["B"].prompt_idx == 9 and refs["B"].step == 5
    group_a = rb.load_bank_group(refs["A"].path)
    # The reference block's source travels with the group.
    metadata = group_a.extra_env_info["responses_create_params"]["metadata"]
    assert metadata["instance_id"] == "repo__7"
    assert group_a.sample_ids == ["A_g0", "A_g0_t1"]
    assert torch.equal(group_a.fields["input_ids"][1], rows["input_ids"][1])
    assert group_a.sequence_lengths == [5, 3]


def test_bank_stage_without_an_instance_id_keys_on_the_prompt(tmp_path):
    data, meta, _ = _stage_batch()
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    ctrl._dp_client = _Plane(data)
    ctrl._advantage_cfg = AdvantageConfig()
    ctrl._train_steps = 0
    ctrl._bank = rb.RolloutBankWriter(tmp_path, IDENTITY, restored_step=0)

    async def _dataset_prompt(idx):
        return {"extra_env_info": {}}

    ctrl._dataset_prompt = _dataset_prompt
    asyncio.run(ctrl._bank_stage(meta))
    ctrl._bank.commit_step(1, trainer_version=0)
    ref = rb.list_bank_groups([tmp_path])[0]
    assert ref.instance_id is None and ref.task_key == "prompt:7"


def test_find_banks_expands_a_sharded_collection(tmp_path):
    for shard in ("shard_001", "shard_000"):
        rb.RolloutBankWriter(tmp_path / shard / "bank", IDENTITY, restored_step=0)
    assert rb.find_banks([tmp_path]) == [
        tmp_path / "shard_000" / "bank",
        tmp_path / "shard_001" / "bank",
    ]
    assert rb.find_banks([tmp_path / "shard_000" / "bank"]) == [
        tmp_path / "shard_000" / "bank"
    ]
    with pytest.raises(FileNotFoundError):
        rb.find_banks([tmp_path / "missing"])
