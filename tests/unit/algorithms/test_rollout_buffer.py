# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
from torchdata.stateful_dataloader import StatefulDataLoader

from nemo_rl.algorithms.rollout_buffer import BUFFER_FILENAME, RolloutBuffer
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def _loader(num_prompts, num_workers=0):
    return StatefulDataLoader(
        list(range(num_prompts)),
        num_workers=num_workers,
        batch_size=2,
        drop_last=True,
        collate_fn=lambda rows: BatchedDataDict({"idx": torch.tensor(rows)}),
    )


def _fill(buffer, prompts, step):
    rows = prompts.repeat_interleave(4)
    rows["message_log"] = [
        [{"token_ids": torch.tensor([int(i), 9])}] for i in rows["idx"]
    ]
    rows["total_reward"] = torch.arange(rows.size).float()
    ids = rows["idx"].unsqueeze(-1)
    buffer.fill(rows, ids, step=step)


@pytest.mark.parametrize("multiplier", [1, 2, 3, 8])
@pytest.mark.parametrize("num_prompts,max_steps", [(20, 5), (10, 9), (11, 9), (3, 9)])
def test_drain_exactly_once_with_short_collection(multiplier, num_prompts, max_steps):
    buffer = RolloutBuffer(2, 4, multiplier)
    seen, collections = [], []
    for step, prompts in enumerate(
        buffer.prompt_batches(_loader(num_prompts), max_steps)
    ):
        if prompts is not None:
            collections.append(prompts.size)
            _fill(buffer, prompts, step)
        batch, ids = buffer.peek()
        assert batch.size == 8
        assert torch.equal(ids[:, 0], batch["idx"])
        assert torch.equal(ids.reshape(2, 4), ids.reshape(2, 4)[:, :1].expand(2, 4))
        seen.extend(batch["idx"].tolist()[::4])
        buffer.commit()
    expected = min(num_prompts // 2, max_steps) * 2
    assert seen == list(range(expected))
    assert sum(collections) == expected
    assert len(collections) == (expected // 2 + multiplier - 1) // multiplier
    assert buffer.remaining_updates == 0


@pytest.mark.parametrize("num_workers", [0, 1])
@pytest.mark.parametrize("num_prompts", [10, 4, 6])
@pytest.mark.parametrize("stop_after", [1, 2])
def test_resume_preserves_dataloader_position_and_epoch_tail(
    tmp_path, num_prompts, stop_after, num_workers
):
    loader = _loader(num_prompts, num_workers)
    buffer = RolloutBuffer(2, 4, 3)
    seen = []
    for step, prompts in enumerate(buffer.prompt_batches(loader, 20)):
        if prompts is not None:
            _fill(buffer, prompts, step)
        seen.extend(buffer.peek()[0]["idx"].tolist()[::4])
        buffer.commit()
        if step + 1 == stop_after:
            break
    buffer.save(tmp_path, completed_steps=stop_after)
    restored = RolloutBuffer(2, 4, 3)
    restored.restore(tmp_path, completed_steps=stop_after)
    restored_loader = _loader(num_prompts, num_workers)
    restored_loader.load_state_dict(loader.state_dict())
    for step, prompts in enumerate(
        restored.prompt_batches(restored_loader, 20), stop_after
    ):
        if prompts is not None:
            _fill(restored, prompts, step)
        seen.extend(restored.peek()[0]["idx"].tolist()[::4])
        restored.commit()
    assert seen == list(range(num_prompts))
    # A new epoch is allowed only after the restored old epoch has drained.
    assert next(restored.prompt_batches(restored_loader, 20))["idx"].tolist() == list(
        range(min(6, num_prompts))
    )


def test_mutable_processing_cannot_change_retained_rollouts(tmp_path):
    buffer = RolloutBuffer(2, 4, 3)
    prompts = next(buffer.prompt_batches(_loader(10), 5))
    _fill(buffer, prompts, 0)
    batch, ids = buffer.peek()
    batch["total_reward"].fill_(99)
    batch["message_log"][0][0]["token_ids"].fill_(99)
    ids.fill_(99)
    again, ids_again = buffer.peek()
    assert again["total_reward"][0] == 0
    assert again["message_log"][0][0]["token_ids"].tolist() == [0, 9]
    assert ids_again[0].item() == 0
    buffer.commit()
    buffer.save(tmp_path, completed_steps=1)
    saved = torch.load(tmp_path / BUFFER_FILENAME, weights_only=False)
    assert len(saved["batch"]["idx"]) == 16
    assert saved["batch"]["idx"].min() == 2


@pytest.mark.parametrize(
    "key,value",
    [
        ("version", 999),
        ("completed_steps", 2),
        ("multiplier", 2),
        ("prompts_per_update", 1),
        ("responses_per_prompt", 8),
        ("collection_step", 3),
        ("epoch_exhausted", "yes"),
    ],
)
def test_incompatible_checkpoint_fails(tmp_path, key, value):
    buffer = RolloutBuffer(2, 4, 3)
    _fill(buffer, next(buffer.prompt_batches(_loader(10), 5)), 0)
    buffer.commit()
    buffer.save(tmp_path, completed_steps=1)
    state = torch.load(tmp_path / BUFFER_FILENAME, weights_only=False)
    state[key] = value
    torch.save(state, tmp_path / BUFFER_FILENAME)
    with pytest.raises(ValueError):
        RolloutBuffer(2, 4, 3).restore(tmp_path, completed_steps=1)


def test_missing_checkpoint_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        RolloutBuffer(2, 4, 3).restore(tmp_path, completed_steps=1)


def test_invalid_group_order_and_partial_update_fail():
    buffer = RolloutBuffer(2, 4, 3)
    with pytest.raises(ValueError, match="whole optimizer"):
        buffer.fill(
            BatchedDataDict({"idx": torch.arange(7)}), torch.zeros(7, 1), step=0
        )
    with pytest.raises(ValueError, match="contiguous prompt"):
        buffer.fill(
            BatchedDataDict({"idx": torch.arange(8)}),
            torch.arange(8).view(8, 1),
            step=0,
        )
