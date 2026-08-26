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
"""Skipping reference logprobs with a non-zero KL penalty must fail loudly.

The two sync paths guarded this with a bare, message-less ``assert``. Under
``python -O`` an assert is removed entirely, so the run would proceed to train
against a KL term whose reference logprobs were never computed -- the same
defect class raised on PR #3262, where the async twin was converted to a
``ValueError`` and these two were missed.

CPU-only: the guard sits above everything in these functions except a Timer
and a MemoryTracker, so mock arguments reach it.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nemo_rl.algorithms.grpo_sync import grpo_train_sync
from nemo_rl.algorithms.ppo import ppo_train


def _master_config(block: str, *, kl_penalty: float):
    return SimpleNamespace(
        **{
            block: SimpleNamespace(skip_reference_policy_logprobs_calculation=True),
            "loss_fn": SimpleNamespace(reference_policy_kl_penalty=kl_penalty),
            "checkpointing": {"checkpoint_must_save_by": None},
        }
    )


def _call_ppo(master_config):
    ppo_train(
        policy=MagicMock(),
        policy_generation=MagicMock(),
        value_model=MagicMock(),
        dataloader=MagicMock(),
        val_dataloader=None,
        tokenizer=MagicMock(),
        loss_fn=MagicMock(),
        value_loss_fn=MagicMock(),
        task_to_env={},
        val_task_to_env=None,
        logger=MagicMock(),
        checkpointer=MagicMock(),
        ppo_save_state=MagicMock(),
        master_config=master_config,
    )


def _call_grpo(master_config):
    grpo_train_sync(
        policy=MagicMock(),
        policy_generation=MagicMock(),
        wrapped_dataloader=MagicMock(),
        val_dataloader=None,
        tokenizer=MagicMock(),
        loss_fn=MagicMock(),
        task_to_env={},
        val_task_to_env=None,
        logger=MagicMock(),
        checkpointer=MagicMock(),
        grpo_save_state=MagicMock(),
        master_config=master_config,
    )


def test_the_grpo_guard_that_actually_fires_is_in_setup():
    """`grpo_train_sync`'s guard is shadowed and never reached.

    `grpo.setup` checks the same pairing before the train loop starts, so on
    GRPO the setup one is what a user hits. It was an assert too, so under
    `python -O` both were stripped and nothing was left. Converting only the
    train-loop copy would have fixed the unreachable one.

    Asserted on the source rather than by calling `setup`, which would need a
    cluster: the point is that the reachable guard is not an assert.
    """
    import inspect

    from nemo_rl.algorithms import grpo

    src = inspect.getsource(grpo.setup)
    marker = "skip_reference_policy_logprobs_calculation:"
    assert marker in src
    guard = src[src.index(marker) : src.index(marker) + 400]
    assert "raise ValueError" in guard, "the reachable GRPO guard must not be an assert"
    assert "reference_policy_kl_penalty=0" in guard


@pytest.mark.parametrize(
    ("call", "block"),
    [(_call_ppo, "ppo"), (_call_grpo, "grpo")],
    ids=["sync-ppo", "sync-grpo"],
)
def test_a_nonzero_kl_penalty_with_skipped_reference_logprobs_raises(call, block):
    """A ValueError, not an assert: `python -O` strips asserts, and this one
    was message-less on top of that."""
    with pytest.raises(ValueError, match="reference_policy_kl_penalty=0"):
        call(_master_config(block, kl_penalty=0.1))


@pytest.mark.parametrize(
    ("call", "block"),
    [(_call_ppo, "ppo"), (_call_grpo, "grpo")],
    ids=["sync-ppo", "sync-grpo"],
)
def test_the_supported_pairing_gets_past_the_guard(call, block):
    """kl_penalty=0 is the combination the guard exists to allow. It must not
    raise ValueError here -- these mocks fail later, which is fine and is what
    keeps the test from passing vacuously if the guard were made
    unconditional."""
    with pytest.raises(Exception) as excinfo:
        call(_master_config(block, kl_penalty=0.0))
    assert "reference_policy_kl_penalty=0" not in str(excinfo.value)
