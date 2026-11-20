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
"""grpo.agent_overrides: per-environment overlong filtering and reward normalization."""

import pytest
import torch
from pydantic import ValidationError

from nemo_rl.algorithms.advantage_estimator import (
    AdvEstimatorConfig,
    GRPOAdvantageEstimator,
)
from nemo_rl.algorithms.grpo import GRPOConfig, _apply_async_sample_masks
from nemo_rl.algorithms.single_controller_utils.advantage_stage import (
    agent_override_rows,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

GRADED = "rubric_judge_agent"
OTHER = "other_agent"


# ---- config -------------------------------------------------------------------------------------


def test_agent_overrides_default_is_off():
    assert GRPOConfig().agent_overrides is None


def test_agent_overrides_parse_and_fall_back_to_run_wide_values():
    cfg = GRPOConfig(
        overlong_filtering=False,
        agent_overrides={
            GRADED: {"overlong_filtering": True, "normalize_rewards": False}
        },
    )
    assert cfg.agent_overrides[GRADED].overlong_filtering is True
    assert cfg.agent_override_value(GRADED, "overlong_filtering", False) is True
    assert cfg.agent_override_value(GRADED, "normalize_rewards", True) is False
    # Unlisted and unknown agents keep the run-wide value.
    assert cfg.agent_override_value(OTHER, "overlong_filtering", False) is False
    assert cfg.agent_override_value(None, "normalize_rewards", True) is True


def test_agent_override_leaves_unset_fields_to_the_run():
    cfg = GRPOConfig(agent_overrides={GRADED: {"overlong_filtering": True}})
    assert cfg.agent_override_value(GRADED, "normalize_rewards", True) is True
    assert cfg.agent_override_value(GRADED, "normalize_rewards", False) is False


def test_agent_overrides_reject_empty_mapping():
    with pytest.raises(ValidationError, match="at least one attribution key"):
        GRPOConfig(agent_overrides={})


def test_agent_overrides_reject_an_override_that_sets_nothing():
    with pytest.raises(ValidationError, match="sets nothing"):
        GRPOConfig(agent_overrides={GRADED: {}})


def test_agent_overrides_reject_unknown_fields():
    with pytest.raises(ValidationError):
        GRPOConfig(agent_overrides={GRADED: {"token_level_loss": True}})


def test_agent_overrides_normalize_needs_grpo_estimator():
    with pytest.raises(ValidationError, match="adv_estimator.name 'grpo'"):
        GRPOConfig(
            adv_estimator={"name": "gdpo"},
            agent_overrides={GRADED: {"normalize_rewards": False}},
        )


@pytest.mark.parametrize("run_wide", [False, True])
def test_agent_overrides_reject_a_differing_overlong_flag_under_exclude_dynamic_sampling(
    run_wide,
):
    with pytest.raises(ValidationError, match="masked_reward_policy 'exclude'"):
        GRPOConfig(
            overlong_filtering=run_wide,
            use_dynamic_sampling=True,
            masked_reward_policy="exclude",
            agent_overrides={GRADED: {"overlong_filtering": not run_wide}},
        )


def test_agent_overrides_allow_the_run_wide_overlong_flag_or_include_policy():
    # Repeating the run-wide value is harmless under exclude.
    GRPOConfig(
        overlong_filtering=True,
        use_dynamic_sampling=True,
        masked_reward_policy="exclude",
        agent_overrides={GRADED: {"overlong_filtering": True}},
    )
    # Under include neither side drops truncated rows from the vote.
    GRPOConfig(
        overlong_filtering=False,
        use_dynamic_sampling=True,
        masked_reward_policy="include",
        agent_overrides={GRADED: {"overlong_filtering": True}},
    )


# ---- per-row switches from row tags --------------------------------------------------------------


def test_agent_override_rows_is_none_without_overrides():
    overlong, normalize, agents = agent_override_rows(GRPOConfig(), None, 3)
    assert overlong is None and normalize is None
    assert agents == [None, None, None]


def test_agent_override_rows_maps_each_row_to_its_agent():
    cfg = GRPOConfig(
        overlong_filtering=False,
        adv_estimator={"normalize_rewards": True},
        agent_overrides={
            GRADED: {"overlong_filtering": True, "normalize_rewards": False}
        },
    )
    tags = [{"agent_name": GRADED}, {"agent_name": OTHER}, {}]
    overlong, normalize, agents = agent_override_rows(cfg, tags, 3)
    assert agents == [GRADED, OTHER, None]
    assert overlong.tolist() == [True, False, False]
    assert normalize.tolist() == [False, True, True]


def test_agent_override_rows_skips_normalize_when_no_override_sets_it():
    cfg = GRPOConfig(agent_overrides={GRADED: {"overlong_filtering": True}})
    overlong, normalize, _ = agent_override_rows(cfg, [{"agent_name": GRADED}], 1)
    assert overlong.tolist() == [True]
    assert normalize is None


def test_agent_override_rows_rejects_tag_count_mismatch():
    cfg = GRPOConfig(agent_overrides={GRADED: {"overlong_filtering": True}})
    with pytest.raises(ValueError, match="row tags cover 1 rows; expected 2"):
        agent_override_rows(cfg, [{"agent_name": GRADED}], 2)


# ---- per-row overlong filtering ------------------------------------------------------------------


def _mask_batch(truncated, mask_sample=None):
    n = len(truncated)
    return BatchedDataDict(
        {
            "loss_multiplier": torch.ones(n),
            "truncated": torch.tensor(truncated),
            "mask_sample": torch.tensor(mask_sample or [False] * n),
        }
    )


def test_sample_masks_accept_per_row_overlong_filtering():
    batch = _mask_batch(
        [True, True, False, True], mask_sample=[False, False, False, True]
    )
    metrics, filtered = _apply_async_sample_masks(
        batch, overlong_filtering=torch.tensor([True, False, True, True])
    )
    # Row 0: listed and truncated -> masked; row 1: truncated but not listed;
    # row 2: listed, not truncated; row 3: masked once, by overlong filtering.
    assert batch["loss_multiplier"].tolist() == [0.0, 1.0, 1.0, 0.0]
    assert metrics["num_masked_seqs_by_overlong_filtering"] == 2
    assert metrics["num_masked_seqs_by_rollout"] == 0
    assert filtered == 1


def test_sample_masks_all_false_rows_equal_filtering_off():
    batch = _mask_batch([True, False])
    metrics, _ = _apply_async_sample_masks(
        batch, overlong_filtering=torch.tensor([False, False])
    )
    assert batch["loss_multiplier"].tolist() == [1.0, 1.0]
    assert metrics["num_masked_seqs_by_overlong_filtering"] == 0


def test_sample_masks_reject_wrong_row_count():
    with pytest.raises(ValueError, match="overlong_filtering has 1 rows; expected 2"):
        _apply_async_sample_masks(
            _mask_batch([True, False]), overlong_filtering=torch.tensor([True])
        )


# ---- per-row reward normalization ----------------------------------------------------------------


def _estimator(normalize_rewards: bool) -> GRPOAdvantageEstimator:
    return GRPOAdvantageEstimator(
        AdvEstimatorConfig(normalize_rewards=normalize_rewards), None
    )


def test_normalize_rows_mix_normalized_and_raw_groups():
    # Two groups of four: group 0 (graded, small spread) keeps raw differences,
    # group 1 (pass/fail) is divided by its std.
    prompt_ids = torch.tensor([[0], [0], [0], [0], [1], [1], [1], [1]])
    rewards = torch.tensor([0.80, 0.85, 0.80, 0.85, 1.0, 0.0, 1.0, 0.0])
    mask = torch.ones(8, 3)
    normalize_rows = torch.tensor([False] * 4 + [True] * 4)

    mixed = _estimator(True).compute_advantage(
        prompt_ids, rewards, mask, normalize_rows=normalize_rows
    )
    raw = _estimator(False).compute_advantage(prompt_ids, rewards, mask)
    normalized = _estimator(True).compute_advantage(prompt_ids, rewards, mask)

    torch.testing.assert_close(mixed[:4], raw[:4])
    torch.testing.assert_close(mixed[4:], normalized[4:])
    # The run-wide flag does not matter once rows decide.
    torch.testing.assert_close(
        _estimator(False).compute_advantage(
            prompt_ids, rewards, mask, normalize_rows=normalize_rows
        ),
        mixed,
    )


def test_normalize_rows_none_keeps_the_run_wide_flag():
    prompt_ids = torch.tensor([[0], [0], [0]])
    rewards = torch.tensor([0.2, 0.5, 0.9])
    mask = torch.ones(3, 2)
    torch.testing.assert_close(
        _estimator(True).compute_advantage(
            prompt_ids, rewards, mask, normalize_rows=None
        ),
        _estimator(True).compute_advantage(prompt_ids, rewards, mask),
    )


def test_normalize_rows_rejects_wrong_row_count():
    with pytest.raises(ValueError, match="normalize_rows has 1 rows; expected 2"):
        _estimator(True).compute_advantage(
            torch.tensor([[0], [0]]),
            torch.tensor([0.0, 1.0]),
            torch.ones(2, 1),
            normalize_rows=torch.tensor([True]),
        )
