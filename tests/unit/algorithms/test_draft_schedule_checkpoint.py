# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import pytest

from nemo_rl.algorithms.grpo import restore_draft_update_scheduler
from nemo_rl.models.policy.draft_config import (
    AlwaysDraftUpdateScheduleConfig,
    FixedDraftUpdateScheduleConfig,
)


def test_fresh_fixed_run_without_saved_state_is_allowed() -> None:
    """A fresh launch must not be mistaken for a legacy resume."""
    config = FixedDraftUpdateScheduleConfig(
        mode="fixed", action="sparse_update", fixed_interval=10
    )

    scheduler = restore_draft_update_scheduler(
        config, None, origin_step=0, resuming_from_checkpoint=False
    )

    assert scheduler.state.schedule_origin_step == 0


def test_legacy_checkpoint_is_allowed_only_for_always() -> None:
    """Changing a legacy fixed cadence during resume must fail closed."""
    always = AlwaysDraftUpdateScheduleConfig()
    assert (
        restore_draft_update_scheduler(
            always, None, origin_step=4, resuming_from_checkpoint=True
        ).state.schedule_origin_step
        == 4
    )

    fixed = FixedDraftUpdateScheduleConfig(
        mode="fixed", action="sparse_update", fixed_interval=10
    )
    with pytest.raises(ValueError, match="legacy checkpoint.*always"):
        restore_draft_update_scheduler(
            fixed, None, origin_step=4, resuming_from_checkpoint=True
        )


def test_restore_rejects_resolved_config_mismatch() -> None:
    """A changed cadence parameter cannot silently reinterpret a checkpoint."""
    original = FixedDraftUpdateScheduleConfig(
        mode="fixed", action="sparse_update", fixed_interval=10
    )
    changed = FixedDraftUpdateScheduleConfig(
        mode="fixed", action="sparse_update", fixed_interval=40
    )
    saved = restore_draft_update_scheduler(
        original, None, origin_step=0, resuming_from_checkpoint=False
    ).state_dict()

    with pytest.raises(ValueError, match="resolved draft update schedule"):
        restore_draft_update_scheduler(
            changed, saved, origin_step=0, resuming_from_checkpoint=True
        )
