"""Tests for supported-path validation."""

from types import SimpleNamespace

import pytest
from turn_level_credit.config import TurnCreditConfig
from turn_level_credit.validation import validate_supported_path


def _master_config(
    *,
    estimator_name="grpo",
    async_grpo=False,
    data_plane=False,
    nemo_gym=False,
):
    return SimpleNamespace(
        grpo={
            "adv_estimator": {"name": estimator_name},
            "async_grpo": {"enabled": async_grpo},
        },
        data_plane={"enabled": data_plane},
        env={"should_use_nemo_gym": nemo_gym},
        policy={
            "generation": {
                "backend": "vllm",
                "vllm_cfg": {"async_engine": False},
            }
        },
    )


@pytest.mark.parametrize(
    ("master_config", "message"),
    [
        (_master_config(async_grpo=True), "async GRPO"),
        (_master_config(data_plane=True), "data_plane"),
        (_master_config(nemo_gym=True), "NeMo Gym"),
    ],
)
def test_disabled_config_rejects_launcher_paths_it_cannot_dispatch(
    master_config,
    message,
):
    with pytest.raises(ValueError, match=message):
        validate_supported_path(
            master_config,
            TurnCreditConfig(enabled=False),
        )


@pytest.mark.parametrize(
    ("master_config", "message"),
    [
        (_master_config(estimator_name="gdpo"), "only.*grpo"),
        (_master_config(async_grpo=True), "async GRPO"),
        (_master_config(data_plane=True), "data_plane"),
        (_master_config(nemo_gym=True), "NeMo Gym"),
    ],
)
def test_enabled_config_rejects_unsupported_paths(master_config, message):
    with pytest.raises(ValueError, match=message):
        validate_supported_path(
            master_config,
            TurnCreditConfig(enabled=True),
        )
