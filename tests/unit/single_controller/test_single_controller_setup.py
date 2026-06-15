"""Unit tests for setup_single_controller.

setup_single_controller is heavy (it spins up Ray clusters, TQPolicy, generation
backend, ...) so it's exercised through monkey-patching rather than as a real e2e —
the unit tests cover the shape of the contract, not the underlying initialization.
The full path is covered by the functional test at
tests/functional/grpo_dp_single_controller.sh.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import nemo_rl.algorithms.single_controller_utils.setup as sc_setup_mod
from nemo_rl.algorithms.loss import ClippedPGLossConfig
from nemo_rl.algorithms.single_controller_utils import (
    MasterConfig,
    SingleControllerBundle,
    setup_single_controller,
)


def _make_master_config(
    *,
    dp_enabled: bool = True,
    use_multiple_dataloader: bool = False,
    colocated: bool = True,
    backend: str = "vllm",
    megatron_enabled: bool = False,
    env: dict | None = None,
    max_num_steps: int = 100,
    max_num_epochs: int = 1,
    num_prompts_per_step: int = 4,
) -> MasterConfig:
    """Build a partially-populated MasterConfig for unit tests.

    Cross-cutting components (cluster/checkpointing/...) are required by pydantic for
    normal load but unused here — model_construct skips validation, and we hand-fill
    only the dict-shaped fields setup reads.
    """
    return MasterConfig.model_construct(
        data_plane={"enabled": dp_enabled, "impl": "transfer_queue"},
        data={
            "use_multiple_dataloader": use_multiple_dataloader,
            "shuffle": False,
            "num_workers": 0,
            "train": [{"env_name": "math"}],
        },
        grpo={
            "max_num_steps": max_num_steps,
            "max_num_epochs": max_num_epochs,
            "num_prompts_per_step": num_prompts_per_step,
            "num_generations_per_prompt": 2,
            "max_rollout_turns": 1,
        },
        policy={
            "max_total_sequence_length": 32,
            "megatron_cfg": {"enabled": megatron_enabled},
            "generation": {
                "backend": backend,
                "colocated": {"enabled": colocated, "resources": {}},
            },
        },
        loss_fn=ClippedPGLossConfig(),
        env=env if env is not None else {},
    )


@pytest.fixture
def patched_factories():
    """Patch every external factory setup calls.

    Returns a dict of mocks keyed by name so individual tests can assert on call args
    without re-importing the patch handles.
    """
    fake_dataset = list(range(8))
    fake_dataloader = MagicMock(name="dataloader")
    # len(dataloader) used by the Megatron train_iters injection.
    fake_dataloader.__len__ = MagicMock(return_value=4)
    fake_env_handles = {"math": MagicMock(name="math_env")}

    with (
        patch.object(
            sc_setup_mod,
            "setup_response_data",
            return_value=(fake_dataset, None, fake_env_handles, {}),
        ) as mock_setup_response,
        patch.object(
            sc_setup_mod,
            "StatefulDataLoader",
            return_value=fake_dataloader,
        ) as mock_dataloader,
        patch.object(
            sc_setup_mod,
            "_build_clusters",
            return_value=(
                MagicMock(name="train_cluster"),
                MagicMock(name="inference_cluster"),
            ),
        ) as mock_clusters,
        patch.object(
            sc_setup_mod, "_build_generation", return_value=MagicMock(name="gen")
        ) as mock_gen,
        patch.object(
            sc_setup_mod, "_build_trainer", return_value=MagicMock(name="policy")
        ) as mock_trainer,
        patch.object(
            sc_setup_mod,
            "build_data_plane_client",
            return_value=MagicMock(name="dp_client"),
        ) as mock_dp_client,
        patch.object(
            sc_setup_mod,
            "create_weight_synchronizer",
            return_value=MagicMock(name="weight_sync"),
        ) as mock_weight_sync,
        patch.object(
            sc_setup_mod,
            "_create_advantage_estimator",
            return_value=MagicMock(name="adv"),
        ) as mock_adv,
        patch.object(
            sc_setup_mod, "ClippedPGLossFn", return_value=MagicMock(name="loss_fn")
        ) as mock_loss,
        patch.object(
            sc_setup_mod,
            "_generation_max_seq_len",
            return_value=32,
        ),
    ):
        yield {
            "setup_response_data": mock_setup_response,
            "StatefulDataLoader": mock_dataloader,
            "_build_clusters": mock_clusters,
            "_build_generation": mock_gen,
            "_build_trainer": mock_trainer,
            "build_data_plane_client": mock_dp_client,
            "create_weight_synchronizer": mock_weight_sync,
            "_create_advantage_estimator": mock_adv,
            "ClippedPGLossFn": mock_loss,
            "dataloader": fake_dataloader,
            "env_handles": fake_env_handles,
        }


class TestSetup:
    """setup arg validation + bundle assembly."""

    def test_raises_when_data_plane_disabled(self):
        mc = _make_master_config(dp_enabled=False)
        with pytest.raises(ValueError, match="data_plane.enabled=True"):
            setup_single_controller(mc, MagicMock())

    def test_multiple_dataloader_not_supported(self):
        mc = _make_master_config(use_multiple_dataloader=True)
        with pytest.raises(NotImplementedError, match="use_multiple_dataloader"):
            setup_single_controller(mc, MagicMock(pad_token_id=0))

    def test_returns_bundle(self, patched_factories):
        mc = _make_master_config(colocated=True)
        tokenizer = MagicMock(pad_token_id=0)

        bundle = setup_single_controller(mc, tokenizer)

        assert isinstance(bundle, SingleControllerBundle)
        assert bundle.gen_handle is patched_factories["_build_generation"].return_value
        assert bundle.trainer_handle is patched_factories["_build_trainer"].return_value
        assert bundle.env_handles is patched_factories["env_handles"]
        assert (
            bundle.dp_client
            is patched_factories["build_data_plane_client"].return_value
        )
        assert bundle.dataloader is patched_factories["dataloader"]
        assert bundle.weight_synchronizer is (
            patched_factories["create_weight_synchronizer"].return_value
        )
        assert bundle.advantage_estimator is (
            patched_factories["_create_advantage_estimator"].return_value
        )
        assert bundle.loss_fn is patched_factories["ClippedPGLossFn"].return_value
        # tq_buffer + rollout_manager are constructed inline (not mocked).
        assert bundle.tq_buffer is not None
        assert bundle.rollout_manager is not None
        # rollout_manager binds the same tq_buffer for the writer + sampler.
        assert bundle.rollout_manager._tq_buffer is bundle.tq_buffer
        # tq_buffer wires the dp_client + default partition.
        assert bundle.tq_buffer._dp_client is bundle.dp_client
        assert bundle.partition_id == "rollout_data"
        assert bundle.tq_buffer._partition_id == "rollout_data"

    def test_env_handles_sourced_from_setup_response_data(self, patched_factories):
        """setup_response_data receives master_config.env and supplies env handles."""
        math_env_cfg = {"some": "value"}
        mc = _make_master_config(env={"math": math_env_cfg})

        bundle = setup_single_controller(mc, MagicMock(pad_token_id=0))

        _, call_kwargs = patched_factories["setup_response_data"].call_args
        assert call_kwargs["env_configs"] == {"math": math_env_cfg}
        assert bundle.env_handles is patched_factories["env_handles"]

    def test_weight_sync_factory_args(self, patched_factories):
        """create_weight_synchronizer receives policy / generation / topology."""
        mc = _make_master_config(colocated=False, backend="vllm")
        tokenizer = MagicMock(pad_token_id=0)

        setup_single_controller(mc, tokenizer)

        _, factory_kwargs = patched_factories["create_weight_synchronizer"].call_args
        assert (
            factory_kwargs["policy"] is patched_factories["_build_trainer"].return_value
        )
        assert (
            factory_kwargs["generation"]
            is patched_factories["_build_generation"].return_value
        )
        assert factory_kwargs["generation_backend"] == "vllm"
        assert factory_kwargs["colocated"] is False

    def test_custom_partition_id(self, patched_factories):
        mc = _make_master_config()
        tokenizer = MagicMock(pad_token_id=7)

        bundle = setup_single_controller(mc, tokenizer, partition_id="custom_partition")

        assert bundle.partition_id == "custom_partition"
        assert bundle.tq_buffer._partition_id == "custom_partition"
        assert bundle.tq_buffer._pad_value_dict == {
            "token_ids": 7,
            "input_ids": 7,
        }

    def test_megatron_train_iters_capped_by_max_num_steps(self, patched_factories):
        """train_iters = min(max_num_steps, max_num_epochs * len(dataloader))."""
        mc = _make_master_config(
            megatron_enabled=True,
            max_num_steps=2,
            max_num_epochs=1,
        )
        # patched dataloader has len() == 4, so the min picks max_num_steps.
        setup_single_controller(mc, MagicMock(pad_token_id=0))

        assert mc.policy["megatron_cfg"]["train_iters"] == 2

    def test_megatron_train_iters_capped_by_dataloader_epochs(self, patched_factories):
        """train_iters drops to max_num_epochs * len(dataloader) when smaller."""
        mc = _make_master_config(
            megatron_enabled=True,
            max_num_steps=1000,
            max_num_epochs=2,
        )
        # patched dataloader has len() == 4 → 2 * 4 = 8 < 1000.
        setup_single_controller(mc, MagicMock(pad_token_id=0))

        assert mc.policy["megatron_cfg"]["train_iters"] == 8

    def test_megatron_train_iters_not_set_when_disabled(self, patched_factories):
        mc = _make_master_config(megatron_enabled=False)
        setup_single_controller(mc, MagicMock(pad_token_id=0))

        assert "train_iters" not in mc.policy.get("megatron_cfg", {})
