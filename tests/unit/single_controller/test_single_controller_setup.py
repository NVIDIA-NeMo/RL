"""Unit tests for ``nemo_rl.algorithms.single_controller_utils.setup``.

setup_handle is heavy (calls ``grpo.setup`` which spins up Ray clusters,
policies, and generation backends) so it's exercised through
monkey-patching rather than as a real e2e — the unit tests cover the
shape of the contract, not the underlying initialization. The full path
is covered by the functional test at
``tests/functional/grpo_dp_single_controller.sh``.

setup_single_controller_component takes flat kwargs (no wrapper); the
tests build the inputs directly and unpack the returned tuple.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from nemo_rl.algorithms.single_controller_utils import (
    MasterConfig,
    setup_handle,
    setup_single_controller_component,
)
from nemo_rl.algorithms.single_controller_utils import setup as setup_module


def _make_master_config(
    *,
    dp_enabled: bool = True,
    use_multiple_dataloader: bool = False,
    colocated: bool = True,
    backend: str = "vllm",
) -> MasterConfig:
    """Build a partially-populated MasterConfig for unit tests.

    Cross-cutting components (policy/data/cluster/...) are required by
    pydantic for normal load but unused here — ``model_construct``
    skips validation, and we hand-fill only the dict-shaped fields the
    setup helpers read.
    """
    return MasterConfig.model_construct(
        data_plane={"enabled": dp_enabled, "impl": "transfer_queue"},
        data={
            "use_multiple_dataloader": use_multiple_dataloader,
            "shuffle": False,
            "num_workers": 0,
        },
        grpo={
            "num_prompts_per_step": 4,
            "num_generations_per_prompt": 2,
            "max_rollout_turns": 1,
        },
        policy={
            "max_total_sequence_length": 32,
            "generation": {
                "backend": backend,
                "colocated": {"enabled": colocated, "resources": {}},
            },
        },
        env={},
    )


class TestSetupHandle:
    """``setup_handle`` arg validation + return-tuple assembly."""

    def test_raises_when_data_plane_disabled(self):
        mc = _make_master_config(dp_enabled=False)
        with pytest.raises(ValueError, match="data_plane.enabled=True"):
            setup_handle(mc, MagicMock(), object(), None)

    def test_passes_dp_cfg_to_tq_policy_and_returns_six_tuple(self):
        """setup_handle wires grpo.setup with a TQPolicy factory and returns the 6-tuple."""
        mc = _make_master_config(dp_enabled=True)
        policy = MagicMock()
        policy.dp_client = MagicMock(name="dp_client_handle")
        policy_generation = MagicMock(name="policy_generation")
        train_cluster, inference_cluster = MagicMock(), MagicMock()
        loss_fn = MagicMock(name="loss_fn")

        fake_grpo_setup_return = (
            policy,                # policy
            policy_generation,     # policy_generation
            None,                  # nemo_gym
            (train_cluster, inference_cluster),
            MagicMock(),           # dataloader (ignored — SC rebuilds it)
            None,                  # val_dataloader
            loss_fn,
            MagicMock(),           # logger
            MagicMock(),           # checkpointer
            {},                    # grpo_save_state
            mc,                    # master_config
        )
        env_handles = {"math": MagicMock(name="math_env")}

        with patch.object(
            setup_module, "grpo_setup", return_value=fake_grpo_setup_return
        ) as mock_grpo_setup:
            result = setup_handle(
                mc,
                tokenizer=MagicMock(),
                dataset=object(),
                val_dataset=None,
                env_handles=env_handles,
            )

        assert mock_grpo_setup.call_count == 1
        _, call_kwargs = mock_grpo_setup.call_args
        assert callable(call_kwargs["policy_factory"])

        dp_client, gen_handle, trainer_handle, returned_env_handles, tc, ic = result
        assert dp_client is policy.dp_client
        assert gen_handle is policy_generation
        assert trainer_handle is policy
        assert returned_env_handles is env_handles
        assert tc is train_cluster
        assert ic is inference_cluster

    def test_builds_env_handles_when_not_passed(self):
        """When env_handles is None, setup_response_data is invoked."""
        mc = _make_master_config(dp_enabled=True)
        policy = MagicMock()
        policy.dp_client = MagicMock()
        derived_env_handles = {"derived": MagicMock()}

        fake_grpo_setup_return = (
            policy,
            MagicMock(),
            None,
            (MagicMock(), MagicMock()),
            MagicMock(),
            None,
            MagicMock(),
            MagicMock(),
            MagicMock(),
            {},
            mc,
        )

        with (
            patch.object(
                setup_module,
                "grpo_setup",
                return_value=fake_grpo_setup_return,
            ),
            patch.object(
                setup_module,
                "setup_response_data",
                return_value=(object(), None, derived_env_handles, {}),
            ) as mock_setup_data,
        ):
            _, _, _, returned_env_handles, _, _ = setup_handle(
                mc, MagicMock(), object(), None, env_handles=None
            )

        mock_setup_data.assert_called_once()
        assert returned_env_handles is derived_env_handles


def _component_kwargs(*, dp_client=None, master_config=None, **overrides):
    """Common kwargs for setup_single_controller_component tests."""
    kwargs = dict(
        dp_client=dp_client or MagicMock(),
        gen_handle=MagicMock(),
        trainer_handle=MagicMock(),
        env_handles={},
        train_cluster=MagicMock(),
        inference_cluster=MagicMock(),
        dataset=[1, 2, 3, 4, 5, 6, 7, 8],
    )
    kwargs.update(overrides)
    return master_config or _make_master_config(), kwargs


class TestSetupSingleControllerComponent:
    """``setup_single_controller_component`` assembles the five locals."""

    def test_returns_five_tuple(self):
        mc, kwargs = _component_kwargs()
        tokenizer = MagicMock(pad_token_id=0)

        with patch.object(
            setup_module,
            "create_weight_synchronizer",
            return_value=MagicMock(name="weight_sync"),
        ):
            result = setup_single_controller_component(mc, tokenizer, **kwargs)

        dataloader, weight_sync, advantage_estimator, rollout_manager, tq_buffer = result
        assert dataloader is not None
        assert weight_sync is not None
        assert advantage_estimator is not None
        assert rollout_manager is not None
        assert tq_buffer is not None

        # tq_buffer wires dp_client + default partition.
        assert tq_buffer._dp_client is kwargs["dp_client"]
        assert tq_buffer._partition_id == "rollout_data"

        # rollout_manager binds the same tq_buffer so writer + sampler share state.
        assert rollout_manager._tq_buffer is tq_buffer

    def test_multiple_dataloader_not_supported(self):
        mc = _make_master_config(use_multiple_dataloader=True)
        _, kwargs = _component_kwargs(master_config=mc)

        with (
            patch.object(
                setup_module,
                "create_weight_synchronizer",
                return_value=MagicMock(),
            ),
            pytest.raises(NotImplementedError, match="use_multiple_dataloader"),
        ):
            setup_single_controller_component(
                mc, MagicMock(pad_token_id=0), **kwargs
            )

    def test_weight_sync_factory_args(self):
        """create_weight_synchronizer receives the right policy/generation/topology."""
        mc = _make_master_config(colocated=False, backend="vllm")
        _, kwargs = _component_kwargs(master_config=mc)
        tokenizer = MagicMock(pad_token_id=0)

        with patch.object(
            setup_module,
            "create_weight_synchronizer",
            return_value=MagicMock(),
        ) as mock_factory:
            setup_single_controller_component(mc, tokenizer, **kwargs)

        _, factory_kwargs = mock_factory.call_args
        assert factory_kwargs["policy"] is kwargs["trainer_handle"]
        assert factory_kwargs["generation"] is kwargs["gen_handle"]
        assert factory_kwargs["generation_backend"] == "vllm"
        assert factory_kwargs["colocated"] is False
        assert factory_kwargs["train_cluster"] is kwargs["train_cluster"]
        assert factory_kwargs["inference_cluster"] is kwargs["inference_cluster"]

    def test_custom_partition_id(self):
        mc, kwargs = _component_kwargs()
        tokenizer = MagicMock(pad_token_id=7)

        with patch.object(
            setup_module,
            "create_weight_synchronizer",
            return_value=MagicMock(),
        ):
            _, _, _, _, tq_buffer = setup_single_controller_component(
                mc, tokenizer, partition_id="custom_partition", **kwargs
            )

        assert tq_buffer._partition_id == "custom_partition"
        assert tq_buffer._pad_value_dict == {"token_ids": 7, "input_ids": 7}
