"""Unit tests for ``nemo_rl.algorithms.single_controller_setup``.

setup_handle is heavy (calls ``grpo.setup`` which spins up Ray clusters,
policies, and generation backends) so it's exercised through
monkey-patching rather than as a real e2e — the unit tests cover the
shape of the contract, not the underlying initialization. The full path
is covered by the functional test at
``tests/test_suites/llm/sc-grpo-llama3.2-1b-instruct-1n8g-fsdp2tp1.sh``.

setup_single_controller_component takes a fully built
``SingleControllerHandles`` so it is exercised directly with fakes.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from nemo_rl.algorithms import single_controller_setup
from nemo_rl.algorithms.single_controller_setup import (
    SingleControllerComponents,
    SingleControllerHandles,
    setup_handle,
    setup_single_controller_component,
)


def _make_master_config(
    *,
    dp_enabled: bool = True,
    use_multiple_dataloader: bool = False,
    colocated: bool = True,
    backend: str = "vllm",
) -> Any:
    """Return a MagicMock master_config with just the fields the helpers read."""
    mc = MagicMock()
    mc.data_plane = {"enabled": dp_enabled, "impl": "transfer_queue"}
    mc.data = {
        "use_multiple_dataloader": use_multiple_dataloader,
        "shuffle": False,
        "num_workers": 0,
    }
    mc.grpo = {
        "num_prompts_per_step": 4,
        "num_generations_per_prompt": 2,
        "max_rollout_turns": 1,
    }
    mc.policy = {
        "max_total_sequence_length": 32,
        "generation": {
            "backend": backend,
            "colocated": {"enabled": colocated, "resources": {}},
        },
    }
    mc.env = {}
    return mc


def _make_handles(
    *,
    master_config: Any | None = None,
) -> SingleControllerHandles:
    if master_config is None:
        master_config = _make_master_config()
    return SingleControllerHandles(
        dp_client=MagicMock(),
        gen_handle=MagicMock(),
        trainer_handle=MagicMock(),
        env_handles={},
        train_cluster=MagicMock(),
        inference_cluster=MagicMock(),
        loss_fn=MagicMock(),
        dataset=[1, 2, 3, 4, 5, 6, 7, 8],
        val_dataset=None,
        master_config=master_config,
    )


class TestSetupHandle:
    """``setup_handle`` arg validation + handle assembly."""

    def test_raises_when_data_plane_disabled(self):
        mc = _make_master_config(dp_enabled=False)
        with pytest.raises(ValueError, match="data_plane.enabled=True"):
            setup_handle(mc, MagicMock(), object(), None)

    def test_passes_dp_cfg_to_tq_policy_and_returns_handles(self):
        """setup_handle wires grpo.setup with a TQPolicy factory and packs the four handles."""
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

        # Patch grpo.setup so we don't bootstrap clusters / vLLM. The
        # patched return value is the 11-tuple grpo.setup produces.
        with patch.object(
            single_controller_setup, "grpo_setup", return_value=fake_grpo_setup_return
        ) as mock_grpo_setup:
            handles = setup_handle(
                mc,
                tokenizer=MagicMock(),
                dataset=object(),
                val_dataset=None,
                env_handles=env_handles,
            )

        # grpo_setup was called once with a policy_factory kwarg.
        assert mock_grpo_setup.call_count == 1
        _, call_kwargs = mock_grpo_setup.call_args
        assert callable(call_kwargs["policy_factory"])

        # Handles are wired from the grpo_setup return values.
        assert isinstance(handles, SingleControllerHandles)
        assert handles.dp_client is policy.dp_client
        assert handles.gen_handle is policy_generation
        assert handles.trainer_handle is policy
        assert handles.env_handles is env_handles
        assert handles.train_cluster is train_cluster
        assert handles.inference_cluster is inference_cluster
        assert handles.loss_fn is loss_fn
        assert handles.master_config is mc

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
                single_controller_setup,
                "grpo_setup",
                return_value=fake_grpo_setup_return,
            ),
            patch.object(
                single_controller_setup,
                "setup_response_data",
                return_value=(object(), None, derived_env_handles, {}),
            ) as mock_setup_data,
        ):
            handles = setup_handle(mc, MagicMock(), object(), None, env_handles=None)

        mock_setup_data.assert_called_once()
        assert handles.env_handles is derived_env_handles


class TestSetupSingleControllerComponent:
    """``setup_single_controller_component`` assembles the six locals."""

    def test_returns_all_six_components(self):
        handles = _make_handles()
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0

        with patch.object(
            single_controller_setup,
            "create_weight_synchronizer",
            return_value=MagicMock(name="weight_sync"),
        ):
            components = setup_single_controller_component(handles, tokenizer)

        assert isinstance(components, SingleControllerComponents)
        # Six fields populated.
        assert components.dataloader is not None
        assert components.weight_synchronizer is not None
        assert components.advantage_estimator is not None
        assert components.tokenizer is tokenizer
        assert components.rollout_manager is not None
        assert components.tq_buffer is not None

        # tq_buffer shares the partition_id default and the dp_client.
        assert components.tq_buffer._dp_client is handles.dp_client
        assert components.tq_buffer._partition_id == "rollout_data"

        # rollout_manager binds the same tq_buffer instance (so add/sampler share state).
        assert components.rollout_manager._tq_buffer is components.tq_buffer

    def test_multiple_dataloader_not_supported(self):
        mc = _make_master_config(use_multiple_dataloader=True)
        handles = _make_handles(master_config=mc)

        with (
            patch.object(
                single_controller_setup,
                "create_weight_synchronizer",
                return_value=MagicMock(),
            ),
            pytest.raises(NotImplementedError, match="use_multiple_dataloader"),
        ):
            setup_single_controller_component(handles, MagicMock(pad_token_id=0))

    def test_weight_sync_factory_args(self):
        """create_weight_synchronizer receives the right policy/generation/topology."""
        mc = _make_master_config(colocated=False, backend="vllm")
        handles = _make_handles(master_config=mc)
        tokenizer = MagicMock(pad_token_id=0)

        with patch.object(
            single_controller_setup,
            "create_weight_synchronizer",
            return_value=MagicMock(),
        ) as mock_factory:
            setup_single_controller_component(handles, tokenizer)

        _, kwargs = mock_factory.call_args
        assert kwargs["policy"] is handles.trainer_handle
        assert kwargs["generation"] is handles.gen_handle
        assert kwargs["generation_backend"] == "vllm"
        assert kwargs["colocated"] is False
        assert kwargs["train_cluster"] is handles.train_cluster
        assert kwargs["inference_cluster"] is handles.inference_cluster

    def test_custom_partition_id(self):
        handles = _make_handles()
        tokenizer = MagicMock(pad_token_id=7)

        with patch.object(
            single_controller_setup,
            "create_weight_synchronizer",
            return_value=MagicMock(),
        ):
            components = setup_single_controller_component(
                handles, tokenizer, partition_id="custom_partition"
            )

        assert components.tq_buffer._partition_id == "custom_partition"
        # pad value derived from tokenizer.pad_token_id.
        assert components.tq_buffer._pad_value_dict == {"token_ids": 7, "input_ids": 7}
