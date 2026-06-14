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
    env: dict | None = None,
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
            "train": [{"env_name": "math"}],
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
        env=env if env is not None else {},
    )


class TestSetupHandle:
    """``setup_handle`` arg validation + return-tuple assembly."""

    def test_raises_when_data_plane_disabled(self):
        mc = _make_master_config(dp_enabled=False)
        with pytest.raises(ValueError, match="data_plane.enabled=True"):
            setup_handle(mc, MagicMock())

    def test_assembles_handles_from_inline_helpers(self):
        """setup_handle returns gen/policy via the inline _build_* helpers."""
        mc = _make_master_config(dp_enabled=True, colocated=True)
        policy = MagicMock()
        generation = MagicMock(name="generation")
        train_cluster, inference_cluster = MagicMock(), MagicMock()
        env_handles = {"math": MagicMock(name="math_env")}

        with (
            patch.object(
                setup_module,
                "_build_clusters",
                return_value=(train_cluster, inference_cluster),
            ) as mock_build_clusters,
            patch.object(
                setup_module, "_build_generation", return_value=generation
            ) as mock_build_generation,
            patch.object(
                setup_module, "_build_trainer", return_value=policy
            ) as mock_build_trainer,
        ):
            result = setup_handle(mc, MagicMock(), env_handles=env_handles)

        mock_build_clusters.assert_called_once_with(mc)
        mock_build_generation.assert_called_once()
        mock_build_trainer.assert_called_once()

        gen_handle, trainer_handle, returned_env_handles, tc, ic = result
        assert gen_handle is generation
        assert trainer_handle is policy
        assert returned_env_handles is env_handles
        assert tc is train_cluster
        assert ic is inference_cluster

    def test_env_handles_built_from_config_when_not_passed(self):
        """When env_handles is None, setup_handle iterates config.env + create_env."""
        math_env_cfg = {"some": "value"}
        mc = _make_master_config(env={"math": math_env_cfg}, colocated=True)
        policy = MagicMock()
        fake_math_env = MagicMock(name="math_env")

        with (
            patch.object(
                setup_module,
                "_build_clusters",
                return_value=(MagicMock(), MagicMock()),
            ),
            patch.object(setup_module, "_build_generation", return_value=MagicMock()),
            patch.object(setup_module, "_build_trainer", return_value=policy),
            patch.object(
                setup_module, "create_env", return_value=fake_math_env
            ) as mock_create_env,
        ):
            _, _, returned_env_handles, _, _ = setup_handle(
                mc, MagicMock(), env_handles=None
            )

        mock_create_env.assert_called_once_with(
            env_name="math", env_config=math_env_cfg
        )
        assert returned_env_handles == {"math": fake_math_env}


def _component_kwargs(*, master_config=None, **overrides):
    """Common kwargs for setup_single_controller_component tests."""
    kwargs = dict(
        gen_handle=MagicMock(),
        trainer_handle=MagicMock(),
        env_handles={},
        train_cluster=MagicMock(),
        inference_cluster=MagicMock(),
    )
    kwargs.update(overrides)
    return master_config or _make_master_config(), kwargs


class TestSetupSingleControllerComponent:
    """``setup_single_controller_component`` assembles the six locals."""

    def test_returns_six_tuple(self):
        mc, kwargs = _component_kwargs()
        tokenizer = MagicMock(pad_token_id=0)
        fake_dp_client = MagicMock(name="dp_client")

        with (
            patch.object(
                setup_module,
                "build_data_plane_client",
                return_value=fake_dp_client,
            ),
            patch.object(
                setup_module,
                "create_weight_synchronizer",
                return_value=MagicMock(name="weight_sync"),
            ),
            patch.object(
                setup_module,
                "setup_response_data",
                return_value=([1, 2, 3, 4, 5, 6, 7, 8], None),
            ),
        ):
            result = setup_single_controller_component(mc, tokenizer, **kwargs)

        dp_client, dataloader, weight_sync, advantage_estimator, rollout_manager, tq_buffer = result
        assert dp_client is fake_dp_client
        assert dataloader is not None
        assert weight_sync is not None
        assert advantage_estimator is not None
        assert rollout_manager is not None
        assert tq_buffer is not None

        # tq_buffer wires the dp_client built inside + default partition.
        assert tq_buffer._dp_client is fake_dp_client
        assert tq_buffer._partition_id == "rollout_data"

        # rollout_manager binds the same tq_buffer so writer + sampler share state.
        assert rollout_manager._tq_buffer is tq_buffer

    def test_multiple_dataloader_not_supported(self):
        mc = _make_master_config(use_multiple_dataloader=True)
        _, kwargs = _component_kwargs(master_config=mc)

        with (
            patch.object(
                setup_module, "build_data_plane_client", return_value=MagicMock()
            ),
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

        with (
            patch.object(
                setup_module, "build_data_plane_client", return_value=MagicMock()
            ),
            patch.object(
                setup_module,
                "create_weight_synchronizer",
                return_value=MagicMock(),
            ) as mock_factory,
            patch.object(
                setup_module,
                "setup_response_data",
                return_value=([1, 2, 3, 4], None),
            ),
        ):
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

        with (
            patch.object(
                setup_module, "build_data_plane_client", return_value=MagicMock()
            ),
            patch.object(
                setup_module,
                "create_weight_synchronizer",
                return_value=MagicMock(),
            ),
            patch.object(
                setup_module,
                "setup_response_data",
                return_value=([1, 2, 3, 4], None),
            ),
        ):
            _, _, _, _, _, tq_buffer = setup_single_controller_component(
                mc, tokenizer, partition_id="custom_partition", **kwargs
            )

        assert tq_buffer._partition_id == "custom_partition"
        assert tq_buffer._pad_value_dict == {"token_ids": 7, "input_ids": 7}
