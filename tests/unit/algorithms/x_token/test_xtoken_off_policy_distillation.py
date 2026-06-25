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
"""Unit tests for ``nemo_rl/algorithms/xtoken_off_policy_distillation.py``.

Mirrors the style of ``tests/unit/algorithms/test_distillation.py``:
a single ``mock_xtoken_components`` fixture builds all the Ray/policy/
data plumbing as ``MagicMock``s, then top-level ``def test_*``
functions exercise the high-level invariants the PR-2508 reviewer
flagged. CPU-only, no Ray, no CUDA.
"""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torchdata.stateful_dataloader import StatefulDataLoader

import nemo_rl.algorithms.xtoken_off_policy_distillation as xt_mod
from nemo_rl.algorithms.xtoken_off_policy_distillation import (
    MasterConfig,
    _default_off_policy_distillation_save_state,
    setup,
    validate,
    xtoken_off_policy_distillation_train,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_batch(
    batch_size: int = 1, t_student: int = 4, t_teacher: int = 4
) -> BatchedDataDict:
    """Synthetic batch with every key the algo's train_data packer reads."""
    return BatchedDataDict(
        {
            "input_ids": torch.zeros((batch_size, t_student), dtype=torch.long),
            "input_lengths": torch.full((batch_size,), t_student, dtype=torch.long),
            "token_mask": torch.ones((batch_size, t_student), dtype=torch.long),
            "sample_mask": torch.ones((batch_size,), dtype=torch.long),
            "teacher_input_ids": torch.zeros((batch_size, t_teacher), dtype=torch.long),
            "teacher_input_lengths": torch.full(
                (batch_size,), t_teacher, dtype=torch.long
            ),
            "teacher_token_mask": torch.ones((batch_size, t_teacher), dtype=torch.long),
            "alignment_pair_valid": torch.ones((batch_size, 2), dtype=torch.bool),
            "alignment_pair_is_correct": torch.ones((batch_size, 2), dtype=torch.bool),
            "alignment_student_exact_partition_mask": torch.zeros(
                (batch_size, t_student), dtype=torch.bool
            ),
            "alignment_teacher_exact_partition_mask": torch.zeros(
                (batch_size, t_teacher), dtype=torch.bool
            ),
            "alignment_student_chunk_id": torch.zeros(
                (batch_size, t_student), dtype=torch.long
            ),
            "alignment_teacher_chunk_id": torch.zeros(
                (batch_size, t_teacher), dtype=torch.long
            ),
            "alignment_num_chunks": torch.tensor([1] * batch_size, dtype=torch.long),
        }
    )


def _mock_dataloader(num_batches: int) -> MagicMock:
    batch = _make_batch()
    dl = MagicMock(spec=StatefulDataLoader)
    dl.__iter__ = lambda self: iter([batch] * num_batches)
    dl.__len__ = MagicMock(return_value=num_batches)
    dl.state_dict = MagicMock(return_value={})
    return dl


def _make_master_config(
    *,
    max_num_steps: int = 5,
    max_num_epochs: int = 10,
    val_period: int = 100,
    val_at_start: bool = False,
    val_at_end: bool = False,
    save_enabled: bool = False,
) -> MasterConfig:
    """MasterConfig that passes the setup() backend asserts and the
    train loop's lookups. Built via ``MasterConfig.model_construct`` to
    bypass strict TypedDict field validation (matches the pattern used
    in tests/unit/algorithms/test_rm.py).
    """
    return MasterConfig.model_construct(
        **{
            "distillation": {
                "num_prompts_per_step": 1,
                "max_num_steps": max_num_steps,
                "max_num_epochs": max_num_epochs,
                "seed": 42,
                "val_period": val_period,
                "val_at_start": val_at_start,
                "val_at_end": val_at_end,
                "val_teacher_micro_batch_size": 1,
            },
            "policy": {
                "dtensor_cfg": {
                    "enabled": True,
                    "_v2": True,
                    "tensor_parallel_size": 1,
                    "context_parallel_size": 1,
                },
                "max_total_sequence_length": 64,
                "make_sequence_length_divisible_by": 8,
                "train_global_batch_size": 1,
                "train_micro_batch_size": 1,
                "tokenizer": {"name": "student-tok"},
            },
            "teacher": {
                "dtensor_cfg": {
                    "enabled": True,
                    "_v2": True,
                    "tensor_parallel_size": 1,
                    "context_parallel_size": 1,
                },
                "max_total_sequence_length": 64,
                "make_sequence_length_divisible_by": 8,
                "train_global_batch_size": 1,
                "train_micro_batch_size": 1,
                "tokenizer": {"name": "teacher-tok"},
            },
            "loss_fn": {
                "projection_matrix_path": "/tmp/dummy-projection.pt",
                "gold_loss": False,
                "xtoken_loss": False,
                "temperature": 1.0,
                "vocab_topk": 8,
                "uncommon_topk": 4,
                "reverse_kl": False,
                "exact_token_match_only": False,
                "kl_loss_weight": 1.0,
                "ce_loss_scale": 1.0,
                "dynamic_loss_scaling": False,
            },
            "data": {
                "shuffle": False,
                "num_workers": 0,
            },
            "logger": {"log_dir": "/tmp/logger"},
            "cluster": {"num_nodes": 1, "gpus_per_node": 1},
            "checkpointing": {
                "enabled": save_enabled,
                "checkpoint_must_save_by": None,
                "save_period": 100,
                "metric_name": None,
            },
        }
    )


def _make_tokenizer(vocab_size: int) -> MagicMock:
    tok = MagicMock()
    tok.__len__ = MagicMock(return_value=vocab_size)
    return tok


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_xtoken_components():
    student_policy = MagicMock()
    student_policy.train.return_value = {
        "loss": torch.tensor(0.5),
        "grad_norm": torch.tensor(1.0),
        "all_mb_metrics": {"global_valid_toks": [10], "kl_loss": [0.3]},
    }
    # validate() derives the DP degree for the val-batch pad quantum.
    student_policy.data_parallel_size = 1

    teacher_policy = MagicMock()
    teacher_policy.get_full_logits_ipc.return_value = [{"shape": (4, 32)}]
    teacher_policy.data_parallel_size = 1

    train_dataloader = _mock_dataloader(num_batches=10)
    val_dataloader = _mock_dataloader(num_batches=2)

    loss_fn = MagicMock()
    logger = MagicMock()

    checkpointer = MagicMock()
    checkpointer.save_optimizer = False

    return SimpleNamespace(
        student_policy=student_policy,
        teacher_policy=teacher_policy,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        loss_fn=loss_fn,
        logger=logger,
        checkpointer=checkpointer,
        save_state=_default_off_policy_distillation_save_state(),
        master_config=_make_master_config(),
    )


# ---------------------------------------------------------------------------
# setup() backend & vocab-injection asserts
# ---------------------------------------------------------------------------


def _patched_setup_call(master_config, *, student_vocab=32, teacher_vocab=24):
    """Drive setup() with every heavy collaborator patched out."""
    student_tok = _make_tokenizer(student_vocab)
    teacher_tok = _make_tokenizer(teacher_vocab)
    train_ds = MagicMock()
    train_ds.__len__ = MagicMock(return_value=4)
    val_ds = MagicMock()
    val_ds.__len__ = MagicMock(return_value=2)

    with (
        patch.object(xt_mod, "RayVirtualCluster") as mock_cluster,
        patch.object(xt_mod, "Policy") as mock_policy_cls,
        patch.object(xt_mod, "Logger"),
        patch.object(xt_mod, "CheckpointManager") as mock_cp_cls,
        patch.object(xt_mod, "TokenAligner"),
        patch.object(xt_mod, "CrossTokenizerCollator"),
        patch.object(xt_mod, "CrossTokenizerDistillationLossFn") as mock_loss_cls,
        patch.object(xt_mod, "StatefulDataLoader") as mock_dl_cls,
    ):
        mock_cp_cls.return_value.get_latest_checkpoint_path.return_value = None
        mock_cp_cls.return_value.load_training_info.return_value = None
        mock_cp_cls.return_value.get_resume_paths.return_value = (None, None)
        mock_dl_cls.side_effect = lambda *a, **kw: MagicMock(spec=StatefulDataLoader)

        def _make_policy(*a, **kw):
            # Policy.data_parallel_size feeds the diff-DP grid assert in
            # setup(); it needs a real int, not a MagicMock.
            policy = MagicMock()
            policy.data_parallel_size = 1
            return policy

        mock_policy_cls.side_effect = _make_policy

        result = setup(
            master_config,
            student_tokenizer=student_tok,
            teacher_tokenizer=teacher_tok,
            train_dataset=train_ds,
            val_dataset=val_ds,
        )
        return result, {
            "cluster": mock_cluster,
            "policy": mock_policy_cls,
            "loss": mock_loss_cls,
            "checkpointer": mock_cp_cls,
        }


def test_setup_requires_dtensor_v2_student():
    cfg = _make_master_config()
    cfg.policy["dtensor_cfg"]["_v2"] = False
    with (
        patch.object(xt_mod, "RayVirtualCluster") as mock_cluster,
        pytest.raises(AssertionError),
    ):
        setup(
            cfg,
            student_tokenizer=_make_tokenizer(32),
            teacher_tokenizer=_make_tokenizer(24),
            train_dataset=MagicMock(),
            val_dataset=None,
        )
    assert mock_cluster.call_count == 0


def test_setup_requires_dtensor_v2_teacher():
    cfg = _make_master_config()
    cfg.teacher["dtensor_cfg"]["_v2"] = False
    with (
        patch.object(xt_mod, "RayVirtualCluster") as mock_cluster,
        pytest.raises(AssertionError),
    ):
        setup(
            cfg,
            student_tokenizer=_make_tokenizer(32),
            teacher_tokenizer=_make_tokenizer(24),
            train_dataset=MagicMock(),
            val_dataset=None,
        )
    assert mock_cluster.call_count == 0


def test_setup_injects_vocab_sizes_into_loss_config():
    cfg = _make_master_config()
    original_loss_cfg = deepcopy(cfg.loss_fn)

    _, mocks = _patched_setup_call(cfg, student_vocab=128, teacher_vocab=256)

    mocks["loss"].assert_called_once()
    injected_cfg = mocks["loss"].call_args.args[0]
    assert injected_cfg["student_vocab_size"] == 128
    assert injected_cfg["teacher_vocab_size"] == 256
    # YAML defaults preserved.
    assert injected_cfg["projection_matrix_path"] == "/tmp/dummy-projection.pt"
    # Original master_config not mutated by the injection.
    assert cfg.loss_fn == original_loss_cfg


@pytest.mark.parametrize(
    "val_dataset, val_period, val_at_start, val_at_end, expect_loader",
    [
        ("present", 0, False, False, False),  # all gates off
        (None, 100, True, True, False),  # no dataset
        ("present", 100, False, False, True),  # val_period > 0
        ("present", 0, True, False, True),  # val_at_start
        ("present", 0, False, True, True),  # val_at_end
    ],
)
def test_setup_val_dataloader_gating(
    val_dataset, val_period, val_at_start, val_at_end, expect_loader
):
    cfg = _make_master_config(
        val_period=val_period, val_at_start=val_at_start, val_at_end=val_at_end
    )

    ds = MagicMock()
    ds.__len__ = MagicMock(return_value=2)
    val_ds = ds if val_dataset is not None else None
    student_tok = _make_tokenizer(32)
    teacher_tok = _make_tokenizer(24)
    train_ds = MagicMock()
    train_ds.__len__ = MagicMock(return_value=4)

    with (
        patch.object(xt_mod, "RayVirtualCluster"),
        patch.object(xt_mod, "Policy") as mock_policy_cls,
        patch.object(xt_mod, "Logger"),
        patch.object(xt_mod, "CheckpointManager") as mock_cp_cls,
        patch.object(xt_mod, "TokenAligner"),
        patch.object(xt_mod, "CrossTokenizerCollator"),
        patch.object(xt_mod, "CrossTokenizerDistillationLossFn"),
        patch.object(xt_mod, "StatefulDataLoader") as mock_dl_cls,
    ):
        mock_cp_cls.return_value.get_latest_checkpoint_path.return_value = None
        mock_cp_cls.return_value.load_training_info.return_value = None
        mock_cp_cls.return_value.get_resume_paths.return_value = (None, None)
        mock_dl_cls.side_effect = lambda *a, **kw: MagicMock(spec=StatefulDataLoader)

        def _make_policy(*a, **kw):
            # Policy.data_parallel_size feeds the diff-DP grid assert in
            # setup(); it needs a real int, not a MagicMock.
            policy = MagicMock()
            policy.data_parallel_size = 1
            return policy

        mock_policy_cls.side_effect = _make_policy

        (
            _student,
            _teacher,
            _train_dl,
            val_dl,
            *_,
        ) = setup(
            cfg,
            student_tokenizer=student_tok,
            teacher_tokenizer=teacher_tok,
            train_dataset=train_ds,
            val_dataset=val_ds,
        )

    if expect_loader:
        assert val_dl is not None
    else:
        assert val_dl is None


# ---------------------------------------------------------------------------
# Training loop: exit conditions
# ---------------------------------------------------------------------------


def _run_train(c):
    xtoken_off_policy_distillation_train(
        c.student_policy,
        c.teacher_policy,
        c.train_dataloader,
        c.val_dataloader,
        c.loss_fn,
        c.logger,
        c.checkpointer,
        c.save_state,
        c.master_config,
    )


def test_exit_on_max_steps(mock_xtoken_components):
    mock_xtoken_components.master_config.distillation["max_num_steps"] = 3
    mock_xtoken_components.master_config.distillation["max_num_epochs"] = 10

    _run_train(mock_xtoken_components)

    assert mock_xtoken_components.student_policy.train.call_count == 3


def test_exit_on_max_epochs(mock_xtoken_components):
    # max_num_steps high so it doesn't fire first; max_num_epochs caps the loop.
    mock_xtoken_components.master_config.distillation["max_num_steps"] = 10_000
    mock_xtoken_components.master_config.distillation["max_num_epochs"] = 2
    # Two batches per epoch * 2 epochs = 4 student.train calls.
    mock_xtoken_components.train_dataloader = _mock_dataloader(num_batches=2)

    _run_train(mock_xtoken_components)

    assert mock_xtoken_components.student_policy.train.call_count == 4


def test_exit_on_timeout(mock_xtoken_components, capsys):
    mock_xtoken_components.master_config.distillation["max_num_steps"] = 100

    with patch.object(xt_mod, "TimeoutChecker") as mock_timeout_class:
        mock_timeout_instance = MagicMock()
        # False for 4 steps, then True (timeout).
        mock_timeout_instance.check_save.side_effect = [False] * 4 + [True]
        mock_timeout_class.return_value = mock_timeout_instance

        _run_train(mock_xtoken_components)

    # Loop should have run exactly 5 steps before tripping the timeout return.
    assert mock_xtoken_components.student_policy.train.call_count == 5

    captured = capsys.readouterr()
    assert "Timeout reached, stopping training early." in captured.out


# ---------------------------------------------------------------------------
# validate() loss-path branches
# ---------------------------------------------------------------------------


def _make_train_results_with(mb_metrics: dict) -> dict:
    return {
        "loss": torch.tensor(0.5),
        "grad_norm": torch.tensor(1.0),
        "all_mb_metrics": mb_metrics,
    }


def test_validate_emits_pkl_metrics_only(mock_xtoken_components):
    c = mock_xtoken_components
    c.student_policy.train.return_value = _make_train_results_with(
        {"kl_loss": [0.3], "ce_loss": [0.4]}
    )

    metrics, _timings = validate(
        c.student_policy,
        c.teacher_policy,
        c.val_dataloader,
        c.loss_fn,
        c.master_config,
    )

    assert "loss" in metrics
    assert "kl_loss" in metrics
    assert "ce_loss" in metrics
    assert "kl_common" not in metrics
    assert "l1_uncommon" not in metrics


def test_validate_emits_gold_metrics_only(mock_xtoken_components):
    c = mock_xtoken_components
    # The gold path combines KD with next-token CE, so it emits ce_loss
    # alongside kl_common/l1_uncommon. kl_loss stays P-KL-only.
    c.student_policy.train.return_value = _make_train_results_with(
        {"kl_common": [0.2], "l1_uncommon": [0.1], "ce_loss": [0.4]}
    )

    metrics, _timings = validate(
        c.student_policy,
        c.teacher_policy,
        c.val_dataloader,
        c.loss_fn,
        c.master_config,
    )

    assert "loss" in metrics
    assert "kl_common" in metrics
    assert "l1_uncommon" in metrics
    assert "ce_loss" in metrics
    assert "kl_loss" not in metrics


# ---------------------------------------------------------------------------
# IPC buffer release-on-failure
# ---------------------------------------------------------------------------


def test_ipc_buffer_released_on_student_train_failure(mock_xtoken_components):
    c = mock_xtoken_components
    c.student_policy.train.side_effect = RuntimeError("boom")
    c.master_config.distillation["max_num_steps"] = 1

    with pytest.raises(RuntimeError):
        _run_train(c)

    # Even though student.train raised, the except branch must have invoked
    # teacher_policy.release_ipc_buffer before propagating.
    assert c.teacher_policy.release_ipc_buffer.call_count >= 1


def test_ipc_buffer_not_released_on_happy_path(mock_xtoken_components):
    c = mock_xtoken_components
    c.master_config.distillation["max_num_steps"] = 3
    c.master_config.distillation["max_num_epochs"] = 10

    _run_train(c)

    # The teacher reuses one persistent IPC buffer across steps via copy_, so
    # successful steps must not release it (releasing every step would free +
    # realloc the large teacher-logits buffer and fragment the GPU into an OOM).
    # It is released exactly once at termination -- not per step, which the old
    # try/finally would have done (3 steps + 1 exit = 4 releases).
    assert c.teacher_policy.release_ipc_buffer.call_count == 1


# ---------------------------------------------------------------------------
# Entrypoint scope assert: examples/run_xtoken_off_policy_distillation.py
# ---------------------------------------------------------------------------


def _drive_run_main(config_dict: dict):
    """Drive ``run_xtoken_off_policy_distillation.main()`` with every
    heavy collaborator stubbed. Returns nothing — caller asserts on
    raise / no-raise behavior.
    """
    from examples import run_xtoken_off_policy_distillation as runner

    with (
        patch.object(runner, "register_omegaconf_resolvers"),
        patch.object(runner, "load_config", return_value=config_dict),
        patch.object(runner, "parse_hydra_overrides", side_effect=lambda c, o: c),
        patch.object(runner, "OmegaConf") as mock_om,
        # Bypass strict Pydantic validation; minimal stub config isn't a
        # valid full MasterConfig but is sufficient for the scope-gate test.
        patch.object(
            runner,
            "MasterConfig",
            lambda **kw: MasterConfig.model_construct(**kw),
        ),
        patch.object(runner, "get_next_experiment_dir", return_value="/tmp/exp"),
        patch.object(runner, "init_ray"),
        patch.object(runner, "get_tokenizer"),
        patch.object(
            runner, "setup_response_data", return_value=(MagicMock(), MagicMock())
        ),
        patch.object(runner, "setup") as mock_setup,
        patch.object(runner, "xtoken_off_policy_distillation_train"),
    ):
        mock_om.to_container.side_effect = lambda c, resolve=True: c
        mock_setup.return_value = tuple(MagicMock() for _ in range(9))
        runner.main()


def _runner_config(
    *,
    same_tokenizer: bool = False,
    projection_path: str | None = "/tmp/p.pt",
) -> dict:
    policy_tok = "shared-tok" if same_tokenizer else "student-tok"
    teacher_tok = "shared-tok" if same_tokenizer else "teacher-tok"
    return {
        "policy": {"tokenizer": {"name": policy_tok}},
        "teacher": {"tokenizer": {"name": teacher_tok}},
        "loss_fn": {"projection_matrix_path": projection_path},
        "logger": {"log_dir": "/tmp/logger"},
        "checkpointing": {"enabled": False, "checkpoint_dir": "/tmp/ckpt"},
        "data": {},
    }


def test_entrypoint_requires_distinct_tokenizers_and_projection_path():
    # Same tokenizer → AssertionError.
    with pytest.raises(AssertionError, match="cross-tokenizer"):
        _drive_run_main(_runner_config(same_tokenizer=True))

    # Distinct tokenizers but null projection path → AssertionError.
    with pytest.raises(AssertionError, match="cross-tokenizer"):
        _drive_run_main(_runner_config(projection_path=None))

    # Distinct tokenizers + non-null path → assert passes (no raise).
    _drive_run_main(_runner_config())  # should not raise
