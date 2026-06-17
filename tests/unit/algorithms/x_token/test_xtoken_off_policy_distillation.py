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
from nemo_rl.algorithms.loss.loss_functions import CrossTokenizerDistillationLossFn
from nemo_rl.algorithms.xtoken_off_policy_distillation import (
    MasterConfig,
    TeacherConfig,
    _default_off_policy_distillation_save_state,
    _run_teacher_forwards_and_pack,
    setup,
    validate,
    xtoken_non_student_seq_keys,
    xtoken_off_policy_distillation_train,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_batch(
    batch_size: int = 1,
    t_student: int = 4,
    t_teacher: int = 4,
    num_teachers: int = 1,
) -> dict:
    """Synthetic batch with every teacher-indexed key the packer reads.

    Each teacher ``i`` is cross-tokenizer here, so it carries
    ``teacher_{i}_*`` tokenization + ``alignment_{i}_*`` keys (matching what
    ``CrossTokenizerCollator`` emits for a cross-tokenizer teacher).
    """
    batch = {
        "input_ids": torch.zeros((batch_size, t_student), dtype=torch.long),
        "input_lengths": torch.full((batch_size,), t_student, dtype=torch.long),
        "token_mask": torch.ones((batch_size, t_student), dtype=torch.long),
        "sample_mask": torch.ones((batch_size,), dtype=torch.long),
    }
    for i in range(num_teachers):
        batch[f"teacher_{i}_input_ids"] = torch.zeros(
            (batch_size, t_teacher), dtype=torch.long
        )
        batch[f"teacher_{i}_input_lengths"] = torch.full(
            (batch_size,), t_teacher, dtype=torch.long
        )
        batch[f"teacher_{i}_token_mask"] = torch.ones(
            (batch_size, t_teacher), dtype=torch.long
        )
        batch[f"alignment_{i}_pair_valid"] = torch.ones(
            (batch_size, 2), dtype=torch.bool
        )
        batch[f"alignment_{i}_pair_is_correct"] = torch.ones(
            (batch_size, 2), dtype=torch.bool
        )
        batch[f"alignment_{i}_student_exact_partition_mask"] = torch.zeros(
            (batch_size, t_student), dtype=torch.bool
        )
        batch[f"alignment_{i}_teacher_exact_partition_mask"] = torch.zeros(
            (batch_size, t_teacher), dtype=torch.bool
        )
        batch[f"alignment_{i}_student_chunk_id"] = torch.zeros(
            (batch_size, t_student), dtype=torch.long
        )
        batch[f"alignment_{i}_teacher_chunk_id"] = torch.zeros(
            (batch_size, t_teacher), dtype=torch.long
        )
        batch[f"alignment_{i}_num_chunks"] = torch.tensor(
            [1] * batch_size, dtype=torch.long
        )
    return batch


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
                "tokenizer": {"name": "student-tok"},
            },
            "teachers": [
                TeacherConfig(
                    **{
                        "projection_matrix_path": "/tmp/dummy-projection.pt",
                        "weight": 1.0,
                        "dtensor_cfg": {
                            "enabled": True,
                            "_v2": True,
                            "tensor_parallel_size": 1,
                            "context_parallel_size": 1,
                        },
                        "max_total_sequence_length": 64,
                        "make_sequence_length_divisible_by": 8,
                        "tokenizer": {"name": "teacher-tok"},
                    }
                )
            ],
            "loss_fn": {
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
                "kd_loss_mode": "sum",
                "sum_weights_metric": None,
                "token_level_weights": False,
                "alpha": 1.0,
                "normalize_teacher_by_vocab": False,
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

    teacher_policy = MagicMock()
    teacher_policy.get_teacher_logits_ipc.return_value = [{"rank_logits_ipc": (4, 32)}]

    train_dataloader = _mock_dataloader(num_batches=10)
    val_dataloader = _mock_dataloader(num_batches=2)

    # The trainer reads per-teacher metadata off the loss fn to build the
    # skip-keys set and drive the teacher-forward loop. One cross-tokenizer
    # teacher: non-null projection path, ships full logits.
    loss_fn = MagicMock()
    loss_fn.num_teachers = 1
    loss_fn.projection_matrix_paths = ["/tmp/dummy-projection.pt"]
    loss_fn.teacher_ships_full = [True]
    loss_fn.vocab_topk = 8
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
        mock_policy_cls.side_effect = lambda *a, **kw: MagicMock()

        result = setup(
            master_config,
            student_tokenizer=student_tok,
            teacher_tokenizers=[teacher_tok],
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
            teacher_tokenizers=[_make_tokenizer(24)],
            train_dataset=MagicMock(),
            val_dataset=None,
        )
    assert mock_cluster.call_count == 0


def test_setup_requires_dtensor_v2_teacher():
    cfg = _make_master_config()
    cfg.teachers[0].dtensor_cfg["_v2"] = False
    with (
        patch.object(xt_mod, "RayVirtualCluster") as mock_cluster,
        pytest.raises(AssertionError),
    ):
        setup(
            cfg,
            student_tokenizer=_make_tokenizer(32),
            teacher_tokenizers=[_make_tokenizer(24)],
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
    # Per-teacher metadata is injected as parallel lists (one teacher here).
    assert injected_cfg["teacher_vocab_sizes"] == [256]
    assert injected_cfg["projection_matrix_paths"] == ["/tmp/dummy-projection.pt"]
    assert injected_cfg["teacher_weights"] == [1.0]
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
        mock_policy_cls.side_effect = lambda *a, **kw: MagicMock()

        (
            _student,
            _teacher,
            _train_dl,
            val_dl,
            *_,
        ) = setup(
            cfg,
            student_tokenizer=student_tok,
            teacher_tokenizers=[teacher_tok],
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
        [c.teacher_policy],
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
        [c.teacher_policy],
        c.val_dataloader,
        c.loss_fn,
        c.master_config,
    )

    assert "loss" in metrics
    assert "kl_loss" in metrics
    assert "ce_loss" in metrics
    assert "kl_common" not in metrics
    assert "l1_uncommon" not in metrics


def test_validate_collects_only_aggregate_metrics(mock_xtoken_components):
    c = mock_xtoken_components
    # validate summarizes only the aggregate loss / kl_loss / ce_loss. The
    # gold path's per-teacher components (kl_common_t{i} / l1_uncommon_t{i})
    # are not collected here, so they don't appear in the val metrics.
    c.student_policy.train.return_value = _make_train_results_with(
        {"kl_common_t0": [0.2], "l1_uncommon_t0": [0.1], "ce_loss": [0.4]}
    )

    metrics, _timings = validate(
        c.student_policy,
        [c.teacher_policy],
        c.val_dataloader,
        c.loss_fn,
        c.master_config,
    )

    assert "loss" in metrics
    assert "ce_loss" in metrics
    assert "kl_common" not in metrics
    assert "l1_uncommon" not in metrics
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

    # Even though student.train raised, the try/finally must have invoked
    # teacher_policy.release_ipc_buffer.
    assert c.teacher_policy.release_ipc_buffer.call_count >= 1


# ---------------------------------------------------------------------------
# Back-compat shim: legacy single `teacher:` block -> `teachers:` list
# ---------------------------------------------------------------------------


def test_normalize_multi_teacher_config_shim():
    from nemo_rl.algorithms.xtoken_off_policy_distillation import (
        normalize_multi_teacher_config,
    )

    # Legacy single `teacher:` + loss_fn.projection_matrix_path -> 1-element
    # teachers list; projection path moves onto the teacher, weight defaults 1.
    raw = {
        "teacher": {
            "model_name": "Qwen/Qwen3-4B",
            "tokenizer": {"name": "Qwen/Qwen3-4B"},
        },
        "loss_fn": {"projection_matrix_path": "/p.pt"},
    }
    out = normalize_multi_teacher_config(raw)
    assert len(out["teachers"]) == 1
    t0 = out["teachers"][0]
    assert t0["projection_matrix_path"] == "/p.pt"
    assert t0["weight"] == 1.0
    assert t0["model_name"] == "Qwen/Qwen3-4B"

    # An already-`teachers:` config passes through untouched.
    raw2 = {"teachers": [{"model_name": "A"}, {"model_name": "B"}]}
    assert normalize_multi_teacher_config(raw2)["teachers"] == [
        {"model_name": "A"},
        {"model_name": "B"},
    ]

    # Neither `teachers:` nor `teacher:` -> error.
    with pytest.raises(ValueError, match="teachers"):
        normalize_multi_teacher_config({"loss_fn": {}})


# ---------------------------------------------------------------------------
# Multi-teacher wiring (CPU-only, no GPU loss math)
# ---------------------------------------------------------------------------


class _FakeLossFn:
    """Minimal stand-in exposing the per-teacher metadata the trainer reads."""

    def __init__(self, projection_matrix_paths, teacher_ships_full, vocab_topk=8):
        self.num_teachers = len(projection_matrix_paths)
        self.projection_matrix_paths = projection_matrix_paths
        self.teacher_ships_full = teacher_ships_full
        self.vocab_topk = vocab_topk


def test_skip_keys_builder_cross_and_same_vocab():
    # teacher 0 cross-tokenizer (full logits); teacher 1 same-vocab top-K.
    loss_fn = _FakeLossFn(
        projection_matrix_paths=["/p0.pt", None],
        teacher_ships_full=[True, False],
    )
    keys = xtoken_non_student_seq_keys(loss_fn)
    # Cross-tokenizer teacher 0: IPC handle list + teacher tokens + the
    # teacher-seq / max_pairs alignment keys are skipped.
    assert "teacher_0_full_logits_ipc" in keys
    assert "teacher_0_input_ids" in keys
    assert "teacher_0_token_mask" in keys
    assert "alignment_0_pair_valid" in keys
    assert "alignment_0_teacher_chunk_id" in keys
    # Student-seq alignment keys ([B, T_s]) and num_chunks ([B]) are NOT skipped.
    assert "alignment_0_student_chunk_id" not in keys
    assert "alignment_0_num_chunks" not in keys
    # Same-vocab top-K teacher 1 now ships top-K over IPC, so its handle-list
    # key (a non-tensor) is skipped; it has no other seq-axis keys.
    assert "teacher_1_topk_ipc" in keys
    assert "teacher_1_full_logits_ipc" not in keys
    assert not any(k.startswith("alignment_1_") for k in keys)
    assert "teacher_1_input_ids" not in keys


def test_skip_keys_builder_same_vocab_full_logits():
    loss_fn = _FakeLossFn(
        projection_matrix_paths=[None],
        teacher_ships_full=[True],
    )
    # Same-vocab full-logits teacher: only the IPC handle list is skipped.
    assert xtoken_non_student_seq_keys(loss_fn) == frozenset(
        {"teacher_0_full_logits_ipc"}
    )


def test_run_teacher_forwards_packs_indexed_keys_and_runs_serially():
    # teacher 0 cross-tokenizer (full-logits IPC); teacher 1 same-vocab top-K
    # IPC. Both ship over the unified get_teacher_logits_ipc producer.
    loss_fn = _FakeLossFn(
        projection_matrix_paths=["/p0.pt", None],
        teacher_ships_full=[True, False],
        vocab_topk=8,
    )
    t0 = MagicMock()
    t0.get_teacher_logits_ipc.return_value = [{"rank_logits_ipc": 0}]
    t1 = MagicMock()
    t1.get_teacher_logits_ipc.return_value = [
        {"rank_topk_vals_ipc": 0, "rank_topk_idx_ipc": 1}
    ]
    # Cross-tokenizer teacher 0 needs teacher_0_*/alignment_0_*; same-vocab
    # teacher 1 reuses the student tokenization (no teacher_1_* token keys).
    batch = _make_batch(num_teachers=1)

    train_data = _run_teacher_forwards_and_pack([t0, t1], loss_fn, batch)

    # Cross-tokenizer teacher 0 -> full-logits IPC + its alignment payload.
    assert "teacher_0_full_logits_ipc" in train_data
    assert "alignment_0_pair_valid" in train_data
    assert "teacher_0_input_ids" in train_data
    # Same-vocab teacher 1 -> top-K IPC handle list, no dense tensors, no
    # projection/alignment keys.
    assert "teacher_1_topk_ipc" in train_data
    assert "teacher_1_topk_logits" not in train_data
    assert "teacher_1_topk_indices" not in train_data
    assert "alignment_1_pair_valid" not in train_data
    assert "teacher_1_full_logits_ipc" not in train_data
    # Unified producer: full mode for teacher 0, top-K mode (with k) for 1.
    t0.get_teacher_logits_ipc.assert_called_once()
    assert t0.get_teacher_logits_ipc.call_args.kwargs["mode"] == "full"
    t1.get_teacher_logits_ipc.assert_called_once()
    assert t1.get_teacher_logits_ipc.call_args.kwargs["mode"] == "topk"
    assert t1.get_teacher_logits_ipc.call_args.kwargs["k"] == 8
    # Serial collocated execution: each teacher onloaded then offloaded.
    for t in (t0, t1):
        t.prepare_for_lp_inference.assert_called_once()
        t.offload_after_refit.assert_called_once()


def test_setup_builds_one_policy_per_teacher():
    cfg = _make_master_config()
    # Add a second (same-vocab) teacher.
    cfg.teachers.append(
        TeacherConfig(
            **{
                "projection_matrix_path": None,
                "weight": 0.5,
                "dtensor_cfg": {
                    "enabled": True,
                    "_v2": True,
                    "tensor_parallel_size": 1,
                    "context_parallel_size": 1,
                },
                "max_total_sequence_length": 64,
                "make_sequence_length_divisible_by": 8,
                "tokenizer": {"name": "student-tok"},
            }
        )
    )
    student_tok = _make_tokenizer(32)
    teacher_toks = [_make_tokenizer(24), _make_tokenizer(32)]
    train_ds = MagicMock()
    train_ds.__len__ = MagicMock(return_value=4)

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
        mock_policy_cls.side_effect = lambda *a, **kw: MagicMock()

        (_student, teachers, *_rest) = setup(
            cfg,
            student_tokenizer=student_tok,
            teacher_tokenizers=teacher_toks,
            train_dataset=train_ds,
            val_dataset=None,
        )

    # One teacher Policy per entry (+ the student), and the colocation cap
    # accounts for all teacher groups + the student.
    assert isinstance(teachers, list) and len(teachers) == 2
    assert mock_policy_cls.call_count == 3  # 2 teachers + 1 student
    assert mock_cluster.call_args.kwargs["max_colocated_worker_groups"] == 3
    # Per-teacher metadata injected as parallel lists.
    injected_cfg = mock_loss_cls.call_args.args[0]
    assert injected_cfg["projection_matrix_paths"] == ["/tmp/dummy-projection.pt", None]
    assert injected_cfg["teacher_weights"] == [1.0, 0.5]
    assert injected_cfg["teacher_vocab_sizes"] == [24, 32]


def test_setup_rejects_same_vocab_teacher_with_mismatched_vocab():
    # A null-projection (same-vocab) teacher whose vocab != the student's is a
    # config error — same-vocab detection is by null projection path, but the
    # teacher must actually share the student vocab for the direct KL. Caught
    # in setup() (the right place — it has the real tokenizers; tokenizer
    # *names* would wrongly flag Llama-3.2-3B vs -1B, which share a vocab).
    cfg = _make_master_config()
    cfg.teachers[0].projection_matrix_path = None  # mark same-vocab
    student_tok = _make_tokenizer(32)
    teacher_toks = [_make_tokenizer(24)]  # 24 != 32 -> mismatch
    with (
        patch.object(xt_mod, "RayVirtualCluster") as mock_cluster,
        pytest.raises(AssertionError, match="same-vocab"),
    ):
        setup(
            cfg,
            student_tokenizer=student_tok,
            teacher_tokenizers=teacher_toks,
            train_dataset=MagicMock(),
            val_dataset=None,
        )
    # The check fires before any cluster/policy construction.
    assert mock_cluster.call_count == 0


# ---------------------------------------------------------------------------
# averaged_logits direct-KL guard
# ---------------------------------------------------------------------------


def _bare_averaged_logits_loss_fn(projection_matrix_paths):
    """A ``CrossTokenizerDistillationLossFn`` carrying only the attrs the
    ``averaged_logits`` guard reads, built via ``__new__`` to skip the heavy
    config-driven ``__init__``."""
    fn = CrossTokenizerDistillationLossFn.__new__(CrossTokenizerDistillationLossFn)
    fn.num_teachers = len(projection_matrix_paths)
    fn.projection_matrix_paths = list(projection_matrix_paths)
    fn.teacher_weights = [1.0] * fn.num_teachers
    return fn


def test_averaged_logits_cross_tokenizer_skips_direct_kl_fast_path():
    # Two cross-tokenizer teachers (non-null projection paths) whose logits
    # happen to share a shape must NOT take the direct per-position KL fast
    # path: it assumes the student's tokenizer and would mismatch the
    # student's token_mask when the teacher length differs (T_t != T_s).
    fn = _bare_averaged_logits_loss_fn(["t0_proj.pt", "t1_proj.pt"])

    fallback_calls = []

    def _fallback(i, *args, **kwargs):
        fallback_calls.append(i)
        return torch.tensor(0.0), {}

    def _fast_path(*args, **kwargs):
        raise AssertionError("direct_full_vocab_kl reached for cross-tokenizer teachers")

    fn._compute_teacher_kd = _fallback
    fn._direct_full_vocab_kl = _fast_path

    # Equal-shaped teacher logits (the misleading signal) but teacher length 13
    # vs the student's length 10.
    teacher_logits = torch.zeros(2, 13, 24)
    teacher_full = {0: teacher_logits, 1: teacher_logits.clone()}
    student_logits = torch.zeros(2, 10, 32)

    total_kd, _ = fn._averaged_logits_kd(
        student_logits, {}, teacher_full, torch.tensor(20.0)
    )
    assert fallback_calls == [0, 1]  # per-teacher fallback path, one call each
    assert total_kd is not None


def test_averaged_logits_same_tokenizer_takes_direct_kl_fast_path():
    # All-null projection paths => genuine same-tokenizer teachers, so the
    # averaging + single direct-KL fast path is the correct branch.
    fn = _bare_averaged_logits_loss_fn([None, None])

    fast_calls = []

    def _fast_path(*args, **kwargs):
        fast_calls.append(1)
        return torch.tensor(1.23)

    fn._direct_full_vocab_kl = _fast_path
    fn._compute_teacher_kd = MagicMock(
        side_effect=AssertionError("fallback reached for same-tokenizer teachers")
    )

    logits = torch.zeros(2, 10, 32)
    teacher_full = {0: logits.clone(), 1: logits.clone()}

    kd, metrics = fn._averaged_logits_kd(logits, {}, teacher_full, torch.tensor(20.0))
    assert fast_calls == [1]  # fast path taken exactly once
    assert "kl_loss" in metrics
    assert kd.item() == pytest.approx(1.23)


# ---------------------------------------------------------------------------
# teacher weight-metric score masking
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("metric", ["ce", "entropy", "max_prob"])
def test_teacher_weight_score_ignores_padded_positions(metric):
    # The per-teacher weight/selection score must exclude padded positions:
    # padding logits are near-uniform noise and would otherwise dominate the
    # score on long-padded batches. Corrupting everything in the padded tail
    # must not change the score.
    fn = CrossTokenizerDistillationLossFn.__new__(CrossTokenizerDistillationLossFn)
    fn.sum_weights_metric = metric

    torch.manual_seed(0)
    batch, seqlen, vocab = 2, 5, 8
    logits = torch.randn(batch, seqlen, vocab)
    ids = torch.randint(0, vocab, (batch, seqlen))
    # Last two positions of each sample are padding.
    token_mask = torch.tensor(
        [[1, 1, 1, 0, 0], [1, 1, 1, 0, 0]], dtype=torch.float32
    )
    sample_mask = torch.ones(batch)

    score = fn._teacher_weight_score(logits, ids, token_mask, sample_mask)

    # Corrupt the padded tail; a properly masked score must not move.
    logits_corrupt = logits.clone()
    logits_corrupt[:, 3:, :] = 1e4
    ids_corrupt = ids.clone()
    ids_corrupt[:, 3:] = 0
    score_corrupt = fn._teacher_weight_score(
        logits_corrupt, ids_corrupt, token_mask, sample_mask
    )

    assert torch.allclose(score, score_corrupt, atol=1e-5)


def test_teacher_weight_score_masks_dropped_samples():
    # sample_mask=0 rows must not contribute either.
    fn = CrossTokenizerDistillationLossFn.__new__(CrossTokenizerDistillationLossFn)
    fn.sum_weights_metric = "max_prob"

    torch.manual_seed(0)
    logits = torch.randn(2, 4, 8)
    ids = torch.randint(0, 8, (2, 4))
    token_mask = torch.ones(2, 4)
    sample_mask = torch.tensor([1.0, 0.0])  # second sample dropped

    score = fn._teacher_weight_score(logits, ids, token_mask, sample_mask)

    logits_corrupt = logits.clone()
    logits_corrupt[1] = 1e4  # corrupt the dropped sample
    score_corrupt = fn._teacher_weight_score(
        logits_corrupt, ids, token_mask, sample_mask
    )

    assert torch.allclose(score, score_corrupt, atol=1e-5)
