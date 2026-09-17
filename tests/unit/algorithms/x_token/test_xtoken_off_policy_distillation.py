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
functions exercise the high-level invariants the reviewer flagged.
CPU-only, no Ray, no CUDA.

Cross-tokenizer teachers may ship either dense full-vocab logits or row-wise
top-k logits plus full-vocab logZ over CUDA IPC. Same-vocab teachers remain
dense because their direct-KL and teacher-scoring paths require full logits.
"""

from __future__ import annotations

import math
import os
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pydantic import ValidationError
from torchdata.stateful_dataloader import StatefulDataLoader

import nemo_rl.algorithms.xtoken_off_policy_distillation as xt_mod
from nemo_rl.algorithms.loss.loss_functions import CrossTokenizerDistillationLossFn
from nemo_rl.algorithms.xtoken_off_policy_distillation import (
    MasterConfig,
    TeacherAlignerConfig,
    TeacherConfig,
    _build_teacher_force_include_token_ids,
    _default_off_policy_distillation_save_state,
    _packing_batch_uids_from_state,
    build_xtoken_lockstep_packing_plan,
    build_xtoken_logical_batch_digest_record,
    export_teacher_logits_and_pack,
    log_xtoken_logical_batch_digest,
    log_xtoken_packing_telemetry,
    reduce_mb_metric,
    setup,
    validate,
    validate_xtoken_packing_setup,
    validate_xtoken_tokenizer_reuse,
    xtoken_non_student_seq_keys,
    xtoken_off_policy_distillation_train,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def has_gloo() -> bool:
    """Whether torch.distributed has a usable gloo backend."""
    return torch.distributed.is_available() and torch.distributed.is_gloo_available()


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
        "sample_id": [f"sample-{i}" for i in range(batch_size)],
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
    # validate() pads ragged val batches via BatchedDataDict.size, so the mock
    # batches must be BatchedDataDict (the train path reads them as a dict too).
    return BatchedDataDict(batch)


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
                "train_global_batch_size": 1,
                "train_micro_batch_size": 1,
                "tokenizer": {"name": "student-tok"},
            },
            "teachers": [
                TeacherConfig(
                    **{
                        "aligner": {
                            "projection_matrix_path": "/tmp/dummy-projection.pt"
                        },
                        "weight": 1.0,
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
                    }
                )
            ],
            "loss_fn": {
                "temperature": 1.0,
                "vocab_topk": 8,
                "reverse_kl": False,
                "kl_loss_weight": 1.0,
                "ce_loss_scale": 1.0,
                "dynamic_loss_scaling": False,
                "kd_loss_mode": "sum",
                "sum_weights_metric": None,
                "alpha": 1.0,
                "normalize_teacher_by_vocab": False,
                "kl_chunk_shift": False,
                "prefix_bidir_v3_noise_filter_topk": 0,
                "teacher_topk_ipc_k": 0,
                "teacher_topk_ipc_support_mode": "row_topk",
                "teacher_topk_ipc_keep_realized": True,
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
    tok.get_vocab.return_value = {
        f"token-{token_id}": token_id for token_id in range(vocab_size)
    }
    tok.is_fast = True
    tok.backend_tokenizer.to_str.return_value = "stable-fast-backend"
    tok.special_tokens_map_extended = {}
    tok.all_special_ids = []
    tok.chat_template = None
    return tok


def _enable_lockstep_packing(
    cfg: MasterConfig, *, global_batch_size: int = 1, capacity: int = 64
) -> None:
    cfg.distillation["num_prompts_per_step"] = global_batch_size
    cfg.data["train"] = {
        "dataset_name": "arrow_text",
        "characters_per_sample": None,
    }
    cfg.data["num_packed_rows"] = 1

    def enable_policy_config(policy_config):
        policy_config["model_name"] = "transformer-test-model"
        policy_config["train_global_batch_size"] = global_batch_size
        policy_config["train_micro_batch_size"] = 1
        policy_config["dynamic_batching"] = {"enabled": False}
        policy_config["sequence_packing"] = {
            "enabled": True,
            "train_mb_tokens": capacity,
            "logprob_mb_tokens": capacity,
            "algorithm": "lockstep_first_fit_decreasing",
            "fuse_loss": False,
        }

    enable_policy_config(cfg.policy)
    for teacher_idx, teacher in enumerate(cfg.teachers):
        teacher_config = teacher.policy_config()
        enable_policy_config(teacher_config)
        cfg.teachers[teacher_idx] = TeacherConfig(
            **teacher_config,
            aligner=teacher.aligner.model_dump(),
            weight=teacher.weight,
        )


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_xtoken_components():
    student_policy = MagicMock()
    student_policy.data_parallel_size = 1
    student_policy.train.return_value = {
        "loss": torch.tensor(0.5),
        "grad_norm": torch.tensor(1.0),
        "all_mb_metrics": {
            "global_valid_toks": [10],
            "kl_loss": [0.3],
            "ipc_reconstruction_fallbacks": [1],
            "ipc_reconstruction_fallbacks_t0": [1],
        },
    }

    teacher_policy = MagicMock()
    teacher_policy.data_parallel_size = 1
    teacher_policy.get_full_logits_ipc.return_value = [
        {"teacher_shards": [{"actual_shape": (4, 32), "dtype": torch.float32}]}
    ]

    train_dataloader = _mock_dataloader(num_batches=10)
    val_dataloader = _mock_dataloader(num_batches=2)

    # The trainer reads per-teacher metadata off the loss fn to build the
    # skip-keys set and drive the teacher-forward loop. One cross-tokenizer
    # teacher: non-null projection path (every teacher ships full logits).
    loss_fn = MagicMock()
    loss_fn.num_teachers = 1
    loss_fn.projection_matrix_paths = ["/tmp/dummy-projection.pt"]
    loss_fn.teacher_vocab_sizes = [24]
    loss_fn.cfg = {
        "temperature": 1.0,
        "kl_chunk_shift": False,
        "prefix_bidir_v3_noise_filter_topk": 0,
        "teacher_topk_ipc_k": 0,
        "teacher_topk_ipc_support_mode": "row_topk",
        "teacher_topk_ipc_keep_realized": True,
    }
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


@pytest.fixture(autouse=True)
def _resolve_synthetic_packing_model_metadata(monkeypatch):
    """Keep synthetic lockstep configs offline while exercising metadata checks."""
    original = xt_mod._load_xtoken_packed_model_config

    def load(config):
        if config.get("model_name") == "transformer-test-model":
            return SimpleNamespace(
                model_type="llama",
                architectures=["LlamaForCausalLM"],
                to_dict=lambda: {
                    "model_type": "llama",
                    "architectures": ["LlamaForCausalLM"],
                },
            )
        return original(config)

    monkeypatch.setattr(xt_mod, "_load_xtoken_packed_model_config", load)


# ---------------------------------------------------------------------------
# setup() backend & vocab-injection asserts
# ---------------------------------------------------------------------------


def _patched_setup_call(
    master_config, *, student_vocab=32, teacher_vocab=24, train_batches=4
):
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
        patch.object(xt_mod, "CrossTokenizerCollator") as mock_collator_cls,
        patch.object(xt_mod, "CrossTokenizerDistillationLossFn") as mock_loss_cls,
        patch.object(xt_mod, "StatefulDataLoader") as mock_dl_cls,
        patch.object(xt_mod, "assert_teacher_student_batch_grid"),
        patch.object(xt_mod, "assert_xtoken_ipc_node_local"),
    ):
        mock_cp_cls.return_value.get_latest_checkpoint_path.return_value = None
        mock_cp_cls.return_value.load_training_info.return_value = None
        mock_cp_cls.return_value.get_resume_paths.return_value = (None, None)
        train_dl = MagicMock(spec=StatefulDataLoader)
        train_dl.__len__ = MagicMock(return_value=train_batches)
        val_dl = MagicMock(spec=StatefulDataLoader)
        val_dl.__len__ = MagicMock(return_value=2)
        mock_dl_cls.side_effect = [train_dl, val_dl]
        mock_policy_cls.side_effect = lambda *a, **kw: MagicMock(data_parallel_size=1)

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
            "collator": mock_collator_cls,
            "checkpointer": mock_cp_cls,
        }


def test_teacher_aligner_config_defaults():
    teacher = TeacherConfig(model_name="teacher")

    assert isinstance(teacher.aligner, TeacherAlignerConfig)
    assert teacher.aligner.projection_matrix_path is None
    assert teacher.aligner.drop_first_assistant_chunk_kl is False
    assert teacher.aligner.pseudo_target_path is None
    assert teacher.aligner.reverse_pseudo_target_path is None


def test_teacher_aligner_config_explicit_values_serialize_and_stay_out_of_policy():
    teacher = TeacherConfig(
        model_name="teacher",
        aligner={
            "projection_matrix_path": "/tmp/projection.pt",
            "drop_first_assistant_chunk_kl": True,
            "pseudo_target_path": "/tmp/forward.pt",
            "reverse_pseudo_target_path": "/tmp/reverse.pt",
        },
    )

    dumped = teacher.model_dump()
    assert dumped["aligner"] == {
        "projection_matrix_path": "/tmp/projection.pt",
        "drop_first_assistant_chunk_kl": True,
        "pseudo_target_path": "/tmp/forward.pt",
        "reverse_pseudo_target_path": "/tmp/reverse.pt",
    }
    assert "projection_matrix_path" not in dumped
    assert "aligner" not in teacher.policy_config()


def test_legacy_teacher_projection_path_migrates_and_warns():
    with pytest.warns(FutureWarning, match="aligner.projection_matrix_path"):
        teacher = TeacherConfig(projection_matrix_path="/tmp/legacy.pt")

    assert teacher.aligner.projection_matrix_path == "/tmp/legacy.pt"
    assert "projection_matrix_path" not in teacher.model_dump()


def test_conflicting_legacy_and_nested_projection_paths_fail():
    with pytest.raises(ValidationError, match="conflicting projection matrix paths"):
        TeacherConfig(
            projection_matrix_path="/tmp/legacy.pt",
            aligner={"projection_matrix_path": "/tmp/nested.pt"},
        )


@pytest.mark.parametrize("field", ["pseudo_target_path", "reverse_pseudo_target_path"])
def test_root_pseudo_target_paths_are_rejected(field):
    with pytest.raises(
        ValidationError,
        match="must be nested under teachers\\[i\\]\\.aligner",
    ):
        TeacherConfig(**{field: "/tmp/table.pt"})


def test_shared_drop_first_assistant_chunk_kl_is_rejected():
    with pytest.raises(
        ValidationError,
        match=r"teachers\[i\]\.aligner\.drop_first_assistant_chunk_kl",
    ):
        MasterConfig.model_validate({"data": {"drop_first_assistant_chunk_kl": True}})


def test_empty_teachers_list_rejected_at_config_load():
    """An empty ``teachers:`` list fails validation at config load."""
    with pytest.raises(ValidationError) as exc_info:
        MasterConfig(teachers=[])
    assert any(
        e["loc"] == ("teachers",) and e["type"] == "too_short"
        for e in exc_info.value.errors()
    )


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


def test_setup_router_replay_requires_megatron_student():
    cfg = _make_master_config()
    cfg.policy["router_replay"] = {"enabled": True}
    with (
        patch.object(xt_mod, "RayVirtualCluster") as mock_cluster,
        pytest.raises(ValueError, match="requires the Megatron student policy backend"),
    ):
        setup(
            cfg,
            student_tokenizer=_make_tokenizer(32),
            teacher_tokenizers=[_make_tokenizer(24)],
            train_dataset=MagicMock(),
            val_dataset=None,
        )
    assert mock_cluster.call_count == 0


def _dtensor_cfg(*, enabled=True, v2=True, tp=1, cp=1):
    return {
        "enabled": enabled,
        "_v2": v2,
        "tensor_parallel_size": tp,
        "context_parallel_size": cp,
    }


def _megatron_cfg(*, enabled=True, tp=1, pp=1, cp=1):
    return {
        "enabled": enabled,
        "tensor_model_parallel_size": tp,
        "pipeline_model_parallel_size": pp,
        "context_parallel_size": cp,
    }


def test_entity_parallelism_dtensor_v2_returns_tp_cp_pp():
    # DTensor has no pipeline axis, so it always reports pp == 1.
    cfg = {"dtensor_cfg": _dtensor_cfg(tp=4, cp=2)}
    assert xt_mod._xtoken_entity_parallelism(cfg, label="teachers[0]") == (4, 2, 1)


def test_entity_parallelism_megatron_pp1cp1_returns_tp_cp_pp():
    cfg = {
        "dtensor_cfg": _dtensor_cfg(enabled=False, v2=False),
        "megatron_cfg": _megatron_cfg(tp=2, pp=1, cp=1),
    }
    assert xt_mod._xtoken_entity_parallelism(cfg, label="teachers[0]") == (2, 1, 1)


def test_entity_parallelism_megatron_accepts_pp_gt_1():
    # PP is supported: only the last stage holds logits, so only it contributes
    # full-logits IPC handles and the earlier stages drop out in
    # aggregate_per_sample_handles.
    cfg = {
        "dtensor_cfg": _dtensor_cfg(enabled=False, v2=False),
        "megatron_cfg": _megatron_cfg(tp=2, pp=2, cp=1),
    }
    assert xt_mod._xtoken_entity_parallelism(cfg, label="teachers[0]") == (2, 1, 2)


def test_entity_parallelism_megatron_accepts_cp_gt_1():
    # Megatron TP/CP/PP are all supported (the loss is parallelism-invariant).
    cfg = {
        "dtensor_cfg": _dtensor_cfg(enabled=False, v2=False),
        "megatron_cfg": _megatron_cfg(tp=2, pp=1, cp=2),
    }
    assert xt_mod._xtoken_entity_parallelism(cfg, label="teachers[0]") == (2, 2, 1)


def test_entity_parallelism_rejects_neither_backend():
    # dtensor enabled but not _v2, and no megatron_cfg -> neither backend.
    cfg = {"dtensor_cfg": _dtensor_cfg(enabled=True, v2=False)}
    with pytest.raises(AssertionError, match="either DTensor-V2"):
        xt_mod._xtoken_entity_parallelism(cfg, label="teachers[0]")


def test_setup_injects_vocab_sizes_into_loss_config():
    cfg = _make_master_config()
    cfg.teachers[0].aligner.pseudo_target_path = "/tmp/forward.pt"
    cfg.teachers[0].aligner.reverse_pseudo_target_path = "/tmp/reverse.pt"
    original_loss_cfg = deepcopy(cfg.loss_fn)

    _, mocks = _patched_setup_call(cfg, student_vocab=128, teacher_vocab=256)

    mocks["loss"].assert_called_once()
    injected_cfg = mocks["loss"].call_args.args[0]
    assert injected_cfg["student_vocab_size"] == 128
    # Per-teacher metadata is injected as parallel lists (one teacher here).
    assert injected_cfg["teacher_vocab_sizes"] == [256]
    assert injected_cfg["projection_matrix_paths"] == ["/tmp/dummy-projection.pt"]
    assert injected_cfg["teacher_weights"] == [1.0]
    assert injected_cfg["pseudo_target_paths"] == ["/tmp/forward.pt"]
    assert injected_cfg["reverse_pseudo_target_paths"] == ["/tmp/reverse.pt"]
    assert mocks["collator"].call_args.kwargs[
        "drop_first_assistant_chunk_kl_by_teacher"
    ] == [False]
    # Original master_config not mutated by the injection.
    assert cfg.loss_fn == original_loss_cfg


def test_setup_sets_derived_train_iters_on_megatron_teacher_and_student():
    cfg = _make_master_config(max_num_steps=10, max_num_epochs=2)
    cfg.policy["dtensor_cfg"]["enabled"] = False
    cfg.policy["megatron_cfg"] = _megatron_cfg()
    cfg.teachers[0].dtensor_cfg["enabled"] = False
    cfg.teachers[0].megatron_cfg = _megatron_cfg()

    _, mocks = _patched_setup_call(cfg, train_batches=3)

    # min(max_num_steps=10, max_num_epochs=2 * train_batches=3) == 6.
    teacher_config = mocks["policy"].call_args_list[0].kwargs["config"]
    student_config = mocks["policy"].call_args_list[1].kwargs["config"]
    assert teacher_config["megatron_cfg"]["train_iters"] == 6
    assert student_config["megatron_cfg"]["train_iters"] == 6


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
        patch.object(xt_mod, "assert_teacher_student_batch_grid"),
        patch.object(xt_mod, "assert_xtoken_ipc_node_local"),
    ):
        mock_cp_cls.return_value.get_latest_checkpoint_path.return_value = None
        mock_cp_cls.return_value.load_training_info.return_value = None
        mock_cp_cls.return_value.get_resume_paths.return_value = (None, None)
        mock_dl_cls.side_effect = lambda *a, **kw: MagicMock(spec=StatefulDataLoader)
        mock_policy_cls.side_effect = lambda *a, **kw: MagicMock(data_parallel_size=1)

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


def test_train_surfaces_reduced_teacher_routing_metrics(mock_xtoken_components):
    c = mock_xtoken_components
    c.master_config.distillation.update(max_num_steps=1, max_num_epochs=1)
    c.val_dataloader = None
    c.student_policy.train.return_value["all_mb_metrics"].update(
        {
            "teacher_0/routed_samples": [2, 3],
            "teacher_0/routed_tokens": [7, 11],
            "teacher_0/weighted_kl": [0.2, 0.3],
        }
    )

    _run_train(c)

    train_log_call = next(
        call
        for call in c.logger.log_metrics.call_args_list
        if call.kwargs.get("prefix") == "train"
    )
    metrics = train_log_call.args[0]
    assert metrics["teacher_0/routed_samples"] == pytest.approx(5)
    assert metrics["teacher_0/routed_tokens"] == pytest.approx(18)
    assert metrics["teacher_0/weighted_kl"] == pytest.approx(0.5)


def test_ft_save_period_triggers_periodic_saves(mock_xtoken_components):
    """ft_save_period triggers checkpoint saves independent of save_period."""
    c = mock_xtoken_components
    c.master_config.distillation["max_num_steps"] = 5
    c.master_config.distillation["max_num_epochs"] = 1
    c.master_config.checkpointing["enabled"] = True
    c.master_config.checkpointing["save_period"] = 100  # only the final step saves
    c.master_config.checkpointing["ft_save_period"] = 2
    c.master_config.checkpointing["metric_name"] = None
    c.checkpointer.init_tmp_checkpoint.return_value = "/tmp/ft_ckpt_test/tmp_step"

    with patch("nemo_rl.algorithms.xtoken_off_policy_distillation.torch.save"):
        _run_train(c)

    # ft_save_period=2 -> steps 2, 4; save_period=100 contributes only the last step (5).
    saved_steps = [
        call.args[0] for call in c.checkpointer.init_tmp_checkpoint.call_args_list
    ]
    assert saved_steps == [2, 4, 5]


@pytest.mark.parametrize(
    ("offload_student", "offload_optimizer", "restore_method", "unused_method"),
    [
        (True, True, "offload_after_refit", "offload_before_refit"),
        (False, True, "offload_before_refit", "offload_after_refit"),
    ],
)
def test_checkpoint_restores_student_between_teacher_state_after_save(
    mock_xtoken_components,
    offload_student,
    offload_optimizer,
    restore_method,
    unused_method,
):
    c = mock_xtoken_components
    c.master_config.distillation.update(
        max_num_steps=1,
        max_num_epochs=1,
        offload_student_after_step=offload_student,
    )
    c.master_config.policy["offload_optimizer_for_logprob"] = offload_optimizer
    c.master_config.checkpointing.update(enabled=True, save_period=1)
    c.checkpointer.init_tmp_checkpoint.return_value = "/tmp/residency/tmp_step"

    events = MagicMock()
    events.attach_mock(c.student_policy, "student")
    events.attach_mock(c.checkpointer, "checkpointer")
    with patch("nemo_rl.algorithms.xtoken_off_policy_distillation.torch.save"):
        _run_train(c)

    event_names = [event[0] for event in events.mock_calls]
    save_index = event_names.index("student.save_checkpoint")
    finalize_index = event_names.index("checkpointer.begin_finalization")
    restore_indices = [
        index
        for index, event_name in enumerate(event_names)
        if event_name == f"student.{restore_method}"
    ]

    assert save_index < finalize_index < restore_indices[-1]
    assert len(restore_indices) == 2
    getattr(c.student_policy, unused_method).assert_not_called()


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


def test_packing_batch_uid_high_water_mark_survives_multibatch_validation_resume(
    mock_xtoken_components,
):
    c = mock_xtoken_components
    _enable_lockstep_packing(c.master_config, global_batch_size=1, capacity=64)
    c.master_config.distillation.update(
        max_num_steps=1,
        max_num_epochs=10,
        val_period=1,
        val_at_start=False,
        val_at_end=False,
    )
    c.val_dataloader = _mock_dataloader(num_batches=2)
    c.master_config.checkpointing.update(
        enabled=True,
        save_period=1,
        ft_save_period=None,
        metric_name=None,
    )
    c.checkpointer.init_tmp_checkpoint.return_value = "/tmp/uid_resume/tmp_step"

    with patch("nemo_rl.algorithms.xtoken_off_policy_distillation.torch.save"):
        _run_train(c)

    # Reconstruct the state as JSON checkpoint loading would, then continue for
    # one more step. The first run consumed one train UID plus two validation
    # UIDs, so resume must start at 3 rather than deriving 2 from total_steps.
    c.save_state = dict(c.save_state)
    c.master_config.distillation["max_num_steps"] = 2
    with patch("nemo_rl.algorithms.xtoken_off_policy_distillation.torch.save"):
        _run_train(c)

    observed_uids = [
        call.kwargs["packing_plan"].batch_uid
        for call in c.student_policy.train.call_args_list
    ]
    assert observed_uids == [0, 1, 2, 3, 4, 5]
    assert c.save_state["next_packing_batch_uid"] == 6


def test_packing_batch_uid_accepts_checkpoint_without_high_water_mark():
    state = _default_off_policy_distillation_save_state()
    del state["next_packing_batch_uid"]
    state["total_steps"] = 7

    batch_uids = _packing_batch_uids_from_state(state)

    assert [next(batch_uids), next(batch_uids)] == [14, 15]
    assert state["next_packing_batch_uid"] == 16


@pytest.mark.parametrize(
    "fallback_metrics,error_match",
    [
        (
            {"ipc_reconstruction_fallbacks": [1]},
            "missing consumer-side metrics",
        ),
        (
            {
                "ipc_reconstruction_fallbacks": [2],
                "ipc_reconstruction_fallbacks_t0": [1],
            },
            "internally inconsistent",
        ),
    ],
)
def test_packed_dense_ipc_telemetry_fails_closed_and_releases_buffers(
    mock_xtoken_components, fallback_metrics, error_match
):
    c = mock_xtoken_components
    _enable_lockstep_packing(c.master_config, global_batch_size=1, capacity=64)
    c.master_config.distillation.update(
        max_num_steps=1,
        max_num_epochs=1,
        val_period=0,
        val_at_start=False,
        val_at_end=False,
    )
    c.student_policy.train.return_value = _make_train_results_with(
        {"global_valid_toks": [10], "kl_loss": [0.3], **fallback_metrics}
    )

    with pytest.raises(RuntimeError, match=error_match):
        _run_train(c)

    c.teacher_policy.release_ipc_buffer.assert_called_once_with()


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
        skip_keys=xtoken_non_student_seq_keys(c.loss_fn),
    )

    assert "loss" in metrics
    assert "kl_loss" in metrics
    assert "ce_loss" in metrics
    assert "kl_common" not in metrics
    assert "l1_uncommon" not in metrics


def test_validate_collects_only_aggregate_metrics(mock_xtoken_components):
    c = mock_xtoken_components
    # validate summarizes only the aggregate loss / kl_loss / ce_loss / the
    # aggregate gold path's kl_common / l1_uncommon. The per-teacher suffixed
    # components (kl_common_t{i} / l1_uncommon_t{i}) are not collected here, so
    # they don't appear in the val metrics.
    c.student_policy.train.return_value = _make_train_results_with(
        {"kl_common_t0": [0.2], "l1_uncommon_t0": [0.1], "ce_loss": [0.4]}
    )

    metrics, _timings = validate(
        c.student_policy,
        [c.teacher_policy],
        c.val_dataloader,
        c.loss_fn,
        c.master_config,
        skip_keys=xtoken_non_student_seq_keys(c.loss_fn),
    )

    assert "loss" in metrics
    assert "ce_loss" in metrics
    assert "kl_common" not in metrics
    assert "l1_uncommon" not in metrics
    assert "kl_loss" not in metrics


def test_validate_passes_ragged_packed_target_size_as_gbs(mock_xtoken_components):
    c = mock_xtoken_components
    _enable_lockstep_packing(c.master_config, global_batch_size=8, capacity=64)
    c.student_policy.data_parallel_size = 2
    c.teacher_policy.data_parallel_size = 2
    ragged_batch = _make_batch(batch_size=3)
    c.val_dataloader = MagicMock(spec=StatefulDataLoader)
    c.val_dataloader.__iter__ = lambda self: iter([ragged_batch])

    validate(
        c.student_policy,
        [c.teacher_policy],
        c.val_dataloader,
        c.loss_fn,
        c.master_config,
        skip_keys=xtoken_non_student_seq_keys(c.loss_fn),
    )

    train_kwargs = c.student_policy.train.call_args.kwargs
    assert train_kwargs["gbs"] == 4
    assert len(train_kwargs["packing_plan"].canonical_batch_item_ids) == 4


@pytest.mark.parametrize(
    ("offload_student", "offload_optimizer", "restore_method", "unused_method"),
    [
        (True, True, "offload_after_refit", "offload_before_refit"),
        (False, True, "offload_before_refit", "offload_after_refit"),
    ],
)
def test_validate_restores_student_between_teacher_state(
    mock_xtoken_components,
    offload_student,
    offload_optimizer,
    restore_method,
    unused_method,
):
    c = mock_xtoken_components
    c.master_config.distillation["offload_student_after_step"] = offload_student
    c.master_config.policy["offload_optimizer_for_logprob"] = offload_optimizer

    events = MagicMock()
    events.attach_mock(c.student_policy, "student")
    events.attach_mock(c.teacher_policy, "teacher")
    validate(
        c.student_policy,
        [c.teacher_policy],
        c.val_dataloader,
        c.loss_fn,
        c.master_config,
        skip_keys=xtoken_non_student_seq_keys(c.loss_fn),
    )

    event_names = [call[0] for call in events.mock_calls]
    train_indices = [
        index
        for index, event_name in enumerate(event_names)
        if event_name == "student.train"
    ]
    restore_indices = [
        index
        for index, event_name in enumerate(event_names)
        if event_name == f"student.{restore_method}"
    ]
    teacher_prepare_indices = [
        index
        for index, event_name in enumerate(event_names)
        if event_name == "teacher.prepare_for_lp_inference"
    ]

    assert len(train_indices) == len(restore_indices) == 2
    assert train_indices[0] < restore_indices[0] < teacher_prepare_indices[1]
    assert train_indices[1] < restore_indices[1]
    assert getattr(c.student_policy, restore_method).call_count == 2
    getattr(c.student_policy, unused_method).assert_not_called()


# ---------------------------------------------------------------------------
# IPC buffer release-on-failure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_teachers", [1, 2])
def test_ipc_buffer_released_for_every_teacher_on_train_failure(
    mock_xtoken_components, num_teachers
):
    c = mock_xtoken_components
    c.student_policy.train.side_effect = RuntimeError("boom")
    c.master_config.distillation["max_num_steps"] = 1
    # No validation passes, so release is driven solely by the failing train
    # step's finally (one release per teacher).
    c.master_config.distillation["val_at_start"] = False
    c.master_config.distillation["val_period"] = 0
    c.master_config.distillation["val_at_end"] = False

    # N same-vocab teachers: the release loop is teacher-type-agnostic, and
    # same-vocab teachers reuse the student tokenization (the mock batch only
    # carries teacher_0_* keys, so this stays valid for any N).
    teacher_policies = [MagicMock(data_parallel_size=1) for _ in range(num_teachers)]
    for t in teacher_policies:
        t.get_full_logits_ipc.return_value = [{"payload_ipc": (4, 32)}]
    c.loss_fn.num_teachers = num_teachers
    c.loss_fn.projection_matrix_paths = [None] * num_teachers
    # teacher_mbs is derived per entry in master_config.teachers, so it must
    # have one entry per teacher policy.
    c.master_config.teachers = [
        c.master_config.teachers[0] for _ in range(num_teachers)
    ]

    with pytest.raises(RuntimeError):
        xtoken_off_policy_distillation_train(
            c.student_policy,
            teacher_policies,
            c.train_dataloader,
            c.val_dataloader,
            c.loss_fn,
            c.logger,
            c.checkpointer,
            c.save_state,
            c.master_config,
        )

    # The finally must release EVERY teacher's IPC buffer, not just the first
    # (a misplaced finally / early break that frees only teacher 0 would pass a
    # `>= 1` check while leaking the rest).
    for t in teacher_policies:
        assert t.release_ipc_buffer.call_count == 1


# ---------------------------------------------------------------------------
# Multi-teacher wiring (CPU-only, no GPU loss math)
# ---------------------------------------------------------------------------


class _FakeLossFn:
    """Minimal stand-in exposing the per-teacher metadata the trainer reads.

    Exposes the indexed metadata needed by the trainer and skip-key builder.
    """

    def __init__(
        self,
        projection_matrix_paths,
        *,
        teacher_topk_ipc_k=0,
        teacher_vocab_sizes=None,
    ):
        self.num_teachers = len(projection_matrix_paths)
        self.projection_matrix_paths = projection_matrix_paths
        self.teacher_vocab_sizes = teacher_vocab_sizes or [32] * self.num_teachers
        self.cfg = {
            "temperature": 1.0,
            "kl_chunk_shift": False,
            "prefix_bidir_v3_noise_filter_topk": 0,
            "teacher_topk_ipc_k": teacher_topk_ipc_k,
            "teacher_topk_ipc_support_mode": "row_topk",
            "teacher_topk_ipc_keep_realized": True,
        }


def test_skip_keys_builder_cross_and_same_vocab():
    # teacher 0 cross-tokenizer (full logits); teacher 1 same-vocab.
    loss_fn = _FakeLossFn(projection_matrix_paths=["/p0.pt", None])
    keys = xtoken_non_student_seq_keys(loss_fn)
    # Cross-tokenizer teacher 0: IPC handle list + teacher tokens + the
    # teacher-seq / max_pairs alignment keys are skipped.
    assert "teacher_0_full_logits_ipc" in keys
    assert "teacher_0_sparse_logits_ipc" in keys
    assert "teacher_0_input_ids" in keys
    assert "teacher_0_token_mask" in keys
    assert "alignment_0_pair_valid" in keys
    assert "alignment_0_teacher_chunk_id" in keys
    # Student-seq alignment keys ([B, T_s]) and num_chunks ([B]) are NOT skipped.
    assert "alignment_0_student_chunk_id" not in keys
    assert "alignment_0_num_chunks" not in keys
    # Same-vocab teacher 1 also ships full logits over IPC, so its handle-list
    # key (a non-tensor) is skipped; it reuses the student tokenization, so it
    # has no teacher-seq token keys and no teacher-indexed alignment keys.
    assert "teacher_1_full_logits_ipc" in keys
    assert "teacher_1_sparse_logits_ipc" in keys
    assert not any(k.startswith("alignment_1_") for k in keys)
    assert "teacher_1_input_ids" not in keys


def test_reduce_mb_metric_means_per_chunk_diagnostics_and_sums_shares():
    """v6 per-chunk diagnostics are per-mb averages: mean-reduce, never sum.

    Summing them scales the reported value with the microbatch count — the
    defect that produced a ``top1_acc_per_chunk`` of 110.60 on a 128-microbatch
    step. The KD/CE terms are normalized by a *global* denominator in the loss,
    so those stay sum-reduced, as do the ``num_*`` counts.
    """
    n_mb = 128
    acc = [0.8] * n_mb
    # Mean-reduced, both bare and with the per-teacher ``_t{i}`` suffix.
    for key in (
        "top1_acc_per_chunk",
        "kl_common_per_chunk",
        "kl_partition_first_per_chunk",
        "kl_partition_last_per_chunk",
    ):
        assert reduce_mb_metric(key, acc) == pytest.approx(0.8)
        assert reduce_mb_metric(f"{key}_t0", acc) == pytest.approx(0.8)
        assert reduce_mb_metric(f"{key}_t11", acc) == pytest.approx(0.8)
    assert reduce_mb_metric("kl_loss_scale", [4.0] * n_mb) == pytest.approx(4.0)
    # Sum-reduced: global-denominator loss shares and counts, suffixed or not.
    assert reduce_mb_metric("kl_loss", [0.5] * 4) == pytest.approx(2.0)
    assert reduce_mb_metric("kl_loss_t0", [0.5] * 4) == pytest.approx(2.0)
    assert reduce_mb_metric("ce_loss", [0.5] * 4) == pytest.approx(2.0)
    assert reduce_mb_metric("num_common_chunks_t0", [10] * 4) == pytest.approx(40)
    assert reduce_mb_metric("teacher_0/routed_samples", [2, 3]) == pytest.approx(5)
    assert reduce_mb_metric("teacher_0/routed_tokens", [7, 11]) == pytest.approx(18)
    assert reduce_mb_metric("teacher_0/weighted_kl", [0.2, 0.3]) == pytest.approx(0.5)


def test_dense_ipc_telemetry_counts_tp2_cp2_fallbacks_once_per_logical_row(capsys):
    """TP/CP-replicated worker results must not multiply the logical count.

    The packed TP2/CP2 policy returns one replicated result per DP rank, and
    each of the four DP ranks contributes 32 logical MBS1 records.  A TP2
    teacher cannot use the full-vocabulary zero-copy path, so all 128 records
    report one actual reconstruction fallback for teacher 0.
    """
    per_dp_records = [[1] * 32 for _ in range(4)]
    model_parallel_deduplicated = [
        value for dp_records in per_dp_records for value in dp_records
    ]

    xt_mod._log_dense_ipc_reconstruction_telemetry(
        {
            "ipc_reconstruction_fallbacks": model_parallel_deduplicated,
            "ipc_reconstruction_fallbacks_t0": model_parallel_deduplicated,
        },
        packing_plan=SimpleNamespace(batch_uid=17),
        num_teachers=1,
    )

    assert capsys.readouterr().out == (
        "XTOKEN_IPC_RECONSTRUCTION batch_uid=17 "
        "reconstruction_fallbacks=128 teacher_0=128\n"
    )


def test_skip_keys_builder_same_vocab_full_logits():
    loss_fn = _FakeLossFn(projection_matrix_paths=[None])
    # Same-vocab full-logits teacher: only the IPC handle list is skipped.
    assert xtoken_non_student_seq_keys(loss_fn) == frozenset(
        {"teacher_0_full_logits_ipc", "teacher_0_sparse_logits_ipc"}
    )


def test_export_teacher_logits_packs_indexed_keys_and_runs_serially():
    # teacher 0 cross-tokenizer; teacher 1 same-vocab. Both ship full-vocab
    # logits over the unified get_full_logits_ipc producer (always-full).
    loss_fn = _FakeLossFn(projection_matrix_paths=["/p0.pt", None])
    # Attach both teacher mocks to one parent so their calls land in a single
    # ordered list (lets us assert the serial interleaving below). Configure
    # the child return values after attaching.
    parent = MagicMock()
    t0 = MagicMock()
    t1 = MagicMock()
    parent.attach_mock(t0, "t0")
    parent.attach_mock(t1, "t1")
    t0.get_full_logits_ipc.return_value = [{"payload_ipc": 0}]
    t1.get_full_logits_ipc.return_value = [{"payload_ipc": 1}]
    # Cross-tokenizer teacher 0 needs teacher_0_*/alignment_0_*; same-vocab
    # teacher 1 reuses the student tokenization (no teacher_1_* token keys).
    batch = _make_batch(num_teachers=1)

    train_data = export_teacher_logits_and_pack(
        [t0, t1], loss_fn, batch, teacher_mbs=[1, 1]
    )

    # Cross-tokenizer teacher 0 -> full-logits IPC + its alignment payload.
    assert "teacher_0_full_logits_ipc" in train_data
    assert "alignment_0_pair_valid" in train_data
    assert "teacher_0_input_ids" in train_data
    # Same-vocab teacher 1 -> full-logits IPC handle list, no dense tensors,
    # no projection/alignment keys (it reuses the student tokenization).
    assert "teacher_1_full_logits_ipc" in train_data
    assert "teacher_1_input_ids" not in train_data
    assert "alignment_1_pair_valid" not in train_data
    # Both teachers ship full logits via get_full_logits_ipc with their own MBS.
    t0.get_full_logits_ipc.assert_called_once()
    assert t0.get_full_logits_ipc.call_args.kwargs["micro_batch_size"] == 1
    t1.get_full_logits_ipc.assert_called_once()
    assert t1.get_full_logits_ipc.call_args.kwargs["micro_batch_size"] == 1
    # Serial collocated execution: each teacher onloaded then offloaded, AND
    # teacher 0 is offloaded before teacher 1 is onloaded (never both resident).
    for t in (t0, t1):
        t.prepare_for_lp_inference.assert_called_once()
        t.offload_after_refit.assert_called_once()
    call_names = [c[0] for c in parent.mock_calls]
    assert call_names.index("t0.offload_after_refit") < call_names.index(
        "t1.prepare_for_lp_inference"
    )


def test_export_teacher_logits_preserves_student_routed_experts():
    teacher = MagicMock()
    teacher.get_full_logits_ipc.return_value = [{"payload_ipc": 0}]
    loss_fn = _FakeLossFn(projection_matrix_paths=["/projection.pt"])
    batch = _make_batch(num_teachers=1)
    routed_experts = torch.arange(
        batch["input_ids"].shape[0] * batch["input_ids"].shape[1] * 2 * 2
    ).reshape(batch["input_ids"].shape[0], batch["input_ids"].shape[1], 2, 2)
    batch["routed_experts"] = routed_experts

    train_data = export_teacher_logits_and_pack(
        [teacher], loss_fn, batch, teacher_mbs=[1]
    )

    assert train_data["routed_experts"] is routed_experts


def test_export_teacher_logits_threads_lockstep_plan_and_occurrence_ids():
    cfg = _make_master_config()
    _enable_lockstep_packing(cfg, global_batch_size=2, capacity=64)
    batch = _make_batch(batch_size=2, t_student=4, t_teacher=6)
    batch["student_semantic_regions"] = [
        ((0, "assistant", "content", 1, 3),),
        ((2, "assistant", "eot", 3, 4),),
    ]
    batch["teacher_0_semantic_regions"] = [
        ((0, "assistant", "content", 2, 4),),
        ((2, "assistant", "eot", 4, 5),),
    ]
    plan = build_xtoken_lockstep_packing_plan(
        batch, cfg, batch_uid=17, data_parallel_size=1
    )
    assert plan is not None
    teacher = MagicMock()
    teacher.get_full_logits_ipc.return_value = [
        {"batch_item_id": item_id, "teacher_shards": []}
        for item_id in plan.canonical_batch_item_ids
    ]
    loss_fn = _FakeLossFn(projection_matrix_paths=["/projection.pt"])

    train_data = export_teacher_logits_and_pack(
        [teacher],
        loss_fn,
        batch,
        teacher_mbs=[1],
        packing_plan=plan,
    )

    assert train_data["sample_id"] == batch["sample_id"]
    assert torch.equal(train_data["batch_item_id"], batch["batch_item_id"])
    first_occurrence = plan.canonical_batch_item_ids[0]
    assert train_data["student_semantic_regions"][0][0] == (
        first_occurrence,
        0,
        "assistant",
        "content",
        1,
        3,
    )
    assert train_data["teacher_0_semantic_regions"][0][0][0] == first_occurrence
    teacher_data = teacher.get_full_logits_ipc.call_args.args[0]
    assert torch.equal(teacher_data["batch_item_id"], batch["batch_item_id"])
    assert teacher.get_full_logits_ipc.call_args.kwargs["packing_plan"] is plan
    assert (
        teacher.get_full_logits_ipc.call_args.kwargs["packing_side_id"] == "teacher_0"
    )


def test_build_teacher_force_ids_respects_position_zero_and_shift():
    batch = _make_batch(batch_size=1, t_student=5, t_teacher=5)
    batch["teacher_0_input_ids"] = torch.tensor([[10, 11, 12, 13, 14]])
    batch["alignment_0_pair_valid"] = torch.tensor([[True, True, False]])
    batch["alignment_0_student_chunk_id"] = torch.tensor([[0, 0, -1, 1, 1]])
    batch["alignment_0_teacher_chunk_id"] = torch.tensor([[0, 0, 1, 1, -1]])
    loss_config = {
        "kl_chunk_shift": True,
        "prefix_bidir_v3_noise_filter_topk": 0,
        "prefix_bidir_v3_pure_alm": False,
        "teacher_topk_ipc_k": 8192,
        "teacher_topk_ipc_keep_realized": True,
    }

    force_ids = _build_teacher_force_include_token_ids(
        batch,
        teacher_idx=0,
        loss_config=loss_config,
    )

    # Chunk 0 starts at position zero, so it remains unshifted. Chunk 1 starts
    # later and its labels at teacher positions 2,3 are attached to predictors
    # 1,2, with the latter write taking precedence at position 1.
    assert torch.equal(force_ids, torch.tensor([[10, 12, 13, -1, -1]]))


def test_export_teacher_logits_mixes_sparse_cross_and_dense_same_vocab():
    loss_fn = _FakeLossFn(
        projection_matrix_paths=["/p0.pt", None],
        teacher_topk_ipc_k=8192,
        teacher_vocab_sizes=[151669, 128256],
    )
    t0 = MagicMock()
    t1 = MagicMock()
    t0.get_topk_logits_ipc.return_value = [{"teacher_shards": ["sparse"]}]
    t1.get_full_logits_ipc.return_value = [{"teacher_shards": ["dense"]}]
    batch = _make_batch(num_teachers=1)

    train_data = export_teacher_logits_and_pack(
        [t0, t1], loss_fn, batch, teacher_mbs=[1, 2]
    )

    assert train_data["teacher_0_sparse_logits_ipc"] == [{"teacher_shards": ["sparse"]}]
    assert train_data["teacher_1_full_logits_ipc"] == [{"teacher_shards": ["dense"]}]
    assert "teacher_0_full_logits_ipc" not in train_data
    assert "teacher_1_sparse_logits_ipc" not in train_data
    t0.get_full_logits_ipc.assert_not_called()
    t1.get_topk_logits_ipc.assert_not_called()
    sparse_kwargs = t0.get_topk_logits_ipc.call_args.kwargs
    assert sparse_kwargs["k"] == 8192
    assert sparse_kwargs["temperature"] == 1.0
    assert sparse_kwargs["vocab_size"] == 151669
    assert sparse_kwargs["micro_batch_size"] == 1
    assert sparse_kwargs["support_mode"] == "row_topk"
    assert sparse_kwargs["gt_filter_topk"] is None
    assert torch.equal(
        t0.get_topk_logits_ipc.call_args.args[0]["force_include_token_ids"],
        torch.zeros((1, 4), dtype=torch.long),
    )
    assert t1.get_full_logits_ipc.call_args.kwargs["micro_batch_size"] == 2


def test_setup_preserves_aligner_config_across_interleaved_teacher_types():
    cfg = _make_master_config()
    # Interleave a same-vocab teacher between two cross-tokenizer teachers.
    cfg.teachers.append(
        TeacherConfig(
            **{
                "aligner": {"projection_matrix_path": None},
                "weight": 0.5,
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
            }
        )
    )
    cfg.teachers.append(
        TeacherConfig(
            **{
                "aligner": {
                    "projection_matrix_path": "/tmp/dummy-projection-2.pt",
                    "drop_first_assistant_chunk_kl": True,
                },
                "weight": 0.25,
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
                "tokenizer": {"name": "teacher-2-tok"},
            }
        )
    )
    student_tok = _make_tokenizer(32)
    teacher_toks = [_make_tokenizer(24), _make_tokenizer(32), _make_tokenizer(28)]
    train_ds = MagicMock()
    train_ds.__len__ = MagicMock(return_value=4)

    with (
        patch.object(xt_mod, "RayVirtualCluster") as mock_cluster,
        patch.object(xt_mod, "Policy") as mock_policy_cls,
        patch.object(xt_mod, "Logger"),
        patch.object(xt_mod, "CheckpointManager") as mock_cp_cls,
        patch.object(xt_mod, "TokenAligner") as mock_aligner_cls,
        patch.object(xt_mod, "CrossTokenizerCollator") as mock_collator_cls,
        patch.object(xt_mod, "CrossTokenizerDistillationLossFn") as mock_loss_cls,
        patch.object(xt_mod, "StatefulDataLoader") as mock_dl_cls,
        patch.object(xt_mod, "assert_teacher_student_batch_grid"),
        patch.object(xt_mod, "assert_xtoken_ipc_node_local"),
    ):
        mock_cp_cls.return_value.get_latest_checkpoint_path.return_value = None
        mock_cp_cls.return_value.load_training_info.return_value = None
        mock_cp_cls.return_value.get_resume_paths.return_value = (None, None)
        mock_dl_cls.side_effect = lambda *a, **kw: MagicMock(spec=StatefulDataLoader)
        mock_policy_cls.side_effect = lambda *a, **kw: MagicMock(data_parallel_size=1)

        (_student, teachers, *_rest) = setup(
            cfg,
            student_tokenizer=student_tok,
            teacher_tokenizers=teacher_toks,
            train_dataset=train_ds,
            val_dataset=None,
        )

    # One teacher Policy per entry (+ the student), and the colocation cap
    # accounts for all teacher groups + the student.
    assert isinstance(teachers, list) and len(teachers) == 3
    assert mock_policy_cls.call_count == 4  # 3 teachers + 1 student
    assert mock_cluster.call_args.kwargs["max_colocated_worker_groups"] == 4
    assert mock_aligner_cls.call_count == 2
    aligners = mock_collator_cls.call_args.kwargs["aligners"]
    assert aligners[0] is not None and aligners[1] is None and aligners[2] is not None
    assert mock_collator_cls.call_args.kwargs[
        "drop_first_assistant_chunk_kl_by_teacher"
    ] == [False, False, True]
    # Per-teacher metadata injected as parallel lists.
    injected_cfg = mock_loss_cls.call_args.args[0]
    assert injected_cfg["projection_matrix_paths"] == [
        "/tmp/dummy-projection.pt",
        None,
        "/tmp/dummy-projection-2.pt",
    ]
    assert injected_cfg["teacher_weights"] == [1.0, 0.5, 0.25]
    assert injected_cfg["teacher_vocab_sizes"] == [24, 32, 28]


def test_setup_rejects_same_vocab_teacher_with_mismatched_vocab():
    # A null-projection (same-vocab) teacher whose vocab != the student's is a
    # config error — same-vocab detection is by null projection path, but the
    # teacher must actually share the student vocab for the direct KL. Caught
    # in setup() (the right place — it has the real tokenizers; tokenizer
    # *names* would wrongly flag Llama-3.2-3B vs -1B, which share a vocab).
    cfg = _make_master_config()
    cfg.teachers[0].aligner.projection_matrix_path = None  # mark same-vocab
    student_tok = _make_tokenizer(32)
    teacher_toks = [_make_tokenizer(24)]  # 24 != 32 -> mismatch
    with (
        patch.object(xt_mod, "RayVirtualCluster") as mock_cluster,
        pytest.raises(ValueError, match="safe student-token reuse"),
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


def test_same_tokenizer_reuse_rejects_equal_size_different_mapping():
    cfg = _make_master_config()
    cfg.teachers[0].aligner.projection_matrix_path = None
    student = _make_tokenizer(4)
    teacher = _make_tokenizer(4)
    teacher.get_vocab.return_value = {
        "token-1": 0,
        "token-0": 1,
        "token-2": 2,
        "token-3": 3,
    }

    with pytest.raises(ValueError, match="differing fields: .*vocab"):
        validate_xtoken_tokenizer_reuse(cfg, student, [teacher])


def test_same_tokenizer_reuse_rejects_template_kwargs_difference():
    cfg = _make_master_config()
    cfg.teachers[0].aligner.projection_matrix_path = None
    cfg.policy["tokenizer"]["chat_template_kwargs"] = {"enable_thinking": False}
    cfg.teachers[0].tokenizer["chat_template_kwargs"] = {"enable_thinking": True}

    with pytest.raises(ValueError, match="chat_template_kwargs"):
        validate_xtoken_tokenizer_reuse(cfg, _make_tokenizer(4), [_make_tokenizer(4)])


def test_same_tokenizer_reuse_rejects_slow_tokenizer_even_with_same_vocab():
    cfg = _make_master_config()
    cfg.teachers[0].aligner.projection_matrix_path = None
    student = _make_tokenizer(4)
    teacher = _make_tokenizer(4)
    teacher.is_fast = False

    with pytest.raises(ValueError, match="slow-tokenizer.*cannot be proven"):
        validate_xtoken_tokenizer_reuse(cfg, student, [teacher])


def test_same_tokenizer_reuse_rejects_different_fast_backend_behavior():
    cfg = _make_master_config()
    cfg.teachers[0].aligner.projection_matrix_path = None
    student = _make_tokenizer(4)
    teacher = _make_tokenizer(4)
    teacher.backend_tokenizer.to_str.return_value = "different-normalizer"

    with pytest.raises(ValueError, match="differing fields: .*backend"):
        validate_xtoken_tokenizer_reuse(cfg, student, [teacher])


def test_lockstep_setup_accepts_supported_dtensor_cp1():
    cfg = _make_master_config()
    _enable_lockstep_packing(cfg)

    assert validate_xtoken_packing_setup(cfg) == 1


def test_lockstep_setup_rejects_renamed_mamba_from_architecture_metadata(
    monkeypatch,
):
    cfg = _make_master_config()
    _enable_lockstep_packing(cfg)
    cfg.policy["model_name"] = "/models/innocuous-local-name"
    mamba_config = SimpleNamespace(
        model_type="nemotron_h",
        architectures=["NemotronHForCausalLM"],
        to_dict=lambda: {
            "model_type": "nemotron_h",
            "architectures": ["NemotronHForCausalLM"],
            "hybrid_override_pattern": "M*-M*-",
            "mamba_d_state": 128,
        },
    )
    monkeypatch.setattr(
        xt_mod, "_load_xtoken_packed_model_config", lambda config: mamba_config
    )

    with pytest.raises(ValueError, match="Nano/Mamba recurrent state isolation"):
        validate_xtoken_packing_setup(cfg)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda cfg: cfg.teachers[0].sequence_packing.update(enabled=False),
            "student and every teacher",
        ),
        (
            lambda cfg: cfg.policy["dynamic_batching"].update(enabled=True),
            "dynamic batching",
        ),
        (
            lambda cfg: cfg.data["train"].update(characters_per_sample=128),
            "characters_per_sample=null",
        ),
        (
            lambda cfg: cfg.data.update(validation={"characters_per_sample": 128}),
            "data.validation.characters_per_sample=null",
        ),
        (
            lambda cfg: cfg.data.update(num_packed_rows=2),
            "num_packed_rows must be 1",
        ),
        (
            lambda cfg: cfg.policy["sequence_packing"].update(
                algorithm="modified_first_fit_decreasing"
            ),
            "lockstep_first_fit_decreasing",
        ),
        (
            lambda cfg: cfg.policy["sequence_packing"].pop("fuse_loss"),
            "fuse_loss must be set explicitly",
        ),
        (
            lambda cfg: cfg.loss_fn.update(teacher_topk_ipc_k=8),
            "dense teacher IPC only",
        ),
        (
            lambda cfg: cfg.loss_fn.update(kd_loss_mode="select_teacher"),
            "static additive teacher aggregation",
        ),
        (
            lambda cfg: cfg.policy["dtensor_cfg"].update(context_parallel_size=2),
            "DTensor-V2 CP=1 only",
        ),
        (
            lambda cfg: cfg.policy["dtensor_cfg"].update(sequence_parallel=True),
            "sequence parallelism",
        ),
    ],
)
def test_lockstep_setup_rejects_unsupported_modes(mutate, message):
    cfg = _make_master_config()
    _enable_lockstep_packing(cfg)
    mutate(cfg)

    with pytest.raises(ValueError, match=message):
        validate_xtoken_packing_setup(cfg)


def test_lockstep_setup_resolves_character_grouping_for_validation_defaults():
    cfg = _make_master_config()
    _enable_lockstep_packing(cfg)
    cfg.data["train"]["characters_per_sample"] = None
    cfg.data["default"] = {"characters_per_sample": 128}
    cfg.data["validation"] = {"dataset_name": "validation"}

    with pytest.raises(ValueError, match="data.validation.characters_per_sample=null"):
        validate_xtoken_packing_setup(cfg)


def test_lockstep_plan_assigns_unique_occurrence_ids_after_validation_padding():
    cfg = _make_master_config()
    cfg.cluster = {"num_nodes": 1, "gpus_per_node": 2}
    cfg.policy["dtensor_cfg"]["tensor_parallel_size"] = 1
    cfg.teachers[0].dtensor_cfg["tensor_parallel_size"] = 1
    _enable_lockstep_packing(cfg, global_batch_size=4, capacity=64)
    batch = xt_mod.pad_distillation_val_batch(_make_batch(batch_size=3), 4)

    plan = build_xtoken_lockstep_packing_plan(
        batch,
        cfg,
        batch_uid=7,
        data_parallel_size=2,
    )

    assert plan is not None
    assert batch["sample_id"] == ["sample-0", "sample-1", "sample-2", "sample-2"]
    assert len(set(batch["batch_item_id"].tolist())) == 4
    assert plan.canonical_batch_item_ids == tuple(batch["batch_item_id"].tolist())
    assert plan.sides["student"].raw_lengths == (4, 4, 4, 4)
    assert plan.sides["teacher_0"].raw_lengths == (4, 4, 4, 4)
    assert plan.sides["student"].rank_bin_indices == ((0,), (1,))
    assert plan.bins == (
        tuple(plan.canonical_batch_item_ids[:2]),
        tuple(plan.canonical_batch_item_ids[2:]),
    )


def test_dtensor_tp_student_plan_preserves_fixed_training_tail():
    cfg = _make_master_config()
    _enable_lockstep_packing(cfg, global_batch_size=2, capacity=64)
    cfg.policy["dtensor_cfg"]["tensor_parallel_size"] = 2
    batch = _make_batch(batch_size=2, t_student=4, t_teacher=6)

    plan = build_xtoken_lockstep_packing_plan(
        batch,
        cfg,
        batch_uid=8,
        data_parallel_size=1,
    )

    assert plan is not None
    assert len(plan.bins) == 1
    assert plan.sides["student"].physical_tokens_by_bin == (64,)
    assert plan.sides["student"].padded_cu_seqlens_by_bin[0][-1] == 64
    # The inference-only teacher does not inherit the student's TP training
    # tail; its side-local physical geometry remains independently minimal.
    assert plan.sides["teacher_0"].physical_tokens_by_bin == (16,)


def test_lockstep_telemetry_proves_multi_sample_bins(capsys):
    cfg = _make_master_config()
    _enable_lockstep_packing(cfg, global_batch_size=2, capacity=64)
    batch = _make_batch(batch_size=2, t_student=4, t_teacher=6)
    plan = build_xtoken_lockstep_packing_plan(
        batch,
        cfg,
        batch_uid=9,
        data_parallel_size=1,
    )
    assert plan is not None

    log_xtoken_packing_telemetry(plan, batch)

    output = capsys.readouterr().out
    assert "logical_samples=2 physical_bins=1 multi_sample_bins=1" in output
    assert "side=student" in output
    assert "side=teacher_0" in output


def test_logical_batch_digest_is_deterministic_and_content_bound(capsys):
    batch = _make_batch(batch_size=2, t_student=4, t_teacher=6)
    batch["kd_token_mask"] = batch["token_mask"].clone()
    batch["student_semantic_regions"] = [
        ((0, "assistant", "content", 1, 3),),
        ((0, "assistant", "eot", 3, 4),),
    ]
    batch["teacher_0_semantic_regions"] = [
        ((0, "assistant", "content", 2, 5),),
        ((0, "assistant", "eot", 5, 6),),
    ]

    first = build_xtoken_logical_batch_digest_record(batch, batch_uid=9)
    second = build_xtoken_logical_batch_digest_record(batch, batch_uid=9)
    assert first == second
    assert first["logical_samples"] == 2
    assert first["sample_ids"]["count"] == 2
    assert first["fields"]["input_ids"]["shape"] == [2, 4]
    assert first["fields"]["teacher_0_input_ids"]["shape"] == [2, 6]
    assert "student_semantic_regions" in first["fields"]
    assert "alignment_0_pair_valid" in first["fields"]

    batch["input_ids"][0, 0] = 1
    changed = build_xtoken_logical_batch_digest_record(batch, batch_uid=9)
    assert (
        changed["fields"]["input_ids"]["sha256"]
        != first["fields"]["input_ids"]["sha256"]
    )
    assert changed["record_sha256"] != first["record_sha256"]

    logged = log_xtoken_logical_batch_digest(batch, batch_uid=9)
    output = capsys.readouterr().out
    assert logged == changed
    assert output.startswith("XTOKEN_LOGICAL_BATCH_DIGEST {")
    assert "sample-0" not in output


def test_dense_ipc_telemetry_counts_logical_shard_bytes():
    handles = [
        {
            "teacher_shards": [
                {"actual_shape": (4, 8), "dtype": torch.float32},
                {"actual_shape": (4, 8), "dtype": torch.float32},
            ]
        },
        {
            "teacher_shards": [
                {"actual_shape": (2, 8), "dtype": torch.bfloat16},
            ]
        },
    ]

    assert xt_mod._dense_ipc_logical_bytes(handles) == 288


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
    fn.teacher_weights = [2.0, 3.0]  # distinct weights so weighting is exercised

    fallback_calls = []

    def _fallback(i, *args, **kwargs):
        fallback_calls.append(i)
        # Distinct nonzero per-teacher KD so the weighted sum is non-trivial:
        # a "last teacher overwrites" or "weight ignored" bug would change it.
        return torch.tensor(float(i + 1)), {}

    def _fast_path(*args, **kwargs):
        raise AssertionError(
            "direct_full_vocab_kl reached for cross-tokenizer teachers"
        )

    fn._compute_teacher_kd = _fallback
    fn._direct_full_vocab_kl = _fast_path

    # Equal-shaped teacher logits (the misleading signal) but teacher length 13
    # vs the student's length 10.
    teacher_logits = torch.zeros(2, 13, 24)
    teacher_full = {0: teacher_logits, 1: teacher_logits.clone()}
    student_logits = torch.zeros(2, 10, 32)

    total_kd, metrics = fn._averaged_logits_kd(
        student_logits,
        {},
        teacher_full,
        {},
        torch.tensor(20.0),
        teacher_sparse_logits_by_idx={},
        tp_group=None,
        cp_group=None,
    )
    assert fallback_calls == [0, 1]  # per-teacher fallback path, one call each
    # total_kd = Σ_i weight_i * kd_i = 2*1 + 3*2 = 8.
    assert total_kd.item() == pytest.approx(2.0 * 1.0 + 3.0 * 2.0)
    assert metrics["teacher_0/weighted_kl"] == pytest.approx(2.0)
    assert metrics["teacher_1/weighted_kl"] == pytest.approx(6.0)


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

    kd, metrics = fn._averaged_logits_kd(
        logits,
        {},
        teacher_full,
        {0: MagicMock()},
        torch.tensor(20.0),
        teacher_sparse_logits_by_idx={},
        tp_group=None,
        cp_group=None,
    )
    assert fast_calls == [1]  # fast path taken exactly once
    assert "kl_loss" in metrics
    assert kd.item() == pytest.approx(1.23)
    assert metrics["teacher_0/weighted_kl"] == pytest.approx(1.23 / 2)
    assert metrics["teacher_1/weighted_kl"] == pytest.approx(1.23 / 2)


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
    token_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 0, 0]], dtype=torch.float32)
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


# ---------------------------------------------------------------------------
# sum_weights_metric is a sum-mode-only knob
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["averaged_logits", "select_teacher"])
def test_sum_weights_metric_rejected_outside_sum_mode(mode):
    # Dynamic weighting is only applied in kd_loss_mode="sum" (matching the
    # reference). Combining sum_weights_metric with another mode used to be
    # silently dropped; it must now fail loudly at construction.
    with pytest.raises(ValueError, match="sum_weights_metric"):
        CrossTokenizerDistillationLossFn(
            {
                "kd_loss_mode": mode,
                "sum_weights_metric": "ce",
            }
        )


@pytest.mark.parametrize("weights", [[0.0, 0.0], [1.0, -1.0]])
def test_averaged_logits_rejects_zero_weight_sum(weights):
    # averaged_logits divides by sum(teacher_weights) to form a convex average,
    # so a zero weight-sum (all zeros, or a signed set cancelling to 0) is
    # rejected at construction with a clear message instead of failing with a
    # deep ZeroDivisionError mid-step.
    with pytest.raises(ValueError, match="must not sum to zero"):
        CrossTokenizerDistillationLossFn(
            {
                "kd_loss_mode": "averaged_logits",
                "teacher_weights": weights,
            }
        )


# ---------------------------------------------------------------------------
# DP-global teacher score (cross-rank agreement primitive)
# ---------------------------------------------------------------------------


def test_dp_global_masked_mean_single_process_is_local_masked_mean():
    # Without a process group the DP all-reduce is a no-op, so the global
    # masked mean reduces to the plain masked mean: padded positions excluded,
    # and the result is detached (it gates selection / weighting, never
    # back-propagated). Cross-rank agreement across DP ranks is verified by
    # ``test_dp_global_masked_mean_agrees_across_ranks`` below.
    fn = CrossTokenizerDistillationLossFn.__new__(CrossTokenizerDistillationLossFn)
    values = torch.tensor([[1.0, 2.0, 100.0]], requires_grad=True)
    mask = torch.tensor([[1.0, 1.0, 0.0]])

    out = fn._dp_global_masked_mean(values, mask)

    assert out.item() == pytest.approx(1.5)  # (1+2)/2; padded 100.0 excluded
    assert not out.requires_grad  # detached


def _dp_global_masked_mean_worker(rank, world_size, init_file, q):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        fn = CrossTokenizerDistillationLossFn.__new__(CrossTokenizerDistillationLossFn)
        # Deliberately uneven shards: rank 1's masked-out 1e9 must not leak,
        # and the mean must combine across ranks (not stay rank-local).
        if rank == 0:
            values, mask = torch.tensor([[1.0, 2.0]]), torch.tensor([[1.0, 1.0]])
        else:
            values, mask = torch.tensor([[10.0, 1e9]]), torch.tensor([[1.0, 0.0]])
        q.put((rank, fn._dp_global_masked_mean(values, mask).item()))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not has_gloo(), reason="gloo backend unavailable")
def test_dp_global_masked_mean_agrees_across_ranks(tmp_path):
    """Both DP ranks compute the identical DP-global masked mean = 13/3.

    rank 0 contributes (1+2) over 2 valid positions, rank 1 contributes 10 over
    1 (its 1e9 is masked out); the all-reduced sum/count give (3+10)/(2+1),
    proving the score is DP-global (not rank-local) and padding is excluded.
    """
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [
        ctx.Process(
            target=_dp_global_masked_mean_worker,
            args=(rank, 2, str(tmp_path / "init"), q),
        )
        for rank in range(2)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0
    results = dict(q.get() for _ in range(len(procs)))
    assert results[0] == results[1]
    assert results[0] == pytest.approx(13.0 / 3.0)


# ---------------------------------------------------------------------------
# select_teacher chooses the lowest-CE teacher
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("better", [0, 1])
def test_select_teacher_picks_lowest_ce_teacher(better):
    # select_teacher must use the teacher with the lowest next-token CE on its
    # own tokens. Two same-vocab teachers: the "better" one peaks on the true
    # next token (CE ~ 0); the other is uniform (CE = log V). Parametrized over
    # which index is better so it can't pass by always returning 0.
    fn = CrossTokenizerDistillationLossFn.__new__(CrossTokenizerDistillationLossFn)
    fn.num_teachers = 2
    fn.projection_matrix_paths = [None, None]  # same-vocab: scored on student ids

    vocab, seqlen = 8, 4
    input_ids = torch.tensor([[1, 2, 3, 0]])
    data = {
        "input_ids": input_ids,
        "token_mask": torch.ones(1, seqlen),
        "sample_mask": torch.ones(1),
    }

    # Same-vocab teachers are scored on the student tokens via the thin
    # alignment (``align.student_input_ids`` / ``align.student_token_mask``).
    align = SimpleNamespace(
        student_input_ids=input_ids,
        student_kd_token_mask=None,
        student_token_mask=torch.ones(1, seqlen),
    )
    aligns_by_idx = {0: align, 1: align}

    confident = torch.full((1, seqlen, vocab), -10.0)
    for t in range(seqlen - 1):
        confident[0, t, input_ids[0, t + 1]] = 10.0  # peak on the true next token
    uniform = torch.zeros(1, seqlen, vocab)
    teacher_full = {better: confident, 1 - better: uniform}

    selected = []

    def _fake_kd(i, *args, **kwargs):
        selected.append(i)
        return torch.tensor(2.5), {"kl_loss": 2.5}

    fn._compute_teacher_kd = _fake_kd

    _kd, metrics = fn._select_teacher_kd(
        torch.zeros(1, seqlen, vocab),
        data,
        teacher_full,
        aligns_by_idx,
        torch.tensor(3.0),
        teacher_sparse_logits_by_idx={},
        tp_group=None,
        cp_group=None,
    )

    assert metrics["selected_teacher"] == better
    assert selected == [better]  # KD computed only for the selected teacher
    assert metrics[f"teacher_{better}/weighted_kl"] == pytest.approx(2.5)
    assert metrics[f"teacher_{1 - better}/weighted_kl"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# sum-mode aggregation: weighted sum + normalize_teacher_by_vocab rescaling
# ---------------------------------------------------------------------------


def _bare_sum_kd_loss_fn(
    *,
    num_teachers,
    teacher_weights,
    normalize_teacher_by_vocab=False,
    teacher_vocab_sizes=None,
):
    """A ``CrossTokenizerDistillationLossFn`` carrying only the attrs ``_sum_kd``
    reads, built via ``__new__`` to skip the config-driven ``__init__``.
    Static weights only (``sum_weights_metric=None``)."""
    fn = CrossTokenizerDistillationLossFn.__new__(CrossTokenizerDistillationLossFn)
    fn.num_teachers = num_teachers
    fn.teacher_weights = list(teacher_weights)
    fn.sum_weights_metric = None
    fn.normalize_teacher_by_vocab = normalize_teacher_by_vocab
    fn.teacher_vocab_sizes = (
        list(teacher_vocab_sizes) if teacher_vocab_sizes else [1] * num_teachers
    )
    return fn


def test_sum_kd_weighted_sum_of_per_teacher_kd():
    # total_kd = Σ_i weight_i * KD_i. Distinct nonzero per-teacher KD + distinct
    # weights so "last teacher overwrites" or "per-teacher weight ignored" can't
    # pass silently.
    fn = _bare_sum_kd_loss_fn(num_teachers=2, teacher_weights=[2.0, 3.0])

    def _fake_kd(i, *args, **kwargs):
        return torch.tensor(float(i + 1)), {"kl_loss": float(i + 1)}

    fn._compute_teacher_kd = _fake_kd

    total_kd, per_metrics = fn._sum_kd(
        torch.zeros(1, 4, 8),  # student_logits_contig (only used for device/dtype)
        {},
        {},
        {},
        torch.tensor(10.0),
        teacher_sparse_logits_by_idx={},
        tp_group=None,
        cp_group=None,
    )
    # 2*1 + 3*2 = 8.
    assert total_kd.item() == pytest.approx(2.0 * 1.0 + 3.0 * 2.0)
    # Per-teacher KD metrics are suffixed and the static weights are reported.
    assert per_metrics["kl_loss_t0"] == 1.0
    assert per_metrics["kl_loss_t1"] == 2.0
    assert per_metrics["weight_t0"] == 2.0
    assert per_metrics["weight_t1"] == 3.0
    assert per_metrics["teacher_0/weighted_kl"] == pytest.approx(2.0)
    assert per_metrics["teacher_1/weighted_kl"] == pytest.approx(6.0)


def test_sum_kd_normalize_teacher_by_vocab_rescales_by_log_ratio():
    # With normalize_teacher_by_vocab, each teacher's KD is scaled by
    # log(V_t_i) / log(min_j V_t_j). V0 = min so its scale is exactly 1; V1 = V0^2
    # so its scale = 2. A min<->max or numerator/denominator swap changes this.
    v0, v1 = 100, 10000  # log(v1)/log(v0) == 2
    fn = _bare_sum_kd_loss_fn(
        num_teachers=2,
        teacher_weights=[1.0, 1.0],
        normalize_teacher_by_vocab=True,
        teacher_vocab_sizes=[v0, v1],
    )

    def _fake_kd(i, *args, **kwargs):
        return torch.tensor(float(i + 1)), {}

    fn._compute_teacher_kd = _fake_kd

    total_kd, per_metrics = fn._sum_kd(
        torch.zeros(1, 4, 8),
        {},
        {},
        {},
        torch.tensor(10.0),
        teacher_sparse_logits_by_idx={},
        tp_group=None,
        cp_group=None,
    )
    s0 = math.log(v0) / math.log(v0)  # 1.0
    s1 = math.log(v1) / math.log(v0)  # 2.0
    # kd = [1, 2], weights = [1, 1]: 1*1*s0 + 2*1*s1.
    assert total_kd.item() == pytest.approx(1.0 * s0 + 2.0 * s1)
    assert per_metrics["teacher_0/weighted_kl"] == pytest.approx(1.0 * s0)
    assert per_metrics["teacher_1/weighted_kl"] == pytest.approx(2.0 * s1)


def test_teacher_routing_metrics_use_each_teachers_tokenization():
    """Routing counts exclude padded rows and use each teacher's token axis."""
    fn = CrossTokenizerDistillationLossFn.__new__(CrossTokenizerDistillationLossFn)
    fn.num_teachers = 2
    fn.projection_matrix_paths = ["/tmp/projection.pt", None]
    data = BatchedDataDict(
        {
            "sample_mask": torch.tensor([1, 0]),
            "token_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]]),
            "teacher_0_token_mask": torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]]),
        }
    )

    metrics = fn._teacher_routing_metrics(data)

    assert metrics == {
        "teacher_0/routed_samples": 1,
        "teacher_0/routed_tokens": 4,
        "teacher_1/routed_samples": 1,
        "teacher_1/routed_tokens": 3,
    }


# ---------------------------------------------------------------------------
# dynamic teacher weights: softmax(alpha * per-teacher scores) across teachers
# ---------------------------------------------------------------------------


def _dynamic_weights_fn(alpha, scores):
    """``CrossTokenizerDistillationLossFn`` stubbed to feed fixed per-teacher
    scores into ``_compute_dynamic_weights`` (isolates the aggregation)."""
    fn = CrossTokenizerDistillationLossFn.__new__(CrossTokenizerDistillationLossFn)
    fn.num_teachers = len(scores)
    fn.alpha = alpha
    fn.normalize_teacher_by_vocab = False
    fn._teacher_score_inputs = lambda i, *a, **k: (None, None, None)
    it = iter([torch.tensor(float(s)) for s in scores])
    fn._teacher_weight_score = lambda *a, **k: next(it)
    return fn


def test_compute_dynamic_weights_softmax_over_teachers():
    # Weights are softmax(alpha * scores) over the teacher axis. Stubbed scores
    # isolate the aggregation, catching "softmax over wrong axis" or "alpha
    # applied to logits not scores".
    data = {"input_ids": torch.zeros(1, 1), "sample_mask": torch.ones(1)}

    weights = _dynamic_weights_fn(1.0, [1.0, 0.0])._compute_dynamic_weights(
        data, {}, {}
    )
    expected = torch.softmax(torch.tensor([1.0, 0.0]), dim=0)
    assert len(weights) == 2
    assert weights[0].item() == pytest.approx(expected[0].item())
    assert weights[1].item() == pytest.approx(expected[1].item())
    assert (weights[0] + weights[1]).item() == pytest.approx(1.0)
    assert weights[0].item() > weights[1].item()  # higher score -> higher weight

    # alpha multiplies the scores before softmax: a larger alpha sharpens onto
    # the top-scoring teacher.
    sharp = _dynamic_weights_fn(8.0, [1.0, 0.0])._compute_dynamic_weights(data, {}, {})
    assert sharp[0].item() == pytest.approx(
        torch.softmax(torch.tensor([8.0, 0.0]), dim=0)[0].item()
    )
    assert sharp[0].item() > weights[0].item()
