# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torchdata.stateful_dataloader import StatefulDataLoader

import nemo_rl.algorithms.distillation as distil_mod
from nemo_rl.algorithms.distillation import (
    _default_distillation_save_state,
    check_vocab_equality,
    distillation_train,
    validate,
)
from nemo_rl.algorithms.loss import DistillationLossFn
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


@pytest.fixture
def mock_components():
    # Create mock components
    student_policy = MagicMock()
    student_policy.train.return_value = {
        "loss": torch.tensor(0.5),
        "grad_norm": torch.tensor(1.0),
        "all_mb_metrics": {"global_valid_toks": [10]},
    }
    # Add generate method since student_generation will be set to student_policy
    student_policy.generate.return_value = {
        "output_ids": torch.randint(0, 8, (2, 10)),
        "generation_lengths": torch.tensor([5, 7]),
        "unpadded_sequence_lengths": torch.tensor([8, 10]),
        "logprobs": torch.randn(2, 10, 8),
    }

    teacher_policy = MagicMock()
    teacher_policy.get_topk_logits.return_value = {
        "topk_logits": torch.randn(2, 10, 64),
        "topk_indices": torch.randint(0, 8, (2, 10, 64)),
    }

    # Set student_generation to None to avoid Ray-related refit issues
    # This makes NEED_REFIT = False, so refit_policy_generation won't be called
    student_generation = None

    # Create a proper message log structure with token_ids (similar to SFT)
    # Use BatchedDataDict instead of regular dict to support repeat_interleave
    mock_batch = BatchedDataDict[DatumSpec](
        {
            "message_log": [
                [
                    {
                        "token_ids": torch.tensor([1, 2, 3]),
                        "role": "user",
                        "content": "What is 1+1?",
                    },
                    {
                        "token_ids": torch.tensor([4, 5, 6]),
                        "role": "assistant",
                        "content": "The answer is 2.",
                    },
                ]
            ],
            "loss_multiplier": torch.tensor(
                [1.0]
            ),  # Make it 1D tensor for batch dimension
            "task_name": ["math"],
            "extra_env_info": [{}],
            "length": torch.tensor([6]),  # Make it 1D tensor for batch dimension
            "idx": torch.tensor([0]),  # Make it 1D tensor for batch dimension
        }
    )

    # Create mock dataloader with 10 batches that can be iterated multiple times
    train_dataloader = MagicMock(spec=StatefulDataLoader)

    def train_iter(self):
        return iter([mock_batch] * 10)

    train_dataloader.__iter__ = train_iter
    train_dataloader.__len__ = MagicMock(return_value=10)

    val_dataloader = MagicMock(spec=StatefulDataLoader)

    def val_iter(self):
        return iter([mock_batch] * 10)

    val_dataloader.__iter__ = val_iter
    val_dataloader.__len__ = MagicMock(return_value=10)

    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.image_processor = None
    tokenizer.video_processor = None
    tokenizer.feature_extractor = None

    loss_fn = DistillationLossFn(
        {
            "kl_type": "forward",
            "mixed_kl_weight": 0.5,
            "zero_outside_topk": False,
        }
    )

    logger = MagicMock()
    checkpointer = MagicMock()

    # Create mock environments
    task_to_env = {"math": MagicMock()}
    val_task_to_env = {"math": MagicMock()}

    # Create mock master config
    master_config = {
        "distillation": {
            "max_num_steps": 5,
            "max_num_epochs": 10,
            "val_period": 100,
            "val_batch_size": 1,
            "val_at_start": False,
            "val_at_end": False,
            "max_val_samples": 10,
            "topk_logits_k": 64,
            "num_prompts_per_step": 1,
            "num_generations_per_prompt": 1,
            "max_rollout_turns": 0,  # No environment interaction needed for distillation
            "seed": 42,
        },
        "policy": {
            "train_global_batch_size": 1,
            "make_sequence_length_divisible_by": 8,
            "max_total_sequence_length": 2048,
            "generation": {
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": None,
                "colocated": {
                    "enabled": False,
                },
            },
        },
        "teacher": {
            "model_name": "test-teacher",
        },
        "loss_fn": {
            "kl_type": "forward",
            "mixed_kl_weight": 0.5,
            "zero_outside_topk": False,
        },
        "data": {
            "dataset_name": "test_dataset",
        },
        "logger": {
            "num_val_samples_to_print": 5,
        },
        "cluster": {
            "num_nodes": 1,
            "gpus_per_node": 2,
        },
        "checkpointing": {
            "enabled": False,
            "checkpoint_must_save_by": None,
            "save_period": 10,
            "metric_name": None,
        },
    }

    return {
        "student_policy": student_policy,
        "teacher_policy": teacher_policy,
        "student_generation": student_generation,
        "train_dataloader": train_dataloader,
        "val_dataloader": val_dataloader,
        "tokenizer": tokenizer,
        "loss_fn": loss_fn,
        "logger": logger,
        "checkpointer": checkpointer,
        "task_to_env": task_to_env,
        "val_task_to_env": val_task_to_env,
        "master_config": master_config,
    }


def test_distillation_train_max_steps(mock_components):
    """Test that training terminates correctly when maximum steps are reached."""
    mock_components["master_config"]["distillation"]["max_num_steps"] = 5

    distillation_save_state = _default_distillation_save_state()

    # Run training
    distillation_train(
        mock_components["student_policy"],
        mock_components["teacher_policy"],
        mock_components["student_generation"],
        mock_components["train_dataloader"],
        mock_components["val_dataloader"],
        mock_components["tokenizer"],
        mock_components["loss_fn"],
        mock_components["task_to_env"],
        mock_components["val_task_to_env"],
        mock_components["logger"],
        mock_components["checkpointer"],
        distillation_save_state,
        mock_components["master_config"],
    )

    assert mock_components["student_policy"].train.call_count == 5


def test_distillation_train_logs_legacy_opd_metrics_without_changing_teacher_payload(
    mock_components,
):
    """Test that legacy OPD metrics are logged and teacher top-k payload stays unchanged."""
    mock_components["master_config"]["distillation"]["max_num_steps"] = 1
    mock_components["master_config"]["distillation"]["max_num_epochs"] = 1
    mock_components["master_config"]["distillation"]["val_period"] = 0
    mock_components["master_config"]["distillation"]["val_at_end"] = False

    fake_teacher_topk = {
        "topk_logits": torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]]
        ),
        "topk_indices": torch.tensor([[[9, 10], [11, 12], [13, 14], [15, 16]]]),
    }
    mock_components["teacher_policy"].get_topk_logits.return_value = fake_teacher_topk

    distillation_save_state = _default_distillation_save_state()

    with (
        patch.object(
            distil_mod,
            "_get_teacher_annotation_bytes",
            return_value=4321,
        ) as mock_teacher_annotation_bytes,
        patch.object(
            distil_mod,
            "_get_driver_rss_bytes",
            side_effect=[100, 150],
        ) as mock_driver_rss_bytes,
    ):
        distillation_train(
            mock_components["student_policy"],
            mock_components["teacher_policy"],
            mock_components["student_generation"],
            mock_components["train_dataloader"],
            mock_components["val_dataloader"],
            mock_components["tokenizer"],
            mock_components["loss_fn"],
            mock_components["task_to_env"],
            mock_components["val_task_to_env"],
            mock_components["logger"],
            mock_components["checkpointer"],
            distillation_save_state,
            mock_components["master_config"],
        )

    assert mock_driver_rss_bytes.call_count == 2
    mock_teacher_annotation_bytes.assert_called_once_with(fake_teacher_topk)

    train_call = mock_components["student_policy"].train.call_args
    train_payload = train_call.args[0]
    assert torch.equal(
        train_payload["teacher_topk_logits"], fake_teacher_topk["topk_logits"]
    )
    assert torch.equal(
        train_payload["teacher_topk_indices"], fake_teacher_topk["topk_indices"]
    )

    train_metrics = None
    for call in mock_components["logger"].log_metrics.call_args_list:
        if call.kwargs.get("prefix") == "train":
            train_metrics = call.args[0]
            break

    assert train_metrics is not None
    assert train_metrics["opd/data/teacher_annotation_bytes"] == 4321
    assert train_metrics["opd/data/manifest_batch_size"] == 1
    assert train_metrics["opd/data/manifest_loss_positions"] == 3
    assert train_metrics["opd/data/driver_rss_bytes"] == 150


def test_distillation_train_stream_teacher_uses_stream_consumer(mock_components):
    mock_components["master_config"]["distillation"]["max_num_steps"] = 1
    mock_components["master_config"]["distillation"]["max_num_epochs"] = 1
    mock_components["master_config"]["distillation"]["val_period"] = 0
    mock_components["master_config"]["distillation"]["val_at_end"] = False
    mock_components["master_config"]["distillation"]["data_pipeline"] = {
        "mode": "stream_teacher",
        "max_chunk_tokens": 1024,
    }

    layout = MagicMock()
    layout.dp = 1
    mock_components["teacher_policy"].stage_layout.return_value = layout
    mock_components["teacher_policy"].annotate_topk_stream.return_value = "annotated"
    mock_components["student_policy"].train_distillation_stream.return_value = {
        "loss": torch.tensor(0.25),
        "grad_norm": torch.tensor(0.75),
        "all_mb_metrics": {"global_valid_toks": [3]},
    }

    distillation_save_state = _default_distillation_save_state()

    with patch.object(
        distil_mod,
        "drain_annotated_stream",
        return_value="drained",
    ) as mock_drain:
        distillation_train(
            mock_components["student_policy"],
            mock_components["teacher_policy"],
            mock_components["student_generation"],
            mock_components["train_dataloader"],
            mock_components["val_dataloader"],
            mock_components["tokenizer"],
            mock_components["loss_fn"],
            mock_components["task_to_env"],
            mock_components["val_task_to_env"],
            mock_components["logger"],
            mock_components["checkpointer"],
            distillation_save_state,
            mock_components["master_config"],
        )

    mock_components["teacher_policy"].get_topk_logits.assert_not_called()
    mock_components["student_policy"].train.assert_not_called()
    mock_components["teacher_policy"].annotate_topk_stream.assert_called_once()
    mock_drain.assert_called_once_with("annotated")
    stream_train_call = mock_components["student_policy"].train_distillation_stream.call_args
    assert stream_train_call.args[1:] == ("drained", mock_components["loss_fn"])
    assert stream_train_call.args[0]["sample_ids"].tolist() == [0]
    assert "timer" in stream_train_call.kwargs


def test_distillation_train_stream_rollout_uses_ref_stream_consumer(mock_components):
    mock_components["master_config"]["distillation"]["max_num_steps"] = 1
    mock_components["master_config"]["distillation"]["data_pipeline"] = {
        "mode": "stream_rollout",
        "max_chunk_tokens": 128,
    }

    layout = MagicMock()
    layout.dp = 1
    mock_components["teacher_policy"].stage_layout.return_value = layout
    mock_components["teacher_policy"].annotate_topk_stream.return_value = "annotated"
    mock_components["student_policy"].train_distillation_stream_from_refs.return_value = {
        "loss": torch.tensor(0.25),
        "grad_norm": torch.tensor(0.75),
        "all_mb_metrics": {"global_valid_toks": [3]},
    }

    distillation_save_state = _default_distillation_save_state()

    with (
        patch.object(
            distil_mod,
            "drain_annotated_stream",
            return_value="drained",
        ) as mock_drain,
        patch.object(
            distil_mod,
            "build_token_stream_from_rollout_batch",
            wraps=distil_mod.build_token_stream_from_rollout_batch,
        ) as mock_rollout_normalizer,
        patch.object(
            distil_mod,
            "batched_message_log_to_flat_message",
            side_effect=AssertionError("stream_rollout should not use dense flatten"),
        ),
    ):
        distillation_train(
            mock_components["student_policy"],
            mock_components["teacher_policy"],
            mock_components["student_generation"],
            mock_components["train_dataloader"],
            mock_components["val_dataloader"],
            mock_components["tokenizer"],
            mock_components["loss_fn"],
            mock_components["task_to_env"],
            mock_components["val_task_to_env"],
            mock_components["logger"],
            mock_components["checkpointer"],
            distillation_save_state,
            mock_components["master_config"],
        )

    mock_rollout_normalizer.assert_called_once()
    mock_components["teacher_policy"].get_topk_logits.assert_not_called()
    mock_components["student_policy"].train.assert_not_called()
    mock_components["student_policy"].train_distillation_stream.assert_not_called()
    mock_components[
        "student_policy"
    ].train_distillation_stream_from_refs.assert_called_once()
    ref_stream_call = mock_components[
        "student_policy"
    ].train_distillation_stream_from_refs.call_args
    assert ref_stream_call.args == ("drained", mock_components["loss_fn"])
    assert "timer" in ref_stream_call.kwargs
    mock_drain.assert_called_once_with("annotated")


def test_distillation_train_stream_rollout_rejects_async_rollout(mock_components):
    mock_components["master_config"]["distillation"]["data_pipeline"] = {
        "mode": "stream_rollout"
    }
    mock_components["master_config"]["policy"]["generation"]["backend"] = "vllm"
    mock_components["master_config"]["policy"]["generation"]["vllm_cfg"] = {
        "async_engine": True
    }

    with pytest.raises(AssertionError, match="async rollouts"):
        distillation_train(
            mock_components["student_policy"],
            mock_components["teacher_policy"],
            mock_components["student_generation"],
            mock_components["train_dataloader"],
            mock_components["val_dataloader"],
            mock_components["tokenizer"],
            mock_components["loss_fn"],
            mock_components["task_to_env"],
            mock_components["val_task_to_env"],
            mock_components["logger"],
            mock_components["checkpointer"],
            _default_distillation_save_state(),
            mock_components["master_config"],
        )


def test_distillation_train_stream_rollout_allows_nemo_gym_metadata_restore(
    mock_components,
):
    mock_components["master_config"]["distillation"]["max_num_steps"] = 1
    mock_components["master_config"]["distillation"]["max_num_epochs"] = 1
    mock_components["master_config"]["distillation"]["val_period"] = 0
    mock_components["master_config"]["distillation"]["val_at_end"] = False
    mock_components["master_config"]["distillation"]["data_pipeline"] = {
        "mode": "stream_rollout",
        "max_chunk_tokens": 128,
    }
    mock_components["master_config"]["policy"]["generation"]["backend"] = "vllm"
    mock_components["master_config"]["policy"]["generation"]["vllm_cfg"] = {
        "async_engine": True,
        "expose_http_server": True,
    }
    mock_components["master_config"]["env"] = {"should_use_nemo_gym": True}

    layout = MagicMock()
    layout.dp = 1
    mock_components["teacher_policy"].stage_layout.return_value = layout
    mock_components["teacher_policy"].annotate_topk_stream.return_value = "annotated"
    mock_components["student_policy"].train_distillation_stream_from_refs.return_value = {
        "loss": torch.tensor(0.25),
        "grad_norm": torch.tensor(0.75),
        "all_mb_metrics": {"global_valid_toks": [3]},
    }

    final_batch = BatchedDataDict[DatumSpec](
        {
            "message_log": [
                [
                    {"token_ids": torch.tensor([1, 2]), "role": "user"},
                    {"token_ids": torch.tensor([3, 4]), "role": "assistant"},
                ]
            ],
            "loss_multiplier": torch.tensor([1.0]),
            "task_name": ["math"],
            "extra_env_info": [{}],
            "length": torch.tensor([4]),
            "idx": torch.tensor([0]),
        }
    )
    fake_manifest = SimpleNamespace(
        input_lengths=torch.tensor([4]),
        batch_size=1,
        loss_spans=object(),
    )
    fake_token_stream = SimpleNamespace(shard_streams=[["chunk"]])

    def fake_rollout_normalizer(**kwargs):
        rollout_batch = kwargs["rollout_batch"]
        for key in distil_mod.STREAM_METADATA_KEYS:
            assert key in rollout_batch
            assert rollout_batch[key].numel() == 1
        return fake_manifest, fake_token_stream

    with (
        patch.object(
            distil_mod,
            "run_async_nemo_gym_rollout",
            return_value=SimpleNamespace(
                final_batch=final_batch,
                rollout_metrics={
                    "mean_gen_tokens_per_sample": 2.0,
                    "agent/full_result": "raw-response-payload",
                },
            ),
        ) as mock_nemo_gym_rollout,
        patch.object(
            distil_mod,
            "build_token_stream_from_rollout_batch",
            side_effect=fake_rollout_normalizer,
        ) as mock_rollout_normalizer,
        patch.object(distil_mod, "estimate_chunk_bytes", return_value=1),
        patch.object(distil_mod, "estimate_active_loss_positions", return_value=3),
        patch.object(
            distil_mod,
            "drain_annotated_stream",
            return_value="drained",
        ),
    ):
        distillation_train(
            mock_components["student_policy"],
            mock_components["teacher_policy"],
            mock_components["student_generation"],
            mock_components["train_dataloader"],
            mock_components["val_dataloader"],
            mock_components["tokenizer"],
            mock_components["loss_fn"],
            mock_components["task_to_env"],
            mock_components["val_task_to_env"],
            mock_components["logger"],
            mock_components["checkpointer"],
            _default_distillation_save_state(),
            mock_components["master_config"],
        )

    mock_nemo_gym_rollout.assert_called_once()
    mock_rollout_normalizer.assert_called_once()
    mock_components["student_policy"].train.assert_not_called()
    mock_components[
        "student_policy"
    ].train_distillation_stream_from_refs.assert_called_once()

    train_metrics = None
    for call in mock_components["logger"].log_metrics.call_args_list:
        if call.kwargs.get("prefix") == "train":
            train_metrics = call.args[0]
            break

    assert train_metrics is not None
    assert "agent/full_result" not in train_metrics


def _configure_mixed_generation_test(
    mock_components,
    *,
    student_generations_per_prompt: int,
    teacher_generations_per_prompt: int,
) -> None:
    mock_components["master_config"]["distillation"]["max_num_steps"] = 1
    mock_components["master_config"]["distillation"]["max_num_epochs"] = 1
    mock_components["master_config"]["distillation"]["val_period"] = 0
    mock_components["master_config"]["distillation"]["val_at_end"] = False
    mock_components["master_config"]["distillation"]["num_generations_per_prompt"] = (
        student_generations_per_prompt + teacher_generations_per_prompt
    )
    mock_components["master_config"]["distillation"]["data_pipeline"] = {
        "mode": "stream_rollout",
        "max_chunk_tokens": 128,
    }
    mixed_generation = {
        "enabled": True,
        "student_generations_per_prompt": student_generations_per_prompt,
        "teacher_generations_per_prompt": teacher_generations_per_prompt,
    }
    if teacher_generations_per_prompt > 0:
        mixed_generation["teacher_rollout_path"] = "/tmp/teacher-rollouts.jsonl"
    mock_components["master_config"]["distillation"]["mixed_generation"] = (
        mixed_generation
    )
    mock_components["master_config"]["policy"]["train_global_batch_size"] = (
        student_generations_per_prompt + teacher_generations_per_prompt
    )
    mock_components["master_config"]["policy"]["generation"]["backend"] = "vllm"
    mock_components["master_config"]["policy"]["generation"]["vllm_cfg"] = {
        "async_engine": True,
        "expose_http_server": True,
    }
    mock_components["master_config"]["env"] = {"should_use_nemo_gym": True}

    layout = MagicMock()
    layout.dp = 1
    mock_components["teacher_policy"].stage_layout.return_value = layout
    mock_components["teacher_policy"].annotate_topk_stream.return_value = "annotated"
    mock_components["student_policy"].train_distillation_stream_from_refs.return_value = {
        "loss": torch.tensor(0.25),
        "grad_norm": torch.tensor(0.75),
        "all_mb_metrics": {"global_valid_toks": [3]},
    }


def _student_rollout_final_batch(
    row_count: int,
    *,
    prompt_token_ids: list[int] | None = None,
) -> BatchedDataDict:
    if prompt_token_ids is None:
        prompt_token_ids = [1] * row_count
    return BatchedDataDict[DatumSpec](
        {
            "agent_ref": [
                {"type": "student", "name": f"student-{idx}"}
                for idx in range(row_count)
            ],
            "message_log": [
                [
                    {"token_ids": torch.tensor([prompt_token_ids[idx]]), "role": "user"},
                    {
                        "token_ids": torch.tensor([10 + idx]),
                        "generation_logprobs": torch.tensor([-0.2]),
                        "role": "assistant",
                    },
                ]
                for idx in range(row_count)
            ],
            "loss_multiplier": torch.ones(row_count),
            "length": torch.ones(row_count, dtype=torch.int32),
            "total_reward": torch.arange(1, row_count + 1, dtype=torch.float32),
            "truncated": torch.zeros(row_count, dtype=torch.bool),
        }
    )


def _teacher_rollout_record(
    generation_id: int,
    *,
    prompt_uid: str = "train:test_dataset:0",
    dataset_index: int = 0,
    prompt_token_id: int = 1,
    assistant_token_id: int | None = None,
) -> dict:
    if assistant_token_id is None:
        assistant_token_id = 99 + generation_id
    return {
        "schema_version": 1,
        "source": "teacher",
        "prompt_uid": prompt_uid,
        "dataset_index": dataset_index,
        "teacher_generation_id": generation_id,
        "agent_ref": {"type": "teacher", "name": f"teacher-{generation_id}"},
        "message_log": [
            {"token_ids": [prompt_token_id], "role": "user", "content": ""},
            {
                "token_ids": [assistant_token_id],
                "generation_logprobs": [-0.1],
                "role": "assistant",
                "content": "",
            },
        ],
        "length": 1,
        "loss_multiplier": 1.0,
        "total_reward": 4.0 + generation_id,
        "truncated": False,
        "sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": None},
        "model": {"name_or_path": "test-teacher", "tokenizer": "test-teacher"},
    }


def _train_metrics_from_logger(logger: MagicMock) -> dict | None:
    for call in logger.log_metrics.call_args_list:
        if call.kwargs.get("prefix") == "train":
            return call.args[0]
    return None


def test_distillation_train_mixed_generation_uses_student_subset_and_full_metadata(
    mock_components,
):
    _configure_mixed_generation_test(
        mock_components,
        student_generations_per_prompt=3,
        teacher_generations_per_prompt=1,
    )

    student_final_batch = _student_rollout_final_batch(3)
    teacher_record = _teacher_rollout_record(0)
    teacher_store = MagicMock()
    teacher_store.select_for_step.return_value = [teacher_record]
    fake_manifest = SimpleNamespace(
        input_lengths=torch.tensor([2, 2, 2, 2]),
        batch_size=4,
        loss_spans=object(),
    )
    fake_token_stream = SimpleNamespace(shard_streams=[["chunk"]])

    def fake_rollout_normalizer(**kwargs):
        rollout_batch = kwargs["rollout_batch"]
        assert rollout_batch["rollout_source"] == [
            "student",
            "student",
            "student",
            "teacher",
        ]
        assert rollout_batch["sample_ids"].tolist() == [0, 1, 2, 3]
        assert rollout_batch["generation_ids"].tolist() == [0, 1, 2, 3]
        return fake_manifest, fake_token_stream

    with (
        patch.object(
            distil_mod,
            "TeacherRolloutStore",
            return_value=teacher_store,
        ) as mock_store_cls,
        patch.object(
            distil_mod,
            "run_async_nemo_gym_rollout",
            return_value=SimpleNamespace(
                final_batch=student_final_batch,
                rollout_metrics={"mean_gen_tokens_per_sample": 1.0},
            ),
        ) as mock_nemo_gym_rollout,
        patch.object(
            distil_mod,
            "build_token_stream_from_rollout_batch",
            side_effect=fake_rollout_normalizer,
        ),
        patch.object(distil_mod, "estimate_chunk_bytes", return_value=1),
        patch.object(distil_mod, "estimate_active_loss_positions", return_value=3),
        patch.object(
            distil_mod,
            "drain_annotated_stream",
            return_value="drained",
        ),
    ):
        distillation_train(
            mock_components["student_policy"],
            mock_components["teacher_policy"],
            mock_components["student_generation"],
            mock_components["train_dataloader"],
            mock_components["val_dataloader"],
            mock_components["tokenizer"],
            mock_components["loss_fn"],
            mock_components["task_to_env"],
            mock_components["val_task_to_env"],
            mock_components["logger"],
            mock_components["checkpointer"],
            _default_distillation_save_state(),
            mock_components["master_config"],
        )

    mock_store_cls.assert_called_once()
    _, store_kwargs = mock_store_cls.call_args
    assert store_kwargs["dataset_namespace"] == "train:test_dataset"
    assert store_kwargs["expected_model_name"] == "test-teacher"
    assert store_kwargs["expected_tokenizer_name_or_path"] == "test-teacher"
    assert store_kwargs["require_done"] is True
    teacher_store.select_for_step.assert_called_once_with(
        ["train:test_dataset:0"],
        teacher_generations_per_prompt=1,
        step=0,
        seed=42,
    )
    rollout_input = mock_nemo_gym_rollout.call_args.kwargs["input_batch"]
    assert rollout_input.size == 3
    for key in distil_mod.STREAM_METADATA_KEYS:
        assert key not in rollout_input
    train_metrics = _train_metrics_from_logger(mock_components["logger"])
    assert train_metrics is not None
    assert train_metrics["rollout/source/student/count"] == 3.0
    assert train_metrics["rollout/source/teacher/count"] == 1.0
    assert train_metrics["rollout/source/teacher/reward_mean"] == 4.0


def test_distillation_train_mixed_generation_uses_real_store_and_preserves_alignment(
    mock_components,
    tmp_path,
):
    _configure_mixed_generation_test(
        mock_components,
        student_generations_per_prompt=3,
        teacher_generations_per_prompt=1,
    )
    teacher_rollout_path = tmp_path / "teacher-rollouts.jsonl"
    teacher_records = [
        _teacher_rollout_record(
            0,
            prompt_uid="train:test_dataset:0",
            dataset_index=0,
            prompt_token_id=101,
            assistant_token_id=900,
        ),
        _teacher_rollout_record(
            0,
            prompt_uid="train:test_dataset:1",
            dataset_index=1,
            prompt_token_id=102,
            assistant_token_id=901,
        ),
    ]
    with teacher_rollout_path.open("w", encoding="utf-8") as handle:
        for record in teacher_records:
            handle.write(json.dumps(record) + "\n")
    teacher_rollout_path.with_name(f"{teacher_rollout_path.name}.done").write_text(
        "{}\n",
        encoding="utf-8",
    )
    mock_components["master_config"]["distillation"]["mixed_generation"][
        "teacher_rollout_path"
    ] = str(teacher_rollout_path)
    mock_components["master_config"]["distillation"]["num_prompts_per_step"] = 2
    mock_components["master_config"]["policy"]["train_global_batch_size"] = 8

    prompt_batch = BatchedDataDict[DatumSpec](
        {
            "message_log": [
                [
                    {"token_ids": torch.tensor([101]), "role": "user", "content": "p0"},
                    {
                        "token_ids": torch.tensor([201]),
                        "role": "assistant",
                        "content": "a0",
                    },
                ],
                [
                    {"token_ids": torch.tensor([102]), "role": "user", "content": "p1"},
                    {
                        "token_ids": torch.tensor([202]),
                        "role": "assistant",
                        "content": "a1",
                    },
                ],
            ],
            "loss_multiplier": torch.ones(2),
            "task_name": ["math", "math"],
            "extra_env_info": [{}, {}],
            "length": torch.tensor([1, 1]),
            "idx": torch.tensor([0, 1]),
        }
    )

    def train_iter(self):
        return iter([prompt_batch])

    mock_components["train_dataloader"].__iter__ = train_iter
    mock_components["train_dataloader"].__len__ = MagicMock(return_value=1)

    fake_manifest = SimpleNamespace(
        input_lengths=torch.full((8,), 2),
        batch_size=8,
        loss_spans=object(),
    )
    fake_token_stream = SimpleNamespace(shard_streams=[["chunk"]])

    def fake_nemo_gym_rollout(**kwargs):
        rollout_input = kwargs["input_batch"]
        prompt_tokens = [
            int(message_log[0]["token_ids"].item())
            for message_log in rollout_input["message_log"]
        ]
        assert prompt_tokens == [101, 101, 101, 102, 102, 102]
        return SimpleNamespace(
            final_batch=_student_rollout_final_batch(
                6,
                prompt_token_ids=[101, 101, 101, 102, 102, 102],
            ),
            rollout_metrics={"mean_gen_tokens_per_sample": 1.0},
        )

    def fake_rollout_normalizer(**kwargs):
        rollout_batch = kwargs["rollout_batch"]
        assert rollout_batch["rollout_source"] == [
            "student",
            "student",
            "student",
            "teacher",
            "student",
            "student",
            "student",
            "teacher",
        ]
        assert rollout_batch["prompt_ids"].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
        assert rollout_batch["generation_ids"].tolist() == [0, 1, 2, 3, 0, 1, 2, 3]
        prompt_tokens = [
            int(message_log[0]["token_ids"].item())
            for message_log in rollout_batch["message_log"]
        ]
        assistant_tokens = [
            int(message_log[1]["token_ids"].item())
            for message_log in rollout_batch["message_log"]
        ]
        assert prompt_tokens == [101, 101, 101, 101, 102, 102, 102, 102]
        assert assistant_tokens == [10, 11, 12, 900, 13, 14, 15, 901]
        return fake_manifest, fake_token_stream

    with (
        patch.object(
            distil_mod,
            "run_async_nemo_gym_rollout",
            side_effect=fake_nemo_gym_rollout,
        ),
        patch.object(
            distil_mod,
            "build_token_stream_from_rollout_batch",
            side_effect=fake_rollout_normalizer,
        ),
        patch.object(distil_mod, "estimate_chunk_bytes", return_value=1),
        patch.object(distil_mod, "estimate_active_loss_positions", return_value=8),
        patch.object(
            distil_mod,
            "drain_annotated_stream",
            return_value="drained",
        ),
    ):
        distillation_train(
            mock_components["student_policy"],
            mock_components["teacher_policy"],
            mock_components["student_generation"],
            mock_components["train_dataloader"],
            mock_components["val_dataloader"],
            mock_components["tokenizer"],
            mock_components["loss_fn"],
            mock_components["task_to_env"],
            mock_components["val_task_to_env"],
            mock_components["logger"],
            mock_components["checkpointer"],
            _default_distillation_save_state(),
            mock_components["master_config"],
        )

    train_metrics = _train_metrics_from_logger(mock_components["logger"])
    assert train_metrics is not None
    assert train_metrics["rollout/source/student/count"] == 6.0
    assert train_metrics["rollout/source/teacher/count"] == 2.0


def test_distillation_train_mixed_generation_supports_student_only_layout(
    mock_components,
):
    _configure_mixed_generation_test(
        mock_components,
        student_generations_per_prompt=4,
        teacher_generations_per_prompt=0,
    )
    student_final_batch = _student_rollout_final_batch(4)
    fake_manifest = SimpleNamespace(
        input_lengths=torch.tensor([2, 2, 2, 2]),
        batch_size=4,
        loss_spans=object(),
    )
    fake_token_stream = SimpleNamespace(shard_streams=[["chunk"]])

    def fake_rollout_normalizer(**kwargs):
        rollout_batch = kwargs["rollout_batch"]
        assert rollout_batch["rollout_source"] == [
            "student",
            "student",
            "student",
            "student",
        ]
        assert rollout_batch["generation_ids"].tolist() == [0, 1, 2, 3]
        return fake_manifest, fake_token_stream

    with (
        patch.object(distil_mod, "TeacherRolloutStore") as mock_store_cls,
        patch.object(
            distil_mod,
            "run_async_nemo_gym_rollout",
            return_value=SimpleNamespace(
                final_batch=student_final_batch,
                rollout_metrics={"mean_gen_tokens_per_sample": 1.0},
            ),
        ) as mock_nemo_gym_rollout,
        patch.object(
            distil_mod,
            "build_token_stream_from_rollout_batch",
            side_effect=fake_rollout_normalizer,
        ),
        patch.object(distil_mod, "estimate_chunk_bytes", return_value=1),
        patch.object(distil_mod, "estimate_active_loss_positions", return_value=3),
        patch.object(
            distil_mod,
            "drain_annotated_stream",
            return_value="drained",
        ),
    ):
        distillation_train(
            mock_components["student_policy"],
            mock_components["teacher_policy"],
            mock_components["student_generation"],
            mock_components["train_dataloader"],
            mock_components["val_dataloader"],
            mock_components["tokenizer"],
            mock_components["loss_fn"],
            mock_components["task_to_env"],
            mock_components["val_task_to_env"],
            mock_components["logger"],
            mock_components["checkpointer"],
            _default_distillation_save_state(),
            mock_components["master_config"],
        )

    mock_store_cls.assert_not_called()
    assert mock_nemo_gym_rollout.call_args.kwargs["input_batch"].size == 4
    train_metrics = _train_metrics_from_logger(mock_components["logger"])
    assert train_metrics is not None
    assert train_metrics["rollout/source/student/count"] == 4.0
    assert train_metrics["rollout/source/teacher/count"] == 0.0


def test_distillation_train_mixed_generation_supports_teacher_only_layout(
    mock_components,
):
    _configure_mixed_generation_test(
        mock_components,
        student_generations_per_prompt=0,
        teacher_generations_per_prompt=4,
    )
    teacher_store = MagicMock()
    teacher_store.select_for_step.return_value = [
        _teacher_rollout_record(generation_id) for generation_id in range(4)
    ]
    fake_manifest = SimpleNamespace(
        input_lengths=torch.tensor([2, 2, 2, 2]),
        batch_size=4,
        loss_spans=object(),
    )
    fake_token_stream = SimpleNamespace(shard_streams=[["chunk"]])

    def fake_rollout_normalizer(**kwargs):
        rollout_batch = kwargs["rollout_batch"]
        assert rollout_batch["rollout_source"] == [
            "teacher",
            "teacher",
            "teacher",
            "teacher",
        ]
        assert rollout_batch["generation_ids"].tolist() == [0, 1, 2, 3]
        return fake_manifest, fake_token_stream

    with (
        patch.object(
            distil_mod,
            "TeacherRolloutStore",
            return_value=teacher_store,
        ) as mock_store_cls,
        patch.object(
            distil_mod,
            "run_async_nemo_gym_rollout",
            side_effect=AssertionError("teacher-only layout should skip student rollout"),
        ) as mock_nemo_gym_rollout,
        patch.object(
            distil_mod,
            "build_token_stream_from_rollout_batch",
            side_effect=fake_rollout_normalizer,
        ),
        patch.object(distil_mod, "estimate_chunk_bytes", return_value=1),
        patch.object(distil_mod, "estimate_active_loss_positions", return_value=3),
        patch.object(
            distil_mod,
            "drain_annotated_stream",
            return_value="drained",
        ),
    ):
        distillation_train(
            mock_components["student_policy"],
            mock_components["teacher_policy"],
            mock_components["student_generation"],
            mock_components["train_dataloader"],
            mock_components["val_dataloader"],
            mock_components["tokenizer"],
            mock_components["loss_fn"],
            mock_components["task_to_env"],
            mock_components["val_task_to_env"],
            mock_components["logger"],
            mock_components["checkpointer"],
            _default_distillation_save_state(),
            mock_components["master_config"],
        )

    mock_store_cls.assert_called_once()
    mock_nemo_gym_rollout.assert_not_called()
    teacher_store.select_for_step.assert_called_once_with(
        ["train:test_dataset:0"],
        teacher_generations_per_prompt=4,
        step=0,
        seed=42,
    )
    train_metrics = _train_metrics_from_logger(mock_components["logger"])
    assert train_metrics is not None
    assert train_metrics["rollout/source/student/count"] == 0.0
    assert train_metrics["rollout/source/teacher/count"] == 4.0


def test_distillation_train_mixed_generation_rejects_non_nemo_gym_rollouts(
    mock_components,
):
    mock_components["master_config"]["distillation"]["num_generations_per_prompt"] = 1
    mock_components["master_config"]["distillation"]["mixed_generation"] = {
        "enabled": True,
        "student_generations_per_prompt": 1,
        "teacher_generations_per_prompt": 0,
    }

    with pytest.raises(AssertionError, match="NeMo-Gym"):
        distillation_train(
            mock_components["student_policy"],
            mock_components["teacher_policy"],
            mock_components["student_generation"],
            mock_components["train_dataloader"],
            mock_components["val_dataloader"],
            mock_components["tokenizer"],
            mock_components["loss_fn"],
            mock_components["task_to_env"],
            mock_components["val_task_to_env"],
            mock_components["logger"],
            mock_components["checkpointer"],
            _default_distillation_save_state(),
            mock_components["master_config"],
        )


def test_distillation_train_mixed_generation_sparse_loss_uses_sparse_stream_consumer(
    mock_components,
):
    _configure_mixed_generation_test(
        mock_components,
        student_generations_per_prompt=3,
        teacher_generations_per_prompt=1,
    )
    mock_components["master_config"]["distillation"]["data_pipeline"] = {
        "mode": "sparse_loss",
        "max_chunk_tokens": 128,
    }
    mock_components["student_policy"].train_distillation_sparse_stream.return_value = {
        "loss": torch.tensor(0.25),
        "grad_norm": torch.tensor(0.75),
        "all_mb_metrics": {"global_valid_toks": [3]},
    }
    teacher_store = MagicMock()
    teacher_store.select_for_step.return_value = [_teacher_rollout_record(0)]

    with (
        patch.object(
            distil_mod,
            "TeacherRolloutStore",
            return_value=teacher_store,
        ),
        patch.object(
            distil_mod,
            "run_async_nemo_gym_rollout",
            return_value=SimpleNamespace(
                final_batch=_student_rollout_final_batch(3),
                rollout_metrics={"mean_gen_tokens_per_sample": 1.0},
            ),
        ),
        patch.object(
            distil_mod,
            "drain_annotated_stream",
            return_value="drained",
        ),
    ):
        distillation_train(
            mock_components["student_policy"],
            mock_components["teacher_policy"],
            mock_components["student_generation"],
            mock_components["train_dataloader"],
            mock_components["val_dataloader"],
            mock_components["tokenizer"],
            mock_components["loss_fn"],
            mock_components["task_to_env"],
            mock_components["val_task_to_env"],
            mock_components["logger"],
            mock_components["checkpointer"],
            _default_distillation_save_state(),
            mock_components["master_config"],
        )

    mock_components["teacher_policy"].get_topk_logits.assert_not_called()
    mock_components[
        "student_policy"
    ].train_distillation_sparse_stream.assert_called_once()
    sparse_call = mock_components[
        "student_policy"
    ].train_distillation_sparse_stream.call_args
    assert sparse_call.args[1:] == ("drained", mock_components["loss_fn"])
    assert sparse_call.args[0]["sample_ids"].tolist() == [0, 1, 2, 3]
    assert sparse_call.args[0]["generation_ids"].tolist() == [0, 1, 2, 3]


def test_distillation_train_mixed_generation_dense_path_calls_get_topk_logits(
    mock_components,
):
    _configure_mixed_generation_test(
        mock_components,
        student_generations_per_prompt=3,
        teacher_generations_per_prompt=1,
    )
    mock_components["master_config"]["distillation"].pop("data_pipeline")
    mock_components["teacher_policy"].get_topk_logits.return_value = {
        "topk_logits": torch.randn(4, 2, 64),
        "topk_indices": torch.randint(0, 8, (4, 2, 64)),
    }
    teacher_store = MagicMock()
    teacher_store.select_for_step.return_value = [_teacher_rollout_record(0)]

    with (
        patch.object(
            distil_mod,
            "TeacherRolloutStore",
            return_value=teacher_store,
        ),
        patch.object(
            distil_mod,
            "run_async_nemo_gym_rollout",
            return_value=SimpleNamespace(
                final_batch=_student_rollout_final_batch(3),
                rollout_metrics={"mean_gen_tokens_per_sample": 1.0},
            ),
        ),
    ):
        distillation_train(
            mock_components["student_policy"],
            mock_components["teacher_policy"],
            mock_components["student_generation"],
            mock_components["train_dataloader"],
            mock_components["val_dataloader"],
            mock_components["tokenizer"],
            mock_components["loss_fn"],
            mock_components["task_to_env"],
            mock_components["val_task_to_env"],
            mock_components["logger"],
            mock_components["checkpointer"],
            _default_distillation_save_state(),
            mock_components["master_config"],
        )

    mock_components["teacher_policy"].get_topk_logits.assert_called_once()
    mock_components["student_policy"].train.assert_called_once()
    train_payload = mock_components["student_policy"].train.call_args.args[0]
    assert train_payload["sample_ids"].tolist() == [0, 1, 2, 3]
    assert train_payload["generation_ids"].tolist() == [0, 1, 2, 3]
    assert train_payload["teacher_topk_logits"].shape[0] == 4


def test_distillation_train_sparse_loss_uses_sparse_stream_consumer(mock_components):
    mock_components["master_config"]["distillation"]["max_num_steps"] = 1
    mock_components["master_config"]["distillation"]["max_num_epochs"] = 1
    mock_components["master_config"]["distillation"]["val_period"] = 0
    mock_components["master_config"]["distillation"]["val_at_end"] = False
    mock_components["master_config"]["distillation"]["data_pipeline"] = {
        "mode": "sparse_loss",
        "max_chunk_tokens": 128,
    }

    layout = MagicMock()
    layout.dp = 1
    mock_components["teacher_policy"].stage_layout.return_value = layout
    mock_components["teacher_policy"].annotate_topk_stream.return_value = "annotated"
    mock_components["student_policy"].train_distillation_sparse_stream.return_value = {
        "loss": torch.tensor(0.25),
        "grad_norm": torch.tensor(0.75),
        "all_mb_metrics": {"global_valid_toks": [3]},
    }

    with patch.object(
        distil_mod,
        "drain_annotated_stream",
        return_value="drained",
    ):
        distillation_train(
            mock_components["student_policy"],
            mock_components["teacher_policy"],
            mock_components["student_generation"],
            mock_components["train_dataloader"],
            mock_components["val_dataloader"],
            mock_components["tokenizer"],
            mock_components["loss_fn"],
            mock_components["task_to_env"],
            mock_components["val_task_to_env"],
            mock_components["logger"],
            mock_components["checkpointer"],
            _default_distillation_save_state(),
            mock_components["master_config"],
        )

    mock_components["teacher_policy"].get_topk_logits.assert_not_called()
    mock_components["student_policy"].train_distillation_stream.assert_not_called()
    mock_components[
        "student_policy"
    ].train_distillation_sparse_stream.assert_called_once()
    sparse_call = mock_components[
        "student_policy"
    ].train_distillation_sparse_stream.call_args
    assert sparse_call.args[1:] == ("drained", mock_components["loss_fn"])
    assert "sample_ids" in sparse_call.args[0]


def test_exit_on_timeout(mock_components, capsys):
    """Test that training loop exits when timeout is reached"""
    # Set max steps to large number
    mock_components["master_config"]["distillation"]["max_num_steps"] = 100

    distillation_save_state = _default_distillation_save_state()

    # Mock TimeoutChecker to return False for first 7 checks, then True (timeout)
    with patch("nemo_rl.algorithms.distillation.TimeoutChecker") as mock_timeout_class:
        mock_timeout_instance = MagicMock()
        # Create a side_effect that returns False 7 times, then True
        check_results = [False] * 7 + [True]
        mock_timeout_instance.check_save.side_effect = check_results
        mock_timeout_class.return_value = mock_timeout_instance

        # Run training
        distillation_train(
            mock_components["student_policy"],
            mock_components["teacher_policy"],
            mock_components["student_generation"],
            mock_components["train_dataloader"],
            mock_components["val_dataloader"],
            mock_components["tokenizer"],
            mock_components["loss_fn"],
            mock_components["task_to_env"],
            mock_components["val_task_to_env"],
            mock_components["logger"],
            mock_components["checkpointer"],
            distillation_save_state,
            mock_components["master_config"],
        )

        # Verify training stopped at 8 steps (when check_save returned True)
        assert mock_components["student_policy"].train.call_count == 8

        # Verify the timeout message was printed and training actually stopped
        captured = capsys.readouterr()
        output_lines = captured.out.strip().split("\n")

        # Find the timeout message
        timeout_line_idx = None
        for i, line in enumerate(output_lines):
            if "Timeout has been reached, stopping training early" in line:
                timeout_line_idx = i
                break

        assert timeout_line_idx is not None, "Timeout message not found in output"

        # For distillation, verify we don't see more step messages after timeout
        remaining_lines = output_lines[timeout_line_idx:]
        for line in remaining_lines:
            # Distillation doesn't have epochs, but check for step markers
            assert not line.startswith("Step ") or "Step 8" in line, (
                f"Training continued after timeout: {line}"
            )


def test_validate_function(mock_components):
    """Test independent validation function to ensure validation logic correctness."""
    # Run validation
    val_metrics, validation_timings = validate(
        mock_components["student_generation"],
        mock_components["val_dataloader"],
        mock_components["tokenizer"],
        mock_components["val_task_to_env"],
        step=0,
        master_config=mock_components["master_config"],
    )

    # Verify validation results
    assert isinstance(val_metrics, dict)
    assert isinstance(validation_timings, dict)
    # For distillation, we don't need environment interaction since max_rollout_turns=0
    # The validation focuses on generation and teacher-student knowledge transfer
    # Note: validate() function itself doesn't call logger.log_metrics - that's done by the caller


def test_validate_nemo_gym_filters_full_result_metrics(mock_components):
    mock_components["master_config"]["distillation"]["max_val_samples"] = 1
    mock_components["master_config"]["distillation"]["val_batch_size"] = 1
    mock_components["master_config"]["policy"]["generation"]["backend"] = "vllm"
    mock_components["master_config"]["policy"]["generation"]["vllm_cfg"] = {
        "async_engine": True,
        "expose_http_server": True,
    }
    mock_components["master_config"]["env"] = {"should_use_nemo_gym": True}

    final_batch = BatchedDataDict[DatumSpec](
        {
            "message_log": [
                [
                    {"role": "user", "content": "What is 1+1?"},
                    {"role": "assistant", "content": "2"},
                ]
            ],
            "total_reward": torch.tensor([1.0]),
        }
    )

    with (
        patch.object(
            distil_mod,
            "run_async_nemo_gym_rollout",
            return_value=SimpleNamespace(
                final_batch=final_batch,
                rollout_metrics={
                    "mean_gen_tokens_per_sample": 2.0,
                    "agent/full_result": "raw-response-payload",
                    "agent/score/mean": 1.0,
                },
            ),
        ) as mock_nemo_gym_rollout,
        patch.object(distil_mod, "print_message_log_samples"),
    ):
        val_metrics, validation_timings = validate(
            mock_components["student_generation"],
            mock_components["val_dataloader"],
            mock_components["tokenizer"],
            mock_components["val_task_to_env"],
            step=0,
            master_config=mock_components["master_config"],
        )

    mock_nemo_gym_rollout.assert_called_once()
    assert val_metrics["accuracy"] == 1.0
    assert val_metrics["agent/score/mean"] == 1.0
    assert "agent/full_result" not in val_metrics
    assert isinstance(validation_timings, dict)


def test_check_vocab_equality_pass(monkeypatch):
    student_tokenizer = MagicMock()
    student_tokenizer.get_vocab.return_value = {"a": 0, "b": 1}
    student_tokenizer.__len__.return_value = 2

    teacher_tokenizer = MagicMock()
    teacher_tokenizer.get_vocab.return_value = {"a": 0, "b": 1}
    teacher_tokenizer.__len__.return_value = 2

    student_config = MagicMock()
    student_config.vocab_size = 2
    teacher_config = MagicMock()
    teacher_config.vocab_size = 2

    monkeypatch.setattr(
        distil_mod.AutoTokenizer,
        "from_pretrained",
        lambda name: teacher_tokenizer,
    )
    monkeypatch.setattr(
        distil_mod.AutoConfig,
        "from_pretrained",
        lambda name: student_config if name == "student-model" else teacher_config,
    )

    # Should not raise
    check_vocab_equality(student_tokenizer, "student-model", "teacher-model")


def test_check_vocab_equality_vocab_mismatch_raises(monkeypatch):
    student_tokenizer = MagicMock()
    student_tokenizer.get_vocab.return_value = {"a": 0, "b": 1}
    student_tokenizer.__len__.return_value = 2

    teacher_tokenizer = MagicMock()
    teacher_tokenizer.get_vocab.return_value = {"a": 0, "c": 2}
    teacher_tokenizer.__len__.return_value = 2

    student_config = MagicMock()
    student_config.vocab_size = 2
    teacher_config = MagicMock()
    teacher_config.vocab_size = 2

    monkeypatch.setattr(
        distil_mod.AutoTokenizer,
        "from_pretrained",
        lambda name: teacher_tokenizer,
    )
    monkeypatch.setattr(
        distil_mod.AutoConfig,
        "from_pretrained",
        lambda name: student_config if name == "student-model" else teacher_config,
    )

    with pytest.raises(AssertionError):
        check_vocab_equality(student_tokenizer, "student-model", "teacher-model")


def test_check_vocab_equality_length_mismatch_raises(monkeypatch):
    # Same vocab mapping but different __len__ values
    vocab = {"a": 0, "b": 1}
    student_tokenizer = MagicMock()
    student_tokenizer.get_vocab.return_value = vocab
    student_tokenizer.__len__.return_value = 2

    teacher_tokenizer = MagicMock()
    teacher_tokenizer.get_vocab.return_value = vocab
    teacher_tokenizer.__len__.return_value = 3

    student_config = MagicMock()
    student_config.vocab_size = 2
    teacher_config = MagicMock()
    teacher_config.vocab_size = 2

    monkeypatch.setattr(
        distil_mod.AutoTokenizer,
        "from_pretrained",
        lambda name: teacher_tokenizer,
    )
    monkeypatch.setattr(
        distil_mod.AutoConfig,
        "from_pretrained",
        lambda name: student_config if name == "student-model" else teacher_config,
    )

    with pytest.raises(AssertionError):
        check_vocab_equality(student_tokenizer, "student-model", "teacher-model")


def test_check_vocab_equality_config_vocab_size_mismatch_raises(monkeypatch):
    vocab = {"a": 0, "b": 1}
    student_tokenizer = MagicMock()
    student_tokenizer.get_vocab.return_value = vocab
    student_tokenizer.__len__.return_value = 2

    teacher_tokenizer = MagicMock()
    teacher_tokenizer.get_vocab.return_value = vocab
    teacher_tokenizer.__len__.return_value = 2

    student_config = MagicMock()
    student_config.vocab_size = 2
    teacher_config = MagicMock()
    teacher_config.vocab_size = 3

    monkeypatch.setattr(
        distil_mod.AutoTokenizer,
        "from_pretrained",
        lambda name: teacher_tokenizer,
    )
    monkeypatch.setattr(
        distil_mod.AutoConfig,
        "from_pretrained",
        lambda name: student_config if name == "student-model" else teacher_config,
    )

    with pytest.raises(AssertionError):
        check_vocab_equality(student_tokenizer, "student-model", "teacher-model")


def test_noncolocated_inference_requires_explicit_gpus_per_node_single_node():
    """Test that non-colocated inference requires explicit gpus_per_node when cluster.num_nodes=1."""
    from unittest.mock import MagicMock, patch

    from nemo_rl.algorithms.distillation import setup

    # Create minimal config with non-colocated inference but gpus_per_node=None
    master_config = {
        "policy": {
            "generation": {
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": None,
                "backend": "vllm",
                "colocated": {
                    "enabled": False,  # Non-colocated
                    "resources": {
                        "gpus_per_node": None,  # This should trigger error
                        "num_nodes": None,
                    },
                },
            },
            "dtensor_cfg": {
                "enabled": False,
            },
        },
        "teacher": {
            "dtensor_cfg": {
                "enabled": False,
            },
        },
        "loss_fn": {},
        "distillation": {
            "seed": 42,
            "topk_logits_k": 64,
            "num_prompts_per_step": 1,  # Config extraction requires this key
            "val_period": 0,  # Config extraction requires this key
            "val_at_start": False,  # Config extraction requires this key
            "val_at_end": False,  # Config extraction requires this key
        },
        "data": {"shuffle": False},
        "logger": {},  # Config extraction requires this key
        "checkpointing": {},  # Config extraction requires this key
        "cluster": {
            "num_nodes": 1,  # Single node
            "gpus_per_node": 8,
        },
    }

    tokenizer = MagicMock()
    dataset = MagicMock()
    dataset.__len__ = MagicMock(return_value=10)

    # Mock everything we don't need to test
    with (
        patch("nemo_rl.algorithms.distillation.Logger") as mock_logger,
        patch("nemo_rl.algorithms.distillation.CheckpointManager") as mock_checkpointer,
        patch("nemo_rl.algorithms.distillation.StatefulDataLoader"),
        pytest.raises(
            AssertionError,
            match="policy.generation.colocated.resources.gpus_per_node must be explicitly set",
        ),
    ):
        # Configure mocks to skip checkpoint loading
        mock_checkpointer.return_value.get_latest_checkpoint_path.return_value = None
        setup(master_config, tokenizer, dataset, None)


def test_distillation_setup_non_colocated_smoke(monkeypatch):
    """Smoke test: calling setup with a non-colocated config should succeed."""
    from unittest.mock import MagicMock, patch

    import nemo_rl.algorithms.distillation as distil_mod

    # Single node cluster; inference uses a subset of GPUs on same node
    master_config = {
        "policy": {
            "generation": {
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": None,
                "backend": "vllm",
                "colocated": {
                    "enabled": False,
                    "resources": {
                        "gpus_per_node": 8,  # inference on 8 GPU
                        "num_nodes": 1,
                    },
                },
            },
            "dtensor_cfg": {
                "enabled": False,
            },
            "model_name": "test-policy",
        },
        "teacher": {
            "model_name": "test-teacher",
            "dtensor_cfg": {
                "enabled": False,
            },
        },
        "loss_fn": {
            "kl_type": "forward",
            "mixed_kl_weight": 0.5,
            "zero_outside_topk": False,
        },
        "distillation": {
            "seed": 42,
            "topk_logits_k": 64,
            "num_prompts_per_step": 1,
            "max_num_epochs": 10,
            "max_num_steps": 100,
            "val_period": 0,
            "val_at_start": False,
            "val_at_end": False,
        },
        "data": {"shuffle": False},
        "logger": {},
        "checkpointing": {},
        "cluster": {"num_nodes": 2, "gpus_per_node": 8},
    }

    tokenizer = MagicMock()
    dataset = MagicMock()
    dataset.__len__ = MagicMock(return_value=1)

    # Skip tokenizer/vocab equality check inside setup
    monkeypatch.setenv("NRL_SKIP_DISTILLATION_TOKENIZER_CHECK", "1")

    ip_port = ("127.0.0.1", 12345)

    class DummyCluster:
        def __init__(self, *args, **kwargs):
            pass

        def world_size(self):
            return 1

        def get_master_address_and_port(self):
            return ip_port

    class DummyPolicy:
        def __init__(self, *args, **kwargs):
            pass

        def prepare_refit_info(self):
            return {}

        def offload_after_refit(self):
            return None

        def init_collective(self, *args, **kwargs):
            return [MagicMock()]

    class DummyVllmGeneration:
        def __init__(self, *args, **kwargs):
            pass

        def finish_generation(self):
            return None

        def prepare_refit_info(self, *args, **kwargs):
            return None

        def init_collective(self, *args, **kwargs):
            return [MagicMock()]

    with (
        patch.object(distil_mod, "RayVirtualCluster", DummyCluster),
        patch.object(distil_mod, "Logger"),
        patch.object(distil_mod, "CheckpointManager") as mock_ckpt_mgr,
        patch.object(distil_mod, "StatefulDataLoader"),
        patch.object(distil_mod, "Policy", DummyPolicy),
        patch.object(distil_mod, "VllmGeneration", DummyVllmGeneration),
        patch.object(distil_mod, "ray") as mock_ray,
    ):
        mock_ckpt_mgr.return_value.get_latest_checkpoint_path.return_value = None
        mock_ckpt_mgr.return_value.get_resume_paths.return_value = (None, None)
        mock_ray.get = MagicMock(return_value=None)

        # Should not raise
        result = distil_mod.setup(master_config, tokenizer, dataset, None)

        # Basic shape check of returned tuple
        assert isinstance(result, tuple)


def test_noncolocated_inference_requires_explicit_gpus_per_node_multi_node():
    """Test that non-colocated inference requires explicit gpus_per_node when cluster.num_nodes>1."""
    from unittest.mock import MagicMock, patch

    from nemo_rl.algorithms.distillation import setup

    # Create minimal config with non-colocated inference but gpus_per_node=None
    master_config = {
        "policy": {
            "generation": {
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": None,
                "backend": "vllm",
                "colocated": {
                    "enabled": False,  # Non-colocated
                    "resources": {
                        "gpus_per_node": None,  # This should trigger error
                        "num_nodes": 1,  # Use 1 node for inference
                    },
                },
            },
            "dtensor_cfg": {
                "enabled": False,
            },
        },
        "teacher": {
            "dtensor_cfg": {
                "enabled": False,
            },
        },
        "loss_fn": {},
        "distillation": {
            "seed": 42,
            "topk_logits_k": 64,
            "max_num_epochs": 10,
            "max_num_steps": 100,
            "num_prompts_per_step": 1,  # Config extraction requires this key
            "val_period": 0,  # Config extraction requires this key
            "val_at_start": False,  # Config extraction requires this key
            "val_at_end": False,  # Config extraction requires this key
        },
        "data": {"shuffle": False},
        "logger": {},  # Config extraction requires this key
        "checkpointing": {},  # Config extraction requires this key
        "cluster": {
            "num_nodes": 2,  # Multi-node
            "gpus_per_node": 8,
        },
    }

    tokenizer = MagicMock()
    dataset = MagicMock()
    dataset.__len__ = MagicMock(return_value=10)

    # Mock everything we don't need to test
    with (
        patch("nemo_rl.algorithms.distillation.Logger") as mock_logger,
        patch("nemo_rl.algorithms.distillation.CheckpointManager") as mock_checkpointer,
        patch("nemo_rl.algorithms.distillation.StatefulDataLoader"),
        pytest.raises(
            AssertionError,
            match="policy.generation.colocated.resources.gpus_per_node must be explicitly set",
        ),
    ):
        # Configure mocks to skip checkpoint loading
        mock_checkpointer.return_value.get_latest_checkpoint_path.return_value = None
        setup(master_config, tokenizer, dataset, None)
