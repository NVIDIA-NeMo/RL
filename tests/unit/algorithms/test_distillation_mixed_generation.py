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

import importlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
import torch

from nemo_rl.algorithms.distillation_streaming import (
    STREAM_METADATA_KEYS,
    build_batch_manifest_from_train_data,
    build_conservation_oracle,
    build_step_manifest,
)
from nemo_rl.data.llm_message_utils import batched_message_log_to_flat_message
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def _load_mixed_generation_module():
    module_name = "nemo_rl.algorithms.distillation_mixed_generation"
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            pytest.fail(
                "Missing expected Phase 0-3 module "
                "`nemo_rl.algorithms.distillation_mixed_generation`."
            )
        raise


def _load_rollout_writer_module():
    return importlib.import_module("examples.nemo_gym.generate_nemo_gym_rollouts")


def _get_api(name: str):
    module = _load_mixed_generation_module()
    if not hasattr(module, name):
        pytest.fail(
            "Missing expected mixed-generation API "
            f"`nemo_rl.algorithms.distillation_mixed_generation.{name}`."
        )
    return getattr(module, name)


def _value(container: Any, key: str):
    if isinstance(container, Mapping):
        if key not in container:
            pytest.fail(f"Expected key `{key}` in {container!r}.")
        return container[key]
    if hasattr(container, key):
        return getattr(container, key)
    pytest.fail(f"Expected field `{key}` on {container!r}.")


def _message(role: str, token_ids: list[int], *, generation_logprobs: list[float] | None = None):
    message = {
        "role": role,
        "content": "",
        "token_ids": torch.tensor(token_ids, dtype=torch.int64),
    }
    if generation_logprobs is not None:
        message["generation_logprobs"] = torch.tensor(
            generation_logprobs,
            dtype=torch.float32,
        )
    return message


def _final_batch_sample(
    prompt_token_ids: list[int],
    assistant_token_ids: list[int],
    *,
    prompt_uid: str,
    dataset_index: int,
    teacher_generation_id: int,
    agent_name: str = "teacher-agent",
):
    return {
        "agent_ref": {"type": "policy", "name": agent_name},
        "message_log": [
            _message("user", prompt_token_ids),
            _message(
                "assistant",
                assistant_token_ids,
                generation_logprobs=[-0.1] * len(assistant_token_ids),
            ),
        ],
        "length": len(prompt_token_ids),
        "loss_multiplier": 1.0,
        "total_reward": 0.0,
        "truncated": False,
        "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": None},
        "model": {"name_or_path": "teacher", "tokenizer": "teacher-tokenizer"},
        "extra_env_info": {"source_prompt_uid": prompt_uid},
        "source": "teacher",
        "schema_version": 1,
        "prompt_uid": prompt_uid,
        "dataset_index": dataset_index,
        "teacher_generation_id": teacher_generation_id,
    }


def _single_sample_final_batch(sample: Mapping[str, Any]) -> BatchedDataDict:
    return BatchedDataDict(
        {
            "agent_ref": [sample["agent_ref"]],
            "message_log": [[dict(message) for message in sample["message_log"]]],
            "length": torch.tensor([sample["length"]], dtype=torch.int32),
            "loss_multiplier": torch.tensor(
                [sample["loss_multiplier"]],
                dtype=torch.float32,
            ),
            "total_reward": torch.tensor([sample["total_reward"]], dtype=torch.float32),
            "truncated": torch.tensor([sample["truncated"]], dtype=torch.bool),
            "extra_env_info": [sample["extra_env_info"]],
        }
    )


def _metadata_from_sample(sample: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": sample["schema_version"],
        "source": sample["source"],
        "prompt_uid": sample["prompt_uid"],
        "dataset_index": sample["dataset_index"],
        "teacher_generation_id": sample["teacher_generation_id"],
        "sampling": sample["sampling"],
        "model": sample["model"],
    }


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, default=_json_default) + "\n")


def _json_default(value: Any):
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _selection_signature(records: Any) -> list[tuple[str, int, int]]:
    signature = []
    for record in records:
        signature.append(
            (
                _value(record, "prompt_uid"),
                int(_value(record, "teacher_generation_id")),
                int(_value(record, "dataset_index")),
            )
        )
    return signature


def _parse_config(raw_config: Mapping[str, Any] | None, *, num_generations_per_prompt: int):
    parse_mixed_generation_config = _get_api("parse_mixed_generation_config")
    return parse_mixed_generation_config(
        raw_config,
        num_generations_per_prompt=num_generations_per_prompt,
    )


def _build_source_plan(
    *,
    prompt_count: int,
    num_generations_per_prompt: int,
    student_generations_per_prompt: int,
    teacher_generations_per_prompt: int,
):
    config = _parse_config(
        {
            "enabled": True,
            "student_generations_per_prompt": student_generations_per_prompt,
            "teacher_generations_per_prompt": teacher_generations_per_prompt,
            "teacher_rollout_path": "/tmp/teacher-rollouts.jsonl"
            if teacher_generations_per_prompt
            else None,
        },
        num_generations_per_prompt=num_generations_per_prompt,
    )
    build_source_plan = _get_api("build_source_plan")
    return build_source_plan(
        prompt_count=prompt_count,
        num_generations_per_prompt=num_generations_per_prompt,
        config=config,
    )


def _simple_prompt_batch(prompt_count: int) -> BatchedDataDict:
    message_logs = []
    for prompt_index in range(prompt_count):
        message_logs.append(
            [
                _message("user", [100 + prompt_index, 200 + prompt_index]),
                _message("assistant", [300 + prompt_index]),
            ]
        )
    return BatchedDataDict(
        {
            "message_log": message_logs,
            "loss_multiplier": torch.ones(prompt_count, dtype=torch.float32),
            "length": torch.full((prompt_count,), 2, dtype=torch.int32),
            "idx": torch.arange(prompt_count, dtype=torch.int64),
            "task_name": ["math"] * prompt_count,
            "extra_env_info": [{} for _ in range(prompt_count)],
        }
    )


def test_dataset_namespace_from_config_uses_train_dataset_identity():
    dataset_namespace_from_config = _get_api("dataset_namespace_from_config")

    assert dataset_namespace_from_config(
        {
            "train": {
                "dataset_name": "NemoGymDataset",
                "data_path": "/tmp/train.jsonl",
            }
        }
    ) == "train:NemoGymDataset:/tmp/train.jsonl"
    assert dataset_namespace_from_config({"dataset_name": "flat"}) == "train:flat"


def test_dataset_namespace_from_config_accepts_omegaconf_dictconfig(tmp_path: Path):
    from omegaconf import OmegaConf

    dataset_namespace_from_config = _get_api("dataset_namespace_from_config")
    data_path = tmp_path / "train.jsonl"
    data_path.write_text("", encoding="utf-8")
    data_config = OmegaConf.create(
        {
            "train": {
                "dataset_name": "NemoGymDataset",
                "data_path": str(data_path),
            }
        }
    )

    assert dataset_namespace_from_config(data_config) == (
        f"train:NemoGymDataset:{data_path.resolve()}"
    )


def test_dataset_namespace_from_config_accepts_single_dataset_list(tmp_path: Path):
    dataset_namespace_from_config = _get_api("dataset_namespace_from_config")
    data_path = tmp_path / "train.jsonl"
    data_path.write_text("", encoding="utf-8")

    assert dataset_namespace_from_config(
        {
            "train": [
                {
                    "dataset_name": "NemoGymDataset",
                    "data_path": str(data_path),
                }
            ]
        }
    ) == f"train:NemoGymDataset:{data_path.resolve()}"


def test_dataset_namespace_from_config_canonicalizes_data_path(tmp_path: Path):
    dataset_namespace_from_config = _get_api("dataset_namespace_from_config")
    data_path = tmp_path / "train.jsonl"
    data_path.write_text("", encoding="utf-8")

    assert dataset_namespace_from_config(
        {
            "train": {
                "dataset_name": "NemoGymDataset",
                "data_path": str(data_path),
            }
        }
    ) == f"train:NemoGymDataset:{data_path.resolve()}"


def test_build_prompt_identities_from_batch_prefers_explicit_prompt_uid():
    build_prompt_identities_from_batch = _get_api("build_prompt_identities_from_batch")
    batch = {
        "idx": torch.tensor([3, 4]),
        "extra_env_info": [
            {"prompt_uid": "explicit:3"},
            {"source_prompt_uid": "generated:4"},
        ],
        "task_name": ["math", "math"],
    }

    assert build_prompt_identities_from_batch(batch) == ["explicit:3", "generated:4"]


def test_build_prompt_identities_from_batch_uses_namespace_fallbacks():
    build_prompt_identities_from_batch = _get_api("build_prompt_identities_from_batch")
    batch = {
        "idx": torch.tensor([5, 6]),
        "extra_env_info": [{}, {}],
        "task_name": ["math", "code"],
    }
    no_task_batch = {
        "idx": torch.tensor([7]),
        "extra_env_info": [{}],
    }

    assert build_prompt_identities_from_batch(batch) == ["math:5", "code:6"]
    assert build_prompt_identities_from_batch(
        no_task_batch,
        dataset_namespace="train:/tmp/train.jsonl",
    ) == ["train:/tmp/train.jsonl:7"]
    assert build_prompt_identities_from_batch(
        {
            "idx": torch.tensor([8]),
            "extra_env_info": [None],
        },
        dataset_namespace="train:/tmp/train.jsonl",
    ) == ["train:/tmp/train.jsonl:8"]
    assert build_prompt_identities_from_batch(no_task_batch) == ["dataset:7"]


def test_build_prompt_identities_from_batch_rejects_namespace_mismatch():
    build_prompt_identities_from_batch = _get_api("build_prompt_identities_from_batch")
    batch = {
        "idx": torch.tensor([3]),
        "extra_env_info": [{"prompt_uid": "other:3"}],
    }

    with pytest.raises((AssertionError, ValueError), match="dataset namespace"):
        build_prompt_identities_from_batch(batch, dataset_namespace="train:data")


def _rollout_row(source: str, row_id: int) -> list[dict[str, Any]]:
    return [
        _message("user", [10_000 + row_id]),
        _message(
            "assistant",
            [20_000 + row_id, 30_000 + row_id],
            generation_logprobs=[-0.25, -0.5],
        ),
    ]


def _rollout_batch(source: str, row_ids: Sequence[int]) -> BatchedDataDict:
    batch_size = len(row_ids)
    return BatchedDataDict(
        {
            "agent_ref": [{"type": source, "name": f"{source}-{row_id}"} for row_id in row_ids],
            "message_log": [_rollout_row(source, row_id) for row_id in row_ids],
            "length": torch.ones(batch_size, dtype=torch.int32),
            "loss_multiplier": torch.ones(batch_size, dtype=torch.float32),
            "total_reward": torch.tensor(
                [float(row_id) for row_id in row_ids],
                dtype=torch.float32,
            ),
            "truncated": torch.zeros(batch_size, dtype=torch.bool),
            "extra_env_info": [{"row_id": row_id, "source": source} for row_id in row_ids],
        }
    )


def test_serialize_and_deserialize_round_trip_preserves_message_token_ids():
    serialize_final_batch_sample = _get_api("serialize_final_batch_sample")
    deserialize_teacher_rollout_record = _get_api("deserialize_teacher_rollout_record")
    records_to_final_batch = _get_api("records_to_final_batch")

    sample = _final_batch_sample(
        prompt_token_ids=[11, 12, 13],
        assistant_token_ids=[21, 22],
        prompt_uid="train:/tmp/train.jsonl:7",
        dataset_index=7,
        teacher_generation_id=3,
    )
    final_batch = _single_sample_final_batch(sample)
    metadata = _metadata_from_sample(sample)

    record = serialize_final_batch_sample(final_batch, index=0, metadata=metadata)

    assert json.loads(json.dumps(record))["message_log"][0]["token_ids"] == [11, 12, 13]

    deserialized = deserialize_teacher_rollout_record(record)
    round_tripped_batch = records_to_final_batch([deserialized])

    assert round_tripped_batch["message_log"][0][0]["token_ids"].tolist() == [11, 12, 13]
    assert round_tripped_batch["message_log"][0][1]["token_ids"].tolist() == [21, 22]
    assert round_tripped_batch["message_log"][0][1]["generation_logprobs"].tolist() == [-0.1, -0.1]
    assert round_tripped_batch["length"].tolist() == [3]


def test_deserialize_teacher_rollout_record_rejects_mismatched_generation_logprobs_length():
    deserialize_teacher_rollout_record = _get_api("deserialize_teacher_rollout_record")
    record = _final_batch_sample(
        prompt_token_ids=[11, 12],
        assistant_token_ids=[21, 22],
        prompt_uid="train:/tmp/train.jsonl:3",
        dataset_index=3,
        teacher_generation_id=0,
    )
    record["message_log"][1]["generation_logprobs"] = [-0.1]

    with pytest.raises((AssertionError, ValueError), match="generation_logprobs|length"):
        deserialize_teacher_rollout_record(record)


def test_deserialize_teacher_rollout_record_rejects_unsupported_schema_version():
    deserialize_teacher_rollout_record = _get_api("deserialize_teacher_rollout_record")
    record = _final_batch_sample(
        prompt_token_ids=[11, 12],
        assistant_token_ids=[21, 22],
        prompt_uid="train:/tmp/train.jsonl:4",
        dataset_index=4,
        teacher_generation_id=0,
    )
    record["schema_version"] = 999

    with pytest.raises((AssertionError, ValueError), match="schema_version"):
        deserialize_teacher_rollout_record(record)


def test_deserialize_teacher_rollout_record_rejects_rank_two_token_ids():
    deserialize_teacher_rollout_record = _get_api("deserialize_teacher_rollout_record")
    record = _final_batch_sample(
        prompt_token_ids=[11, 12],
        assistant_token_ids=[21, 22],
        prompt_uid="train:/tmp/train.jsonl:4",
        dataset_index=4,
        teacher_generation_id=0,
    )
    record["message_log"][1]["token_ids"] = [[21, 22]]

    with pytest.raises((AssertionError, ValueError), match="token_ids|rank"):
        deserialize_teacher_rollout_record(record)


def test_records_to_final_batch_rejects_duplicate_prompt_uid_generation_pairs():
    deserialize_teacher_rollout_record = _get_api("deserialize_teacher_rollout_record")
    records_to_final_batch = _get_api("records_to_final_batch")
    record = _final_batch_sample(
        prompt_token_ids=[11, 12, 13],
        assistant_token_ids=[21, 22],
        prompt_uid="train:/tmp/train.jsonl:9",
        dataset_index=9,
        teacher_generation_id=1,
    )

    with pytest.raises(
        (AssertionError, ValueError),
        match="prompt_uid|teacher_generation_id|duplicate",
    ):
        records_to_final_batch(
            [
                deserialize_teacher_rollout_record(record),
                deserialize_teacher_rollout_record(record),
            ]
        )


def test_teacher_rollout_store_selection_is_deterministic_across_jsonl_append_order(
    tmp_path: Path,
):
    TeacherRolloutStore = _get_api("TeacherRolloutStore")
    records = [
        _final_batch_sample(
            [11, 12],
            [21],
            prompt_uid="train:/tmp/train.jsonl:0",
            dataset_index=0,
            teacher_generation_id=0,
        ),
        _final_batch_sample(
            [11, 12],
            [22],
            prompt_uid="train:/tmp/train.jsonl:0",
            dataset_index=0,
            teacher_generation_id=1,
        ),
        _final_batch_sample(
            [13, 14],
            [23],
            prompt_uid="train:/tmp/train.jsonl:1",
            dataset_index=1,
            teacher_generation_id=0,
        ),
        _final_batch_sample(
            [13, 14],
            [24],
            prompt_uid="train:/tmp/train.jsonl:1",
            dataset_index=1,
            teacher_generation_id=1,
        ),
    ]
    ordered_path = tmp_path / "ordered.jsonl"
    permuted_path = tmp_path / "permuted.jsonl"
    _write_jsonl(ordered_path, records)
    _write_jsonl(permuted_path, list(reversed(records)))

    ordered_store = TeacherRolloutStore(ordered_path)
    permuted_store = TeacherRolloutStore(permuted_path)
    prompt_identities = ["train:/tmp/train.jsonl:0", "train:/tmp/train.jsonl:1"]

    ordered_selection = ordered_store.select_for_step(
        prompt_identities,
        teacher_generations_per_prompt=1,
        step=5,
        seed=99,
    )
    permuted_selection = permuted_store.select_for_step(
        prompt_identities,
        teacher_generations_per_prompt=1,
        step=5,
        seed=99,
    )

    assert _selection_signature(ordered_selection) == _selection_signature(
        permuted_selection
    )


def test_teacher_rollout_store_selection_changes_when_step_changes(tmp_path: Path):
    TeacherRolloutStore = _get_api("TeacherRolloutStore")
    records = [
        _final_batch_sample(
            [11, 12],
            [21],
            prompt_uid="train:/tmp/train.jsonl:0",
            dataset_index=0,
            teacher_generation_id=0,
        ),
        _final_batch_sample(
            [11, 12],
            [22],
            prompt_uid="train:/tmp/train.jsonl:0",
            dataset_index=0,
            teacher_generation_id=1,
        ),
    ]
    path = tmp_path / "teacher.jsonl"
    _write_jsonl(path, records)
    store = TeacherRolloutStore(path)

    step_0_selection = store.select_for_step(
        ["train:/tmp/train.jsonl:0"],
        teacher_generations_per_prompt=1,
        step=0,
        seed=123,
    )
    step_1_selection = store.select_for_step(
        ["train:/tmp/train.jsonl:0"],
        teacher_generations_per_prompt=1,
        step=1,
        seed=123,
    )

    assert _selection_signature(step_0_selection) != _selection_signature(
        step_1_selection
    )


def test_teacher_rollout_store_selection_changes_when_seed_changes(tmp_path: Path):
    TeacherRolloutStore = _get_api("TeacherRolloutStore")
    records = [
        _final_batch_sample(
            [11],
            [20 + gen_id],
            prompt_uid="train:/tmp/train.jsonl:0",
            dataset_index=0,
            teacher_generation_id=gen_id,
        )
        for gen_id in range(4)
    ]
    path = tmp_path / "teacher.jsonl"
    _write_jsonl(path, records)
    store = TeacherRolloutStore(path)

    seed_1_selection = store.select_for_step(
        ["train:/tmp/train.jsonl:0"],
        teacher_generations_per_prompt=1,
        step=0,
        seed=1,
    )
    seed_2_selection = store.select_for_step(
        ["train:/tmp/train.jsonl:0"],
        teacher_generations_per_prompt=1,
        step=0,
        seed=2,
    )

    assert _selection_signature(seed_1_selection) != _selection_signature(
        seed_2_selection
    )


def test_teacher_rollout_store_selects_multiple_teacher_slots_per_prompt(tmp_path: Path):
    TeacherRolloutStore = _get_api("TeacherRolloutStore")
    records = [
        _final_batch_sample(
            [11],
            [20 + gen_id],
            prompt_uid="train:/tmp/train.jsonl:0",
            dataset_index=0,
            teacher_generation_id=gen_id,
        )
        for gen_id in range(4)
    ]
    path = tmp_path / "teacher.jsonl"
    _write_jsonl(path, records)
    store = TeacherRolloutStore(path)

    selection = store.select_for_step(
        ["train:/tmp/train.jsonl:0"],
        teacher_generations_per_prompt=2,
        step=0,
        seed=1,
    )

    signature = _selection_signature(selection)
    assert len(signature) == 2
    assert len({teacher_generation_id for _, teacher_generation_id, _ in signature}) == 2


def test_teacher_rollout_store_selection_is_stable_under_prompt_order_permutation(
    tmp_path: Path,
):
    TeacherRolloutStore = _get_api("TeacherRolloutStore")
    records = [
        _final_batch_sample(
            [11],
            [20 + gen_id],
            prompt_uid="train:/tmp/train.jsonl:0",
            dataset_index=0,
            teacher_generation_id=gen_id,
        )
        for gen_id in range(3)
    ] + [
        _final_batch_sample(
            [12],
            [30 + gen_id],
            prompt_uid="train:/tmp/train.jsonl:1",
            dataset_index=1,
            teacher_generation_id=gen_id,
        )
        for gen_id in range(3)
    ]
    path = tmp_path / "teacher.jsonl"
    _write_jsonl(path, records)
    store = TeacherRolloutStore(path)

    forward = store.select_for_step(
        ["train:/tmp/train.jsonl:0", "train:/tmp/train.jsonl:1"],
        teacher_generations_per_prompt=1,
        step=7,
        seed=3,
    )
    reversed_order = store.select_for_step(
        ["train:/tmp/train.jsonl:1", "train:/tmp/train.jsonl:0"],
        teacher_generations_per_prompt=1,
        step=7,
        seed=3,
    )

    assert {
        prompt_uid: generation_id
        for prompt_uid, generation_id, _ in _selection_signature(forward)
    } == {
        prompt_uid: generation_id
        for prompt_uid, generation_id, _ in _selection_signature(reversed_order)
    }


def test_teacher_rollout_store_uses_prompt_uid_namespace_to_avoid_dataset_index_collisions(
    tmp_path: Path,
):
    TeacherRolloutStore = _get_api("TeacherRolloutStore")
    records = [
        _final_batch_sample(
            [11],
            [21],
            prompt_uid="train:/tmp/train-a.jsonl:7",
            dataset_index=7,
            teacher_generation_id=0,
        ),
        _final_batch_sample(
            [12],
            [22],
            prompt_uid="train:/tmp/train-b.jsonl:7",
            dataset_index=7,
            teacher_generation_id=0,
        ),
    ]
    path = tmp_path / "teacher.jsonl"
    _write_jsonl(path, records)
    store = TeacherRolloutStore(path)

    selection = store.select_for_step(
        ["train:/tmp/train-b.jsonl:7"],
        teacher_generations_per_prompt=1,
        step=0,
        seed=11,
    )

    assert _selection_signature(selection) == [("train:/tmp/train-b.jsonl:7", 0, 7)]


def test_teacher_rollout_store_rejects_prompt_uid_dataset_index_mismatch(
    tmp_path: Path,
):
    TeacherRolloutStore = _get_api("TeacherRolloutStore")
    record = _final_batch_sample(
        [11],
        [21],
        prompt_uid="train:/tmp/train.jsonl:7",
        dataset_index=8,
        teacher_generation_id=0,
    )
    path = tmp_path / "teacher.jsonl"
    _write_jsonl(path, [record])

    with pytest.raises((AssertionError, ValueError), match="prompt_uid|dataset_index"):
        TeacherRolloutStore(path)


def test_teacher_rollout_store_rejects_unexpected_dataset_namespace(tmp_path: Path):
    TeacherRolloutStore = _get_api("TeacherRolloutStore")
    record = _final_batch_sample(
        [11],
        [21],
        prompt_uid="train:/tmp/train-a.jsonl:7",
        dataset_index=7,
        teacher_generation_id=0,
    )
    path = tmp_path / "teacher.jsonl"
    _write_jsonl(path, [record])

    with pytest.raises((AssertionError, ValueError), match="namespace|prompt_uid"):
        TeacherRolloutStore(path, dataset_namespace="train:/tmp/train-b.jsonl")


def test_teacher_rollout_store_lists_missing_prompt_identities_in_error(tmp_path: Path):
    TeacherRolloutStore = _get_api("TeacherRolloutStore")
    path = tmp_path / "teacher.jsonl"
    _write_jsonl(
        path,
        [
            _final_batch_sample(
                [11],
                [21],
                prompt_uid="train:/tmp/train.jsonl:0",
                dataset_index=0,
                teacher_generation_id=0,
            )
        ],
    )
    store = TeacherRolloutStore(path)

    with pytest.raises((AssertionError, KeyError, ValueError), match="train:/tmp/train.jsonl:999"):
        store.select_for_step(
            ["train:/tmp/train.jsonl:999"],
            teacher_generations_per_prompt=1,
            step=0,
            seed=5,
        )


def test_teacher_rollout_store_rejects_corrupt_jsonl(tmp_path: Path):
    TeacherRolloutStore = _get_api("TeacherRolloutStore")
    path = tmp_path / "teacher.jsonl"
    path.write_text("{not valid json\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line 1"):
        TeacherRolloutStore(path)


def test_teacher_rollout_store_rejects_sampling_mismatch(tmp_path: Path):
    TeacherRolloutStore = _get_api("TeacherRolloutStore")
    record = _final_batch_sample(
        [11],
        [21],
        prompt_uid="train:/tmp/train.jsonl:0",
        dataset_index=0,
        teacher_generation_id=0,
    )
    path = tmp_path / "teacher.jsonl"
    _write_jsonl(path, [record])

    with pytest.raises((AssertionError, ValueError), match="sampling mismatch"):
        TeacherRolloutStore(
            path,
            sampling_config={"temperature": 1.0, "top_p": 0.95, "top_k": None},
        )

    TeacherRolloutStore(
        path,
        sampling_config={"temperature": 1.0, "top_p": 0.95, "top_k": None},
        require_sampling_match=False,
    )


def test_teacher_rollout_store_rejects_model_metadata_mismatch(tmp_path: Path):
    TeacherRolloutStore = _get_api("TeacherRolloutStore")
    record = _final_batch_sample(
        [11],
        [21],
        prompt_uid="train:/tmp/train.jsonl:0",
        dataset_index=0,
        teacher_generation_id=0,
    )
    path = tmp_path / "teacher.jsonl"
    _write_jsonl(path, [record])

    with pytest.raises((AssertionError, ValueError), match="model mismatch"):
        TeacherRolloutStore(path, expected_model_name="other-teacher")
    with pytest.raises((AssertionError, ValueError), match="tokenizer mismatch"):
        TeacherRolloutStore(path, expected_tokenizer_name_or_path="other-tokenizer")


def test_teacher_rollout_store_accepts_canonical_equivalent_model_paths(tmp_path: Path):
    TeacherRolloutStore = _get_api("TeacherRolloutStore")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    model_link = tmp_path / "model-link"
    model_link.symlink_to(model_dir, target_is_directory=True)
    record = _final_batch_sample(
        [11],
        [21],
        prompt_uid="train:/tmp/train.jsonl:0",
        dataset_index=0,
        teacher_generation_id=0,
    )
    record["model"] = {
        "name_or_path": str(model_dir),
        "tokenizer": str(model_dir),
    }
    path = tmp_path / "teacher.jsonl"
    _write_jsonl(path, [record])

    TeacherRolloutStore(
        path,
        expected_model_name=str(model_link),
        expected_tokenizer_name_or_path=str(model_link),
    )


def test_teacher_rollout_store_require_done_rejects_unfinalized_files(tmp_path: Path):
    TeacherRolloutStore = _get_api("TeacherRolloutStore")
    record = _final_batch_sample(
        [11],
        [21],
        prompt_uid="train:/tmp/train.jsonl:0",
        dataset_index=0,
        teacher_generation_id=0,
    )
    path = tmp_path / "teacher.jsonl"
    _write_jsonl(path, [record])

    with pytest.raises((AssertionError, ValueError), match="not finalized|done"):
        TeacherRolloutStore(path, require_done=True)

    path.with_name(f"{path.name}.done").write_text("{}\n", encoding="utf-8")
    path.with_name(f"{path.name}.inprogress").write_text("{}\n", encoding="utf-8")
    with pytest.raises((AssertionError, ValueError), match="in progress"):
        TeacherRolloutStore(path, require_done=True)

    path.with_name(f"{path.name}.inprogress").unlink()
    TeacherRolloutStore(path, require_done=True)


def test_teacher_rollout_store_rejects_insufficient_generations(tmp_path: Path):
    TeacherRolloutStore = _get_api("TeacherRolloutStore")
    record = _final_batch_sample(
        [11],
        [21],
        prompt_uid="train:/tmp/train.jsonl:0",
        dataset_index=0,
        teacher_generation_id=0,
    )
    path = tmp_path / "teacher.jsonl"
    _write_jsonl(path, [record])
    store = TeacherRolloutStore(path)

    with pytest.raises((AssertionError, ValueError), match="insufficient|needs 2"):
        store.select_for_step(
            ["train:/tmp/train.jsonl:0"],
            teacher_generations_per_prompt=2,
            step=0,
            seed=1,
        )


def test_parse_mixed_generation_config_returns_disabled_defaults_when_omitted():
    config = _parse_config(None, num_generations_per_prompt=4)

    assert _value(config, "enabled") is False
    assert _value(config, "source_layout") == "interleave"
    assert _value(config, "require_sampling_match") is True
    assert _value(config, "log_source_metrics") is True


def test_rollout_writer_resume_repairs_partial_tail_and_tracks_existing_keys(
    tmp_path: Path,
):
    writer = _load_rollout_writer_module()
    output_path = tmp_path / "teacher-rollouts.jsonl"
    first_record = {
        "prompt_uid": "train:data:0",
        "teacher_generation_id": 0,
    }
    second_record = {
        "prompt_uid": "train:data:1",
        "teacher_generation_id": 0,
    }
    output_path.write_text(
        json.dumps(first_record)
        + "\n"
        + json.dumps(second_record)
        + "\n"
        + '{"prompt_uid": "partial"',
        encoding="utf-8",
    )

    state = writer._prepare_output_for_write(
        output_path,
        overwrite=False,
        resume=True,
    )

    assert state.existing_record_count == 2
    assert state.existing_keys == {
        ("train:data:0", 0),
        ("train:data:1", 0),
    }
    assert output_path.read_text(encoding="utf-8") == (
        json.dumps(first_record) + "\n" + json.dumps(second_record) + "\n"
    )
    assert writer._missing_row_indices(
        [
            {"prompt_uid": "train:data:0", "teacher_generation_id": 0},
            {"prompt_uid": "train:data:0", "teacher_generation_id": 1},
            {"prompt_uid": "train:data:1", "teacher_generation_id": 0},
        ],
        state.existing_keys,
    ) == [1]


def test_rollout_writer_resume_requires_explicit_output_path():
    writer = _load_rollout_writer_module()

    with pytest.raises(ValueError, match="explicit --output"):
        writer._resolve_output_path(
            {"logger": {"log_dir": "/tmp/run"}},
            None,
            overwrite=False,
            resume=True,
        )


def test_rollout_writer_generation_config_loads_real_teacher_weights():
    writer = _load_rollout_writer_module()

    class _Tokenizer:
        pad_token_id = 0
        eos_token_id = 2

    generation_config = {
        "backend": "vllm",
        "stop_token_ids": None,
        "stop_strings": None,
        "vllm_cfg": {
            "expose_http_server": True,
        },
    }

    configured = writer._configure_teacher_rollout_generation(
        generation_config,
        _Tokenizer(),
    )

    assert configured["vllm_cfg"]["load_format"] == "auto"
    assert configured["vllm_cfg"]["skip_tokenizer_init"] is False
    assert configured["stop_token_ids"] == [2]


def test_rollout_writer_resume_done_sentinel_exits_without_rewrite(tmp_path: Path):
    writer = _load_rollout_writer_module()
    output_path = tmp_path / "teacher-rollouts.jsonl"
    output_path.write_text(
        json.dumps({"prompt_uid": "train:data:0", "teacher_generation_id": 0}) + "\n",
        encoding="utf-8",
    )
    writer._done_path(output_path).write_text("{}\n", encoding="utf-8")

    state = writer._prepare_output_for_write(
        output_path,
        overwrite=False,
        resume=True,
    )

    assert state.already_complete is True
    assert state.existing_record_count == 1


def test_rollout_writer_resume_rejects_metadata_mismatch(tmp_path: Path):
    writer = _load_rollout_writer_module()
    output_path = tmp_path / "teacher-rollouts.jsonl"
    output_path.write_text(
        json.dumps(
            {
                "prompt_uid": "train:data:0",
                "teacher_generation_id": 0,
                "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": None},
                "model": {"name_or_path": "teacher", "tokenizer": "tokenizer"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sampling mismatch"):
        writer._validate_existing_records_for_resume(
            output_path,
            sampling_config={"temperature": 1.0, "top_p": 0.95, "top_k": None},
            model_name="teacher",
            tokenizer_name="tokenizer",
        )


def test_parse_mixed_generation_config_rejects_source_counts_that_do_not_sum_to_total():
    with pytest.raises(
        (AssertionError, ValueError),
        match="num_generations_per_prompt|student|teacher",
    ):
        _parse_config(
            {
                "enabled": True,
                "student_generations_per_prompt": 1,
                "teacher_generations_per_prompt": 1,
                "teacher_rollout_path": "/tmp/teacher.jsonl",
            },
            num_generations_per_prompt=4,
        )


def test_parse_mixed_generation_config_requires_teacher_rollout_path_for_teacher_slots():
    with pytest.raises((AssertionError, ValueError), match="teacher_rollout_path"):
        _parse_config(
            {
                "enabled": True,
                "student_generations_per_prompt": 3,
                "teacher_generations_per_prompt": 1,
            },
            num_generations_per_prompt=4,
        )


@pytest.mark.parametrize(
    "raw_config",
    [
        {
            "enabled": True,
            "student_generations_per_prompt": 4,
            "teacher_generations_per_prompt": 0,
            "unknown_key": True,
        },
        {
            "enabled": True,
            "student_generations_per_prompt": 4,
            "teacher_generations_per_prompt": 0,
            "source_layout": "prefix",
        },
        {
            "enabled": True,
            "teacher_generations_per_prompt": 0,
        },
        {
            "enabled": True,
            "student_generations_per_prompt": -1,
            "teacher_generations_per_prompt": 5,
        },
    ],
)
def test_parse_mixed_generation_config_rejects_invalid_enabled_configs(raw_config):
    with pytest.raises((AssertionError, ValueError)):
        _parse_config(raw_config, num_generations_per_prompt=4)


def test_build_source_plan_interleaves_student_then_teacher_slots_per_prompt():
    source_plan = _build_source_plan(
        prompt_count=2,
        num_generations_per_prompt=4,
        student_generations_per_prompt=3,
        teacher_generations_per_prompt=1,
    )

    plan_rows = [
        (
            _value(item, "prompt_id"),
            _value(item, "generation_id"),
            _value(item, "source"),
        )
        for item in source_plan
    ]

    assert plan_rows == [
        (0, 0, "student"),
        (0, 1, "student"),
        (0, 2, "student"),
        (0, 3, "teacher"),
        (1, 0, "student"),
        (1, 1, "student"),
        (1, 2, "student"),
        (1, 3, "teacher"),
    ]


def test_build_student_rollout_input_repeats_only_student_slots_and_omits_stream_metadata():
    build_student_rollout_input = _get_api("build_student_rollout_input")
    batch = _simple_prompt_batch(prompt_count=2)
    source_plan = _build_source_plan(
        prompt_count=2,
        num_generations_per_prompt=4,
        student_generations_per_prompt=3,
        teacher_generations_per_prompt=1,
    )

    student_batch, final_slot_indices = build_student_rollout_input(batch, source_plan)

    assert len(student_batch["message_log"]) == 6
    assert list(final_slot_indices) == [0, 1, 2, 4, 5, 6]
    for key in STREAM_METADATA_KEYS:
        assert key not in student_batch


def test_build_student_rollout_input_deep_copies_repeated_list_rows():
    build_student_rollout_input = _get_api("build_student_rollout_input")
    batch = _simple_prompt_batch(prompt_count=1)
    batch["extra_env_info"][0] = {
        "responses_create_params": {"input": [{"role": "user", "content": "p0"}]},
        "agent_ref": {"type": "responses_api_agents", "name": "agent"},
    }
    source_plan = _build_source_plan(
        prompt_count=1,
        num_generations_per_prompt=4,
        student_generations_per_prompt=3,
        teacher_generations_per_prompt=1,
    )

    student_batch, _ = build_student_rollout_input(batch, source_plan)

    assert len({id(row) for row in student_batch["extra_env_info"]}) == 3
    student_batch["extra_env_info"][0]["_rowidx"] = 0
    assert "_rowidx" not in student_batch["extra_env_info"][1]
    assert "_rowidx" not in batch["extra_env_info"][0]

    assert len({id(row) for row in student_batch["message_log"]}) == 3
    student_batch["message_log"][0][0]["content"] = "changed"
    assert student_batch["message_log"][1][0]["content"] == ""
    assert batch["message_log"][0][0]["content"] == ""


@pytest.mark.parametrize(
    ("student_rows", "teacher_rows", "expected_sources"),
    [
        ([100, 101, 102, 103], [], ["student", "student", "student", "student"]),
        ([], [200, 201, 202, 203], ["teacher", "teacher", "teacher", "teacher"]),
        ([300, 301, 302], [400], ["student", "student", "student", "teacher"]),
    ],
)
def test_mix_rollout_batches_sets_rollout_source_for_supported_source_layouts(
    student_rows: list[int],
    teacher_rows: list[int],
    expected_sources: list[str],
):
    mix_rollout_batches = _get_api("mix_rollout_batches")
    source_plan = _build_source_plan(
        prompt_count=1,
        num_generations_per_prompt=4,
        student_generations_per_prompt=len(student_rows),
        teacher_generations_per_prompt=len(teacher_rows),
    )
    full_step_manifest = build_step_manifest(
        batch_id="mixed-step-0",
        step=0,
        prompt_count=1,
        num_generations_per_prompt=4,
        train_global_batch_size=4,
        max_sequence_length=16,
        tokenizer_name_or_path="unit-test-tokenizer",
        tokenizer_config={},
    )

    mixed_batch = mix_rollout_batches(
        _rollout_batch("student", student_rows),
        _rollout_batch("teacher", teacher_rows),
        source_plan,
        full_step_manifest,
    )

    assert mixed_batch["rollout_source"] == expected_sources
    assert mixed_batch["sample_ids"].tolist() == [0, 1, 2, 3]
    assert mixed_batch["generation_ids"].tolist() == [0, 1, 2, 3]


def test_mix_rollout_batches_builds_batch_that_flattens_and_restores_manifest_metadata():
    mix_rollout_batches = _get_api("mix_rollout_batches")
    source_plan = _build_source_plan(
        prompt_count=2,
        num_generations_per_prompt=4,
        student_generations_per_prompt=3,
        teacher_generations_per_prompt=1,
    )
    full_step_manifest = build_step_manifest(
        batch_id="mixed-step-7",
        step=7,
        prompt_count=2,
        num_generations_per_prompt=4,
        train_global_batch_size=4,
        max_sequence_length=16,
        tokenizer_name_or_path="unit-test-tokenizer",
        tokenizer_config={},
    )

    mixed_batch = mix_rollout_batches(
        _rollout_batch("student", [100, 101, 102, 103, 104, 105]),
        _rollout_batch("teacher", [200, 201]),
        source_plan,
        full_step_manifest,
    )

    flat_messages, input_lengths = batched_message_log_to_flat_message(
        mixed_batch["message_log"],
        pad_value_dict={"token_ids": 0},
    )
    train_data = BatchedDataDict(
        {
            "input_ids": flat_messages["token_ids"],
            "input_lengths": input_lengths,
            "token_mask": flat_messages["token_loss_mask"],
            "sample_mask": mixed_batch["loss_multiplier"],
        }
    )
    for key in STREAM_METADATA_KEYS:
        train_data[key] = mixed_batch[key]

    manifest = build_batch_manifest_from_train_data(
        batch_id=full_step_manifest.batch_id,
        step=full_step_manifest.step,
        train_data=train_data,
        metadata=mixed_batch,
        max_sequence_length=full_step_manifest.max_sequence_length,
        tokenizer_name_or_path=full_step_manifest.tokenizer_name_or_path,
        tokenizer_config=full_step_manifest.tokenizer_config,
    )
    conservation_oracle = build_conservation_oracle(manifest)

    assert mixed_batch["rollout_source"] == [
        "student",
        "student",
        "student",
        "teacher",
        "student",
        "student",
        "student",
        "teacher",
    ]
    conservation_oracle.validate_student_boundary(
        mixed_batch["sample_ids"],
        mixed_batch["sample_order"],
        mixed_batch["update_group"],
        mixed_batch["global_batch_slot"],
    )


def test_source_metrics_uses_contract_key_names_and_values():
    mix_rollout_batches = _get_api("mix_rollout_batches")
    source_metrics = _get_api("source_metrics")
    source_plan = _build_source_plan(
        prompt_count=1,
        num_generations_per_prompt=4,
        student_generations_per_prompt=3,
        teacher_generations_per_prompt=1,
    )
    full_step_manifest = build_step_manifest(
        batch_id="mixed-step-8",
        step=8,
        prompt_count=1,
        num_generations_per_prompt=4,
        train_global_batch_size=4,
        max_sequence_length=16,
        tokenizer_name_or_path="unit-test-tokenizer",
        tokenizer_config={},
    )
    mixed_batch = mix_rollout_batches(
        _rollout_batch("student", [10, 11, 12]),
        _rollout_batch("teacher", [20]),
        source_plan,
        full_step_manifest,
    )
    mixed_batch["truncated"][3] = True

    metrics = source_metrics(mixed_batch)

    assert metrics["mean_gen_tokens_per_sample"] == 2.0
    assert metrics["rollout/source/student/count"] == 3.0
    assert metrics["rollout/source/teacher/count"] == 1.0
    assert metrics["rollout/source/student/gen_tokens_mean"] == 2.0
    assert metrics["rollout/source/teacher/gen_tokens_mean"] == 2.0
    assert metrics["rollout/source/student/capped_ratio"] == 0.0
    assert metrics["rollout/source/teacher/capped_ratio"] == 1.0
    assert metrics["rollout/source/student/reward_mean"] == 11.0
    assert metrics["rollout/source/teacher/reward_mean"] == 20.0


def test_mix_rollout_batches_orders_rows_by_final_slot_when_source_plan_is_permuted():
    mix_rollout_batches = _get_api("mix_rollout_batches")
    source_plan = _build_source_plan(
        prompt_count=1,
        num_generations_per_prompt=4,
        student_generations_per_prompt=3,
        teacher_generations_per_prompt=1,
    )
    full_step_manifest = build_step_manifest(
        batch_id="mixed-step-9",
        step=9,
        prompt_count=1,
        num_generations_per_prompt=4,
        train_global_batch_size=4,
        max_sequence_length=16,
        tokenizer_name_or_path="unit-test-tokenizer",
        tokenizer_config={},
    )

    mixed_batch = mix_rollout_batches(
        _rollout_batch("student", [300, 301, 302]),
        _rollout_batch("teacher", [400]),
        list(reversed(source_plan)),
        full_step_manifest,
    )

    assistant_token_ids = [
        row[1]["token_ids"].tolist()[0] for row in mixed_batch["message_log"]
    ]
    assert mixed_batch["rollout_source"] == [
        "student",
        "student",
        "student",
        "teacher",
    ]
    assert assistant_token_ids == [20_300, 20_301, 20_302, 20_400]
    assert mixed_batch["sample_ids"].tolist() == [36, 37, 38, 39]
    assert mixed_batch["generation_ids"].tolist() == [0, 1, 2, 3]


def test_mix_rollout_batches_preserves_row_identity_for_fake_teacher_annotation():
    mix_rollout_batches = _get_api("mix_rollout_batches")
    source_plan = _build_source_plan(
        prompt_count=2,
        num_generations_per_prompt=4,
        student_generations_per_prompt=3,
        teacher_generations_per_prompt=1,
    )
    full_step_manifest = build_step_manifest(
        batch_id="mixed-step-10",
        step=10,
        prompt_count=2,
        num_generations_per_prompt=4,
        train_global_batch_size=4,
        max_sequence_length=16,
        tokenizer_name_or_path="unit-test-tokenizer",
        tokenizer_config={},
    )

    mixed_batch = mix_rollout_batches(
        _rollout_batch("student", [100, 101, 102, 103, 104, 105]),
        _rollout_batch("teacher", [200, 201]),
        list(reversed(source_plan)),
        full_step_manifest,
    )

    fake_teacher_topk_row_id = mixed_batch["sample_ids"] * 100 + mixed_batch[
        "generation_ids"
    ]
    row_identity = [
        (source, int(sample_id), int(row_id))
        for source, sample_id, row_id in zip(
            mixed_batch["rollout_source"],
            mixed_batch["sample_ids"],
            fake_teacher_topk_row_id,
            strict=True,
        )
    ]

    assert row_identity == [
        ("student", 80, 8000),
        ("student", 81, 8101),
        ("student", 82, 8202),
        ("teacher", 83, 8303),
        ("student", 84, 8400),
        ("student", 85, 8501),
        ("student", 86, 8602),
        ("teacher", 87, 8703),
    ]


def test_mix_rollout_batches_rejects_source_size_mismatch():
    mix_rollout_batches = _get_api("mix_rollout_batches")
    source_plan = _build_source_plan(
        prompt_count=1,
        num_generations_per_prompt=4,
        student_generations_per_prompt=3,
        teacher_generations_per_prompt=1,
    )
    full_step_manifest = build_step_manifest(
        batch_id="mixed-step-11",
        step=11,
        prompt_count=1,
        num_generations_per_prompt=4,
        train_global_batch_size=4,
        max_sequence_length=16,
        tokenizer_name_or_path="unit-test-tokenizer",
        tokenizer_config={},
    )

    with pytest.raises((AssertionError, ValueError), match="student_final_batch"):
        mix_rollout_batches(
            _rollout_batch("student", [100, 101]),
            _rollout_batch("teacher", [200]),
            source_plan,
            full_step_manifest,
        )


def test_mix_rollout_batches_rejects_manifest_mismatch():
    mix_rollout_batches = _get_api("mix_rollout_batches")
    source_plan = _build_source_plan(
        prompt_count=1,
        num_generations_per_prompt=4,
        student_generations_per_prompt=3,
        teacher_generations_per_prompt=1,
    )
    bad_manifest = build_step_manifest(
        batch_id="mixed-step-12",
        step=12,
        prompt_count=1,
        num_generations_per_prompt=4,
        train_global_batch_size=4,
        max_sequence_length=16,
        tokenizer_name_or_path="unit-test-tokenizer",
        tokenizer_config={},
    )
    bad_manifest.generation_ids[0] = 3

    with pytest.raises((AssertionError, ValueError), match="generation_id"):
        mix_rollout_batches(
            _rollout_batch("student", [100, 101, 102]),
            _rollout_batch("teacher", [200]),
            source_plan,
            bad_manifest,
        )
