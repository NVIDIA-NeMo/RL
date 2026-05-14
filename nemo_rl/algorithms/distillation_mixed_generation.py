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

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, MutableMapping, Sequence, cast

import torch

from nemo_rl.algorithms.distillation_streaming import (
    STREAM_METADATA_KEYS,
    StepManifest,
)
from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


TEACHER_ROLLOUT_SCHEMA_VERSION = 1
ROLLOUT_SOURCE_STUDENT = "student"
ROLLOUT_SOURCE_TEACHER = "teacher"
RolloutSource = Literal["student", "teacher"]
_FLOAT_JSON_PRECISION = 8

_ROLLOUT_BATCH_KEYS = (
    "agent_ref",
    "message_log",
    "length",
    "loss_multiplier",
    "total_reward",
    "truncated",
    "extra_env_info",
)
_TENSOR_ROLLOUT_BATCH_KEYS = {
    "length",
    "loss_multiplier",
    "total_reward",
    "truncated",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json_scalar(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        _require(value.numel() == 1, "expected a scalar tensor")
        return _json_scalar(value.detach().cpu().item())
    if isinstance(value, bool | int | str) or value is None:
        return value
    if isinstance(value, float):
        return round(float(value), _FLOAT_JSON_PRECISION)
    return value


def _jsonify(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.ndim == 0:
            return _json_scalar(tensor)
        if torch.is_floating_point(tensor):
            return [
                round(float(item), _FLOAT_JSON_PRECISION)
                for item in tensor.tolist()
            ]
        return tensor.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonify(item) for item in value]
    return _json_scalar(value)


def _canonical_path_string(value: Any) -> str:
    text = str(value).replace("\n", " ")
    path = Path(text).expanduser()
    if path.is_absolute() or text.startswith((".", "~")) or path.exists():
        return str(path.resolve())
    return text


def _identity_part(key: str, value: Any) -> str:
    if key == "data_path":
        return _canonical_path_string(value)
    return str(value).replace("\n", " ")


def _metadata_value_matches(actual: Any, expected: Any) -> bool:
    if actual == expected:
        return True
    if actual is None or expected is None:
        return False
    return _canonical_path_string(actual) == _canonical_path_string(expected)


def _mapping_get(value: Any, key: str) -> Any:
    getter = getattr(value, "get", None)
    if callable(getter):
        return getter(key)
    if isinstance(value, Mapping):
        return value.get(key)
    return None


def _is_mapping_like(value: Any) -> bool:
    return callable(getattr(value, "get", None)) or isinstance(value, Mapping)


def _is_non_string_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value,
        str | bytes | bytearray,
    )


def _sidecar_path(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.name}{suffix}")


def _tensor_row(value: torch.Tensor, index: int) -> torch.Tensor:
    row = value[index]
    if row.device.type != "cpu":
        row = row.cpu()
    return row.detach().clone()


def _row_value(batch: Mapping[str, Any], key: str, index: int) -> Any:
    value = batch[key]
    if isinstance(value, torch.Tensor):
        return _tensor_row(value, index)
    return copy.deepcopy(value[index])


def _message_to_json(message: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _jsonify(value) for key, value in message.items()}


def _message_from_json(message: Mapping[str, Any]) -> dict[str, Any]:
    converted = dict(message)
    if "token_ids" in converted:
        if isinstance(converted["token_ids"], torch.Tensor):
            converted["token_ids"] = converted["token_ids"].detach().clone().to(
                dtype=torch.int64,
                device="cpu",
            )
        else:
            converted["token_ids"] = torch.tensor(
                converted["token_ids"],
                dtype=torch.int64,
            )
    if "generation_logprobs" in converted:
        if isinstance(converted["generation_logprobs"], torch.Tensor):
            converted["generation_logprobs"] = converted[
                "generation_logprobs"
            ].detach().clone().to(dtype=torch.float64, device="cpu")
        else:
            converted["generation_logprobs"] = torch.tensor(
                converted["generation_logprobs"],
                dtype=torch.float64,
            )
    if "token_loss_mask" in converted:
        if isinstance(converted["token_loss_mask"], torch.Tensor):
            converted["token_loss_mask"] = converted["token_loss_mask"].detach().clone().to(
                dtype=torch.int64,
                device="cpu",
            )
        else:
            converted["token_loss_mask"] = torch.tensor(
                converted["token_loss_mask"],
                dtype=torch.int64,
            )
    return converted


def _normalize_generation_logprobs(message: MutableMapping[str, Any]) -> None:
    if "generation_logprobs" in message and isinstance(
        message["generation_logprobs"], torch.Tensor
    ):
        message["generation_logprobs"] = message["generation_logprobs"].to(
            dtype=torch.float64,
            device="cpu",
        )


def _ensure_loss_mask(message_log: LLMMessageLogType) -> None:
    for message in message_log:
        token_ids = message.get("token_ids")
        if not isinstance(token_ids, torch.Tensor):
            continue
        if message.get("role") == "assistant":
            message["token_loss_mask"] = torch.ones_like(token_ids)
        else:
            message["token_loss_mask"] = torch.zeros_like(token_ids)


def _copy_message_log(message_log: Sequence[Mapping[str, Any]]) -> LLMMessageLogType:
    copied: LLMMessageLogType = []
    for message in message_log:
        copied_message = copy.deepcopy(dict(message))
        _normalize_generation_logprobs(cast(MutableMapping[str, Any], copied_message))
        copied.append(copied_message)
    _ensure_loss_mask(copied)
    return copied


def _validate_message_log(record: Mapping[str, Any]) -> None:
    message_log = record.get("message_log")
    _require(isinstance(message_log, list), "message_log must be a list")
    _require(bool(message_log), "message_log must not be empty")

    first_message = cast(Mapping[str, Any], message_log[0])
    first_tokens = first_message.get("token_ids")
    _require(
        isinstance(first_tokens, torch.Tensor),
        "message_log[0].token_ids must be a tensor after deserialization",
    )
    _require(
        int(record["length"]) == int(first_tokens.numel()),
        "length must match the first prompt message token length",
    )

    for message_index, message in enumerate(cast(list[Mapping[str, Any]], message_log)):
        token_ids = message.get("token_ids")
        _require(
            isinstance(token_ids, torch.Tensor),
            f"message_log[{message_index}].token_ids must be present",
        )
        _require(
            token_ids.ndim == 1,
            f"message_log[{message_index}].token_ids must be rank-1",
        )
        _require(
            not torch.is_floating_point(token_ids),
            f"message_log[{message_index}].token_ids must be integral",
        )
        generation_logprobs = message.get("generation_logprobs")
        if generation_logprobs is not None:
            _require(
                isinstance(generation_logprobs, torch.Tensor),
                f"message_log[{message_index}].generation_logprobs must be a tensor",
            )
            _require(
                generation_logprobs.ndim == 1,
                f"message_log[{message_index}].generation_logprobs must be rank-1",
            )
            _require(
                torch.is_floating_point(generation_logprobs),
                (
                    f"message_log[{message_index}].generation_logprobs "
                    "must be floating point"
                ),
            )
            _require(
                int(generation_logprobs.numel()) == int(token_ids.numel()),
                (
                    f"message_log[{message_index}].generation_logprobs length must "
                    "match token_ids length"
                ),
            )
        token_loss_mask = message.get("token_loss_mask")
        if token_loss_mask is not None:
            _require(
                isinstance(token_loss_mask, torch.Tensor),
                f"message_log[{message_index}].token_loss_mask must be a tensor",
            )
            _require(
                token_loss_mask.ndim == 1,
                f"message_log[{message_index}].token_loss_mask must be rank-1",
            )
            _require(
                int(token_loss_mask.numel()) == int(token_ids.numel()),
                (
                    f"message_log[{message_index}].token_loss_mask length must "
                    "match token_ids length"
                ),
            )


def _validate_record_identity(
    record: Mapping[str, Any],
    *,
    dataset_namespace: str | None = None,
) -> None:
    prompt_uid = str(record["prompt_uid"])
    dataset_index = int(record["dataset_index"])
    expected_suffix = f":{dataset_index}"
    _require(
        prompt_uid.endswith(expected_suffix),
        (
            "prompt_uid and dataset_index are inconsistent: "
            f"prompt_uid={prompt_uid!r}, dataset_index={dataset_index}"
        ),
    )
    if dataset_namespace is None:
        return
    expected_prompt_uid = f"{dataset_namespace}:{dataset_index}"
    _require(
        prompt_uid == expected_prompt_uid,
        (
            "prompt_uid does not match the expected dataset namespace: "
            f"expected {expected_prompt_uid!r}, got {prompt_uid!r}"
        ),
    )


def serialize_final_batch_sample(
    final_batch: Mapping[str, Any],
    *,
    index: int,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialize one NeMo-Gym final_batch row as a JSONL-ready teacher record."""
    metadata = {} if metadata is None else dict(metadata)
    record: dict[str, Any] = {str(key): _jsonify(value) for key, value in metadata.items()}
    record.setdefault("schema_version", TEACHER_ROLLOUT_SCHEMA_VERSION)
    record.setdefault("source", ROLLOUT_SOURCE_TEACHER)

    for key in _ROLLOUT_BATCH_KEYS:
        if key not in final_batch:
            if key == "extra_env_info":
                record[key] = {}
                continue
            raise ValueError(f"final_batch is missing required key `{key}`")
        if key == "message_log":
            record[key] = [
                _message_to_json(message)
                for message in cast(Sequence[Mapping[str, Any]], final_batch[key][index])
            ]
        else:
            record[key] = _jsonify(_row_value(final_batch, key, index))

    _require("prompt_uid" in record, "metadata must include prompt_uid")
    _require("dataset_index" in record, "metadata must include dataset_index")
    _require(
        "teacher_generation_id" in record,
        "metadata must include teacher_generation_id",
    )
    return record


def deserialize_teacher_rollout_record(
    record: Mapping[str, Any],
    *,
    dataset_namespace: str | None = None,
) -> dict[str, Any]:
    """Validate and tensorize one serialized teacher rollout record."""
    _require(
        int(record.get("schema_version", -1)) == TEACHER_ROLLOUT_SCHEMA_VERSION,
        f"schema_version must be {TEACHER_ROLLOUT_SCHEMA_VERSION}",
    )
    _require(
        record.get("source") == ROLLOUT_SOURCE_TEACHER,
        "teacher rollout record source must be `teacher`",
    )
    for key in (
        "prompt_uid",
        "dataset_index",
        "teacher_generation_id",
        "agent_ref",
        "message_log",
        "length",
        "loss_multiplier",
        "total_reward",
        "truncated",
    ):
        _require(key in record, f"teacher rollout record missing `{key}`")

    converted = copy.deepcopy(dict(record))
    converted["dataset_index"] = int(converted["dataset_index"])
    converted["teacher_generation_id"] = int(converted["teacher_generation_id"])
    converted["length"] = int(converted["length"])
    converted["loss_multiplier"] = float(converted["loss_multiplier"])
    converted["total_reward"] = float(converted["total_reward"])
    converted["truncated"] = bool(converted["truncated"])
    converted["message_log"] = [
        _message_from_json(cast(Mapping[str, Any], message))
        for message in cast(Sequence[Mapping[str, Any]], converted["message_log"])
    ]
    converted.setdefault("extra_env_info", {})
    _validate_record_identity(converted, dataset_namespace=dataset_namespace)
    _validate_message_log(converted)
    return converted


def records_to_final_batch(records: Sequence[Mapping[str, Any]]) -> BatchedDataDict:
    """Convert deserialized teacher records into the final_batch layout."""
    seen: set[tuple[str, int]] = set()
    normalized: list[dict[str, Any]] = []
    for record in records:
        if isinstance(record["message_log"][0]["token_ids"], torch.Tensor):
            converted = copy.deepcopy(dict(record))
        else:
            converted = deserialize_teacher_rollout_record(record)
        key = (str(converted["prompt_uid"]), int(converted["teacher_generation_id"]))
        _require(
            key not in seen,
            (
                "duplicate teacher rollout for prompt_uid and "
                f"teacher_generation_id: {key}"
            ),
        )
        seen.add(key)
        normalized.append(converted)

    return BatchedDataDict(
        {
            "agent_ref": [record["agent_ref"] for record in normalized],
            "message_log": [record["message_log"] for record in normalized],
            "length": torch.tensor(
                [record["length"] for record in normalized],
                dtype=torch.int32,
            ),
            "loss_multiplier": torch.tensor(
                [record["loss_multiplier"] for record in normalized],
                dtype=torch.float32,
            ),
            "total_reward": torch.tensor(
                [record["total_reward"] for record in normalized],
                dtype=torch.float32,
            ),
            "truncated": torch.tensor(
                [record["truncated"] for record in normalized],
                dtype=torch.bool,
            ),
            "extra_env_info": [record.get("extra_env_info", {}) for record in normalized],
        }
    )


def build_prompt_identities_from_batch(
    batch: Mapping[str, Any],
    *,
    dataset_namespace: str | None = None,
) -> list[str]:
    """Build stable prompt identities shared by generation and mixed training."""
    batch_size = len(batch["idx"])
    extra_env_info = batch.get("extra_env_info", [{} for _ in range(batch_size)])
    task_names = batch.get("task_name")
    prompt_identities: list[str] = []
    for row_index, dataset_index in enumerate(batch["idx"]):
        namespaced_prompt_uid = None
        if dataset_namespace is not None:
            namespaced_prompt_uid = f"{dataset_namespace}:{int(dataset_index)}"
            prompt_identities.append(namespaced_prompt_uid)

        row_extra = extra_env_info[row_index] if extra_env_info is not None else {}
        explicit_prompt_uid = None
        if isinstance(row_extra, Mapping):
            for key in ("prompt_uid", "source_prompt_uid"):
                if key in row_extra:
                    explicit_prompt_uid = str(row_extra[key])
                    break
        if "prompt_uid" in batch:
            explicit_prompt_uid = str(batch["prompt_uid"][row_index])

        if namespaced_prompt_uid is not None:
            if explicit_prompt_uid is not None:
                _require(
                    explicit_prompt_uid == namespaced_prompt_uid,
                    (
                        "explicit prompt_uid does not match dataset namespace: "
                        f"expected {namespaced_prompt_uid!r}, "
                        f"got {explicit_prompt_uid!r}"
                    ),
                )
            continue

        if explicit_prompt_uid is not None:
            prompt_identities.append(explicit_prompt_uid)
            continue

        if isinstance(row_extra, Mapping):
            namespace = dataset_namespace
            if namespace is None and task_names is not None:
                namespace = str(task_names[row_index])
            if namespace is None:
                namespace = "dataset"
            prompt_identities.append(f"{namespace}:{int(dataset_index)}")
            continue

        namespace = dataset_namespace
        if namespace is None and task_names is not None:
            namespace = str(task_names[row_index])
        if namespace is None:
            namespace = "dataset"
        prompt_identities.append(f"{namespace}:{int(dataset_index)}")
    return prompt_identities


class TeacherRolloutStore:
    """Deterministic prompt_uid-indexed store for pre-generated teacher samples."""

    def __init__(
        self,
        path: str | Path,
        *,
        dataset_namespace: str | None = None,
        sampling_config: Mapping[str, Any] | None = None,
        require_sampling_match: bool = True,
        expected_model_name: str | None = None,
        expected_tokenizer_name_or_path: str | None = None,
        require_done: bool = False,
    ) -> None:
        self.path = Path(path)
        if require_done:
            _require(
                _sidecar_path(self.path, ".done").exists(),
                f"teacher rollout file is not finalized: missing {self.path.name}.done",
            )
            _require(
                not _sidecar_path(self.path, ".inprogress").exists(),
                (
                    "teacher rollout file appears to still be in progress: "
                    f"{self.path.name}.inprogress"
                ),
            )
        self.dataset_namespace = dataset_namespace
        self.sampling_config = None if sampling_config is None else dict(sampling_config)
        self.require_sampling_match = require_sampling_match
        self.expected_model_name = expected_model_name
        self.expected_tokenizer_name_or_path = expected_tokenizer_name_or_path
        self._by_prompt_uid: dict[str, list[TeacherRolloutRef]] = {}

        with self.path.open("r", encoding="utf-8") as handle:
            line_number = 0
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                line_number += 1
                if not line.strip():
                    continue
                try:
                    raw_record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"failed to parse teacher rollout JSONL at line {line_number}"
                    ) from exc
                ref = self._build_ref(
                    raw_record,
                    offset=offset,
                    line_number=line_number,
                )
                self._by_prompt_uid.setdefault(ref.prompt_uid, []).append(ref)

        seen: set[tuple[str, int]] = set()
        for prompt_uid, prompt_refs in self._by_prompt_uid.items():
            prompt_refs.sort(
                key=lambda ref: (
                    ref.teacher_generation_id,
                    ref.dataset_index,
                    ref.line_number,
                )
            )
            for ref in prompt_refs:
                key = (prompt_uid, ref.teacher_generation_id)
                _require(
                    key not in seen,
                    (
                        "duplicate teacher rollout for prompt_uid and "
                        f"teacher_generation_id: {key}"
                    ),
                )
                seen.add(key)

    def _build_ref(
        self,
        raw_record: Mapping[str, Any],
        *,
        offset: int,
        line_number: int,
    ) -> TeacherRolloutRef:
        _require(
            int(raw_record.get("schema_version", -1))
            == TEACHER_ROLLOUT_SCHEMA_VERSION,
            f"schema_version must be {TEACHER_ROLLOUT_SCHEMA_VERSION}",
        )
        _require(
            raw_record.get("source") == ROLLOUT_SOURCE_TEACHER,
            "teacher rollout record source must be `teacher`",
        )
        for key in ("prompt_uid", "dataset_index", "teacher_generation_id"):
            _require(key in raw_record, f"teacher rollout record missing `{key}`")

        _validate_record_identity(
            raw_record,
            dataset_namespace=self.dataset_namespace,
        )
        self._validate_sampling(raw_record, line_number=line_number)
        self._validate_model_metadata(raw_record, line_number=line_number)

        return TeacherRolloutRef(
            prompt_uid=str(raw_record["prompt_uid"]),
            dataset_index=int(raw_record["dataset_index"]),
            teacher_generation_id=int(raw_record["teacher_generation_id"]),
            offset=offset,
            line_number=line_number,
        )

    def _load_records(
        self,
        refs: Sequence[TeacherRolloutRef],
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for ref in refs:
                handle.seek(ref.offset)
                line = handle.readline()
                try:
                    raw_record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        "failed to parse teacher rollout JSONL at selected "
                        f"line {ref.line_number}"
                    ) from exc
                record = deserialize_teacher_rollout_record(
                    raw_record,
                    dataset_namespace=self.dataset_namespace,
                )
                self._validate_sampling(record, line_number=ref.line_number)
                self._validate_model_metadata(record, line_number=ref.line_number)
                records.append(record)
        return records

    def _validate_model_metadata(
        self,
        record: Mapping[str, Any],
        *,
        line_number: int,
    ) -> None:
        if self.expected_model_name is None and self.expected_tokenizer_name_or_path is None:
            return
        model_metadata = record.get("model")
        _require(
            isinstance(model_metadata, Mapping),
            f"teacher rollout line {line_number} missing model metadata",
        )
        if self.expected_model_name is not None:
            _require(
                _metadata_value_matches(
                    model_metadata.get("name_or_path"),
                    self.expected_model_name,
                ),
                (
                    f"teacher rollout line {line_number} model mismatch: "
                    f"expected {self.expected_model_name!r}, "
                    f"got {model_metadata.get('name_or_path')!r}"
                ),
            )
        if self.expected_tokenizer_name_or_path is not None:
            _require(
                _metadata_value_matches(
                    model_metadata.get("tokenizer"),
                    self.expected_tokenizer_name_or_path,
                ),
                (
                    f"teacher rollout line {line_number} tokenizer mismatch: "
                    f"expected {self.expected_tokenizer_name_or_path!r}, "
                    f"got {model_metadata.get('tokenizer')!r}"
                )
            )

    def _validate_sampling(self, record: Mapping[str, Any], *, line_number: int) -> None:
        if not self.require_sampling_match or self.sampling_config is None:
            return
        record_sampling = record.get("sampling")
        _require(
            isinstance(record_sampling, Mapping),
            f"teacher rollout line {line_number} missing sampling metadata",
        )
        for key in ("temperature", "top_p", "top_k"):
            if key not in self.sampling_config:
                continue
            _require(
                record_sampling.get(key) == self.sampling_config.get(key),
                (
                    f"teacher rollout line {line_number} sampling mismatch for "
                    f"{key}: expected {self.sampling_config.get(key)!r}, "
                    f"got {record_sampling.get(key)!r}"
                ),
            )

    def select_for_step(
        self,
        prompt_identities: Sequence[str],
        *,
        teacher_generations_per_prompt: int,
        step: int,
        seed: int,
    ) -> list[dict[str, Any]]:
        _require(
            teacher_generations_per_prompt >= 0,
            "teacher_generations_per_prompt must be non-negative",
        )
        if teacher_generations_per_prompt == 0:
            return []

        selected: list[TeacherRolloutRef] = []
        missing: list[str] = []
        insufficient: list[str] = []
        for prompt_uid in prompt_identities:
            candidates = self._by_prompt_uid.get(prompt_uid)
            if not candidates:
                missing.append(prompt_uid)
                continue
            if len(candidates) < teacher_generations_per_prompt:
                insufficient.append(
                    f"{prompt_uid} has {len(candidates)}, "
                    f"needs {teacher_generations_per_prompt}"
                )
                continue
            base = _stable_int(f"{seed}:{prompt_uid}")
            start = (base + step * teacher_generations_per_prompt) % len(candidates)
            for offset in range(teacher_generations_per_prompt):
                selected.append(candidates[(start + offset) % len(candidates)])

        _require(
            not missing,
            f"teacher rollout store missing prompt identities: {missing}",
        )
        _require(
            not insufficient,
            f"teacher rollout store has insufficient generations: {insufficient}",
        )
        return self._load_records(selected)


def _stable_int(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest, 16)


@dataclass(frozen=True)
class MixedGenerationConfig:
    enabled: bool = False
    teacher_rollout_path: str | None = None
    student_generations_per_prompt: int | None = None
    teacher_generations_per_prompt: int | None = None
    source_layout: Literal["interleave"] = "interleave"
    require_sampling_match: bool = True
    log_source_metrics: bool = True


@dataclass(frozen=True)
class TeacherRolloutRef:
    prompt_uid: str
    dataset_index: int
    teacher_generation_id: int
    offset: int
    line_number: int


def dataset_namespace_from_config(
    data_config: Mapping[str, Any],
    *,
    split: str = "train",
) -> str:
    """Build the prompt namespace used to join training prompts with rollouts."""
    split_config = _mapping_get(data_config, split)
    if _is_non_string_sequence(split_config):
        split_items = list(split_config)
        if len(split_items) == 1:
            split_config = split_items[0]
    if not _is_mapping_like(split_config):
        split_config = data_config

    parts = [split]
    for key in ("dataset_name", "data_path", "split", "subset"):
        value = _mapping_get(split_config, key)
        if value is not None:
            parts.append(_identity_part(key, value))
    if len(parts) == 1:
        parts.append("dataset")
    return ":".join(part.replace("\n", " ") for part in parts)


def parse_mixed_generation_config(
    raw_config: Mapping[str, Any] | None,
    *,
    num_generations_per_prompt: int,
) -> MixedGenerationConfig:
    if raw_config is None:
        return MixedGenerationConfig()

    allowed_keys = {
        "enabled",
        "teacher_rollout_path",
        "student_generations_per_prompt",
        "teacher_generations_per_prompt",
        "source_layout",
        "require_sampling_match",
        "log_source_metrics",
    }
    unknown_keys = sorted(set(raw_config) - allowed_keys)
    _require(not unknown_keys, f"unknown mixed_generation keys: {unknown_keys}")

    enabled = bool(raw_config.get("enabled", False))
    config = MixedGenerationConfig(
        enabled=enabled,
        teacher_rollout_path=cast(str | None, raw_config.get("teacher_rollout_path")),
        student_generations_per_prompt=cast(
            int | None,
            raw_config.get("student_generations_per_prompt"),
        ),
        teacher_generations_per_prompt=cast(
            int | None,
            raw_config.get("teacher_generations_per_prompt"),
        ),
        source_layout=cast(
            Literal["interleave"],
            raw_config.get("source_layout", "interleave"),
        ),
        require_sampling_match=bool(raw_config.get("require_sampling_match", True)),
        log_source_metrics=bool(raw_config.get("log_source_metrics", True)),
    )
    if not enabled:
        return config

    _require(config.source_layout == "interleave", "source_layout must be `interleave`")
    _require(
        config.student_generations_per_prompt is not None,
        "student_generations_per_prompt is required when mixed_generation is enabled",
    )
    _require(
        config.teacher_generations_per_prompt is not None,
        "teacher_generations_per_prompt is required when mixed_generation is enabled",
    )
    student_count = int(config.student_generations_per_prompt)
    teacher_count = int(config.teacher_generations_per_prompt)
    _require(student_count >= 0, "student_generations_per_prompt must be non-negative")
    _require(teacher_count >= 0, "teacher_generations_per_prompt must be non-negative")
    _require(
        student_count + teacher_count == num_generations_per_prompt,
        (
            "student_generations_per_prompt + teacher_generations_per_prompt "
            "must equal num_generations_per_prompt"
        ),
    )
    if teacher_count > 0:
        _require(
            bool(config.teacher_rollout_path),
            "teacher_rollout_path is required when teacher slots are requested",
        )
    return config


@dataclass(frozen=True)
class SourcePlanEntry:
    final_slot: int
    prompt_id: int
    generation_id: int
    source: RolloutSource
    source_index: int


def build_source_plan(
    *,
    prompt_count: int,
    num_generations_per_prompt: int,
    config: MixedGenerationConfig,
) -> list[SourcePlanEntry]:
    _require(prompt_count > 0, "prompt_count must be positive")
    _require(
        num_generations_per_prompt > 0,
        "num_generations_per_prompt must be positive",
    )
    _require(config.enabled, "mixed_generation config must be enabled")
    student_count = int(config.student_generations_per_prompt or 0)
    teacher_count = int(config.teacher_generations_per_prompt or 0)
    _require(
        student_count + teacher_count == num_generations_per_prompt,
        "source counts must match num_generations_per_prompt",
    )

    source_plan: list[SourcePlanEntry] = []
    student_index = 0
    teacher_index = 0
    final_slot = 0
    for prompt_id in range(prompt_count):
        for generation_id in range(student_count):
            source_plan.append(
                SourcePlanEntry(
                    final_slot=final_slot,
                    prompt_id=prompt_id,
                    generation_id=generation_id,
                    source=ROLLOUT_SOURCE_STUDENT,
                    source_index=student_index,
                )
            )
            student_index += 1
            final_slot += 1
        for generation_id in range(student_count, num_generations_per_prompt):
            source_plan.append(
                SourcePlanEntry(
                    final_slot=final_slot,
                    prompt_id=prompt_id,
                    generation_id=generation_id,
                    source=ROLLOUT_SOURCE_TEACHER,
                    source_index=teacher_index,
                )
            )
            teacher_index += 1
            final_slot += 1

    return source_plan


def build_student_rollout_input(
    prompt_batch: BatchedDataDict,
    source_plan: Sequence[SourcePlanEntry],
) -> tuple[BatchedDataDict, list[int]]:
    _validate_source_plan_slots(source_plan)
    student_entries = [
        entry
        for entry in sorted(source_plan, key=lambda item: item.final_slot)
        if entry.source == ROLLOUT_SOURCE_STUDENT
    ]
    prompt_indices = [entry.prompt_id for entry in student_entries]
    selected_batch = prompt_batch.select_indices(prompt_indices)
    for key, value in list(selected_batch.items()):
        if isinstance(value, list):
            selected_batch[key] = [copy.deepcopy(item) for item in value]
    for key in STREAM_METADATA_KEYS:
        selected_batch.pop(key, None)
    return selected_batch, [entry.final_slot for entry in student_entries]


def _validate_source_plan_slots(source_plan: Sequence[SourcePlanEntry]) -> None:
    final_slots = [entry.final_slot for entry in source_plan]
    _require(
        sorted(final_slots) == list(range(len(source_plan))),
        "source_plan final_slot values must be contiguous and unique",
    )


def _source_plan_by_final_slot(
    source_plan: Sequence[SourcePlanEntry],
    full_step_manifest: StepManifest,
) -> list[SourcePlanEntry]:
    _validate_source_plan_slots(source_plan)
    by_slot = sorted(source_plan, key=lambda entry: entry.final_slot)
    for entry in by_slot:
        _require(
            int(full_step_manifest.prompt_ids[entry.final_slot].item())
            == entry.prompt_id,
            "source_plan prompt_id must match full step manifest prompt_ids",
        )
        _require(
            int(full_step_manifest.generation_ids[entry.final_slot].item())
            == entry.generation_id,
            "source_plan generation_id must match full step manifest generation_ids",
        )
    return by_slot


def mix_rollout_batches(
    student_final_batch: BatchedDataDict,
    teacher_final_batch: BatchedDataDict,
    source_plan: Sequence[SourcePlanEntry],
    full_step_manifest: StepManifest,
) -> BatchedDataDict:
    source_plan_by_slot = _source_plan_by_final_slot(source_plan, full_step_manifest)
    student_count = sum(
        1 for entry in source_plan_by_slot if entry.source == ROLLOUT_SOURCE_STUDENT
    )
    teacher_count = sum(
        1 for entry in source_plan_by_slot if entry.source == ROLLOUT_SOURCE_TEACHER
    )
    _require(
        student_final_batch.size == student_count,
        (
            "student_final_batch size must match student source plan entries: "
            f"{student_final_batch.size} != {student_count}"
        ),
    )
    _require(
        teacher_final_batch.size == teacher_count,
        (
            "teacher_final_batch size must match teacher source plan entries: "
            f"{teacher_final_batch.size} != {teacher_count}"
        ),
    )

    output: dict[str, Any] = {}
    for key in _ROLLOUT_BATCH_KEYS:
        if key not in student_final_batch and key not in teacher_final_batch:
            continue
        rows: list[Any] = []
        for entry in source_plan_by_slot:
            batch = (
                student_final_batch
                if entry.source == ROLLOUT_SOURCE_STUDENT
                else teacher_final_batch
            )
            if key not in batch:
                rows.append({} if key == "extra_env_info" else None)
                continue
            row = _row_value(batch, key, entry.source_index)
            if key == "message_log":
                row = _copy_message_log(cast(Sequence[Mapping[str, Any]], row))
            rows.append(row)

        if key in _TENSOR_ROLLOUT_BATCH_KEYS:
            output[key] = torch.stack(cast(list[torch.Tensor], rows))
        else:
            output[key] = rows

    output["rollout_source"] = [entry.source for entry in source_plan_by_slot]
    for key in STREAM_METADATA_KEYS:
        metadata = getattr(full_step_manifest, key)
        _require(
            metadata.numel() == len(source_plan_by_slot),
            f"step manifest {key} length does not match source plan",
        )
        output[key] = metadata.clone()
    return BatchedDataDict(output)


def source_metrics(batch: Mapping[str, Any]) -> dict[str, float]:
    if "rollout_source" not in batch:
        return {}
    sources = cast(Sequence[str], batch["rollout_source"])
    gen_tokens = _assistant_token_counts(batch)
    if gen_tokens:
        mean_gen_tokens = sum(gen_tokens) / len(gen_tokens)
    else:
        mean_gen_tokens = 0.0
    metrics: dict[str, float] = {}
    metrics["mean_gen_tokens_per_sample"] = float(mean_gen_tokens)
    for source in (ROLLOUT_SOURCE_STUDENT, ROLLOUT_SOURCE_TEACHER):
        indices = [index for index, value in enumerate(sources) if value == source]
        prefix = f"rollout/source/{source}"
        if not indices:
            metrics[f"{prefix}/count"] = 0.0
            metrics[f"{prefix}/gen_tokens_mean"] = 0.0
            metrics[f"{prefix}/capped_ratio"] = 0.0
            metrics[f"{prefix}/reward_mean"] = 0.0
            continue
        metrics[f"{prefix}/count"] = float(len(indices))
        metrics[f"{prefix}/gen_tokens_mean"] = float(
            sum(gen_tokens[index] for index in indices) / len(indices)
        )
        if "total_reward" in batch:
            total_reward = cast(torch.Tensor, batch["total_reward"])[indices].float()
            metrics[f"{prefix}/reward_mean"] = float(total_reward.mean().item())
        else:
            metrics[f"{prefix}/reward_mean"] = 0.0
        if "truncated" in batch:
            truncated = cast(torch.Tensor, batch["truncated"])[indices].float()
            metrics[f"{prefix}/capped_ratio"] = float(truncated.mean().item())
        else:
            metrics[f"{prefix}/capped_ratio"] = 0.0
    return metrics


def _assistant_token_counts(batch: Mapping[str, Any]) -> list[int]:
    if "message_log" not in batch:
        return []
    counts: list[int] = []
    for message_log in cast(Sequence[Sequence[Mapping[str, Any]]], batch["message_log"]):
        count = 0
        for message in message_log:
            if message.get("role") != "assistant":
                continue
            token_ids = message.get("token_ids")
            if isinstance(token_ids, torch.Tensor):
                count += int(token_ids.numel())
            elif isinstance(token_ids, Sequence):
                count += len(token_ids)
        counts.append(count)
    return counts
