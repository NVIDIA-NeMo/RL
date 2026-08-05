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

"""Runtime identities used to admit exact NeMo-Gym traces to training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


RUNTIME_GENERATION_CONTRACT_SCHEMA_VERSION = 1
_RUNTIME_COMPONENT_FIELDS = (
    "model_contract_id",
    "tokenizer_contract_id",
    "template_contract_id",
    "processor_contract_id",
)
_ALLOWED_GYM_IDENTITY_GAPS = frozenset(
    {
        "exact_tokenizer_identity_not_reported_by_generation_server",
        "exact_chat_template_identity_not_reported_by_generation_server",
        "exact_multimodal_processor_fingerprint_not_reported_by_generation_server",
    }
)


def _canonical_value(value: Any) -> Any:
    """Convert common runtime/config objects into stable JSON-compatible data."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return {"bytes_sha256": hashlib.sha256(value).hexdigest()}
    if is_dataclass(value) and not isinstance(value, type):
        return _canonical_value(asdict(value))
    if hasattr(value, "model_dump"):
        return _canonical_value(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (set, frozenset)):
        normalized = [_canonical_value(item) for item in value]
        return sorted(normalized, key=_canonical_json)
    if isinstance(value, Sequence):
        return [_canonical_value(item) for item in value]

    added_token_fields = (
        "content",
        "single_word",
        "lstrip",
        "rstrip",
        "normalized",
        "special",
    )
    if all(hasattr(value, field) for field in added_token_fields):
        return {
            field: _canonical_value(getattr(value, field))
            for field in added_token_fields
        }
    if hasattr(value, "to_dict"):
        return _canonical_value(value.to_dict())
    raise TypeError(
        "Generation-contract inputs must be canonically serializable; "
        f"unsupported value type {type(value)!r}"
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _canonical_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def stable_id(prefix: str, *parts: Any) -> str:
    return f"{prefix}-{canonical_digest(parts)[:24]}"


def _class_identity(value: Any) -> str | None:
    if value is None:
        return None
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _tokenizer_definition(
    tokenizer: Any,
    *,
    tokenizer_config: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    reasons: list[str] = []
    vocab = None
    if hasattr(tokenizer, "get_vocab"):
        vocab = tokenizer.get_vocab()
    if not isinstance(vocab, Mapping) or not vocab:
        reasons.append("exact_tokenizer_vocabulary_unavailable")

    backend_serialization = None
    backend_tokenizer = getattr(tokenizer, "backend_tokenizer", None)
    if backend_tokenizer is not None and hasattr(backend_tokenizer, "to_str"):
        backend_serialization = backend_tokenizer.to_str()

    definition = {
        "implementation": _class_identity(tokenizer),
        "configured_name": tokenizer_config.get("name"),
        "runtime_name_or_path": getattr(tokenizer, "name_or_path", None),
        "revision": tokenizer_config.get("revision"),
        "vocab_digest": canonical_digest(vocab) if vocab else None,
        "backend_serialization_digest": (
            hashlib.sha256(backend_serialization.encode("utf-8")).hexdigest()
            if isinstance(backend_serialization, str)
            else None
        ),
        "special_tokens_map": getattr(tokenizer, "special_tokens_map_extended", None),
        "special_token_ids": {
            field: getattr(tokenizer, field, None)
            for field in (
                "bos_token_id",
                "eos_token_id",
                "pad_token_id",
                "unk_token_id",
                "sep_token_id",
                "cls_token_id",
                "mask_token_id",
            )
        },
        "added_vocab": (
            tokenizer.get_added_vocab()
            if hasattr(tokenizer, "get_added_vocab")
            else None
        ),
    }
    if not definition["implementation"]:
        reasons.append("exact_tokenizer_implementation_unavailable")
    return definition, reasons


def _template_definition(
    tokenizer: Any,
    *,
    tokenizer_config: Mapping[str, Any],
    generation_config: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    template = getattr(tokenizer, "chat_template", None)

    policy_kwargs = tokenizer_config.get("chat_template_kwargs") or {}
    serving_config = generation_config.get("vllm_cfg") or {}
    serving_kwargs = serving_config.get("http_server_serving_chat_kwargs") or {}
    server_template = serving_kwargs.get("chat_template")
    server_template_kwargs = serving_kwargs.get("chat_template_kwargs") or {}

    return (
        {
            "template_content": template,
            "serving_chat_template_content": server_template,
            "policy_chat_template_setting": tokenizer_config.get("chat_template"),
            "policy_chat_template_kwargs": policy_kwargs,
            "serving_chat_template_kwargs": server_template_kwargs,
            "policy_and_serving_kwargs_match": (
                policy_kwargs == server_template_kwargs
            ),
            "policy_and_serving_template_match": (
                server_template is None or server_template == template
            ),
            "serving_chat_template_content_format": serving_kwargs.get(
                "chat_template_content_format", "auto"
            ),
            "reasoning_parser": serving_kwargs.get("reasoning_parser"),
            "reasoning_parser_plugin": serving_config.get("reasoning_parser_plugin"),
            "tool_parser": serving_kwargs.get("tool_parser"),
            "tool_parser_plugin": serving_config.get("tool_parser_plugin"),
            "serving_adapter_contract": "nemo_rl_vllm_responses_v1",
        },
        [],
    )


def _processor_component(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    definition: dict[str, Any] = {
        "implementation": _class_identity(value),
    }
    if hasattr(value, "to_dict"):
        definition["configuration"] = value.to_dict()
    if hasattr(value, "model_input_names"):
        definition["model_input_names"] = list(value.model_input_names)
    return definition


def _processor_definition(
    processor: Any,
    *,
    generation_config: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    reasons: list[str] = []
    if processor is None:
        reasons.append("exact_multimodal_processor_unavailable")
        return {}, reasons

    vllm_kwargs = generation_config.get("vllm_kwargs") or {}
    definition = {
        "implementation": _class_identity(processor),
        "processor": _processor_component(processor),
        "image_processor": _processor_component(
            getattr(processor, "image_processor", None)
        ),
        "feature_extractor": _processor_component(
            getattr(processor, "feature_extractor", None)
        ),
        "image_token": getattr(processor, "image_token", None),
        "image_token_id": getattr(processor, "image_token_id", None),
        "model_input_names": list(getattr(processor, "model_input_names", [])),
        "vllm_multimodal_configuration": {
            key: value
            for key, value in vllm_kwargs.items()
            if key.startswith("mm_")
            or key
            in {
                "limit_mm_per_prompt",
                "skip_mm_profiling",
                "media_io_kwargs",
            }
        },
        "training_materializer_contract": (
            "nemo_rl_exact_trace_multimodal_materializer_v1"
        ),
    }
    if not definition["implementation"]:
        reasons.append("exact_multimodal_processor_implementation_unavailable")
    if (
        definition["processor"] is None
        and definition["image_processor"] is None
        and definition["feature_extractor"] is None
    ):
        reasons.append("exact_multimodal_processor_configuration_unavailable")
    return definition, reasons


def build_runtime_generation_contract(
    *,
    model_name: str,
    model_revision: str,
    tokenizer: Any,
    processor: Any,
    tokenizer_config: Mapping[str, Any],
    generation_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Fingerprint the exact launcher-owned generation/training runtime.

    Sampling and compaction-policy identities are request-owned and are added by
    Gym. This contract covers the components NeMo-RL can independently know and
    compare before accepting Gym evidence for training.
    """
    reasons: list[str] = []
    if not isinstance(model_name, str) or not model_name:
        reasons.append("exact_model_identity_unavailable")

    tokenizer_definition, tokenizer_reasons = _tokenizer_definition(
        tokenizer,
        tokenizer_config=tokenizer_config,
    )
    template_definition, template_reasons = _template_definition(
        tokenizer,
        tokenizer_config=tokenizer_config,
        generation_config=generation_config,
    )
    processor_definition, processor_reasons = _processor_definition(
        processor,
        generation_config=generation_config,
    )
    reasons.extend(tokenizer_reasons)
    reasons.extend(template_reasons)
    reasons.extend(processor_reasons)

    model_definition = {
        "model_name": model_name,
        "model_revision": model_revision,
        "policy_model_name": tokenizer_config.get("name"),
        "generation_backend": "vllm",
        "precision": (generation_config.get("vllm_cfg") or {}).get("precision"),
        "quantization": generation_config.get("quant_cfg"),
        "hf_config_overrides": (generation_config.get("vllm_kwargs") or {}).get(
            "hf_overrides"
        ),
        "logprobs_mode": (generation_config.get("vllm_cfg") or {}).get("logprobs_mode"),
        "generation_adapter_contract": "nemo_rl_vllm_http_v1",
    }
    if tokenizer_config.get("name") != model_name:
        reasons.append("policy_model_and_tokenizer_checkpoint_differ")

    definitions = {
        "model": model_definition,
        "tokenizer": tokenizer_definition,
        "template": template_definition,
        "processor": processor_definition,
    }
    component_ids = {
        "model_contract_id": stable_id("model-contract", model_definition),
        "tokenizer_contract_id": stable_id("tokenizer-contract", tokenizer_definition),
        "template_contract_id": stable_id("template-contract", template_definition),
        "processor_contract_id": stable_id("processor-contract", processor_definition),
    }
    runtime_contract_id = stable_id(
        "generation-runtime-contract",
        canonical_digest(component_ids),
    )
    return {
        "schema_version": RUNTIME_GENERATION_CONTRACT_SCHEMA_VERSION,
        **component_ids,
        "runtime_contract_id": runtime_contract_id,
        "component_definitions": definitions,
        "training_eligible": not reasons,
        "incomplete_reasons": sorted(set(reasons)),
    }


def bind_runtime_generation_contract(
    contract: Mapping[str, Any],
    *,
    generation_policy_version: str,
) -> dict[str, Any]:
    """Bind a static runtime fingerprint to one synchronized weight version."""
    validate_runtime_generation_contract(contract)
    if not isinstance(generation_policy_version, str) or not generation_policy_version:
        raise ValueError("generation_policy_version must be a non-empty string")

    bound = deepcopy(dict(contract))
    definitions = bound["component_definitions"]
    definitions["model"]["generation_policy_version"] = generation_policy_version
    bound["model_contract_id"] = stable_id(
        "model-contract",
        definitions["model"],
    )
    component_ids = {field: bound[field] for field in _RUNTIME_COMPONENT_FIELDS}
    bound["runtime_contract_id"] = stable_id(
        "generation-runtime-contract",
        canonical_digest(component_ids),
    )
    validate_runtime_generation_contract(bound)
    return bound


def validate_runtime_generation_contract(contract: Mapping[str, Any]) -> None:
    """Independently validate a serialized launcher-owned runtime contract."""
    if contract.get("schema_version") != RUNTIME_GENERATION_CONTRACT_SCHEMA_VERSION:
        raise ValueError("Unsupported runtime generation contract schema")
    definitions = contract.get("component_definitions")
    if not isinstance(definitions, Mapping):
        raise ValueError("Runtime generation contract has no component definitions")
    expected_definitions = {
        "model_contract_id": ("model-contract", "model"),
        "tokenizer_contract_id": ("tokenizer-contract", "tokenizer"),
        "template_contract_id": ("template-contract", "template"),
        "processor_contract_id": ("processor-contract", "processor"),
    }
    for field, (prefix, definition_name) in expected_definitions.items():
        expected = stable_id(prefix, definitions.get(definition_name))
        if contract.get(field) != expected:
            raise ValueError(f"Runtime generation contract has invalid {field}")
    component_ids = {field: contract.get(field) for field in _RUNTIME_COMPONENT_FIELDS}
    expected_runtime_id = stable_id(
        "generation-runtime-contract",
        canonical_digest(component_ids),
    )
    if contract.get("runtime_contract_id") != expected_runtime_id:
        raise ValueError("Runtime generation contract identity is corrupted")
    reasons = contract.get("incomplete_reasons")
    if not isinstance(reasons, list) or not all(
        isinstance(reason, str) and reason for reason in reasons
    ):
        raise ValueError("Runtime generation contract has invalid incomplete reasons")
    if contract.get("training_eligible") is True and reasons:
        raise ValueError(
            "A training-eligible runtime generation contract cannot be incomplete"
        )


def build_training_admission_contract(
    generation_contract: Mapping[str, Any],
    runtime_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind immutable Gym evidence to NeMo-RL's synchronized training runtime."""
    validate_runtime_generation_contract(runtime_contract)
    model_definition = runtime_contract["component_definitions"]["model"]
    policy_version = model_definition.get("generation_policy_version")
    if not isinstance(policy_version, str) or not policy_version:
        raise ValueError(
            "NeMo-RL runtime generation contract has no synchronized "
            "generation-policy version"
        )
    if runtime_contract.get("training_eligible") is not True:
        raise ValueError("NeMo-RL runtime generation contract is incomplete")

    required_source_fields = (
        "generation_contract_id",
        "sampling_contract_id",
        "compaction_policy_id",
    )
    for field in required_source_fields:
        if (
            not isinstance(generation_contract.get(field), str)
            or not (generation_contract[field])
        ):
            raise ValueError(f"Gym generation contract has no {field}")
    if generation_contract.get("loss_normalization") != "global_action_token_mean":
        raise ValueError("Gym generation contract has unsupported loss normalization")

    source_reasons = generation_contract.get("incomplete_reasons") or []
    if not isinstance(source_reasons, (list, tuple)) or not all(
        isinstance(reason, str) and reason for reason in source_reasons
    ):
        raise ValueError("Gym generation contract has invalid incomplete reasons")
    unexpected_reasons = set(source_reasons) - _ALLOWED_GYM_IDENTITY_GAPS
    if unexpected_reasons:
        raise ValueError(
            "Gym generation contract has unsupported admission gaps: "
            f"{sorted(unexpected_reasons)!r}"
        )

    component_ids = {
        "source_generation_contract_id": generation_contract["generation_contract_id"],
        "runtime_contract_id": runtime_contract["runtime_contract_id"],
        "sampling_contract_id": generation_contract["sampling_contract_id"],
        "compaction_policy_id": generation_contract["compaction_policy_id"],
        "generation_policy_version": policy_version,
    }
    return {
        "schema_version": 1,
        **component_ids,
        "admission_contract_id": stable_id(
            "training-admission-contract",
            canonical_digest(component_ids),
        ),
        "runtime_contract": deepcopy(dict(runtime_contract)),
        "source_incomplete_reasons": list(source_reasons),
        "training_eligible": True,
        "incomplete_reasons": [],
    }


def validate_training_admission_contract(
    admission_contract: Mapping[str, Any],
    generation_contract: Mapping[str, Any],
) -> None:
    """Independently validate NeMo-RL's serialized training admission."""
    if admission_contract.get("schema_version") != 1:
        raise ValueError("Unsupported training admission contract schema")
    runtime_contract = admission_contract.get("runtime_contract")
    if not isinstance(runtime_contract, Mapping):
        raise ValueError("Training admission has no runtime contract")
    validate_runtime_generation_contract(runtime_contract)
    model_definition = runtime_contract["component_definitions"]["model"]
    policy_version = model_definition.get("generation_policy_version")
    if (
        admission_contract.get("generation_policy_version") != policy_version
        or not policy_version
    ):
        raise ValueError("Training admission policy version is corrupted")
    if admission_contract.get("runtime_contract_id") != runtime_contract.get(
        "runtime_contract_id"
    ):
        raise ValueError("Training admission runtime identity is corrupted")

    source_fields = {
        "source_generation_contract_id": "generation_contract_id",
        "sampling_contract_id": "sampling_contract_id",
        "compaction_policy_id": "compaction_policy_id",
    }
    for admission_field, source_field in source_fields.items():
        if admission_contract.get(admission_field) != generation_contract.get(
            source_field
        ):
            raise ValueError(f"Training admission disagrees with Gym {source_field}")
    source_reasons = admission_contract.get("source_incomplete_reasons")
    if source_reasons != list(generation_contract.get("incomplete_reasons") or []):
        raise ValueError("Training admission source gaps are corrupted")
    if set(source_reasons) - _ALLOWED_GYM_IDENTITY_GAPS:
        raise ValueError("Training admission contains unsupported source gaps")

    component_ids = {
        "source_generation_contract_id": admission_contract.get(
            "source_generation_contract_id"
        ),
        "runtime_contract_id": admission_contract.get("runtime_contract_id"),
        "sampling_contract_id": admission_contract.get("sampling_contract_id"),
        "compaction_policy_id": admission_contract.get("compaction_policy_id"),
        "generation_policy_version": admission_contract.get(
            "generation_policy_version"
        ),
    }
    expected_id = stable_id(
        "training-admission-contract",
        canonical_digest(component_ids),
    )
    if admission_contract.get("admission_contract_id") != expected_id:
        raise ValueError("Training admission contract identity is corrupted")
    if admission_contract.get("training_eligible") is not True:
        raise ValueError("Training admission is not training-eligible")
    if admission_contract.get("incomplete_reasons"):
        raise ValueError("Training admission is incomplete")
