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

from copy import deepcopy

import pytest

from nemo_rl.environments.generation_contract import (
    bind_runtime_generation_contract,
    build_training_admission_contract,
    build_runtime_generation_contract,
    validate_runtime_generation_contract,
    validate_training_admission_contract,
)
from nemo_rl.environments.nemo_gym import _stamp_context_compaction_rollout_ids


class _BackendTokenizer:
    def to_str(self):
        return '{"normalizer":"test","model":"test"}'


class _Tokenizer:
    name_or_path = "checkpoint"
    chat_template = "template {{ messages }}"
    backend_tokenizer = _BackendTokenizer()
    special_tokens_map_extended = {"eos_token": "</s>"}
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 3
    unk_token_id = 4
    sep_token_id = None
    cls_token_id = None
    mask_token_id = None

    def get_vocab(self):
        return {"a": 10, "b": 11}

    def get_added_vocab(self):
        return {"</s>": 2}


class _ImageProcessor:
    model_input_names = ["pixel_values", "imgs_sizes"]

    def to_dict(self):
        return {"size": {"height": 512, "width": 512}, "do_normalize": True}


class _Processor:
    image_processor = _ImageProcessor()
    feature_extractor = None
    image_token = "<image>"
    image_token_id = 99
    model_input_names = ["input_ids", "pixel_values", "imgs_sizes"]

    def to_dict(self):
        return {"processor_class": "_Processor"}


def _build_contract(
    *,
    policy_kwargs=None,
    server_kwargs=None,
):
    policy_kwargs = policy_kwargs or {
        "enable_thinking": True,
        "truncate_history_thinking": False,
    }
    server_kwargs = server_kwargs or dict(policy_kwargs)
    return build_runtime_generation_contract(
        model_name="checkpoint",
        model_revision="checkpoint@revision",
        tokenizer=_Tokenizer(),
        processor=_Processor(),
        tokenizer_config={
            "name": "checkpoint",
            "chat_template": "default",
            "chat_template_kwargs": policy_kwargs,
        },
        generation_config={
            "vllm_cfg": {
                "precision": "bf16",
                "logprobs_mode": "raw_logprobs",
                "http_server_serving_chat_kwargs": {
                    "chat_template_kwargs": server_kwargs,
                    "reasoning_parser": "nano_v3",
                },
            },
            "vllm_kwargs": {
                "limit_mm_per_prompt": {"image": 8},
                "mm_processor_cache_gb": 0,
            },
        },
    )


def test_runtime_generation_contract_is_stable_and_complete():
    first = _build_contract()
    second = _build_contract()

    assert first == second
    assert first["training_eligible"]
    assert first["incomplete_reasons"] == []
    validate_runtime_generation_contract(first)


def test_runtime_generation_contract_records_template_mismatch_as_diagnostic():
    contract = _build_contract(
        server_kwargs={
            "enable_thinking": True,
            "truncate_history_thinking": True,
        }
    )

    assert contract["training_eligible"]
    assert contract["incomplete_reasons"] == []
    assert not contract["component_definitions"]["template"][
        "policy_and_serving_kwargs_match"
    ]
    validate_runtime_generation_contract(contract)


def test_runtime_generation_contract_detects_definition_corruption():
    contract = _build_contract()
    corrupted = deepcopy(contract)
    corrupted["component_definitions"]["processor"]["image_token"] = "<img>"

    with pytest.raises(ValueError, match="processor_contract_id"):
        validate_runtime_generation_contract(corrupted)


def _gym_generation_contract() -> dict:
    return {
        "generation_contract_id": "gym-generation-contract",
        "model_contract_id": "gym-model-contract",
        "tokenizer_contract_id": "gym-tokenizer-unavailable",
        "template_contract_id": "gym-template-unavailable",
        "sampling_contract_id": "gym-sampling-contract",
        "processor_contract_id": "gym-processor-unavailable",
        "compaction_policy_id": "gym-compaction-policy",
        "loss_normalization": "global_action_token_mean",
        "training_eligible": False,
        "incomplete_reasons": [
            "exact_tokenizer_identity_not_reported_by_generation_server",
            "exact_chat_template_identity_not_reported_by_generation_server",
            "exact_multimodal_processor_fingerprint_not_reported_by_generation_server",
        ],
    }


def test_training_admission_binds_gym_evidence_to_launcher_runtime():
    runtime = bind_runtime_generation_contract(
        _build_contract(),
        generation_policy_version="sync-policy-step-00000000",
    )
    generation_contract = _gym_generation_contract()
    admission = build_training_admission_contract(
        generation_contract,
        runtime,
    )
    validate_training_admission_contract(admission, generation_contract)

    assert admission["training_eligible"]
    assert (
        admission["source_generation_contract_id"]
        == (generation_contract["generation_contract_id"])
    )
    assert (
        admission["source_incomplete_reasons"]
        == generation_contract["incomplete_reasons"]
    )


def test_training_admission_requires_synchronized_policy_version():
    runtime = _build_contract()
    with pytest.raises(ValueError, match="generation-policy version"):
        build_training_admission_contract(
            _gym_generation_contract(),
            runtime,
        )


def test_training_admission_rejects_unknown_gym_gap():
    runtime = bind_runtime_generation_contract(
        _build_contract(),
        generation_policy_version="sync-policy-step-00000000",
    )
    generation_contract = _gym_generation_contract()
    generation_contract["incomplete_reasons"].append("unknown_generation_gap")

    with pytest.raises(ValueError, match="unsupported admission gaps"):
        build_training_admission_contract(generation_contract, runtime)


def test_training_admission_detects_source_contract_corruption():
    runtime = bind_runtime_generation_contract(
        _build_contract(),
        generation_policy_version="sync-policy-step-00000000",
    )
    generation_contract = _gym_generation_contract()
    admission = build_training_admission_contract(generation_contract, runtime)
    generation_contract["sampling_contract_id"] = "different-sampling-contract"

    with pytest.raises(ValueError, match="sampling_contract_id"):
        validate_training_admission_contract(admission, generation_contract)


def test_rollout_stamp_carries_bound_runtime_contract():
    runtime = bind_runtime_generation_contract(
        _build_contract(),
        generation_policy_version="sync-policy-step-00000000",
    )
    rows = [
        {
            "_rowidx": 0,
            "context_compaction_contract_version": 2,
            "context_compaction_group_id": "group",
            "context_compaction_task_id": "task",
            "context_compaction_rollout_index": 0,
            "context_compaction_attempt_index": 0,
        }
    ]

    _stamp_context_compaction_rollout_ids(
        rows,
        rollout_batch_index=99,
        runtime_contract=runtime,
    )

    assert rows[0]["context_compaction_runtime_contract"] == runtime
