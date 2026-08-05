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
import hashlib
import json
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import ray
import requests
import torch
from PIL import Image
from yaml import safe_load

from nemo_rl.algorithms.grpo import MasterConfig
from nemo_rl.distributed.ray_actor_environment_registry import (
    get_actor_python_env,
)
from nemo_rl.environments.generation_contract import canonical_digest, stable_id
from nemo_rl.environments.nemo_gym import (
    NemoGym,
    NemoGymConfig,
    _actor_peak_rss_gib,
    _attach_multimodal_data_to_user_message,
    _compact_json_size,
    _index_per_turn_images,
    _resolve_images_by_media_id,
    _stamp_context_compaction_rollout_ids,
    build_reward_component_columns,
    extract_reward_components,
    setup_nemo_gym_config,
    validate_reward_components_match_scalar,
)
from nemo_rl.models.generation.vllm import VllmGeneration

# cluster and tokenizer are fixture imports
from tests.unit.models.generation.test_vllm_generation import (
    basic_vllm_test_config,
    cluster,  # noqa: F401
)
from tests.unit.models.generation.test_vllm_generation import (
    tokenizer as nemo_gym_tokenizer,  # noqa: F401
)


def test_extract_reward_components():
    assert extract_reward_components({"reward": 1.0}) is None
    assert extract_reward_components({"reward": 1.0, "reward_components": {}}) is None
    assert extract_reward_components(
        {
            "reward": 2.0,
            "reward_components": {"correctness": 1, "format": 0.5},
        }
    ) == {"correctness": 1.0, "format": 0.5}


def test_build_reward_component_columns():
    from nemo_rl.algorithms.utils import get_gdpo_reward_component_keys

    columns = build_reward_component_columns(
        [
            {"correctness": 1.0, "format": 0.0},
            {"correctness": 0.0, "format": 1.0},
        ]
    )
    assert set(columns) == {"reward/correctness", "reward/format"}
    assert torch.equal(columns["reward/correctness"], torch.tensor([1.0, 0.0]))
    assert torch.equal(columns["reward/format"], torch.tensor([0.0, 1.0]))

    columns = build_reward_component_columns([{"b": 2.0}, {"a": 1.0, "b": 3.0}, None])
    assert list(columns) == ["reward/a", "reward/b"]
    assert torch.equal(columns["reward/a"], torch.tensor([0.0, 1.0, 0.0]))
    assert torch.equal(columns["reward/b"], torch.tensor([2.0, 3.0, 0.0]))
    assert get_gdpo_reward_component_keys(columns) == ["reward/a", "reward/b"]
    assert build_reward_component_columns([None, None]) == {}


def test_validate_reward_components_match_scalar():
    validate_reward_components_match_scalar(
        [{"reward": 1.5, "reward_components": {"correctness": 1.0, "format": 0.5}}]
    )
    validate_reward_components_match_scalar(
        [
            {
                "reward": 1.5000001,
                "reward_components": {"correctness": 1.0, "format": 0.5},
            }
        ]
    )
    validate_reward_components_match_scalar([{"reward": 2.0}])
    with pytest.raises(ValueError, match="result 1"):
        validate_reward_components_match_scalar(
            [
                {
                    "reward": 1.5,
                    "reward_components": {"correctness": 1.0, "format": 0.5},
                },
                {
                    "reward": 2.0,
                    "reward_components": {"correctness": 1.0, "format": 0.5},
                },
            ]
        )


_TEST_GENERATION_CONTRACT = {
    "generation_contract_id": "generation-contract-test",
    "sampling_contract_id": "sampling-contract-test",
    "compaction_policy_id": "compaction-policy-test",
    "loss_normalization": "global_action_token_mean",
    "training_eligible": False,
    "incomplete_reasons": [
        "exact_tokenizer_identity_not_reported_by_generation_server",
        "exact_chat_template_identity_not_reported_by_generation_server",
        "exact_multimodal_processor_fingerprint_not_reported_by_generation_server",
    ],
}
_TEST_FINAL_POLICY_DECISION = {
    "policy_name": "recency",
    "policy_version": "1",
    "config_digest": "policy-config",
    "retained_part_count": 1,
    "omitted_part_count": 0,
    "lineage": {
        "transformation_id": "transform-final",
        "transformation_type": "visual_recency",
        "transformation_version": "1",
        "configuration_digest": "policy-config",
        "deterministic": True,
        "lossy": False,
        "generator_contract_id": None,
        "unit_records": [
            {
                "source_unit_id": "part-final",
                "source_digest": "digest-final",
                "disposition": "kept",
                "output_unit_ids": ["part-final"],
                "output_digests": ["digest-final"],
            }
        ],
        "validator_result": "passed",
    },
}


def _test_runtime_contract() -> dict:
    definitions = {
        "model": {"generation_policy_version": "sync-policy-step-00000000"},
        "tokenizer": {"vocab": "test"},
        "template": {"template": "test"},
        "processor": {"processor": "test"},
    }
    component_ids = {
        "model_contract_id": stable_id("model-contract", definitions["model"]),
        "tokenizer_contract_id": stable_id(
            "tokenizer-contract", definitions["tokenizer"]
        ),
        "template_contract_id": stable_id("template-contract", definitions["template"]),
        "processor_contract_id": stable_id(
            "processor-contract", definitions["processor"]
        ),
    }
    return {
        "schema_version": 1,
        **component_ids,
        "runtime_contract_id": stable_id(
            "generation-runtime-contract",
            canonical_digest(component_ids),
        ),
        "component_definitions": definitions,
        "training_eligible": True,
        "incomplete_reasons": [],
    }


def test_actor_peak_rss_gib_converts_linux_kib(monkeypatch):
    peak_rss_gib = 3.25
    monkeypatch.setattr(
        "nemo_rl.environments.nemo_gym.resource.getrusage",
        lambda _: SimpleNamespace(ru_maxrss=peak_rss_gib * 1024**2),
    )

    assert _actor_peak_rss_gib() == peak_rss_gib


def _test_lineage_deltas(num_turns: int) -> list[dict]:
    final_record = _TEST_FINAL_POLICY_DECISION["lineage"]["unit_records"][0]
    payload = json.dumps(
        [final_record],
        sort_keys=True,
        separators=(",", ":"),
    )
    state_digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    parent = None
    deltas = []
    for turn_id in range(1, num_turns + 1):
        transformation_id = f"transform-{turn_id}"
        deltas.append(
            {
                "transformation_id": transformation_id,
                "parent_transformation_id": parent,
                "transformation_type": "visual_recency",
                "transformation_version": "1",
                "configuration_digest": "policy-config",
                "deterministic": True,
                "lossy": False,
                "generator_contract_id": None,
                "unit_upserts": ([final_record] if turn_id == 1 else []),
                "source_unit_count": 1,
                "state_digest": state_digest,
                "validator_result": "passed",
            }
        )
        parent = transformation_id
    return deltas


def _exact_evidence_contract_fields(
    *,
    turn_id: int,
    segment_index: int,
    media_ids: list[str],
    expected_append_compatible: bool,
    compaction_event_id: str | None = None,
) -> dict:
    action_id = f"action-{turn_id}"
    model_call_id = f"model-call-{turn_id}"
    return {
        "prepared_request_id": f"prepared-{turn_id}",
        "request_id": f"request-{turn_id}",
        "context_epoch": segment_index,
        "segment_index": segment_index,
        "segment_id": f"segment-{segment_index}",
        "expected_append_compatible": expected_append_compatible,
        "compaction_event_id": compaction_event_id,
        "generation_contract_id": _TEST_GENERATION_CONTRACT["generation_contract_id"],
        "policy_decision": {
            "policy_name": "recency",
            "policy_version": "1",
            "config_digest": "policy-config",
            "decision_turn": turn_id,
            "selection_digest": f"selection-{turn_id}",
            "transformation_id": f"transform-{turn_id}",
        },
        "policy_output_spans": [
            {
                "policy_output_span_id": f"span-{turn_id}",
                "model_call_id": model_call_id,
                "action_ids": [action_id],
                "start": 0,
                "end": 1,
                "eligible": turn_id == 1,
                "old_logprobs_alignment": "sampled_tokens",
            }
        ],
        "media_occurrences": [
            {
                "media_id": media_id,
                "occurrence_ordinal": ordinal,
                "model_call_id": model_call_id,
                "placeholder_span_or_position": None,
                "processed_dimensions": None,
                "model_specific_sidecars": {},
            }
            for ordinal, media_id in enumerate(media_ids)
        ],
    }


@pytest.mark.nemo_gym
def test_nemo_gym_stub_module():
    from nemo_gym import config_types

    print(
        f"NeMo-Gym test successfully run! NeMo-Gym config_types module: {config_types}"
    )


@pytest.fixture(scope="function")
def nemo_gym_vllm_generation(cluster, nemo_gym_tokenizer):  # noqa: F811
    generation_config = deepcopy(basic_vllm_test_config)
    master_config = MasterConfig.model_construct(
        policy={"generation": generation_config}
    )
    setup_nemo_gym_config(master_config, nemo_gym_tokenizer)

    generation_config["vllm_cfg"]["max_model_len"] = 16_384
    # This is the tool parser for Qwen/Qwen3-0.6B. This needs to be changed for other models.
    generation_config["vllm_cfg"]["http_server_serving_chat_kwargs"] = {
        "enable_auto_tools": True,
        "tool_parser": "hermes",
    }

    vllm_generation = VllmGeneration(cluster, generation_config)

    yield vllm_generation

    vllm_generation.shutdown()


@pytest.fixture(scope="function")
def nemo_gym(nemo_gym_vllm_generation):
    """Create a NeMo-Gym actor for testing."""

    yaml_str = r"""example_multi_step_resources_server:
  resources_servers:
    example_multi_step:
      entrypoint: app.py
      domain: instruction_following
example_multi_step_simple_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: example_multi_step_resources_server
      model_server:
        type: responses_api_models
        name: openai_model
openai_model:
  responses_api_models:
    vllm_model:
      entrypoint: app.py
      base_url: ${policy_base_url}
      api_key: ${policy_api_key}
      model: ${policy_model_name}
      return_token_id_information: true
      uses_reasoning_parser: true
rollout_max_attempts_to_avoid_lp_nan: 1
"""

    config = NemoGymConfig(
        model_name=nemo_gym_vllm_generation.cfg["model_name"],
        base_urls=nemo_gym_vllm_generation.dp_openai_server_base_urls,
        initial_global_config_dict=safe_load(yaml_str),
    )
    env = NemoGym.options(
        runtime_env={
            "py_executable": get_actor_python_env(
                "nemo_rl.environments.nemo_gym.NemoGym"
            ),
        }
    ).remote(config)

    # Blocking wait for NeMo-Gym to spin up
    ray.get(env._spinup.remote())

    yield env
    # Clean up the actor and wait for it to be killed
    env.shutdown.remote()
    ray.kill(env)
    # Give some time for cleanup
    time.sleep(0.1)


@pytest.fixture(scope="function")
def nemo_gym_sanity_test_data():
    fpath = Path(__file__).parent / "nemo_gym_test_data/test_nemo_gym_sanity.json"
    with open(fpath) as f:
        data = json.load(f)
    return data


def _write_actual_test_data(original_input: list, actual_result: list):
    """Write actual rollout results to actual_test_nemo_gym_sanity.json.

    This makes it easy to update the expected output after a Gym commit bump:
        cp nemo_gym_test_data/actual_test_nemo_gym_sanity.json nemo_gym_test_data/test_nemo_gym_sanity.json
    """

    def _convert(obj):
        """Recursively convert torch tensors to Python lists for JSON serialization."""
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj

    cleaned = deepcopy(actual_result)
    for r in cleaned:
        r.pop("full_result", None)
        for msg in r.get("message_log", [])[1:]:
            if "token_ids" in msg:
                msg["token_ids"] = []
            if "generation_logprobs" in msg:
                msg["generation_logprobs"] = []

    output_path = (
        Path(__file__).parent / "nemo_gym_test_data/actual_test_nemo_gym_sanity.json"
    )
    data = _convert({"input": original_input, "expected_output": cleaned})
    with open(output_path, "w") as f:
        json.dump(data, f)
        f.write("\n")
    print(f"Wrote updated test data to {output_path}")


def test_nemo_gym_postprocess_uses_batch_decode():
    class _Tokenizer:
        def __init__(self):
            self.batch_decode_calls = []

        def batch_decode(self, batch):
            self.batch_decode_calls.append([list(token_ids) for token_ids in batch])
            return [" ".join(map(str, token_ids)) for token_ids in batch]

    tokenizer = _Tokenizer()
    nemo_gym_result = {
        "response": {
            "output": [
                {
                    "prompt_token_ids": [1, 2],
                    "generation_token_ids": [3],
                    "generation_log_probs": [-0.1],
                },
                {
                    "prompt_token_ids": [1, 2, 3, 4, 5],
                    "generation_token_ids": [6, 7],
                    "generation_log_probs": [-0.2, -0.3],
                },
            ]
        },
        "responses_create_params": {"input": []},
    }

    class _MockSelf:
        cfg = {}

    result = (
        NemoGym.__ray_metadata__.modified_class._postprocess_nemo_gym_to_nemo_rl_result(
            _MockSelf(), {}, nemo_gym_result, tokenizer
        )
    )

    assert tokenizer.batch_decode_calls == [
        [[1, 2], [1, 2, 3, 4, 5]],
        [[3], [6, 7]],
    ]
    assert result["message_log"][0]["token_ids"].tolist() == [1, 2]
    assert result["message_log"][1]["token_ids"].tolist() == [3]
    assert result["message_log"][2]["token_ids"].tolist() == [4, 5]
    assert result["message_log"][3]["token_ids"].tolist() == [6, 7]
    assert len(result["physical_message_logs"]) == 1
    assert result["rollout_trace_bundle"]["checks"]["ok"]
    assert nemo_gym_result["response"]["output"][0]["prompt_str"] == "1 2"
    assert nemo_gym_result["response"]["output"][0]["generation_str"] == "3"
    assert nemo_gym_result["response"]["output"][1]["prompt_str"] == "1 2 3 4 5"
    assert nemo_gym_result["response"]["output"][1]["generation_str"] == "6 7"


def test_nemo_gym_postprocess_no_generation_data_raises():
    class _Tokenizer:
        def apply_chat_template(self, input_messages, tokenize=True):
            return list(range(1234))

    nemo_gym_result = {
        "response": {
            "output": [
                {"type": "reasoning"},
                {"type": "function_call"},
            ]
        },
        "responses_create_params": {"input": [{"role": "user", "content": "hi"}]},
    }

    class _MockSelf:
        cfg = {}

    with pytest.raises(ValueError) as excinfo:
        NemoGym.__ray_metadata__.modified_class._postprocess_nemo_gym_to_nemo_rl_result(
            _MockSelf(), {}, nemo_gym_result, _Tokenizer()
        )

    message = str(excinfo.value)
    assert "no generation data" in message
    assert "1234 tokens" in message
    assert "['reasoning', 'function_call']" in message


def test_nemo_gym_postprocess_no_generation_data_chat_template_failure():
    class _Tokenizer:
        def apply_chat_template(self, input_messages, tokenize=True):
            raise RuntimeError("boom")

    nemo_gym_result = {
        "response": {"output": [{"type": "reasoning"}]},
        "responses_create_params": {"input": [{"role": "user", "content": "hi"}]},
    }

    class _MockSelf:
        cfg = {}

    with pytest.raises(ValueError) as excinfo:
        NemoGym.__ray_metadata__.modified_class._postprocess_nemo_gym_to_nemo_rl_result(
            _MockSelf(), {}, nemo_gym_result, _Tokenizer()
        )

    message = str(excinfo.value)
    assert "no generation data" in message
    assert "apply_chat_template failed" in message
    assert "RuntimeError" in message
    assert "['reasoning']" in message


def test_nemo_gym_postprocess_allows_rewrite_only_for_generation_collection():
    class _Tokenizer:
        def batch_decode(self, batch):
            return [" ".join(map(str, token_ids)) for token_ids in batch]

    def make_result():
        return {
            "response": {
                "output": [
                    {
                        "prompt_token_ids": [1, 2],
                        "generation_token_ids": [3],
                        "generation_log_probs": [-0.1],
                    },
                    {
                        "prompt_token_ids": [1, 4],
                        "generation_token_ids": [5],
                        "generation_log_probs": [-0.2],
                    },
                ]
            },
            "responses_create_params": {"input": []},
        }

    class _MockSelf:
        cfg = {}

    postprocess = (
        NemoGym.__ray_metadata__.modified_class._postprocess_nemo_gym_to_nemo_rl_result
    )
    with pytest.raises(AssertionError, match="Non-contiguous messages"):
        postprocess(_MockSelf(), {}, make_result(), _Tokenizer())

    result = postprocess(
        _MockSelf(),
        {},
        make_result(),
        _Tokenizer(),
        generation_only=True,
    )

    assert [message["token_ids"].tolist() for message in result["message_log"]] == [
        [1, 2],
        [3],
        [1, 4],
        [5],
    ]
    assert [
        [message["token_ids"].tolist() for message in physical_trace]
        for physical_trace in result["physical_message_logs"]
    ] == [[[1, 2], [3]], [[1, 4], [5]]]
    assert result["rollout_trace_bundle"]["checks"]["physical_trace_count"] == 2
    assert not result["rollout_trace_bundle"]["checks"]["ok"]


def test_nemo_gym_postprocess_builds_exact_compacted_trace_bundle():
    class _Tokenizer:
        def batch_decode(self, batch):
            return [" ".join(map(str, token_ids)) for token_ids in batch]

    boundary = {
        "event_id": "boundary-2",
        "applies_to_step": 2,
        "reason": "history_policy_rewrite",
        "policy_name": "recency",
        "policy_version": "1",
        "config_digest": "policy-config",
    }
    rollout_id = "group-cc:batch-000000:row-000003"
    result_payload = {
        "response": {
            "context_compaction_contract": {
                "schema_version": 2,
                "mode": "exact_trace_authority",
                "rollout_id": rollout_id,
                "group_id": "group-cc",
                "task_id": "task-cc",
                "rollout_index": 3,
                "attempt_index": 0,
                "generation_contract": _TEST_GENERATION_CONTRACT,
            },
            "media_assets": {
                "screen-a": {
                    "type": "input_image",
                    "image_url": "data:image/png;base64,A",
                },
                "screen-b": {
                    "type": "input_image",
                    "image_url": "data:image/png;base64,B",
                },
            },
            "final_policy_decision": _TEST_FINAL_POLICY_DECISION,
            "lineage_deltas": _test_lineage_deltas(2),
            "boundary_events": [boundary],
            "completion_evidence": [
                {
                    "rollout_id": rollout_id,
                    "turn_id": 1,
                    "completion_id": "completion-1",
                    "action_id": "action-1",
                    "prompt_token_ids": [1],
                    "sampled_token_ids": [2],
                    "sampled_logprobs": [-0.1],
                    "media_ids": ["screen-a"],
                    "policy_decision": {
                        "policy_name": "recency",
                        "policy_version": "1",
                        "config_digest": "policy-config",
                    },
                    "finish_reason": "stop",
                    "eligible": True,
                    "evidence_source": "generation_response",
                    **_exact_evidence_contract_fields(
                        turn_id=1,
                        segment_index=0,
                        media_ids=["screen-a"],
                        expected_append_compatible=False,
                    ),
                },
                {
                    "rollout_id": rollout_id,
                    "turn_id": 2,
                    "completion_id": "completion-2",
                    "action_id": "action-2",
                    "prompt_token_ids": [8],
                    "sampled_token_ids": [9],
                    "sampled_logprobs": [-0.2],
                    "media_ids": ["screen-a", "screen-b"],
                    "policy_decision": {
                        "policy_name": "recency",
                        "policy_version": "1",
                        "config_digest": "policy-config",
                    },
                    "finish_reason": "stop",
                    "eligible": False,
                    "evidence_source": "generation_response",
                    **_exact_evidence_contract_fields(
                        turn_id=2,
                        segment_index=1,
                        media_ids=["screen-a", "screen-b"],
                        expected_append_compatible=False,
                        compaction_event_id="boundary-2",
                    ),
                },
            ],
            "output": [
                {
                    "prompt_token_ids": [1],
                    "generation_token_ids": [2],
                    "generation_log_probs": [-0.1],
                },
                {
                    "prompt_token_ids": [8],
                    "generation_token_ids": [9],
                    "generation_log_probs": [-0.2],
                },
            ],
        },
        "responses_create_params": {"input": []},
        "reward": 0.75,
    }
    row = {
        "_rowidx": 3,
        "context_compaction_rollout_id": rollout_id,
        "context_compaction_group_id": "group-cc",
        "context_compaction_task_id": "task-cc",
        "context_compaction_rollout_index": 3,
        "context_compaction_attempt_index": 0,
        "context_compaction_runtime_contract": _test_runtime_contract(),
    }

    class _MockSelf:
        cfg = {}

    training_result = (
        NemoGym.__ray_metadata__.modified_class._postprocess_nemo_gym_to_nemo_rl_result(
            _MockSelf(),
            row,
            deepcopy(result_payload),
            _Tokenizer(),
            generation_only=False,
        )
    )
    assert [
        [message["token_ids"].tolist() for message in physical_trace]
        for physical_trace in training_result["physical_message_logs"]
    ] == [[[1], [2]], [[8], [9]]]

    result = (
        NemoGym.__ray_metadata__.modified_class._postprocess_nemo_gym_to_nemo_rl_result(
            _MockSelf(),
            row,
            result_payload,
            _Tokenizer(),
            generation_only=True,
        )
    )

    bundle = result["rollout_trace_bundle"]
    assert bundle["checks"]["ok"]
    assert bundle["rollout_id"] == rollout_id
    assert bundle["group_id"] == "group-cc"
    assert bundle["source_row_index"] == 3
    assert bundle["reward"] == 0.75
    assert [trace["source_turn_ids"] for trace in bundle["physical_traces"]] == [
        [1],
        [2],
    ]
    assert bundle["physical_traces"][1]["segments"][1]["loss_mask"] == [0]
    assert bundle["training_admission"]["training_eligible"]
    assert bundle["generation_contract"]["training_eligible"] is False
    assert "nemo_rl_trace_bundle" not in result_payload
    assert result["full_result"]["nemo_rl_trace_bundle"] == bundle
    assert (
        result["full_result"]["context_compaction_gym_http_bytes"]
        > (result["full_result"]["context_compaction_ray_env_extras_bytes"])
    )
    assert (
        result["full_result"]["context_compaction_trajectory_record_bytes"]
        > (result["full_result"]["context_compaction_ray_env_extras_bytes"])
    )
    ray_projection = {
        key: value
        for key, value in result["full_result"].items()
        if key
        not in {
            "nemo_rl_trace_bundle",
            "context_compaction_trajectory_record_bytes",
        }
    }
    assert result["full_result"][
        "context_compaction_ray_env_extras_bytes"
    ] == _compact_json_size(ray_projection)
    assert result["full_result"][
        "context_compaction_trajectory_record_bytes"
    ] == _compact_json_size(result["full_result"])
    assert (
        0.0
        < result["full_result"]["context_compaction_transport_reduction_ratio"]
        < 1.0
    )
    projected_response = result["full_result"]["response"]
    assert set(projected_response).isdisjoint(
        {
            "agent_input",
            "seed_obs",
            "media_assets",
            "completion_evidence",
            "final_policy_decision",
            "lineage_deltas",
        }
    )


def test_nemo_gym_postprocess_exact_authority_rejects_missing_evidence():
    class _Tokenizer:
        def batch_decode(self, batch):
            return [" ".join(map(str, token_ids)) for token_ids in batch]

    rollout_id = "group-cc:batch-000000:row-000000"
    payload = {
        "response": {
            "context_compaction_contract": {
                "schema_version": 2,
                "mode": "exact_trace_authority",
                "rollout_id": rollout_id,
                "group_id": "group-cc",
                "task_id": "task-cc",
                "rollout_index": 0,
                "attempt_index": 0,
                "generation_contract": _TEST_GENERATION_CONTRACT,
            },
            "media_assets": {},
            "completion_evidence": [],
            "output": [
                {
                    "prompt_token_ids": [1],
                    "generation_token_ids": [2],
                    "generation_log_probs": [-0.1],
                }
            ],
        },
        "responses_create_params": {"input": []},
    }
    row = {
        "_rowidx": 0,
        "context_compaction_rollout_id": rollout_id,
        "context_compaction_group_id": "group-cc",
        "context_compaction_task_id": "task-cc",
        "context_compaction_rollout_index": 0,
        "context_compaction_attempt_index": 0,
    }

    postprocess = (
        NemoGym.__ray_metadata__.modified_class._postprocess_nemo_gym_to_nemo_rl_result
    )
    with pytest.raises(ValueError, match="missing completion_evidence"):
        postprocess(
            type("_MockSelf", (), {"cfg": {}})(),
            row,
            payload,
            _Tokenizer(),
            generation_only=True,
        )


def test_context_compaction_rollout_ids_are_unique_within_and_across_batches():
    rows = [
        {
            "_rowidx": row_index,
            "context_compaction_contract_version": 1,
            "context_compaction_group_id": "group-cc",
        }
        for row_index in range(2)
    ]
    _stamp_context_compaction_rollout_ids(rows, rollout_batch_index=4)

    assert [row["context_compaction_rollout_id"] for row in rows] == [
        "group-cc:batch-000004:row-000000",
        "group-cc:batch-000004:row-000001",
    ]

    next_batch = [dict(rows[0])]
    _stamp_context_compaction_rollout_ids(next_batch, rollout_batch_index=5)
    assert next_batch[0]["context_compaction_rollout_id"] == (
        "group-cc:batch-000005:row-000000"
    )


def test_context_compaction_training_rejects_rows_without_identity_contract():
    rows = [{"_rowidx": 0}]

    with pytest.raises(
        ValueError,
        match="training rows require context_compaction_contract_version",
    ):
        _stamp_context_compaction_rollout_ids(
            rows,
            rollout_batch_index=0,
            runtime_contract=_test_runtime_contract(),
        )


def test_v2_context_compaction_rollout_ids_are_retry_and_order_stable():
    rows = [
        {
            "_rowidx": 19,
            "context_compaction_contract_version": 2,
            "context_compaction_group_id": "group-cc",
            "context_compaction_task_id": task_id,
            "context_compaction_rollout_index": rollout_index,
            "context_compaction_attempt_index": 0,
        }
        for task_id, rollout_index in (("task-a", 0), ("task-b", 1))
    ]
    reordered = [dict(rows[1]), dict(rows[0])]

    _stamp_context_compaction_rollout_ids(rows, rollout_batch_index=4)
    _stamp_context_compaction_rollout_ids(reordered, rollout_batch_index=99)

    by_task = {
        row["context_compaction_task_id"]: row["context_compaction_rollout_id"]
        for row in rows
    }
    reordered_by_task = {
        row["context_compaction_task_id"]: row["context_compaction_rollout_id"]
        for row in reordered
    }
    assert by_task == reordered_by_task
    assert len(set(by_task.values())) == 2

    new_attempt = [dict(rows[0], context_compaction_attempt_index=1)]
    _stamp_context_compaction_rollout_ids(new_attempt, rollout_batch_index=4)
    assert new_attempt[0]["context_compaction_rollout_id"] != by_task["task-a"]


def test_index_per_turn_images_preserves_initial_and_observation_order():
    image_a = Image.new("RGB", (2, 2), "red")
    image_b = Image.new("RGB", (2, 2), "green")
    image_c = Image.new("RGB", (2, 2), "blue")
    initial_input = [
        {
            "role": "user",
            "content": [{"type": "input_image", "image": image_a}],
        }
    ]
    seed_obs = [
        {
            "role": "user",
            "content": [{"type": "input_image", "image": image_b}],
        }
    ]
    output = [
        {"role": "assistant", "generation_token_ids": [1]},
        {"role": "user", "content": [{"type": "input_text", "text": "none"}]},
        {"role": "assistant", "generation_token_ids": [2]},
        {
            "role": "user",
            "content": [
                {"type": "input_image", "image": image_c},
                {"type": "input_image", "image": image_a},
            ],
        },
        {"role": "assistant", "generation_token_ids": [3]},
    ]

    per_turn = _index_per_turn_images(initial_input, seed_obs, output)

    assert per_turn == [[image_a, image_b], [], [image_c, image_a]]


def test_media_arena_order_is_preserved_in_processor_packed_tensors():
    image_a = Image.new("RGB", (2, 2), (10, 20, 30))
    image_b = Image.new("RGB", (2, 2), (40, 50, 60))
    media_assets = {
        "image-a": {"type": "input_image", "image": image_a},
        "image-b": {
            "media_id": "image-b",
            "content_digest": "digest-b",
            "source_part": {"type": "input_image", "image": image_b},
            "original_dimensions": (2, 2),
            "color_mode": "RGB",
            "source_format": "png",
        },
    }
    images = _resolve_images_by_media_id(
        media_assets,
        ["image-b", "image-a", "image-b"],
    )

    class _Tokenizer:
        model_input_names = ["input_ids"]

    class _Processor:
        image_token = "<image>"
        model_input_names = ["input_ids", "pixel_values", "imgs_sizes"]
        tokenizer = _Tokenizer()

        def __init__(self):
            self.observed_colors = []

        def __call__(self, *, text, images, return_tensors):
            assert text == "<image><image><image>"
            assert return_tensors == "pt"
            self.observed_colors = [image.getpixel((0, 0)) for image in images]
            return {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "pixel_values": torch.tensor(self.observed_colors),
                "imgs_sizes": torch.tensor([[2, 2]] * len(images)),
            }

    processor = _Processor()
    user_message = {"role": "user", "content": "", "token_ids": torch.tensor([1])}
    _attach_multimodal_data_to_user_message(
        user_message,
        images=images,
        processor=processor,
    )

    assert processor.observed_colors == [
        (40, 50, 60),
        (10, 20, 30),
        (40, 50, 60),
    ]
    assert user_message["pixel_values"].tensors[0].tolist() == [
        [40, 50, 60],
        [10, 20, 30],
        [40, 50, 60],
    ]
    assert user_message["imgs_sizes"].tensors[0].dtype == torch.int32
    assert user_message["num_frames"].tensors[0].tolist() == [1, 1, 1]


@pytest.mark.nemo_gym
def test_nemo_gym_sanity(
    nemo_gym,
    nemo_gym_sanity_test_data,
    nemo_gym_vllm_generation,
    nemo_gym_tokenizer,  # noqa: F811
):
    """Test basic functionality of MathEnvironment step with simple messages."""

    # Save original input before mutation for writing the actual test data file
    original_input = deepcopy(nemo_gym_sanity_test_data["input"])

    # We need to match NeMo RL generation config params before sending to NeMo-Gym
    generation_config = nemo_gym_vllm_generation.cfg
    examples = nemo_gym_sanity_test_data["input"]
    for idx, example in enumerate(examples):
        example["responses_create_params"]["temperature"] = generation_config[
            "temperature"
        ]
        example["responses_create_params"]["top_p"] = generation_config["top_p"]
        example["_rowidx"] = idx

    actual_result, _ = ray.get(
        nemo_gym.run_rollouts.remote(
            nemo_gym_sanity_test_data["input"], nemo_gym_tokenizer, ""
        )
    )
    expected_result = nemo_gym_sanity_test_data["expected_output"]

    # These are tensors originally and we swap them back to a list for comparison below
    for d in actual_result:
        for message in d["input_message_log"]:
            message["token_ids"] = message["token_ids"].tolist()
        # Right now, we don't need to swap the token ids in the message log since they pointto the same underlying dictionary as above.
        # for message in d["message_log"][:1]:
        #     message["token_ids"] = message["token_ids"].tolist()

    # Write the actual result to a file so it can be used to update the expected output.
    # To update: cp actual_test_nemo_gym_sanity.json test_nemo_gym_sanity.json
    _write_actual_test_data(original_input, actual_result)

    def _standardize_single_result(d: dict):
        d = deepcopy(d)
        d.pop("full_result", None)
        d.pop("physical_message_logs", None)
        d.pop("rollout_trace_bundle", None)

        # We remove these fields and message from comparison since we cannot guarantee exact generation reproducibility
        d["message_log"] = d["message_log"][:2]
        for message in d["message_log"][1:]:
            if "token_ids" in message:
                message["token_ids"] = []
            if "generation_logprobs" in message:
                message["generation_logprobs"] = []
            if "prompt_str" in message:
                message["prompt_str"] = "dummy prompt_str"
            if "generation_str" in message:
                message["generation_str"] = "dummy generation_str"
            message.setdefault("is_invalid_tool_call", False)
            message.setdefault("has_malformed_thinking", False)

        return d

    def _standardize(l: list[dict]):
        return list(map(_standardize_single_result, l))

    assert _standardize(expected_result) == _standardize(actual_result)


# Sentinel for omitting the top_logprobs field entirely, which is distinct from sending null.
_OMIT_TOP_LOGPROBS = object()


@pytest.mark.nemo_gym
def test_vllm_http_logprobs_contract(nemo_gym_vllm_generation):
    """Pin the vLLM OpenAI HTTP logprobs contract that NeMo-Gym capture depends on.

    NeMo-Gym's vllm_model sets logprobs=True and return_tokens_as_token_ids=True to extract
    per-token ids and logprobs for training (Gym omits top_logprobs on the capture path, so
    vLLM applies its default; Gym PR #1612 additionally pins top_logprobs=0, which is
    equivalent). vLLM computes `logprobs = top_logprobs if logprobs else None`, so omitting
    top_logprobs (default 0) or sending 0 returns logprobs, while an explicit null returns
    none and silently empties the captured token ids. This exercises the real HTTP path where
    that translation lives (the offline LLM API does not), so a vLLM bump that changes the
    contract fails here instead of silently freezing training.

    All three cases share the (expensive) vLLM fixture, so they run in a single test rather
    than as separate parametrized cases.
    """
    base_url = nemo_gym_vllm_generation.dp_openai_server_base_urls[0]
    gen_cfg = nemo_gym_vllm_generation.cfg

    def _chat(top_logprobs_field):
        body = {
            "model": gen_cfg["model_name"],
            "messages": [{"role": "user", "content": "Say hello."}],
            "max_tokens": 8,
            # The RL HTTP wrapper asserts these match the generation config exactly.
            "temperature": gen_cfg["temperature"],
            "top_p": gen_cfg["top_p"],
            # The fields NeMo-Gym sets to capture token ids.
            "logprobs": True,
            "return_tokens_as_token_ids": True,
        }
        if top_logprobs_field is not _OMIT_TOP_LOGPROBS:
            body["top_logprobs"] = top_logprobs_field

        # The base URL is known once the fixture is ready, but retry briefly to avoid racing
        # the very first connection to the server.
        last_exc = None
        for _ in range(30):
            try:
                return requests.post(
                    f"{base_url}/chat/completions", json=body, timeout=60
                )
            except requests.exceptions.ConnectionError as e:
                last_exc = e
                time.sleep(1)
        raise AssertionError(f"vLLM HTTP server never became reachable: {last_exc}")

    def _assert_has_token_ids(resp, label):
        resp.raise_for_status()
        content = resp.json()["choices"][0]["logprobs"]["content"]
        assert content, f"expected per-token logprobs for {label}"
        # return_tokens_as_token_ids makes each token a "token_id:<int>" string; capture
        # parses these into ints, so they must all parse.
        token_ids = [int(c["token"].removeprefix("token_id:")) for c in content]
        assert len(token_ids) == len(content)

    # Omitting top_logprobs (what Gym does on the capture path; vLLM default 0) and sending 0
    # (the equivalent explicit pin) must both yield per-token logprobs whose tokens decode to ints.
    _assert_has_token_ids(_chat(_OMIT_TOP_LOGPROBS), "omitted top_logprobs")
    _assert_has_token_ids(_chat(0), "top_logprobs=0")

    # Explicit null is the divergence that motivates the Gym fix: vLLM returns no logprobs
    # (200 with logprobs=None) or rejects the request outright. Both mean capture gets
    # nothing. If a future vLLM makes null behave like 0, this fails and signals the Gym
    # workaround can be relaxed.
    null_resp = _chat(None)
    if null_resp.status_code == 200:
        assert null_resp.json()["choices"][0].get("logprobs") is None
    else:
        # A rejection must be a client-side validation error, not an unrelated server failure
        # that would let this branch pass vacuously.
        assert 400 <= null_resp.status_code < 500, (
            f"expected null top_logprobs accepted-with-None or rejected as 4xx, "
            f"got {null_resp.status_code}: {null_resp.text}"
        )
