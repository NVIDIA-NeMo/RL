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

import weakref
from typing import Any

import pytest
import ray.cloudpickle
import torch

from nemo_rl.environments.nemo_gym import NemoGym
from nemo_rl.utils.routed_experts_codec import decode_routed_experts


class _Tokenizer:
    def batch_decode(self, batch):
        return [" ".join(map(str, token_ids)) for token_ids in batch]


def _routes(num_tokens: int) -> list[list[list[int]]]:
    return [[[token_idx, token_idx + 100]] for token_idx in range(num_tokens)]


def test_nemo_gym_postprocess_slices_routed_experts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decoded_refs: list[weakref.ReferenceType[torch.Tensor]] = []

    def track_decode(payload: Any, dtype: torch.dtype) -> torch.Tensor:
        assert all(reference() is None for reference in decoded_refs)
        decoded = decode_routed_experts(payload, dtype=dtype)
        decoded_refs.append(weakref.ref(decoded))
        return decoded

    monkeypatch.setattr(
        "nemo_rl.environments.nemo_gym.decode_routed_experts", track_decode
    )
    first_turn_routes = _routes(3)
    first_turn_routes[-1] = [[0, 1]]
    second_turn_routes = _routes(7)
    second_turn_routes[2] = [[30, 31]]
    nemo_gym_result = {
        "response": {
            "output": [
                {
                    "prompt_token_ids": [1, 2],
                    "generation_token_ids": [3],
                    "generation_log_probs": [-0.1],
                    "routed_experts": first_turn_routes,
                },
                {
                    "prompt_token_ids": [1, 2, 3, 4, 5],
                    "generation_token_ids": [6, 7],
                    "generation_log_probs": [-0.2, -0.3],
                    "routed_experts": second_turn_routes,
                },
            ]
        },
        "responses_create_params": {"input": []},
    }

    class _MockSelf:
        cfg = {"require_routed_experts": True}

    result = (
        NemoGym.__ray_metadata__.modified_class._postprocess_nemo_gym_to_nemo_rl_result(
            _MockSelf(), {}, nemo_gym_result, _Tokenizer()
        )
    )

    message_log = result["message_log"]
    assert message_log[0]["token_ids"].tolist() == [1, 2]
    assert message_log[0]["routed_experts"].tolist() == first_turn_routes[:2]
    assert message_log[1]["token_ids"].tolist() == [3]
    assert message_log[1]["routed_experts"].tolist() == second_turn_routes[2:3]
    assert message_log[2]["token_ids"].tolist() == [4, 5]
    assert message_log[2]["routed_experts"].tolist() == second_turn_routes[3:5]
    assert message_log[3]["token_ids"].tolist() == [6, 7]
    assert message_log[3]["routed_experts"].tolist() == second_turn_routes[5:7]
    assert all(reference() is None for reference in decoded_refs)

    restored = ray.cloudpickle.loads(ray.cloudpickle.dumps(result, protocol=5))
    for payload in (result, restored):
        for message, expected in zip(payload["message_log"], message_log):
            routes = message["routed_experts"]
            assert torch.equal(routes, expected["routed_experts"])
            assert routes.storage_offset() == 0
            assert (
                routes.untyped_storage().nbytes()
                == routes.numel() * routes.element_size()
            )


def test_nemo_gym_postprocess_tolerates_missing_routed_experts_when_configured():
    """A trainable item without routes is kept routeless, not rejected.

    Routes can be legitimately unrecoverable on the echo path (e.g. a
    context-overflow rollout); ``backfill_missing_routed_experts`` sentinel-fills
    such messages at flatten time and a batch-wide absence still fails at the
    rollout actor's ``ROUTED_EXPERTS_FIELD`` guard.
    """
    nemo_gym_result = {
        "response": {
            "output": [
                {
                    "prompt_token_ids": [1, 2],
                    "generation_token_ids": [3],
                    "generation_log_probs": [-0.1],
                },
            ]
        },
        "responses_create_params": {"input": []},
    }

    class _MockSelf:
        cfg = {"require_routed_experts": True}

    result = (
        NemoGym.__ray_metadata__.modified_class._postprocess_nemo_gym_to_nemo_rl_result(
            _MockSelf(), {}, nemo_gym_result, _Tokenizer()
        )
    )

    message_log = result["message_log"]
    assert [message["role"] for message in message_log] == ["user", "assistant"]
    assert message_log[1]["token_ids"].tolist() == [3]
    assert all("routed_experts" not in message for message in message_log)


def test_nemo_gym_postprocess_casts_routed_experts_to_configured_dtype():
    nemo_gym_result = {
        "response": {
            "output": [
                {
                    "prompt_token_ids": [1, 2],
                    "generation_token_ids": [3],
                    "generation_log_probs": [-0.1],
                    "routed_experts": _routes(3),
                },
            ]
        },
        "responses_create_params": {"input": []},
    }

    class _MockSelf:
        cfg = {"require_routed_experts": True, "routed_experts_dtype": "int8"}

    result = (
        NemoGym.__ray_metadata__.modified_class._postprocess_nemo_gym_to_nemo_rl_result(
            _MockSelf(), {}, nemo_gym_result, _Tokenizer()
        )
    )

    for message in result["message_log"]:
        if "routed_experts" in message:
            assert message["routed_experts"].dtype == torch.int8
