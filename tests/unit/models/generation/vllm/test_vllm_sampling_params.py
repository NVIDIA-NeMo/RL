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

import pytest
import torch

from nemo_rl.models.generation.vllm.vllm_generation import VllmGeneration
from nemo_rl.models.generation.vllm.vllm_worker import BaseVllmGenerationWorker


def make_worker() -> BaseVllmGenerationWorker:
    worker = object.__new__(BaseVllmGenerationWorker)
    worker.cfg = {
        "top_k": None,
        "temperature": 0.7,
        "top_p": 0.9,
        "max_new_tokens": 128,
        "stop_token_ids": [2],
    }
    worker.SamplingParams = lambda **kwargs: kwargs
    return worker


def test_sampling_params_keep_normal_rollout_stop_semantics() -> None:
    params = make_worker()._build_sampling_params(
        greedy=False,
        stop_strings=["stop"],
        max_new_tokens=64,
    )

    assert params["max_tokens"] == 64
    assert "min_tokens" not in params
    assert "ignore_eos" not in params


def test_sampling_params_can_force_exact_trace_length() -> None:
    params = make_worker()._build_sampling_params(
        greedy=True,
        stop_strings=None,
        max_new_tokens=37,
        force_generation_length=True,
    )

    assert params["temperature"] == 0.0
    assert params["max_tokens"] == 37
    assert params["min_tokens"] == 37
    assert params["ignore_eos"] is True


def test_sampling_params_can_force_exact_trace_content() -> None:
    params = make_worker()._build_sampling_params(
        greedy=True,
        stop_strings=None,
        max_new_tokens=37,
        force_generation_length=True,
        allowed_token_ids=[1000],
    )

    assert params["temperature"] == 0.0
    assert params["allowed_token_ids"] == [1000]


def test_sampling_params_can_force_a_recorded_sequence() -> None:
    params = make_worker()._build_sampling_params(
        greedy=True,
        stop_strings=None,
        max_new_tokens=3,
        force_generation_length=True,
        forced_generation_token_ids=[7, 8, 9],
    )

    assert params["extra_args"] == {
        "nrl_forced_token_ids": [7, 8, 9]
    }
    assert "min_tokens" not in params
    assert "ignore_eos" not in params


def test_forced_sequence_processor_selects_each_recorded_position() -> None:
    from types import SimpleNamespace

    pytest.importorskip(
        "vllm",
        reason=(
            "the custom processor is loaded in the isolated vLLM worker "
            "environment and is validated there by "
            "validate_forced_sequence_processor.py"
        ),
    )
    from nemo_rl.models.generation.vllm.forced_sequence_logits_processor import (
        ForcedSequenceLogitsProcessor,
    )

    params = SimpleNamespace(
        max_tokens=3,
        extra_args={"nrl_forced_token_ids": [7, 8, 9]},
    )
    ForcedSequenceLogitsProcessor.validate_params(params)
    adapter = object.__new__(ForcedSequenceLogitsProcessor)
    processor = adapter.new_req_logits_processor(params)
    assert processor is not None
    for output_ids, target in (([], 7), ([7], 8), ([7, 8], 9)):
        logits = torch.arange(16, dtype=torch.float32)
        transformed = processor(output_ids, logits)
        assert int(transformed.argmax().item()) == target
        assert torch.isneginf(transformed).sum().item() == 15


def test_oracle_probe_metadata_does_not_replace_unknown_fallback() -> None:
    class NoopWorkerGroup:
        def shutdown(self, **_kwargs) -> bool:
            return True

    generation = object.__new__(VllmGeneration)
    generation._cross_dp_dispatcher = object()
    generation.worker_group = NoopWorkerGroup()
    generation._cross_dp_scheduler_mode = "oracle_probe_lfs"
    generation.cfg = {"max_new_tokens": 16384}

    _, catalog = generation._build_cross_dp_session(
        ["a", "a"],
        participant_count=1,
        participant_indices=None,
        request_costs=[10, 100],
    )

    assert [item["fallback_cost"] for item in catalog] == [16384, 16384]
    assert [item["oracle_cost"] for item in catalog] == [10, 100]
    generation._cross_dp_dispatcher = None


def test_predicted_lfs_metadata_is_group_aligned() -> None:
    generation = make_cross_dp_generation("predicted_lfs")

    _, catalog = generation._build_cross_dp_session(
        ["a", "a", "b"],
        participant_count=1,
        participant_indices=None,
        request_costs=None,
        predicted_group_costs=[100, 100, 20],
    )

    assert [item["predicted_cost"] for item in catalog] == [100, 100, 20]
    assert [item["fallback_cost"] for item in catalog] == [16384] * 3
    generation._cross_dp_dispatcher = None


def make_cross_dp_generation(mode: str = "lfs") -> VllmGeneration:
    class NoopWorkerGroup:
        def shutdown(self, **_kwargs) -> bool:
            return True

    generation = object.__new__(VllmGeneration)
    generation._cross_dp_dispatcher = object()
    generation.worker_group = NoopWorkerGroup()
    generation._cross_dp_scheduler_mode = mode
    generation.dp_size = 2
    generation.cfg = {"max_new_tokens": 16384}
    return generation


def test_cross_dp_session_builder_transmits_designated_probe_flags() -> None:
    generation = make_cross_dp_generation()

    _, catalog = generation._build_cross_dp_session(
        ["a", "a", "b"],
        participant_count=1,
        participant_indices=None,
        request_costs=None,
        designated_probe_flags=[False, True, True],
    )

    assert [item["group_id"] for item in catalog] == ["a", "a", "b"]
    assert [item["is_designated_probe"] for item in catalog] == [
        False,
        True,
        True,
    ]
    assert [item["fallback_cost"] for item in catalog] == [16384] * 3
    assert all("oracle_cost" not in item for item in catalog)
    generation._cross_dp_dispatcher = None


@pytest.mark.parametrize(
    ("flags", "message"),
    [
        ([True], "must align with group_ids"),
        ([False, 1, True], "only bool values"),
        ([False, False, True], "exactly one request per group"),
        ([True, True, True], "exactly one request per group"),
    ],
)
def test_cross_dp_session_builder_validates_designated_probe_flags(
    flags: list[bool],
    message: str,
) -> None:
    generation = make_cross_dp_generation()

    with pytest.raises(ValueError, match=message):
        generation._build_cross_dp_session(
            ["a", "a", "b"],
            participant_count=1,
            participant_indices=None,
            request_costs=None,
            designated_probe_flags=flags,
        )

    generation._cross_dp_dispatcher = None


def test_cross_dp_session_builder_transmits_exact_length_dp_pinning() -> None:
    generation = make_cross_dp_generation("exact_length_lpt")

    _, catalog = generation._build_cross_dp_session(
        ["a", "b", "c"],
        participant_count=1,
        participant_indices=None,
        request_costs=[100, 20, 10],
        preferred_dp_indices=[0, 1, 1],
    )

    assert [item["fallback_cost"] for item in catalog] == [100, 20, 10]
    assert [item["preferred_dp_idx"] for item in catalog] == [0, 1, 1]
    generation._cross_dp_dispatcher = None


@pytest.mark.parametrize(
    ("mode", "request_costs", "preferred_dp_indices", "message"),
    [
        ("lfs", [10, 20], [0, 1], "require.*exact_length_lpt"),
        (
            "exact_length_lpt",
            None,
            [0, 1],
            "require explicit exact-length request_costs",
        ),
        (
            "exact_length_lpt",
            [10, 20],
            [0],
            "must align with group_ids",
        ),
        (
            "exact_length_lpt",
            [10, 20],
            [True, 1],
            "ints \\(not bool\\)",
        ),
        (
            "exact_length_lpt",
            [10, 20],
            [0, 2],
            r"in \[0, 2\)",
        ),
    ],
)
def test_cross_dp_session_builder_validates_exact_length_dp_pinning(
    mode: str,
    request_costs: list[int] | None,
    preferred_dp_indices: list[int],
    message: str,
) -> None:
    generation = make_cross_dp_generation(mode)

    with pytest.raises(ValueError, match=message):
        generation._build_cross_dp_session(
            ["a", "b"],
            participant_count=1,
            participant_indices=None,
            request_costs=request_costs,
            preferred_dp_indices=preferred_dp_indices,
        )

    generation._cross_dp_dispatcher = None
