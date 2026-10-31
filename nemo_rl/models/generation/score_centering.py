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

"""vLLM request preparation and sample extraction for score centering.

The current strategy collects the top-k head. Future strategies must keep
request preparation, validation and extraction consistent, and return the same
pair of sampler token IDs and their sampler logprobs. Staging consumes this
tensor contract regardless of how the candidates were selected.
"""

import math
from numbers import Real
from typing import Any, TypedDict

import torch

VLLM_LOGPROB_FLOOR = -9999.0


class ScoreCenteringRequestOverrides(TypedDict):
    """HTTP request fields used by the current top-k collection strategy."""

    logprobs: bool
    top_logprobs: int


def build_score_centering_request_overrides(
    *, num_candidates: int
) -> ScoreCenteringRequestOverrides:
    """Build request overrides for collecting score-centering candidates.

    Gym applies these fields to each outgoing policy request. Extend this
    function together with validation and extraction for another selection
    strategy, keeping the HTTP request consistent with engine sampling.

    Args:
        num_candidates: Number of sampler candidates to collect per token.

    Returns:
        HTTP request overrides. Callers merge them with other request settings.
    """
    if num_candidates <= 0:
        raise ValueError("Score-centering candidate count must be positive")
    return {"logprobs": True, "top_logprobs": num_candidates}


def validate_score_centering_request(request: Any, *, num_candidates: int) -> None:
    """Check that the HTTP request advertises the configured collection strategy.

    Args:
        request: Resolved OpenAI-compatible chat request on the vLLM worker.
        num_candidates: Candidate count configured for capture.
    """
    if getattr(request, "logprob_token_ids", None):
        raise ValueError(
            "Score-centering requests must omit logprob_token_ids: vLLM would "
            "return specific IDs instead of the sampler's top-k head."
        )
    expected = build_score_centering_request_overrides(num_candidates=num_candidates)
    if any(getattr(request, key, None) != value for key, value in expected.items()):
        raise ValueError(
            "Captured score-centering requests must set logprobs=true and "
            f"top_logprobs={num_candidates} over HTTP. "
            "Use a Gym capture adapter that preserves top_logprobs."
        )


def extract_score_centering_samples(
    output: Any, *, num_candidates: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract sampler candidate IDs and logprobs from one vLLM completion.

    The current strategy takes the lowest-ranked candidates from each engine
    logprob mapping, excluding an extra sampled token outside the top-k head.
    Candidate logprobs retain their original mass rather than being normalized
    over the selected set. Candidates the sampler filtered out (top-k, top-p)
    keep their ``-inf`` processed logprob.

    Args:
        output: One engine CompletionOutput, including per-position logprobs.
        num_candidates: Number of sampler candidates to retain per position.

    Returns:
        CPU ``(sampler_ids, sampler_logprobs)`` tensors, both shaped
        ``[generated_len, num_candidates]``, with int32 IDs and float32
        logprobs. Callers retain the existing ``sampler_topk_token_ids`` and
        ``sampler_topk_logprobs`` transport fields.
    """
    if num_candidates <= 0:
        raise ValueError("Score-centering candidate count must be positive")
    if len(output.logprobs or []) != len(output.token_ids):
        raise RuntimeError("Sampler top-k rows must match the generated token count")
    max_token_id = torch.iinfo(torch.int32).max
    min_logprob = torch.finfo(torch.float32).min
    for position in output.logprobs or []:
        for token_id, entry in position.items():
            lp = entry.logprob
            if (
                type(token_id) is not int
                or not 0 <= token_id <= max_token_id
                or isinstance(lp, bool)
                or not isinstance(lp, Real)
                or math.isnan(lp)
                or (float(lp) < min_logprob and lp != -math.inf)
                or float(lp) > 1e-5
            ):
                raise ValueError("Invalid sampler top-k token id or log probability")
    ranked = [
        sorted(position.items(), key=lambda item: item[1].rank)[:num_candidates]
        for position in output.logprobs or []
    ]
    if any(len(row) != num_candidates for row in ranked):
        raise RuntimeError(
            "vLLM returned fewer than top_k log-probs for a generated position "
            f"while extracting score-centering samples: choice_idx={output.index}, "
            f"top_k={num_candidates}."
        )
    for row in ranked:
        if len({token_id for token_id, _ in row}) != num_candidates:
            raise ValueError("Sampler top-k token ids must be unique")
        if sum(math.exp(entry.logprob) for _, entry in row) > 1.0001:
            raise ValueError("Sampler top-k probabilities exceed unit mass")
    sampler_ids = torch.tensor(
        [[token_id for token_id, _ in row] for row in ranked],
        dtype=torch.int32,
        device="cpu",
    ).reshape(-1, num_candidates)
    sampler_logprobs = torch.tensor(
        [[float(entry.logprob) for _, entry in row] for row in ranked],
        dtype=torch.float32,
        device="cpu",
    ).reshape(-1, num_candidates)
    return sampler_ids, sampler_logprobs
