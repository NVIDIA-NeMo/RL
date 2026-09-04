"""Benchmark-only exact token-sequence replay for vLLM 0.17."""

from __future__ import annotations

import os
from typing import Any, Optional

import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import (
    AdapterLogitsProcessor,
    RequestLogitsProcessor,
)


class _ForcedSequence:
    def __init__(self, token_ids: list[int]) -> None:
        self.token_ids = token_ids

    def __call__(
        self, output_ids: list[int], logits: torch.Tensor
    ) -> torch.Tensor:
        position = len(output_ids)
        if position >= len(self.token_ids):
            raise RuntimeError(
                "forced-sequence processor called beyond recorded sequence: "
                f"position={position}, length={len(self.token_ids)}"
            )
        target = self.token_ids[position]
        if target >= logits.numel():
            raise RuntimeError(
                f"forced token {target} is outside logits size {logits.numel()}"
            )
        value = logits[target].clone()
        logits[:] = float("-inf")
        logits[target] = value
        return logits


class ForcedSequenceLogitsProcessor(AdapterLogitsProcessor):
    """Force each request to emit its recorded output token IDs."""

    ARGUMENT = "nrl_forced_token_ids"

    @classmethod
    def validate_params(cls, params: SamplingParams) -> None:
        value: Any = params.extra_args and params.extra_args.get(cls.ARGUMENT)
        if os.environ.get("NRL_FORCED_SEQUENCE_AUDIT") == "1":
            print(
                "[FORCED-SEQUENCE-AUDIT] validate_params "
                f"has_value={value is not None} "
                f"length={len(value) if isinstance(value, list) else None}",
                flush=True,
            )
        if value is None:
            return
        if (
            not isinstance(value, list)
            or not value
            or any(type(token_id) is not int or token_id < 0 for token_id in value)
        ):
            raise ValueError(
                f"{cls.ARGUMENT} must be a non-empty list of non-negative ints"
            )
        if params.max_tokens != len(value):
            raise ValueError(
                f"{cls.ARGUMENT} length ({len(value)}) must equal max_tokens "
                f"({params.max_tokens})"
            )

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(
        self, params: SamplingParams
    ) -> Optional[RequestLogitsProcessor]:
        self.validate_params(params)
        value = params.extra_args and params.extra_args.get(self.ARGUMENT)
        if os.environ.get("NRL_FORCED_SEQUENCE_AUDIT") == "1":
            print(
                "[FORCED-SEQUENCE-AUDIT] new_req_logits_processor "
                f"has_value={value is not None}",
                flush=True,
            )
        return None if value is None else _ForcedSequence(list(value))
