# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Detach native decoded logprob buffers before the frontend queues them."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.utils.serial_utils import MsgpackDecoder, bytestr
    from vllm.v1.engine import EngineCoreOutputs


class LogprobBufferDetachment:
    """Copy only logprob views; preserve native output types and all other fields.

    Native decoded arrays can retain GC-tracked memoryviews and managed buffers
    while waiting in the output queue. Owned NumPy arrays preserve the numeric
    payload without those backing wrappers. This does not bound the queue or
    change the logprob container, selection, ranks, or speculative boundaries.
    Install on the live decoder before compact/timing wrappers and serving.
    """

    def __init__(self, decoder: MsgpackDecoder) -> None:
        # vLLM is optional in CPU-only tooling. Import its live output contract
        # only when installing the opt-in engine hook.
        from vllm.v1.engine import EngineCoreOutputs
        from vllm.v1.outputs import LogprobsLists

        if decoder.decoder.type is not EngineCoreOutputs:
            raise RuntimeError(
                "Logprob detachment requires the EngineCoreOutputs decoder"
            )
        if LogprobsLists._fields != (
            "logprob_token_ids",
            "logprobs",
            "sampled_token_ranks",
            "cu_num_generated_tokens",
        ):
            raise RuntimeError("Unverified LogprobsLists layout for queued detachment")
        self._decode = decoder.decode
        self._stats: dict[str, int | float] = {
            "messages": 0,
            "logprob_outputs": 0,
            "copied_outputs": 0,
            "arrays_copied": 0,
            "bytes_copied": 0,
            "detach_wall_s": 0.0,
            "detach_thread_cpu_s": 0.0,
        }
        decoder.decode = self.decode

    def stats(self) -> dict[str, int | float]:
        """Read numeric counters on the engine owner loop without heap scans."""
        return dict(self._stats)

    def decode(self, frames: bytestr | Sequence[bytestr]) -> EngineCoreOutputs:
        """Use the original decoder, then detach sample logprobs before queueing."""
        output = self._decode(frames)
        started, cpu_started = time.perf_counter(), time.thread_time()
        self._stats["messages"] += 1
        logprob_outputs = copied_outputs = arrays_copied = bytes_copied = 0
        try:
            for item in output.outputs:
                logprobs = item.new_logprobs
                if logprobs is None:
                    continue
                logprob_outputs += 1
                arrays = (
                    logprobs.logprob_token_ids,
                    logprobs.logprobs,
                    logprobs.sampled_token_ranks,
                )
                if all(array.flags.owndata for array in arrays):
                    continue
                owned = []
                for array in arrays:
                    if array.flags.owndata:
                        owned.append(array)
                    else:
                        owned.append(array.copy(order="C"))
                        arrays_copied += 1
                        bytes_copied += array.nbytes
                item.new_logprobs = logprobs._replace(
                    logprob_token_ids=owned[0],
                    logprobs=owned[1],
                    sampled_token_ranks=owned[2],
                )
                copied_outputs += 1
            return output
        finally:
            self._stats["logprob_outputs"] += logprob_outputs
            self._stats["copied_outputs"] += copied_outputs
            self._stats["arrays_copied"] += arrays_copied
            self._stats["bytes_copied"] += bytes_copied
            self._stats["detach_wall_s"] += time.perf_counter() - started
            self._stats["detach_thread_cpu_s"] += time.thread_time() - cpu_started
