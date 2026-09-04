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

"""Live in-flight rollout batch profiler for vLLM generation.

At a fixed cadence *during* a ``generate()`` call this samples the in-process
vLLM V1 scheduler's running queue, capturing the per-data-parallel-rank in-flight
batch composition: the number of running sequences (batch size) and the context
length of every running sequence (prompt + tokens generated so far). Sampling
each data-parallel replica concurrently over the same wall-clock window gives the
"global" view of how each worker's batch and context lengths evolve over time
(batch decay, straggler tails, KV pressure) for rollout performance analysis.

Enabled via ``NRL_PROFILE_INFLIGHT=1``. It reads vLLM internals directly, which
requires the in-process EngineCore (``VLLM_ENABLE_V1_MULTIPROCESSING=0``, which
NeMo-RL sets by default) so the scheduler's ``running`` queue is reachable from
the worker process. If the scheduler cannot be located it degrades to a no-op
(after logging once) rather than failing generation.
"""

import os
import threading
import time
from typing import Any, Callable, Optional


def inflight_profiling_enabled() -> bool:
    """Whether to sample the in-flight rollout batch (NRL_PROFILE_INFLIGHT)."""
    return os.environ.get("NRL_PROFILE_INFLIGHT", "0").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def inflight_interval_s() -> float:
    """Sampling cadence in seconds (NRL_PROFILE_INFLIGHT_INTERVAL, default 0.5)."""
    raw = os.environ.get("NRL_PROFILE_INFLIGHT_INTERVAL", "0.5")
    try:
        # Floor the interval so a typo cannot spin the sampler thread.
        return max(0.02, float(raw))
    except ValueError:
        return 0.5


def inflight_output_dir() -> str:
    """Directory the driver writes the timeline JSONL to (NRL_PROFILE_INFLIGHT_DIR)."""
    return os.environ.get("NRL_PROFILE_INFLIGHT_DIR", "dp_inflight_profiles")


def find_vllm_scheduler(llm: Any) -> Optional[Any]:
    """Locate the in-process vLLM V1 scheduler from a ``vllm.LLM`` instance.

    Navigates ``LLM -> LLMEngine -> EngineCoreClient -> EngineCore -> scheduler``.
    With multiprocessing disabled the client is an ``InprocClient`` whose
    ``.engine_core`` is the in-process ``EngineCore`` holding the scheduler; the
    extra candidates make this robust to minor vLLM version differences. Returns
    the scheduler (which exposes ``running: list[Request]``) or None if it is not
    reachable (e.g. EngineCore lives in a separate process).
    """
    if llm is None:
        return None
    engine = getattr(llm, "llm_engine", None)
    candidates = []
    if engine is not None:
        engine_core_client = getattr(engine, "engine_core", None)
        if engine_core_client is not None:
            # InprocClient.engine_core is the in-process EngineCore.
            candidates.append(getattr(engine_core_client, "engine_core", None))
            candidates.append(engine_core_client)
        candidates.append(engine)
    for candidate in candidates:
        if candidate is None:
            continue
        scheduler = getattr(candidate, "scheduler", None)
        if scheduler is not None and hasattr(scheduler, "running"):
            return scheduler
    return None


def read_scheduler_sample(scheduler: Any) -> Optional[dict[str, Any]]:
    """Snapshot the running queue of an in-process vLLM scheduler (sync engine).

    Returns ``{batch_size, waiting, ctx_lens, prompt_lens, gen_lens}`` for the
    currently-running requests, or None if the scheduler is unavailable. Used as
    the ``sample_fn`` for the synchronous engine, where the scheduler lives in the
    worker process. (The async engine runs its scheduler out-of-process, so it
    uses a front-end streaming sample source instead — see the async worker.)
    """
    if scheduler is None:
        return None
    # list(...) over a CPython list is atomic under the GIL, so this is safe
    # against the engine thread mutating scheduler.running concurrently.
    try:
        running = list(getattr(scheduler, "running", []))
    except Exception:
        return None

    ctx_lens: list[int] = []
    prompt_lens: list[int] = []
    gen_lens: list[int] = []
    for request in running:
        try:
            # Request.num_tokens == len(all_token_ids): prompt + generated so far.
            ctx_lens.append(int(request.num_tokens))
            prompt_lens.append(int(getattr(request, "num_prompt_tokens", 0)))
            gen_lens.append(int(getattr(request, "num_output_tokens", 0)))
        except Exception:
            continue
    try:
        waiting = len(getattr(scheduler, "waiting", []) or [])
    except Exception:
        waiting = -1
    return {
        "batch_size": len(running),
        "waiting": waiting,
        "ctx_lens": ctx_lens,
        "prompt_lens": prompt_lens,
        "gen_lens": gen_lens,
    }


class InflightProfiler:
    """Background sampler of a vLLM scheduler's running queue for one DP replica.

    The owning worker calls :meth:`mark_call_start` / :meth:`mark_call_end` around
    each ``generate()`` so samples are scoped to a single generation call (relative
    timestamps reset each call), and the driver pulls the buffer with
    :meth:`snapshot` afterwards. A daemon thread does the sampling so it runs while
    the synchronous ``llm.generate()`` blocks the main thread.
    """

    # Hard cap so a pathologically long generation cannot grow the buffer without
    # bound; at the default 0.5s cadence this is far more than any real rollout.
    _MAX_SAMPLES_PER_CALL = 100_000

    def __init__(
        self,
        sample_fn: Callable[[], Optional[dict[str, Any]]],
        dp_label: str,
        interval_s: float,
    ):
        # sample_fn returns {batch_size, waiting, ctx_lens, prompt_lens, gen_lens}
        # for the current in-flight set, or None if the source is unavailable.
        # Sync engine reads the in-process scheduler; async engine reads a live
        # dict maintained from the streamed RequestOutputs (scheduler is remote).
        self._sample_fn = sample_fn
        self._dp_label = dp_label
        self._interval_s = interval_s
        self._lock = threading.Lock()
        self._buffer: list[dict[str, Any]] = []
        self._active = False
        self._t0 = 0.0
        self._thread: Optional[threading.Thread] = None
        self._sample_missing_warned = False

    def start(self) -> None:
        """Launch the daemon sampling thread (idempotent)."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._loop, name="nrl-inflight-profiler", daemon=True
        )
        self._thread.start()

    def mark_call_start(self) -> None:
        """Begin a new generation window: clear the buffer and start sampling."""
        with self._lock:
            self._buffer = []
            self._t0 = time.monotonic()
            self._active = True

    def mark_call_end(self) -> None:
        """Stop sampling; take one final sample to capture the decode tail."""
        self._sample_once()
        with self._lock:
            self._active = False

    def snapshot(self) -> list[dict[str, Any]]:
        """Return a copy of the samples collected for the last generation window."""
        with self._lock:
            return list(self._buffer)

    def _loop(self) -> None:
        while True:
            if self._active:
                self._sample_once()
            time.sleep(self._interval_s)

    def _sample_once(self) -> None:
        try:
            data = self._sample_fn()
        except Exception:
            data = None
        if data is None:
            if not self._sample_missing_warned:
                self._sample_missing_warned = True
                print(
                    f"[INFLIGHT-PROFILER] sample source unavailable for "
                    f"{self._dp_label}; the in-flight timeline will be empty.",
                    flush=True,
                )
            return

        sample = {
            "t": round(time.monotonic() - self._t0, 4),
            "batch_size": data.get("batch_size", 0),
            "waiting": data.get("waiting", -1),
            "ctx_lens": data.get("ctx_lens", []),
            "prompt_lens": data.get("prompt_lens", []),
            "gen_lens": data.get("gen_lens", []),
        }
        with self._lock:
            if self._active and len(self._buffer) < self._MAX_SAMPLES_PER_CALL:
                self._buffer.append(sample)
