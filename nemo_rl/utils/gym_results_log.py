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

"""Append-only JSONL logs of NeMo-Gym rollouts and of what RL did with them.

Two files share ``<log_dir>/gym_results/`` (``async_rl.gym_results_log``):

- ``gym_results.<pid>.jsonl``, written by each NemoGym actor: one line per Gym
  result as it lands, before RL post-processing, or one ``error`` line when the
  rollout failed.
- ``rl_outcomes.<pid>.jsonl``, written wherever RL decides a rollout's fate:
  buffered or evicted (replay buffer, per group), trained or masked (advantage
  stage, per sample).

Lines join on ``rollout_id`` / ``group_id``. Each process writes its own file, so
no two writers share a file handle.
"""

import json
import os
import threading
import time
from collections.abc import Iterable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Optional, TextIO

GYM_RESULTS_SUBDIR = "gym_results"
GYM_RESULTS_PREFIX = "gym_results"
RL_OUTCOMES_PREFIX = "rl_outcomes"


def gym_results_dir(log_dir: str) -> str:
    """Absolute directory both logs are written to for a run's ``log_dir``."""
    return os.path.join(os.path.abspath(log_dir), GYM_RESULTS_SUBDIR)


class JsonlAppender:
    """Append JSON lines to ``<directory>/<prefix>.<pid>.jsonl`` off the caller's thread.

    ``append`` serializes immediately, so later in-place edits to the record (RL
    post-processing pops token arrays from Gym results) cannot change what was
    logged, and hands the write to one background thread so the caller's event
    loop never blocks on file IO. The file is opened on first write, in the
    process that writes it.
    """

    def __init__(self, directory: str, prefix: str) -> None:
        self._directory = directory
        self._prefix = prefix
        self._file: Optional[TextIO] = None
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"{prefix}-writer"
        )
        self._lock = threading.Lock()

    def __getstate__(self) -> dict[str, str]:
        # Setup builds owners of an appender (the replay buffer) in the driver and
        # ships them to the controller actor; the writer thread, its queue and the
        # file handle cannot cross that boundary and belong to the receiving process.
        return {"directory": self._directory, "prefix": self._prefix}

    def __setstate__(self, state: dict[str, str]) -> None:
        self.__init__(state["directory"], state["prefix"])

    @property
    def path(self) -> str:
        return os.path.join(self._directory, f"{self._prefix}.{os.getpid()}.jsonl")

    def append(self, record: Mapping[str, Any]) -> Future:
        """Queue one record; the returned future resolves once it is written."""
        line = json.dumps(record, separators=(",", ":"), default=str) + "\n"
        return self._executor.submit(self._write, line)

    def flush(self) -> None:
        """Block until every queued record is on disk."""
        self._executor.submit(lambda: None).result()

    def close(self) -> None:
        self._executor.shutdown(wait=True)
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None

    def _write(self, line: str) -> None:
        with self._lock:
            if self._file is None:
                os.makedirs(self._directory, exist_ok=True)
                self._file = open(self.path, "a", buffering=1)
            self._file.write(line)


# Top-level keys Gym's enrichment adds that are not a copy of something else.
_ENRICHMENT_KEYS = frozenset({"ng_model_call_capture", "ng_trajectory", "ng_perf"})
# Where a call field that repeats the agent's own result points instead.
REQUEST_INPUT_REF = "=full_result.responses_create_params.input"
TOOLS_REF = "=full_result.responses_create_params.tools"
RESPONSE_OUTPUT_REF = "=full_result.response.output"


def _drop_nulls(value: Any) -> Any:
    """Copy ``value`` without null-valued keys (Pydantic defaults the log need not repeat)."""
    if isinstance(value, Mapping):
        return {k: _drop_nulls(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_drop_nulls(v) for v in value]
    return value


def _normalized_input(items: Any) -> Any:
    """Responses input as the model server receives it: no nulls, message type explicit.

    A message item may omit ``type``, which defaults to ``message``; Gym's request
    models fill it in, so the two forms are the same input.
    """
    items = _drop_nulls(items)
    if not isinstance(items, list):
        return items
    return [
        {"type": "message", **item}
        if isinstance(item, dict) and "role" in item and "type" not in item
        else item
        for item in items
    ]


def compact_gym_result(
    result: Mapping[str, Any], *, added_keys: Iterable[str]
) -> dict[str, Any]:
    """Return ``result`` without what it repeats, for the Gym-results log.

    Every captured model call keeps its entry, metadata and any body that is not
    already in the line. Removed:

    - keys enrichment stamped on the result (``added_keys``, e.g. the rollout
      identity, which the log line carries at top level),
    - ``ng_model_call_capture.calls`` (the same calls as ``ng_trajectory.model_calls``;
      its ``metrics`` and ``gaps`` stay),
    - null-valued fields,
    - a call's ``request.input``, ``request.tools``, ``response.output`` or
      ``response.tools`` that equals the agent's own ``responses_create_params``
      / ``response`` field, replaced by a ``"=full_result...."`` reference.

    Never mutates ``result``: the logged copy shares nested objects with the
    result RL post-processes.
    """
    added = set(added_keys)
    out: dict[str, Any] = {
        k: v for k, v in result.items() if k not in added or k in _ENRICHMENT_KEYS
    }
    rcp = out.get("responses_create_params")
    rcp = rcp if isinstance(rcp, Mapping) else {}
    tools = _drop_nulls(rcp.get("tools")) if rcp.get("tools") is not None else None
    request_input = _normalized_input(rcp.get("input"))
    response = out.get("response")
    output = (
        _drop_nulls(response.get("output")) if isinstance(response, Mapping) else None
    )

    def ref_tools(container: Mapping[str, Any]) -> dict[str, Any]:
        container = dict(container)
        if tools is not None and _drop_nulls(container.get("tools")) == tools:
            container["tools"] = TOOLS_REF
        return container

    if isinstance(response, Mapping):
        out["response"] = ref_tools(response)

    trajectory = out.get("ng_trajectory")
    if isinstance(trajectory, Mapping) and trajectory.get("model_calls"):
        calls = []
        for call in trajectory["model_calls"]:
            call = dict(call)
            request = call.get("request")
            if isinstance(request, Mapping):
                request = ref_tools(request)
                if (
                    rcp.get("input") is not None
                    and _normalized_input(request.get("input")) == request_input
                ):
                    request["input"] = REQUEST_INPUT_REF
                call["request"] = request
            call_response = call.get("response")
            if isinstance(call_response, Mapping):
                call_response = ref_tools(call_response)
                if (
                    output is not None
                    and _drop_nulls(call_response.get("output")) == output
                ):
                    call_response["output"] = RESPONSE_OUTPUT_REF
                call["response"] = call_response
            calls.append(call)
        out["ng_trajectory"] = {**trajectory, "model_calls": calls}

    capture = out.get("ng_model_call_capture")
    if isinstance(capture, Mapping) and "calls" in capture:
        out["ng_model_call_capture"] = {
            k: v for k, v in capture.items() if k != "calls"
        }

    return _drop_nulls(out)


def outcome_record(
    *,
    outcome: str,
    group_id: Optional[str],
    sample_id: Optional[str] = None,
    train_step: Optional[int] = None,
    final_reward: Optional[float] = None,
    weight_version: Optional[int] = None,
    gym_group_id: Optional[str] = None,
    gym_group_attempt: Optional[int] = None,
) -> dict[str, Any]:
    """One ``rl_outcomes`` line: what RL did with a rollout (or a whole group).

    ``group_id`` and ``sample_id`` are TransferQueue ids (``<group_id>_g<i>``).
    ``buffered`` lines also carry the ``gym_group_id``/``gym_group_attempt`` the
    Gym rows were dispatched under, which is how they join ``gym_results``.
    """
    return {
        "outcome": outcome,
        "group_id": group_id,
        "sample_id": sample_id,
        "train_step": train_step,
        "final_reward": final_reward,
        "weight_version": weight_version,
        "gym_group_id": gym_group_id,
        "gym_group_attempt": gym_group_attempt,
        "logged_at": time.time(),
    }
