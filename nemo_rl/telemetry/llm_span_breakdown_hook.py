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
"""NeMo Gym ``llm_span_breakdown_hook`` for NeMo-RL's engines.

With ``NRL_COLLECT_TIMELINE_FIELDS=1`` the vLLM and TRT-LLM servers stamp each response with
epoch-microsecond milestones (``nemo_llm_*``, plus ``nemo_fe_*`` and ``nemo_ctx_*`` under
prefill/decode disaggregation) and the host whose clock stamped each leg (``nemo_*_host``).
:func:`llm_span_breakdown` turns a response into attributes for the Gym model server's
``gym.model.*`` span (token counts, the prefix-cache hit rate, the generation replica) and the
segments between consecutive milestones, which the model server draws as child spans of its
``gym.model.llm_execute`` span::

    policy_model:
      responses_api_models:
        vllm_model:
          llm_span_breakdown_hook: /path/to/nemo_rl/telemetry/llm_span_breakdown_hook.py:llm_span_breakdown

Gym loads this file by path inside its own server process, which does not have ``nemo_rl``, so this
module must import the standard library only.

With ``NRL_CLOCK_CALIBRATION_PATH`` set to the ``clock_calibration.jsonl`` NeMo-RL writes beside its
logs, each engine timestamp is mapped onto the model server's clock before the segments are cut, so
segments that cross hosts line up with their parent. Without it, or for a host it does not list, a
timestamp is used as stamped.
"""

import json
import math
import os
import socket
from time import monotonic
from typing import Any, Callable, Mapping, Optional

CALIBRATION_PATH_ENV = "NRL_CLOCK_CALIBRATION_PATH"

#: Every attribute this hook sets is under this prefix; each segment span carries its class under it.
ATTRIBUTE_PREFIX = "nemo.gym.model."
SEGMENT_CLASS_ATTRIBUTE = ATTRIBUTE_PREFIX + "segment_class"

#: Milestones, in the order a request passes them. ``sent`` and ``received`` are stamped by the
#: model server itself; the model server's own work before ``sent`` and after ``received`` is its
#: spans, not this hook's. The ``fe`` and ``ctx`` legs exist only under prefill/decode disaggregation.
_ENGINE_KEYS = {
    "fe_recv": "nemo_fe_recv_ts_us",
    "ctx_arrival": "nemo_ctx_arrival_ts_us",
    "ctx_queued": "nemo_ctx_queued_ts_us",
    "ctx_first_scheduled": "nemo_ctx_first_scheduled_ts_us",
    "ctx_done": "nemo_ctx_done_ts_us",
    "llm_arrival": "nemo_llm_arrival_ts_us",
    "queued": "nemo_llm_queued_ts_us",
    "first_scheduled": "nemo_llm_first_scheduled_ts_us",
    "first_token": "nemo_llm_first_token_ts_us",
    "last_token": "nemo_llm_last_token_ts_us",
    "llm_response_ready": "nemo_llm_response_ready_ts_us",
}

#: (name, start milestone, end milestone, class). ``compute`` is engine work, ``queue`` is waiting on
#: an engine scheduler, ``plumbing`` is everything else. ``proxy_to_engine`` stands in for the
#: frontend and context legs when there is no disaggregation. Only the engine's side is here, from
#: ``sent`` to ``received``; the model server draws its own work before and after.
_SEGMENTS = (
    ("proxy_to_frontend", "sent", "fe_recv", "plumbing"),
    ("frontend_route", "fe_recv", "ctx_arrival", "plumbing"),
    ("ctx_submit", "ctx_arrival", "ctx_queued", "plumbing"),
    ("ctx_sched_queue", "ctx_queued", "ctx_first_scheduled", "queue"),
    ("ctx_prefill", "ctx_first_scheduled", "ctx_done", "compute"),
    ("kv_handoff", "ctx_done", "llm_arrival", "plumbing"),
    ("proxy_to_engine", "sent", "llm_arrival", "plumbing"),
    ("gen_submit", "llm_arrival", "queued", "plumbing"),
    ("gen_sched_queue", "queued", "first_scheduled", "queue"),
    ("gen_first_token", "first_scheduled", "first_token", "compute"),
    ("decode", "first_token", "last_token", "compute"),
    ("response_build", "last_token", "llm_response_ready", "plumbing"),
    ("llm_to_proxy", "llm_response_ready", "received", "plumbing"),
)

#: The response field naming the host whose clock stamped each engine milestone: the ``nemo_llm``
#: in ``nemo_llm_first_token_ts_us`` is stamped by the host in ``nemo_llm_host``.
_HOST_KEYS = {
    key: "_".join(key.split("_")[:2]) + "_host" for key in _ENGINE_KEYS.values()
}

#: The id NeMo-RL gives this host's clock in ``clock_calibration.jsonl``: its host name.
_LOCAL_HOST = socket.gethostname()

#: A calibration read that found nothing is retried after this long, in case the file is not written yet.
_CALIBRATION_RETRY_S = 30.0

#: path -> (monotonic time of the read, {host: offset_us}).
_calibration_cache: dict = {}


def _is_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _read_clock_offsets(path: str) -> dict:
    """``{host: offset_us}`` from a ``clock_calibration.jsonl``, empty if it is missing or unusable.

    An offset is the host's clock minus the reference clock, so a host timestamp maps onto the
    reference as ``ts - offset``. The reference host's own offset is 0.
    """
    try:
        with open(path) as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    except (OSError, ValueError):
        return {}
    rows = [row for row in rows if isinstance(row, dict)]
    references = {
        row["reference_clock_id"] for row in rows if row.get("reference_clock_id")
    }
    if len(references) != 1:
        return {}
    offsets = {
        row["host_clock_id"]: float(row["offset_us"])
        for row in rows
        if row.get("host_clock_id")
        and isinstance(row.get("offset_us"), (int, float))
        and math.isfinite(row["offset_us"])
    }
    offsets[references.pop()] = 0.0
    return offsets


def load_clock_offsets(path: str) -> Mapping[str, float]:
    """:func:`_read_clock_offsets`, read once per path; an empty result is retried after a while."""
    cached = _calibration_cache.get(path)
    now = monotonic()
    if cached is not None and (cached[1] or now - cached[0] < _CALIBRATION_RETRY_S):
        return cached[1]
    offsets = _read_clock_offsets(path)
    _calibration_cache[path] = (now, offsets)
    return offsets


def _milestones(
    engine_response: Mapping[str, Any],
    model_server: Mapping[str, Optional[int]],
    offsets: Optional[Mapping[str, float]],
) -> tuple:
    """Every milestone the response carries, plus the model server's own, on this server's clock.

    With *offsets*, an engine milestone is moved from the clock of the host that stamped it onto this
    server's. A milestone whose host is unknown, or that does not name one, is left as stamped.

    Returns:
        The milestones, and whether every engine milestone could be mapped (``None`` without *offsets*).
    """
    milestones = {
        name: value for name, value in model_server.items() if value is not None
    }
    local = None if offsets is None else offsets.get(_LOCAL_HOST)
    mapped = offsets is not None and local is not None
    for name, key in _ENGINE_KEYS.items():
        value = engine_response.get(key)
        if not _is_count(value):
            continue
        if offsets is not None:
            offset = offsets.get(engine_response.get(_HOST_KEYS[key]))
            if local is not None and offset is not None:
                value += round(local - offset)
            else:
                mapped = False
        milestones[name] = value
    return milestones, (mapped if offsets is not None else None)


def _usage_attributes(engine_response: Mapping[str, Any]) -> dict:
    """Token counts and prefix-cache reuse from the response ``usage`` block.

    A hit rate of ``0`` is a real miss; the keys are left out when the engine did not report
    the numbers, so a missing value is never confused with a miss.
    """
    attributes: dict = {}
    usage = engine_response.get("usage")
    if not isinstance(usage, Mapping):
        return attributes
    prompt_tokens = usage.get("prompt_tokens")
    if _is_count(prompt_tokens):
        attributes["prompt_tokens"] = prompt_tokens
    completion_tokens = usage.get("completion_tokens")
    if _is_count(completion_tokens):
        attributes["decode_tokens"] = completion_tokens
    details = usage.get("prompt_tokens_details")
    cached = details.get("cached_tokens") if isinstance(details, Mapping) else None
    if _is_count(prompt_tokens) and _is_count(cached) and cached <= prompt_tokens:
        attributes["cached_prompt_tokens"] = cached
        attributes["kv_cache_hit_rate"] = (
            cached / prompt_tokens if prompt_tokens else 0.0
        )
    return attributes


def _breakdown(
    engine_response: Mapping[str, Any],
    *,
    sent_us: int,
    received_us: int,
    replica_idx: Optional[int] = None,
) -> dict:
    """The attributes and segments for one engine call, with segment times on the model server's clock.

    Without disaggregation there is no ``fe``/``ctx`` leg, so ``proxy_to_engine`` is used and
    ``proxy_to_frontend`` is not; with it, the reverse. A segment with a missing end is left out.
    """
    path = os.environ.get(CALIBRATION_PATH_ENV)
    offsets = load_clock_offsets(path) if path else None
    model_server = {
        "sent": sent_us,
        "received": received_us,
    }
    milestones, mapped = _milestones(engine_response, model_server, offsets)

    attributes = _usage_attributes(engine_response)
    if _is_count(replica_idx):
        attributes["generation_replica_idx"] = replica_idx
    if mapped is not None:
        attributes["clock_corrected"] = mapped
    # Under disaggregation the prefill leg's reuse is the one that matters, not the decode leg's usage.
    for key in (
        "nemo_ctx_cached_tokens",
        "nemo_ctx_computed_tokens",
        "nemo_ctx_num_chunks",
    ):
        value = engine_response.get(key)
        if _is_count(value):
            attributes[key.removeprefix("nemo_")] = value

    disaggregated = "fe_recv" in milestones
    segments = [
        (name, milestones[start], milestones[end], segment_class)
        for name, start, end, segment_class in _SEGMENTS
        if start in milestones
        and end in milestones
        and not (name == "proxy_to_engine" and disaggregated)
    ]
    return {"attributes": attributes, "segments": segments}


def llm_span_breakdown(
    engine_response: Mapping[str, Any],
    *,
    span: Any,
    emit_span: Callable[..., None],
    sent_us: int,
    received_us: int,
    replica_idx: Optional[int] = None,
) -> None:
    """The Gym ``llm_span_breakdown_hook`` entry point: break down one engine call.

    Sets the attributes on *span*, and draws one span per segment with ``emit_span``, nested under ``llm_execute``. A
    segment that ends before it starts compares clocks of different hosts that disagree; it is
    not drawn and is counted in ``nemo.gym.model.timeline_skew_segments``.
    """
    result = _breakdown(
        engine_response,
        sent_us=sent_us,
        received_us=received_us,
        replica_idx=replica_idx,
    )
    attributes = {
        ATTRIBUTE_PREFIX + key: value for key, value in result["attributes"].items()
    }
    skewed = 0
    for name, start_us, end_us, segment_class in result["segments"]:
        if end_us < start_us:
            skewed += 1
        else:
            emit_span(name, start_us, end_us, {SEGMENT_CLASS_ATTRIBUTE: segment_class})
    attributes[ATTRIBUTE_PREFIX + "timeline_skew_segments"] = skewed
    span.set_attributes(attributes)
