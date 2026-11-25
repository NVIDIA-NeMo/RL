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
"""Turn-judge coordinator: the turn judge as a group-level verification step (``grpo.turn_credit``).

The judge is part of a group's verification: after the group seals (every rollout verified), the group's
dispatch coroutine awaits ``judge_sealed_group`` and only then finalizes and commits the group as ready. Nothing
waits at the advantage stage, the sampler is unchanged, and a committed group's labels are already known.

1. ``observe_completion`` (rollout manager's per-rollout completion callback): keep the small fields the judge
   needs (rewards, test status and log, final patch, the path of the Gym ``generation.json`` holding the
   trajectory, the call count of ``response.output``); never the trajectory itself.
2. ``judge_sealed_group``: groups with an outcome contrast are POSTed to the turn-judge service (a standalone
   HTTP service running the offline judge; request id = group id + rollout fingerprint, so re-sending is
   idempotent) and polled until a result or ``judge.budget_s`` from the seal; anything else (uniform group, agent
   not judged, missing payload, breaker open, timeout, transport error) returns at once or at the deadline and the
   group trains as plain GRPO.
3. ``labels_for``: the advantage stage's labels, by canonical sample id.
4. Restart: every full checkpoint carries ``state_dict()`` (taken under the checkpoint barrier, with the replay
   metadata and the recovery ledger): the results of its replay and in-flight groups and the stash of its in-flight
   groups, whose sealed rollouts a restart reuses without observing them again. ``resume`` loads it. A judgment
   still running when the job ended is re-sent and judged again. Optional (``state_dir``): an append-only event log,
   one file per job, folded at start instead of the checkpoint file.
"""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import hashlib
import json
import logging
import os
import queue
import re
import threading
import time
import zlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from nemo_rl.algorithms.single_controller_utils.turn_credit import (
    GOOD,
    HARMFUL,
    RolloutTurnLabels,
    TurnCreditConfig,
    TurnJudgeClientConfig,
    call_starts_of,
)
from nemo_rl.experience.sample_ids import try_parse_sample_id

log = logging.getLogger(__name__)

# Text fields stored zlib-compressed in the event log.
_PACKED_FIELDS = ("test_output", "model_patch", "response_output")
_MARGIN_S = 60.0
# Statuses of the service's result that carry labels.
_LABEL_STATUSES = ("ok", "partial")


def _pack(value: Any) -> Any:
    if value is None:
        return None
    raw = json.dumps(value, ensure_ascii=False).encode()
    return {"z": base64.b64encode(zlib.compress(raw, 6)).decode("ascii")}


def _unpack(value: Any) -> Any:
    if isinstance(value, dict) and set(value) == {"z"}:
        return json.loads(zlib.decompress(base64.b64decode(value["z"])))
    return value


def _packed(stash: dict[str, Any]) -> dict[str, Any]:
    """The stash as kept in memory and in the restart state: big text fields compressed once."""
    return {k: (_pack(v) if k in _PACKED_FIELDS else v) for k, v in stash.items()}


def rollout_fingerprint(rollout_ids: Sequence[str]) -> str:
    """Identity of the exact rollouts a judgment covers (a regenerated sibling gets a new gate rollout id)."""
    return hashlib.sha256("\n".join(rollout_ids).encode()).hexdigest()[:16]


def stash_from_env_extras(
    env_extras: Mapping[str, Any], reward: float
) -> dict[str, Any]:
    """The judge's per-rollout fields from a completion's ``env_extras`` (the Gym ``/run`` result)."""
    response = env_extras.get("response") or {}
    output = response.get("output") if isinstance(response, Mapping) else None
    output = output if isinstance(output, list) else []
    instance_config = env_extras.get("instance_config") or {}
    env_reward = env_extras.get("reward")
    fpath = env_extras.get("opencode_results_fpath") or None
    return {
        "gate_rollout_id": env_extras.get("ng_rollout_id"),
        # The environment's outcome (the judge's PASS / FAIL); the trained reward may carry penalties.
        "reward": float(env_reward)
        if isinstance(env_reward, (int, float))
        else float(reward),
        "mask_sample": bool(instance_config.get("mask_sample", False))
        if isinstance(instance_config, Mapping)
        else False,
        "context_limited": bool(env_extras.get("opencode_context_limited")),
        "cap_killed": "signal: killed"
        in str(env_extras.get("opencode_run_stderr") or ""),
        "evaluation_completed": env_extras.get("evaluation_completed"),
        "resolved": env_extras.get("resolved"),
        "test_results": env_extras.get("test_results"),
        "test_output": env_extras.get("test_output"),
        "test_return_code": env_extras.get("test_return_code"),
        "model_patch": env_extras.get("model_patch"),
        "patch_applied": env_extras.get("patch_applied"),
        "opencode_results_fpath": fpath,
        "n_calls": len(call_starts_of(output)),
        # Only when Gym wrote no generation.json path: the service then needs the trajectory inline.
        "response_output": output if not fpath else None,
    }


# ------------------------------------------------------------------------------------------------ event log


class EventLog:
    """Append-only JSONL file written by one background thread (no Lustre latency on the event loop).

    ``append(event, durable=True)`` returns a future that resolves once the line is flushed and fsynced.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._queue: queue.Queue = queue.Queue()
        self.bytes_written = 0
        self.write_errors = 0
        self._thread = threading.Thread(
            target=self._run, name="turn-judge-event-log", daemon=True
        )
        self._thread.start()

    def append(
        self, event: dict[str, Any], durable: bool = False
    ) -> concurrent.futures.Future:
        future: concurrent.futures.Future = concurrent.futures.Future()
        self._queue.put(
            (json.dumps(event, ensure_ascii=False, default=str) + "\n", durable, future)
        )
        return future

    def close(self, timeout: float = 30.0) -> None:
        self._queue.put(None)
        self._thread.join(timeout)

    def _run(self) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            while True:
                item = self._queue.get()
                if item is None:
                    return
                # Write everything queued so far, then one flush (and one fsync if any line asked for it).
                batch, stop = [item], False
                while not self._queue.empty():
                    nxt = self._queue.get_nowait()
                    if nxt is None:
                        stop = True
                        break
                    batch.append(nxt)
                try:
                    for line, _, _ in batch:
                        f.write(line)
                    f.flush()
                    if any(durable for _, durable, _ in batch):
                        os.fsync(f.fileno())
                    self.bytes_written += sum(len(line) for line, _, _ in batch)
                    ok = True
                except Exception as error:  # the log must never fail training
                    self.write_errors += 1
                    log.warning("turn-judge event log write failed: %s", error)
                    ok = False
                for _, _, future in batch:
                    future.set_result(ok)
                if stop:
                    return


EVENT_FILE = re.compile(r"^events\.(.+)\.(\d+)\.jsonl$")


def read_events(state_dir: str) -> list[dict[str, Any]]:
    """Events from the newest file holding a complete snapshot onwards (older files are superseded by it).

    Files are ``events.<job id>.<unix start time>.jsonl``, ordered by start time.
    """
    try:
        names = os.listdir(state_dir)
    except FileNotFoundError:
        return []
    found = [(int(m.group(2)), n) for n in names if (m := EVENT_FILE.match(n))]
    paths = [os.path.join(state_dir, n) for _, n in sorted(found)]
    # Newest first, stopping at the first file whose snapshot completed: only that file and later ones are read.
    selected: list[list[dict[str, Any]]] = []
    for path in reversed(paths):
        events = []
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # a torn last line of a killed job
        selected.append(events)
        if any(e.get("e") == "snapshot_done" for e in events):
            break
    return [e for events in reversed(selected) for e in events]


# ------------------------------------------------------------------------------------------------ service client


class TurnJudgeClient:
    """POST / poll / DELETE against the turn-judge service; the address comes from an endpoint file."""

    def __init__(
        self, config: TurnJudgeClientConfig, *, clock: Callable[[], float] = time.time
    ) -> None:
        self._cfg = config
        self._clock = clock
        self._url: Optional[str] = None
        self._session: Any = None
        self.consecutive_failures = 0
        self.breaker_open_until = 0.0

    def breaker_open(self) -> bool:
        return self._clock() < self.breaker_open_until

    def _base_url(self) -> str:
        if self._url is None:
            with open(self._cfg.endpoint_file, encoding="utf-8") as f:
                env = dict(re.findall(r"export (\w+)=(\S+)", f.read()))
            self._url = env["TURN_JUDGE_URL"].rstrip("/")
        return self._url

    async def _request(
        self, method: str, path: str, body: Optional[dict] = None
    ) -> tuple[int, Any]:
        import aiohttp

        if self._session is None or self._session.closed:
            # No keep-alive: a pooled connection the service already closed as idle fails the next request with
            # "connection reset" / "broken pipe" (seen 3 times in run 20261115-134722). A few requests per second
            # need no connection reuse.
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._cfg.request_timeout_s),
                connector=aiohttp.TCPConnector(force_close=True),
            )
        async with self._session.request(
            method, self._base_url() + path, json=body
        ) as resp:
            data = await resp.json(content_type=None) if resp.status != 204 else None
            return resp.status, data

    def _failed(self, error: BaseException, *, count: bool = True) -> None:
        self._url = None  # re-read the endpoint file: the service may have moved
        if count:
            self.consecutive_failures += 1
            if self.consecutive_failures >= self._cfg.breaker_failures:
                self.breaker_open_until = self._clock() + self._cfg.breaker_cooloff_s
                self.consecutive_failures = 0
        log.warning("turn-judge service request failed: %r", error)

    async def judge(
        self, payload: dict[str, Any], *, deadline: float
    ) -> tuple[str, Optional[dict]]:
        """Run one judgment to its result: ``("done", result)``, ``("timeout", None)`` or ``("transport_error", None)``.

        A missing endpoint file means the service is not up yet (a job's first minutes): wait for it until the
        deadline without counting a failure.
        """
        request_id = payload["request_id"]
        submitted = False
        while True:
            if self._url is None and not os.path.exists(self._cfg.endpoint_file):
                remaining = deadline - self._clock()
                if remaining <= 0:
                    return "no_service", None
                await asyncio.sleep(min(self._cfg.poll_interval_s, remaining))
                continue
            try:
                if not submitted:
                    status, body = await self._request("POST", "/judge_group", payload)
                else:
                    status, body = await self._request(
                        "GET", f"/judge_group/{request_id}"
                    )
                if status == 404 and submitted:
                    submitted = False  # the service restarted and lost it: send again
                elif 400 <= status < 500 and status != 404:
                    # The service rejected the request itself (bad payload): retrying cannot help.
                    log.warning(
                        "turn-judge service rejected %s: HTTP %s %s",
                        request_id,
                        status,
                        str(body)[:300],
                    )
                    return "rejected", None
                elif status != 200:
                    raise RuntimeError(f"HTTP {status}: {str(body)[:300]}")
                else:
                    submitted = True
                    self.consecutive_failures = 0
                    if body.get("state") == "done":
                        return "done", body.get("result") or {}
            except asyncio.CancelledError:
                raise
            except Exception as error:
                # Only a failed submission is evidence that the service is down. A slow poll of a request it already
                # accepted is not (its event loop can stall for a minute; LOG 096): it never opens the breaker, and
                # the request keeps polling until its deadline.
                self._failed(error, count=not submitted)
                if not submitted and self.breaker_open():
                    return "transport_error", None
            remaining = deadline - self._clock()
            if remaining <= 0:
                break
            await asyncio.sleep(min(self._cfg.poll_interval_s, remaining))
        if submitted:
            try:
                await self._request("DELETE", f"/judge_group/{request_id}")
            except (
                Exception
            ) as error:  # best effort: the service also stops at its own deadline
                log.warning("turn-judge cancel of %s failed: %r", request_id, error)
        return ("timeout", None) if submitted else ("transport_error", None)

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()


# ------------------------------------------------------------------------------------------------ coordinator


@dataclass
class _GroupResult:
    fingerprint: str
    status: str
    labels: dict[
        str, RolloutTurnLabels
    ]  # canonical sample id -> labels (rollouts with >= 1 label)


@dataclass
class _StepCounters:
    counts: dict[str, float] = field(default_factory=dict)
    latencies: list[float] = field(default_factory=list)

    def add(self, key: str, value: float = 1.0) -> None:
        self.counts[key] = self.counts.get(key, 0.0) + value


def _result_to_json(result: "_GroupResult") -> dict[str, Any]:
    return {
        "fp": result.fingerprint,
        "status": result.status,
        # [generation index, n_calls, [[call, j], ...], trailing empty calls, aligned (None: not checked),
        #  per-call output tokens (None: not given)]
        "labels": {
            sid: [
                lab.generation_index,
                lab.n_calls,
                [list(x) for x in lab.labels],
                lab.trailing_empty_calls,
                lab.aligned,
                None if lab.step_tokens is None else list(lab.step_tokens),
            ]
            for sid, lab in result.labels.items()
        },
    }


def _result_from_json(gid: str, raw: Mapping[str, Any]) -> "_GroupResult":
    return _GroupResult(
        fingerprint=raw["fp"],
        status=raw["status"],
        labels={
            sid: RolloutTurnLabels(
                group_id=gid,
                generation_index=int(v[0]),
                n_calls=int(v[1]),
                labels=tuple((int(k), int(j)) for k, j in v[2]),
                trailing_empty_calls=int(v[3]) if len(v) > 3 else 0,
                aligned=v[4] if len(v) > 4 else None,
                step_tokens=(
                    tuple(None if t is None else int(t) for t in v[5])
                    if len(v) > 5 and v[5] is not None
                    else None
                ),
            )
            for sid, v in raw["labels"].items()
        },
    )


class TurnJudgeCoordinator:
    """Stash -> judge at seal -> labels at the advantage stage (see the module docstring)."""

    def __init__(
        self,
        config: TurnCreditConfig,
        *,
        state_dir: Optional[str],
        job_id: Optional[str] = None,
        client: Optional[TurnJudgeClient] = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        assert config.judge is not None
        self._cfg = config
        self._judge_cfg = config.judge
        self._clock = clock
        self._client = client or TurnJudgeClient(config.judge, clock=clock)
        self._state_dir = state_dir
        self._job_id = job_id or os.environ.get("SLURM_JOB_ID", "nojob")
        self._log: Optional[EventLog] = None
        # (group id) -> {generation index -> stash}
        self._stash: dict[str, dict[int, dict[str, Any]]] = {}
        self._results: dict[str, _GroupResult] = {}
        self._consumed_at: dict[str, int] = {}
        self._in_flight = 0
        self._step = _StepCounters()
        self._resumed = False

    @property
    def config(self) -> TurnCreditConfig:
        """Return the validated configuration used for judging and shaping."""
        return self._cfg

    # ---------------------------------------------------------------- restart

    def state_dict(
        self,
        replay_group_ids: Iterable[Optional[str]],
        inflight_group_ids: Iterable[str],
    ) -> dict[str, Any]:
        """Restart state for a full checkpoint (called under its barrier; plain data, copied, no I/O).

        Results of the checkpoint's replay groups and in-flight groups; stash of its in-flight groups.
        """
        replay = {g for g in replay_group_ids if g}
        inflight = set(inflight_group_ids)
        return {
            "version": 1,
            "results": {
                gid: _result_to_json(result)
                for gid, result in self._results.items()
                if gid in replay or gid in inflight
            },
            "stash": {
                gid: dict[int, dict[str, Any]](per_gen)
                for gid, per_gen in self._stash.items()
                if gid in inflight
            },
        }

    def resume(
        self,
        replay_group_ids: Iterable[Optional[str]],
        inflight_group_ids: Iterable[str],
        checkpoint_state: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Load the restored run's judge state; called once before the rollout and train pumps start.

        ``checkpoint_state`` (the restored checkpoint's ``state_dict``) is used when given; else, with a
        ``state_dir``, the event log is folded. Only the restored groups are kept: replay groups (committed: their
        labels) and in-flight recovery-ledger groups (their rollouts' stash for the sealed rollouts the restart
        reuses, and a result, reused if the rollouts are unchanged).
        """
        replay = {g for g in replay_group_ids if g}
        inflight = set(inflight_group_ids)
        live = replay | inflight
        source = "none"
        if checkpoint_state is not None:
            source = "checkpoint"
            for gid, raw in (checkpoint_state.get("results") or {}).items():
                if gid in live:
                    self._results[gid] = _result_from_json(gid, raw)
            for gid, per_gen in (checkpoint_state.get("stash") or {}).items():
                if gid in inflight:
                    self._stash[gid] = {
                        int(gen): stash for gen, stash in per_gen.items()
                    }
        elif self._state_dir is not None:
            source = "event log"
            for event in read_events(self._state_dir):
                kind, gid = event.get("e"), event.get("gid")
                if not isinstance(gid, str) or gid not in live:
                    continue
                if kind == "stash" and gid in inflight:
                    self._stash.setdefault(gid, {})[int(event["gen"])] = event["d"]
                elif kind == "result":
                    self._results[gid] = _result_from_json(gid, event)
        restored_results = len(self._results)
        restored_stash = sum(len(v) for v in self._stash.values())
        if self._state_dir is not None:
            path = os.path.join(
                self._state_dir, f"events.{self._job_id}.{int(self._clock())}.jsonl"
            )
            self._log = EventLog(path)
            self._log.append(
                {"e": "snapshot_start", "t": self._clock(), "job": self._job_id}
            )
            for gid, per_gen in self._stash.items():
                for gen, stash in per_gen.items():
                    self._append_stash(gid, gen, stash)
            for gid, result in self._results.items():
                self._append_result(gid, result, latency=None, info=None)
            assert self._log is not None
            self._log.append({"e": "snapshot_done", "t": self._clock()})
        self._step.add("state_restored_results", restored_results)
        self._step.add("state_restored_stash", restored_stash)
        self._resumed = True
        print(
            f"turn_judge: resumed {restored_results} group result(s) and {restored_stash} rollout stash(es) "
            f"for {len(live)} restored group(s) from {source}; event log {self._log.path if self._log else '(off)'}",
            flush=True,
        )

    def _append_stash(self, gid: str, gen: int, stash: dict[str, Any]) -> None:
        if self._log is None:
            return
        self._log.append({"e": "stash", "gid": gid, "gen": gen, "d": stash})

    def _append_result(
        self,
        gid: str,
        result: _GroupResult,
        *,
        latency: Optional[float],
        info: Optional[dict],
    ) -> Optional[concurrent.futures.Future]:
        if self._log is None:
            return None
        return self._log.append(
            {
                "e": "result",
                "gid": gid,
                "t": self._clock(),
                "latency": latency,
                "info": info,
                **_result_to_json(result),
            },
            durable=True,
        )

    # ---------------------------------------------------------------- stash

    def observe_completion(
        self,
        group_id: str,
        generation_index: int,
        env_extras: Mapping[str, Any],
        reward: float,
    ) -> None:
        """Keep one rollout's judge fields (rollout manager callback; never raises)."""
        try:
            stash = _packed(stash_from_env_extras(env_extras, reward))
            self._stash.setdefault(group_id, {})[int(generation_index)] = stash
            self._append_stash(group_id, int(generation_index), stash)
        except Exception as error:
            self._step.add("observer_errors")
            log.warning(
                "turn_judge observer failed for %s/%s: %r",
                group_id,
                generation_index,
                error,
            )

    # ---------------------------------------------------------------- judge

    async def judge_sealed_group(
        self, request: Any, prompt: Mapping[str, Any], agent_name: str
    ) -> str:
        """Judge one sealed group before it is finalized; returns the outcome (for logs and tests).

        ``request`` is the group's ReassemblyRequest (group id, gate rollout ids, canonical sample ids, rewards).
        """
        gid = request.group_id
        fingerprint = rollout_fingerprint(request.rollout_ids)
        self._step.add("sealed_groups")
        previous = self._results.get(gid)
        if previous is not None and previous.fingerprint == fingerprint:
            # Judged before a restart (or before a failed finalize): reuse.
            self._step.add("status/reused")
            return "reused"
        self._results.pop(gid, None)
        agents = self._judge_cfg.agents
        if agents is not None and agent_name not in agents:
            return self._skip(gid, "agent")
        if len({float(r) for r in request.rewards}) < 2:
            return self._skip(gid, "uniform")
        stash = self._stash.get(gid, {})
        rollouts = []
        for i, rollout_id in enumerate(request.rollout_ids):
            s = stash.get(i)
            if s is None or s.get("gate_rollout_id") != rollout_id:
                return self._skip(gid, "incomplete_payload")
            rollouts.append(
                {
                    "generation_index": i,
                    **{k: _unpack(v) for k, v in s.items() if k != "gate_rollout_id"},
                }
            )
        if self._client.breaker_open():
            return self._skip(gid, "breaker")
        sealed_at = self._clock()
        deadline = sealed_at + self._judge_cfg.budget_s
        task = {
            k: v
            for k, v in dict(prompt.get("extra_env_info") or {}).items()
            if not str(k).startswith("_ng")
        }
        payload = {
            "request_id": f"{gid}:{fingerprint}",
            "group_id": gid,
            "agent_name": agent_name,
            "deadline_unix": deadline - _MARGIN_S,
            "stage1_timeout_s": self._judge_cfg.stage1_timeout_s,
            "pair_timeout_s": self._judge_cfg.pair_timeout_s,
            "overrides": dict(self._judge_cfg.request_overrides),
            "task": json.loads(json.dumps(task, default=str)),
            "rollouts": rollouts,
        }
        self._step.add("judged_groups")
        self._in_flight += 1
        try:
            outcome, result = await self._client.judge(payload, deadline=deadline)
        finally:
            self._in_flight -= 1
        latency = self._clock() - sealed_at
        self._step.latencies.append(latency)
        if outcome != "done" or result is None:
            self._step.add(f"status/{outcome}")
            return outcome
        status = str(result.get("status") or "error")
        self._step.add(f"status/{status}")
        raw_info = result.get("info")
        info: dict[str, Any] = raw_info if isinstance(raw_info, dict) else {}
        for key in (
            "pairs_total",
            "pairs_ok",
            "pairs_capped",
            "pairs_failed",
            "sessions",
        ):
            if isinstance(info.get(key), (int, float)):
                self._step.add(f"service/{key}", float(info[key]))
        labels: dict[str, RolloutTurnLabels] = {}
        if status in _LABEL_STATUSES:
            labels = self._convert_labels(
                gid, request.canonical_sample_ids, result.get("labels") or {}, stash
            )
        group_result = _GroupResult(
            fingerprint=fingerprint, status=status, labels=labels
        )
        self._results[gid] = group_result
        written = self._append_result(gid, group_result, latency=latency, info=info)
        if written is not None:
            # Durable before the commit: a committed group's labels must survive a restart.
            try:
                ok = await asyncio.wait_for(asyncio.wrap_future(written), timeout=30.0)
                if not ok:
                    self._step.add("state_write_errors")
            except asyncio.TimeoutError:
                self._step.add("state_write_timeouts")
        return status

    def _skip(self, gid: str, reason: str) -> str:
        self._step.add(f"skipped/{reason}")
        return f"skipped:{reason}"

    def _convert_labels(
        self,
        gid: str,
        sample_ids: Sequence[str],
        raw: Mapping[str, Any],
        stash: Mapping[int, Mapping[str, Any]],
    ) -> dict[str, RolloutTurnLabels]:
        out: dict[str, RolloutTurnLabels] = {}
        for gen_key, entry in raw.items():
            try:
                i = int(gen_key)
                n_calls = int(entry["n_calls"])
            except (TypeError, ValueError, KeyError):
                self._step.add("labels/bad_entry")
                continue
            if not 0 <= i < len(sample_ids):
                self._step.add("labels/bad_entry")
                continue
            if n_calls != int(stash.get(i, {}).get("n_calls", -1)):
                # The service read another trajectory than the trainer saw: never apply these labels.
                self._step.add("labels/n_calls_mismatch")
                continue
            # The service's proof that its calls are OpenCode's model calls one to one (absent: an older service).
            aligned: Optional[bool] = None
            trailing = 0
            if "aligned" in entry:
                aligned = entry.get("aligned") is True
                if not aligned:
                    self._step.add("labels/alignment_not_proven")
                    continue
                try:
                    trailing = max(0, int(entry.get("trailing_empty_calls") or 0))
                except (TypeError, ValueError):
                    trailing = 0
                if trailing:
                    self._step.add("labels/trailing_empty_rollouts")
            per_call: dict[int, int] = {}
            conflicts: set[int] = set()
            for lab in entry.get("labels") or []:
                value = {"good": GOOD, "harmful": HARMFUL}.get(str(lab.get("label")))
                k = lab.get("call")
                if value is None or not isinstance(k, int) or not 0 <= k < n_calls:
                    self._step.add("labels/bad_label")
                    continue
                if per_call.get(k, value) != value:
                    conflicts.add(k)
                per_call[k] = value
            for k in conflicts:
                per_call.pop(k, None)
                self._step.add("labels/conflicting_call")
            if not per_call:
                continue
            parsed = try_parse_sample_id(sample_ids[i])
            if parsed is not None and parsed[0] != gid:
                self._step.add("labels/bad_entry")
                continue
            # Per-call generated tokens of the export the judge read (None entry: a step that never finished), for
            # the advantage stage's identity check (anything malformed -> None: that check then applies no label).
            raw_tokens = entry.get("step_output_tokens")
            step_tokens: Optional[tuple[Optional[int], ...]] = None
            if isinstance(raw_tokens, list) and all(
                t is None or (isinstance(t, int) and not isinstance(t, bool) and t >= 0)
                for t in raw_tokens
            ):
                step_tokens = tuple(raw_tokens)
            else:
                self._step.add("labels/no_step_tokens")
            out[sample_ids[i]] = RolloutTurnLabels(
                group_id=gid,
                generation_index=i,
                n_calls=n_calls,
                labels=tuple(sorted(per_call.items())),
                aligned=aligned,
                trailing_empty_calls=trailing,
                step_tokens=step_tokens,
            )
            self._step.add("labels/rollouts")
            self._step.add(
                "labels/good_calls", sum(1 for v in per_call.values() if v == GOOD)
            )
            self._step.add(
                "labels/harmful_calls",
                sum(1 for v in per_call.values() if v == HARMFUL),
            )
        return out

    def discard_group(self, group_id: str) -> None:
        """The group will never train (dropped, below threshold, cancelled): forget its stash and labels."""
        self._stash.pop(group_id, None)
        self._results.pop(group_id, None)

    def committed(self, group_id: str) -> None:
        """The group is in the replay buffer: its stash is no longer needed in memory."""
        self._stash.pop(group_id, None)

    # ---------------------------------------------------------------- advantage stage

    def labels_for(
        self, sample_ids: Sequence[str], step: int
    ) -> dict[str, RolloutTurnLabels]:
        """Labels of the batch's rows (canonical sample ids); records the groups as consumed at ``step``."""
        out: dict[str, RolloutTurnLabels] = {}
        groups: set[str] = set()
        for sid in sample_ids:
            parsed = try_parse_sample_id(str(sid))
            gid = parsed[0] if parsed is not None else str(sid)
            groups.add(gid)
            result = self._results.get(gid)
            if result is not None and sid in result.labels:
                out[sid] = result.labels[sid]
        for gid in groups:
            result = self._results.get(gid)
            self._step.add("batch/groups")
            if result is not None:
                self._step.add(f"batch/groups_{result.status}")
                if result.labels:
                    self._step.add("batch/groups_labeled")
            self._consumed_at.setdefault(gid, step)
        # Keep consumed groups for two steps (a retried step re-reads them), then drop them from memory; a
        # restart re-reads them from the log if the checkpoint restores them.
        for gid, at in list(self._consumed_at.items()):
            if at < step - 2:
                self._results.pop(gid, None)
                self._stash.pop(gid, None)
                del self._consumed_at[gid]
        return out

    def record_shaping(self, step: int, records: list[dict[str, Any]]) -> None:
        """Side log: the advantage stage's per-rollout records (A, lambda_eff, labeled calls)."""
        if self._log is None or not self._cfg.side_log or not records:
            return
        self._log.append(
            {"e": "shaped", "step": step, "t": self._clock(), "rows": records}
        )

    # ---------------------------------------------------------------- metrics

    def pop_step_metrics(self) -> dict[str, float]:
        step, self._step = self._step, _StepCounters()
        m = {f"turn_judge/{k}": v for k, v in step.counts.items()}
        m["turn_judge/in_flight_groups"] = float(self._in_flight)
        m["turn_judge/breaker_open"] = float(self._client.breaker_open())
        if step.latencies:
            lat = sorted(step.latencies)
            m["turn_judge/latency_s_mean"] = sum(lat) / len(lat)
            m["turn_judge/latency_s_p90"] = lat[min(len(lat) - 1, int(0.9 * len(lat)))]
            m["turn_judge/latency_s_max"] = lat[-1]
        judged = step.counts.get("judged_groups", 0.0)
        if judged:
            m["turn_judge/timeout_rate"] = (
                step.counts.get("status/timeout", 0.0) / judged
            )
        if self._log is not None:
            m["turn_judge/state_bytes"] = float(self._log.bytes_written)
            m["turn_judge/state_write_errors_total"] = float(self._log.write_errors)
        return m

    async def close(self) -> None:
        await self._client.close()
        if self._log is not None:
            await asyncio.to_thread(self._log.close)
