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
"""RemoteGeneration — GenerationInterface wrapper for disaggregated vLLM.

Two modes:

  Co-located (``generation`` provided):
    Wraps a VllmGeneration instance in the SAME Ray cluster. All calls delegate
    to the underlying VllmGeneration; a GenerationRouter runs alongside
    for external HTTP clients (e.g. NemoGym).

  HTTP-only (``generation=None`` + ``server_url``):
    VllmGeneration lives in a SEPARATE Ray cluster (or standalone process).
    The unified GenerationRouter at ``server_url`` exposes both the data plane
    (``/v1/completions`` with cordon + replay) and the control plane (weight
    sync, lifecycle, refit gate, metrics) on one port.
"""

from __future__ import annotations

import asyncio
import io
import time
from typing import Any, AsyncGenerator, Optional

import aiohttp
import ray
import requests
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationInterface,
    GenerationOutputSpec,
)

# Timeout for HTTP requests to the generation server (seconds).
# Weight sync / NCCL operations can take minutes.
_HTTP_TIMEOUT = 600


@ray.remote(num_cpus=0)
def _http_call_blocking(url: str, json_body: dict | None = None, raw_body: bytes | None = None, timeout: int = _HTTP_TIMEOUT, raise_on_status: bool = True) -> dict:  # pragma: no cover
    """Fire-and-forget HTTP POST wrapped in a Ray task.

    Returns the JSON response dict. Using a Ray remote function lets the
    caller get back a future immediately, which is critical for NCCL
    rendezvous: both training and inference sides must enter simultaneously.

    ``raise_on_status``: when False, return the body on 4xx/5xx instead of
    raising. Caller is responsible for inspecting status. Used by
    ``init_collective`` HTTP-mode so the router's structured 503 body
    (``current_gen_world_size`` / ``failed_indices`` / ``evicted_shard_ids``)
    survives the round trip — without this flag, ``raise_for_status`` collapses
    503s into ``HTTPError`` and the caller loses the structured info needed
    to decide whether to retry vs fail. Default True preserves the
    raise-on-5xx semantics expected by broadcast / refit callers.
    """
    if raw_body is not None:
        resp = requests.post(url, data=raw_body, timeout=timeout, headers={"Content-Type": "application/octet-stream"})
    elif json_body is not None:
        resp = requests.post(url, json=json_body, timeout=timeout)
    else:
        resp = requests.post(url, timeout=timeout)
    if raise_on_status:
        resp.raise_for_status()
        return resp.json()
    body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {"raw_text": resp.text}
    body["_status_code"] = resp.status_code
    return body


class RemoteGeneration(GenerationInterface):
    """GenerationInterface wrapper that supports direct delegation or HTTP-only mode."""

    def __init__(
        self,
        generation: Optional[GenerationInterface],
        server_url: str,
        config: dict,
    ):
        self._generation = generation
        # The unified GenerationRouter exposes data plane + control plane on
        # one port. Both /v1/completions and /init_collective live here.
        self.server_url = server_url.rstrip("/")
        self._http_mode = generation is None

        # Per-step cache for the consolidated /step_metrics_snapshot fetch.
        # Train side calls get_step_metrics_snapshot() multiple times around
        # a wandb log (once for spec decode, once for vllm logger, once for
        # router); we don't want to cross the cluster boundary three times.
        # Caller passes ``step`` to invalidate; if None, the cache is
        # bypassed (always re-fetch).
        self._step_snapshot_cache: Optional[dict[str, Any]] = None
        self._step_snapshot_cache_step: Optional[int] = None

        if self._http_mode:
            self.cfg = self._fetch_remote_config(config)
        else:
            self.cfg = dict(generation.cfg)

        # Merge caller-provided overrides
        for key in (
            "remote_generation_url",
            "max_new_tokens",
            "temperature",
            "top_p",
            "top_k",
            "stop_token_ids",
            "stop_strings",
        ):
            if key in config:
                self.cfg[key] = config[key]

        # In HTTP mode, /v1/{completions,chat/completions} go to the
        # unified router (which proxies to a healthy shard with replay on
        # failure). The shard URLs are still pulled once for diagnostics
        # and so colocated NemoGym (in the train cluster) can see them.
        self._shard_urls: list[str] = []
        if self._http_mode:
            self._shard_urls = self._fetch_shard_urls()
            print(
                f"  ✓ Disagg HTTP routing via unified router {self.server_url} "
                f"({len(self._shard_urls)} backing DP shard(s))",
                flush=True,
            )

        # NemoGym round-robins across whatever URLs we publish here. In
        # HTTP-only mode we publish a single URL = the router (transparent
        # cordon + replay for gym too). Co-located mode points gym at the
        # local router's /v1 surface.
        if self._http_mode:
            self.dp_openai_server_base_urls = [f"{self.server_url}/v1"]
        else:
            self.dp_openai_server_base_urls = [f"{self.server_url}/v1"]

    def _fetch_shard_urls(self) -> list[str]:
        """Fetch per-shard vLLM URLs from the gen server."""
        resp = requests.get(f"{self.server_url}/dp_openai_server_base_urls", timeout=30)
        resp.raise_for_status()
        urls = [u for u in resp.json() if u is not None]
        if not urls:
            raise RuntimeError("No shard URLs returned from generation server")
        return urls

    # =====================================================================
    # Refit gate + train↔gen NCCL world-size synchronization
    # =====================================================================

    def _wait_until_refit_ready(
        self, timeout: float = 600.0, poll_interval: float = 0.5
    ) -> None:
        """Block until the gen server reports refit_ready=True.

        Raises TimeoutError if the gate doesn't open in time. Older gen
        servers don't expose ``/refit_ready`` and 404 — treat as ready
        (back-compat with the static-cluster disagg path).
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                resp = requests.get(f"{self.server_url}/refit_ready", timeout=10)
                if resp.status_code == 404:
                    return
                resp.raise_for_status()
                payload = resp.json()
                if payload.get("ready"):
                    return
                last_reason = payload.get("reason", "unknown")
            except requests.RequestException as e:
                last_reason = f"transport: {e}"
            time.sleep(poll_interval)
        raise TimeoutError(
            f"refit_ready did not become true within {timeout}s "
            f"(last reason: {last_reason})"
        )

    def _fetch_current_gen_world_size(self) -> Optional[int]:
        """Read /current_gen_world_size from the gen server.

        Returns None when the endpoint isn't available — caller should
        skip world-size re-sync (back-compat with the static path).
        """
        try:
            resp = requests.get(
                f"{self.server_url}/current_gen_world_size", timeout=10
            )
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return int(resp.json()["world_size"])
        except (requests.RequestException, KeyError, ValueError):
            return None

    def _evict_gen_stragglers(self, gen_idxs: list[int]) -> None:
        """Tell the router to remove gen-side shards whose init_collective
        future didn't complete in the rendezvous timeout window.

        ``gen_idxs`` is the index into the futures list returned by
        ``self.init_collective(...)`` — that list is in shard-rank order
        as the gen worker_group dispatched, so we map index → shard_id
        via ``GET /shards`` (the router orders shards by insertion).

        Best-effort: a 5xx from the router or a network error is logged
        and swallowed so the surrounding retry loop keeps going. The
        worst case is the next attempt rendezvouses at the same broken
        world, times out again, and exits via ``max_attempts``.
        """
        if not gen_idxs:
            return
        try:
            resp = requests.get(f"{self.server_url}/shards", timeout=5)
            resp.raise_for_status()
            shards = resp.json()
        except Exception as e:  # noqa: BLE001
            print(
                f"  ⚠ _evict_gen_stragglers: /shards lookup failed ({e}); "
                f"skipping eviction",
                flush=True,
            )
            return
        # init_collective is dispatched on workers in the same order
        # they appear in worker_group. Shards are listed in router
        # insertion order; per-shard world size = 1 for the 4B/30B
        # FT recipes, so idx-into-futures == idx-into-shards. If
        # per_shard_world_size > 1 (TP>1) we'd need an idx → shard map
        # via worker_metadata; left as a follow-up.
        for idx in gen_idxs:
            if idx >= len(shards):
                continue
            shard_id = shards[idx].get("shard_id")
            if not shard_id:
                continue
            try:
                resp = requests.post(
                    f"{self.server_url}/admin/remove_shard",
                    json={
                        "shard_id": shard_id,
                        "reason": "ensure_collective_synced: rendezvous straggler",
                        # Skip the 30s drain — the shard is already
                        # wedged in NCCL rendezvous, no real inflight
                        # traffic to drain. Bound the call-side HTTP
                        # timeout at 30s so a wedged router doesn't
                        # pin this retry attempt.
                        "drain_timeout_s": 0.0,
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                print(
                    f"  ✓ evicted straggler {shard_id} (idx {idx}): "
                    f"{resp.json()}",
                    flush=True,
                )
            except Exception as e:  # noqa: BLE001
                print(
                    f"  ⚠ failed to evict straggler {shard_id} (idx {idx}): {e}",
                    flush=True,
                )

    def ensure_collective_synced(
        self,
        policy: Any,
        rendezvous_timeout_s: float = 150.0,
        max_attempts: int = 6,
    ) -> None:
        """Re-init the train↔gen NCCL group if the gen-side world size
        changed since the last refit.

        Robust to mid-rendezvous shard removal: if a peer dies between
        the world-size read and rendezvous completion (e.g. a fault
        injector firing in the middle of post-recover refit), the
        rendezvous would otherwise hang forever waiting for the dead
        rank. We bound each rendezvous attempt at
        ``rendezvous_timeout_s``, abort both sides on timeout, re-read
        ``/current_gen_world_size`` (now reflects the new fault), and
        retry up to ``max_attempts`` times. If rendezvous never settles,
        we raise — the run will fail loudly rather than silently wedge.

        The train driver calls this once at the top of
        ``refit_policy_generation`` (in disagg mode). It:
          1. Polls ``/refit_ready`` until the router reports the gen-side
             reset_collective is done.
          2. Fetches the current gen world size.
          3. If different from ``self._last_synced_world_size``, picks a
             fresh ``(ip, port)`` from the train cluster, dispatches
             ``init_collective`` symmetrically on both sides, and
             ``ray.wait`` blocks (with timeout) until rendezvous
             completes — retrying if a peer goes missing mid-flight.

        Idempotent / no-op when world size hasn't changed (the steady
        state). Safe to call on the first refit too — sets
        ``_last_synced_world_size`` from the initial init_collective.
        """
        # Only meaningful in HTTP-only disagg mode. Co-located /
        # delegating-RemoteGeneration use the wrapped instance directly.
        if not self._http_mode:
            inner = self._generation
            if inner is not None and hasattr(inner, "ensure_collective_synced"):
                inner.ensure_collective_synced(policy)
            return

        for attempt in range(1, max_attempts + 1):
            self._wait_until_refit_ready()
            new_gen_ws = self._fetch_current_gen_world_size()
            if new_gen_ws is None:
                return  # Older gen server: no dynamic world size, nothing to do.

            # RL-412 follow-up: under the RefitWorker architecture only ONE
            # train-side actor (the RefitWorker) participates in the
            # cross-cluster NCCL group. Effective ``train_world_size`` for
            # rank-numbering purposes is therefore 1, regardless of how
            # many physical Megatron workers exist. Legacy DTensor path
            # still has every train rank in the group.
            uses_refit_worker = bool(getattr(policy, "_use_refit_worker", False))
            if uses_refit_worker:
                effective_train_ws = 1
            else:
                effective_train_ws = policy.worker_group.cluster.world_size()
            new_total_ws = effective_train_ws + new_gen_ws
            last = getattr(self, "_last_synced_world_size", None)
            # NB: with per-refit teardown (RefitWorker.broadcast_weights_
            # until_complete and vllm_backend.update_weights_from_collective
            # both call ``group.destroy()`` at the end of every refit), the
            # cross-cluster ``model_update_group`` does NOT exist between
            # refits. So we ALWAYS need to re-rendezvous here, even when
            # the world size hasn't changed since the last refit. The
            # ``_last_synced_world_size`` field is kept only for log
            # output; it no longer gates the init.

            ip, port = policy.worker_group.cluster.get_master_address_and_port()
            print(
                f"  ↻ ensure_collective_synced [attempt {attempt}/{max_attempts}]: "
                f"world_size {last} → {new_total_ws} "
                f"(effective_train={effective_train_ws} "
                f"refit_worker={uses_refit_worker}, gen={new_gen_ws}); "
                f"rendezvous on {ip}:{port}",
                flush=True,
            )
            futures_train = policy.init_collective(
                ip, port, new_total_ws, train_world_size=effective_train_ws
            )
            futures_inf = self.init_collective(
                ip, port, new_total_ws, train_world_size=effective_train_ws
            )
            all_futures = list(futures_train) + list(futures_inf)
            ready, pending = ray.wait(
                all_futures,
                num_returns=len(all_futures),
                timeout=rendezvous_timeout_s,
            )
            if not pending:
                # Router-owned classification: in HTTP mode, the gen-side
                # ``init_collective`` future resolves to the router's
                # response body (``raise_on_status=False`` preserves the
                # structured 503 payload). 200 = success across all
                # surviving shards; 503 = router did per-worker eviction
                # and the new world size has shrunk. Train just retries
                # at the next world; we do NOT re-do the classification
                # here. Index-based eviction is gone — that's bug #1.
                try:
                    results = ray.get(ready)
                    inf_results = [r for r in results if isinstance(r, dict)]
                    inf_status_codes = [
                        r.get("_status_code", 200) for r in inf_results
                    ]
                    if any(code != 200 for code in inf_status_codes):
                        # Router classified + evicted internally. Refresh
                        # world size on next attempt and retry.
                        bad = [r for r in inf_results if r.get("_status_code", 200) != 200]
                        print(
                            f"  ⚠ ensure_collective_synced attempt {attempt}: "
                            f"router returned {bad[0].get('_status_code')}: "
                            f"{bad[0].get('error', '<no error>')}; "
                            f"new gen world={bad[0].get('current_gen_world_size', '?')} "
                            f"evicted={bad[0].get('evicted_shard_ids', [])} — "
                            f"retrying at smaller world",
                            flush=True,
                        )
                    else:
                        self._last_synced_world_size = new_total_ws
                        if attempt > 1:
                            print(
                                f"  ✓ ensure_collective_synced recovered on attempt {attempt}",
                                flush=True,
                            )
                        return
                except Exception as e:  # noqa: BLE001 — covers RayActorError, transport errors, etc.
                    print(
                        f"  ⚠ ensure_collective_synced attempt {attempt} raised "
                        f"{type(e).__name__}: {e}",
                        flush=True,
                    )
            else:
                # Outer timeout: gen-side router took longer than our
                # ``rendezvous_timeout_s``. With router-owned classification
                # (raise_on_status=False on the gen future), this should
                # only happen on transport / proxy hangs — the router's
                # 90s per-worker classifier completes well within our
                # 150s default. Train side just cleans up its NCCL state
                # and retries; we do NOT try to identify "straggler" gen
                # shards by index, because the gen-side future is a single
                # HTTP call covering all shards (bug #1).
                pending_set = set(pending)
                pending_gen = sum(1 for f in futures_inf if f in pending_set)
                pending_train = sum(1 for f in futures_train if f in pending_set)
                print(
                    f"  ⚠ ensure_collective_synced attempt {attempt} timed out "
                    f"after {rendezvous_timeout_s}s; pending: gen={pending_gen} "
                    f"train={pending_train} — retrying (router does its own "
                    f"per-worker classification on the gen side)",
                    flush=True,
                )

            # Tear down the train-side comm so the next attempt starts
            # from a clean state. The gen side recovers on its own
            # whenever the next fault completes (remove_shard/add_shard
            # both call reset_collective on surviving workers), so we
            # don't need to drive a gen-side abort here — re-reading
            # /current_gen_world_size on the next attempt picks up
            # whatever world the gen side has converged to.
            try:
                abort_train = policy.abort_collective()
                if abort_train:
                    ray.wait(list(abort_train), timeout=15.0)
            except Exception as e:  # noqa: BLE001
                print(
                    f"  ⚠ abort_collective dispatch (train) raised {type(e).__name__}: {e}",
                    flush=True,
                )
            for f in pending:
                try:
                    ray.cancel(f, force=True)
                except Exception:  # noqa: BLE001
                    pass

            # CRITICAL: wait for the cancelled futures to actually settle
            # before starting the next attempt. Without this, we just call
            # ``ray.cancel`` and loop back to dispatch a NEW init_collective
            # — but the in-flight ones haven't finished yet. On the gen
            # side, ``ray.cancel(force=True)`` raises ``CancelledError`` at
            # the next ``await`` point, but vLLM's ``collective_rpc``
            # internals dispatch to TP subprocesses via ``MessageQueue``
            # which serializes — the dispatched work CONTINUES even if the
            # leader's coroutine is cancelled (the TP subprocesses don't
            # see the cancel). So the leader has a queue: the cancelled
            # coroutine, then the new one we're about to dispatch. The
            # new one waits behind the old one's TP work, and meanwhile
            # train has moved its master port — so the new dispatch
            # rendezvous against a stale master, fails, train retries
            # again, repeat until eviction kicks in and (incorrectly)
            # kills the healthy-but-slow leaders.
            #
            # Concrete failure mode (30B TP=2 mid-broadcast pod-kill):
            # train retries 3x at 45s each. Eviction at attempt 3 of 6
            # tears down survivor shards that were just queued behind a
            # stale 30s rendezvous; world cascades to 0.
            #
            # Fix: after cancel, ``ray.wait`` on the cancelled futures
            # with a timeout that exceeds gen-side rendezvous timeout
            # (StatelessProcessGroup uses 30s). 60s gives slack for
            # cancel propagation + any internal vLLM cleanup. Train's
            # next attempt is GUARANTEED to start on a clean gen side
            # — the leaders' actor task queue is empty when train
            # dispatches. No queue buildup, no spurious eviction.
            try:
                ray.wait(pending, timeout=60.0)
            except Exception:  # noqa: BLE001
                pass
            # Brief settle for gen-side reset_collective + router state
            # consistency before the next /current_gen_world_size read.
            time.sleep(1.0)

        raise RuntimeError(
            f"ensure_collective_synced failed to rendezvous after {max_attempts} "
            f"attempts; gen cluster may be in an unrecoverable state"
        )

    def _fetch_remote_config(self, local_config: dict) -> dict:
        """Fetch generation config from the remote gen server.

        vLLM init on cold cache (model load + CUDA graph capture +
        inductor compile) takes ~3 min on GB300, so we poll for ~10 min
        before giving up — long enough to cover the full bootstrap
        without blocking dev iteration too much when gen is genuinely
        broken.
        """
        max_attempts = 120  # 120 × 5s = 600s = 10 min
        for attempt in range(max_attempts):
            try:
                resp = requests.get(f"{self.server_url}/config", timeout=10)
                resp.raise_for_status()
                remote_cfg = resp.json()
                print(f"  ✓ Fetched remote generation config from {self.server_url}")
                return remote_cfg
            except Exception as e:
                if attempt < max_attempts - 1:
                    print(
                        f"  Waiting for gen server at {self.server_url} "
                        f"(attempt {attempt + 1}/{max_attempts}): {e}"
                    )
                    time.sleep(5)
                else:
                    raise RuntimeError(
                        f"Failed to reach generation server at {self.server_url}/config "
                        f"after {max_attempts} attempts"
                    ) from e

    # =====================================================================
    # Data plane — generation
    # =====================================================================

    def generate(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> BatchedDataDict[GenerationOutputSpec]:
        if not self._http_mode:
            return self._generation.generate(data, greedy)
        return asyncio.run(self._generate_json_completions(data, greedy))

    async def _generate_json_completions(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool
    ) -> BatchedDataDict[GenerationOutputSpec]:
        """Send batch via /v1/completions JSON endpoint.

        Uses the OpenAI-compatible completions API with prompt_token_ids.
        Goes through vLLM's full OpenAI serving layer (tokenization,
        validation, etc). The unified router proxies to a healthy shard
        with replay on failure.
        """
        # Aligned with the router's ``proxy_timeout_s`` (default 600s,
        # set explicitly in the recipe YAML for 30B-class shards under
        # heavy gen burst). The mismatch the previous client-side 300s
        # introduced — train giving up at 5 min while the router would
        # still happily wait 10 — caused a hard ``asyncio.TimeoutError``
        # on the 30B TP=2 post-fault path: 4 shards' worth of inflight
        # was rebalanced onto 3 surviving shards, per-request latency
        # nearly doubled, and the train-side cap fired before the
        # router gave up. By matching router proxy_timeout we let the
        # router's own retry/replay handle transient slow shards.
        gen_timeout = aiohttp.ClientTimeout(total=600)

        completions_url = f"{self.server_url}/v1/completions"

        input_ids = data["input_ids"]
        input_lengths = data["input_lengths"]
        batch_size = input_ids.shape[0]
        max_new_tokens = self.cfg.get("max_new_tokens", 2048)
        max_model_len = self.cfg.get("vllm_cfg", {}).get("max_model_len", 4096)

        temperature = 0.0 if greedy else self.cfg.get("temperature", 1.0)
        top_p = 1.0 if greedy else self.cfg.get("top_p", 1.0)

        # Build per-sample requests
        requests_list = []
        for i in range(batch_size):
            length = input_lengths[i].item()
            prompt_tokens = input_ids[i, :length].tolist()
            max_tokens = min(max_new_tokens, max_model_len - length)
            req = {
                "model": self.cfg.get("model_name", "default"),
                "prompt": prompt_tokens,  # vLLM accepts list[int] as token IDs
                "max_tokens": max(max_tokens, 1),
                "temperature": temperature,
                "top_p": top_p,
                "logprobs": 1,
            }
            if self.cfg.get("stop_token_ids"):
                req["stop_token_ids"] = self.cfg["stop_token_ids"]
            requests_list.append(req)

        # Send all requests concurrently to the router. Per-request retry
        # on transient transport errors (TimeoutError, ServerDisconnectedError,
        # ClientConnectionError, ClientPayloadError) so a single slow shard
        # under post-fault rebalanced load doesn't sink the whole step. Each
        # retry goes through the router's load balancer fresh, so the next
        # attempt is routed to whichever shard has the smallest queue.
        # 5xx responses are NOT retried here — the router itself retries
        # 5xx via its replay buffer (cordon + replay-to-different-shard).
        # Only client-observed transport failures (where the router didn't
        # even respond) get retried, since the router has no chance to
        # apply its own logic in that case.
        async with aiohttp.ClientSession(timeout=gen_timeout) as session:
            async def _send_one(req):
                # Up to 3 attempts: original + 2 retries. Linear backoff
                # 0.5s → 1.0s. 30B generation latency dominates anyway,
                # so backoff just prevents instant retry from hitting the
                # same overloaded shard before the router rebalances.
                last_err: Exception | None = None
                for attempt in range(3):
                    try:
                        async with session.post(completions_url, json=req) as resp:
                            resp.raise_for_status()
                            return await resp.json()
                    except (
                        asyncio.TimeoutError,
                        aiohttp.ServerDisconnectedError,
                        aiohttp.ClientConnectionError,
                        aiohttp.ClientPayloadError,
                    ) as e:
                        last_err = e
                        if attempt < 2:
                            await asyncio.sleep(0.5 * (attempt + 1))
                            continue
                        raise
                # Unreachable (loop either returns or raises), but mypy.
                raise last_err  # type: ignore[misc]
            responses = await asyncio.gather(*[_send_one(r) for r in requests_list])

        # Parse responses into GenerationOutputSpec
        pad_token_id = self.cfg.get("_pad_token_id", 0)
        all_output_ids = []
        all_gen_lengths = []
        all_unpadded_lengths = []
        all_logprobs = []
        all_truncated = []

        for i, resp_json in enumerate(responses):
            choice = resp_json["choices"][0]
            finish_reason = choice.get("finish_reason", "stop")
            input_length = input_lengths[i].item()

            # Extract generated token IDs from logprobs.tokens ("token_id:NNN" format)
            gen_token_ids = []
            lp_list = []
            logprobs_data = choice.get("logprobs")
            if logprobs_data and "tokens" in logprobs_data:
                for tok_str in logprobs_data["tokens"]:
                    if tok_str.startswith("token_id:"):
                        gen_token_ids.append(int(tok_str.split(":")[1]))
                    else:
                        gen_token_ids.append(0)  # fallback
                lp_list = [lp if lp is not None else 0.0 for lp in logprobs_data.get("token_logprobs", [])]

            gen_length = len(gen_token_ids)
            unpadded_length = input_length + gen_length
            prompt_tokens = input_ids[i, :input_length].tolist()
            full_ids = prompt_tokens + gen_token_ids

            # Pad logprobs: zeros for input tokens, then actual logprobs
            full_logprobs = [0.0] * input_length + lp_list
            # Pad to same length as full_ids
            while len(full_logprobs) < len(full_ids):
                full_logprobs.append(0.0)

            all_output_ids.append(full_ids)
            all_gen_lengths.append(gen_length)
            all_unpadded_lengths.append(unpadded_length)
            all_logprobs.append(full_logprobs)
            all_truncated.append(finish_reason == "length")

        # Pad to uniform sequence length
        max_seq_len = max(len(ids) for ids in all_output_ids)
        for i in range(batch_size):
            pad_len = max_seq_len - len(all_output_ids[i])
            all_output_ids[i].extend([pad_token_id] * pad_len)
            all_logprobs[i].extend([0.0] * pad_len)

        return BatchedDataDict[GenerationOutputSpec]({
            "output_ids": torch.tensor(all_output_ids, dtype=torch.long),
            "generation_lengths": torch.tensor(all_gen_lengths, dtype=torch.long),
            "unpadded_sequence_lengths": torch.tensor(all_unpadded_lengths, dtype=torch.long),
            "logprobs": torch.tensor(all_logprobs, dtype=torch.float32),
            "truncated": torch.tensor(all_truncated, dtype=torch.bool),
        })

    async def generate_async(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> AsyncGenerator[tuple[int, BatchedDataDict[GenerationOutputSpec]], None]:
        if not self._http_mode:
            async for result in self._generation.generate_async(data, greedy):
                yield result
            return

        result = await self._generate_json_completions(data, greedy)
        batch_size = result["output_ids"].shape[0]
        for i in range(batch_size):
            single = BatchedDataDict[GenerationOutputSpec]({
                k: v[i:i+1] if isinstance(v, torch.Tensor) else ([v[i]] if isinstance(v, list) else v)
                for k, v in result.items()
            })
            yield (i, single)

    # =====================================================================
    # Weight sync and lifecycle
    # =====================================================================

    def init_collective(
        self, ip: str, port: int, world_size: int, *, train_world_size: int
    ) -> list[ray.ObjectRef]:
        # Cache the world size we last rendezvoused at so
        # ensure_collective_synced can skip work in steady state.
        self._last_synced_world_size = world_size
        if not self._http_mode:
            return self._generation.init_collective(
                ip, port, world_size, train_world_size=train_world_size
            )

        # HTTP mode: dispatch as a Ray task so it returns a future.
        # The training side calls ray.get(futures_train + futures_inference)
        # and both sides must enter NCCL rendezvous simultaneously.
        url = f"{self.server_url}/init_collective"
        print(
            f"    [RemoteGeneration] dispatching _http_call_blocking POST {url} "
            f"world_size={world_size} train_world_size={train_world_size}",
            flush=True,
        )
        # ``raise_on_status=False``: the router's 503 response carries the
        # structured per-worker classifier result (current_gen_world_size,
        # evicted_shard_ids, failed_indices). ``raise_for_status`` would
        # collapse it into a generic HTTPError on the train side, losing
        # that info — which is exactly the bug behind the index-based
        # straggler eviction that incorrectly evicted dp-0 on outer
        # timeouts. ensure_collective_synced inspects the body to decide
        # retry vs success without trying to do its own classification.
        ref = _http_call_blocking.remote(
            url,
            json_body={
                "ip": ip,
                "port": port,
                "world_size": world_size,
                "train_world_size": train_world_size,
            },
            raise_on_status=False,
        )
        print(f"    [RemoteGeneration] Ray ObjectRef: {ref}", flush=True)
        return [ref]

    def update_weights_from_collective(self) -> list[ray.ObjectRef]:
        if not self._http_mode:
            return self._generation.update_weights_from_collective()

        return [
            _http_call_blocking.remote(
                f"{self.server_url}/update_weights_from_collective",
            )
        ]

    def reset_collective(self) -> list[ray.ObjectRef]:
        """Tear down the gen-side cross-cluster weight-sync NCCL group.

        Symmetric to the train-side ``abort_collective``: invoked by the
        broadcast-retry path in ``grpo.py`` when a fault is detected, to
        force gen workers to drop the stale comm before the next
        ``ensure_collective_synced`` re-rendezvouses at the smaller
        world size. Not called on the steady-state path — the comm is
        long-lived across refits.
        """
        if not self._http_mode:
            return self._generation.reset_collective()

        return [
            _http_call_blocking.remote(
                f"{self.server_url}/reset_collective",
            )
        ]

    def prepare_for_generation(self, *args: Any, **kwargs: Any) -> bool:
        if not self._http_mode:
            return self._generation.prepare_for_generation(*args, **kwargs)

        resp = requests.post(f"{self.server_url}/prepare_for_generation", timeout=_HTTP_TIMEOUT)
        resp.raise_for_status()
        return resp.json().get("success", False)

    def finish_generation(self, *args: Any, **kwargs: Any) -> bool:
        if not self._http_mode:
            return self._generation.finish_generation(*args, **kwargs)

        resp = requests.post(f"{self.server_url}/finish_generation", timeout=_HTTP_TIMEOUT)
        resp.raise_for_status()
        return resp.json().get("success", False)

    def prepare_refit_info(self, state_dict_info: dict[str, Any]) -> None:
        if not self._http_mode:
            self._generation.prepare_refit_info(state_dict_info)
            return

        buf = io.BytesIO()
        torch.save(state_dict_info, buf)
        resp = requests.post(
            f"{self.server_url}/prepare_refit_info",
            data=buf.getvalue(),
            headers={"Content-Type": "application/octet-stream"},
            timeout=_HTTP_TIMEOUT,
        )
        resp.raise_for_status()

    def update_weights_via_ipc_zmq(self) -> list[ray.ObjectRef]:
        if not self._http_mode:
            return self._generation.update_weights_via_ipc_zmq()
        raise NotImplementedError("update_weights_via_ipc_zmq not supported in HTTP mode")

    def invalidate_kv_cache(self) -> bool:
        if not self._http_mode:
            return self._generation.invalidate_kv_cache()

        resp = requests.post(f"{self.server_url}/invalidate_kv_cache", timeout=_HTTP_TIMEOUT)
        resp.raise_for_status()
        return resp.json().get("success", False)

    @property
    def requires_kv_scale_sync(self) -> bool:
        if not self._http_mode:
            return getattr(self._generation, "requires_kv_scale_sync", False)
        return False

    def clear_logger_metrics(self) -> None:
        if not self._http_mode:
            self._generation.clear_logger_metrics()
            return

        try:
            requests.post(f"{self.server_url}/clear_logger_metrics", timeout=30)
        except requests.RequestException:
            pass

    def get_logger_metrics(self) -> dict[str, Any]:
        if not self._http_mode:
            return self._generation.get_logger_metrics()

        try:
            resp = requests.get(f"{self.server_url}/get_logger_metrics", timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            return {}

    def snapshot_step_metrics(self) -> None:
        if not self._http_mode:
            if hasattr(self._generation, "snapshot_step_metrics"):
                self._generation.snapshot_step_metrics()
            return

        try:
            requests.post(f"{self.server_url}/snapshot_step_metrics", timeout=30)
        except requests.RequestException:
            pass

    def get_step_metrics(self) -> dict[str, float]:
        if not self._http_mode:
            if hasattr(self._generation, "get_step_metrics"):
                return self._generation.get_step_metrics()
            return {}

        try:
            resp = requests.get(f"{self.server_url}/get_step_metrics", timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            return {}

    def get_step_metrics_snapshot(
        self, step: Optional[int] = None
    ) -> dict[str, Any]:
        """Consolidated per-step gen-side metrics for wandb logging.

        Returns a dict with a stable shape so callers can index without
        is-None checks:

        ::

            {
              "vllm_logger_metrics": {...},   # same shape as get_logger_metrics()
              "spec_decode_metrics": {},      # always empty, see note
              "router_metrics": {...},        # only populated in disagg mode
            }

        Note: ``spec_decode_metrics`` is intentionally empty here. Spec
        decode counters are a destructive delta-since-snapshot read; grpo
        consumes them through the existing ``get_step_metrics()`` path so
        we don't bake a second consumer that would race for the same
        baseline. The key is preserved for shape stability.

        In HTTP mode this is a single round-trip to
        ``GET /step_metrics_snapshot`` on the gen server. The result is
        cached per ``step``: callers in the same training step share one
        fetch. Pass ``step=None`` (the default) to bypass the cache.

        In co-located mode we synthesize the same dict locally with no
        HTTP — the wrapped generation lives in the same Ray cluster.
        Co-located runs don't have a router so ``router_metrics`` is
        ``{}``.

        On HTTP transport failure we return an empty-shape dict rather
        than raising — wandb logging must never block training. This
        matches the existing ``get_logger_metrics()`` failure semantics.
        """
        if step is not None and self._step_snapshot_cache_step == step:
            cached = self._step_snapshot_cache
            if cached is not None:
                return cached

        if not self._http_mode:
            inner = self._generation
            # Spec-decode metrics are NOT included — they're a destructive
            # delta-since-snapshot read that grpo consumes through the
            # existing get_step_metrics() pathway. Mirroring the HTTP
            # endpoint's contract so callers don't accidentally double-
            # consume the baseline.
            payload: dict[str, Any] = {
                "vllm_logger_metrics": (
                    inner.get_logger_metrics() if inner is not None else {}
                ),
                "spec_decode_metrics": {},
                "router_metrics": {},
            }
        else:
            try:
                resp = requests.get(
                    f"{self.server_url}/step_metrics_snapshot", timeout=30
                )
                resp.raise_for_status()
                payload = resp.json()
            except requests.RequestException:
                payload = {
                    "vllm_logger_metrics": {},
                    "spec_decode_metrics": {},
                    "router_metrics": {},
                }

        if step is not None:
            self._step_snapshot_cache = payload
            self._step_snapshot_cache_step = step
        return payload
