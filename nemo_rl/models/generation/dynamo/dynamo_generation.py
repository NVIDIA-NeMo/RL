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
"""Dynamo backend for nemo-rl: a thin URL forwarder to a DynamoGraphDeployment.

On Kubernetes, a ``DynamoGraphDeployment`` (DGD) owns the entire inference
stack — etcd, NATS, the dynamo frontend, and the vLLM/sglang/trtllm workers.
This class does not bring any of that up; it only resolves the cluster-internal
URL of the DGD's frontend Service so nemo-gym can dispatch rollout requests to
it over HTTP.
"""

import asyncio
import threading
import time
import warnings
from typing import Any, AsyncGenerator, Optional, Union

import ray
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.generation.dynamo.config import DynamoConfig
from nemo_rl.models.generation.dynamo.token_wrapper import DynamoTokenWrapperServer
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationInterface,
    GenerationOutputSpec,
    verify_right_padding,
)
from nemo_rl.utils.k8s import is_in_kubernetes, read_pod_namespace

DEFAULT_FRONTEND_PORT = 8000
DEFAULT_DYN_SYSTEM_PORT = 9090


def _http_post_json(
    url: str, payload: dict[str, Any], timeout_s: float
) -> dict[str, Any]:
    """POST a JSON body to a URL and parse the JSON response.

    Returns the parsed dict (or a ``{"status": "error", ...}`` shape on
    transport / HTTP error). Never raises — caller decides how to handle
    a non-ok status.
    """
    import json
    import urllib.error
    import urllib.request

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read()
    except urllib.error.HTTPError as exc:
        err = exc.read().decode("utf-8", "replace") if exc.fp else ""
        return {"status": "error", "http_status": exc.code, "raw": err}
    except (urllib.error.URLError, TimeoutError) as exc:
        return {"status": "error", "transport_error": f"{type(exc).__name__}: {exc}"}
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return {"status": "error", "raw": body.decode("utf-8", "replace")}


def _format_dynamo_error(response: dict[str, Any]) -> str:
    """Format the internal error shape returned by ``_http_post_json``."""
    if "http_status" in response:
        return f"HTTP {response['http_status']}: {response.get('raw', '')}"
    if "transport_error" in response:
        return str(response["transport_error"])
    if "raw" in response:
        return str(response["raw"])
    return str(response)


def _parse_dynamo_completion_response(
    response: dict[str, Any], *, request_url: str
) -> tuple[list[int], list[float], bool]:
    """Parse the Dynamo OpenAI completion response for direct generation."""
    if not isinstance(response, dict):
        raise RuntimeError(
            f"Dynamo completion response from {request_url} was not a JSON object."
        )
    if response.get("status") == "error":
        raise RuntimeError(
            f"Dynamo completion request to {request_url} failed: "
            f"{_format_dynamo_error(response)}"
        )

    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(
            f"Dynamo completion response from {request_url} did not include choices."
        )
    choice = choices[0]
    if not isinstance(choice, dict):
        raise RuntimeError(
            f"Dynamo completion response from {request_url} has invalid choice shape."
        )

    nvext = response.get("nvext")
    if not isinstance(nvext, dict):
        raise RuntimeError(
            f"Dynamo completion response from {request_url} did not include nvext."
        )
    completion_token_ids = nvext.get("completion_token_ids")
    if not isinstance(completion_token_ids, list):
        raise RuntimeError(
            "Dynamo completion response did not include "
            "nvext.completion_token_ids. Ensure the DGD is based on "
            "jthomson04/tokenize-endpoint-merge-main-06-09 or newer."
        )
    generated_token_ids = [int(token_id) for token_id in completion_token_ids]

    generated_logprobs = [0.0] * len(generated_token_ids)
    logprobs = choice.get("logprobs")
    if isinstance(logprobs, dict):
        token_logprobs = logprobs.get("token_logprobs")
        if isinstance(token_logprobs, list):
            for idx, logprob in enumerate(token_logprobs[: len(generated_logprobs)]):
                if logprob is not None:
                    generated_logprobs[idx] = float(logprob)

    return (
        generated_token_ids,
        generated_logprobs,
        choice.get("finish_reason") == "length",
    )


# Interpreter/process noise (prometheus_client internals) — not engine
# telemetry. Override via policy.generation.dynamo_cfg.metrics_exclude_prefixes.
_DEFAULT_METRICS_EXCLUDE_PREFIXES = ("python_", "process_")

# Default collection allow-list, in two tiers. Bounding volume matters: a worker
# exposes ~120 metric families, and each becomes a per-worker timeline figure that
# wandb renders to a multi-MB Plotly string — logging all of them chokes ALL metric
# sync (train/reward/gpu never reach the cloud). Override / widen via
# policy.generation.dynamo_cfg.metrics_include_prefixes (pass [] to scrape all).
#
# Tier 1 — Dynamo *runtime* metrics (dynamo_component_* / dynamo_work_handler_*),
# emitted identically by ANY Dynamo engine, so backend-agnostic.
_DYNAMO_RUNTIME_METRIC_PREFIXES = (
    "dynamo_component_gpu_cache_usage",  # kv-cache utilization
    "dynamo_component_inflight_requests",  # inflight (running) requests
    "dynamo_work_handler_queue_depth",  # pending queue depth
    "dynamo_component_requests_total",  # request throughput (requests)
    "dynamo_work_handler_time_to_first_response",  # time-to-first-response latency
)
# Tier 2 — engine-specific passthroughs for high-value signals the Dynamo runtime
# does NOT expose: it has no token-level metrics and only coarse latency. vLLM is
# the only engine wired today; as others are onboarded add their equivalents here
# (e.g. "sglang:..."), and a non-matching engine just falls back to Tier 1.
_ENGINE_PASSTHROUGH_METRIC_PREFIXES = (
    "vllm:generation_tokens",  # generation throughput (tokens)
    "vllm:prompt_tokens_total",  # prompt throughput (tokens)
    "vllm:inter_token_latency",  # inter-token (decode) latency
)
_CURATED_METRICS_INCLUDE_PREFIXES = (
    _DYNAMO_RUNTIME_METRIC_PREFIXES + _ENGINE_PASSTHROUGH_METRIC_PREFIXES
)

# nemo-rl's print_performance_metrics (algorithms/utils.py) hard-asserts the
# vLLM backend's canonical generation-metric names are present in
# get_logger_metrics() output (inflight_batch_sizes / num_pending_samples), and
# log_generation_metrics_to_wandb plots them. We ALWAYS surface these four
# canonical keys — mapped from the generically-scraped vllm_* values (post ':'
# -> '_' sanitization), or an empty dict if absent — so (a) that assert never
# trips, (b) panels match the vLLM backend's names for direct comparison, and
# (c) a missing/renamed source metric degrades to a valid empty dict rather than
# crashing the training loop. Additive: the generic bulk collection is unchanged.
# Sources are tried in order. The dynamo_component_* / dynamo_work_handler_* names
# are backend-agnostic (Dynamo's runtime emits them identically for any engine);
# the engine-specific vllm_* names are kept as a fallback for the colocated-vLLM
# backend (only collected if metrics_include_prefixes is widened to include them).
_CANONICAL_LOGGER_ALIASES: "dict[str, list[str]]" = {
    "inflight_batch_sizes": [
        "dynamo_component_inflight_requests",
        "vllm_num_requests_running",
    ],
    "num_pending_samples": [
        "dynamo_work_handler_queue_depth",
        "vllm_num_requests_waiting",
    ],
    "kv_cache_usage_perc": [
        "dynamo_component_gpu_cache_usage_percent",
        "vllm_kv_cache_usage_perc",
        "vllm_gpu_cache_usage_perc",
    ],
    "generation_tokens": ["vllm_generation_tokens_total", "vllm_generation_tokens"],
}


def _http_get_text(url: str, timeout_s: float) -> "Optional[str]":
    """GET a URL, returning the decoded body or None on any transport/HTTP error.

    Sibling of :func:`_http_post_json` for the Prometheus ``/metrics`` scrape.
    Never raises — a best-effort telemetry scrape must not perturb training.
    """
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            return resp.read().decode("utf-8", "replace")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return None


def _parse_prometheus_metrics(
    text: str,
    include_prefixes: "Optional[tuple[str, ...]]" = None,
    exclude_prefixes: "tuple[str, ...]" = _DEFAULT_METRICS_EXCLUDE_PREFIXES,
) -> "dict[str, float]":
    """Parse Prometheus text exposition into ``{metric_name: summed_value}``.

    Deliberately schema-agnostic: it collects *every* scalar sample line rather
    than a fixed allow-list, so whatever the Dynamo workers emit today
    (``vllm:*``, ``dynamo_component_*``, runtime gauges) — and whatever they
    emit after an upgrade — flows through with no code change. Rules:

      * ``# HELP`` / ``# TYPE`` comment lines are skipped;
      * histogram ``*_bucket`` lines are skipped (cumulative-by-``le``; summing
        across buckets is meaningless — the scalar ``_sum`` / ``_count`` are
        kept);
      * ``*_created`` lines are skipped (prometheus_client per-series creation
        timestamps — a constant epoch value, not telemetry);
      * label sets are dropped and values for the same metric name are summed
        (each worker pod serves one engine, so this is a no-op in the common
        single-line case and a sane aggregate otherwise);
      * ``include_prefixes`` (if given) restricts to those families;
        ``exclude_prefixes`` drops matches (interpreter noise by default);
      * ``:`` in names is mapped to ``_`` so keys are wandb-safe (Prometheus
        names are ``[a-zA-Z0-9_:]``, so ``:`` is the only unsafe character).

    Never raises; unparseable lines are skipped.
    """
    out: dict[str, float] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line[0] == "#":
            continue
        if "{" in line:
            name = line[: line.index("{")]
            tail = line[line.rindex("}") + 1 :]
        else:
            sp = line.split(None, 1)
            if len(sp) != 2:
                continue
            name, tail = sp[0], sp[1]
        if name.endswith(("_bucket", "_created")):
            continue
        if include_prefixes and not name.startswith(include_prefixes):
            continue
        if exclude_prefixes and name.startswith(exclude_prefixes):
            continue
        tok = tail.split()
        if not tok:
            continue
        try:
            val = float(tok[0])
        except ValueError:
            continue
        key = name.replace(":", "_")
        out[key] = out.get(key, 0.0) + val
    return out


def _discover_worker_instances(
    *,
    frontend_host: str,
    frontend_port: int,
    dyn_namespaces: "set[str]",
    dyn_system_port: int,
    timeout_s: float = 15.0,
) -> list[dict[str, Any]]:
    """Discover live worker instances via the frontend's ``GET /health``.

    Each entry in the response's ``instances`` array has an
    ``instance_id`` (per-worker-pod stable identifier) and a
    ``transport.tcp`` URL of the form
    ``tcp://<pod_ip>:<tcp_port>/<channel>/<endpoint_name>``. We extract
    the pod IP, pair it with the well-known ``DYN_SYSTEM_PORT`` (default
    9090) where the ``/engine/<route>`` HTTP admin server listens, and
    dedupe per ``instance_id`` (each pod registers multiple endpoint
    instances but we only need one system URL per pod).

    Returns a list of ``{instance_id, system_url}`` dicts. Empty on
    transport / parse error — caller decides whether that's fatal.
    """
    import json
    import re
    import urllib.error
    import urllib.request

    url = f"http://{frontend_host}:{frontend_port}/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            data = json.loads(resp.read())
    except (
        urllib.error.HTTPError,
        urllib.error.URLError,
        TimeoutError,
        json.JSONDecodeError,
    ):
        return []
    instances = data.get("instances", []) if isinstance(data, dict) else []
    if not isinstance(instances, list):
        return []

    seen_by_id: dict[Any, dict[str, Any]] = {}
    # ``transport.tcp`` is ``<pod_ip>:<port>/<channel>/<endpoint>`` with no
    # scheme prefix; we just need the host part.
    tcp_re = re.compile(r"^(?:tcp://)?([^:/]+):")
    refit_discovery_endpoints = {"update_weights_via_mx", "rl"}
    for inst in instances:
        if not isinstance(inst, dict):
            continue
        if inst.get("namespace") not in dyn_namespaces:
            continue
        # Only the vLLM worker (component "backend") serves the /engine/*
        # admin routes. Older Dynamo images expose update_weights_via_mx as a
        # top-level endpoint; newer --enable-rl images expose a single rl
        # endpoint whose route descriptors include update_weights_via_mx.
        if inst.get("component") != "backend":
            continue
        if inst.get("endpoint") not in refit_discovery_endpoints:
            continue
        inst_id = inst.get("instance_id")
        if inst_id is None:
            continue
        transport = inst.get("transport") or {}
        tcp = transport.get("tcp") if isinstance(transport, dict) else None
        if not isinstance(tcp, str):
            continue
        m = tcp_re.match(tcp)
        if not m:
            continue
        pod_ip = m.group(1)
        worker = seen_by_id.setdefault(
            inst_id,
            {
                "instance_id": inst_id,
                "system_url": f"http://{pod_ip}:{dyn_system_port}",
                "requires_pause": False,
            },
        )
        if inst.get("endpoint") == "rl":
            worker["requires_pause"] = True
    return list(seen_by_id.values())


@ray.remote(num_cpus=0)
def _dispatch_update_weights_via_mx_remote(
    *,
    k8s_namespace: str,
    dgd_name: str,
    version: int,
    mx_config_dict: dict[str, Any],
    worker_namespaces: "list[str] | None" = None,
    frontend_port: int = DEFAULT_FRONTEND_PORT,
    dyn_system_port: int = DEFAULT_DYN_SYSTEM_PORT,
    refit_timeout_s: float = 300.0,
    admin_timeout_s: float = 30.0,
    max_convergence_iterations: int = 5,
) -> dict[str, Any]:
    """Synchronously orchestrate an MX refit cycle that converges over scaling.

    Architecture (minimum surgical, no biswapanda PR dependency):

      1. GET http://<dgd>-frontend.<ns>.svc:8000/health
         → enumerate worker pods from ``instances[*]``
         → key by ``instance_id``; system_url = http://<pod_ip>:<DYN_SYSTEM_PORT>
      2. For each NEW worker (not already refitted in this cycle):
         a. New --enable-rl workers: pause generation with mode="keep".
         b. POST /engine/update_weights_via_mx  (real NIXL receive, blocks)
         c. POST /engine/flush_cache            (drop stale prefix cache)
         d. New --enable-rl workers: resume generation.
      3. Re-discover via /health. If new instance_ids appeared (a worker
         scaled in or restarted during the cycle), go to step 2.
      4. Once a discovery shows no new instance_ids, return.

    Workers that DISAPPEARED mid-cycle (instance_id no longer in /health)
    are silently dropped — they're gone, so they can't be serving stale
    weights. Caller-visible failure only on per-worker step errors or if
    the loop fails to converge within ``max_convergence_iterations``
    passes (defense against pathological scale-thrash).
    """
    frontend_host = f"{dgd_name}-frontend.{k8s_namespace}.svc.cluster.local"
    # Which Dynamo namespace(s) hold the MX workers to refit:
    #   * Flat single-DGD deployment: the workers live in the frontend's own
    #     namespace ({k8s_ns}-{dgd_name}) — the default.
    #   * Hierarchical GlobalRouter deployment: the public Frontend is in the
    #     control DGD, but the MX workers live in the *pool* DGD namespaces.
    #     The caller passes those explicitly via ``worker_namespaces``. /health
    #     is cluster-wide, so the control frontend can still enumerate them.
    if worker_namespaces:
        dyn_namespaces = set(worker_namespaces)
    else:
        dyn_namespaces = {f"{k8s_namespace}-{dgd_name}"}
    payload = {"version": version, "mx_config": mx_config_dict}

    def _step(
        sys_url: str, route: str, body: dict[str, Any], timeout_s: float
    ) -> dict[str, Any]:
        return _http_post_json(f"{sys_url}/engine/{route}", body, timeout_s)

    refitted_ids: set[Any] = set()
    iteration_logs: list[dict[str, Any]] = []
    failures: list[str] = []

    # Shared deadline for retrying transient refit failures across the whole
    # cycle. The receiver's one-shot discover_v2_sources can miss the brief
    # (~1s) window between the trainer's mark_ready() and that READY status
    # propagating into the server's list_sources(status_filter=READY) index —
    # at 16 workers the first receivers fire inside that lag and get
    # "no v2 source available". Retrying (rather than raising) is what fixes
    # it: as long as THIS dispatcher keeps running, the trainer stays blocked
    # in ray.get(futures_inference) and alive, so its heartbeat holds the
    # published sources READY (server reaper heartbeat_timeout=90s) — and a
    # re-issued refit then discovers them. Raising on the first failure
    # instead crashes the trainer, which immediately STALEs the sources and
    # dooms every remaining worker. Bounded by mx_config.timeout_seconds so a
    # genuinely broken refit still surfaces instead of hanging forever.
    import time as _time

    _cycle_deadline = _time.monotonic() + float(
        mx_config_dict.get("timeout_seconds", 300.0)
    )

    for iteration in range(max_convergence_iterations):
        if iteration == 0:
            # On the first pass, the worker pod may be container-Ready but
            # not yet registered in the frontend's discovery system. Retry
            # with backoff before giving up.
            import time as _time

            instances = []
            for _attempt in range(20):
                instances = _discover_worker_instances(
                    frontend_host=frontend_host,
                    frontend_port=frontend_port,
                    dyn_namespaces=dyn_namespaces,
                    dyn_system_port=dyn_system_port,
                )
                if instances:
                    break
                _time.sleep(3.0)
            if not instances:
                raise RuntimeError(
                    f"[mx] GET http://{frontend_host}:{frontend_port}/health "
                    f"returned no update_weights_via_mx workers in namespaces="
                    f"{sorted(dyn_namespaces)} after 20 retries (60s). Verify the "
                    f"worker DGD(s) are healthy and refit_worker_namespaces is "
                    f"correct. Refit aborted."
                )
        else:
            instances = _discover_worker_instances(
                frontend_host=frontend_host,
                frontend_port=frontend_port,
                dyn_namespaces=dyn_namespaces,
                dyn_system_port=dyn_system_port,
            )

        new_instances = [i for i in instances if i["instance_id"] not in refitted_ids]
        iter_log = {
            "iteration": iteration,
            "discovered": len(instances),
            "new": len(new_instances),
            "already_refitted": len(refitted_ids),
        }
        if not new_instances:
            iteration_logs.append(iter_log)
            break  # converged: every live worker has been refitted

        # Tree fan-out: fire refits in exponentially-growing waves.
        # Wave k has up to FANOUT**k pods running in parallel. After
        # each wave, the freshly-published inference_replicas become
        # sources for the next wave; the picker random-picks among
        # them so load spreads across NICs rather than serializing
        # on the trainer. FANOUT=4 picked to match the receiver-side
        # observation that a single source NIC saturates around 4
        # concurrent NIXL pulls. Per-pod work (pause/refit/resume)
        # is unchanged — only the outer loop becomes wave-parallel.
        FANOUT = 4
        import concurrent.futures

        def _refit_one(inst: dict[str, Any]) -> tuple[Any, list[str], dict[str, Any]]:
            """One pod's refit (with retry) → flush.

            New Dynamo ``--enable-rl`` workers expose a single ``rl`` endpoint
            and require an explicit pause before weight mutation. Use
            ``mode="keep"`` so in-flight requests are held rather than aborted.
            Legacy workers that expose ``update_weights_via_mx`` directly keep
            the old no-pause path. ``flush_cache`` mirrors the direct backend's
            reset_prefix_cache, dropping prefix-cache entries computed on the
            old weights.

            Returns ``(instance_id, failure_msgs, steps)``. Designed to
            run in a worker thread inside the wave-parallel executor.
            """
            sys_url = inst["system_url"]
            inst_id = inst["instance_id"]
            requires_pause = bool(inst.get("requires_pause", False))
            steps: dict[str, Any] = {}
            failure_msgs: list[str] = []

            if requires_pause:
                r_pause = _step(
                    sys_url,
                    "pause_generation",
                    {"mode": "keep", "clear_cache": False},
                    admin_timeout_s,
                )
                steps["pause"] = r_pause
                if r_pause.get("status") != "ok":
                    failure_msgs.append(f"pause@{sys_url}({inst_id}): {r_pause}")
                    return inst_id, failure_msgs, steps

            attempt = 0
            r_refit = {"status": "error", "reason": "not attempted"}
            try:
                while True:
                    attempt += 1
                    r_refit = _step(
                        sys_url, "update_weights_via_mx", payload, refit_timeout_s
                    )
                    steps["refit"] = r_refit
                    if r_refit.get("status") == "ok":
                        break
                    if _time.monotonic() >= _cycle_deadline:
                        failure_msgs.append(
                            f"refit@{sys_url}({inst_id}) after {attempt} attempts "
                            f"(deadline exceeded): {r_refit}"
                        )
                        break
                    _time.sleep(min(3.0, 0.5 * attempt))
                steps["refit_attempts"] = attempt

                r_flush = _step(sys_url, "flush_cache", {}, admin_timeout_s)
                steps["flush"] = r_flush
                if r_flush.get("status") not in ("ok", None):
                    failure_msgs.append(f"flush@{sys_url}({inst_id}): {r_flush}")
            finally:
                if requires_pause:
                    r_resume = _step(sys_url, "resume_generation", {}, admin_timeout_s)
                    steps["resume"] = r_resume
                    if r_resume.get("status") != "ok":
                        failure_msgs.append(
                            f"resume@{sys_url}({inst_id}): {r_resume}"
                        )

            return inst_id, failure_msgs, steps

        wave_logs: list[dict[str, Any]] = []
        remaining = list(new_instances)
        wave_idx = 0
        while remaining:
            wave_idx += 1
            wave_size = min(len(remaining), FANOUT**wave_idx)
            wave = remaining[:wave_size]
            remaining = remaining[wave_size:]
            wave_start = _time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(wave)) as ex:
                futures = [ex.submit(_refit_one, inst) for inst in wave]
                for fut in concurrent.futures.as_completed(futures):
                    inst_id, fmsgs, _steps = fut.result()
                    refitted_ids.add(inst_id)
                    if fmsgs:
                        failures.extend(fmsgs)
            wave_logs.append(
                {
                    "wave": wave_idx,
                    "size": len(wave),
                    "wall_s": round(_time.monotonic() - wave_start, 3),
                }
            )

        iter_log["refitted_this_pass"] = len(new_instances)
        iter_log["waves"] = wave_logs
        iteration_logs.append(iter_log)
    else:
        # Loop hit max iterations without converging — pathological churn.
        raise RuntimeError(
            f"[mx] update_weights_via_mx(version={version}) did not converge "
            f"after {max_convergence_iterations} passes "
            f"(workers refitted: {len(refitted_ids)}). Worker pool may be "
            f"churning faster than refit can keep up. Iteration log: "
            f"{iteration_logs}"
        )

    if failures:
        raise RuntimeError(
            f"[mx] update_weights_via_mx(version={version}) refit cycle "
            f"failed: " + " | ".join(failures[:3])
        )
    return {
        "status": "ok",
        "version": version,
        "workers_refitted": len(refitted_ids),
        "iterations": iteration_logs,
    }


def _derive_frontend_url_from_dgd(dynamo_cfg: dict[str, Any]) -> str:
    """Build the cluster-internal URL of the DGD's frontend Service.

    The dynamo operator names the frontend Service ``<dgd-name>-frontend``,
    so the URL is fully determined by ``dgd_name`` + namespace + port.
    """
    dgd_name = dynamo_cfg["dgd_name"]
    namespace = dynamo_cfg.get("namespace") or read_pod_namespace()
    if not namespace:
        # Falling back to "default" is almost certainly wrong, but failing
        # outright would be over-eager — the cluster might be configured
        # without serviceaccount projection.
        warnings.warn(
            "Could not determine pod namespace; falling back to 'default'. "
            "Set policy.generation.dynamo_cfg.namespace explicitly to silence this.",
            UserWarning,
            stacklevel=3,
        )
        namespace = "default"

    port = dynamo_cfg.get("frontend_port", DEFAULT_FRONTEND_PORT)
    return f"http://{dgd_name}-frontend.{namespace}.svc.cluster.local:{port}/v1"


def _resolve_frontend_url(dynamo_cfg: dict[str, Any]) -> tuple[str, bool]:
    """Resolve the frontend URL from a DynamoCfg.

    Returns ``(url, requires_k8s)``. ``requires_k8s`` is True only on the
    ``dgd_name`` path; an explicit ``frontend_url`` opts out of the
    in-pod check so the backend works against any reachable endpoint.
    """
    if "frontend_url" in dynamo_cfg:
        url = dynamo_cfg["frontend_url"]
        if not url:
            raise RuntimeError(
                "policy.generation.dynamo_cfg.frontend_url is set but empty."
            )
        return url, False

    if "dgd_name" not in dynamo_cfg:
        raise RuntimeError(
            "DynamoGeneration requires either policy.generation.dynamo_cfg.dgd_name "
            "(the metadata.name of the DynamoGraphDeployment) or "
            "policy.generation.dynamo_cfg.frontend_url (an explicit reachable URL)."
        )
    return _derive_frontend_url_from_dgd(dynamo_cfg), True


class DynamoGeneration(GenerationInterface):
    """Forwards rollout requests to a DynamoGraphDeployment frontend.

    The DGD must already exist in the cluster — this class does not create or
    wait on it. nrl-k8s is the orchestration layer that brings up the DGD and
    waits for readiness before the training entrypoint runs.
    """

    def __init__(
        self,
        cluster: Optional[RayVirtualCluster],
        config: DynamoConfig,
        name_prefix: str = "dynamo",
        workers_per_node: Optional[Union[int, list[int]]] = None,
        tokenizer: Any | None = None,
        tokenizer_config: Optional[dict[str, Any]] = None,
    ):
        self.cfg = config
        dynamo_cfg = config.get("dynamo_cfg", {}) or {}
        url, requires_k8s = _resolve_frontend_url(dynamo_cfg)
        if requires_k8s and not is_in_kubernetes():
            raise RuntimeError(
                "DynamoGeneration with dgd_name requires running inside a "
                "Kubernetes pod (KUBERNETES_SERVICE_HOST is not set). "
                "Either run inside a pod, or set "
                "policy.generation.dynamo_cfg.frontend_url to a reachable URL."
            )
        self._dynamo_frontend_base_url = url
        self._token_wrapper_server: Optional[DynamoTokenWrapperServer] = None

        vllm_cfg = config.get("vllm_cfg") or {}
        if vllm_cfg.get("expose_http_server"):
            if tokenizer is None:
                raise RuntimeError(
                    "DynamoGeneration requires a tokenizer when exposing an "
                    "OpenAI-compatible rollout server."
                )
            tokenizer_chat_template_kwargs: Optional[dict[str, Any]] = None
            if (
                tokenizer_config is not None
                and "chat_template_kwargs" in tokenizer_config
                and tokenizer_config["chat_template_kwargs"] is not None
            ):
                chat_template_kwargs = tokenizer_config["chat_template_kwargs"]
                if not isinstance(chat_template_kwargs, dict):
                    raise RuntimeError(
                        "policy.tokenizer.chat_template_kwargs must be a dictionary."
                    )
                tokenizer_chat_template_kwargs = dict(chat_template_kwargs)

            request_timeout_s: Optional[float] = None
            if (
                "request_timeout_s" in dynamo_cfg
                and dynamo_cfg["request_timeout_s"] is not None
            ):
                request_timeout_s = float(dynamo_cfg["request_timeout_s"])
            self._token_wrapper_server = DynamoTokenWrapperServer(
                dynamo_frontend_base_url=url,
                tokenizer=tokenizer,
                tokenizer_chat_template_kwargs=tokenizer_chat_template_kwargs,
                request_timeout_s=request_timeout_s,
            )
            wrapper_url = self._token_wrapper_server.start()
            self.dp_openai_server_base_urls: list[Optional[str]] = [wrapper_url]
            print(
                "  [Dynamo] Forwarding rollout chat requests through token "
                f"wrapper {wrapper_url} -> {url}",
                flush=True,
            )
        else:
            self.dp_openai_server_base_urls = [url]
            print(f"  [Dynamo] Forwarding rollouts to {url}", flush=True)

        # --- Engine-telemetry sampler (Dynamo → nemo-rl generation_metrics/*) ---
        # Gated by vllm_cfg.enable_vllm_metrics_logger — the same flag grpo.py
        # reads to decide whether to call log_generation_metrics_to_wandb — so a
        # single recipe switch turns both the gate and this sampler on together.
        # Worker discovery needs the in-cluster /health path, so a frontend_url /
        # non-k8s construction (local tests) skips the sampler.
        self._metrics_enabled = bool(vllm_cfg.get("enable_vllm_metrics_logger", False))
        self._metrics_interval_s = float(
            vllm_cfg.get("vllm_metrics_logger_interval", 0.5) or 0.5
        )
        # Schema-agnostic collection: by default include everything and drop only
        # interpreter noise. Both lists are config-overridable (no metric names
        # are baked in, so changes to what Dynamo emits are picked up for free).
        # Default to the backend-agnostic curated allow-list (Dynamo runtime
        # metrics, see _CURATED_METRICS_INCLUDE_PREFIXES). A non-empty config list
        # overrides it; an explicit empty list ([]) opts back into scraping all.
        _inc = dynamo_cfg.get(
            "metrics_include_prefixes", _CURATED_METRICS_INCLUDE_PREFIXES
        )
        self._metrics_include_prefixes: Optional[tuple[str, ...]] = (
            tuple(_inc) if _inc else None
        )
        _exc = dynamo_cfg.get("metrics_exclude_prefixes")
        self._metrics_exclude_prefixes: tuple[str, ...] = (
            tuple(_exc) if _exc is not None else _DEFAULT_METRICS_EXCLUDE_PREFIXES
        )
        self._dyn_logger_metrics: dict[str, dict[int, list[float]]] = {}
        self._dyn_metrics_lock = threading.Lock()
        self._dyn_metrics_stop: Optional[threading.Event] = None
        self._dyn_metrics_thread: Optional[threading.Thread] = None
        self._dyn_worker_ordinals: dict[Any, int] = {}
        self._metrics_discovery_kwargs: Optional[dict[str, Any]] = None
        if self._metrics_enabled and requires_k8s and "dgd_name" in dynamo_cfg:
            self._metrics_discovery_kwargs = self._build_metrics_discovery_kwargs(
                dynamo_cfg
            )
            self._start_metrics_sampler()

    # ------------------------------------------------------------------
    # GenerationInterface — lifecycle
    # ------------------------------------------------------------------

    def prepare_for_generation(self, *args: Any, **kwargs: Any) -> bool:
        return True

    def finish_generation(self, *args: Any, **kwargs: Any) -> bool:
        return True

    # ------------------------------------------------------------------
    # Engine telemetry — scrape each DGD worker's Prometheus /metrics and
    # surface it into nemo-rl's generation_metrics/* wandb panels. The workers
    # already run Dynamo's system_status_server on DYN_SYSTEM_PORT (9090) — the
    # same server MX refit POSTs /engine/* to — which serves a combined
    # /metrics (vLLM engine stats + Dynamo-native gauges). One driver-side
    # sampler polls all workers; the pickled actor-side rollout copies never
    # sample (__getstate__ omits the thread/lock; the guards below no-op there).
    # ------------------------------------------------------------------

    def _build_metrics_discovery_kwargs(
        self, dynamo_cfg: dict[str, Any]
    ) -> dict[str, Any]:
        """Freeze the worker-discovery args (mirrors update_weights_via_mx)."""
        dgd_name = dynamo_cfg["dgd_name"]
        namespace = dynamo_cfg.get("namespace") or read_pod_namespace() or "default"
        worker_namespaces = dynamo_cfg.get("refit_worker_namespaces") or None
        if worker_namespaces:
            dyn_namespaces = set(worker_namespaces)
        else:
            dyn_namespaces = {f"{namespace}-{dgd_name}"}
        return {
            "frontend_host": f"{dgd_name}-frontend.{namespace}.svc.cluster.local",
            "frontend_port": int(
                dynamo_cfg.get("frontend_port", DEFAULT_FRONTEND_PORT)
            ),
            "dyn_namespaces": dyn_namespaces,
            "dyn_system_port": int(
                dynamo_cfg.get("dyn_system_port", DEFAULT_DYN_SYSTEM_PORT)
            ),
        }

    def _start_metrics_sampler(self) -> None:
        stop = threading.Event()
        self._dyn_metrics_stop = stop
        t = threading.Thread(
            target=self._metrics_loop, name="dynamo-metrics-sampler", daemon=True
        )
        self._dyn_metrics_thread = t
        t.start()
        print(
            "📋[Dynamo Metrics] sampler thread started "
            f"(interval={self._metrics_interval_s}s, "
            f"frontend={self._metrics_discovery_kwargs['frontend_host']})",
            flush=True,
        )

    def _metrics_ordinal(self, instance_id: Any) -> int:
        """Stable 0-based worker index for a discovery instance_id."""
        idx = self._dyn_worker_ordinals.get(instance_id)
        if idx is None:
            idx = len(self._dyn_worker_ordinals)
            self._dyn_worker_ordinals[instance_id] = idx
        return idx

    def _metrics_loop(self) -> None:
        interval = self._metrics_interval_s
        include = self._metrics_include_prefixes
        exclude = self._metrics_exclude_prefixes
        stop = self._dyn_metrics_stop
        kwargs = self._metrics_discovery_kwargs
        assert stop is not None and kwargs is not None

        stop.wait(min(2.0, interval))  # let the DGD settle before first scrape
        instances: list[dict[str, Any]] = []
        last_discover = 0.0
        while not stop.is_set():
            try:
                now = time.monotonic()
                # Re-discover at most every 5s — cheap, and picks up worker
                # restarts / autoscale without re-hitting /health every tick.
                if not instances or (now - last_discover) > 5.0:
                    instances = _discover_worker_instances(**kwargs)
                    last_discover = now
                for inst in instances:
                    text = _http_get_text(
                        f"{inst['system_url']}/metrics", timeout_s=interval + 2.0
                    )
                    if not text:
                        continue
                    found = _parse_prometheus_metrics(text, include, exclude)
                    if not found:
                        continue
                    ordinal = self._metrics_ordinal(inst["instance_id"])
                    with self._dyn_metrics_lock:
                        for name, value in found.items():
                            self._dyn_logger_metrics.setdefault(name, {}).setdefault(
                                ordinal, []
                            ).append(value)
            except Exception as exc:  # daemon telemetry: never perturb training
                print(
                    f"⚠️[Dynamo Metrics] sampler tick failed: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
            stop.wait(interval)

    def get_logger_metrics(self) -> dict[str, Any]:
        """Per-worker engine-metric timelines for generation_metrics/* panels.

        Shape matches the vLLM backend's get_logger_metrics():
        ``{metric_name: {worker_idx: [samples]}}`` — consumed by
        log_generation_metrics_to_wandb / log_plot_per_worker_timeline_metrics.
        Empty until the sampler accumulates at least one scrape.
        """
        if not getattr(self, "_metrics_enabled", False):
            return {}
        with self._dyn_metrics_lock:
            out = {
                name: {idx: list(samples) for idx, samples in per_worker.items()}
                for name, per_worker in self._dyn_logger_metrics.items()
            }
        # Surface the vLLM-backend canonical keys (print_performance_metrics
        # asserts inflight_batch_sizes/num_pending_samples; all four also give
        # panel parity), mapping from the generic scrape. Drop the raw source once
        # aliased so the same per-worker timeline isn't logged twice (canonical +
        # raw) — that duplication doubled the figure volume. Default to {} so a
        # missing/renamed source can never trip the assert or crash the loop.
        for canon, sources in _CANONICAL_LOGGER_ALIASES.items():
            if canon in out:
                continue
            src = next((s for s in sources if s in out), None)
            out[canon] = dict(out[src]) if src is not None else {}
            if src is not None:
                del out[src]
        return out

    def clear_logger_metrics(self) -> None:
        """Reset the timelines after each refit (start a fresh logging cycle)."""
        if not getattr(self, "_metrics_enabled", False):
            return
        with self._dyn_metrics_lock:
            self._dyn_logger_metrics = {}

    def shutdown(self) -> bool:
        # The DGD lifecycle is owned by Kubernetes (the dynamo operator); we
        # have nothing to tear down on the nemo-rl side — just stop the
        # telemetry sampler and token-wrapper threads if they are running.
        stop = getattr(self, "_dyn_metrics_stop", None)
        if stop is not None:
            stop.set()
        token_wrapper_server = getattr(self, "_token_wrapper_server", None)
        if token_wrapper_server is not None:
            token_wrapper_server.shutdown()
        return True

    # ------------------------------------------------------------------
    # Pickling — async rollouts ship the GenerationInterface across Ray actors
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict[str, Any]:
        return {
            "cfg": self.cfg,
            "dp_openai_server_base_urls": self.dp_openai_server_base_urls,
            "_dynamo_frontend_base_url": self._dynamo_frontend_base_url,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.cfg = state["cfg"]
        self.dp_openai_server_base_urls = state["dp_openai_server_base_urls"]
        self._dynamo_frontend_base_url = state.get(
            "_dynamo_frontend_base_url",
            self.dp_openai_server_base_urls[0],
        )
        self._token_wrapper_server = None

    def _completion_url(self) -> str:
        base_url = self._dynamo_frontend_base_url
        if not base_url:
            raise RuntimeError("DynamoGeneration does not have a frontend URL.")
        return f"{base_url.rstrip('/')}/completions"

    def _request_timeout_s(self) -> float:
        dynamo_cfg = self.cfg["dynamo_cfg"]
        if (
            "request_timeout_s" not in dynamo_cfg
            or dynamo_cfg["request_timeout_s"] is None
        ):
            raise RuntimeError(
                "DynamoGeneration direct generate() requires "
                "policy.generation.dynamo_cfg.request_timeout_s."
            )
        return float(dynamo_cfg["request_timeout_s"])

    def _merge_stop_strings(self, batch_stop_strings: Any) -> Optional[list[str]]:
        stop_set: set[str] = set()

        if self.cfg.get("stop_strings"):
            stop_set.update(self.cfg["stop_strings"])

        if batch_stop_strings is not None:
            for sample_stop_strings in batch_stop_strings:
                if not sample_stop_strings:
                    continue
                if isinstance(sample_stop_strings, str):
                    stop_set.add(sample_stop_strings)
                else:
                    stop_set.update(sample_stop_strings)

        return list(stop_set) if stop_set else None

    def _prompt_token_ids(
        self,
        data: BatchedDataDict["GenerationDatumSpec"],
        sample_idx: int,
    ) -> list[int]:
        if "vllm_content" in data:
            raise NotImplementedError(
                "DynamoGeneration direct generate() supports token-ID LLM "
                "prompts only; multimodal vllm_content is not supported."
            )

        input_length = int(data["input_lengths"][sample_idx].item())
        return data["input_ids"][sample_idx, :input_length].tolist()

    def _build_completion_request(
        self,
        *,
        prompt_token_ids: list[int],
        greedy: bool,
        stop_strings: Optional[list[str]],
        max_new_tokens: int,
    ) -> dict[str, Any]:
        top_k_cfg = self.cfg["top_k"]
        top_k_val = 1 if greedy else (top_k_cfg if top_k_cfg is not None else -1)

        payload: dict[str, Any] = {
            "model": self.cfg["model_name"],
            "prompt": prompt_token_ids,
            "max_tokens": int(max_new_tokens),
            "temperature": 0.0 if greedy else self.cfg["temperature"],
            "top_p": self.cfg["top_p"],
            "top_k": top_k_val,
            "n": 1,
            "return_tokens_as_token_ids": True,
            "include_stop_str_in_output": True,
            "nvext": {"extra_fields": ["completion_token_ids"]},
        }

        if self.cfg["stop_token_ids"] is not None:
            payload["stop_token_ids"] = self.cfg["stop_token_ids"]
        if stop_strings is not None:
            payload["stop"] = stop_strings

        return payload

    def _post_completion_request(
        self,
        *,
        prompt_token_ids: list[int],
        greedy: bool,
        stop_strings: Optional[list[str]],
        max_new_tokens: int,
    ) -> tuple[list[int], list[float], bool]:
        request_url = self._completion_url()
        payload = self._build_completion_request(
            prompt_token_ids=prompt_token_ids,
            greedy=greedy,
            stop_strings=stop_strings,
            max_new_tokens=max_new_tokens,
        )
        response = _http_post_json(request_url, payload, self._request_timeout_s())
        return _parse_dynamo_completion_response(response, request_url=request_url)

    def _single_sample_output(
        self,
        *,
        input_ids: torch.Tensor,
        input_length: int,
        generated_token_ids: list[int],
        generated_logprobs: list[float],
        truncated: bool,
    ) -> BatchedDataDict["GenerationOutputSpec"]:
        output_length = input_length + len(generated_token_ids)
        output_ids = torch.full(
            (output_length,),
            self.cfg["_pad_token_id"],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        output_ids[:input_length] = input_ids[:input_length]
        if generated_token_ids:
            output_ids[input_length:output_length] = torch.tensor(
                generated_token_ids,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )

        logprobs = torch.zeros(
            (1, output_length),
            dtype=torch.float32,
            device=input_ids.device,
        )
        for idx, logprob in enumerate(generated_logprobs[: len(generated_token_ids)]):
            logprobs[0, input_length + idx] = logprob

        return BatchedDataDict[GenerationOutputSpec](
            {
                "output_ids": output_ids.unsqueeze(0),
                "logprobs": logprobs,
                "generation_lengths": torch.tensor(
                    [len(generated_token_ids)],
                    dtype=torch.long,
                    device=input_ids.device,
                ),
                "unpadded_sequence_lengths": torch.tensor(
                    [output_length],
                    dtype=torch.long,
                    device=input_ids.device,
                ),
                "truncated": torch.tensor(
                    [truncated],
                    dtype=torch.bool,
                    device=input_ids.device,
                ),
            }
        )

    def generate(
        self,
        data: BatchedDataDict["GenerationDatumSpec"],
        greedy: bool = False,
    ) -> BatchedDataDict["GenerationOutputSpec"]:
        """Generate a batch of token-ID prompts using the DGD completions route."""
        assert isinstance(data, BatchedDataDict), (
            f"data must be a BatchedDataDict, got type: {type(data)}"
        )
        assert "input_ids" in data and "input_lengths" in data, (
            "input_ids and input_lengths are required in data for Dynamo generation"
        )

        input_ids = data["input_ids"]
        input_lengths = data["input_lengths"]
        if len(input_ids) == 0:
            return BatchedDataDict[GenerationOutputSpec](
                {
                    "output_ids": torch.zeros(
                        (0, 0), dtype=torch.long, device=input_ids.device
                    ),
                    "logprobs": torch.zeros(
                        (0, 0), dtype=torch.float, device=input_ids.device
                    ),
                    "generation_lengths": torch.zeros(
                        0, dtype=torch.long, device=input_ids.device
                    ),
                    "unpadded_sequence_lengths": torch.zeros(
                        0, dtype=torch.long, device=input_ids.device
                    ),
                    "truncated": torch.zeros(
                        0, dtype=torch.bool, device=input_ids.device
                    ),
                }
            )

        verify_right_padding(data, pad_value=self.cfg["_pad_token_id"])

        batch_stop_strings = data.get("stop_strings", [])
        stop_strings = self._merge_stop_strings(batch_stop_strings)
        padded_input_length = input_ids.size(1)

        per_sample_results = []
        max_generated_length = 0
        for sample_idx in range(input_ids.shape[0]):
            generated_token_ids, generated_logprobs, truncated = (
                self._post_completion_request(
                    prompt_token_ids=self._prompt_token_ids(data, sample_idx),
                    greedy=greedy,
                    stop_strings=stop_strings,
                    max_new_tokens=self.cfg["max_new_tokens"],
                )
            )
            per_sample_results.append(
                (generated_token_ids, generated_logprobs, truncated)
            )
            max_generated_length = max(max_generated_length, len(generated_token_ids))

        total_length = padded_input_length + max_generated_length
        output_ids_list = []
        logprobs_list = []
        generation_lengths = []
        unpadded_sequence_lengths = []
        truncated_list = []

        for sample_idx, (
            generated_token_ids,
            generated_logprobs,
            truncated,
        ) in enumerate(per_sample_results):
            input_length = int(input_lengths[sample_idx].item())
            full_output = torch.full(
                (total_length,),
                self.cfg["_pad_token_id"],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            full_output[:input_length] = input_ids[sample_idx, :input_length]
            if generated_token_ids:
                full_output[input_length : input_length + len(generated_token_ids)] = (
                    torch.tensor(
                        generated_token_ids,
                        dtype=input_ids.dtype,
                        device=input_ids.device,
                    )
                )

            full_logprobs = torch.zeros(
                total_length,
                dtype=torch.float32,
                device=input_ids.device,
            )
            for idx, logprob in enumerate(
                generated_logprobs[: len(generated_token_ids)]
            ):
                full_logprobs[input_length + idx] = logprob

            response_length = input_length + len(generated_token_ids)
            if (
                "vllm_cfg" in self.cfg
                and "max_model_len" in self.cfg["vllm_cfg"]
                and response_length > self.cfg["vllm_cfg"]["max_model_len"]
            ):
                raise AssertionError(
                    "Dynamo response length exceeded "
                    f"vllm_cfg.max_model_len: {response_length} > "
                    f"{self.cfg['vllm_cfg']['max_model_len']}"
                )

            output_ids_list.append(full_output)
            logprobs_list.append(full_logprobs)
            generation_lengths.append(len(generated_token_ids))
            unpadded_sequence_lengths.append(response_length)
            truncated_list.append(truncated)

        return BatchedDataDict[GenerationOutputSpec](
            {
                "output_ids": torch.stack(output_ids_list),
                "logprobs": torch.stack(logprobs_list),
                "generation_lengths": torch.tensor(
                    generation_lengths,
                    dtype=torch.long,
                    device=input_ids.device,
                ),
                "unpadded_sequence_lengths": torch.tensor(
                    unpadded_sequence_lengths,
                    dtype=torch.long,
                    device=input_ids.device,
                ),
                "truncated": torch.tensor(
                    truncated_list,
                    dtype=torch.bool,
                    device=input_ids.device,
                ),
            }
        )

    async def generate_async(
        self,
        data: BatchedDataDict["GenerationDatumSpec"],
        greedy: bool = False,
    ) -> AsyncGenerator[tuple[int, BatchedDataDict["GenerationOutputSpec"]], None]:
        """Generate one token-ID prompt asynchronously using the DGD route."""
        assert isinstance(data, BatchedDataDict), (
            f"data must be a BatchedDataDict, got type: {type(data)}"
        )
        assert "input_ids" in data and "input_lengths" in data, (
            "input_ids and input_lengths are required in data for Dynamo generation"
        )
        if len(data["input_ids"]) == 0:
            return

        verify_right_padding(data, pad_value=self.cfg["_pad_token_id"])

        input_ids_batch = data["input_ids"]
        input_lengths_batch = data["input_lengths"]
        batch_size = input_ids_batch.shape[0]
        assert batch_size == 1, (
            "generate_async is restricted to handle only single samples, "
            f"but received batch_size={batch_size}. Please handle batching "
            "outside this method."
        )
        if "vllm_cfg" not in self.cfg or "max_model_len" not in self.cfg["vllm_cfg"]:
            raise RuntimeError(
                "DynamoGeneration.generate_async requires "
                "policy.generation.vllm_cfg.max_model_len for vLLM-parity "
                "context budgeting."
            )

        sample_idx = 0
        input_length = int(input_lengths_batch[sample_idx].item())
        batch_stop_strings = data.get("stop_strings", [[] for _ in range(batch_size)])
        per_sample_stop_strings = None
        if batch_stop_strings and sample_idx < len(batch_stop_strings):
            per_sample_stop_strings = batch_stop_strings[sample_idx]
        final_stop_strings = self._merge_stop_strings(
            [per_sample_stop_strings] if per_sample_stop_strings else None
        )

        remaining_ctx = self.cfg["vllm_cfg"]["max_model_len"] - input_length
        allowed_new_tokens = max(0, min(self.cfg["max_new_tokens"], remaining_ctx))
        input_ids = input_ids_batch[sample_idx]
        if allowed_new_tokens == 0:
            yield (
                sample_idx,
                self._single_sample_output(
                    input_ids=input_ids,
                    input_length=input_length,
                    generated_token_ids=[],
                    generated_logprobs=[],
                    truncated=False,
                ),
            )
            return

        request_url = self._completion_url()
        payload = self._build_completion_request(
            prompt_token_ids=self._prompt_token_ids(data, sample_idx),
            greedy=greedy,
            stop_strings=final_stop_strings,
            max_new_tokens=allowed_new_tokens,
        )
        response = await asyncio.to_thread(
            _http_post_json,
            request_url,
            payload,
            self._request_timeout_s(),
        )
        generated_token_ids, generated_logprobs, truncated = (
            _parse_dynamo_completion_response(response, request_url=request_url)
        )

        yield (
            sample_idx,
            self._single_sample_output(
                input_ids=input_ids,
                input_length=input_length,
                generated_token_ids=generated_token_ids,
                generated_logprobs=generated_logprobs,
                truncated=truncated,
            ),
        )

    def init_collective(
        self, ip: str, port: int, world_size: int, **kwargs: Any
    ) -> list[ray.ObjectRef]:
        raise NotImplementedError(
            "DynamoGeneration does not support collective initialization "
            "(weight refit is not implemented in this phase)."
        )

    def prepare_refit_info(self, state_dict_info: dict[str, Any]) -> None:
        """No-op on the trainer side.

        With the receiver-side polling architecture, every DGD worker watches
        the MX server for new versions and refits itself — there's no
        trainer→worker RPC to forward state_dict_info on. If FP8 / assertion
        checks ever need this info, the worker can read it from the MX
        publisher's tensor descriptors at receive time.
        """
        return

    @property
    def requires_kv_scale_sync(self) -> bool:
        """Whether Dynamo/vLLM generation needs FP8 KV-cache scale refit."""
        vllm_cfg = self.cfg.get("vllm_cfg", None)
        if vllm_cfg is None:
            return False
        kv_cache_dtype = vllm_cfg.get("kv_cache_dtype", None)
        return kv_cache_dtype is not None and str(kv_cache_dtype).startswith("fp8")

    def update_weights_via_ipc_zmq(self) -> list[ray.ObjectRef]:
        raise NotImplementedError(
            "DynamoGeneration does not support IPC ZMQ weight sync — use "
            "weight_sync.method='mx' for non-colocated refit."
        )

    def update_weights_from_collective(self) -> list[ray.ObjectRef]:
        raise NotImplementedError(
            "DynamoGeneration does not support NCCL collective weight sync — "
            "use weight_sync.method='mx' for non-colocated refit."
        )

    # ------------------------------------------------------------------
    # ModelExpress v2 mid-training refit (cluster.weight_sync.method='mx')
    # ------------------------------------------------------------------

    def update_weights_via_mx(
        self,
        *,
        version: int,
        mx_config: Any,
    ) -> list[ray.ObjectRef]:
        """Synchronously trigger refit on every DGD VllmDecodeWorker.

        After the trainer has published the new weight version to the MX
        server (via ``policy.stream_weights_via_mx``), call the Dynamo
        Endpoint ``{namespace}.{component}.update_weights_via_mx`` on each
        worker. The handler at ``BaseWorkerHandler.update_weights_via_mx``
        fans into ``AsyncLLM.collective_rpc("update_weights_via_mx", …)``
        which runs the real NIXL receive synchronously inside every vLLM
        worker process; the endpoint streams back one JSON object whose
        ``status`` field tells us whether the receive succeeded.

        Returns a list of Ray ObjectRefs (one per worker). The caller
        ``ray.get(...)`` raises if any worker reported a non-ok status,
        producing a loud failure if the publish was invisible or the
        receive errored — replaces the silent stale-weight failure mode
        of the prior poll-only architecture.
        """
        dynamo_cfg = self.cfg.get("dynamo_cfg", {}) or {}
        dgd_name = dynamo_cfg.get("dgd_name")
        if not dgd_name:
            raise RuntimeError(
                "DynamoGeneration.update_weights_via_mx requires "
                "policy.generation.dynamo_cfg.dgd_name to identify the DGD."
            )
        namespace = dynamo_cfg.get("namespace") or read_pod_namespace() or "default"

        # Serialize MxConfig (dataclass or already-a-dict) to a JSON-safe dict.
        if isinstance(mx_config, dict):
            mx_config_dict = dict(mx_config)
        elif hasattr(mx_config, "__dataclass_fields__"):
            import dataclasses

            mx_config_dict = dataclasses.asdict(mx_config)
        else:
            mx_config_dict = {}

        # Hierarchical (GlobalRouter) deployments put the MX workers in pool
        # DGD namespaces distinct from the control DGD's frontend namespace.
        # When set, refit discovery targets these instead of {namespace}-{dgd_name}.
        worker_namespaces = dynamo_cfg.get("refit_worker_namespaces") or None
        if worker_namespaces is not None:
            worker_namespaces = list(worker_namespaces)

        ref = _dispatch_update_weights_via_mx_remote.remote(
            k8s_namespace=namespace,
            dgd_name=dgd_name,
            version=int(version),
            mx_config_dict=mx_config_dict,
            worker_namespaces=worker_namespaces,
        )
        return [ref]
