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

import warnings
from typing import Any, Optional, Union

import ray

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.generation.dynamo.config import DynamoConfig
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationInterface,
    GenerationOutputSpec,
)
from nemo_rl.utils.k8s import is_in_kubernetes, read_pod_namespace

DEFAULT_FRONTEND_PORT = 8000
DEFAULT_DYN_SYSTEM_PORT = 9090


def _http_post_json(url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
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
    except (urllib.error.HTTPError, urllib.error.URLError,
            TimeoutError, json.JSONDecodeError):
        return []
    instances = data.get("instances", []) if isinstance(data, dict) else []
    if not isinstance(instances, list):
        return []

    seen_ids: set[Any] = set()
    out: list[dict[str, Any]] = []
    # ``transport.tcp`` is ``<pod_ip>:<port>/<channel>/<endpoint>`` with no
    # scheme prefix; we just need the host part.
    tcp_re = re.compile(r"^(?:tcp://)?([^:/]+):")
    for inst in instances:
        if not isinstance(inst, dict):
            continue
        if inst.get("namespace") not in dyn_namespaces:
            continue
        # Only the vLLM worker (component "backend") serves the
        # /engine/update_weights_via_mx admin route. Filter on that endpoint
        # so we skip the LocalRouter / Planner / GlobalRouter / GlobalPlanner
        # pods that also register in these namespaces — POSTing /engine/* to
        # them resets the connection (they don't serve it). This is correct in
        # the flat topology too (the single worker registers this endpoint).
        if inst.get("endpoint") != "update_weights_via_mx":
            continue
        inst_id = inst.get("instance_id")
        if inst_id is None or inst_id in seen_ids:
            continue
        transport = inst.get("transport") or {}
        tcp = transport.get("tcp") if isinstance(transport, dict) else None
        if not isinstance(tcp, str):
            continue
        m = tcp_re.match(tcp)
        if not m:
            continue
        pod_ip = m.group(1)
        seen_ids.add(inst_id)
        out.append({
            "instance_id": inst_id,
            "system_url": f"http://{pod_ip}:{dyn_system_port}",
        })
    return out


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
         a. POST /engine/update_weights_via_mx  (real NIXL receive, blocks)
         b. POST /engine/flush_cache            (drop stale prefix cache)
         No pause/resume: update_weights_via_mx is a vLLM collective_rpc that
         runs between engine steps, pausing (not aborting) in-flight requests
         which resume on the new weights — matching NeMo-RL's direct vLLM
         backend. See _refit_one for the rationale.
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

    def _step(sys_url: str, route: str, body: dict[str, Any], timeout_s: float) -> dict[str, Any]:
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

        new_instances = [
            i for i in instances if i["instance_id"] not in refitted_ids
        ]
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

            No pause/resume around the refit: ``update_weights_via_mx`` runs as
            a vLLM ``collective_rpc``, which executes between engine steps and
            therefore *pauses* (not aborts) any in-flight requests — they resume
            on the new weights once the receive returns. This matches NeMo-RL's
            direct vLLM backend (vllm_worker.update_weights_via_mx), which calls
            the same collective RPC with no surrounding pause/abort. The old
            pause_generation step defaulted to mode="abort", which *killed*
            in-flight rollouts → empty completions → downstream crashes
            (BackendUnknown in the LocalRouter, IndexError on choices[0] in the
            gym). flush_cache is kept — it mirrors the direct backend's
            reset_prefix_cache, dropping prefix-cache entries computed on the
            old weights.

            Returns ``(instance_id, failure_msgs, steps)``. Designed to
            run in a worker thread inside the wave-parallel executor.
            """
            sys_url = inst["system_url"]
            inst_id = inst["instance_id"]
            steps: dict[str, Any] = {}
            failure_msgs: list[str] = []

            attempt = 0
            r_refit = {"status": "error", "reason": "not attempted"}
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

            return inst_id, failure_msgs, steps

        wave_logs: list[dict[str, Any]] = []
        remaining = list(new_instances)
        wave_idx = 0
        while remaining:
            wave_idx += 1
            wave_size = min(len(remaining), FANOUT ** wave_idx)
            wave = remaining[:wave_size]
            remaining = remaining[wave_size:]
            wave_start = _time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(wave)
            ) as ex:
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
        "status": "ok", "version": version,
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
        self.dp_openai_server_base_urls: list[Optional[str]] = [url]
        print(f"  [Dynamo] Forwarding rollouts to {url}", flush=True)

    # ------------------------------------------------------------------
    # GenerationInterface — lifecycle
    # ------------------------------------------------------------------

    def prepare_for_generation(self, *args: Any, **kwargs: Any) -> bool:
        return True

    def finish_generation(self, *args: Any, **kwargs: Any) -> bool:
        return True

    def shutdown(self) -> bool:
        # The DGD lifecycle is owned by Kubernetes (the dynamo operator); we
        # have nothing to tear down on the nemo-rl side.
        return True

    # ------------------------------------------------------------------
    # Pickling — async rollouts ship the GenerationInterface across Ray actors
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict[str, Any]:
        return {
            "cfg": self.cfg,
            "dp_openai_server_base_urls": self.dp_openai_server_base_urls,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.cfg = state["cfg"]
        self.dp_openai_server_base_urls = state["dp_openai_server_base_urls"]

    def generate(
        self,
        data: BatchedDataDict["GenerationDatumSpec"],
        greedy: bool = False,
    ) -> BatchedDataDict["GenerationOutputSpec"]:
        raise NotImplementedError(
            "DynamoGeneration does not support direct generate(). "
            "Use the nemo-gym HTTP rollout path instead."
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
