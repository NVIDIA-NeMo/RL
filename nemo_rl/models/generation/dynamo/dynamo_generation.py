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
"""Dynamo generation and NCCL refit through external or Ray-managed Dynamo.

On Kubernetes, a ``DynamoGraphDeployment`` (DGD) owns the entire inference
stack — etcd, NATS, the dynamo frontend, and the vLLM/sglang/trtllm workers.
This class resolves the frontend for direct generation or NeMo-Gym requests
and coordinates native vLLM NCCL weight updates with the fixed worker fleet.
On Slurm/Ray, the same adapter can own a fixed Dynamo vLLM deployment.
"""

import asyncio
import logging
import time
import warnings
from typing import TYPE_CHECKING, Any, AsyncGenerator, Optional

import ray
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.generation.dynamo.config import DynamoCfg, DynamoConfig
from nemo_rl.models.generation.dynamo.token_wrapper import DynamoTokenWrapperServer
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationInterface,
    GenerationOutputSpec,
    verify_right_padding,
)
from nemo_rl.utils.k8s import is_in_kubernetes, read_pod_namespace

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from nemo_rl.models.generation.dynamo.managed_runtime import ManagedDynamoRuntime

_HTTP_MAX_ATTEMPTS = 3
_HTTP_RETRY_DELAY_S = 1.0
_RETRYABLE_HTTP_STATUS_CODES = {408, 429}


class _WorkerDiscoveryError(RuntimeError):
    """Dynamo worker discovery failed without confirming a fleet change."""


class _RetryableWorkerDiscoveryError(_WorkerDiscoveryError):
    """Dynamo worker discovery failed in a way that may be transient."""


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
        return {
            "status": "error",
            "json_decode_error": True,
            "raw": body.decode("utf-8", "replace"),
        }


def _format_dynamo_error(response: dict[str, Any]) -> str:
    """Format the internal error shape returned by ``_http_post_json``."""
    if "http_status" in response:
        return f"HTTP {response['http_status']}: {response.get('raw', '')}"
    if "transport_error" in response:
        return str(response["transport_error"])
    if "raw" in response:
        return str(response["raw"])
    return str(response)


def _is_retryable_http_response(response: Any) -> bool:
    """Return whether an internal HTTP error shape represents a transient error."""
    if not isinstance(response, dict):
        return False
    if "transport_error" in response or response.get("json_decode_error") is True:
        return True
    status = response.get("http_status")
    return isinstance(status, int) and (
        status in _RETRYABLE_HTTP_STATUS_CODES or 500 <= status < 600
    )


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
            "nvext.completion_token_ids. Ensure the Dynamo frontend is "
            "configured to return completion token IDs."
        )
    generated_token_ids = [int(token_id) for token_id in completion_token_ids]

    if not generated_token_ids:
        return (
            generated_token_ids,
            [],
            choice.get("finish_reason") == "length",
        )

    logprobs = choice.get("logprobs")
    if not isinstance(logprobs, dict):
        raise RuntimeError(
            f"Dynamo completion response from {request_url} did not include "
            "choice.logprobs."
        )
    token_logprobs = logprobs.get("token_logprobs")
    if not isinstance(token_logprobs, list):
        raise RuntimeError(
            f"Dynamo completion response from {request_url} did not include "
            "choice.logprobs.token_logprobs."
        )
    if len(token_logprobs) != len(generated_token_ids):
        raise RuntimeError(
            f"Dynamo completion response from {request_url} returned "
            f"{len(token_logprobs)} token logprobs for "
            f"{len(generated_token_ids)} generated tokens."
        )

    generated_logprobs = []
    for idx, logprob in enumerate(token_logprobs):
        if not isinstance(logprob, (int, float)) or isinstance(logprob, bool):
            raise RuntimeError(
                f"Dynamo completion response from {request_url} returned invalid "
                f"logprob {logprob!r} for generated token {idx}."
            )
        generated_logprobs.append(float(logprob))

    return (
        generated_token_ids,
        generated_logprobs,
        choice.get("finish_reason") == "length",
    )


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

    Returns a list of ``{instance_id, system_url}`` dicts. A successfully
    fetched empty fleet returns an empty list. Fetch and parse failures raise
    ``_WorkerDiscoveryError`` so they cannot be mistaken for membership loss.
    """
    import json
    import urllib.error
    import urllib.parse
    import urllib.request

    url = f"http://{frontend_host}:{frontend_port}/health"
    data: Any = None
    for attempt in range(1, _HTTP_MAX_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout_s) as resp:
                response_body = resp.read()
        except urllib.error.HTTPError as exc:
            error_type = (
                _RetryableWorkerDiscoveryError
                if exc.code in _RETRYABLE_HTTP_STATUS_CODES or 500 <= exc.code < 600
                else _WorkerDiscoveryError
            )
            error = error_type(
                f"Dynamo worker discovery request to {url} failed with HTTP {exc.code}."
            )
        except (urllib.error.URLError, TimeoutError) as exc:
            error = _RetryableWorkerDiscoveryError(
                f"Dynamo worker discovery request to {url} failed: "
                f"{type(exc).__name__}: {exc}"
            )
        else:
            try:
                data = json.loads(response_body)
            except (json.JSONDecodeError, UnicodeDecodeError):
                error = _RetryableWorkerDiscoveryError(
                    f"Dynamo worker discovery response from {url} was not valid JSON."
                )
            else:
                break

        if not isinstance(error, _RetryableWorkerDiscoveryError):
            raise error
        if attempt == _HTTP_MAX_ATTEMPTS:
            raise _WorkerDiscoveryError(
                f"Dynamo worker discovery through {url} failed after "
                f"{_HTTP_MAX_ATTEMPTS} attempts: {error}"
            ) from error
        LOGGER.warning(
            "Dynamo worker discovery attempt %d/%d failed; retrying in %.1fs: %s",
            attempt,
            _HTTP_MAX_ATTEMPTS,
            _HTTP_RETRY_DELAY_S,
            error,
        )
        time.sleep(_HTTP_RETRY_DELAY_S)

    if not isinstance(data, dict):
        raise _WorkerDiscoveryError(
            f"Dynamo worker discovery response from {url} was not a JSON object."
        )
    instances = data.get("instances", [])
    if not isinstance(instances, list):
        raise _WorkerDiscoveryError(
            f"Dynamo worker discovery response from {url} had a non-list "
            "instances field."
        )

    seen_by_id: dict[Any, dict[str, Any]] = {}
    refit_discovery_endpoints = {"rl"}
    for inst in instances:
        if not isinstance(inst, dict):
            continue
        if inst.get("namespace") not in dyn_namespaces:
            continue
        # Only the vLLM worker (component "backend") serves the /engine/*
        # admin routes. --enable-rl exposes those routes through the rl
        # endpoint advertised by the worker.
        if inst.get("component") != "backend":
            continue
        if inst.get("endpoint") not in refit_discovery_endpoints:
            continue
        inst_id = inst.get("instance_id")
        if inst_id is None:
            raise _WorkerDiscoveryError(
                "Dynamo worker discovery returned an rl backend without an instance_id."
            )
        transport = inst.get("transport") or {}
        tcp = transport.get("tcp") if isinstance(transport, dict) else None
        if not isinstance(tcp, str):
            raise _WorkerDiscoveryError(
                f"Dynamo worker {inst_id!r} did not advertise a TCP transport URL."
            )
        transport_url = tcp if "://" in tcp else f"tcp://{tcp}"
        try:
            parsed_transport = urllib.parse.urlsplit(transport_url)
            pod_ip = parsed_transport.hostname
            transport_port = parsed_transport.port
        except ValueError as exc:
            raise _WorkerDiscoveryError(
                f"Dynamo worker {inst_id!r} advertised invalid TCP transport "
                f"URL {tcp!r}."
            ) from exc
        if pod_ip is None or transport_port is None:
            raise _WorkerDiscoveryError(
                f"Dynamo worker {inst_id!r} advertised invalid TCP transport "
                f"URL {tcp!r}."
            )
        system_host = f"[{pod_ip}]" if ":" in pod_ip else pod_ip
        seen_by_id.setdefault(
            inst_id,
            {
                "instance_id": inst_id,
                "system_url": f"http://{system_host}:{dyn_system_port}",
            },
        )
    return sorted(seen_by_id.values(), key=lambda worker: str(worker["instance_id"]))


@ray.remote(num_cpus=0)
def _post_dynamo_worker_route_remote(
    *,
    system_url: str,
    route: str,
    payload: dict[str, Any],
    timeout_s: float,
) -> bool:
    """POST one Dynamo worker admin route and raise on failure."""
    response = _http_post_json(
        f"{system_url}/engine/{route}",
        payload,
        timeout_s,
    )
    if response.get("status") != "ok":
        raise RuntimeError(
            f"Dynamo worker {system_url} route {route} failed: "
            f"{_format_dynamo_error(response)}"
        )
    print(f"  [Dynamo] worker={system_url} route={route} status=ok", flush=True)
    return True


@ray.remote(num_cpus=0)
def _update_dynamo_worker_weights_remote(
    *,
    system_url: str,
    update_info: dict[str, Any],
    timeout_s: float,
) -> bool:
    """Run one native vLLM NCCL weight-update transaction."""
    common: dict[str, Any] = {
        "allow_unpaused": True,
        "reset_prefix_cache": False,
    }
    steps: tuple[tuple[str, dict[str, Any]], ...] = (
        (
            "start_weight_update",
            {"is_checkpoint_format": True},
        ),
        (
            "update_weights",
            {"update_info": update_info},
        ),
        (
            "finish_weight_update",
            {},
        ),
    )
    for engine_rpc, kwargs in steps:
        response = _http_post_json(
            f"{system_url}/engine/update_weights_from_distributed",
            {
                **common,
                "engine_rpc": engine_rpc,
                **kwargs,
            },
            timeout_s,
        )
        if response.get("status") != "ok":
            raise RuntimeError(
                f"Dynamo worker {system_url} RPC {engine_rpc} failed: "
                f"{_format_dynamo_error(response)}"
            )
        print(
            f"  [Dynamo] worker={system_url} refit_rpc={engine_rpc} status=ok",
            flush=True,
        )
    return True


def _derive_frontend_url_from_dgd(dynamo_cfg: DynamoCfg) -> str:
    """Build the cluster-internal URL of the DGD's frontend Service.

    The dynamo operator names the frontend Service ``<dgd-name>-frontend``,
    so the URL is fully determined by ``dgd_name`` + namespace + port.
    """
    dgd_name = dynamo_cfg.dgd_name
    assert dgd_name is not None
    namespace = dynamo_cfg.namespace or read_pod_namespace()
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

    return (
        f"http://{dgd_name}-frontend.{namespace}.svc.cluster.local:"
        f"{dynamo_cfg.frontend_port}/v1"
    )


def _resolve_frontend_url(dynamo_cfg: DynamoCfg) -> tuple[str, bool]:
    """Resolve the frontend URL from a DynamoCfg.

    Returns ``(url, requires_k8s)``. ``requires_k8s`` is True only on the
    ``dgd_name`` path; an explicit ``frontend_url`` opts out of the
    in-pod check so the backend works against any reachable endpoint.
    """
    if dynamo_cfg.frontend_url is not None:
        url = dynamo_cfg.frontend_url
        if not url:
            raise RuntimeError(
                "policy.generation.dynamo_cfg.frontend_url is set but empty."
            )
        return url, False

    if dynamo_cfg.dgd_name is None:
        raise RuntimeError(
            "DynamoGeneration requires either policy.generation.dynamo_cfg.dgd_name "
            "(the metadata.name of the DynamoGraphDeployment) or "
            "policy.generation.dynamo_cfg.frontend_url (an explicit reachable URL)."
        )
    return _derive_frontend_url_from_dgd(dynamo_cfg), True


class DynamoGeneration(GenerationInterface):
    """Forward rollouts to an external or Ray-managed Dynamo frontend.

    External mode assumes the DGD already exists. Ray mode owns a fixed local
    service fleet and keeps lifecycle ownership on the driver instance.
    """

    def __init__(
        self,
        cluster: Optional[RayVirtualCluster],
        config: DynamoConfig,
        tokenizer: Any | None = None,
        tokenizer_config: Optional[dict[str, Any]] = None,
    ):
        self.cfg = config
        self._dynamo_cfg = DynamoCfg.model_validate(config["dynamo_cfg"])
        dynamo_cfg = self._dynamo_cfg
        vllm_cfg = config.get("vllm_cfg") or {}
        expose_http_server = bool(vllm_cfg.get("expose_http_server"))
        tokenizer_chat_template_kwargs: Optional[dict[str, Any]] = None
        if expose_http_server:
            if tokenizer is None:
                raise RuntimeError(
                    "DynamoGeneration requires a tokenizer when exposing an "
                    "OpenAI-compatible rollout server."
                )
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
        self._managed_runtime: Optional["ManagedDynamoRuntime"] = None
        self._owns_managed_runtime = False
        if dynamo_cfg.deployment == "ray":
            from nemo_rl.models.generation.dynamo.managed_runtime import (
                ManagedDynamoRuntime,
            )

            if cluster is None:
                raise RuntimeError(
                    "Dynamo deployment='ray' requires an inference RayVirtualCluster."
                )
            self._managed_runtime = ManagedDynamoRuntime(
                cluster=cluster,
                config=config,
                dynamo_cfg=dynamo_cfg,
            )
            self._owns_managed_runtime = True
            url = self._managed_runtime.frontend_url
            requires_k8s = False
        else:
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

        if expose_http_server:
            request_timeout_s: Optional[float] = None
            if dynamo_cfg.request_timeout_s is not None:
                request_timeout_s = dynamo_cfg.request_timeout_s
            self._token_wrapper_server = DynamoTokenWrapperServer(
                dynamo_frontend_base_url=url,
                tokenizer=tokenizer,
                tokenizer_chat_template_kwargs=tokenizer_chat_template_kwargs,
                request_timeout_s=request_timeout_s,
            )
            try:
                wrapper_url = self._token_wrapper_server.start()
            except Exception:
                if self._owns_managed_runtime and self._managed_runtime is not None:
                    self._managed_runtime.shutdown()
                    self._managed_runtime = None
                    self._owns_managed_runtime = False
                self._token_wrapper_server = None
                raise
            self.dp_openai_server_base_urls: list[Optional[str]] = [wrapper_url]
            print(
                "  [Dynamo] Forwarding rollout chat requests through token "
                f"wrapper {wrapper_url} -> {url}",
                flush=True,
            )
        else:
            self.dp_openai_server_base_urls = [url]
            print(f"  [Dynamo] Forwarding rollouts to {url}", flush=True)

        self._refit_workers: Optional[list[dict[str, Any]]] = None
        self._refit_discovery_kwargs: Optional[dict[str, Any]] = None
        self._refit_update_info: Optional[dict[str, Any]] = None

    # ------------------------------------------------------------------
    # GenerationInterface — lifecycle
    # ------------------------------------------------------------------

    def prepare_for_generation(self, *args: Any, **kwargs: Any) -> bool:
        return True

    def finish_generation(self, *args: Any, **kwargs: Any) -> bool:
        return True

    def _build_worker_discovery_kwargs(self, dynamo_cfg: DynamoCfg) -> dict[str, Any]:
        """Build worker-discovery arguments for NCCL refit."""
        dgd_name = dynamo_cfg.dgd_name
        assert dgd_name is not None
        namespace = dynamo_cfg.namespace or read_pod_namespace() or "default"
        return {
            "frontend_host": f"{dgd_name}-frontend.{namespace}.svc.cluster.local",
            "frontend_port": dynamo_cfg.frontend_port,
            "dyn_namespaces": {f"{namespace}-{dgd_name}"},
            "dyn_system_port": dynamo_cfg.dyn_system_port,
        }

    def _get_refit_workers(self) -> list[dict[str, Any]]:
        """Discover and freeze the fixed Dynamo vLLM worker fleet."""
        if self._refit_workers is not None:
            return self._refit_workers

        dynamo_cfg = self._dynamo_cfg
        if dynamo_cfg.deployment == "ray":
            if self._managed_runtime is None:
                raise RuntimeError(
                    "Ray-managed Dynamo refit state is only available on the "
                    "driver instance that owns the runtime."
                )
            workers = self._managed_runtime.refit_workers()
            if not workers:
                raise RuntimeError("Ray-managed Dynamo started with no vLLM workers.")
            self._refit_workers = workers
            return workers
        if dynamo_cfg.dgd_name is None:
            raise RuntimeError(
                "Dynamo NCCL weight transfer requires "
                "policy.generation.dynamo_cfg.dgd_name."
            )
        self._refit_discovery_kwargs = self._build_worker_discovery_kwargs(dynamo_cfg)
        workers = _discover_worker_instances(**self._refit_discovery_kwargs)
        if not workers:
            raise RuntimeError(
                "No Dynamo vLLM workers advertising the rl endpoint were "
                "discovered through the DGD frontend /health response."
            )
        self._refit_workers = workers
        return workers

    def _validate_refit_workers(self) -> list[dict[str, Any]]:
        """Fail if the fixed worker fleet changed after collective setup."""
        expected = self._get_refit_workers()
        if self._dynamo_cfg.deployment == "ray":
            if self._managed_runtime is None:
                raise RuntimeError("Ray-managed Dynamo runtime is not owned here.")
            return self._managed_runtime.validate_workers(expected)
        assert self._refit_discovery_kwargs is not None
        current = _discover_worker_instances(**self._refit_discovery_kwargs)
        expected_ids = [
            (worker["instance_id"], worker["system_url"]) for worker in expected
        ]
        current_ids = [
            (worker["instance_id"], worker["system_url"]) for worker in current
        ]
        if current_ids != expected_ids:
            raise RuntimeError(
                "Dynamo worker membership changed after NCCL collective "
                f"initialization: expected={expected_ids}, current={current_ids}. "
                "Restart the training job to establish a new fixed collective."
            )
        return expected

    def get_inference_world_size(self) -> int:
        """Return the number of vLLM ranks across all discovered workers."""
        engine_world_size = self._dynamo_cfg.engine_world_size
        return len(self._get_refit_workers()) * engine_world_size

    def shutdown(self) -> bool:
        """Stop process-local wrappers and any driver-owned managed runtime."""
        token_wrapper_server = self._token_wrapper_server
        if token_wrapper_server is not None:
            token_wrapper_server.shutdown()
            self._token_wrapper_server = None
        if self._owns_managed_runtime and self._managed_runtime is not None:
            self._managed_runtime.shutdown()
            self._managed_runtime = None
            self._owns_managed_runtime = False
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
        self._dynamo_cfg = DynamoCfg.model_validate(self.cfg["dynamo_cfg"])
        self.dp_openai_server_base_urls = state["dp_openai_server_base_urls"]
        frontend_url = state.get(
            "_dynamo_frontend_base_url",
            self.dp_openai_server_base_urls[0],
        )
        if not isinstance(frontend_url, str) or not frontend_url:
            raise RuntimeError("Pickled DynamoGeneration has no frontend URL.")
        self._dynamo_frontend_base_url = frontend_url
        self._token_wrapper_server = None
        self._managed_runtime = None
        self._owns_managed_runtime = False
        # Refit state is process-local; Ray copies must rediscover the worker fleet.
        self._refit_workers = None
        self._refit_discovery_kwargs = None
        self._refit_update_info = None

    def _completion_url(self) -> str:
        base_url = self._dynamo_frontend_base_url
        if not base_url:
            raise RuntimeError("DynamoGeneration does not have a frontend URL.")
        return f"{base_url.rstrip('/')}/completions"

    def _request_timeout_s(self) -> float:
        request_timeout_s = self._dynamo_cfg.request_timeout_s
        if request_timeout_s is None:
            raise RuntimeError(
                "DynamoGeneration direct generate() requires "
                "policy.generation.dynamo_cfg.request_timeout_s."
            )
        return request_timeout_s

    def _merge_stop_strings(self, batch_stop_strings: Any) -> Optional[list[str]]:
        stop_set: set[str] = set()

        configured_stop_strings = self.cfg.get("stop_strings")
        if configured_stop_strings is not None:
            stop_set.update(configured_stop_strings)

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
            "logprobs": 0,
            "return_tokens_as_token_ids": True,
            "include_stop_str_in_output": True,
            "nvext": {"extra_fields": ["completion_token_ids"]},
        }

        if self.cfg["stop_token_ids"] is not None:
            payload["stop_token_ids"] = self.cfg["stop_token_ids"]
        if stop_strings is not None:
            payload["stop"] = stop_strings

        return payload

    def _allowed_new_tokens(self, input_length: int) -> int:
        """Return the generation budget for a prompt."""
        if "vllm_cfg" not in self.cfg or "max_model_len" not in self.cfg["vllm_cfg"]:
            return self.cfg["max_new_tokens"]

        remaining_ctx = int(self.cfg["vllm_cfg"]["max_model_len"]) - input_length
        return max(0, min(self.cfg["max_new_tokens"], remaining_ctx))

    def _assert_response_within_context(
        self, *, input_length: int, generated_length: int
    ) -> None:
        if "vllm_cfg" not in self.cfg or "max_model_len" not in self.cfg["vllm_cfg"]:
            return

        response_length = input_length + generated_length
        max_model_len = int(self.cfg["vllm_cfg"]["max_model_len"])
        if response_length > max_model_len:
            raise AssertionError(
                "Dynamo response length exceeded "
                f"vllm_cfg.max_model_len: {response_length} > {max_model_len}"
            )

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
        response: dict[str, Any] = {}
        for attempt in range(1, _HTTP_MAX_ATTEMPTS + 1):
            response = _http_post_json(request_url, payload, self._request_timeout_s())
            if not _is_retryable_http_response(response):
                break
            if attempt == _HTTP_MAX_ATTEMPTS:
                break
            LOGGER.warning(
                "Dynamo completion attempt %d/%d failed; retrying in %.1fs: %s",
                attempt,
                _HTTP_MAX_ATTEMPTS,
                _HTTP_RETRY_DELAY_S,
                _format_dynamo_error(response),
            )
            time.sleep(_HTTP_RETRY_DELAY_S)
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
        self._assert_response_within_context(
            input_length=input_length,
            generated_length=len(generated_token_ids),
        )
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
        padded_input_length = input_ids.size(1)

        per_sample_results = []
        max_generated_length = 0
        for sample_idx in range(input_ids.shape[0]):
            input_length = int(input_lengths[sample_idx].item())
            per_sample_stop_strings = None
            if batch_stop_strings and sample_idx < len(batch_stop_strings):
                per_sample_stop_strings = batch_stop_strings[sample_idx]
            stop_strings = self._merge_stop_strings(
                [per_sample_stop_strings] if per_sample_stop_strings else None
            )
            allowed_new_tokens = self._allowed_new_tokens(input_length)
            if allowed_new_tokens == 0:
                result = ([], [], False)
            else:
                result = self._post_completion_request(
                    prompt_token_ids=self._prompt_token_ids(data, sample_idx),
                    greedy=greedy,
                    stop_strings=stop_strings,
                    max_new_tokens=allowed_new_tokens,
                )
            generated_token_ids, generated_logprobs, truncated = result
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
            self._assert_response_within_context(
                input_length=input_length,
                generated_length=len(generated_token_ids),
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
        sample_idx = 0
        input_length = int(input_lengths_batch[sample_idx].item())
        batch_stop_strings = data.get("stop_strings", [[] for _ in range(batch_size)])
        per_sample_stop_strings = None
        if batch_stop_strings and sample_idx < len(batch_stop_strings):
            per_sample_stop_strings = batch_stop_strings[sample_idx]
        final_stop_strings = self._merge_stop_strings(
            [per_sample_stop_strings] if per_sample_stop_strings else None
        )

        allowed_new_tokens = self._allowed_new_tokens(input_length)
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

        generated_token_ids, generated_logprobs, truncated = await asyncio.to_thread(
            self._post_completion_request,
            prompt_token_ids=self._prompt_token_ids(data, sample_idx),
            greedy=greedy,
            stop_strings=final_stop_strings,
            max_new_tokens=allowed_new_tokens,
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
        self,
        ip: str,
        port: int,
        world_size: int,
        *,
        train_world_size: int,
    ) -> list[ray.ObjectRef]:
        """Initialize native vLLM NCCL transfer on every fixed DGD worker."""
        workers = self._get_refit_workers()
        engine_world_size = self._dynamo_cfg.engine_world_size
        inference_world_size = len(workers) * engine_world_size
        expected_world_size = train_world_size + inference_world_size
        if world_size != expected_world_size:
            raise ValueError(
                f"NCCL world_size={world_size} does not match "
                f"train_world_size={train_world_size} + Dynamo inference "
                f"world size={inference_world_size}."
            )

        rank_offsets = [
            train_world_size + worker_idx * engine_world_size
            for worker_idx in range(len(workers))
        ]
        print(
            f"  [Dynamo] initializing NCCL world_size={world_size} "
            f"train_world_size={train_world_size} "
            f"engine_world_size={engine_world_size} rank_offsets={rank_offsets}",
            flush=True,
        )

        timeout_s = self._request_timeout_s()
        return [
            _post_dynamo_worker_route_remote.remote(
                system_url=worker["system_url"],
                route="init_weights_update_group",
                payload={
                    "engine_rpc": "init_weight_transfer_engine",
                    "init_info": {
                        "master_address": ip,
                        "master_port": port,
                        "rank_offset": train_world_size
                        + worker_idx * engine_world_size,
                        "world_size": world_size,
                    },
                },
                timeout_s=timeout_s,
            )
            for worker_idx, worker in enumerate(workers)
        ]

    def prepare_refit_info(self, state_dict_info: dict[str, Any]) -> None:
        """Serialize checkpoint-format tensor metadata for native vLLM refit."""
        if state_dict_info is None:
            raise ValueError("state_dict_info must not be None for Dynamo refit.")

        names: list[str] = []
        dtype_names: list[str] = []
        shapes: list[list[int]] = []
        for name, (shape, dtype) in state_dict_info.items():
            names.append(name)
            dtype_names.append(str(dtype).removeprefix("torch."))
            shapes.append(list(shape))
        self._refit_update_info = {
            "names": names,
            "dtype_names": dtype_names,
            "shapes": shapes,
            "packed": True,
        }

    def update_weights_via_ipc_zmq(self) -> list[ray.ObjectRef]:
        raise NotImplementedError(
            "DynamoGeneration only supports NCCL weight transfer."
        )

    def update_weights_from_collective(self) -> list[ray.ObjectRef]:
        """Receive packed checkpoint-format weights on every Dynamo worker."""
        if self._refit_update_info is None:
            raise RuntimeError(
                "prepare_refit_info() must be called before Dynamo weight updates."
            )
        workers = self._validate_refit_workers()
        timeout_s = self._request_timeout_s()
        return [
            _update_dynamo_worker_weights_remote.remote(
                system_url=worker["system_url"],
                update_info=self._refit_update_info,
                timeout_s=timeout_s,
            )
            for worker in workers
        ]

    def invalidate_kv_cache(self) -> bool:
        """Flush every fixed Dynamo worker's prefix/KV cache."""
        workers = self._validate_refit_workers()
        futures = [
            _post_dynamo_worker_route_remote.remote(
                system_url=worker["system_url"],
                route="flush_cache",
                payload={},
                timeout_s=self._request_timeout_s(),
            )
            for worker in workers
        ]
        return all(ray.get(futures))
