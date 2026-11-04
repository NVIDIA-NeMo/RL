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
"""Worker-side hooks for a Megatron engine served through Dynamo."""

import asyncio
import os
import uuid
from concurrent.futures import Future as ConcurrentFuture
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional

import torch

from nemo_rl.distributed.virtual_cluster import (
    DEFAULT_DYNAMO_SYSTEM_PORT_RANGE_HIGH,
    DEFAULT_DYNAMO_SYSTEM_PORT_RANGE_LOW,
    _get_free_port_local,
    _get_node_ip_local,
)

if TYPE_CHECKING:
    from megatron.core.inference.contexts.dynamic_context import (
        DynamicInferenceContext,
    )
    from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine
    from megatron.core.inference.text_generation_controllers.text_generation_controller import (
        TextGenerationController,
    )
    from megatron.inference.integrations.dynamo.telemetry import EngineEventReporter


def require_megatron_dynamo_integration() -> None:
    """Fail before model setup if this Megatron-LM cannot serve an engine behind Dynamo.

    The engine is only built at the first refit, after the model has loaded, so
    a missing integration would otherwise surface late.
    """
    try:
        import megatron.inference.integrations.dynamo.dynamic_engine  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "A Megatron engine behind Dynamo needs a Megatron-LM with the Dynamo "
            "integration (megatron.inference.integrations.dynamo)"
        ) from e


class MegatronDynamoGenerationMixin:
    """Prefill/decode engine hooks for a Megatron worker served through Dynamo.

    Hosted by ``MegatronDynamoPolicyWorker``, the worker class the policy picks
    when ``generation._dynamo_role`` is set, so every override here applies
    unconditionally; a worker without a role is a plain ``MegatronPolicyWorker``
    and never sees this class. Compose it before ``MegatronGenerationMixin`` so
    the overrides win. ``MegatronDynamoGeneration`` sets the role per engine;
    it picks the state-handoff engine variant and its KV transfer agents.

    Borrowed from ``MegatronGenerationMixin``:

     - cfg: policy config (TypedDict).
     - rank: global rank (used for logging).
     - dynamic_inference_engine: the engine, once the first refit built it.
     - coordinator_addr: the engine's coordinator address once it listens.
     - _inference_loop: the engine's asyncio loop.
     - _inference_engine_initialized: whether the engine is built.
    """

    cfg: dict[str, Any]
    rank: int
    dynamic_inference_engine: Optional["DynamicInferenceEngine"]
    coordinator_addr: Optional[str]
    _inference_loop: Optional[asyncio.AbstractEventLoop]
    _inference_engine_initialized: bool

    @property
    def _dynamo_role(self) -> str:
        """``"prefill"`` or ``"decode"``: which half of the handoff this engine is."""
        role = self.cfg["generation"].get("_dynamo_role")
        if role is None:
            raise RuntimeError(
                "MegatronDynamoPolicyWorker needs policy.generation._dynamo_role; "
                "MegatronDynamoGeneration sets it per engine"
            )
        return role

    def _init_inference_engine_state(self) -> None:
        super()._init_inference_engine_state()
        # Set by attach_dynamo_events once the engine serves through a Megatron
        # Dynamo adapter.
        self._dynamo_event_reporter: Optional["EngineEventReporter"] = None
        # Rank zero only: the task serving this engine's Dynamo endpoint, once
        # start_dynamo_endpoint scheduled it on the inference loop.
        self._dynamo_endpoint_future: Optional[ConcurrentFuture] = None
        # What start_dynamo_endpoint returned; kept so a repeat call reads the
        # same ports and handoff owner.
        self._dynamo_endpoint_metadata: Optional[dict[str, Any]] = None

    def _inference_config_overrides(self) -> dict[str, Any]:
        # Prefill and decode engines are separate process groups, so the
        # handoff must cross them; NIXL is the only transport that does.
        return {"kv_transport_backend": "nixl"}

    def _build_dynamic_inference_engine(
        self,
        controller: "TextGenerationController",
        context: "DynamicInferenceContext",
    ) -> "DynamicInferenceEngine":
        # Prefill/decode behind Dynamo: the state-handoff engine pins prefill
        # blocks until decode releases them and imports them on decode.
        # Deferred: the Dynamo integration is optional (absent from older
        # Megatron-LM) and only Dynamo-attached engines load it.
        from megatron.inference.integrations.dynamo.dynamic_engine import (
            DynamoDynamicInferenceEngine,
        )

        engine = DynamoDynamicInferenceEngine(controller, context)
        # Collective over the engine's model-parallel ranks. The Dynamo adapter
        # claims every handoff before decode reads it, so unclaimed prefill
        # offers may expire instead of staying pinned forever.
        engine.setup_kv_transfer(role=self._dynamo_role, require_handoff_claim=True)
        return engine

    def _coordinator_kwargs(self) -> dict[str, Any]:
        # The peer role's engines connect to this coordinator from other
        # processes and nodes.
        return {"hostname": _get_node_ip_local()}

    def attach_dynamo_events(self, *, parent_event_address: str) -> bool:
        """Send this engine's events to its Megatron Dynamo adapter.

        Runs on every rank once the engine and its coordinator are up, after
        rank zero started the adapter, the way the vLLM worker registers an
        engine that already serves. Every rank reports progress; rank zero also
        forwards KV events and sends "ready" with the coordinator address, which
        is what lets the adapter register the engine with the frontend. A second
        call is a no-op.

        Args:
            parent_event_address: ZMQ address the adapter listens on for
                engine events.

        Returns:
            True, so the fan-out can be awaited uniformly.
        """
        if not self._inference_engine_initialized:
            raise RuntimeError(
                "attach_dynamo_events needs the inference engine to be built "
                "(the first refit builds it)"
            )
        if self._dynamo_event_reporter is not None:
            return True

        async def _attach() -> None:
            # On the inference loop: the engine fires its progress and KV-event
            # callbacks there.
            self._start_dynamo_event_reporter(parent_event_address)
            self._report_dynamo_ready()

        asyncio.run_coroutine_threadsafe(_attach(), self._inference_loop).result()
        return True

    def start_dynamo_endpoint(
        self,
        *,
        namespace: str,
        manager_env: Mapping[str, str],
        model_name: str,
        served_model_name: str,
        endpoint_types: list[str],
        engine_start_timeout_s: float,
        excluded_ports: list[int],
    ) -> dict[str, Any]:
        """Host Megatron-LM's Dynamo adapter for this engine on this rank.

        Runs on rank zero once the engine is built (the first refit builds it),
        as the vLLM worker's ``start_dynamo_endpoint_async`` runs on an engine
        that already serves. The adapter (``MegatronLLMEngine``, an
        implementation of Dynamo's ``LLMEngine``) is driven by a Dynamo
        ``Worker`` task on the inference loop. Unlike ``NativeVllmEngine`` it
        does not hold the engine: in external launch mode it binds an event
        socket, waits for the "ready" message ``attach_dynamo_events`` sends,
        connects an ``InferenceClient`` to the coordinator, and only then
        registers the engine with the frontend. The engine's role comes from
        its config. A second call returns the first call's metadata.

        Args:
            namespace: Dynamo namespace shared with the frontend.
            manager_env: etcd / NATS connection variables from the managed runtime.
            model_name: Model the engine serves; also what the frontend lists.
            served_model_name: Name the frontend serves it under.
            endpoint_types: ``["completions"]`` or ``["chat", "completions"]``.
            engine_start_timeout_s: How long the adapter waits for the engine's
                ready message.
            excluded_ports: Ports earlier engines on this node took.

        Returns:
            ``node_ip``, ``event_port``, ``system_port``, ``parent_event_address``
            (what every rank's event reporter connects to) and ``handoff_owner``
            (decode only: the identity prefill pins KV blocks for).
        """
        role = self._dynamo_role
        if self._dynamo_endpoint_metadata is not None:
            return dict(self._dynamo_endpoint_metadata)
        if not self._inference_engine_initialized:
            raise RuntimeError(
                "start_dynamo_endpoint needs the inference engine to be built "
                "(the first refit builds it)"
            )
        # Deferred: ai-dynamo and the Megatron-LM Dynamo integration are
        # imported only by engines that serve through the frontend.
        from dynamo.common.backend import Worker
        from dynamo.common.model_fetch import fetch_model
        from megatron.inference.integrations.dynamo.args import Config
        from megatron.inference.integrations.dynamo.llm_engine import (
            MegatronLLMEngine,
        )

        node_ip = _get_node_ip_local()
        used = set(excluded_ports)
        event_port = _get_free_port_local(
            DEFAULT_DYNAMO_SYSTEM_PORT_RANGE_LOW,
            DEFAULT_DYNAMO_SYSTEM_PORT_RANGE_HIGH,
            max_retries=None,
            excluded_ports=used,
        )
        used.add(event_port)
        system_port = _get_free_port_local(
            DEFAULT_DYNAMO_SYSTEM_PORT_RANGE_LOW,
            DEFAULT_DYNAMO_SYSTEM_PORT_RANGE_HIGH,
            max_retries=None,
            excluded_ports=used,
        )
        # A fresh owner per launch: prefill pins KV blocks per decode owner.
        handoff_owner = uuid.uuid4().hex if role == "decode" else None
        # The Rust runtime reads these when the first Worker is constructed.
        os.environ.update({**manager_env, "DYN_SYSTEM_PORT": str(system_port)})
        config = Config(
            model=model_name,
            served_model_name=served_model_name,
            namespace=namespace,
            component="prefill" if role == "prefill" else "backend",
            endpoint="generate",
            discovery_backend=manager_env.get("DYN_DISCOVERY_BACKEND", "etcd"),
            request_plane=manager_env.get("DYN_REQUEST_PLANE", "tcp"),
            event_plane=manager_env.get("DYN_EVENT_PLANE", "nats"),
            role=role,
            nproc_per_node=None,
            coordinator_host=node_ip,
            coordinator_port=None,
            worker_id_file=None,
            # Local-mode launch settings; unused in external mode.
            megatron_root="",
            drain_timeout=30.0,
            megatron_argv=[],
            engine_launch_mode="external",
            engine_start_timeout=float(engine_start_timeout_s),
            parent_event_host=node_ip,
            parent_event_port=event_port,
            endpoint_types=",".join(endpoint_types),
            handoff_owner=handoff_owner,
        )

        async def _serve() -> None:
            # As the adapter's own entry point does: a local path registers as
            # itself, a hub id as a metadata-only snapshot, never a second copy
            # of the weights.
            model_path = Path(model_name)
            registration_model = (
                str(model_path.resolve())
                if model_path.exists()
                else str(await fetch_model(model_name, ignore_weights=True))
            )
            engine = MegatronLLMEngine(config, registration_model)
            worker_config = MegatronLLMEngine.worker_config(config, registration_model)
            await Worker(engine, worker_config).run()

        self._dynamo_endpoint_future = asyncio.run_coroutine_threadsafe(
            _serve(), self._inference_loop
        )
        print(
            f"[Rank {self.rank}] Dynamo {role} endpoint started: events on "
            f"tcp://{node_ip}:{event_port}, system port {system_port}",
            flush=True,
        )
        self._dynamo_endpoint_metadata = {
            "node_ip": node_ip,
            "event_port": event_port,
            "system_port": system_port,
            "parent_event_address": f"tcp://{node_ip}:{event_port}",
            "handoff_owner": handoff_owner,
        }
        return dict(self._dynamo_endpoint_metadata)

    def stop_dynamo_endpoint(self) -> None:
        """Cancel the Dynamo endpoint task; the engine leaves discovery."""
        future = self._dynamo_endpoint_future
        self._dynamo_endpoint_future = None
        self._dynamo_endpoint_metadata = None
        if future is not None:
            future.cancel()

    def _start_dynamo_event_reporter(self, parent_event_address: str) -> None:
        """Connect the engine's progress and KV events to the adapter."""
        # Deferred: the Dynamo integration is optional (absent from older
        # Megatron-LM) and only Dynamo-attached engines load it.
        from megatron.inference.integrations.dynamo.telemetry import (
            EngineEventReporter,
        )

        reporter = EngineEventReporter(
            self.dynamic_inference_engine, parent_event_address
        )
        self.dynamic_inference_engine.set_progress_callback(
            lambda: reporter.observe("progress", {})
        )
        reporter.start()
        self._dynamo_event_reporter = reporter

    def _report_dynamo_ready(self) -> None:
        """Rank zero tells the adapter where the coordinator is and what it serves."""
        reporter = self._dynamo_event_reporter
        if reporter is None or torch.distributed.get_rank() != 0:
            return
        if self.coordinator_addr is None:
            raise RuntimeError("The inference coordinator is not listening yet")
        # Deferred: only Dynamo-attached engines use it, and it ships only with
        # the Megatron-LM versions that carry the Dynamo integration.
        from megatron.core.inference.engine_endpoint import InferenceEngineEndpoint

        endpoint = InferenceEngineEndpoint.from_engine(
            self.coordinator_addr,
            self.dynamic_inference_engine,
            logical_data_parallel_size=1,
        )
        reporter.observe("ready", endpoint.to_dict())
