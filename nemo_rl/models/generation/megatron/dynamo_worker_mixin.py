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

from typing import TYPE_CHECKING, Any

from nemo_rl.distributed.virtual_cluster import _get_node_ip_local

if TYPE_CHECKING:
    from megatron.core.inference.contexts.dynamic_context import (
        DynamicInferenceContext,
    )
    from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine
    from megatron.core.inference.text_generation_controllers.text_generation_controller import (
        TextGenerationController,
    )


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
    """

    cfg: dict[str, Any]

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
