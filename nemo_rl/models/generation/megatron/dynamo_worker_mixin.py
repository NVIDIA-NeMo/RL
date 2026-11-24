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

import dataclasses
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from nemo_rl.distributed.virtual_cluster import _get_node_ip_local
from nemo_rl.models.generation.disagg_layout import ROLES, Role
from nemo_rl.models.generation.megatron.megatron_worker import MegatronGenerationMixin

if TYPE_CHECKING:
    from megatron.core.inference.contexts.dynamic_context import (
        DynamicInferenceContext,
    )
    from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine
    from megatron.core.inference.text_generation_controllers.text_generation_controller import (
        TextGenerationController,
    )

    from nemo_rl.models.policy import PolicyConfig


def _role_from_generation_config(generation: Mapping[str, Any]) -> Role | None:
    """The engine role ``MegatronDynamoGeneration`` wrote into ``generation``, if any.

    Raises:
        ValueError: For a role that is neither ``prefill`` nor ``decode``.
    """
    role = generation.get("_disaggregation_role")
    if role is None:
        return None
    if role not in ROLES:
        raise ValueError(
            "policy.generation._disaggregation_role must be 'prefill' or 'decode' when "
            f"set, got {role!r}; MegatronDynamoGeneration sets it per engine"
        )
    return role


def require_megatron_dynamo_integration() -> None:
    """Fail before model setup if this Megatron-LM cannot serve an engine behind Dynamo.

    The engine is only built at the first refit, after the model has loaded, so
    a missing integration would otherwise surface late. Two things are
    checked: the Dynamo integration package, and an ``InferenceConfig`` with
    ``disaggregation_role``, which every prefill/decode engine passes (a
    decode engine uses it to drop the Mamba prefix-cache budget its KV
    handoff refuses).

    Raises:
        RuntimeError: When either is missing.
    """
    # Deferred: Megatron-LM is only importable on the workers, and the Dynamo
    # integration only exists in recent Megatron-LM.
    try:
        import megatron.inference.integrations.dynamo.dynamic_engine  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "A Megatron engine behind Dynamo needs a Megatron-LM with the Dynamo "
            "integration (megatron.inference.integrations.dynamo)"
        ) from e
    from megatron.core.inference.config import InferenceConfig

    if "disaggregation_role" not in {
        field.name for field in dataclasses.fields(InferenceConfig)
    }:
        raise RuntimeError(
            "Megatron prefill/decode engines behind Dynamo need a Megatron-LM whose "
            "InferenceConfig has disaggregation_role; this one is older. Use a "
            "newer Megatron-LM or remove policy.generation.dynamo."
        )


class MegatronDynamoGenerationMixin(MegatronGenerationMixin):
    """Prefill/decode engine hooks for a Megatron worker served through Dynamo.

    Composed into ``MegatronPolicyWorkerImpl`` ahead of
    ``MegatronGenerationMixin``, so every Megatron worker carries it. A worker
    whose config has no ``generation._disaggregation_role`` (training, plain
    inference) gets the plain behaviour: every override here falls back to
    ``super()`` when the role is unset. ``MegatronDynamoGeneration`` sets the
    role per engine; it picks the state-handoff engine variant and its KV
    transfer agents.

    Any further override of a ``MegatronGenerationMixin`` method must fall back
    the same way; ``test_megatron_dynamo_worker_mixin`` lists the guarded ones.
    """

    cfg: dict[str, Any]

    @property
    def _disaggregation_role(self) -> Role | None:
        """``"prefill"`` or ``"decode"`` for an engine behind Dynamo; ``None`` otherwise."""
        return _role_from_generation_config(self.cfg["generation"])

    def _check_generation_requirements(self, config: "PolicyConfig") -> None:
        # Before the model loads, not at the first refit that builds the engine.
        if _role_from_generation_config(config["generation"]) is not None:
            require_megatron_dynamo_integration()
        super()._check_generation_requirements(config)

    def _inference_config_overrides(self) -> dict[str, Any]:
        role = self._disaggregation_role
        if role is None:
            return super()._inference_config_overrides()
        # Prefill and decode engines are separate process groups, so the
        # handoff must cross them; NIXL is the only transport that does. The
        # role sizes role-specific buffers: a decode context drops the Mamba
        # prefix cache that prefix_caching_mamba_gb gives prefill.
        return {"kv_transport_backend": "nixl", "disaggregation_role": role}

    def _build_dynamic_inference_engine(
        self,
        controller: "TextGenerationController",
        context: "DynamicInferenceContext",
    ) -> "DynamicInferenceEngine":
        role = self._disaggregation_role
        if role is None:
            return super()._build_dynamic_inference_engine(controller, context)
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
        engine.setup_kv_transfer(role=role, require_handoff_claim=True)
        return engine

    def _coordinator_kwargs(self) -> dict[str, Any]:
        if self._disaggregation_role is None:
            return super()._coordinator_kwargs()
        # The peer role's engines connect to this coordinator from other
        # processes and nodes.
        return {"hostname": _get_node_ip_local()}
