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

"""vLLM-specific implementation of the version-neutral refit lifecycle."""

import importlib
import inspect
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch


@dataclass(frozen=True)
class VllmRefitCapabilities:
    """vLLM APIs relevant to refit adapter selection and diagnostics."""

    layerwise_reload: bool
    weight_transfer_engine_registry: bool
    trainer_weight_transfer: bool


@runtime_checkable
class VllmRefitAdapter(Protocol):
    """Lifecycle contract used by component-aware generation refit."""

    def validate_plan(self, refit_info: Mapping[str, Any]) -> None:
        """Validate the component plan before any model mutation."""
        ...

    def prepare(self, refit_info: Mapping[str, Any]) -> None:
        """Index the component plan without importing vLLM reload internals."""
        ...

    def begin_update(self) -> None:
        """Restore checkpoint-format storage and enable wrapped weight loaders."""
        ...

    def load_component(
        self,
        *,
        logical_name: str,
        role: str,
        target: torch.Tensor,
        loaded_weight: torch.Tensor,
        loader_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        """Load one received component through its current checkpoint loader."""
        ...

    def finish_update(self) -> None:
        """Publish a complete update through vLLM's finalizer."""
        ...

    def abort_update(self, error: BaseException) -> None:
        """Fail closed after an incomplete or failed update."""
        ...


@runtime_checkable
class _VllmModelRunner(Protocol):
    model: torch.nn.Module
    vllm_config: object


class Vllm0251RefitAdapter:
    """Pinned-vLLM lifecycle adapter using layerwise checkpoint reload.

    This adapter supports the vLLM 0.25.1 contract pinned by NeMo-RL. Later
    APIs are reported by :func:`probe_vllm_refit_capabilities` only; they are
    not selected as a runtime implementation here.
    """

    def __init__(
        self,
        *,
        model_runner: _VllmModelRunner,
        model_config: object,
        device: torch.device,
    ) -> None:
        self._model_runner = model_runner
        self._model_config = model_config
        self._device = device
        self._expected_components: frozenset[tuple[str, str]] = frozenset()
        self._loaded_components: set[tuple[str, str]] = set()
        self._config_context: AbstractContextManager[Any] | None = None
        self._finalize_layerwise_reload: Callable[..., Any] | None = None
        self._state = "new"
        self._failure: BaseException | None = None

    def validate_plan(self, refit_info: Mapping[str, Any]) -> None:
        """Validate that the plan has one ordered identity for every component."""
        _component_keys(refit_info)

    def prepare(self, refit_info: Mapping[str, Any]) -> None:
        """Index expected component loads without touching the vLLM model."""
        self._require_not_poisoned()
        if self._state == "active":
            raise RuntimeError("cannot prepare a vLLM refit adapter during an update")
        try:
            self._expected_components = frozenset(_component_keys(refit_info))
        except BaseException as error:
            self.abort_update(error)
            raise
        self._loaded_components.clear()
        self._state = "prepared"

    def begin_update(self) -> None:
        """Enter vLLM's layerwise restore/load/process/discard lifecycle."""
        self._require_not_poisoned()
        if self._state != "prepared":
            raise RuntimeError(
                "vLLM refit adapter must be prepared before begin_update"
            )
        try:
            config_module = importlib.import_module("vllm.config")
            reload_module = importlib.import_module(
                "vllm.model_executor.model_loader.reload"
            )
            set_current_vllm_config = getattr(config_module, "set_current_vllm_config")
            initialize_layerwise_reload = getattr(
                reload_module, "initialize_layerwise_reload"
            )
            finalize_layerwise_reload = getattr(
                reload_module, "finalize_layerwise_reload"
            )
            if not callable(set_current_vllm_config):
                raise VllmRefitCompatibilityError(
                    "vLLM is missing set_current_vllm_config for layerwise refit"
                )
            if not _accepts_arguments(initialize_layerwise_reload, (object(),)):
                raise VllmRefitCompatibilityError(
                    "vLLM is missing initialize_layerwise_reload(model)"
                )
            if not _accepts_arguments(finalize_layerwise_reload, (object(), object())):
                raise VllmRefitCompatibilityError(
                    "vLLM is missing finalize_layerwise_reload(model, model_config)"
                )
            config_context = set_current_vllm_config(self._model_runner.vllm_config)
            if not isinstance(config_context, AbstractContextManager):
                raise VllmRefitCompatibilityError(
                    "vLLM set_current_vllm_config did not return a context manager"
                )
            self._config_context = config_context
            self._state = "active"
            config_context.__enter__()
            with torch.device(self._device):
                initialize_layerwise_reload(self._model_runner.model)
            self._finalize_layerwise_reload = finalize_layerwise_reload
        except BaseException as error:
            self.abort_update(error)
            raise

    def load_component(
        self,
        *,
        logical_name: str,
        role: str,
        target: torch.Tensor,
        loaded_weight: torch.Tensor,
        loader_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        """Forward a received component to vLLM's active wrapped loader."""
        self._require_active()
        component = (logical_name, role)
        try:
            if component not in self._expected_components:
                raise ValueError(f"unexpected vLLM refit component {component!r}")
            if component in self._loaded_components:
                raise ValueError(f"duplicate vLLM refit component {component!r}")
            weight_loader = getattr(target, "weight_loader", None)
            if not callable(weight_loader):
                raise RuntimeError(
                    f"vLLM checkpoint parameter for {component!r} has no weight_loader"
                )
            weight_loader(target, loaded_weight, **dict(loader_kwargs or {}))
            self._loaded_components.add(component)
        except BaseException as error:
            self.abort_update(error)
            raise

    def finish_update(self) -> None:
        """Finalize exactly one complete layerwise update and leave its context."""
        self._require_active()
        missing_components = self._expected_components - self._loaded_components
        if missing_components:
            error = RuntimeError(
                "vLLM refit cannot finalize with missing component loads: "
                f"{sorted(missing_components)!r}"
            )
            self.abort_update(error)
            raise error
        try:
            finalize_layerwise_reload = self._finalize_layerwise_reload
            if finalize_layerwise_reload is None:
                raise RuntimeError("vLLM refit adapter has no active finalizer")
            with torch.device(self._device):
                finalize_layerwise_reload(self._model_runner.model, self._model_config)
        except BaseException as error:
            self.abort_update(error)
            raise
        try:
            self._exit_config_context()
        except BaseException as error:
            self.abort_update(error)
            raise
        self._loaded_components.clear()
        self._state = "prepared"

    def abort_update(self, error: BaseException) -> None:
        """Close an active context without finalizing incomplete vLLM storage."""
        self._failure = error
        self._state = "poisoned"
        try:
            self._exit_config_context(error)
        except BaseException:
            pass

    def _require_not_poisoned(self) -> None:
        if self._failure is not None:
            raise RuntimeError(
                "The vLLM worker is unusable after a failed native layerwise refit"
            ) from self._failure

    def _require_active(self) -> None:
        self._require_not_poisoned()
        if self._state != "active":
            raise RuntimeError("vLLM refit adapter has no active update")

    def _exit_config_context(self, error: BaseException | None = None) -> None:
        config_context = self._config_context
        self._config_context = None
        if config_context is not None:
            config_context.__exit__(
                type(error) if error is not None else None,
                error,
                error.__traceback__ if error is not None else None,
            )


class VllmRefitCompatibilityError(RuntimeError):
    """Raised when installed vLLM APIs cannot satisfy the selected adapter."""


def create_vllm_refit_adapter(
    *,
    model_runner: _VllmModelRunner,
    model_config: object,
    device: torch.device,
) -> VllmRefitAdapter:
    """Create the pinned adapter using APIs rather than a vLLM version string."""
    capabilities = probe_vllm_refit_capabilities()
    if not capabilities.layerwise_reload:
        raise VllmRefitCompatibilityError(
            "vLLM does not expose the required layerwise reload APIs for native refit"
        )
    return Vllm0251RefitAdapter(
        model_runner=model_runner,
        model_config=model_config,
        device=device,
    )


def probe_vllm_refit_capabilities() -> VllmRefitCapabilities:
    """Probe installed APIs without importing or parsing vLLM version metadata."""
    try:
        config_module = importlib.import_module("vllm.config")
        reload_module = importlib.import_module(
            "vllm.model_executor.model_loader.reload"
        )
    except ModuleNotFoundError:
        layerwise_reload = False
    else:
        layerwise_reload = (
            callable(getattr(config_module, "set_current_vllm_config", None))
            and _accepts_arguments(
                getattr(reload_module, "initialize_layerwise_reload", None), (object(),)
            )
            and _accepts_arguments(
                getattr(reload_module, "finalize_layerwise_reload", None),
                (object(), object()),
            )
        )
    return VllmRefitCapabilities(
        layerwise_reload=layerwise_reload,
        weight_transfer_engine_registry=_has_weight_transfer_engine_registry(),
        trainer_weight_transfer=_has_trainer_weight_transfer(),
    )


def _component_keys(refit_info: Mapping[str, Any]) -> set[tuple[str, str]]:
    """Return the unique ordered component identities from serialized refit metadata."""
    per_layer_params = refit_info.get("per_layer_params")
    layer_names = refit_info.get("layer_names")
    if not isinstance(per_layer_params, Mapping) or not isinstance(
        layer_names, Sequence
    ):
        raise ValueError(
            "vLLM refit plan must contain layer_names and per_layer_params"
        )
    component_keys: set[tuple[str, str]] = set()
    for layer_name in layer_names:
        params = per_layer_params.get(layer_name)
        if not isinstance(params, Sequence):
            raise ValueError(
                f"vLLM refit plan has no parameter list for {layer_name!r}"
            )
        for param_info in params:
            if not isinstance(param_info, Mapping) or not isinstance(
                param_info.get("name"), str
            ):
                raise ValueError("vLLM refit parameter metadata must contain a name")
            logical_name = param_info["name"]
            components = param_info.get("components", ({"role": "weight"},))
            if not isinstance(components, Sequence) or isinstance(
                components, (str, bytes)
            ):
                raise ValueError(
                    f"vLLM refit components for {logical_name!r} must be a sequence"
                )
            for component in components:
                if not isinstance(component, Mapping) or not isinstance(
                    component.get("role"), str
                ):
                    raise ValueError(
                        f"vLLM refit component metadata for {logical_name!r} must contain a role"
                    )
                component_key = (logical_name, component["role"])
                if component_key in component_keys:
                    raise ValueError(
                        f"vLLM refit plan has duplicate component {component_key!r}"
                    )
                component_keys.add(component_key)
    if not component_keys:
        raise ValueError("vLLM refit plan must contain at least one component")
    return component_keys


def _has_weight_transfer_engine_registry() -> bool:
    """Return whether a later vLLM exposes both custom engine registries."""
    try:
        factory_module = importlib.import_module(
            "vllm.distributed.weight_transfer.factory"
        )
    except ModuleNotFoundError:
        return False
    return all(
        _accepts_one_argument_shape(
            getattr(
                getattr(factory_module, factory_name, None), "register_engine", None
            ),
            ((object(), object()), (object(), object(), object())),
        )
        for factory_name in (
            "WeightTransferEngineFactory",
            "WeightTransferTrainerFactory",
        )
    )


def _has_trainer_weight_transfer() -> bool:
    """Return whether a later vLLM exposes worker and trainer transfer methods."""
    try:
        base_module = importlib.import_module("vllm.distributed.weight_transfer.base")
    except ModuleNotFoundError:
        return False
    worker_engine = getattr(base_module, "WeightTransferEngine", None)
    trainer_engine = getattr(base_module, "TrainerWeightTransferEngine", None)
    return (
        all(
            callable(getattr(worker_engine, method_name, None))
            for method_name in (
                "start_weight_update",
                "update_weights",
                "finish_weight_update",
            )
        )
        and _accepts_arguments(
            getattr(trainer_engine, "trainer_init", None),
            (object(),),
            keyword_arguments={"client": object()},
        )
        and _has_keyword_only_parameter(
            getattr(trainer_engine, "trainer_init", None), "client"
        )
        and callable(getattr(trainer_engine, "send_weights", None))
    )


def _accepts_one_argument_shape(
    callable_object: Callable[..., Any] | None,
    argument_shapes: Sequence[tuple[object, ...]],
) -> bool:
    """Return whether a callable accepts one of the documented argument shapes."""
    return any(
        _accepts_arguments(callable_object, arguments) for arguments in argument_shapes
    )


def _accepts_arguments(
    callable_object: Callable[..., Any] | None,
    arguments: tuple[object, ...],
    *,
    keyword_arguments: Mapping[str, object] | None = None,
) -> bool:
    """Return whether a callable can bind the positional arguments used by refit."""
    if not callable(callable_object):
        return False
    try:
        inspect.signature(callable_object).bind(
            *arguments, **dict(keyword_arguments or {})
        )
    except (TypeError, ValueError):
        return False
    return True


def _has_keyword_only_parameter(
    callable_object: Callable[..., Any] | None,
    parameter_name: str,
) -> bool:
    """Return whether a callable exposes the named keyword-only parameter."""
    if not callable(callable_object):
        return False
    try:
        parameter = inspect.signature(callable_object).parameters.get(parameter_name)
    except (TypeError, ValueError):
        return False
    return parameter is not None and parameter.kind is inspect.Parameter.KEYWORD_ONLY
