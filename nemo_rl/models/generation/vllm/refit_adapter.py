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
from torch.distributed._tensor import Shard
from torch.distributed.tensor.placement_types import Replicate

from nemo_rl.weight_sync.nccl_reshard_utils import (
    LocalParamSpec,
    RefitCtx,
    _extract_layer_prefix,
)


_NATIVE_VALUE_DTYPE = torch.float8_e4m3fn
_NATIVE_SCALE_DTYPE = torch.uint8


@dataclass(frozen=True)
class _NativeDestinationBinding:
    logical_name: str
    value_name: str
    merged_slice: tuple[slice, ...] | None
    grouped_expert_proj: str | None
    runtime_value_ptr: int
    runtime_scale_name: str
    runtime_scale_ptr: int
    checkpoint_alias_name: str


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

    def resolve_destination(
        self,
        *,
        logical_name: str,
        role: str,
    ) -> LocalParamSpec:
        """Resolve one active checkpoint-format receive destination."""
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
        self._expected_local_shapes: dict[tuple[str, str], tuple[int, ...]] = {}
        self._loaded_components: set[tuple[str, str]] = set()
        self._native_bindings: dict[str, _NativeDestinationBinding] = {}
        self._bridged_target_ids: set[int] = set()
        self._config_context: AbstractContextManager[Any] | None = None
        self._finalize_layerwise_reload: Callable[..., Any] | None = None
        self._state = "new"
        self._failure: BaseException | None = None

    def validate_plan(self, refit_info: Mapping[str, Any]) -> None:
        """Validate that the plan has one ordered identity for every component."""
        _component_keys(refit_info)
        _native_component_names(refit_info)

    def prepare(self, refit_info: Mapping[str, Any]) -> None:
        """Validate runtime aliases and capture graph-visible storage pointers."""
        self._require_not_poisoned()
        if self._state == "active":
            raise RuntimeError("cannot prepare a vLLM refit adapter during an update")
        try:
            component_metadata = _component_metadata(refit_info)
            native_names = _native_component_names(refit_info)
            component_keys = set(component_metadata)
            self._expected_components = frozenset(
                key for key in component_keys if key[0] in native_names
            )
            if not self._expected_components:
                # Task 6's lifecycle-only contract deliberately uses abbreviated
                # metadata. Keep that fake-compatible path while requiring complete
                # metadata for concrete Task 7 destination binding.
                self._expected_components = frozenset(component_keys)
            self._native_bindings = self._prepare_native_bindings(
                refit_info, native_names
            )
            self._expected_local_shapes = _expected_local_component_shapes(
                refit_info, native_names
            )
        except BaseException as error:
            self.abort_update(error)
            raise
        self._loaded_components.clear()
        self._bridged_target_ids.clear()
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
            self._bridged_target_ids.clear()
        except BaseException as error:
            self.abort_update(error)
            raise

    def resolve_destination(
        self,
        *,
        logical_name: str,
        role: str,
    ) -> LocalParamSpec:
        """Bind a native component to active vLLM checkpoint-format storage."""
        self._require_active()
        component = (logical_name, role)
        if component not in self._expected_components:
            raise ValueError(f"unexpected vLLM refit component {component!r}")
        binding = self._native_bindings.get(logical_name)
        if binding is None:
            raise ValueError(
                f"vLLM refit component {component!r} has no native destination binding"
            )
        try:
            parameters = dict(self._model_runner.model.named_parameters())
            if role == "weight":
                target_name = binding.value_name
            elif role == "weight_scale":
                target_name = self._active_checkpoint_scale_name(binding, parameters)
            else:
                raise ValueError(f"unsupported refit component role {role!r}")
            target = parameters.get(target_name)
            if target is None:
                raise ValueError(
                    f"vLLM checkpoint destination {target_name!r} for {component!r} "
                    "is missing"
                )
            region = _destination_region(target, binding.merged_slice)
            expected_dtype = (
                _NATIVE_VALUE_DTYPE if role == "weight" else _NATIVE_SCALE_DTYPE
            )
            if target.dtype != expected_dtype:
                raise ValueError(
                    f"vLLM checkpoint destination {target_name!r} for {component!r} "
                    f"has dtype {target.dtype}, expected {expected_dtype}"
                )
            expected_shape = self._expected_local_shapes.get(component)
            if expected_shape is None:
                raise ValueError(
                    f"vLLM refit component {component!r} has no destination "
                    "shape derived from Task 2 metadata"
                )
            self._validate_local_component_shape(binding, role, region, expected_shape)
            self._install_local_loader_bridge(target_name, target)
        except BaseException as error:
            self.abort_update(error)
            raise

        def pre(_base: torch.Tensor) -> RefitCtx:
            return RefitCtx(buf=torch.empty_like(region, device=self._device))

        def post(ctx: RefitCtx) -> None:
            self._load_local_component(
                binding=binding,
                role=role,
                target_name=target_name,
                target=target,
                loaded_weight=ctx.buf,
            )

        return LocalParamSpec(base=target, pre=pre, post=post)

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
            owned_weight = loaded_weight.detach().clone()
            weight_loader(target, owned_weight, **dict(loader_kwargs or {}))
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
            self._verify_runtime_bindings()
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

    def _prepare_native_bindings(
        self,
        refit_info: Mapping[str, Any],
        native_names: set[str],
    ) -> dict[str, _NativeDestinationBinding]:
        if not native_names:
            return {}
        destinations = _resolve_vllm_value_destinations(
            self._model_runner.model, refit_info
        )
        parameters = dict(self._model_runner.model.named_parameters())
        names_by_id = {id(parameter): name for name, parameter in parameters.items()}
        bindings: dict[str, _NativeDestinationBinding] = {}
        for logical_name in native_names:
            destination = destinations.get(logical_name)
            if destination is None:
                raise ValueError(
                    f"native MXFP8 parameter {logical_name!r} has no vLLM value destination"
                )
            value_param, merged_slice = destination
            value_name = names_by_id.get(id(value_param))
            if value_name is None:
                raise ValueError(
                    f"native MXFP8 value destination for {logical_name!r} is not a "
                    "registered vLLM parameter"
                )
            runtime_scale_name = f"{value_name}_scale"
            checkpoint_alias_name = f"{runtime_scale_name}_from_checkpoint"
            runtime_scale = parameters.get(runtime_scale_name)
            checkpoint_alias = parameters.get(checkpoint_alias_name)
            if runtime_scale is None:
                raise ValueError(
                    f"native MXFP8 runtime scale {runtime_scale_name!r} for "
                    f"{logical_name!r} is missing"
                )
            if checkpoint_alias is None:
                raise ValueError(
                    f"native MXFP8 checkpoint scale alias {checkpoint_alias_name!r} "
                    f"for {logical_name!r} is missing"
                )
            if runtime_scale.dtype != _NATIVE_SCALE_DTYPE:
                raise ValueError(
                    f"native MXFP8 runtime scale {runtime_scale_name!r} for "
                    f"{logical_name!r} has dtype {runtime_scale.dtype}, expected "
                    "torch.uint8"
                )
            value_region = _destination_region(value_param, merged_slice)
            scale_region = _destination_region(checkpoint_alias, merged_slice)
            _validate_checkpoint_pair(
                logical_name=logical_name,
                value_name=value_name,
                value_region=value_region,
                scale_name=checkpoint_alias_name,
                scale_region=scale_region,
            )
            for parameter_name, parameter in (
                (value_name, value_param),
                (checkpoint_alias_name, checkpoint_alias),
            ):
                if not callable(getattr(parameter, "weight_loader", None)):
                    raise ValueError(
                        f"vLLM runtime parameter {parameter_name!r} for "
                        f"{logical_name!r} has no weight_loader"
                    )
            grouped_proj = _parameter_info_by_name(refit_info)[logical_name].get(
                "grouped_expert_proj"
            )
            bindings[logical_name] = _NativeDestinationBinding(
                logical_name=logical_name,
                value_name=value_name,
                merged_slice=merged_slice,
                grouped_expert_proj=(
                    grouped_proj if isinstance(grouped_proj, str) else None
                ),
                runtime_value_ptr=value_param.data_ptr(),
                runtime_scale_name=runtime_scale_name,
                runtime_scale_ptr=runtime_scale.data_ptr(),
                checkpoint_alias_name=checkpoint_alias_name,
            )
        return bindings

    def _active_checkpoint_scale_name(
        self,
        binding: _NativeDestinationBinding,
        parameters: Mapping[str, torch.Tensor],
    ) -> str:
        candidates = (
            binding.checkpoint_alias_name,
            binding.runtime_scale_name,
        )
        present = [name for name in candidates if name in parameters]
        for name in present:
            loader = getattr(parameters[name], "weight_loader", None)
            if callable(loader) and getattr(loader, "__name__", None) == (
                "online_process_loader"
            ):
                return name
        if present:
            raise ValueError(
                f"vLLM checkpoint scale {present[0]!r} for {binding.logical_name!r} "
                "has no wrapped weight_loader"
            )
        raise ValueError(
            f"vLLM checkpoint scale for {binding.logical_name!r} is missing; "
            f"expected one of {candidates!r}"
        )

    def _validate_local_component_shape(
        self,
        binding: _NativeDestinationBinding,
        role: str,
        region: torch.Tensor,
        expected_shape: tuple[int, ...],
    ) -> None:
        if tuple(region.shape) != expected_shape:
            raise ValueError(
                f"vLLM checkpoint destination for {(binding.logical_name, role)!r} "
                f"has local shape {tuple(region.shape)}, expected {expected_shape} "
                "from Task 2 destination geometry"
            )

    def _install_local_loader_bridge(
        self,
        target_name: str,
        target: torch.Tensor,
    ) -> None:
        if id(target) in self._bridged_target_ids:
            return
        wrapped_loader = getattr(target, "weight_loader", None)
        if (
            not callable(wrapped_loader)
            or getattr(wrapped_loader, "__name__", None) != "online_process_loader"
        ):
            raise ValueError(
                f"vLLM checkpoint parameter {target_name!r} has no wrapped weight_loader"
            )
        owner_name, parameter_name = target_name.rsplit(".", 1)
        owner = dict(self._model_runner.model.named_modules()).get(owner_name)
        if owner is None or getattr(owner, parameter_name, None) is not target:
            raise ValueError(
                f"vLLM checkpoint parameter {target_name!r} has no owning module"
            )
        layerwise_module = importlib.import_module(
            "vllm.model_executor.model_loader.reload.layerwise"
        )
        make_online_process_loader = getattr(
            layerwise_module, "make_online_process_loader", None
        )
        if not _accepts_arguments(make_online_process_loader, (owner, parameter_name)):
            raise VllmRefitCompatibilityError(
                "vLLM 0.25.1 local-shard refit requires "
                "make_online_process_loader(layer, param_name)"
            )
        assert callable(make_online_process_loader)

        def local_shard_loader(
            param: torch.Tensor,
            loaded_weight: torch.Tensor,
            *,
            region: tuple[Any, ...],
            logical_name: str,
            role: str,
            loaded_shard_id: int | None = None,
            weight_name: str | None = None,
            shard_id: str | None = None,
            expert_id: int | None = None,
        ) -> None:
            del logical_name, role, loaded_shard_id, weight_name, shard_id, expert_id
            destination = param.data[region]
            if tuple(destination.shape) != tuple(loaded_weight.shape):
                raise ValueError(
                    f"local-shard loader shape mismatch for {target_name!r}: "
                    f"{tuple(loaded_weight.shape)} != {tuple(destination.shape)}"
                )
            destination.copy_(loaded_weight)

        target.weight_loader = local_shard_loader
        target.weight_loader = make_online_process_loader(owner, parameter_name)
        self._bridged_target_ids.add(id(target))

    def _load_local_component(
        self,
        *,
        binding: _NativeDestinationBinding,
        role: str,
        target_name: str,
        target: torch.Tensor,
        loaded_weight: torch.Tensor,
    ) -> None:
        component = (binding.logical_name, role)
        self._require_active()
        try:
            if component in self._loaded_components:
                raise ValueError(f"duplicate vLLM refit component {component!r}")
            weight_loader = getattr(target, "weight_loader", None)
            if not callable(weight_loader):
                raise RuntimeError(
                    f"vLLM checkpoint parameter for {component!r} has no weight_loader"
                )
            owned_weight = loaded_weight.detach().clone()
            if binding.grouped_expert_proj is not None:
                shard_id = {
                    "gate_proj": "w1",
                    "up_proj": "w3",
                    "down_proj": "w2",
                }[binding.grouped_expert_proj]
                base_region: list[Any] = list(
                    binding.merged_slice
                    or tuple(slice(None) for _ in range(target.ndim))
                )
                for expert_id in range(owned_weight.shape[0]):
                    expert_region = list(base_region)
                    expert_region[0] = expert_id
                    weight_loader(
                        target,
                        owned_weight[expert_id],
                        region=tuple(expert_region),
                        logical_name=binding.logical_name,
                        role=role,
                        weight_name=target_name.rsplit(".", 1)[-1],
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
            else:
                loader_kwargs: dict[str, Any] = {
                    "region": binding.merged_slice
                    or tuple(slice(None) for _ in range(target.ndim)),
                    "logical_name": binding.logical_name,
                    "role": role,
                }
                if binding.logical_name.endswith("gate_proj.weight"):
                    loader_kwargs["loaded_shard_id"] = 0
                elif binding.logical_name.endswith("up_proj.weight"):
                    loader_kwargs["loaded_shard_id"] = 1
                weight_loader(target, owned_weight, **loader_kwargs)
            self._loaded_components.add(component)
        except BaseException as error:
            self.abort_update(error)
            raise

    def _verify_runtime_bindings(self) -> None:
        if not self._native_bindings:
            return
        parameters = dict(self._model_runner.model.named_parameters())
        for binding in self._native_bindings.values():
            value = parameters.get(binding.value_name)
            runtime_scale = parameters.get(binding.runtime_scale_name)
            checkpoint_alias = parameters.get(binding.checkpoint_alias_name)
            if value is None or value.data_ptr() != binding.runtime_value_ptr:
                raise RuntimeError(
                    f"vLLM refit changed CUDA Graph-visible value storage for "
                    f"{binding.logical_name!r}"
                )
            if (
                runtime_scale is None
                or runtime_scale.data_ptr() != binding.runtime_scale_ptr
            ):
                raise RuntimeError(
                    f"vLLM refit changed CUDA Graph-visible scale storage for "
                    f"{binding.logical_name!r}"
                )
            if checkpoint_alias is None:
                raise RuntimeError(
                    f"vLLM refit did not restore checkpoint scale alias "
                    f"{binding.checkpoint_alias_name!r}"
                )
            _validate_checkpoint_pair(
                logical_name=binding.logical_name,
                value_name=binding.value_name,
                value_region=_destination_region(value, binding.merged_slice),
                scale_name=binding.checkpoint_alias_name,
                scale_region=_destination_region(
                    checkpoint_alias, binding.merged_slice
                ),
            )

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


def _component_metadata(
    refit_info: Mapping[str, Any],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    """Return validated component metadata keyed by logical name and role."""
    per_layer_params = refit_info.get("per_layer_params")
    layer_names = refit_info.get("layer_names")
    if not isinstance(per_layer_params, Mapping) or not isinstance(
        layer_names, Sequence
    ):
        raise ValueError(
            "vLLM refit plan must contain layer_names and per_layer_params"
        )
    component_metadata: dict[tuple[str, str], Mapping[str, Any]] = {}
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
            roles: list[str] = []
            for component in components:
                if not isinstance(component, Mapping) or not isinstance(
                    component.get("role"), str
                ):
                    raise ValueError(
                        f"vLLM refit component metadata for {logical_name!r} must contain a role"
                    )
                role = component["role"]
                if role not in ("weight", "weight_scale"):
                    raise ValueError(
                        f"{logical_name!r} has unsupported component role {role!r}"
                    )
                roles.append(role)
                component_key = (logical_name, role)
                if component_key in component_metadata:
                    raise ValueError(
                        f"vLLM refit plan has duplicate component {component_key!r}"
                    )
                component_metadata[component_key] = component
            if tuple(roles) not in (("weight",), ("weight", "weight_scale")):
                raise ValueError(
                    f"{logical_name!r} components must be ordered as "
                    "('weight', 'weight_scale')"
                )
    if not component_metadata:
        raise ValueError("vLLM refit plan must contain at least one component")
    return component_metadata


def _component_keys(refit_info: Mapping[str, Any]) -> set[tuple[str, str]]:
    """Return the unique ordered component identities from serialized refit metadata."""
    return set(_component_metadata(refit_info))


def _native_component_names(refit_info: Mapping[str, Any]) -> set[str]:
    """Validate complete native pairs and return their logical names."""
    metadata = _component_metadata(refit_info)
    param_info_by_name = _parameter_info_by_name(refit_info)
    native_names: set[str] = set()
    for logical_name, param_info in param_info_by_name.items():
        weight = metadata[(logical_name, "weight")]
        scale = metadata.get((logical_name, "weight_scale"))
        if scale is None:
            continue
        if not any(
            field in weight or field in scale for field in ("dtype", "global_shape")
        ):
            continue
        if str(weight.get("dtype")) != "torch.float8_e4m3fn":
            raise ValueError(
                f"{logical_name!r} native weight dtype must be torch.float8_e4m3fn"
            )
        if str(scale.get("dtype")) != "torch.uint8":
            raise ValueError(f"{logical_name!r} weight_scale dtype must be torch.uint8")
        weight_shape = _shape_tuple(weight.get("global_shape"), logical_name, "weight")
        scale_shape = _shape_tuple(
            scale.get("global_shape"), logical_name, "weight_scale"
        )
        if weight_shape[-1] % 32:
            raise ValueError(
                f"{logical_name!r} native weight K must be divisible by 32"
            )
        expected_scale_shape = (*weight_shape[:-1], weight_shape[-1] // 32)
        if scale_shape != expected_scale_shape:
            raise ValueError(
                f"{logical_name!r} weight_scale shape {scale_shape} must equal "
                f"the scale shape {expected_scale_shape}"
            )
        logical_shape = param_info.get("global_shape")
        if logical_shape is not None and tuple(logical_shape) != weight_shape:
            raise ValueError(
                f"{logical_name!r} native weight shape must equal its logical shape"
            )
        native_names.add(logical_name)
    return native_names


def _shape_tuple(value: Any, logical_name: str, role: str) -> tuple[int, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not value
        or any(
            not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0
            for dim in value
        )
    ):
        raise ValueError(
            f"{logical_name!r} {role} shape must contain positive integers"
        )
    return tuple(value)


def _expected_local_component_shapes(
    refit_info: Mapping[str, Any],
    native_names: set[str],
) -> dict[tuple[str, str], tuple[int, ...]]:
    """Derive local receive shapes only from Task 2 mesh metadata."""
    components = _component_metadata(refit_info)
    parameters = _parameter_info_by_name(refit_info)
    result: dict[tuple[str, str], tuple[int, ...]] = {}
    for logical_name in native_names:
        param_info = parameters[logical_name]
        mesh = param_info.get("dst_mesh_info")
        mesh_tensor = getattr(mesh, "mesh", None)
        if mesh_tensor is None:
            mesh_tensor = getattr(mesh, "_mesh", None)
        if not isinstance(mesh_tensor, torch.Tensor) or mesh_tensor.numel() == 0:
            raise ValueError(f"{logical_name!r} has no valid Task 2 destination mesh")
        mesh_shape = tuple(int(size) for size in mesh_tensor.shape)
        for role in ("weight", "weight_scale"):
            component = components[(logical_name, role)]
            global_shape = _shape_tuple(
                component.get("global_shape"), logical_name, role
            )
            placements = component.get(
                "dst_placements", param_info.get("dst_placements")
            )
            if not isinstance(placements, Sequence) or isinstance(
                placements, (str, bytes)
            ):
                raise ValueError(
                    f"{logical_name!r} {role} has no Task 2 destination placements"
                )
            if len(placements) != len(mesh_shape):
                raise ValueError(
                    f"{logical_name!r} {role} has {len(placements)} destination "
                    f"placements for a rank-{len(mesh_shape)} mesh"
                )
            shard_counts: dict[int, int] = {}
            for mesh_dim, placement in enumerate(placements):
                if isinstance(placement, Replicate):
                    continue
                if not isinstance(placement, Shard):
                    raise ValueError(
                        f"{logical_name!r} {role} has unsupported destination "
                        f"placement {placement!r}"
                    )
                if not 0 <= placement.dim < len(global_shape):
                    raise ValueError(
                        f"{logical_name!r} {role} shards dimension "
                        f"{placement.dim} outside rank {len(global_shape)}"
                    )
                shard_counts[placement.dim] = (
                    shard_counts.get(placement.dim, 1) * mesh_shape[mesh_dim]
                )
            local_shape = list(global_shape)
            for tensor_dim, shard_count in shard_counts.items():
                if local_shape[tensor_dim] % shard_count:
                    raise ValueError(
                        f"{logical_name!r} {role} global dimension "
                        f"{local_shape[tensor_dim]} is not evenly divisible by "
                        f"destination shard count {shard_count}"
                    )
                local_shape[tensor_dim] //= shard_count
            result[(logical_name, role)] = tuple(local_shape)
    return result


def _parameter_info_by_name(
    refit_info: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    per_layer_params = refit_info["per_layer_params"]
    return {
        param_info["name"]: param_info
        for layer_name in refit_info["layer_names"]
        for param_info in per_layer_params[layer_name]
    }


def _destination_region(
    parameter: torch.Tensor,
    merged_slice: tuple[slice, ...] | None,
) -> torch.Tensor:
    return parameter if merged_slice is None else parameter[merged_slice]


def _validate_checkpoint_pair(
    *,
    logical_name: str,
    value_name: str,
    value_region: torch.Tensor,
    scale_name: str,
    scale_region: torch.Tensor,
) -> None:
    if value_region.dtype != _NATIVE_VALUE_DTYPE:
        raise ValueError(
            f"vLLM MXFP8 value {value_name!r} for {logical_name!r} has dtype "
            f"{value_region.dtype}, expected {_NATIVE_VALUE_DTYPE}"
        )
    if scale_region.dtype != _NATIVE_SCALE_DTYPE:
        raise ValueError(
            f"vLLM MXFP8 scale {scale_name!r} for {logical_name!r} has dtype "
            f"{scale_region.dtype}, expected torch.uint8"
        )
    if value_region.shape[-1] % 32:
        raise ValueError(
            f"vLLM MXFP8 value {value_name!r} for {logical_name!r} must have K "
            "divisible by 32"
        )
    expected_scale_shape = (
        *value_region.shape[:-1],
        value_region.shape[-1] // 32,
    )
    if tuple(scale_region.shape) != expected_scale_shape:
        raise ValueError(
            f"vLLM MXFP8 scale {scale_name!r} for {logical_name!r} has shape "
            f"{tuple(scale_region.shape)}, expected {expected_scale_shape}"
        )


def _resolve_vllm_value_destinations(
    model: torch.nn.Module,
    refit_info: Mapping[str, Any],
) -> dict[str, tuple[torch.Tensor, tuple[slice, ...] | None]]:
    """Resolve logical FFN names to pinned-vLLM value parameters and regions."""
    vllm_params = dict(model.named_parameters())
    param_info_by_name = _parameter_info_by_name(refit_info)
    hf_shapes = {
        name: tuple(param_info["global_shape"])
        for name, param_info in param_info_by_name.items()
    }
    hf_grouped = {
        name: param_info["grouped_expert_proj"]
        for name, param_info in param_info_by_name.items()
        if param_info.get("grouped_expert_proj")
    }
    has_gate = {
        name.rsplit(".gate_proj.weight", 1)[0]
        for name, projection in hf_grouped.items()
        if projection == "gate_proj"
    }

    def layer_relative(name: str) -> str:
        prefix = _extract_layer_prefix(name)
        return name[len(prefix) + 1 :] if prefix else name

    vllm_by_relative = {layer_relative(name): name for name in vllm_params}
    vllm_by_relative_flat = {
        layer_relative(name).replace(".routed_experts.", "."): name
        for name in vllm_params
    }

    def to_vllm_name(name: str) -> str:
        if name in vllm_params:
            return name
        relative = layer_relative(name)
        matched_name = vllm_by_relative.get(relative)
        if isinstance(matched_name, str):
            return matched_name
        flattened_name = vllm_by_relative_flat.get(relative)
        return flattened_name if isinstance(flattened_name, str) else name

    result: dict[str, tuple[torch.Tensor, tuple[slice, ...] | None]] = {}
    for logical_name in hf_shapes:
        grouped_proj = hf_grouped.get(logical_name)
        if grouped_proj is not None:
            expert_prefix = logical_name.rsplit(f".{grouped_proj}.weight", 1)[0]
            value_suffix = "w2_weight" if grouped_proj == "down_proj" else "w13_weight"
            value_name = to_vllm_name(f"{expert_prefix}.{value_suffix}")
            value = vllm_params.get(value_name)
            if value is None:
                raise ValueError(
                    f"grouped expert {logical_name!r} has no vLLM target {value_name!r}"
                )
            if grouped_proj == "down_proj" or expert_prefix not in has_gate:
                result[logical_name] = (value, None)
            else:
                local_intermediate = value.shape[1] // 2
                output_slice = (
                    slice(0, local_intermediate)
                    if grouped_proj == "gate_proj"
                    else slice(local_intermediate, 2 * local_intermediate)
                )
                result[logical_name] = (
                    value,
                    (slice(None), output_slice, slice(None)),
                )
            continue

        direct_name = to_vllm_name(logical_name)
        if direct_name in vllm_params:
            result[logical_name] = (vllm_params[direct_name], None)
            continue
        if logical_name.endswith(("gate_proj.weight", "up_proj.weight")):
            is_gate = logical_name.endswith("gate_proj.weight")
            suffix = "gate_proj.weight" if is_gate else "up_proj.weight"
            prefix = logical_name[: -len(suffix)]
            value_name = to_vllm_name(f"{prefix}gate_up_proj.weight")
            value = vllm_params.get(value_name)
            if value is not None:
                if value.shape[0] % 2:
                    raise ValueError(
                        f"vLLM fused destination {value_name!r} has odd output "
                        f"dimension {value.shape[0]}"
                    )
                gate_local = value.shape[0] // 2
                up_local = value.shape[0] - gate_local
                output_slice = (
                    slice(0, gate_local)
                    if is_gate
                    else slice(gate_local, gate_local + up_local)
                )
                result[logical_name] = (value, (output_slice,))
                continue
        raise ValueError(f"no vLLM FFN destination for {logical_name!r}")
    return result


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
