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

"""ModelOpt KV compatibility for the pinned native vLLM loader.

The native loader owns scale parameters, persistent storage, and packed KV
layouts. These configs only fill its KV-only / flat-NVFP4 metadata gaps.
"""

from collections.abc import Callable, Mapping
from contextlib import contextmanager
from typing import Any, Iterator

import torch
from vllm.model_executor.layers.attention import Attention, MLAAttention
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptFp8Config,
    ModelOptKVCacheMethod,
    ModelOptMixedPrecisionConfig,
    ModelOptMxFp8Config,
    ModelOptNvFp4Config,
    ModelOptQuantConfigBase,
)

from nemo_rl.modelopt.utils import modelopt_kv_cache_format


class ModelOptKVConfig(ModelOptQuantConfigBase):
    """Native KV scales with unquantized weights; no fictitious weight format."""

    KVCacheMethodCls = ModelOptKVCacheMethod

    def __init__(self, kv_format: str) -> None:
        # vLLM's optional base accepts exclude_modules; Pyrefly infers object's
        # constructor here. Native config tests exercise this exact signature.
        super().__init__(["*"])  # pyrefly: ignore[bad-argument-count]
        self.kv_cache_quant_method = kv_format

    def get_name(self) -> str:
        return "modelopt_kv"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 89

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ModelOptKVConfig":
        kv_format = modelopt_kv_cache_format(config)
        if config.get("quant_algo") or kv_format is None:
            raise ValueError(
                "ModelOpt KV-only loading requires KV metadata and no weight quant_algo"
            )
        return cls(kv_format)

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg: dict[str, Any], user_quant: str | None, hf_config: Any = None
    ) -> str | None:
        if (
            hf_quant_cfg.get("quant_method", "").startswith("modelopt")
            and not hf_quant_cfg.get("quant_algo")
            and hf_quant_cfg.get("kv_cache_scheme") is not None
        ):
            modelopt_kv_cache_format(hf_quant_cfg)
            return "modelopt_kv"
        return None

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> Any:
        if isinstance(layer, MLAAttention):
            raise NotImplementedError(
                "ModelOpt KV refit does not yet support MLA attention"
            )
        return super().get_quant_method(layer, prefix)


class _KVMetadataMixin(ModelOptQuantConfigBase):
    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Any:
        result = super().from_config(config)
        if (
            "quantization" not in config
            and (kv_format := modelopt_kv_cache_format(config)) is not None
        ):
            # vLLM 0.29 reads a flat FP8 dictionary, but drops ModelOpt's NVFP4
            # string. In particular, mixed-precision configs then skip creating
            # attention scale parameters altogether.
            if hasattr(result, "kv_cache_quant_method"):
                result.kv_cache_quant_method = kv_format
            if hasattr(result, "kv_cache_quant_algo"):
                result.kv_cache_quant_algo = kv_format
        return result


class _FP8KVConfig(_KVMetadataMixin, ModelOptFp8Config):
    pass


class _NVFP4KVConfig(_KVMetadataMixin, ModelOptNvFp4Config):
    pass


class _MXFP8KVConfig(_KVMetadataMixin, ModelOptMxFp8Config):
    pass


class _MixedKVConfig(_KVMetadataMixin, ModelOptMixedPrecisionConfig):
    pass


def register_modelopt_kv_configs() -> None:
    """Register configs before engine construction and in native worker processes."""
    for name, config in (
        ("modelopt_kv", ModelOptKVConfig),
        ("modelopt", _FP8KVConfig),
        ("modelopt_fp4", _NVFP4KVConfig),
        ("modelopt_mxfp8", _MXFP8KVConfig),
        ("modelopt_mixed", _MixedKVConfig),
    ):
        register_quantization_config(name)(config)


def _sync_refit_status(
    model_runner: Any, error: Exception | None, changed: bool = False
) -> bool:
    if torch.distributed.is_initialized():
        from vllm.distributed import get_world_group

        # A PP stage can own different (or no) attention layers. Rank-local
        # failures and graph invalidation must not split subsequent collectives.
        flags = torch.tensor(
            (int(error is not None), int(changed)),
            dtype=torch.int32,
            device=model_runner.device,
        )
        torch.distributed.all_reduce(
            flags,
            op=torch.distributed.ReduceOp.MAX,
            group=get_world_group().device_group,
        )
        if flags[0].item() and error is None:
            error = RuntimeError(
                "ModelOpt attention scale refit validation failed on another rollout rank"
            )
        changed = bool(flags[1].item())
    if error is not None:
        raise error
    return changed


@contextmanager
def refit_kv_scales(
    model_runner: Any,
    state_dict_info: Mapping[str, Any],
    kv_format: str,
    *,
    on_scales_changed: Callable[[], None] | None = None,
) -> Iterator[None]:
    """Validate scale coverage and refresh native graphs only when scales change.

    Called with rollout drained, around the complete native reload operation.
    vLLM owns attention copy-back; do not register a second set of scale buffers.
    """
    model = model_runner.model
    if hasattr(model, "unwrap"):
        model = model.unwrap()
    layers = {
        name: layer
        for name, layer in model.named_modules()
        if isinstance(layer, Attention)
    }
    error = None
    try:
        dtype_prefix = "fp8" if kv_format == "FP8" else "nvfp4"
        if any(
            not layer.kv_cache_dtype.startswith(dtype_prefix)
            for layer in layers.values()
        ):
            raise ValueError(
                "Native KV-cache dtype does not match the policy's ModelOpt metadata"
            )
        mapper = model_runner.vllm_config.quant_config.get_cache_scale_mapper()
        if (hf_mapper := getattr(model, "hf_to_vllm_mapper", None)) is not None:
            mapper = hf_mapper | mapper
        mapped_names = set(mapper.apply_list(list(state_dict_info)))
        required = {
            f"{name}.{axis}_scale" for name in layers for axis in ("q", "k", "v")
        }
        if missing := required - mapped_names:
            raise ValueError(
                f"ModelOpt KV refit is missing attention scales: {sorted(missing)}"
            )
        buffers = [
            getattr(layer, f"_{axis}_scale")
            for layer in layers.values()
            for axis in ("q", "k", "v")
        ]
        original_pointers = [value.data_ptr() for value in buffers]
        before = (
            torch.stack([value.detach().reshape(()) for value in buffers]).cpu()
            if buffers
            else torch.empty(0)
        )
    except Exception as exception:
        error = exception
    _sync_refit_status(model_runner, error)
    error = None
    try:
        yield
    except Exception as exception:
        error = exception
    changed = False
    try:
        if error is not None:
            raise error
        buffers = [
            getattr(layer, f"_{axis}_scale")
            for layer in layers.values()
            for axis in ("q", "k", "v")
        ]
        if [value.data_ptr() for value in buffers] != original_pointers:
            raise RuntimeError(
                "Native attention refit did not preserve scale buffer storage"
            )
        after = (
            torch.stack([value.detach().reshape(()) for value in buffers]).cpu()
            if buffers
            else torch.empty(0)
        )
        if not bool(torch.all(torch.isfinite(after) & (after > 0))):
            raise ValueError("Refitted attention scales must be finite and positive")
        changed = not torch.equal(before, after)
    except Exception as exception:
        error = exception
    changed = _sync_refit_status(model_runner, error, changed)
    if not changed:
        return

    for layer in layers.values():
        # FlashInfer caches these Python launch coefficients on its first
        # forward. Tensor copy-back alone cannot refresh them or captured args.
        for name in ("bmm1_scale", "bmm2_scale"):
            if hasattr(layer.impl, name):
                setattr(layer.impl, name, None)
    if on_scales_changed is not None:
        on_scales_changed()
    else:
        recapture_kv_graphs(model_runner)


def recapture_kv_graphs(model_runner: Any) -> None:
    """Recapture native graphs after changed scales, with weights and KV awake."""
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper
    from vllm.compilation.cuda_graph import CUDAGraphWrapper
    from vllm.config import CUDAGraphMode, set_current_vllm_config

    if model_runner.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
        error = None
        retired_graphs: list[Any] = []
        try:
            torch.accelerator.synchronize()
            # Keep graph-pool owners alive until replacement capture completes.
            # Live output buffers alone cannot prevent the allocator's zero-
            # use-count assertion when reusing a pool after its last graph dies.
            retired_graphs.extend(
                entry.cudagraph
                for wrapper in list(CUDAGraphWrapper._all_instances)
                for entry in wrapper.concrete_cudagraph_entries.values()
            )
            retired_graphs.extend(
                entry.capture
                for wrapper in list(BreakableCUDAGraphWrapper._all_instances)
                for entry in wrapper.entries.values()
            )
            manager = getattr(model_runner, "cudagraph_manager", None)
            if manager is not None:
                retired_graphs.extend(manager.graphs.values())
            CUDAGraphWrapper.clear_all_graphs()
            BreakableCUDAGraphWrapper.clear_all_graphs()
            if manager is not None:
                # v0.29's V2 runner owns FULL graphs outside those registries
                # and exposes no reset API. Preserve its pool/output buffers.
                manager.graphs.clear()
                manager._graphs_captured = False
            # capture_model explicitly enables then disables capture. Clearing
            # graphs without recapturing leaves runtime capture disabled in v0.29.
            with set_current_vllm_config(model_runner.vllm_config):
                model_runner.capture_model()
        except Exception as exception:
            error = exception
        finally:
            retired_graphs.clear()
        _sync_refit_status(model_runner, error)
