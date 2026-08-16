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
import gc
import logging
import re
import socket
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any, Literal

import torch
import zmq

from nemo_rl.models.generation.vllm.checkpoint_engine import (
    VllmCheckpointEngineMixin,
    preinit_nixl_from_vllm_config,
    resolve_rollout_rank,
)
from nemo_rl.models.policy.utils import (
    IPCProtocol,
    calculate_aligned_size,
    rebuild_cuda_tensor_from_ipc,
)
from nemo_rl.utils.nsys import wrap_with_nvtx_name
from nemo_rl.utils.packed_tensor import packed_broadcast_consumer
from nemo_rl.weight_sync.nccl_reshard_utils import (
    _STR_TO_DTYPE,
    HFToLocalParamMap,
    LocalParamSpec,
    RefitCtx,
    _extract_layer_prefix,
)

logger = logging.getLogger(__name__)

try:
    import vllm  # noqa: F401
    from vllm.distributed.parallel_state import get_pp_group
    from vllm.v1.worker.gpu_worker import Worker as VllmWorker
except ImportError:
    raise ImportError(
        "vLLM is not installed. Please check that the py_executable in the runtime_env of VllmGenerationWorker "
        "covers the vllm dependency. You may have to update nemo_rl/distributed/ray_actor_environment_registry.py. "
        "This error can also happen if the venv creation was aborted or errored out in the middle. In that case, "
        "please run at least once with the environment variable NRL_FORCE_REBUILD_VENVS=true set to force the rebuild of the environment."
    )


WeightUpdateTransport = Literal[
    "ipc", "collective", "checkpoint_engine", "nccl_reshard"
]
UnsupportedNativeRefitTransport = Literal["checkpoint_engine", "sparse_delta"]
WeightUpdateFinalizer = Callable[[], None]


def _format_refit_key_error(label: str, keys: set[str]) -> str:
    """Format a bounded refit-key diagnostic."""
    ordered = sorted(keys)
    suffix = " ..." if len(ordered) > 8 else ""
    return f"{label} ({len(ordered)}): {ordered[:8]}{suffix}"


def _native_mxfp8_param_names(refit_info: dict[str, Any]) -> set[str]:
    """Return logical parameters carrying canonical value and scale components."""
    result: set[str] = set()
    for layer_name in refit_info.get("layer_names", []):
        for param_info in refit_info.get("per_layer_params", {}).get(layer_name, []):
            components = param_info.get("components", [])
            if [component.get("role") for component in components] == [
                "weight",
                "weight_scale",
            ]:
                result.add(param_info["name"])
    return result


class IPCWeightManifestError(RuntimeError):
    """An IPC transfer did not match the prepared state-dict manifest."""


def _detach_pending_layerwise_weights(
    model: torch.nn.Module, source_storage_ptrs: set[int]
) -> None:
    """Clone deferred reload weights that still alias a transport buffer.

    vLLM 0.25.1 replays deferred weight-loader arguments during layerwise
    finalization, after NeMo-RL may have reused the source transport buffer.
    """
    if not source_storage_ptrs:
        return

    # Keep reload internals off the normal non-layerwise weight-loading path.
    from vllm.model_executor.model_loader.reload.layerwise import get_layerwise_info

    for module in model.modules():
        info = get_layerwise_info(module)
        for _, arguments in info.loaded_weights:
            loaded_weight = arguments.arguments.get("loaded_weight")
            if not isinstance(loaded_weight, torch.Tensor):
                continue
            if loaded_weight.untyped_storage().data_ptr() in source_storage_ptrs:
                arguments.arguments["loaded_weight"] = loaded_weight.clone()


def _refresh_hpc_modules_after_layerwise_reload(model: torch.nn.Module) -> None:
    """Rebuild kernel-specific state omitted by vLLM's layerwise finalizer.

    ``HpcModule`` implementations derive runtime state from loaded weights via
    ``process_weights_after_loading``. vLLM 0.25.1 runs that model-wide pass on
    normal loads, but not after a layerwise reload.
    """
    from vllm.model_executor.layers.hpc import HpcModule

    for module in model.modules():
        if isinstance(module, HpcModule):
            module.process_weights_after_loading(model)


def _unquantized_flashinfer_trtllm_modules(
    model: torch.nn.Module,
) -> list[torch.nn.Module]:
    """Return modules that realized the unquantized TRTLLM MoE backend."""
    # Import backend types only when inspecting a constructed vLLM model. The
    # module layout is version-sensitive; on vLLM builds without the oracle
    # package the TRTLLM backend cannot be realized, so absence means no
    # matches rather than an ImportError on every unquantized-model refit.
    try:
        from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
            UnquantizedMoeBackend,
        )
        from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
            UnquantizedFusedMoEMethod,
        )
    except ImportError:
        return []

    return [
        module
        for module in model.modules()
        if isinstance(
            quant_method := getattr(module, "quant_method", None),
            UnquantizedFusedMoEMethod,
        )
        and quant_method.unquantized_backend is UnquantizedMoeBackend.FLASHINFER_TRTLLM
    ]


def _reload_target_module_ids(
    reload_targets: Sequence[torch.nn.Module],
) -> set[int]:
    """Return every module covered by the native reload targets."""
    return {id(module) for target in reload_targets for module in target.modules()}


def _process_mxfp8_modules_after_native_reload(
    model: torch.nn.Module, reloaded_module_ids: set[int]
) -> None:
    """Rebuild MXFP8 layouts outside the partial native reload."""
    try:
        from vllm.model_executor.layers.quantization.modelopt import (
            ModelOptMxFp8FusedMoE,
            ModelOptMxFp8LinearMethod,
        )
    except ImportError:
        return

    mxfp8_methods = (ModelOptMxFp8FusedMoE, ModelOptMxFp8LinearMethod)
    for module in model.modules():
        if id(module) in reloaded_module_ids:
            continue
        quant_method = getattr(module, "quant_method", None)
        if isinstance(quant_method, mxfp8_methods):
            quant_method.process_weights_after_loading(module)


def _model_uses_unquantized_flashinfer_trtllm(model: torch.nn.Module) -> bool:
    """Return whether a model realized the unquantized TRTLLM MoE backend."""
    return bool(_unquantized_flashinfer_trtllm_modules(model))


def _convert_bf16_moe_weights_to_trtllm_block_layout_batched(
    cache_permute_indices: dict[torch.Size, torch.Tensor],
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    is_gated_act_gemm: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert all BF16 experts to TRTLLM block layout in two gathers."""
    if w13_weight.dtype != torch.bfloat16 or w2_weight.dtype != torch.bfloat16:
        raise ValueError(
            "Unquantized MoE backend FlashInfer TRTLLM requires bfloat16 weights"
        )
    if w13_weight.ndim != 3 or w2_weight.ndim != 3:
        raise ValueError(
            "TRTLLM BF16 MoE weights must have shape [experts, rows, cols]"
        )
    if w13_weight.shape[0] != w2_weight.shape[0]:
        raise ValueError("W13 and W2 must contain the same number of experts")

    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        get_w2_permute_indices_with_cache,
    )

    epilogue_tile_m = 128
    block_k = 128
    w13_expert_uint8 = w13_weight[0].view(torch.uint8)
    w2_expert_uint8 = w2_weight[0].view(torch.uint8)
    w13_permute_indices = _maybe_get_cached_w3_w1_permute_indices(
        cache_permute_indices,
        w13_expert_uint8,
        epilogue_tile_m,
        is_gated_act_gemm=is_gated_act_gemm,
    )
    if is_gated_act_gemm:
        rows = w13_expert_uint8.shape[0]
        w13_permute_indices = (w13_permute_indices + rows // 2) % rows
    w2_permute_indices = get_w2_permute_indices_with_cache(
        cache_permute_indices,
        w2_expert_uint8,
        epilogue_tile_m,
    )

    def _convert(weight: torch.Tensor, source_indices: torch.Tensor) -> torch.Tensor:
        weight_uint8 = weight.view(torch.uint8)
        num_experts, rows, byte_cols = weight_uint8.shape
        if byte_cols % block_k != 0:
            raise ValueError(
                f"TRTLLM BF16 MoE byte columns must be divisible by {block_k}; "
                f"got {byte_cols}"
            )
        expert_blocks = weight_uint8.view(
            num_experts, rows, byte_cols // block_k, block_k
        ).permute(0, 2, 1, 3)
        return (
            torch.index_select(
                expert_blocks,
                2,
                source_indices.to(weight.device),
            )
            .contiguous()
            .view(torch.bfloat16)
        )

    return (
        _convert(w13_weight, w13_permute_indices),
        _convert(w2_weight, w2_permute_indices),
    )


@contextmanager
def _use_batched_bf16_trtllm_layout_conversion() -> Iterator[None]:
    """Use the batched converter only while vLLM rebuilds TRTLLM MoE state."""
    from vllm.model_executor.layers.fused_moe.oracle import unquantized

    original_converter = (
        unquantized.convert_moe_weights_to_flashinfer_trtllm_block_layout
    )
    unquantized.convert_moe_weights_to_flashinfer_trtllm_block_layout = (
        _convert_bf16_moe_weights_to_trtllm_block_layout_batched
    )
    try:
        yield
    finally:
        unquantized.convert_moe_weights_to_flashinfer_trtllm_block_layout = (
            original_converter
        )


def _local_shard_slices(param_info: dict[str, Any], rank: int) -> tuple[slice, ...]:
    """Return this destination rank's slices in an HF-global tensor."""
    from nemo_rl.weight_sync.xferdtensor_python import _compute_shard_slices

    dst_mesh = param_info["dst_mesh_info"]
    mesh_tensor = getattr(dst_mesh, "mesh", None)
    if mesh_tensor is None:
        mesh_tensor = getattr(dst_mesh, "_mesh", None)
    if mesh_tensor is None:
        raise ValueError("Destination DeviceMesh does not expose mesh ranks.")
    coordinates = (mesh_tensor == rank).nonzero(as_tuple=False)
    if coordinates.numel() == 0:
        raise ValueError(f"Rank {rank} is absent from the destination mesh")
    return tuple(
        _compute_shard_slices(
            param_info["global_shape"],
            list(mesh_tensor.shape),
            coordinates[0].tolist(),
            param_info["dst_placements"],
        )
    )


class _IPCWeightManifest:
    """Validate an IPC stream against its prepared state-dict manifest."""

    def __init__(self, expected_keys: Iterable[str]) -> None:
        self.expected_keys = set(expected_keys)
        self.loaded_keys: set[str] = set()
        self.errors: list[str] = []

    def validate_batch(self, keys: Sequence[str]) -> set[str] | None:
        batch_keys: set[str] = set()
        duplicate_keys: set[str] = set()
        for key in keys:
            if key in batch_keys:
                duplicate_keys.add(key)
            batch_keys.add(key)
        duplicate_keys.update(self.loaded_keys & batch_keys)
        unexpected_keys = batch_keys - self.expected_keys
        if duplicate_keys:
            self.errors.append(
                _format_refit_key_error("duplicate keys", duplicate_keys)
            )
        if unexpected_keys:
            self.errors.append(
                _format_refit_key_error("unexpected keys", unexpected_keys)
            )
        return None if self.errors else batch_keys

    def record_loaded(self, keys: set[str]) -> None:
        self.loaded_keys.update(keys)

    def record_load_failure(self, error: Exception) -> None:
        message = f"{type(error).__name__}: {error}"
        if len(message) > 512:
            message = message[:512] + " ..."
        self.errors.append(f"weight load failed: {message}")

    def require_complete(self) -> None:
        details = list(self.errors)
        missing_keys = self.expected_keys - self.loaded_keys
        if missing_keys:
            details.append(_format_refit_key_error("missing keys", missing_keys))
        if details:
            raise IPCWeightManifestError("; ".join(details))


class NixlVllmWorker(VllmWorker):
    """vLLM worker that establishes NIXL/UCX before vLLM initialization."""

    def __new__(cls, vllm_config: Any, *args: Any, **kwargs: Any) -> "NixlVllmWorker":
        worker = super().__new__(cls)
        worker._nrl_nixl_preinit_agent = preinit_nixl_from_vllm_config(vllm_config)
        return worker


def fix_gemma3_vision_weight_name(key: str) -> str:
    """Re-insert the `vision_model` segment into Gemma3 vision-tower weights.

    When performing refit, the vision-tower weight paths are flattened. This unflattens them.
    """
    return re.sub(
        r"vision_tower\.(?!vision_model\.)", "vision_tower.vision_model.", key
    )


def _read_mtp_layer_weights_from_checkpoint(
    model_path: str, mtp_layer_indices: set[int]
) -> list[tuple[str, torch.Tensor]]:
    """Read only the MTP draft layer weights from a sharded HF safetensors checkpoint.

    Uses the checkpoint's ``model.safetensors.index.json`` to open only the
    shards that contain the requested transformer layer indices, so the
    multi-terabyte base-model weights are never read from disk.

    Args:
        model_path: Path to the HF checkpoint directory.
        mtp_layer_indices: Transformer layer indices belonging to the MTP module(s).

    Returns:
        A list of ``(weight_name, tensor)`` pairs for the requested layers, with
        tensors on CPU.
    """
    import json
    import os

    from safetensors import safe_open

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    layer_re = re.compile(r"(?:^|\.)layers\.(\d+)\.")
    shard_to_names: dict[str, list[str]] = {}
    for name, shard in weight_map.items():
        match = layer_re.search(name)
        is_mtp_namespace = "mtp" in name.split(".")
        is_trailing_mtp_layer = (
            match is not None and int(match.group(1)) in mtp_layer_indices
        )
        if is_mtp_namespace or is_trailing_mtp_layer:
            shard_to_names.setdefault(shard, []).append(name)

    weights: list[tuple[str, torch.Tensor]] = []
    for shard, names in shard_to_names.items():
        with safe_open(
            os.path.join(model_path, shard), framework="pt", device="cpu"
        ) as reader:
            for name in names:
                weights.append((name, reader.get_tensor(name)))
    return weights


class VllmInternalWorkerExtension:
    # Per-PP-stage refit groups, None until init_nccl_reshard_comm_group builds them.
    # Declared rather than sprung into existence so a rebuild can release the previous
    # ones without probing, matching AbstractPolicyWorker.model_update_group. None and
    # not {} because a mutable class-level default is shared by every instance.
    pp_comm_groups: Optional[dict[int, Any]] = None
    # False for a checkpoint-loaded static MTP drafter; True only when the
    # trainer exports MTP weights in every policy refit stream.
    _mtp_drafter_weights_from_refit: bool = True
    _sparse_delta_applier: Any = None
    _nrl_named_parameters: dict[str, torch.nn.Parameter]
    _nrl_layerwise_reload_active: bool = False
    # Initialization detaches parameters, so any later failure leaves this
    # worker unsafe to reuse. Keep the original failure for the worker lifetime.
    _nrl_layerwise_reload_failure: Exception | None = None
    _nccl_reshard_refit_adapter: Any | None = None
    nccl_reshard_refit_info: dict[str, Any]
    hf_to_local_param_map: HFToLocalParamMap

    def _get_named_parameters(self) -> dict[str, torch.nn.Parameter]:
        params = getattr(self, "_nrl_named_parameters", None)
        if params is None:
            params = dict(self.model_runner.model.named_parameters())
            self._nrl_named_parameters = params
        return params

    def _load_full_hf_weights(
        self, policy_weights: list[tuple[str, torch.Tensor]]
    ) -> None:
        """Load HF weights and detach any deferred reload tensors from transport storage."""
        if not getattr(self, "_nrl_layerwise_reload_active", False):
            self.model_runner.model.load_weights(weights=policy_weights)
            return

        source_storage_ptrs = {
            tensor.untyped_storage().data_ptr() for _, tensor in policy_weights
        }
        load_error: Exception | None = None
        try:
            self.model_runner.model.load_weights(weights=policy_weights)
        except Exception as error:
            load_error = error
            raise
        finally:
            try:
                # Native layerwise reload may defer weight_loader calls until all
                # shards for a layer have arrived. NCCL/IPC weights are views into
                # reusable receive buffers that subsequent transfers may overwrite,
                # so clone any pending tensors that alias the current buffer and
                # preserve them until deferred replay consumes them.
                _detach_pending_layerwise_weights(
                    self.model_runner.model, source_storage_ptrs
                )
            except Exception:
                if load_error is None:
                    raise
                logger.exception(
                    "Failed to detach deferred weights after a weight load failure"
                )

    def _load_hf_weights(self, policy_weights: list[tuple[str, torch.Tensor]]) -> None:
        from nemo_rl.models.generation.vllm.quantization import fp8

        if fp8.is_fp8_model(self.model_runner.vllm_config):
            fp8.load_weights(
                policy_weights,
                self.model_runner,
                model_load_weights=self._load_full_hf_weights,
            )
            return
        self._load_full_hf_weights(policy_weights)

    def bind_numa(self) -> bool:
        """Pin this TP worker to its GPU's NUMA-local CPUs/memory.

        Invoked via ``collective_rpc`` on each vLLM TP worker once the engine
        (and CUDA) is up, so the worker's physical GPU id is resolved from its
        local device index (see ``resolve_visible_gpu_id``).
        """
        import torch

        from nemo_rl.distributed.numa_utils import (
            bind_to_gpu_numa,
            resolve_visible_gpu_id,
        )

        gpu_id = resolve_visible_gpu_id(torch.cuda.current_device())
        if gpu_id is None:
            return False
        return bind_to_gpu_numa(gpu_id)

    def init_collective(
        self,
        rank_prefix: int,
        ip: str,
        port: int,
        world_size: int,
        train_world_size: int,
    ) -> None:
        """Initialize the collective communication."""
        from nemo_rl.distributed.stateless_process_group import StatelessProcessGroup

        # Place vLLM ranks after all training ranks so all training workers can join
        rank = train_world_size + resolve_rollout_rank(
            rank_prefix, world_size - train_world_size
        )

        self.model_update_group = StatelessProcessGroup(  # pyrefly: ignore[implicitly-defined-attribute]  This class does not define __init__ so assignments like this should be ignored
            master_address=ip, port=port, rank=rank, world_size=world_size
        )
        # Free cached torch-allocator blocks so NCCL's P2P transport buffers
        # (raw cudaMalloc at comm init) have headroom; otherwise comm_init OOMs
        # on memory-tight shapes (mirror the train side).
        torch.cuda.empty_cache()
        self.model_update_group.init_nccl_communicator(device=self.device)

    def init_nccl_reshard_comm_group(
        self,
        rank_prefix: int,
        pp_ips: list[str],
        pp_ports: list[int],
        pp_size: int,
        train_ranks_per_stage: int,
        sub_world_size: int,
    ) -> None:
        """Bootstrap this gen worker's nccl_reshard bulk-path comm group(s).

        One comm group per PP stage; gen workers join ALL ``pp_size`` groups
        (they need every stage's layers), created in stage order so the train
        ranks (each in only their own stage) unblock deterministically.
        Non-PP is simply ``pp_size == 1`` that contains all the gen ranks.
        """
        from nemo_rl.distributed.stateless_process_group import StatelessProcessGroup

        local_rank = torch.distributed.get_rank()
        gen_rank_in_group = train_ranks_per_stage + rank_prefix + local_rank

        # Free cached blocks so NCCL P2P buffers have headroom (see init_collective).
        torch.cuda.empty_cache()
        self.pp_comm_groups = {}  # pyrefly: ignore[implicitly-defined-attribute]
        for stage in range(pp_size):
            group = StatelessProcessGroup(
                master_address=pp_ips[stage],
                port=pp_ports[stage],
                rank=gen_rank_in_group,
                world_size=sub_world_size,
            )
            group.init_nccl_communicator(device=self.device)
            self.pp_comm_groups[stage] = group

    def report_device_id(self) -> str:
        """Retrieve the UUID of the current CUDA device."""
        from nemo_rl.utils.nvml import get_device_uuid

        return get_device_uuid(self.device.index)

    def report_node_hostname(self) -> str:
        """Return the host shared by worker processes on this node."""
        return socket.gethostname()

    def get_zmq_address(self):
        """Get the ZMQ address for the current device."""
        return f"ipc:///tmp/{self.report_device_id()}.sock"

    def maybe_init_zmq(self):
        """Initialize the ZMQ socket if it doesn't exist."""
        if not hasattr(self, "zmq_socket"):
            self.zmq_context = zmq.Context()  # pyrefly: ignore[implicitly-defined-attribute]  This class does not define __init__ so assignments like this should be ignored
            self.zmq_socket = self.zmq_context.socket(  # pyrefly: ignore[implicitly-defined-attribute]  This class does not define __init__ so assignments like this should be ignored
                zmq.REP
            )
            self.zmq_socket.setsockopt(
                zmq.SNDTIMEO, 120000
            )  # set timeout to 120 seconds
            self.zmq_socket.setsockopt(
                zmq.RCVTIMEO, 120000
            )  # set timeout to 120 seconds
            self.zmq_socket.setsockopt(zmq.LINGER, 0)
            self.zmq_socket.connect(self.get_zmq_address())

    def prepare_refit_info(self, state_dict_info: dict[str, Any]) -> None:
        """Prepare state dict metadata for weight refitting and IPC streaming.

        Args:
            state_dict_info (dict): A dictionary containing the info for refit.
                e.g. {tensor_name: (shape, dtype)}

        Raises:
            RuntimeError: If the model realizes the unquantized FlashInfer TRTLLM
                MoE backend while a co-trained MTP drafter is enabled (unsupported
                by the native layerwise refit lifecycle).
        """
        self._validate_native_layerwise_refit()
        self.state_dict_info = state_dict_info  # pyrefly: ignore[implicitly-defined-attribute]  This class does not define __init__ so assignments like this should be ignored

    def prepare_sparse_delta_refit_info(
        self, state_dict_info: dict[str, tuple[tuple[int, ...], torch.dtype]]
    ) -> list[str]:
        """Reserve scratch space and report weights that require overwrite."""
        self._reject_unsupported_native_refit("sparse_delta")
        applier = self._get_sparse_delta_applier()
        return sorted(applier.discover_native_skips(state_dict_info))

    def _uses_fp8_kv_cache(self) -> bool:
        """Return whether this worker owns an FP8 KV cache."""
        vllm_config = getattr(self.model_runner, "vllm_config", None)
        cache_config = getattr(vllm_config, "cache_config", None)
        kv_cache_dtype = getattr(cache_config, "cache_dtype", None)
        return kv_cache_dtype is not None and "fp8" in str(kv_cache_dtype).lower()

    def _maybe_process_fp8_kv_cache(self) -> None:
        """Process weights after loading for FP8 KV cache (static scales)."""
        if not self._uses_fp8_kv_cache():
            return

        # FP8 KV cache: process KV scales after weight loading
        from vllm.config import set_current_vllm_config
        from vllm.model_executor.model_loader.utils import (
            process_weights_after_loading,
        )

        # Get target device for processing
        target_device = next(self.model_runner.model.parameters()).device

        # Call process_weights_after_loading to handle KV scales
        with set_current_vllm_config(self.model_runner.vllm_config):
            process_weights_after_loading(
                self.model_runner.model,
                self.model_runner.model_config,
                target_device,
            )

    @staticmethod
    def _split_policy_and_draft_weights(
        weights: list[tuple[str, torch.Tensor]],
    ) -> tuple[list[tuple[str, torch.Tensor]], list[tuple[str, torch.Tensor]]]:
        """Split trainer-owned draft weights from policy weights.

        This path is only used for the Eagle3 online-training flow, where the
        trainer exports draft parameters under a `draft.` prefix before sending
        them to vLLM. MTP parameters do not use the `draft.` prefix; they remain
        in the policy stream and are forwarded separately by
        ``_maybe_refit_mtp_drafter``.
        The "draft." prefix is added here https://github.com/isomap/RL/blob/d3a5e1396d00f82fb888d9ec6800687a23bb4017/nemo_rl/models/policy/workers/megatron_policy_worker.py#L967-L997
        """
        policy_weights = []
        draft_weights = []
        for key, tensor in weights:
            if key.startswith("draft."):
                draft_weights.append((key.removeprefix("draft."), tensor))
            else:
                policy_weights.append((key, tensor))
        return policy_weights, draft_weights

    @staticmethod
    def _trim_vocab_padding(
        draft_model: torch.nn.Module,
        draft_weights: list[tuple[str, torch.Tensor]],
    ) -> list[tuple[str, torch.Tensor]]:
        """Trim padded vocab dimensions from draft weights.

        Megatron pads vocab to a multiple, but vLLM 0.20's autoloader
        strictly asserts loaded_weight.shape[0] == org_vocab_size on
        VocabParallelEmbedding layers. Each such layer may have a
        different org_vocab_size (e.g. embed_tokens uses vocab_size
        while lm_head uses draft_vocab_size), so we match each weight
        to its target module by name.
        """
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            VocabParallelEmbedding,
        )

        vocab_sizes: dict[str, int] = {}
        for name, module in draft_model.named_modules():
            if isinstance(module, VocabParallelEmbedding):
                vocab_sizes[name] = module.org_vocab_size

        if not vocab_sizes:
            return draft_weights

        trimmed = []
        for key, tensor in draft_weights:
            for mod_name, org_vocab_size in vocab_sizes.items():
                leaf = mod_name.rsplit(".", 1)[-1]
                if leaf in key and tensor.shape[0] > org_vocab_size:
                    tensor = tensor[:org_vocab_size]
                    break
            trimmed.append((key, tensor))
        return trimmed

    def _get_drafter_model(self) -> Any:
        """Return the vLLM drafter's underlying model, or None if absent.

        The drafter holds the speculative-decoding draft model (Eagle3 or MTP),
        which vLLM keeps as a module separate from the main model. Typed ``Any``
        because these are dynamic vLLM model classes whose ``load_weights`` /
        ``mtp_start_layer_idx`` members are not visible through ``nn.Module``.
        """
        draft_owner = getattr(self.model_runner, "drafter", None)
        return getattr(draft_owner, "model", None) if draft_owner else None

    def configure_mtp_drafter_weight_source(self, weights_from_refit: bool) -> None:
        """Record whether the trainer owns and refreshes the MTP weights."""
        self._mtp_drafter_weights_from_refit = weights_from_refit

    def _load_draft_weights(
        self, draft_weights: list[tuple[str, torch.Tensor]]
    ) -> None:
        if not draft_weights:
            return

        draft_model = self._get_drafter_model()
        if draft_model is None:
            logger.warning(
                "[draft] Received draft weights but vLLM drafter is unavailable; skipping draft update."
            )
            return
        draft_weights = self._trim_vocab_padding(draft_model, draft_weights)
        draft_model.load_weights(weights=draft_weights)

    def _mtp_drafter_refit_enabled(self) -> bool:
        """Whether MTP drafter weights should be refreshed from the refit stream.

        For MTP speculative decoding where the trainer co-trains the MTP layer
        (``mtp_num_layers > 0``), the MTP weights are exported as part of the
        policy weight stream during refit (without the ``draft.`` prefix used by
        Eagle3), so the drafter must be fed those weights on every refit.

        Returns False when the MTP weights were instead loaded once from disk
        (see ``load_mtp_weights_from_disk``) — the path used when the trainer
        does not co-train the MTP layer — to avoid clobbering and re-processing
        those static weights.
        """
        if not self._mtp_drafter_weights_from_refit:
            return False
        spec_config = getattr(self.model_runner.vllm_config, "speculative_config", None)
        method = getattr(spec_config, "method", None) if spec_config else None
        if method not in ("deepseek_mtp", "mtp"):
            return False
        return self._get_drafter_model() is not None

    def _maybe_refit_mtp_drafter(self, weights: list[tuple[str, torch.Tensor]]) -> None:
        """Load refit weights into an MTP drafter co-trained with the policy.

        The drafter's ``load_weights`` selects the MTP-specific parameters (and
        shared embed_tokens / lm_head) it needs from the full policy weight
        stream. Megatron pads the vocab dimension, so weights are trimmed to the
        drafter's expected vocab size first, matching ``_load_draft_weights``.
        """
        if not self._mtp_drafter_refit_enabled():
            return
        draft_model = self._get_drafter_model()
        if draft_model is None:
            return
        weights = self._trim_vocab_padding(draft_model, weights)
        draft_model.load_weights(weights=weights)

    def _maybe_process_mtp_drafter_after_loading(self) -> None:
        """Finalize MTP drafter weights after a refit (e.g. MoE grouped-GEMM layout).

        Mirrors the main-model post-processing so the freshly refit MTP layers
        are converted to their runtime layout. Skipped for the disk-load path,
        which already processes its weights once at startup.
        """
        if not self._mtp_drafter_refit_enabled():
            return
        draft_model = self._get_drafter_model()
        if draft_model is None:
            return

        from vllm.config import set_current_vllm_config
        from vllm.model_executor.model_loader.utils import (
            process_weights_after_loading,
        )

        draft_model_config = (
            self.model_runner.vllm_config.speculative_config.draft_model_config
        )
        with set_current_vllm_config(self.model_runner.vllm_config):
            process_weights_after_loading(draft_model, draft_model_config, self.device)

    def load_mtp_weights_from_disk(self, model_path: str) -> bool:
        """Load only the MTP (multi-token-prediction) draft weights from disk.

        Used when an MTP speculative-decoding policy runs with
        ``load_format="dummy"``: the main model receives real weights via refit,
        but the MTP draft layer is not covered by refit (the trainer runs with
        ``mtp_num_layers=0``), so its weights must come from the checkpoint. Only
        the MTP layer(s) are read, avoiding a full base-model load (~1.3 TB for
        DeepSeek-V3) on every inference replica.

        Args:
            model_path: Path to the HF checkpoint directory.

        Returns:
            bool: True if MTP weights were loaded.
        """
        draft_model = self._get_drafter_model()
        if draft_model is None:
            # vLLM places the speculative drafter only on the last pipeline
            # stage. Its absence is expected on every earlier stage, but means
            # the engine cannot serve speculative decoding on the owning stage.
            if get_pp_group().is_last_rank:
                raise RuntimeError(
                    "[mtp] vLLM speculative_config is set for MTP but the drafter "
                    "model is unavailable; cannot load MTP weights from disk."
                )
            return False

        predictor = draft_model.model
        mtp_layer_indices = set(
            range(
                predictor.mtp_start_layer_idx,
                predictor.mtp_start_layer_idx + predictor.num_mtp_layers,
            )
        )
        weights = _read_mtp_layer_weights_from_checkpoint(model_path, mtp_layer_indices)
        if not weights:
            raise ValueError(
                f"No MTP layer weights for layers {sorted(mtp_layer_indices)} "
                f"found in checkpoint at {model_path}. The checkpoint must "
                f"include MTP layer weights to run deepseek_mtp speculative decoding."
            )

        draft_model_config = (
            self.model_runner.vllm_config.speculative_config.draft_model_config
        )

        # The MTP block contains MoE experts whose weights need post-load
        # processing (e.g. grouped-GEMM layout), matching the main-model path.
        # Keep vLLM reload internals off the normal draft-loading path.
        from vllm.config import set_current_vllm_config

        if self._supports_unquantized_flashinfer_trtllm_refit() and (
            _model_uses_unquantized_flashinfer_trtllm(draft_model)
        ):
            from vllm.model_executor.model_loader.reload import (
                finalize_layerwise_reload,
                initialize_layerwise_reload,
            )

            with set_current_vllm_config(self.model_runner.vllm_config):
                with torch.device(self.device):
                    initialize_layerwise_reload(draft_model)
                    self._load_draft_weights(weights)
                    finalize_layerwise_reload(draft_model, draft_model_config)
                    _refresh_hpc_modules_after_layerwise_reload(draft_model)
        else:
            from vllm.model_executor.model_loader.utils import (
                process_weights_after_loading,
            )

            self._load_draft_weights(weights)
            with set_current_vllm_config(self.model_runner.vllm_config):
                process_weights_after_loading(
                    draft_model, draft_model_config, self.device
                )
        # This drafter is served by the checkpoint rather than the refit stream.
        self._mtp_drafter_weights_from_refit = False
        logger.info(
            "[mtp] Loaded MTP draft weights for layers %s from %s",
            sorted(mtp_layer_indices),
            model_path,
        )
        return True

    def _load_weights(self, weights):
        """Load weights with Gemma3 vision-tower weight name fix, FP8, and draft-weight support.

        Applies Gemma3 vision-tower weight name fix if needed, splits policy/draft
        weights, dispatches policy weights through the configured refit loader,
        and loads draft weights into the drafter model.
        """
        if (
            "Gemma3ForConditionalGeneration"
            in self.model_runner.vllm_config.model_config.architectures
        ):
            for idx, (key, weight) in enumerate(weights):
                weights[idx] = (fix_gemma3_vision_weight_name(key), weight)

        policy_weights, draft_weights = self._split_policy_and_draft_weights(weights)
        self._load_hf_weights(policy_weights)
        # Eagle3 draft weights are exported with the `draft.` prefix.
        self._load_draft_weights(draft_weights)
        # MTP drafters co-trained with the policy receive their weights from the
        # policy stream (no `draft.` prefix), so feed it the policy weights too.
        self._maybe_refit_mtp_drafter(policy_weights)

    def _get_sparse_delta_applier(self) -> Any:
        if self._sparse_delta_applier is None:
            # Avoid importing sparse-refit code for existing refit transports.
            from nemo_rl.models.generation.vllm.vllm_sparse_delta import (
                VllmSparseDeltaApplier,
            )

            self._sparse_delta_applier = VllmSparseDeltaApplier(
                self.model_runner,
                self.device,
            )
        return self._sparse_delta_applier

    def _supports_unquantized_flashinfer_trtllm_refit(self) -> bool:
        """Whether this worker supports native unquantized TRTLLM refits."""
        return True

    def _uses_unquantized_flashinfer_trtllm(self) -> bool:
        """Detect a realized unquantized FlashInfer TRTLLM MoE backend."""
        if not self._supports_unquantized_flashinfer_trtllm_refit():
            return False
        model_runner = getattr(self, "model_runner", None)
        model = getattr(model_runner, "model", None)
        if model is None:
            return False

        return _model_uses_unquantized_flashinfer_trtllm(model)

    def _unquantized_flashinfer_trtllm_param_ids(self) -> set[int]:
        """Return parameters owned by realized unquantized TRTLLM MoE modules."""
        if not self._supports_unquantized_flashinfer_trtllm_refit():
            return set()
        model_runner = getattr(self, "model_runner", None)
        model = getattr(model_runner, "model", None)
        if model is None:
            return set()

        return {
            id(param)
            for module in _unquantized_flashinfer_trtllm_modules(model)
            for param in module.parameters(recurse=False)
        }

    def _uses_native_layerwise_refit(self, transport: WeightUpdateTransport) -> bool:
        """Return whether this transport needs vLLM's layerwise lifecycle."""
        return transport in ("ipc", "collective", "nccl_reshard") and (
            self._uses_unquantized_flashinfer_trtllm()
        )

    def _validate_native_layerwise_refit(
        self, transport: WeightUpdateTransport | None = None
    ) -> None:
        """Reject unsupported features on the native layerwise reload path."""
        if not self._uses_unquantized_flashinfer_trtllm():
            return

        if transport in ("ipc", "collective", "nccl_reshard") and (
            self._uses_fp8_kv_cache()
        ):
            raise RuntimeError(
                "BF16 FlashInfer TRTLLM partial refit does not support an "
                "FP8 KV cache because its static scales are outside the "
                "targeted reload lifecycle"
            )

        if transport == "nccl_reshard":
            realized_placements = set()
            for module in _unquantized_flashinfer_trtllm_modules(
                self.model_runner.model
            ):
                strategy = getattr(
                    getattr(module, "expert_map_manager", None),
                    "placement_strategy",
                    getattr(module, "expert_placement_strategy", None),
                )
                if strategy is None:
                    raise RuntimeError(
                        "BF16 FlashInfer TRTLLM nccl_reshard refit could "
                        "not determine the expert placement strategy of "
                        f"{type(module).__name__}; refusing to assume linear "
                        "placement"
                    )
                realized_placements.add(strategy)
            unsupported_placements = sorted(realized_placements - {"linear"})
            if unsupported_placements:
                raise RuntimeError(
                    "BF16 FlashInfer TRTLLM nccl_reshard refit requires "
                    "linear expert placement; realized "
                    f"{unsupported_placements!r}"
                )

        if self._mtp_drafter_refit_enabled():
            raise RuntimeError(
                "Unquantized FlashInfer TRTLLM refit does not yet support "
                "a co-trained MTP drafter"
            )

    def _reject_unsupported_native_refit(
        self, transport: UnsupportedNativeRefitTransport
    ) -> None:
        """Reject transports that cannot run the native layerwise lifecycle."""
        if not self._uses_unquantized_flashinfer_trtllm():
            return

        label = transport.replace("_", "-")
        raise RuntimeError(
            f"{label} refit does not support the unquantized FlashInfer "
            "TRTLLM MoE backend yet because it bypasses vLLM's native "
            "layerwise reload lifecycle"
        )

    @contextmanager
    def _weight_update_lifecycle(
        self, transport: WeightUpdateTransport
    ) -> Iterator[WeightUpdateFinalizer]:
        """Provide setup/finalization around a transport-owned weight update.

        Native reload initialization invalidates the old runtime layout. Any
        subsequent exception therefore marks this worker permanently unusable.
        """
        if self._uses_native_layerwise_refit(transport):
            self._validate_native_layerwise_refit(transport)
            previous_failure = self._nrl_layerwise_reload_failure
            if previous_failure is not None:
                raise RuntimeError(
                    "The vLLM worker is unusable after a failed native layerwise refit"
                ) from previous_failure
            # Load vLLM reload internals only for the native layerwise path.
            from vllm.config import set_current_vllm_config
            from vllm.model_executor.model_loader.reload import (
                finalize_layerwise_reload,
                initialize_layerwise_reload,
            )

            model = self.model_runner.model
            # Restore only the realized BF16 TRTLLM modules. MXFP8 modules own
            # checkpoint-scale parameters created after vLLM recorded reload
            # metadata, so restoring the whole mixed model would delete those
            # parameters. Their layouts are rebuilt separately after transfer.
            reload_targets = _unquantized_flashinfer_trtllm_modules(model)
            reloaded_module_ids = _reload_target_module_ids(reload_targets)

            def finalize() -> None:
                with _use_batched_bf16_trtllm_layout_conversion():
                    with torch.device(self.device):
                        finalize_layerwise_reload(model, self.model_config)
                        _process_mxfp8_modules_after_native_reload(
                            model, reloaded_module_ids
                        )
                        _refresh_hpc_modules_after_layerwise_reload(model)
                        self._maybe_process_mtp_drafter_after_loading()
                torch.cuda.synchronize()

            try:
                with set_current_vllm_config(self.model_runner.vllm_config):
                    with torch.device(self.device):
                        for reload_target in reload_targets:
                            initialize_layerwise_reload(reload_target)
                    self._nrl_layerwise_reload_active = True
                    yield finalize
            except Exception as error:
                self._nrl_layerwise_reload_failure = error
                raise
            finally:
                self._nrl_layerwise_reload_active = False

            return
        from vllm.config import set_current_vllm_config
        from vllm.model_executor.model_loader.utils import (
            process_weights_after_loading,
        )

        def finalize() -> None:
            with set_current_vllm_config(self.model_runner.vllm_config):
                process_weights_after_loading(
                    self.model_runner.model, self.model_config, self.device
                )
            self._maybe_process_mtp_drafter_after_loading()

        yield finalize
        if transport != "checkpoint_engine":
            # Preserve the IPC lifetime boundary: the COMPLETE ACK is sent before
            # this optional second pass, just as it was before lifecycle hooks.
            self._maybe_process_fp8_kv_cache()

    def _weight_update_errors_are_fatal(self) -> bool:
        """Whether transport errors should propagate instead of returning False."""
        return self._uses_unquantized_flashinfer_trtllm()

    def _synchronize_before_ipc_data_ack(self) -> None:
        """Fence work consuming one IPC data batch before its acknowledgment."""
        torch.cuda.current_stream().synchronize()

    @wrap_with_nvtx_name("vllm_internal_worker_extension/update_weights_via_ipc_zmq")
    def update_weights_via_ipc_zmq(self) -> bool:
        """Receive and update model weights via ZMQ IPC socket.

        Returns:
            bool: True if weights were successfully updated.
        """
        buffer = None
        weight = None
        weights = None

        try:
            self.maybe_init_zmq()
            manifest = _IPCWeightManifest(self.state_dict_info)
            with self._weight_update_lifecycle("ipc") as finalize:
                while True:
                    # Blocking receive with timeout (this is the main operation)
                    payload = self.zmq_socket.recv_pyobj()

                    if payload == IPCProtocol.COMPLETE:
                        # A REP socket must reply even when validation or finalization
                        # fails, otherwise the sender remains blocked until timeout.
                        try:
                            manifest.require_complete()
                            finalize()
                        finally:
                            self.zmq_socket.send(IPCProtocol.ACK.value.encode())
                        break

                    batch_keys = None
                    batch_error = None
                    try:
                        ipc_handle, list_keys, used_bytes = payload
                        batch_keys = manifest.validate_batch(list_keys)
                        if batch_keys is None:
                            continue

                        buffer = rebuild_cuda_tensor_from_ipc(
                            ipc_handle, self.device.index
                        )
                        weights = []
                        offset = 0
                        for key in list_keys:
                            shape, dtype = self.state_dict_info[key]  # pyrefly
                            if isinstance(shape, list):
                                shape = torch.Size(shape)

                            size_in_bytes = dtype.itemsize * shape.numel()
                            weight = (
                                buffer[offset : offset + size_in_bytes]
                                .view(dtype=dtype)
                                .view(shape)
                            )
                            weights.append((key, weight))
                            offset += calculate_aligned_size(size_in_bytes)

                        assert offset == used_bytes, (
                            "Offset is not equal to used bytes, usually indicate "
                            "inaccurate info like keys or cached dtype in "
                            "state_dict_info"
                        )
                        self._load_weights(weights)
                    except Exception as error:
                        batch_error = error
                        # The manifest only keeps the exception message; log
                        # the full traceback and the batch contents so loader
                        # failures stay diagnosable from worker logs.
                        batch_desc = ", ".join(
                            f"{k}: {tuple(w.shape)} {w.dtype}"
                            for k, w in (weights or [])[:40]
                        )
                        logger.exception(
                            "IPC weight batch load failed (batch: %s)", batch_desc
                        )
                    finally:
                        # Synchronize before releasing or ACKing an IPC allocation,
                        # including when a loader failed after scheduling CUDA work.
                        if buffer is not None:
                            try:
                                self._synchronize_before_ipc_data_ack()
                            except Exception as error:
                                if batch_error is None:
                                    batch_error = error

                        if batch_error is not None:
                            manifest.record_load_failure(batch_error)
                        elif batch_keys is not None:
                            manifest.record_loaded(batch_keys)

                        # Drop every view before ACK permits sender-side reuse.
                        del weight, weights, buffer
                        weight = None
                        weights = None
                        buffer = None
                        self.zmq_socket.send(IPCProtocol.ACK.value.encode())

            gc.collect()
            torch.cuda.empty_cache()
            return True
        except Exception as e:
            if self._weight_update_errors_are_fatal():
                raise
            logger.exception(
                "Error in VllmInternalWorkerExtension.update_weights_via_ipc_zmq: %s",
                e,
            )
            return False

    @wrap_with_nvtx_name(
        "vllm_internal_worker_extension/update_weights_from_collective"
    )
    def update_weights_from_collective(self) -> bool:
        """Update the model weights from collective communication."""
        assert self.state_dict_info is not None, (
            "state_dict_info is not prepared. "
            "Please call prepare_refit_info when initializing the worker."
        )

        try:
            native_layerwise_refit = self._uses_native_layerwise_refit("collective")
            with self._weight_update_lifecycle("collective") as finalize:
                packed_broadcast_consumer(
                    iterator=iter(self.state_dict_info.items()),
                    group=self.model_update_group,
                    src=0,
                    post_unpack_func=self._load_weights,
                    # Double buffering (num_buffers > 1) causes a race condition
                    # when using native_layerwise_refit: deferred weight_loader
                    # replays may read a buffer while the other stream refills it.
                    num_buffers=1 if native_layerwise_refit else None,
                )
                finalize()

        except Exception as e:
            if self._weight_update_errors_are_fatal():
                raise
            logger.exception(
                "Error in VllmInternalWorkerExtension.update_weights_from_collective: %s",
                e,
            )
            return False

        gc.collect()
        torch.cuda.empty_cache()
        return True

    def update_weights_from_decoded_sparse_payload(
        self, *payloads: bytes | str
    ) -> dict[str, Any]:
        self._reject_unsupported_native_refit("sparse_delta")
        applier = self._get_sparse_delta_applier()
        return applier.update_weights_from_decoded_sparse_payload(*payloads)

    def synchronize_device(self) -> None:
        self._get_sparse_delta_applier().synchronize_device()

    def finish_sparse_delta_refit(self) -> dict[str, Any]:
        return self._get_sparse_delta_applier().finish_sparse_delta_refit()

    def prepare_nccl_reshard_refit_info(self, refit_info: dict) -> None:
        """Restore metadata and preflight the selected destination route."""
        self._validate_native_layerwise_refit("nccl_reshard")

        from nemo_rl.weight_sync.nccl_reshard_utils import (
            restore_refit_info_placements,
        )

        self.nccl_reshard_refit_info = restore_refit_info_placements(refit_info)
        if _native_mxfp8_param_names(self.nccl_reshard_refit_info):
            adapter = self._get_nccl_reshard_refit_adapter()
            adapter.validate_plan(self.nccl_reshard_refit_info)
            adapter.prepare(self.nccl_reshard_refit_info)
            # Active checkpoint tensors replace runtime tensors at begin_update,
            # so the concrete map is intentionally rebuilt for every native refit.
            self.hf_to_local_param_map = HFToLocalParamMap()
        else:
            self.hf_to_local_param_map = self.build_hf_to_local_param_map(
                self.nccl_reshard_refit_info
            )

    def _get_nccl_reshard_refit_adapter(self):
        adapter = self._nccl_reshard_refit_adapter
        if adapter is None:
            # Keep vLLM reload capability probing off legacy NCCL Reshard setup.
            from nemo_rl.models.generation.vllm.refit_adapter import (
                create_vllm_refit_adapter,
            )

            adapter = create_vllm_refit_adapter(
                model_runner=self.model_runner,
                model_config=self.model_config,
                device=self.device,
            )
            self._nccl_reshard_refit_adapter = adapter
        return adapter

    def build_hf_to_local_param_map(self, refit_info: dict) -> HFToLocalParamMap:
        """Build the vLLM-backend ``hf_to_local_param_map`` (HFToLocalParamMap).

        Wraps the ``(vllm_param, merged_slice)`` resolution from
        ``_build_hf_to_gen_backend_mapping`` into ``LocalParamSpec``s:
        - direct (slice ``None``): ``base`` is the live vLLM param; receive in place.
        - merged (dense ``gate_up_proj`` / grouped-expert ``w13``): ``pre`` allocs a
          recv buffer for this component's ``region`` slice, ``post`` copies it back
          (region recomputed each refit to track live storage).
        """

        def _merged_param_spec(vllm_param, merged_slice):
            def pre(_base: torch.Tensor) -> RefitCtx:
                region = vllm_param.data[merged_slice]
                return RefitCtx(buf=torch.empty_like(region), extra={"region": region})

            def post(ctx: RefitCtx) -> None:
                ctx.extra["region"].copy_(ctx.buf)

            return LocalParamSpec(base=vllm_param, pre=pre, post=post)

        def _bf16_to_mxfp8_receiver_quant_spec(
            value_param: torch.Tensor,
            scale_param: torch.Tensor,
            merged_slice: tuple[slice, ...] | None,
        ) -> LocalParamSpec:
            def pre(_base: torch.Tensor) -> RefitCtx:
                value_region = (
                    value_param.data
                    if merged_slice is None
                    else value_param.data[merged_slice]
                )
                scale_region = (
                    scale_param.data
                    if merged_slice is None
                    else scale_param.data[merged_slice]
                )
                return RefitCtx(
                    buf=torch.empty_like(value_region, dtype=torch.bfloat16),
                    extra={"value_region": value_region, "scale_region": scale_region},
                )

            def post(ctx: RefitCtx) -> None:
                from nemo_rl.models.generation.vllm.quantization.fp8 import (
                    quantize_mxfp8_weight,
                )

                value, scale = quantize_mxfp8_weight(ctx.buf)
                ctx.extra["value_region"].copy_(value)
                ctx.extra["scale_region"].copy_(scale)

            return LocalParamSpec(base=value_param.data, pre=pre, post=post)

        param_info_by_name = {
            param_info["name"]: param_info
            for layer_name in refit_info["layer_names"]
            for param_info in refit_info["per_layer_params"][layer_name]
        }
        specs: dict[str | tuple[str, str], LocalParamSpec] = {}
        native_names = _native_mxfp8_param_names(refit_info)
        adapter = self._get_nccl_reshard_refit_adapter() if native_names else None
        if adapter is not None:
            for hf_name in native_names:
                for component in param_info_by_name[hf_name]["components"]:
                    role = component["role"]
                    specs[(hf_name, role)] = adapter.resolve_destination(
                        logical_name=hf_name,
                        role=role,
                    )
        if native_names == set(param_info_by_name):
            return HFToLocalParamMap(specs=specs)

        # Legacy parameters retain the existing direct/merged/receiver-quantized
        # mapping even when they share a refit plan with native components.
        vllm_param_map_and_slices = self._build_hf_to_gen_backend_mapping(refit_info)
        vllm_params = dict(self.model_runner.model.named_parameters())
        vllm_names_by_id = {id(param): name for name, param in vllm_params.items()}
        unquantized_trtllm_param_ids = self._unquantized_flashinfer_trtllm_param_ids()
        for hf_name, (vllm_param, merged_slice) in vllm_param_map_and_slices.items():
            if hf_name in native_names:
                continue
            param_info = param_info_by_name[hf_name]
            if id(vllm_param) in unquantized_trtllm_param_ids and param_info.get(
                "grouped_expert_proj"
            ):
                specs[hf_name] = _trtllm_grouped_expert_spec(param_info, vllm_param)
                continue
            wire_dtype_value = param_info.get("dtype")
            wire_dtype = (
                wire_dtype_value
                if isinstance(wire_dtype_value, torch.dtype)
                else _STR_TO_DTYPE.get(wire_dtype_value)
            )
            if wire_dtype is None:
                raise ValueError(
                    f"build_hf_to_local_param_map: unsupported wire dtype "
                    f"{wire_dtype_value!r} for {hf_name!r}"
                )
            if wire_dtype == torch.bfloat16 and vllm_param.dtype == torch.float8_e4m3fn:
                vllm_name = vllm_names_by_id.get(id(vllm_param))
                if vllm_name is None:
                    raise ValueError(
                        f"build_hf_to_local_param_map: resolved vLLM target for "
                        f"{hf_name!r} is not a registered model parameter"
                    )
                scale_names = (
                    vllm_name + "_scale_from_checkpoint",
                    vllm_name + "_scale",
                )
                scale_name = next(
                    (name for name in scale_names if name in vllm_params), None
                )
                scale_param = (
                    vllm_params.get(scale_name) if scale_name is not None else None
                )
                if scale_param is None:
                    raise ValueError(
                        f"build_hf_to_local_param_map: MXFP8 target {vllm_name!r} "
                        f"for {hf_name!r} has no scale parameter among "
                        f"{scale_names!r}"
                    )
                value_region = (
                    vllm_param if merged_slice is None else vllm_param[merged_slice]
                )
                scale_region = (
                    scale_param if merged_slice is None else scale_param[merged_slice]
                )
                if value_region.shape[-1] % 32 != 0:
                    raise ValueError(
                        f"build_hf_to_local_param_map: MXFP8 target for {hf_name!r} "
                        f"must have K divisible by 32, got {tuple(value_region.shape)}"
                    )
                expected_scale_shape = (
                    *value_region.shape[:-1],
                    value_region.shape[-1] // 32,
                )
                if tuple(scale_region.shape) != expected_scale_shape:
                    raise ValueError(
                        f"build_hf_to_local_param_map: MXFP8 scale target "
                        f"{scale_name!r} for {hf_name!r} has shape "
                        f"{tuple(scale_region.shape)}, expected {expected_scale_shape}"
                    )
                if scale_param.dtype != torch.uint8:
                    raise ValueError(
                        f"build_hf_to_local_param_map: MXFP8 scale target "
                        f"{scale_name!r} has dtype {scale_param.dtype}, expected torch.uint8"
                    )
                specs[hf_name] = _bf16_to_mxfp8_receiver_quant_spec(
                    vllm_param, scale_param, merged_slice
                )
            elif wire_dtype != vllm_param.dtype:
                raise ValueError(
                    f"build_hf_to_local_param_map: wire dtype {wire_dtype} does not "
                    f"match target dtype {vllm_param.dtype} for {hf_name!r}"
                )
            else:
                specs[hf_name] = (
                    LocalParamSpec(base=vllm_param.data)
                    if merged_slice is None
                    else _merged_param_spec(vllm_param, merged_slice)
                )
        return HFToLocalParamMap(specs=specs)

    def _build_hf_to_gen_backend_mapping(self, refit_info):
        """Map each FFN HF param name to its gen-backend param and slice.

        Only ``gate_proj`` / ``up_proj`` / ``down_proj`` ``.weight``
        (dense MLP and MoE experts) reach here.
        Returns ``hf_name -> (vllm_param, merged_param_slice or None)``; the
        slice (``None`` for a 1:1 direct map) is the local region of a fused
        vLLM param this HF piece occupies, applied by the LocalParamSpec
        pre/post hooks.  The three shapes:

          - grouped MoE experts: gate/up -> ``w13_weight`` halves (dim 1),
            down -> ``w2_weight`` (direct).
          - dense MLP gate/up    -> ``gate_up_proj`` halves (dim 0).
          - dense MLP down       -> ``down_proj`` (direct 1:1).
        """
        vllm_params = dict(self.model_runner.model.named_parameters())
        mapping: dict[str, tuple[torch.Tensor, tuple[slice, ...] | None]] = {}

        # Collect FFN param names + global shapes from refit_info, plus the
        # grouped-expert tag (gate_proj/up_proj/down_proj) for MoE params.
        hf_shapes = {}  # hf_name -> global_shape
        hf_grouped = {}  # hf_name -> "gate_proj"|"up_proj"|"down_proj" (MoE only)
        for layer_name in refit_info["layer_names"]:
            # p is a dict of param info
            for p in refit_info["per_layer_params"][layer_name]:
                hf_shapes[p["name"]] = tuple(p["global_shape"])
                if p.get("grouped_expert_proj"):
                    hf_grouped[p["name"]] = p["grouped_expert_proj"]

        # Check if this model uses gated MLP layer (e.g., SwiGLU, Gated ReLU^2)
        has_gate = {
            name.rsplit(".gate_proj.weight", 1)[0]
            for name, proj in hf_grouped.items()
            if proj == "gate_proj"
        }

        # Resolve an HF FFN name to its vLLM param name.  The two differ only in
        # the module prefix before ``layers.N`` (e.g. NemotronH's HF ``backbone.``
        # vs vLLM ``model.``); the layer-relative suffix is identical.  Index the
        # real vLLM names by that suffix so any prefix rename resolves generically
        # instead of hardcoding per-model swaps.  Matching-prefix models (most)
        # hit the exact-name fast path and never touch the index.
        def _layer_relative(name: str) -> str:
            prefix = _extract_layer_prefix(name)
            return name[len(prefix) + 1 :] if prefix else name

        vllm_by_relative = {_layer_relative(n): n for n in vllm_params}

        # vLLM 0.25 moved the fused-MoE expert weights onto a nested
        # ``routed_experts`` submodule, so real names carry a
        # ``.routed_experts.`` segment that the name built from the HF side
        # below does not (``...mlp.experts.w13_weight`` vs
        # ``...mlp.experts.routed_experts.w13_weight``).  Index the real names
        # with that segment dropped so either layout resolves; on a 0.20-style
        # model this index is identical to ``vllm_by_relative``.
        vllm_by_relative_flat = {
            _layer_relative(n).replace(".routed_experts.", "."): n for n in vllm_params
        }

        def _to_vllm_name(n: str) -> str:
            if n in vllm_params:
                return n
            relative = _layer_relative(n)
            if relative in vllm_by_relative:
                return vllm_by_relative[relative]
            return vllm_by_relative_flat.get(relative, n)

        for hf_name in hf_shapes:
            # 1) Grouped MoE expert params (gate_proj/up_proj/down_proj, each
            #    [E, ...]). vLLM fuses them as w13_weight (gate||up on the
            #    intermediate axis) and w2_weight (down). The received
            #    Shard(1)/Shard(2) shard is placed into the right w13/w2 region by
            #    the LocalParamSpec pre/post hooks (for the gated w13 halves).
            # Caveat: Dispatch on the grouped_expert_proj TAG, NOT the suffix,
            #   so dense gate_proj/up_proj (-> gate_up_proj, rule below) don't collide.
            grouped_proj = hf_grouped.get(hf_name)
            if grouped_proj is not None:
                # e.g.) expert_prefix = model.layers.3.mlp.experts
                expert_prefix = hf_name.rsplit(f".{grouped_proj}.weight", 1)[0]
                vllm_suffix = (
                    "w2_weight" if grouped_proj == "down_proj" else "w13_weight"
                )
                # e.g.) vllm_name = model.layers.3.mlp.experts.w13_weight
                vllm_name = _to_vllm_name(f"{expert_prefix}.{vllm_suffix}")
                if vllm_name not in vllm_params:
                    raise ValueError(
                        f"_build_hf_to_gen_backend_mapping: grouped expert {hf_name!r} has "
                        f"no vLLM target {vllm_name!r}; refit would silently drop "
                        f"the expert weights."
                    )
                # vllm_param is a torch.Tensor corresponding to the vllm_name
                vllm_param = vllm_params[vllm_name]
                if grouped_proj == "down_proj" or expert_prefix not in has_gate:
                    # Case for non-gated MLP layer or down_proj (w2)
                    # Weights are not merged, so the mapping is 1:1
                    mapping[hf_name] = (vllm_param, None)
                else:
                    # Gated MLP: vLLM fuses gate (w1) + up (w3) into w13 along the
                    # intermediate axis (dim 1).  Standard layout is [gate; up]:
                    # gate -> [:, :P, :], up -> [:, P:2P, :].  The FlashInfer
                    # CUTLASS unquantized MoE backend instead stores w13 as
                    # [w3; w1] = [up; gate]
                    P = vllm_param.shape[1] // 2
                    # Write canonical [gate; up], following vLLM's load_weights
                    # behavior. Per-MoE-backend layout diversity is resolved later by
                    # process_weights_after_loading at the end of nccl_reshard_refit.
                    sl = slice(0, P) if grouped_proj == "gate_proj" else slice(P, 2 * P)
                    mapping[hf_name] = (vllm_param, (slice(None), sl, slice(None)))
                continue

            # 2) Direct 1:1 (dense down_proj; also non-gated dense up_proj, which
            #    vLLM keeps unmerged).
            vllm_direct = _to_vllm_name(hf_name)
            if vllm_direct in vllm_params:
                mapping[hf_name] = (vllm_params[vllm_direct], None)
                continue

            # 3) Gated dense MLP: gate/up fuse into gate_up_proj along dim 0,
            #    [gate; up] -> gate=[0:I_local], up=[I_local:2*I_local], where
            #    I_local = intermediate // gen TP (even split, gate==up size).
            if hf_name.endswith(("gate_proj.weight", "up_proj.weight")):
                is_gate = hf_name.endswith("gate_proj.weight")
                suffix = "gate_proj.weight" if is_gate else "up_proj.weight"
                prefix = hf_name[: -len(suffix)]
                vllm_name = _to_vllm_name(prefix + "gate_up_proj.weight")
                if vllm_name in vllm_params:
                    tp = refit_info.get("gen_tp_size", 1)
                    gate_local = hf_shapes[prefix + "gate_proj.weight"][0] // tp
                    up_local = hf_shapes[prefix + "up_proj.weight"][0] // tp
                    sl = (
                        slice(0, gate_local)
                        if is_gate
                        else slice(gate_local, gate_local + up_local)
                    )
                    mapping[hf_name] = (vllm_params[vllm_name], (sl,))
                    continue

            raise ValueError(
                f"_build_hf_to_gen_backend_mapping: no vLLM param for {hf_name!r} "
                f"(no grouped-expert / direct / gate_up-merge match). Only FFN "
                f"gate/up/down weights should reach the bulk path."
            )

        return mapping

    def nccl_reshard_refit(self) -> bool:
        """Receive ordered parameter components through NCCL Reshard."""
        import os
        import time
        from collections import OrderedDict

        from nemo_rl.weight_sync.xferdtensor import DTensorRef, xferdtensor

        def _components(param_info: dict[str, Any]) -> list[dict[str, Any]]:
            components = param_info.get("components")
            if components is not None:
                return components
            return [
                {
                    "role": "weight",
                    "global_shape": param_info["global_shape"],
                    "src_placements": param_info["src_placements"],
                    "dst_placements": param_info["dst_placements"],
                }
            ]

        def _recv_one_component(
            param_info: dict[str, Any],
            component: dict[str, Any],
            group: Any,
            stream: torch.cuda.Stream,
        ) -> None:
            role = component["role"]
            spec = self.hf_to_local_param_map.get(param_info["name"], role=role)
            if spec is None:
                raise RuntimeError(
                    f"nccl_reshard_refit: {param_info['name']!r} role {role!r} "
                    "has no destination spec"
                )
            ctx = (
                spec.pre(spec.base) if spec.pre is not None else RefitCtx(buf=spec.base)
            )
            if not isinstance(ctx.buf, torch.Tensor):
                raise RuntimeError(
                    f"nccl_reshard_refit: {param_info['name']!r} role {role!r} "
                    "did not produce a destination tensor"
                )
            dst_tensor = DTensorRef(ctx.buf, component["global_shape"])
            xferdtensor(
                None,
                param_info["src_mesh_info"],
                component.get("src_placements", param_info["src_placements"]),
                dst_tensor,
                param_info["dst_mesh_info"],
                component.get("dst_placements", param_info["dst_placements"]),
                group,
                stream,
            )
            if spec.post is not None:
                spec.post(ctx)

        def _receive_bulk_components() -> None:
            stage_params: OrderedDict[int, list[dict[str, Any]]] = OrderedDict()
            for layer_name in self.nccl_reshard_refit_info["layer_names"]:
                for param_info in self.nccl_reshard_refit_info["per_layer_params"][
                    layer_name
                ]:
                    stage_params.setdefault(param_info.get("pp_stage", 0), []).append(
                        param_info
                    )
            num_streams = max(
                1,
                min(
                    int(os.environ.get("NRL_REFIT_NUM_STREAMS", "2")),
                    len(stage_params),
                ),
            )
            streams = [torch.cuda.Stream() for _ in range(num_streams)]
            events: dict[int, torch.cuda.Event] = {}
            for index, (stage, params) in enumerate(stage_params.items()):
                if (index - num_streams) in events:
                    events[index - num_streams].synchronize()
                stage_stream = streams[index % num_streams]
                with torch.cuda.stream(stage_stream):
                    group = self.pp_comm_groups[stage]
                    for param_info in params:
                        for component in _components(param_info):
                            _recv_one_component(
                                param_info, component, group, stage_stream
                            )
                    event = torch.cuda.Event()
                    event.record()
                    events[index] = event
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        def _receive_misc() -> None:
            misc_t0 = time.perf_counter()
            self._receive_and_load_misc_params()
            torch.cuda.synchronize()
            if torch.distributed.get_rank() == 0:
                print(
                    f"[nccl_reshard_refit] misc recv+load (gen side): "
                    f"{time.perf_counter() - misc_t0:.2f}s",
                    flush=True,
                )
            torch.cuda.empty_cache()

        native_names = _native_mxfp8_param_names(self.nccl_reshard_refit_info)
        if native_names:
            adapter = self._get_nccl_reshard_refit_adapter()
            adapter.begin_update()
            try:
                # Resolving every destination up front ensures a missing role,
                # alias, shape, dtype, or wrapped loader fails before NCCL starts.
                self.hf_to_local_param_map = self.build_hf_to_local_param_map(
                    self.nccl_reshard_refit_info
                )
                _receive_bulk_components()
                _receive_misc()
                adapter.finish_update()
                self._maybe_process_mtp_drafter_after_loading()
                torch.cuda.synchronize()
            except BaseException as error:
                adapter.abort_update(error)
                raise
            torch.cuda.empty_cache()
            return True

        _receive_bulk_components()
        with self._weight_update_lifecycle("nccl_reshard") as finalize:
            _receive_misc()
            finalize()
            torch.cuda.empty_cache()
        return True

    def _receive_and_load_misc_params(self) -> None:
        """Receive misc params via packed_broadcast and load via vLLM."""
        misc_meta = self.nccl_reshard_refit_info.get("misc_meta", {})
        if not misc_meta:
            return

        misc_state_dict_info = {
            name: (tuple(meta["shape"]), _STR_TO_DTYPE[meta["dtype"]])
            for name, meta in misc_meta.items()
        }

        packed_broadcast_consumer(
            iterator=iter(misc_state_dict_info.items()),
            group=self.model_update_group,
            src=0,
            post_unpack_func=self._load_weights,
        )

    def cleanup(self) -> None:
        """Shutdown and cleanup resources."""
        # Close ZMQ socket and context if they exist
        if hasattr(self, "zmq_socket"):
            self.zmq_socket.close()
            self.zmq_context.term()

    def start_gpu_profiling(self) -> None:
        """Start GPU profiling."""
        torch.cuda.profiler.start()

    def stop_gpu_profiling(self) -> None:
        """Stop GPU profiling."""
        torch.cuda.profiler.stop()


class VllmInternalWorkerExtensionWithCheckpointEngine(
    VllmCheckpointEngineMixin, VllmInternalWorkerExtension
):
    """vLLM worker extension with checkpoint-engine refit support."""

    def _validate_checkpoint_engine_weight_update(self) -> None:
        self._reject_unsupported_native_refit("checkpoint_engine")
