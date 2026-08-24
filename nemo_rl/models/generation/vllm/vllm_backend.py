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
import os
import re
import socket
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any, Literal, Optional

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


WeightUpdateTransport = Literal["ipc", "collective", "nccl_reshard"]
UnsupportedNativeRefitTransport = Literal["checkpoint_engine", "sparse_delta"]
WeightUpdateFinalizer = Callable[[], None]

# Speculative methods whose drafter is co-trained by the trainer and refit
# through the ``draft.*`` weight stream (dspark/dflash block drafters and the
# eagle3 TTT drafter). MTP is co-trained too but streams without the prefix.
COTRAINED_SPECULATIVE_METHODS = ("dspark", "dflash", "eagle3")

# Env flag that requests disable_draft_module_sharing() at import time in
# every process that resolves this module (the driver actor and any spawned
# vLLM executor workers, which import it for worker_extension_cls before
# loading the model).
DRAFT_DISABLE_MODULE_SHARING_ENV = "NRL_DRAFT_DISABLE_MODULE_SHARING"


def disable_draft_module_sharing() -> None:
    """Keep the drafter's embed_tokens/lm_head separate from the target's.

    The pinned vLLM's ``load_dspark_model`` / ``load_dflash_model`` /
    ``load_eagle_model`` alias the drafter's embed_tokens and lm_head modules
    to the target model's whenever the drafter's ``load_weights`` has not
    marked them as owned — which is always the case under
    ``load_format="dummy"``, where no checkpoint weights are read at startup.
    With draft co-training the refit stream carries trained
    ``draft.embed_tokens`` / ``draft.lm_head`` weights, and loading them through
    the aliased modules silently replaces the POLICY model's serving embed and
    lm_head with the draft's. Forcing ``_should_share`` to False makes the
    drafter keep the modules it built in ``__init__``; refit fills them before
    the first generation, and CUDA graphs capture drafter-owned storage.

    ``_should_share`` is defined in the eagle utils module and imported by
    value into the dspark/dflash utils modules, so each module's global must
    be rebound individually.

    Must run before engine creation (drafter load and CUDA-graph capture).
    """
    from vllm.v1.worker.gpu.spec_decode.dflash import utils as dflash_utils
    from vllm.v1.worker.gpu.spec_decode.dspark import utils as dspark_utils
    from vllm.v1.worker.gpu.spec_decode.eagle import utils as eagle_utils

    def _never_share(*args: Any, **kwargs: Any) -> bool:
        return False

    eagle_utils._should_share = _never_share
    dspark_utils._should_share = _never_share
    dflash_utils._should_share = _never_share


if os.environ.get(DRAFT_DISABLE_MODULE_SHARING_ENV) == "1":
    disable_draft_module_sharing()

# Incoming draft stream keys each drafter's loader intentionally skips. They
# are tolerated as extras in the refit manifest and excluded from the required
# key set. dspark/dflash: mask_embedding is a placeholder param, the
# confidence head is not wired into inference (dflash drafts never have one),
# and t2d is training-only. eagle3: t2d is training-only.
_DRAFT_SKIPPED_KEY_SUBSTRINGS = {
    "dspark": ("mask_embedding", "confidence_head", "t2d"),
    "dflash": ("mask_embedding", "confidence_head", "t2d"),
    "eagle3": ("t2d",),
}


def _is_full_eagle3_stream(draft_keys: "set[str] | list[str]") -> bool:
    """Whether an eagle3 ``draft.*`` stream is the DTensor-v2 FULL drafter.

    The DTensor-v2 co-training path streams the vendored model's entire
    state_dict (always including ``embed_tokens``); the megatron eagle3
    exporter intentionally omits ``embed_tokens`` and uses the ``midlayer.*``
    alias, relying on drafter module sharing at serve time.
    """
    return any("embed_tokens" in key for key in draft_keys)


def _format_refit_key_error(label: str, keys: set[str]) -> str:
    """Format a bounded refit-key diagnostic."""
    ordered = sorted(keys)
    suffix = " ..." if len(ordered) > 8 else ""
    return f"{label} ({len(ordered)}): {ordered[:8]}{suffix}"


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


def _model_uses_unquantized_flashinfer_trtllm(model: torch.nn.Module) -> bool:
    """Return whether a model realized the unquantized TRTLLM MoE backend."""
    # Import backend types only when inspecting a constructed vLLM model. The
    # module layout is version-sensitive; on vLLM builds without the oracle
    # package the TRTLLM backend cannot be realized, so absence means False
    # rather than an ImportError on every unquantized-model refit.
    try:
        from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
            UnquantizedMoeBackend,
        )
        from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
            UnquantizedFusedMoEMethod,
        )
    except ImportError:
        return False

    return any(
        isinstance(
            quant_method := getattr(module, "quant_method", None),
            UnquantizedFusedMoEMethod,
        )
        and quant_method.unquantized_backend is UnquantizedMoeBackend.FLASHINFER_TRTLLM
        for module in model.modules()
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
        if match is not None and int(match.group(1)) in mtp_layer_indices:
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
    # True once the MTP drafter has been served by a one-time disk load (see
    # load_mtp_weights_from_disk); refit then leaves those static weights alone.
    _mtp_drafter_from_disk: bool = False
    _sparse_delta_applier: Any = None
    _nrl_named_parameters: dict[str, torch.nn.Parameter]
    _nrl_layerwise_reload_active: bool = False
    # Initialization detaches parameters, so any later failure leaves this
    # worker unsafe to reuse. Keep the original failure for the worker lifetime.
    _nrl_layerwise_reload_failure: Exception | None = None
    # None until init_collective builds it. Declared so a rebuild can release the
    # previous group without probing for the attribute's existence.
    model_update_group: Any = None

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
            fp8.load_weights(policy_weights, self.model_runner)
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
        from nemo_rl.distributed.refit_watchdog import RELEASE_GRACE_S, release_within
        from nemo_rl.distributed.stateless_process_group import StatelessProcessGroup

        # Place vLLM ranks after all training ranks so all training workers can join
        rank = train_world_size + resolve_rollout_rank(
            rank_prefix, world_size - train_world_size
        )

        # The other half of the pair printed on the train side.
        print(
            f"  refit: collective rendezvous [gen] addr={ip}:{port} "
            f"rank={rank} world_size={world_size} "
            f"train_world_size={train_world_size} prefix={rank_prefix}",
            flush=True,
        )

        # Connect before releasing, for the reason the train side binds before releasing:
        # the rendezvous has a 300s budget and the release may never return at all.
        old_group, self.model_update_group = (
            self.model_update_group,
            StatelessProcessGroup(
                master_address=ip, port=port, rank=rank, world_size=world_size
            ),
        )
        # Rebuilding is the recovery path for a dead generation rank, so this runs more
        # than once per job. Without the release, each rebuild would strand the previous
        # NCCL communicator and its TCPStore for the life of the worker. Bounded, because
        # the peer this is releasing may be frozen rather than dead.
        if old_group is not None:
            release_within(
                old_group.abort,
                RELEASE_GRACE_S,
                "the previous refit communicator",
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
        from nemo_rl.distributed.refit_watchdog import RELEASE_GRACE_S, release_within
        from nemo_rl.distributed.stateless_process_group import StatelessProcessGroup

        # Through the same helper init_collective uses, and for the same reason: under
        # vLLM's external data parallelism each engine's torch world spans the whole
        # rollout, so get_rank() is ALREADY global and adding rank_prefix on top counts
        # the shard offset twice. The higher shards then get NCCL ranks past the end of
        # the group. Without external DP the two forms agree, which is why this survived:
        # each engine's world is tp x pp, get_rank() indexes within the shard, and the
        # prefix is exactly what is missing.
        #
        # The group is one PP stage: train_ranks_per_stage trainers followed by every gen
        # rank, so the rollout span is sub_world_size - train_ranks_per_stage.
        gen_rank_in_group = train_ranks_per_stage + resolve_rollout_rank(
            rank_prefix, sub_world_size - train_ranks_per_stage
        )

        # Free cached blocks so NCCL P2P buffers have headroom (see init_collective).
        torch.cuda.empty_cache()
        # Aborted before the dict is dropped, not just dereferenced: garbage collecting
        # the handle does not release the NCCL communicator or unbind the TCPStore, so
        # a reshard run would strand one of each per PP stage per recovery. Deferred past
        # the rendezvous below and bounded, for the reason init_collective binds before it
        # releases: these are the split children's parents, and on a frozen peer their
        # abort is the call that never returns.
        stale_groups = list((self.pp_comm_groups or {}).values())
        self.pp_comm_groups = {}
        for stage in range(pp_size):
            # The other half of the pair printed on the train side. A rendezvous that never
            # comes up shows only a 300s connect timeout naming the address, which does not
            # say whether the master never bound it or the two sides disagreed about
            # world_size or rank. These two lines answer that directly.
            print(
                f"  refit: reshard rendezvous [gen] stage={stage} "
                f"addr={pp_ips[stage]}:{pp_ports[stage]} "
                f"rank={gen_rank_in_group} world_size={sub_world_size}",
                flush=True,
            )
            group = StatelessProcessGroup(
                master_address=pp_ips[stage],
                port=pp_ports[stage],
                rank=gen_rank_in_group,
                world_size=sub_world_size,
            )
            group.init_nccl_communicator(device=self.device)
            self.pp_comm_groups[stage] = group

        for previous in stale_groups:
            release_within(
                previous.abort, RELEASE_GRACE_S, "a previous reshard bulk communicator"
            )

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

    def draft_refit_preflight_report(self) -> dict[str, Any]:
        """JSON-safe snapshot of the drafter refit surface for preflight checks.

        Invoked via ``collective_rpc`` to validate, inside the serving
        container, that the drafter loaded, its expected ``draft.*`` key set
        matches the trainer's stream, integer/bool buffers keep their dtype,
        and no embed/lm_head storage is aliased to the target model.
        """
        method = self._speculative_method()
        draft_model = self._get_drafter_model()
        report: dict[str, Any] = {
            "method": method,
            "owns_speculator": self._draft_owns_speculator(),
            "has_drafter": draft_model is not None,
        }
        if draft_model is None:
            return report
        report["expected_draft_keys"] = sorted(self._expected_draft_keys())
        report["param_dtypes"] = {
            name: str(param.dtype) for name, param in draft_model.named_parameters()
        }

        target_model = self.model_runner.model
        target_lm = (
            target_model.get_language_model()
            if hasattr(target_model, "get_language_model")
            else target_model
        )

        def _ptr(module: Any) -> Optional[int]:
            return module.weight.data_ptr() if module is not None else None

        draft_embed = _ptr(
            getattr(getattr(draft_model, "model", None), "embed_tokens", None)
        )
        target_embed = _ptr(
            getattr(getattr(target_lm, "model", None), "embed_tokens", None)
        )
        draft_head = _ptr(getattr(draft_model, "lm_head", None))
        target_head = _ptr(getattr(target_lm, "lm_head", None))
        report["embed_tokens_aliased"] = (
            draft_embed is not None and draft_embed == target_embed
        )
        report["lm_head_aliased"] = draft_head is not None and draft_head == target_head
        return report

    def prepare_refit_info(self, state_dict_info: dict[str, Any]) -> None:
        """Prepare state dict metadata for weight refitting and IPC streaming.

        For co-trained speculative decoding (dspark/dflash/eagle3), the
        trainer-provided ``draft.*`` keys are validated here against the
        drafter's own loadable layout. Combined
        with the per-refit manifests (the IPC path enforces every
        ``state_dict_info`` key exactly once, and the collective path iterates
        ``state_dict_info`` by construction), this proves on every transport
        that the full expected draft key set is delivered.

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
        self._validate_draft_refit_info(state_dict_info)

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

        Megatron pads vocab to a multiple, but vLLM's autoloader
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

    def _speculative_method(self) -> Optional[str]:
        spec_config = getattr(self.model_runner.vllm_config, "speculative_config", None)
        return getattr(spec_config, "method", None) if spec_config else None

    def _draft_owns_speculator(self) -> bool:
        """Whether this rank should own the co-trained speculator.

        vLLM keeps the drafter on the last pipeline stage, so earlier stages
        legitimately have no speculator and must skip draft payloads; every
        rank owns it in single-stage layouts.
        """
        if self._speculative_method() not in COTRAINED_SPECULATIVE_METHODS:
            return False
        try:
            return bool(get_pp_group().is_last_rank)
        except Exception:
            # No initialized PP group (single-process layouts, unit tests).
            return True

    def _expected_draft_keys(self) -> set[str]:
        """Expected incoming ``draft.*`` keys derived from the drafter's layout.

        Inverts the drafter ``load_weights`` name handling shared by
        ``Qwen3DSparkForCausalLM`` / ``DFlashQwen3ForCausalLM`` /
        ``Eagle3Qwen3ForCausalLM``: trainer names are prefixed with ``model.``
        (except ``lm_head.*``, and ``d2t`` which maps to
        ``draft_id_to_target_id``), and fused parameters load from their
        stacked components (qkv_proj <- q/k/v_proj, gate_up_proj <-
        gate/up_proj). Parameters the loader never feeds from the stream (see
        _DRAFT_SKIPPED_KEY_SUBSTRINGS) are excluded.
        """
        draft_model = self._get_drafter_model()
        if draft_model is None:
            raise RuntimeError(
                "[draft] Draft refit validation requires the drafter model, but "
                "none was found at model_runner.drafter.model or "
                "model_runner.speculator.model on a speculator-owning rank."
            )
        fused_expansions = {
            "qkv_proj": ("q_proj", "k_proj", "v_proj"),
            "gate_up_proj": ("gate_proj", "up_proj"),
        }
        method = self._speculative_method()
        if method is None:
            raise RuntimeError(
                "[draft] Expected draft keys require an active "
                "speculative_config method, but none is configured."
            )
        skipped = _DRAFT_SKIPPED_KEY_SUBSTRINGS[method]
        expected: set[str] = set()
        for name, _ in draft_model.named_parameters():
            if any(s in name for s in skipped):
                continue
            if name == "draft_id_to_target_id":
                expected.add("draft.d2t")
                continue
            trainer_name = name.removeprefix("model.")
            segments = trainer_name.split(".")
            fused = next((s for s in segments if s in fused_expansions), None)
            if fused is None:
                expected.add(f"draft.{trainer_name}")
            else:
                for part in fused_expansions[fused]:
                    expected.add(
                        "draft." + ".".join(part if s == fused else s for s in segments)
                    )
        return expected

    def _validate_draft_refit_info(self, state_dict_info: dict[str, Any]) -> None:
        """Hard-error when the trainer's ``draft.*`` manifest mismatches the drafter.

        Owning ranks require exactly the drafter's loadable key set (plus keys
        the loader intentionally skips); non-owning ranks ignore draft payloads
        entirely and are never required to have a drafter.
        """
        method = self._speculative_method()
        if method is None or method not in COTRAINED_SPECULATIVE_METHODS:
            return
        if not self._draft_owns_speculator():
            return
        provided = {k for k in state_dict_info if k.startswith("draft.")}
        if not provided:
            if os.environ.get(DRAFT_DISABLE_MODULE_SHARING_ENV) == "1":
                # The generation worker sets this env exactly when full-draft
                # co-training refit is enabled (dummy-loaded drafter waiting
                # for streamed weights). An empty draft manifest here means the
                # trainer failed to export the draft — serving would silently
                # run a stale (dummy-initialized) drafter forever.
                raise RuntimeError(
                    f"[draft] {method} co-training refit is enabled "
                    f"({DRAFT_DISABLE_MODULE_SHARING_ENV}=1) but the refit "
                    "manifest carries no draft.* keys; the trainer-side draft "
                    "export is missing or misconfigured."
                )
            # Static-drafter mode: with policy.draft.enabled=false the drafter
            # is loaded from its checkpoint at startup and refits legitimately
            # carry only policy weights. Exact-key validation applies only when
            # the trainer co-trains (and therefore streams) the draft.
            return
        if method == "eagle3" and not _is_full_eagle3_stream(provided):
            # Megatron eagle3 co-training streams a PARTIAL drafter (no
            # embed_tokens, midlayer.* alias for the single layer) and relies
            # on the drafter sharing the target's embedding; exact-key
            # validation against the vLLM parameter layout only applies to
            # the DTensor-v2 full stream.
            return
        expected = self._expected_draft_keys()
        skipped = _DRAFT_SKIPPED_KEY_SUBSTRINGS[method]
        missing = expected - provided
        unexpected = {
            key for key in provided - expected if not any(s in key for s in skipped)
        }
        errors = []
        if missing:
            errors.append(_format_refit_key_error("missing draft keys", missing))
        if unexpected:
            errors.append(_format_refit_key_error("unexpected draft keys", unexpected))
        if errors:
            raise RuntimeError(
                f"[draft] {method} refit manifest does not match the vLLM "
                "drafter layout: " + "; ".join(errors)
            )

    def _get_drafter_model(self) -> Any:
        """Return the vLLM drafter's underlying model, or None if absent.

        The drafter holds the speculative-decoding draft model (Eagle3, MTP, or
        DSpark), which vLLM keeps as a module separate from the main model. The
        eagle3 layout exposes it at ``model_runner.drafter.model``; the newer
        gpu-worker speculator layout (used by DSpark) at
        ``model_runner.speculator.model``. Typed ``Any`` because these are
        dynamic vLLM model classes whose ``load_weights`` / ``mtp_start_layer_idx``
        members are not visible through ``nn.Module``.
        """
        for owner_attr in ("drafter", "speculator"):
            draft_owner = getattr(self.model_runner, owner_attr, None)
            draft_model = getattr(draft_owner, "model", None) if draft_owner else None
            if draft_model is not None:
                return draft_model
        return None

    def _load_draft_weights(
        self, draft_weights: list[tuple[str, torch.Tensor]]
    ) -> None:
        if not draft_weights:
            return

        method = self._speculative_method()
        # The megatron eagle3 partial stream predates (and must keep) the
        # lenient path: no alias guard, warn-and-skip on a missing drafter.
        strict_cotraining = method in ("dspark", "dflash") or (
            method == "eagle3"
            and _is_full_eagle3_stream([name for name, _ in draft_weights])
        )
        draft_model = self._get_drafter_model()
        if draft_model is None:
            if strict_cotraining:
                if self._draft_owns_speculator():
                    # Draft co-training streams every draft weight on each
                    # refit; an owning rank without a speculator would silently
                    # generate with stale draft weights.
                    raise RuntimeError(
                        f"[draft] Received {method} draft weights but no "
                        "drafter model was found at model_runner.drafter.model "
                        "or model_runner.speculator.model. The pinned vLLM's "
                        f"{method} speculator layout may have changed."
                    )
                # Non-owning pipeline stages legitimately have no speculator;
                # draft payloads are not theirs to load.
                return
            logger.warning(
                "[draft] Received draft weights but vLLM drafter is unavailable; skipping draft update."
            )
            return
        if strict_cotraining:
            self._assert_drafter_owns_modules(draft_model, draft_weights)
        draft_weights = self._trim_vocab_padding(draft_model, draft_weights)
        draft_model.load_weights(weights=draft_weights)

    def _assert_drafter_owns_modules(
        self, draft_model: Any, draft_weights: list[tuple[str, torch.Tensor]]
    ) -> None:
        """Refuse to load draft embed/lm_head through target-shared modules.

        vLLM's drafter loaders may alias the drafter's embed_tokens and
        lm_head to the target model's modules as a weight-sharing optimization;
        loading refit draft weights through such an alias would overwrite the
        policy's serving weights with the draft's (see
        ``disable_draft_module_sharing``). Guard every refit so any path
        that reintroduces the sharing fails loudly instead of silently
        corrupting generation.
        """
        target_model = self.model_runner.model
        target_lm = (
            target_model.get_language_model()
            if hasattr(target_model, "get_language_model")
            else target_model
        )
        refits_embed = any("embed_tokens" in name for name, _ in draft_weights)
        refits_lm_head = any("lm_head" in name for name, _ in draft_weights)

        def _shares_weight(draft_module: Any, target_module: Any) -> bool:
            return (
                draft_module is not None
                and target_module is not None
                and draft_module.weight.data_ptr() == target_module.weight.data_ptr()
            )

        shared = []
        if refits_embed and _shares_weight(
            getattr(getattr(draft_model, "model", None), "embed_tokens", None),
            getattr(getattr(target_lm, "model", None), "embed_tokens", None),
        ):
            shared.append("embed_tokens")
        if refits_lm_head and _shares_weight(
            getattr(draft_model, "lm_head", None),
            getattr(target_lm, "lm_head", None),
        ):
            shared.append("lm_head")
        if shared:
            raise RuntimeError(
                "[draft] The drafter's "
                + "/".join(shared)
                + " share storage with the target model; loading refit draft "
                "weights through the alias would overwrite the policy's serving "
                "weights. Ensure disable_draft_module_sharing() ran "
                "before engine creation "
                f"({DRAFT_DISABLE_MODULE_SHARING_ENV}=1)."
            )

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
        if self._mtp_drafter_from_disk:
            return False
        if self._speculative_method() not in ("deepseek_mtp", "mtp"):
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
        # Mark that the MTP drafter is served from a one-time disk load so refit
        # does not re-load or re-process these static weights.
        self._mtp_drafter_from_disk = True
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
        vllm_config = getattr(model_runner, "vllm_config", None)
        if vllm_config is None:
            return False
        if getattr(vllm_config, "quant_config", None) is not None:
            return False

        return _model_uses_unquantized_flashinfer_trtllm(self.model_runner.model)

    def _uses_native_layerwise_refit(self, transport: WeightUpdateTransport) -> bool:
        """Return whether this transport needs vLLM's layerwise lifecycle."""
        return transport in ("ipc", "collective") and (
            self._uses_unquantized_flashinfer_trtllm()
        )

    def _validate_native_layerwise_refit(self) -> None:
        """Reject unsupported features on the native layerwise reload path."""
        if not self._uses_unquantized_flashinfer_trtllm():
            return

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
            self._validate_native_layerwise_refit()
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

            def finalize() -> None:
                with torch.device(self.device):
                    finalize_layerwise_reload(model, self.model_config)
                    _refresh_hpc_modules_after_layerwise_reload(model)
                    self._maybe_process_mtp_drafter_after_loading()
                torch.cuda.synchronize()

            try:
                with set_current_vllm_config(self.model_runner.vllm_config):
                    with torch.device(self.device):
                        initialize_layerwise_reload(model)
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
    def update_weights_from_collective(
        self, refit_timeout_s: float | None = None
    ) -> bool:
        """Update the model weights from collective communication.

        Guarded for the same reason as the producer side: if a peer rank dies mid-refit
        this blocks in NCCL forever. Note the buffers hold PARTIAL weights once aborted,
        so the caller must not serve from this engine until a later refit completes.
        """
        from nemo_rl.distributed.refit_watchdog import (
            RefitAborted,
            RefitAbortWatchdog,
            hold_refit_for_fault_injection,
        )

        with RefitAbortWatchdog(self.model_update_group, refit_timeout_s) as guard:
            hold_refit_for_fault_injection()
            result = self._update_weights_from_collective()
        if guard.fired:
            raise RefitAborted(
                f"refit receive exceeded {refit_timeout_s}s and was aborted; "
                "this engine now holds partial weights and must not serve until refit"
            )
        return result

    def _update_weights_from_collective(self) -> bool:
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
        """Restore per-layer param metadata and build the HF→vLLM mapping.

        Done once ahead of refit; the cached mapping is reused by every
        ``nccl_reshard_refit`` call.
        """
        from nemo_rl.weight_sync.nccl_reshard_utils import (
            restore_refit_info_placements,
        )

        self.nccl_reshard_refit_info = (  # pyrefly: ignore[implicitly-defined-attribute]
            restore_refit_info_placements(refit_info)
        )
        # Build HFToLocalParamMap (see nccl_reshard_utils)
        self.hf_to_local_param_map = self.build_hf_to_local_param_map(  # pyrefly: ignore[implicitly-defined-attribute]
            self.nccl_reshard_refit_info
        )

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

        # Get dict of vllm_param and merged_slice for each hf_name
        vllm_param_map_and_slices = self._build_hf_to_gen_backend_mapping(refit_info)
        param_info_by_name = {
            param_info["name"]: param_info
            for layer_name in refit_info["layer_names"]
            for param_info in refit_info["per_layer_params"][layer_name]
        }
        vllm_params = dict(self.model_runner.model.named_parameters())
        vllm_names_by_id = {id(param): name for name, param in vllm_params.items()}
        specs = {}
        for hf_name, (vllm_param, merged_slice) in vllm_param_map_and_slices.items():
            wire_dtype_value = param_info_by_name[hf_name].get("dtype")
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
        mapping = {}

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

    def nccl_reshard_refit(self, refit_timeout_s: float | None = None) -> bool:
        """Receive weights from training workers via xferdtensor, under a deadline.

        Guarded like the collective receive: a peer dying mid-refit blocks this in NCCL
        forever, and the buffers hold PARTIAL weights once aborted, so the caller must
        not serve from this engine until a later refit completes.

        Both communicator families are handed to the watchdog -- the per-PP-stage bulk
        groups and the shared model_update_group -- because the transfer uses them in
        sequence and a hang can be in either.

        Each HF param's ``LocalParamSpec`` (from ``hf_to_local_param_map``,
        built once in ``prepare_nccl_reshard_refit_info``) provides the dst buffer:
        for a direct param xferdtensor receives straight into the live vLLM
        param (no hooks); for a merged param (dense gate_up_proj, grouped w13)
        ``pre`` allocates a temp recv buffer and ``post`` copies the TP-local
        slice back into the live merged param.
        """
        from nemo_rl.distributed.refit_watchdog import (
            RefitAborted,
            RefitAbortWatchdog,
            hold_refit_for_fault_injection,
        )

        groups = [
            *(self.pp_comm_groups or {}).values(),
            self.model_update_group,
        ]
        with RefitAbortWatchdog(groups, refit_timeout_s) as guard:
            hold_refit_for_fault_injection()
            result = self._nccl_reshard_refit()
        if guard.fired:
            raise RefitAborted(
                f"refit nccl_reshard receive exceeded {refit_timeout_s}s and was "
                "aborted; this engine now holds partial weights and must not serve "
                "until refit"
            )
        return result

    def _nccl_reshard_refit(self) -> bool:
        import os
        from collections import OrderedDict

        from nemo_rl.weight_sync.xferdtensor import DTensorRef, xferdtensor

        def _recv_one_param(param_info, group, stream):
            # Coverage guard: every bulk param must have a spec; a missing entry
            # would silently discard its weights.
            spec = self.hf_to_local_param_map.get(param_info["name"])
            assert spec is not None, (
                f"nccl_reshard_refit: {param_info['name']!r} has no spec in "
                "hf_to_local_param_map (would silently discard its weights)"
            )
            # spec.pre/post run on the caller's current stream (this stage's
            # stream); xferdtensor should use the same stream.
            ctx = (
                spec.pre(spec.base) if spec.pre is not None else RefitCtx(buf=spec.base)
            )
            dst_tensor = DTensorRef(ctx.buf, param_info["global_shape"])
            xferdtensor(
                None,
                param_info["src_mesh_info"],
                param_info["src_placements"],
                dst_tensor,
                param_info["dst_mesh_info"],
                param_info["dst_placements"],
                group,
                stream,
            )
            if spec.post is not None:
                spec.post(ctx)

        # Group params by PP stage so different stages' bulk reshards run
        # concurrently on their own streams.  Non-PP = single stage 0 (params
        # carry no "pp_stage" key), so this collapses to one stage / one stream.
        stage_params = OrderedDict()
        for layer_name in self.nccl_reshard_refit_info["layer_names"]:
            for p in self.nccl_reshard_refit_info["per_layer_params"][layer_name]:
                stage_params.setdefault(p.get("pp_stage", 0), []).append(p)

        num_streams = max(
            1,
            min(int(os.environ.get("NRL_REFIT_NUM_STREAMS", "2")), len(stage_params)),
        )

        # Narrowed once here rather than at each use: reaching this without the groups
        # built is a wiring error, and a named failure beats a TypeError on a subscript.
        pp_comm_groups = self.pp_comm_groups
        if pp_comm_groups is None:
            raise RuntimeError(
                "nccl_reshard refit reached before init_nccl_reshard_comm_group built "
                "the per-PP-stage groups"
            )

        streams = [torch.cuda.Stream() for _ in range(num_streams)]
        events = {}
        for idx, (stage, params) in enumerate(stage_params.items()):
            # synchronize the last run in the same stream
            if (idx - num_streams) in events:
                events[idx - num_streams].synchronize()
            stage_stream = streams[idx % num_streams]
            with torch.cuda.stream(stage_stream):
                group = pp_comm_groups[stage]
                for p in params:
                    _recv_one_param(p, group, stage_stream)
                ev = torch.cuda.Event()
                ev.record()
                events[idx] = ev

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        import time

        with self._weight_update_lifecycle("nccl_reshard") as finalize:
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

            # Finalize post-load weight processing: dense Linear + attention/MLA,
            # the per-MoE-backend w13 layout (FlashInfer CUTLASS/TRTLLM) that the
            # canonical [gate; up] bulk write above defers to here, and the MTP
            # drafter's mirror of the same. The FP8 KV-cache per-layer k/v scales
            # are finalized by the lifecycle on exit.
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
