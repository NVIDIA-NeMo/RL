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


WeightUpdateTransport = Literal["ipc", "collective"]
WeightUpdateFinalizer = Callable[[], None]


def _format_refit_key_error(label: str, keys: set[str]) -> str:
    """Format a bounded refit-key diagnostic."""
    ordered = sorted(keys)
    suffix = " ..." if len(ordered) > 8 else ""
    return f"{label} ({len(ordered)}): {ordered[:8]}{suffix}"


class IPCWeightManifestError(RuntimeError):
    """An IPC transfer did not match the prepared state-dict manifest."""


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
    # True once the MTP drafter has been served by a one-time disk load (see
    # load_mtp_weights_from_disk); refit then leaves those static weights alone.
    _mtp_drafter_from_disk: bool = False
    _sparse_delta_applier: Any = None
    _nrl_named_parameters: dict[str, torch.nn.Parameter]

    def _get_named_parameters(self) -> dict[str, torch.nn.Parameter]:
        params = getattr(self, "_nrl_named_parameters", None)
        if params is None:
            params = dict(self.model_runner.model.named_parameters())
            self._nrl_named_parameters = params
        return params

    def _load_full_hf_weights(
        self, policy_weights: list[tuple[str, torch.Tensor]]
    ) -> None:
        self.model_runner.model.load_weights(weights=policy_weights)

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
        """
        self.state_dict_info = state_dict_info  # pyrefly: ignore[implicitly-defined-attribute]  This class does not define __init__ so assignments like this should be ignored

    def prepare_sparse_delta_refit_info(
        self, state_dict_info: dict[str, tuple[tuple[int, ...], torch.dtype]]
    ) -> list[str]:
        """Reserve scratch space and report weights that require overwrite."""
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
        if self._mtp_drafter_from_disk:
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

        self._load_draft_weights(weights)

        # The MTP block contains MoE experts whose weights need post-load
        # processing (e.g. grouped-GEMM layout), matching the main-model path.
        from vllm.config import set_current_vllm_config
        from vllm.model_executor.model_loader.utils import (
            process_weights_after_loading,
        )

        draft_model_config = (
            self.model_runner.vllm_config.speculative_config.draft_model_config
        )
        with set_current_vllm_config(self.model_runner.vllm_config):
            process_weights_after_loading(draft_model, draft_model_config, self.device)
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

    @contextmanager
    def _weight_update_lifecycle(
        self, transport: WeightUpdateTransport
    ) -> Iterator[WeightUpdateFinalizer]:
        """Provide setup/finalization around a transport-owned weight update."""
        del transport
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
        return False

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
            with self._weight_update_lifecycle("collective") as finalize:
                packed_broadcast_consumer(
                    iterator=iter(self.state_dict_info.items()),
                    group=self.model_update_group,
                    src=0,
                    post_unpack_func=self._load_weights,
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
            def pre(_base):
                region = vllm_param.data[merged_slice]
                return RefitCtx(buf=torch.empty_like(region), extra={"region": region})

            def post(ctx):
                ctx.extra["region"].copy_(ctx.buf)

            return LocalParamSpec(base=vllm_param, pre=pre, post=post)

        # Get dict of vllm_param and merged_slice for each hf_name
        vllm_param_map_and_slices = self._build_hf_to_gen_backend_mapping(refit_info)
        return HFToLocalParamMap(
            specs={
                hf_name: (
                    LocalParamSpec(base=vllm_param.data)
                    if merged_slice is None
                    else _merged_param_spec(vllm_param, merged_slice)
                )
                for hf_name, (
                    vllm_param,
                    merged_slice,
                ) in vllm_param_map_and_slices.items()
            }
        )

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
        # Module lookup: to detect the selected backend off the FusedMoE layer
        vllm_modules = dict(self.model_runner.model.named_modules())
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

        def _to_vllm_name(n: str) -> str:
            if n in vllm_params:
                return n
            return vllm_by_relative.get(_layer_relative(n), n)

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
        """Receive weights from training workers via xferdtensor.

        Each HF param's ``LocalParamSpec`` (from ``hf_to_local_param_map``,
        built once in ``prepare_nccl_reshard_refit_info``) provides the dst buffer:
        for a direct param xferdtensor receives straight into the live vLLM
        param (no hooks); for a merged param (dense gate_up_proj, grouped w13)
        ``pre`` allocates a temp recv buffer and ``post`` copies the TP-local
        slice back into the live merged param.
        """
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

        num_streams = min(
            int(os.environ.get("NRL_REFIT_NUM_STREAMS", "2")), len(stage_params)
        )

        streams = [torch.cuda.Stream() for _ in range(num_streams)]
        events = {}
        for idx, (stage, params) in enumerate(stage_params.items()):
            # synchronize the last run in the same stream
            if (idx - num_streams) in events:
                events[idx - num_streams].synchronize()
            stage_stream = streams[idx % num_streams]
            with torch.cuda.stream(stage_stream):
                group = self.pp_comm_groups[stage]
                for p in params:
                    _recv_one_param(p, group, stage_stream)
                ev = torch.cuda.Event()
                ev.record()
                events[idx] = ev

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        import time

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
        from vllm.config import set_current_vllm_config
        from vllm.model_executor.model_loader.utils import (
            process_weights_after_loading,
        )

        # Finalize post-load weight processing: dense Linear + attention/MLA, and
        # crucially the per-MoE-backend w13 layout (FlashInfer CUTLASS/TRTLLM) that
        # the canonical [gate; up] bulk write above defers to here.
        with set_current_vllm_config(self.model_runner.vllm_config):
            process_weights_after_loading(
                self.model_runner.model, self.model_config, self.device
            )

        torch.cuda.empty_cache()

        # Finalize FP8 KV-cache per-layer k/v scales after the misc broadcast.
        self._maybe_process_fp8_kv_cache()
        return True

    def _receive_and_load_misc_params(self) -> None:
        """Receive misc params via packed_broadcast and load via vLLM."""
        from nemo_rl.weight_sync.nccl_reshard_utils import _STR_TO_DTYPE

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
