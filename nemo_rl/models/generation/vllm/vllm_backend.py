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
import re
import socket
import traceback
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any, Literal

import torch
import zmq

from nemo_rl.models.policy.utils import (
    IPCProtocol,
    calculate_aligned_size,
    rebuild_cuda_tensor_from_ipc,
)
from nemo_rl.utils.nsys import wrap_with_nvtx_name
from nemo_rl.utils.packed_tensor import packed_broadcast_consumer

try:
    import vllm  # noqa: F401
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
    _sparse_delta_applier: Any = None

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

        local_rank = torch.distributed.get_rank()
        # Place vLLM ranks after all training ranks so all training workers can join
        rank = train_world_size + rank_prefix + local_rank

        self.model_update_group = StatelessProcessGroup(  # pyrefly: ignore[implicitly-defined-attribute]  This class does not define __init__ so assignments like this should be ignored
            master_address=ip, port=port, rank=rank, world_size=world_size
        )
        self.model_update_group.init_nccl_communicator(device=self.device)

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
        them to vLLM.
        This implementation is specific to the eagle model. For MTP, we can add
        similar logic to this function to split weights and send it to the drafter.
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

    def _load_draft_weights(
        self, draft_weights: list[tuple[str, torch.Tensor]]
    ) -> None:
        if not draft_weights:
            return

        draft_owner = getattr(self.model_runner, "drafter", None)
        draft_model = getattr(draft_owner, "model", None) if draft_owner else None

        if draft_model is None:
            print(
                "[draft] Received draft weights but vLLM drafter is unavailable; skipping draft update."
            )
            return
        draft_weights = self._trim_vocab_padding(draft_model, draft_weights)
        draft_model.load_weights(weights=draft_weights)

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
        draft_owner = getattr(self.model_runner, "drafter", None)
        draft_model = getattr(draft_owner, "model", None) if draft_owner else None
        if draft_model is None:
            print("[mtp] Drafter unavailable; cannot load MTP weights from disk.")
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
        print(
            f"[mtp] Loaded MTP draft weights for layers "
            f"{sorted(mtp_layer_indices)} from {model_path}"
        )
        return True

    def _load_weights(self, weights):
        """Load weights with Gemma3 vision-tower weight name fix, FP8, and draft-weight support.

        Applies Gemma3 vision-tower weight name fix if needed, splits policy/draft
        weights, applies FP8 conversion if needed, and loads draft weights
        into the drafter model.
        """
        from nemo_rl.models.generation.vllm.quantization import fp8

        if (
            "Gemma3ForConditionalGeneration"
            in self.model_runner.vllm_config.model_config.architectures
        ):
            for idx, (key, weight) in enumerate(weights):
                weights[idx] = (fix_gemma3_vision_weight_name(key), weight)

        policy_weights, draft_weights = self._split_policy_and_draft_weights(weights)
        if fp8.is_fp8_model(self.model_runner.vllm_config):
            fp8.load_weights(policy_weights, self.model_runner)
        else:
            self.model_runner.model.load_weights(weights=policy_weights)

        self._load_draft_weights(draft_weights)

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

        yield finalize
        # KV-cache scales are covered by the full process_weights_after_loading
        # pass in finalize(); no second pass is needed.

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
            print(
                f"Error in VllmInternalWorkerExtension.update_weights_via_ipc_zmq: {e}.\n"
                f"{traceback.format_exc()}"
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
            print(
                f"Error in VllmInternalWorkerExtension.update_weights_from_collective: {e}"
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
