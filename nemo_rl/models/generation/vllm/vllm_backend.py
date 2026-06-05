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
import traceback
from typing import Any

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


def fix_gpt_oss_export_transpose(key: str, weight: torch.Tensor) -> torch.Tensor:
    """Apply GPT-OSS down_proj transpose fix to the weight.

    This is a workaround for the issue that the down_proj layout is not the same across different frameworks.
        - HF needs [in, out] layout.
        - Megatron needs [in, out] layout.
        - vLLM needs [out, in] layout.
    See https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/3271 for more details.
    """
    if key.endswith("mlp.experts.down_proj"):
        weight = weight.transpose(-2, -1).contiguous()
    return weight


class VllmInternalWorkerExtension:
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

    def _maybe_process_fp8_kv_cache(self) -> None:
        """Process weights after loading for FP8 KV cache (static scales)."""
        use_fp8_kv_cache = False
        if hasattr(self.model_runner.vllm_config, "cache_config"):
            kv_cache_dtype = getattr(
                self.model_runner.vllm_config.cache_config, "cache_dtype", None
            )
            use_fp8_kv_cache = (
                kv_cache_dtype is not None and "fp8" in str(kv_cache_dtype).lower()
            )

        if not use_fp8_kv_cache:
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

    def _load_weights(self, weights):
        """Load weights with GptOss transpose fix, FP8, and draft-weight support.

        Applies GPT-OSS down_proj transpose if needed, splits policy/draft
        weights, applies FP8 conversion if needed, and loads draft weights
        into the drafter model.
        """
        from nemo_rl.models.generation.vllm.quantization import fp8

        if (
            "GptOssForCausalLM"
            in self.model_runner.vllm_config.model_config.architectures
        ):
            for idx, (key, weight) in enumerate(weights):
                weight = fix_gpt_oss_export_transpose(key, weight)
                weights[idx] = (key, weight)

        policy_weights, draft_weights = self._split_policy_and_draft_weights(weights)
        if fp8.is_fp8_model(self.model_runner.vllm_config):
            fp8.load_weights(policy_weights, self.model_runner)
        else:
            self.model_runner.model.load_weights(weights=policy_weights)

        self._load_draft_weights(draft_weights)

    @wrap_with_nvtx_name("vllm_internal_worker_extension/update_weights_via_ipc_zmq")
    def update_weights_via_ipc_zmq(self) -> bool:
        """Receive and update model weights via ZMQ IPC socket.

        Returns:
            bool: True if weights were successfully updated.
        """
        buffer = None
        weights = None

        try:
            self.maybe_init_zmq()
            while True:
                # Blocking receive with timeout (this is the main operation)
                payload = self.zmq_socket.recv_pyobj()

                if payload == IPCProtocol.COMPLETE:
                    # means the update is done
                    from vllm.config import set_current_vllm_config
                    from vllm.model_executor.model_loader.utils import (
                        process_weights_after_loading,
                    )

                    with set_current_vllm_config(self.model_runner.vllm_config):
                        process_weights_after_loading(
                            self.model_runner.model, self.model_config, self.device
                        )
                    self.zmq_socket.send(IPCProtocol.ACK.value.encode())
                    break

                ipc_handle, list_keys, used_bytes = payload
                buffer = rebuild_cuda_tensor_from_ipc(ipc_handle, self.device.index)

                weight = None
                weights = []
                offset = 0
                for key in list_keys:
                    shape, dtype = self.state_dict_info[key]  # pyrefly
                    if isinstance(shape, list):
                        shape = torch.Size(shape)

                    # Get the weight from the buffer
                    size_in_bytes = dtype.itemsize * shape.numel()
                    weight = (
                        buffer[offset : offset + size_in_bytes]
                        .view(dtype=dtype)
                        .view(shape)
                    )
                    # apply gpt-oss transpose fix
                    if (
                        "GptOssForCausalLM"
                        in self.model_runner.vllm_config.model_config.architectures
                    ):
                        weight = fix_gpt_oss_export_transpose(key, weight)
                    weights.append((key, weight))

                    # Move offset to the next weight
                    aligned_size = calculate_aligned_size(size_in_bytes)
                    offset += aligned_size

                assert offset == used_bytes, (
                    "Offset is not equal to used bytes, usually indicate inaccurate info like keys or cached dtype in state_dict_info"
                )

                # Load weights into the model
                self._load_weights(weights)

                torch.cuda.current_stream().synchronize()

                # CRITICAL: Delete views before ACK to prevent corruption.
                # 'weights' contains views into IPC shared memory. Even though load_weights()
                # copied the data, Python may not garbage collect these view objects immediately.
                # If sender reuses the buffer before GC runs, old views would read corrupted data.
                # Explicit del ensures immediate cleanup before sending ACK.
                del weight, weights, buffer
                weight = None
                weights = None
                buffer = None
                self.zmq_socket.send(IPCProtocol.ACK.value.encode())

            # Process weights after loading for FP8 KV cache
            self._maybe_process_fp8_kv_cache()

            gc.collect()
            torch.cuda.empty_cache()
            return True
        except Exception as e:
            print(
                f"Error in VllmInternalWorkerExtension.update_weights_via_ipc_zmq: {e}.\n"
                f"{traceback.format_exc()}"
            )
            return False

    @wrap_with_nvtx_name("vllm_internal_worker_extension/update_weights_via_mx")
    def update_weights_via_mx(self, *, version: int, mx_config: Any) -> bool:
        """Receive weights via NIXL RDMA from MX server (v2 path).

        Lazy-creates an :class:`MxV2RefitReceiver`, registers our model's
        live parameters once, then for each version: discover same-rank
        source, RDMA receive, slice into per-name views via the trainer's
        published shape registry, hand off to ``_load_weights``, and
        (optionally) republish self as an inference replica for tree
        fan-out.

        Megatron-MX path: if the discovered source's v2 metadata carries
        ``publisher_kind == "megatron"`` (set by Megatron-Core trainers
        per ``modelexpress.megatron_translator.SIDECAR_*`` keys), route
        through :func:`_update_weights_via_mx_megatron` instead. The
        Megatron path uses the receiver-side slice planner +
        Bridge-shaped translator (``modelexpress.megatron_translator``)
        to assemble per-rank shards into HF tensors via the vendored
        QKV un-interleave + gated-MLP split helpers. The translator
        does not depend on Megatron-Bridge being installed in the
        worker image.

        Returns ``True`` on successful refit.
        """
        try:
            assert self.state_dict_info is not None, (
                "state_dict_info not prepared; call prepare_refit_info() first"
            )

            # First-cycle Megatron check: peek at any cached candidates we
            # may have, otherwise discover for the megatron-mode flag and
            # cache. The Megatron path has its own discover/plan loop, so
            # we only do enough discover here to detect the publisher kind.
            if not hasattr(self, "_mx_megatron_mode"):
                self._mx_megatron_mode = None  # None = unknown, True/False = latched

            # ---- Lazy-init receiver and register receive buffers (once) ----
            if not hasattr(self, "_mx_receiver") or self._mx_receiver is None:
                from nemo_rl.distributed.mx_helpers import build_v2_receiver

                rank = (
                    torch.distributed.get_rank()
                    if torch.distributed.is_initialized()
                    else 0
                )
                self._mx_receiver = build_v2_receiver(
                    rank=rank,
                    device_id=self.device.index,
                    mx_config=mx_config,
                )

                # Build receive buffer dict from current model parameters.
                # The trainer publishes local DTensor shards; on the inference
                # side we want vLLM's already-allocated parameters as the
                # destination buffers (no extra copy). vLLM stores them on
                # ``self.model_runner.model``; iterate named_parameters to get
                # them.
                receive_buffers = {
                    name: p.data
                    for name, p in self.model_runner.model.named_parameters()
                    if p.is_cuda
                }
                self._mx_receiver.initialize(model_tensors=receive_buffers)
                self._mx_recv_buffers = receive_buffers

            # ---- Discover, pick, and pull ----
            candidates = self._mx_receiver.discover_v2_sources(
                model_name=self.model_config.model
                if hasattr(self.model_config, "model")
                else getattr(self.model_runner.vllm_config.model_config, "model", "unknown"),
                min_version=int(version),
                same_rank_only=mx_config.same_rank_only,
                include_replicas=mx_config.tree_scale_out,
            )
            if not candidates:
                print(
                    f"[mx] no v2 source available for version>={version} on rank "
                    f"{self._mx_receiver.worker_rank}"
                )
                return False

            # Latch the receiver mode on the first non-empty discovery.
            if self._mx_megatron_mode is None:
                self._mx_megatron_mode = any(
                    c.megatron_meta is not None for c in candidates
                )
                if self._mx_megatron_mode:
                    print(
                        f"[mx] rank={self._mx_receiver.worker_rank} latched "
                        f"Megatron-MX receiver mode (sources advertise "
                        f"publisher_kind=megatron)"
                    )

            if self._mx_megatron_mode:
                return self._update_weights_via_mx_megatron(
                    candidates=candidates, version=int(version), mx_config=mx_config,
                )

            chosen = self._mx_receiver.pick_best_source(candidates)
            if chosen is None:
                print(
                    f"[mx] no candidate covers required experts on rank "
                    f"{self._mx_receiver.worker_rank}"
                )
                return False
            print(
                f"[mx] rank={self._mx_receiver.worker_rank} chosen source "
                f"role={chosen.role} src_rank={chosen.worker_rank} "
                f"version={chosen.ref.training_step}"
            )

            # Drain RDMA receive into our pre-registered buffers.
            for _name, _tensor in self._mx_receiver.receive_from(
                chosen, timeout_seconds=mx_config.timeout_seconds
            ):
                # The yielded tensor is a view into the same buffer we
                # registered. vLLM's model parameters now hold the new bytes;
                # we still call _load_weights below for FP8 / GptOss /
                # draft-weight handling.
                pass

            # Build (name, weight) pairs for _load_weights from buffers.
            weights = []
            for name, buf in self._mx_recv_buffers.items():
                w = buf
                # apply gpt-oss transpose fix on the way in
                if (
                    "GptOssForCausalLM"
                    in self.model_runner.vllm_config.model_config.architectures
                ):
                    w = fix_gpt_oss_export_transpose(name, w)
                weights.append((name, w))

            self._load_weights(weights)
            torch.cuda.current_stream().synchronize()

            # FP8 KV cache hook reuse
            self._maybe_process_fp8_kv_cache()

            # ---- Tree fan-out: republish self as inference_replica ----
            if mx_config.tree_scale_out:
                self._mx_receiver.publish_self_as_source(
                    version=int(version),
                    model_name=self.model_config.model
                    if hasattr(self.model_config, "model")
                    else getattr(
                        self.model_runner.vllm_config.model_config,
                        "model",
                        "unknown",
                    ),
                )

            gc.collect()
            torch.cuda.empty_cache()
            return True
        except Exception as e:
            print(
                f"Error in VllmInternalWorkerExtension.update_weights_via_mx: {e}\n"
                f"{traceback.format_exc()}"
            )
            return False

    @wrap_with_nvtx_name(
        "vllm_internal_worker_extension/update_weights_via_mx_megatron"
    )
    def _update_weights_via_mx_megatron(
        self,
        *,
        candidates: list,
        version: int,
        mx_config: Any,
    ) -> bool:
        """Megatron-MX path of :meth:`update_weights_via_mx`.

        Routes through ``modelexpress.megatron_translator``'s slice
        planner + assembly pipeline. The trainer publishes per-rank
        Megatron-native shards (no allgather); we discover the slice
        plan, pull each rank's contribution into a pre-allocated global
        tensor, and apply role-aware translation (QKV un-interleave,
        gated-MLP split, name remap) using the vendored helpers — no
        Megatron-Bridge import required in the worker image.

        See ``temp/NemoRL_Megatron_MX_Design.md`` §6 + §9b and
        ``temp/NemoRL_Megatron_MX_Phase_C_Handoff.md``.
        """
        from modelexpress.megatron_translator import (
            MegatronReceiverContext,
            ReceiveSpec,
            discover_megatron_context,
            run_refit_cycle,
        )
        from modelexpress.nemo_rl_v2 import TargetTpLayout

        # ---- One-shot: build context from the first cycle's metadata. ----
        if not hasattr(self, "_mx_megatron_ctx") or self._mx_megatron_ctx is None:
            cfg, name_map = discover_megatron_context(candidates)
            if cfg is None:
                print(
                    "[mx-megatron] sources advertise publisher_kind=megatron but "
                    "no transformer_config sidecar; falling back to non-Megatron "
                    "path on next cycle"
                )
                self._mx_megatron_mode = False
                return False

            # Build receive specs: one per Megatron tensor name in the
            # sidecar's name_map. The receiver's TARGET layout is its own
            # vLLM TP × EP shape.
            target_tp = getattr(self.parallel_config, "tensor_parallel_size", 1)
            target_tp_rank = (
                torch.distributed.get_rank(
                    group=getattr(self, "_tp_process_group", None)
                )
                if torch.distributed.is_initialized()
                else 0
            )
            target_tp_layout = TargetTpLayout(
                tp_size=target_tp, tp_rank=target_tp_rank,
            )

            # For each Megatron source-name → list of HF target names, build
            # one ReceiveSpec. Shape + role come from the source's
            # TensorDescriptorV2 in the published shape_registry; the
            # receiver-side parser is in modelexpress.nemo_rl_v2.
            receive_specs: dict[str, ReceiveSpec] = {}
            for cand in candidates:
                if cand.megatron_meta is None or cand.registry is None:
                    continue
                for td in cand.registry.get("tensors", []):
                    if td.name in receive_specs:
                        continue
                    role = td.megatron_role or ""
                    if not role:
                        continue
                    hf_names = name_map.get(td.name, [td.name])
                    receive_specs[td.name] = ReceiveSpec(
                        megatron_name=td.name,
                        hf_names=list(hf_names),
                        role=role,
                        target_shape=tuple(int(s) for s in td.global_shape),
                        target_dtype=td.dtype,
                        shard_axis=int(td.shard_axis),
                        pp_rank=cand.megatron_meta.pp_rank,
                        role_descriptor=dict(td.megatron_extras or {}),
                    )

            self._mx_megatron_ctx = MegatronReceiverContext(
                target_tp_layout=target_tp_layout,
                transformer_config=cfg,
                hf_name_map=name_map,
                receive_specs=receive_specs,
            )
            print(
                f"[mx-megatron] built receive context: tp={target_tp} "
                f"tensors={len(receive_specs)} cfg={cfg}"
            )

        # ---- One refit cycle. ----
        # Pull callback wraps the receiver's existing per-tensor pull.
        # For v0 we synchronously pull each source's bytes; the
        # parallel-pull optimization hooks in the same place.
        def _pull(src, dest):
            self._mx_receiver._receiver._nixl.pull_to(  # type: ignore[attr-defined]
                src.mx_source_id, src.worker_id, dest,
                source_subslice=src.source_subslice,
                timeout_seconds=mx_config.timeout_seconds,
            )

        weights: list[tuple[str, "torch.Tensor"]] = []
        for hf_name, hf_tensor in run_refit_cycle(
            self._mx_receiver,
            candidates=candidates,
            context=self._mx_megatron_ctx,
            pull=_pull,
            device=self.device,
        ):
            weights.append((hf_name, hf_tensor))

        if not weights:
            print(f"[mx-megatron] cycle yielded 0 tensors; refit aborted")
            return False

        self._load_weights(weights)
        torch.cuda.current_stream().synchronize()
        self._maybe_process_fp8_kv_cache()

        if mx_config.tree_scale_out:
            self._mx_receiver.publish_self_as_source(
                version=int(version),
                model_name=self.model_config.model
                if hasattr(self.model_config, "model")
                else getattr(
                    self.model_runner.vllm_config.model_config, "model", "unknown",
                ),
            )

        gc.collect()
        torch.cuda.empty_cache()
        return True

    @wrap_with_nvtx_name(
        "vllm_internal_worker_extension/update_weights_from_collective"
    )
    def update_weights_from_collective(self) -> bool:
        """Update the model weights from collective communication."""
        assert self.state_dict_info is not None, (
            "state_dict_info is not prepared. "
            "Please call prepare_refit_info when initializing the worker."
        )

        load_model_weight_func = self._load_weights

        try:
            packed_broadcast_consumer(
                iterator=iter(self.state_dict_info.items()),
                group=self.model_update_group,
                src=0,
                post_unpack_func=load_model_weight_func,
            )

            # Process weights after loading for FP8 KV cache
            self._maybe_process_fp8_kv_cache()

        except Exception as e:
            print(
                f"Error in VllmInternalWorkerExtension.update_weights_from_collective: {e}"
            )
            return False

        return True

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
