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

    def init_per_pp_refit_comm_group(
        self,
        rank_prefix: int,
        pp_ips: list[str],
        pp_ports: list[int],
        pp_size: int,
        train_ranks_per_stage: int,
        sub_world_size: int,
    ) -> None:
        """Initialize per-PP-stage communication groups for nccl_xfer refit.

        Gen workers join ALL ``pp_size`` groups sequentially (they need all
        layers).  Groups are created in order stage 0, 1, … so that train
        ranks (which only join their own stage) unblock deterministically.
        """
        from nemo_rl.distributed.stateless_process_group import StatelessProcessGroup

        local_rank = torch.distributed.get_rank()
        gen_rank_in_group = train_ranks_per_stage + rank_prefix + local_rank

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
        from vllm.model_executor.model_loader.utils import (
            process_weights_after_loading,
        )

        # Get target device for processing
        target_device = next(self.model_runner.model.parameters()).device

        # Call process_weights_after_loading to handle KV scales
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
                    from vllm.model_executor.model_loader.utils import (
                        process_weights_after_loading,
                    )

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

    def prepare_nccl_xfer_refit_info(self, refit_info: dict) -> None:
        """Restore per-layer param metadata and build the HF→vLLM mapping.

        Done once ahead of refit; the cached mapping is reused by every
        ``nccl_xfer_refit`` call.
        """
        from nemo_rl.distributed.nccl_xfer_utils import (
            restore_refit_info_placements,
        )

        self.nccl_xfer_refit_info = (  # pyrefly: ignore[implicitly-defined-attribute]
            restore_refit_info_placements(refit_info)
        )
        self._hf_to_vllm = self._build_hf_to_vllm_mapping(  # pyrefly: ignore[implicitly-defined-attribute]
            self.nccl_xfer_refit_info
        )

    def _build_hf_to_vllm_mapping(self, refit_info):
        """Build mapping from HF param names to vLLM (param, merged_param_slice).

        vLLM merges certain HF params into combined tensors:
          - q_proj + k_proj + v_proj  → qkv_proj  (concat along dim 0)
          - gate_proj + up_proj       → gate_up_proj (concat along dim 0)
          - lm_head may be tied to embed_tokens

        For TP>1, vLLM shards merged params along dim 0. Each TP rank stores
        [q_shard, k_shard, v_shard] locally. We compute LOCAL slices by scaling
        global offsets proportionally: local_offset = global_offset * local_dim0 / global_dim0.

        Returns:
            dict: hf_name → (vllm_param_tensor, merged_param_slice or None)
                  If merged_param_slice is None, the HF param maps 1:1 to the
                  vLLM param.  Otherwise it is the LOCAL slice into the merged
                  vLLM param that this HF piece occupies.
        """
        vllm_params = dict(self.model_runner.model.named_parameters())
        mapping = {}

        # Collect all HF param names + global shapes from refit_info, plus the
        # grouped-expert tag (gate_proj/up_proj/down_proj) for MoE params.
        hf_shapes = {}
        hf_grouped = {}  # hf_name -> "gate_proj"|"up_proj"|"down_proj" (MoE only)
        for layer_name in refit_info["layer_names"]:
            for p in refit_info["per_layer_params"][layer_name]:
                hf_shapes[p["name"]] = tuple(p["global_shape"])
                if p.get("grouped_expert_proj"):
                    hf_grouped[p["name"]] = p["grouped_expert_proj"]

        # Per-prefix gate presence: a grouped-MoE layer is gated SwiGLU iff it
        # has a gate_proj group (else non-gated ReLU^2 = up_proj only).  Decides
        # whether up_proj maps to the whole w13 (non-gated) or its second half.
        has_gate = {
            name.rsplit(".gate_proj.weight", 1)[0]
            for name, proj in hf_grouped.items()
            if proj == "gate_proj"
        }

        # NemotronH (nemotron_h): HF param names use the ``backbone.*`` prefix
        # while vLLM uses ``model.*`` (e.g. ``backbone.layers.N.mixer.experts.
        # w13_weight`` -> ``model.layers.N.mixer.experts.w13_weight``).  Translate
        # at the point of vLLM-param lookup so the direct-match / merge rules below
        # resolve.  Mamba SSM params + embeddings take the misc path instead (see
        # is_misc_param), so vLLM's load_weights handles their special sharding and
        # the A_log -> A transform; they never reach this bulk mapping.  For
        # non-NemotronH models the name is unchanged, so this is a no-op.
        def _to_vllm_name(n):
            if n == "backbone.embeddings.weight":
                return "model.embed_tokens.weight"
            if n.startswith("backbone."):
                return "model." + n[len("backbone.") :]
            return n

        # Merge rules: (list of HF suffixes) → vLLM suffix, concat along dim 0
        MERGE_RULES = [
            (["q_proj.weight", "k_proj.weight", "v_proj.weight"], "qkv_proj.weight"),
            (["q_proj.bias", "k_proj.bias", "v_proj.bias"], "qkv_proj.bias"),
            (["gate_proj.weight", "up_proj.weight"], "gate_up_proj.weight"),
            # DeepSeek MLA down-projections fused on the vLLM side as
            # `fused_qkv_a_proj` (MergedColumnParallelLinear with
            # ``disable_tp=True`` → each TP rank holds the full concat).
            (
                ["q_a_proj.weight", "kv_a_proj_with_mqa.weight"],
                "fused_qkv_a_proj.weight",
            ),
            # NB: FP8 ``*.weight_scale_inv`` siblings are NOT merged here — every
            # ``_scale_inv`` takes the misc/load_weights path (is_misc_param), so
            # they never reach this bulk mapping and need no merge rule.
        ]

        for hf_name in hf_shapes:
            # 0) Grouped MoE expert params (gate_proj/up_proj/down_proj, each
            #    [E, ...]).  vLLM fuses them as w13_weight (gate||up on the
            #    intermediate axis) and w2_weight (down).  Dispatch on the
            #    grouped_expert_proj TAG, NOT the suffix, so dense gate_proj/
            #    up_proj (-> gate_up_proj, rule below) don't collide.  The
            #    received Shard(1)/Shard(2) shard is placed into the right
            #    w13/w2 region by get_dst_dtensor (+ its post_refit_hook for the
            #    gated w13 halves).
            grouped_proj = hf_grouped.get(hf_name)
            if grouped_proj is not None:
                expert_prefix = hf_name.rsplit(f".{grouped_proj}.weight", 1)[0]
                vllm_suffix = (
                    "w2_weight" if grouped_proj == "down_proj" else "w13_weight"
                )
                vllm_name = _to_vllm_name(f"{expert_prefix}.{vllm_suffix}")
                if vllm_name not in vllm_params:
                    mapping[hf_name] = (None, None)
                    print(
                        f"[WARN] _build_hf_to_vllm_mapping: no vLLM expert param "
                        f"for grouped '{hf_name}'",
                        flush=True,
                    )
                    continue
                vllm_param = vllm_params[vllm_name]
                if grouped_proj == "down_proj" or expert_prefix not in has_gate:
                    # down -> whole w2; non-gated ReLU^2 up -> whole w13.  The
                    # received shard IS the local vLLM param (gen TP shards the
                    # intermediate axis), so this is a direct copy, no sub-slice.
                    mapping[hf_name] = (vllm_param, None)
                else:
                    # gated SwiGLU: w13 = [E, 2P, hidden]; gate -> [:, :P, :],
                    # up -> [:, P:2P, :] along the intermediate axis (dim 1).
                    P = vllm_param.shape[1] // 2
                    sl = slice(0, P) if grouped_proj == "gate_proj" else slice(P, 2 * P)
                    mapping[hf_name] = (vllm_param, (slice(None), sl, slice(None)))
                continue

            # 1) Direct match (1:1 HF -> vLLM name)
            vllm_direct = _to_vllm_name(hf_name)
            if vllm_direct in vllm_params:
                mapping[hf_name] = (vllm_params[vllm_direct], None)
                continue

            # 2) Check merge rules
            matched = False
            for hf_suffixes, vllm_suffix in MERGE_RULES:
                for i, suffix in enumerate(hf_suffixes):
                    if hf_name.endswith(suffix):
                        prefix = hf_name[: -len(suffix)]
                        vllm_name = _to_vllm_name(prefix + vllm_suffix)
                        if vllm_name in vllm_params:
                            vllm_param = vllm_params[vllm_name]
                            local_dim0 = vllm_param.shape[0]

                            # Collect global dim0 sizes for all components
                            global_sizes = []
                            for s in hf_suffixes:
                                full_name = prefix + s
                                global_sizes.append(
                                    hf_shapes[full_name][0]
                                    if full_name in hf_shapes
                                    else 0
                                )
                            global_dim0 = sum(global_sizes)

                            # Compute LOCAL sizes per component.
                            # Linear interpolation (global_size * local_dim0 / global_dim0) fails
                            # when vLLM replicates KV heads (num_kv_heads < tp_size), because
                            # q/k/v proportions change between global and local.
                            #
                            # Use the gen TP size from refit_info, NOT
                            # torch.distributed.get_world_size(): the latter is the vLLM
                            # worker's default-group size, which equals TP only because
                            # gen PP/EP are pinned to 1 (check_nccl_xfer_refit_support).
                            # Reading gen_tp_size stays correct if gen PP is ever added.
                            tp_size = refit_info.get("gen_tp_size", 1)
                            naive_local_sizes = [gs // tp_size for gs in global_sizes]
                            if sum(naive_local_sizes) == local_dim0:
                                local_sizes = naive_local_sizes
                            elif local_dim0 == global_dim0:
                                # Fully replicated merge (e.g. DeepSeek MLA's
                                # fused_qkv_a_proj with disable_tp=True): the
                                # vLLM param holds the full concat on every TP
                                # rank, so local == global per component.
                                local_sizes = list(global_sizes)
                            else:
                                # KV head replication: q divides evenly, k/v are replicated
                                local_sizes = [global_sizes[0] // tp_size]
                                num_rest = len(global_sizes) - 1
                                rest = local_dim0 - local_sizes[0]
                                for _ in range(num_rest):
                                    local_sizes.append(rest // num_rest)
                            local_offset = sum(local_sizes[:i])
                            local_size = local_sizes[i]

                            # merged_param_slice is a multi-dim index TUPLE (not a
                            # bare slice): get_dst_dtensor indexes the vLLM param as
                            # ``vllm_param.data[merged_param_slice]``.  Dim-0 concat
                            # merges (qkv_proj, gate_up_proj, fused_qkv_a_proj) use a
                            # 1-tuple; grouped-expert w13 sub-slices use a dim-1 tuple.
                            mapping[hf_name] = (
                                vllm_param,
                                (slice(local_offset, local_offset + local_size),),
                            )
                            matched = True
                        break
                if matched:
                    break

            # 3) lm_head tied to embed_tokens
            if not matched and hf_name == "lm_head.weight":
                if "model.embed_tokens.weight" in vllm_params:
                    mapping[hf_name] = (
                        vllm_params["model.embed_tokens.weight"],
                        None,
                    )
                    matched = True

            if not matched:
                # (None, None) -> the discard branch in get_dst_dtensor: this
                # rank still joins the collective for symmetry but throws the
                # bytes away. Every supported model maps all of its bulk params,
                # so this WARN firing means a real mapping gap (a new/renamed
                # param vLLM expects under another name), not a benign case.
                mapping[hf_name] = (None, None)
                print(
                    f"[WARN] _build_hf_to_vllm_mapping: no vLLM param for '{hf_name}'",
                    flush=True,
                )

        return mapping

    def get_dst_dtensor(self, param_name, param_info):
        """Get destination tensor info for xferdtensor.

        Takes an HF parameter name and returns a tuple suitable for calling
        the canonical 7-arg xferdtensor. Handles three cases:

        1. **Direct param**: HF name maps 1:1 to a vLLM parameter.
           Returns DTensorRef wrapping the vLLM param with global shape.
        2. **Merged param**: HF name is part of a merged vLLM tensor
           (qkv_proj, gate_up_proj, fused_qkv_a_proj). Returns DTensorRef
           wrapping a buffer shaped like this component's slice of the merged
           param, plus a post_refit_hook that copies it in.
        3. **Unmapped param**: no vLLM param owns this HF param. Returns a
           DTensorRef wrapping a throwaway buffer purely so this rank still
           joins the collective (train broadcasts every bulk param; a missing
           participant would deadlock the broadcast). The bytes are discarded.

        Must be called after ``_hf_to_vllm`` is populated (inside ``nccl_xfer_refit``).

        Returns:
            ``(dst_tensor, dst_placements, post_refit_hook)`` where:
            - ``dst_tensor``: DTensorRef to pass to xferdtensor
            - ``dst_placements``: placement list to use (may differ from param_info)
            - ``post_refit_hook``: callable to run after xferdtensor
              (copies the TP-local slice from the temp buffer into the live
              vLLM merged param), or None for direct/unmapped params
        """
        from nemo_rl.distributed.nccl_xfer_utils import _STR_TO_DTYPE
        from nemo_rl.distributed.xferdtensor import DTensorRef

        vllm_param, merged_param_slice = self._hf_to_vllm.get(param_name, (None, None))
        global_shape = param_info["global_shape"]
        dst_placements = param_info["dst_placements"]

        if vllm_param is None:
            # No vLLM owner for this HF param. This is NOT stale-weight
            # corruption: train broadcasts EVERY bulk param, so each gen rank
            # must call the matching collective even with nowhere to store the
            # bytes, or the broadcast deadlocks. We receive into a throwaway
            # buffer and run no post_refit_hook, discarding the result.
            #
            # The global_shape buffer is fine: golden stages the transfer in its
            # own internal buffers, so this is only the (discarded) final-copy
            # target. In practice every supported model maps all of its bulk
            # params, so reaching here signals a mapping regression (see the
            # WARN in _build_hf_to_vllm_mapping), not a normal path.
            dtype = _STR_TO_DTYPE.get(param_info.get("dtype", ""), torch.bfloat16)
            buf = torch.empty(global_shape, device=self.device, dtype=dtype)
            return DTensorRef(buf, global_shape), dst_placements, None

        if merged_param_slice is not None:
            # Honor the dst placement decided at setup (refit_info) — identical
            # for golden and the real op, so train and gen always agree on the
            # redistribution (the real point-to-point op deadlocks on any
            # disagreement).  Receive this component's slice of the merged vLLM
            # param directly into a like-shaped buffer, then copy it in:
            #   - Shard dst (clean column-parallel): the transfer delivers the
            #     1/gen_tp shard, which is exactly this region.
            #   - all-Replicate dst (disable_tp): the transfer delivers the full
            #     tensor, which IS this region (the gen param is replicated, so
            #     the slice spans the whole tensor).
            # KV-head replication — where the region would be smaller than the
            # received tensor — is routed to the misc/load_weights path instead
            # (see qkv_to_misc), so it never reaches here.
            region = vllm_param.data[merged_param_slice]
            buf = torch.empty_like(region)

            def post_refit_hook(_buf=buf, _region=region):
                _region.copy_(_buf)

            return DTensorRef(buf, global_shape), dst_placements, post_refit_hook

        # Direct 1:1 mapping — xferdtensor handles TP sharding.
        # Use .data to avoid "leaf Variable that requires grad" error from
        # in-place copy_ inside xferdtensor.
        return DTensorRef(vllm_param.data, global_shape), dst_placements, None

    def nccl_xfer_refit(self) -> bool:
        """Receive weights from training workers via xferdtensor.

        Uses ``get_dst_dtensor`` to prepare each parameter, then calls the
        canonical 7-arg ``xferdtensor``. For merged params (qkv_proj,
        gate_up_proj), a post_refit_hook copies the TP-local slice. The
        HF→vLLM mapping is built once in ``prepare_nccl_xfer_refit_info``.
        """
        from nemo_rl.distributed.xferdtensor import xferdtensor

        use_per_stage = hasattr(self, "pp_comm_groups") and self.pp_comm_groups

        for layer_name in self.nccl_xfer_refit_info["layer_names"]:
            for param_info in self.nccl_xfer_refit_info["per_layer_params"][layer_name]:
                if use_per_stage:
                    group = self.pp_comm_groups[param_info["pp_stage"]]
                else:
                    group = self.model_update_group

                dst_tensor, dst_placements, post_refit_hook = self.get_dst_dtensor(
                    param_info["name"], param_info
                )
                xferdtensor(
                    None,
                    param_info["src_mesh_info"],
                    param_info["src_placements"],
                    dst_tensor,
                    param_info["dst_mesh_info"],
                    dst_placements,
                    group,
                )
                if post_refit_hook:
                    post_refit_hook()

        self._receive_and_load_misc_params()

        torch.cuda.synchronize()

        # process_weights_after_loading is intentionally disabled for the
        # FP8-weight refit path: bulk FP8 weights arrive kernel-ready via
        # xferdtensor and misc params go through vLLM's own load_weights, so no
        # extra post-load finalization is needed (verified on 30B FP8 MoE:
        # byte-identical generation/rewards with and without this call).  It is
        # kept (commented) rather than deleted because enabling FP8 *KV cache*
        # will need it to finalize the per-layer k/v scales after refit;
        # uncomment it (guarded on kv_cache_dtype) as part of that work.
        # from vllm.model_executor.model_loader.utils import (
        #     process_weights_after_loading,
        # )
        # process_weights_after_loading(
        #     self.model_runner.model, self.model_config, self.device
        # )
        return True

    def _receive_and_load_misc_params(self) -> None:
        """Receive misc params via packed_broadcast and load via vLLM."""
        from nemo_rl.distributed.nccl_xfer_utils import _STR_TO_DTYPE

        misc_meta = self.nccl_xfer_refit_info.get("misc_meta", {})
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
