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


def fix_gemma3_vision_weight_name(key: str) -> str:
    """Re-insert the `vision_model` segment into Gemma3 vision-tower weights.

    When performing refit, the vision-tower weight paths are flattened. This unflattens them.
    """
    return re.sub(
        r"vision_tower\.(?!vision_model\.)", "vision_tower.vision_model.", key
    )


def _fused_param_merge_slice(
    hf_shapes: dict,
    prefix: str,
    hf_suffixes: list,
    shard_index: int,
    local_dim0: int,
    tp_size: int,
) -> tuple:
    """Dim-0 sub-slice into a TP-sharded *fused* vLLM param for one component.

    vLLM fuses several HF projections into one param along dim 0 (q/k/v ->
    qkv_proj, gate/up -> gate_up_proj, q_a/kv_a -> fused_qkv_a_proj).  Given the
    component list (``hf_suffixes`` at the same layer ``prefix``), the local
    param's dim-0 (``local_dim0``), and the gen TP size, return where component
    ``shard_index``'s received local shard lands.

    Returns a 1-tuple ``(slice(off, off + size),)`` — an index tuple so the
    caller can do ``vllm_param.data[merged_slice]`` uniformly (grouped-expert
    w13 sub-slices use a dim-1 tuple elsewhere).
    """
    # Global (unsharded) dim-0 size of each fused component.
    global_sizes = [
        hf_shapes[prefix + s][0] if (prefix + s) in hf_shapes else 0
        for s in hf_suffixes
    ]
    # LOCAL sizes = global // gen TP.  The KV-head-replication case
    # (tp > num_kv_heads), where that even split breaks, is routed to misc via
    # qkv_to_misc, so the `else` fallback below rarely fires.
    naive = [gs // tp_size for gs in global_sizes]
    if sum(naive) == local_dim0:
        local_sizes = naive
    elif local_dim0 == sum(global_sizes):
        # Fully replicated merge (e.g. DeepSeek MLA fused_qkv_a_proj with
        # disable_tp=True): the param holds the full concat on every TP rank.
        local_sizes = list(global_sizes)
    else:
        # KV-head replication: q divides evenly, k/v are replicated.
        local_sizes = [global_sizes[0] // tp_size]
        num_rest = len(global_sizes) - 1
        rest = local_dim0 - local_sizes[0]
        local_sizes += [rest // num_rest] * num_rest
    offset = sum(local_sizes[:shard_index])
    return (slice(offset, offset + local_sizes[shard_index]),)


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
        self._hf_to_gen_param = self._build_hf_to_gen_backend_mapping(  # pyrefly: ignore[implicitly-defined-attribute]
            self.nccl_xfer_refit_info
        )

    def _build_hf_to_gen_backend_mapping(self, refit_info):
        """Map each HF param name to the generation backend's param and slice.

        Returns a dict ``hf_name -> (param: torch.Tensor, merged_param_slice or
        None)``. merged_param_slice is used in the post_refit_hook to copy the
        received data from the temp buffer into the live merged param. (This
        function could be upstreamed to the generation backend repo later.)

        vLLM merges certain HF params into combined tensors:
          - q_proj + k_proj + v_proj  → qkv_proj      (concat along dim 0)
          - gate_proj + up_proj       → gate_up_proj  (concat along dim 0)
          - lm_head may be tied to embed_tokens

        For TP>1, vLLM shards merged params along dim 0, so each TP rank stores
        [q_shard, k_shard, v_shard] locally; _fused_param_merge_slice computes
        the LOCAL sub-slice each HF component occupies.

        Returns:
            dict: hf_name → (vllm_param_tensor, merged_param_slice or None).
                  If merged_param_slice is None the HF param maps 1:1 to the
                  vLLM param; otherwise it is the LOCAL slice into the merged
                  vLLM param that this HF piece occupies.
        """
        vllm_params = dict(self.model_runner.model.named_parameters())
        # Module lookup: to detect the selected backend off the FusedMoE layer
        vllm_modules = dict(self.model_runner.model.named_modules())
        mapping = {}

        # Collect all HF param names + global shapes from refit_info, plus the
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

        # NemotronH (nemotron_h): HF param names use the ``backbone.*`` prefix
        # while vLLM uses ``model.*`` (e.g. ``backbone.layers.N.mixer.experts.
        # w13_weight`` -> ``model.layers.N.mixer.experts.w13_weight``). For
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
        ]

        for hf_name in hf_shapes:
            # 1) Grouped MoE expert params (gate_proj/up_proj/down_proj, each
            #    [E, ...]). vLLM fuses them as w13_weight (gate||up on the
            #    intermediate axis) and w2_weight (down). The received
            #    Shard(1)/Shard(2) shard is placed into the right w13/w2 region by
            #    get_dst_dtensor (+ its post_refit_hook for the gated w13 halves).
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
                    moe_mod = vllm_modules.get(vllm_name.rsplit(".", 1)[0])
                    backend = getattr(
                        getattr(moe_mod, "quant_method", None),
                        "unquantized_backend",
                        None,
                    )
                    backend = getattr(backend, "name", "")
                    if backend == "FLASHINFER_TRTLLM":
                        # TRTLLM also block-reorders w13 (beyond the swap);
                        # Requires more changes to the refit logic to support.
                        raise ValueError(
                            f"nccl_xfer refit: gen MoE backend {backend!r} reorders "
                            "w13 in a way the refit does not reproduce; run gen with "
                            "the TRITON or FlashInfer CUTLASS MoE backend."
                        )
                    if backend == "FLASHINFER_CUTLASS":  # live w13 is [up; gate]
                        sl = (
                            slice(P, 2 * P)
                            if grouped_proj == "gate_proj"
                            else slice(0, P)
                        )
                    else:  # standard [gate; up]
                        sl = (
                            slice(0, P)
                            if grouped_proj == "gate_proj"
                            else slice(P, 2 * P)
                        )
                    mapping[hf_name] = (vllm_param, (slice(None), sl, slice(None)))
                continue

            # 2) Direct match (1:1 HF -> vLLM name)
            vllm_direct = _to_vllm_name(hf_name)
            if vllm_direct in vllm_params:
                mapping[hf_name] = (vllm_params[vllm_direct], None)
                continue

            # 3) Check merge rules
            matched = False
            for hf_suffixes, vllm_suffix in MERGE_RULES:
                for i, suffix in enumerate(hf_suffixes):
                    if hf_name.endswith(suffix):
                        prefix = hf_name[: -len(suffix)]
                        # e.g.) vllm_name = model.layers.3.mlp.qkv_proj.weight
                        vllm_name = _to_vllm_name(prefix + vllm_suffix)
                        if vllm_name in vllm_params:
                            vllm_param = vllm_params[vllm_name]
                            # Place this component's received shard into its dim-0
                            # region of the fused vLLM param (qkv_proj/gate_up_proj/
                            # fused_qkv_a_proj).  See _fused_param_merge_slice.
                            mapping[hf_name] = (
                                vllm_param,
                                _fused_param_merge_slice(
                                    hf_shapes,
                                    prefix,
                                    hf_suffixes,
                                    i,
                                    vllm_param.shape[0],
                                    refit_info.get("gen_tp_size", 1),
                                ),
                            )
                            matched = True
                        break
                if matched:
                    break

            # 4) lm_head tied to embed_tokens
            if not matched and hf_name == "lm_head.weight":
                if "model.embed_tokens.weight" in vllm_params:
                    mapping[hf_name] = (
                        vllm_params["model.embed_tokens.weight"],
                        None,
                    )
                    matched = True

            if not matched:
                raise ValueError(
                    f"_build_hf_to_gen_backend_mapping: no vLLM param for {hf_name!r} "
                    f"(no direct / merge-rule / tied-lm_head match)."
                )

        return mapping

    def get_dst_dtensor(self, param_name, param_info):
        """Get destination tensor info for xferdtensor.

        Takes an HF parameter name and returns a tuple suitable for calling
        the canonical 7-arg xferdtensor. Handles two cases:

        1. **Direct param**: HF name maps 1:1 to a vLLM parameter.
           Returns DTensorRef wrapping the vLLM param with global shape.
        2. **Merged param**: HF name is part of a merged vLLM tensor
           (qkv_proj, gate_up_proj, fused_qkv_a_proj). Returns DTensorRef
           wrapping a buffer shaped like this component's slice of the merged
           param, plus a post_refit_hook that copies it in.

        ``_build_hf_to_gen_backend_mapping`` guarantees every bulk param falls
        into one of these (it raises if a param has no vLLM target), so a param
        with no mapping entry here is a coverage regression and raises
        ValueError rather than silently discarding its weights.

        Must be called after ``_hf_to_gen_param`` is populated (inside ``nccl_xfer_refit``).

        Returns:
            ``(dst_tensor, dst_placements, post_refit_hook)`` where:
            - ``dst_tensor``: DTensorRef to pass to xferdtensor
            - ``dst_placements``: placement list to use (may differ from param_info)
            - ``post_refit_hook``: callable to run after xferdtensor
              (copies the TP-local slice from the temp buffer into the live
              vLLM merged param), or None for direct params
        """
        from nemo_rl.distributed.xferdtensor import DTensorRef

        vllm_param, merged_param_slice = self._hf_to_gen_param.get(
            param_name, (None, None)
        )
        global_shape = param_info["global_shape"]
        dst_placements = param_info["dst_placements"]

        if vllm_param is None:
            raise ValueError(
                f"get_dst_dtensor: {param_name!r} has no entry in the HF->gen "
                f"mapping; _build_hf_to_gen_backend_mapping did not enumerate it. "
                f"This would silently discard the param's weights."
            )

        if merged_param_slice is not None:
            # If the param is a merged param, we need to copy the received data
            # from the temp buffer into the live generation backend merged param.
            region = vllm_param.data[merged_param_slice]
            buf = torch.empty_like(region)

            def post_refit_hook(_buf=buf, _region=region):
                _region.copy_(_buf)

            return DTensorRef(buf, global_shape), dst_placements, post_refit_hook

        return DTensorRef(vllm_param.data, global_shape), dst_placements, None

    def nccl_xfer_refit(self) -> bool:
        """Receive weights from training workers via xferdtensor.

        Uses ``get_dst_dtensor`` to prepare each parameter, then calls the
        canonical 7-arg ``xferdtensor``. For merged params (qkv_proj,
        gate_up_proj), a post_refit_hook copies the TP-local slice. The
        HF→vLLM mapping is built once in ``prepare_nccl_xfer_refit_info``.
        """
        import os
        from collections import OrderedDict

        from nemo_rl.distributed.xferdtensor import xferdtensor

        def _recv_one_param(param_info, group):
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

        use_per_stage = hasattr(self, "pp_comm_groups") and self.pp_comm_groups
        num_streams = (
            int(os.environ.get("NRL_REFIT_NUM_STREAMS", "2")) if use_per_stage else 1
        )

        # Group params into ordered batches to concurrently recieve params from
        # different pp-stages
        stage_params = OrderedDict()
        for layer_name in self.nccl_xfer_refit_info["layer_names"]:
            for p in self.nccl_xfer_refit_info["per_layer_params"][layer_name]:
                key = p["pp_stage"] if use_per_stage else 0
                stage_params.setdefault(key, []).append(p)

        streams = [torch.cuda.Stream() for _ in range(num_streams)]
        events = {}
        for idx, (stage, params) in enumerate(stage_params.items()):
            # synchronize the last run in the same stream
            if (idx - num_streams) in events:
                events[idx - num_streams].synchronize()
            with torch.cuda.stream(streams[idx % num_streams]):
                group = (
                    self.pp_comm_groups[stage]
                    if use_per_stage
                    else self.model_update_group
                )
                for p in params:
                    _recv_one_param(p, group)
                ev = torch.cuda.Event()
                ev.record()
                events[idx] = ev

        self._receive_and_load_misc_params()

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # Finalize FP8 KV-cache per-layer k/v scales after the misc broadcast.
        self._maybe_process_fp8_kv_cache()
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
