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
import io
import re
import time
import traceback
from dataclasses import dataclass
from typing import Any, cast

import torch
import zmq

from nemo_rl.models.policy.utils import (
    IPCProtocol,
    calculate_aligned_size,
    rebuild_cuda_tensor_from_ipc,
)
from nemo_rl.utils import weight_transfer_sparse_codec as sparse_codec
from nemo_rl.utils.nsys import wrap_with_nvtx_name
from nemo_rl.utils.packed_tensor import packed_broadcast_consumer

_EXPERT_WEIGHT_RE = re.compile(
    r"^(?P<prefix>.*\.experts)\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.weight$"
)

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


@dataclass(frozen=True)
class _SparseDeltaTargetPlan:
    target: torch.Tensor | None
    source_shape: tuple[int, ...] = ()
    source_strides: tuple[int, ...] = ()
    target_strides: tuple[int, ...] = ()
    target_offset: int = 0
    shard_dim: int | None = None
    shard_start: int = 0
    shard_size: int = 0
    segment_shards: tuple[tuple[int, int, int], ...] = ()
    log_delta_transform: bool = False
    identity: bool = False


class VllmInternalWorkerExtension:
    state_dict_info: dict[str, Any] | None = None
    _direct_sparse_delta_targets: dict[str, torch.Tensor] | None = None
    _direct_sparse_delta_modules: dict[str, torch.nn.Module] | None = None
    _direct_sparse_delta_plan_cache: dict[str, _SparseDeltaTargetPlan | None] | None = (
        None
    )

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
        self._direct_sparse_delta_targets = None
        self._direct_sparse_delta_modules = None
        self._direct_sparse_delta_plan_cache = None

    def _process_weights_after_loading(
        self,
        model_config: Any,
        target_device: torch.device,
    ) -> None:
        from vllm.config import set_current_vllm_config
        from vllm.model_executor.model_loader.utils import (
            process_weights_after_loading,
        )

        with set_current_vllm_config(self.model_runner.vllm_config):
            process_weights_after_loading(
                self.model_runner.model,
                model_config,
                target_device,
            )

    def _maybe_process_fp8_kv_cache(self) -> None:
        """Process weights after loading for FP8 KV cache static scales."""
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
        self._process_weights_after_loading(
            self.model_runner.model_config,
            next(self.model_runner.model.parameters()).device,
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
        mtp_start_layer_idx = cast(int, predictor.mtp_start_layer_idx)
        num_mtp_layers = cast(int, predictor.num_mtp_layers)
        mtp_layer_indices = set(
            range(
                mtp_start_layer_idx,
                mtp_start_layer_idx + num_mtp_layers,
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

    def _apply_sparse_weight_deltas(
        self,
        payload_tensors: tuple[torch.Tensor, torch.Tensor],
        metadata: list[dict[str, Any]],
    ) -> None:
        """Apply sparse deltas directly after validating every target plan."""
        if self._direct_sparse_delta_uses_loader_transform():
            raise RuntimeError(
                "Direct sparse delta refit does not support transformed or FP8 weights."
            )

        targets = self._direct_sparse_delta_target_map()
        raw_locations, raw_values = payload_tensors
        plans = []
        for item in metadata:
            plan = self._cached_direct_sparse_delta_target_plan(item, targets)
            if plan is None:
                raise RuntimeError(
                    f"No direct sparse delta target plan for {item['name']!r}."
                )
            plans.append((item, plan))

        with torch.no_grad():
            for item, plan in plans:
                target = plan.target
                if target is None:
                    continue

                value_start = int(item["value_start"])
                value_end = int(item["value_end"])
                values = raw_values[value_start:value_end].to(
                    device=target.device,
                    dtype=target.dtype,
                    non_blocking=True,
                )
                if plan.identity and item["index_encoding"] == "range":
                    range_start = int(item["range_start"])
                    range_count = value_end - value_start
                    target.data.view(-1).narrow(0, range_start, range_count).add_(
                        values
                    )
                    continue

                locations = sparse_codec.sparse_locations_for_item(
                    item,
                    raw_locations,
                    device=target.device,
                )
                locations, values = self._local_sparse_delta_update_inputs(
                    locations,
                    values,
                    plan,
                )
                if locations.numel():
                    target_flat = target.data.view(-1)
                    if plan.log_delta_transform:
                        current = target_flat.index_select(0, locations)
                        updated = current * values.float().exp().to(dtype=current.dtype)
                        target_flat.index_copy_(0, locations, updated)
                    else:
                        target_flat.index_add_(0, locations, values)

    def _direct_sparse_delta_target_map(self) -> dict[str, torch.Tensor]:
        if self._direct_sparse_delta_targets is None:
            self._direct_sparse_delta_targets = dict(
                self.model_runner.model.named_parameters()
            )
            self._direct_sparse_delta_targets.update(
                self.model_runner.model.named_buffers()
            )
        return self._direct_sparse_delta_targets

    def _direct_sparse_delta_named_module(
        self,
        module_name: str,
    ) -> torch.nn.Module | None:
        if self._direct_sparse_delta_modules is None:
            self._direct_sparse_delta_modules = dict(
                self.model_runner.model.named_modules()
            )
        return self._direct_sparse_delta_modules.get(module_name)

    def _direct_sparse_delta_uses_loader_transform(self) -> bool:
        architectures = self.model_runner.vllm_config.model_config.architectures
        if {"GptOssForCausalLM", "Gemma3ForConditionalGeneration"} & set(architectures):
            return True

        from nemo_rl.models.generation.vllm.quantization import fp8

        return fp8.is_fp8_model(self.model_runner.vllm_config)

    def _cached_direct_sparse_delta_target_plan(
        self,
        item: dict[str, Any],
        targets: dict[str, torch.Tensor],
    ) -> _SparseDeltaTargetPlan | None:
        name = str(item["name"])
        if self._direct_sparse_delta_plan_cache is None:
            self._direct_sparse_delta_plan_cache = {}
        if name not in self._direct_sparse_delta_plan_cache:
            self._direct_sparse_delta_plan_cache[name] = (
                self._direct_sparse_delta_target_plan(item, targets)
            )
        return self._direct_sparse_delta_plan_cache[name]

    def _direct_sparse_delta_target_plan(
        self,
        item: dict[str, Any],
        targets: dict[str, torch.Tensor],
    ) -> _SparseDeltaTargetPlan | None:
        name = str(item["name"])
        if name.startswith("mtp."):
            return _SparseDeltaTargetPlan(target=None)
        target_name = self._map_direct_sparse_delta_name(name)
        if target_name is None:
            return None
        if ".mixer." in target_name:
            mamba_plan = self._direct_sparse_delta_mamba2_plan(
                item, target_name, targets
            )
            if mamba_plan is not None:
                return mamba_plan
            if ".mixer.conv1d." in target_name or ".mixer.in_proj." in target_name:
                return None
        if any(f".{candidate}_proj." in target_name for candidate in ("q", "k", "v")):
            return self._direct_sparse_delta_qkv_plan(item, target_name, targets)
        if _EXPERT_WEIGHT_RE.match(target_name):
            return self._direct_sparse_delta_expert_plan(item, target_name, targets)

        target = targets.get(target_name)
        if target is None:
            return None
        source_shape = tuple(item["shape"])
        target_shape = tuple(target.shape)
        if target_shape == source_shape:
            return self._make_sparse_delta_target_plan(target, source_shape)
        return self._direct_sparse_delta_shard_plan(item, target)

    def _direct_sparse_delta_qkv_plan(
        self,
        item: dict[str, Any],
        target_name: str,
        targets: dict[str, torch.Tensor],
    ) -> _SparseDeltaTargetPlan | None:
        shard_id = next(x for x in "qkv" if f".{x}_proj." in target_name)
        packed_name = target_name.replace(f".{shard_id}_proj.", ".qkv_proj.", 1)
        target = targets.get(packed_name)
        if target is None:
            return None
        output_dim = int(cast(Any, target).output_dim) % target.ndim
        module = cast(
            Any,
            getattr(getattr(target, "weight_loader", None), "__self__", None)
            or self._direct_sparse_delta_named_module(packed_name.rsplit(".", 1)[0]),
        )
        shard_offset = int(module._get_shard_offset_mapping(shard_id))
        shard_size = int(module._get_shard_size_mapping(shard_id))
        shard_rank = int(module.tp_rank)
        if shard_id != "q":
            shard_rank //= int(module.num_kv_head_replicas)

        source_shape = tuple(item["shape"])
        shard_start = shard_rank * shard_size
        if source_shape[output_dim] < shard_start:
            return _SparseDeltaTargetPlan(target=None)
        return self._make_sparse_delta_target_plan(
            target,
            source_shape=source_shape,
            shard_dim=output_dim,
            shard_start=shard_start,
            shard_size=min(shard_size, source_shape[output_dim] - shard_start),
            target_offset=shard_offset * target.stride(output_dim),
        )

    def _direct_sparse_delta_mamba2_plan(
        self,
        item: dict[str, Any],
        target_name: str,
        targets: dict[str, torch.Tensor],
    ) -> _SparseDeltaTargetPlan | None:
        target = targets.get(target_name)
        if target is None:
            return None

        if target_name.endswith(".A"):
            source_shape = tuple(item["shape"])
            if tuple(target.shape) == source_shape:
                return self._make_sparse_delta_target_plan(
                    target,
                    source_shape=source_shape,
                    log_delta_transform=True,
                )
            return self._direct_sparse_delta_shard_plan(
                item,
                target,
                log_delta_transform=True,
            )

        if not (".mixer.conv1d." in target_name or ".mixer.in_proj." in target_name):
            return None

        mixer_name = target_name.split(".mixer.", 1)[0] + ".mixer"
        attrs = cast(Any, self._direct_sparse_delta_named_module(mixer_name))
        tp_size = int(attrs.tp_size)
        if tp_size <= 1:
            return None
        intermediate_size = int(attrs.intermediate_size)
        groups_ssm_state_size = int(attrs.groups_ssm_state_size)
        num_heads = int(attrs.num_heads)
        source_shape = tuple(item["shape"])
        fixed_size = intermediate_size
        if ".mixer.in_proj." in target_name:
            fixed_size += intermediate_size + num_heads
        group_size, remainder = divmod(source_shape[0] - fixed_size, 2)
        extra_group_size = groups_ssm_state_size - group_size
        if remainder or group_size <= 0 or extra_group_size < 0:
            return None
        tp_rank = int(
            getattr(target, "tp_rank", self._direct_sparse_delta_tp_rank(tp_size))
        )
        intermediate = (intermediate_size, 0, False)
        group = (groups_ssm_state_size, extra_group_size, extra_group_size > 0)
        segment_specs = (
            (intermediate, group, group)
            if ".mixer.conv1d." in target_name
            else (intermediate, intermediate, group, group, (num_heads, 0, False))
        )

        target_shape = tuple(target.shape)
        source_to_target_dims = tuple(range(len(source_shape)))
        if len(target_shape) == len(source_shape) + 1 and target_shape[1] == 1:
            source_to_target_dims = (0, *range(2, len(target_shape)))
        elif len(target_shape) != len(source_shape):
            return None
        segment_shards: list[tuple[int, int, int]] = []
        target_start = 0
        source_start = 0
        for full_dim, extra, duplicate_groups in segment_specs:
            shard_size = full_dim // tp_size
            rank = 0 if duplicate_groups else tp_rank
            source_dim = full_dim - extra
            source_local_start = source_start + rank * shard_size
            take = min(shard_size, source_dim - rank * shard_size)
            if take > 0:
                segment_shards.append((source_local_start, target_start, take))
            target_start += shard_size
            source_start += source_dim
        if source_shape[0] != source_start or target_shape[0] != target_start:
            return None

        return self._make_sparse_delta_target_plan(
            target,
            source_shape=source_shape,
            source_to_target_dims=source_to_target_dims,
            shard_dim=0,
            segment_shards=tuple(segment_shards),
        )

    def _direct_sparse_delta_expert_plan(
        self,
        item: dict[str, Any],
        target_name: str,
        targets: dict[str, torch.Tensor],
    ) -> _SparseDeltaTargetPlan | None:
        match = cast(re.Match[str], _EXPERT_WEIGHT_RE.match(target_name))

        prefix = match.group("prefix")
        global_expert_id = int(match.group("expert"))
        proj = match.group("proj")
        packed_weight, shard_id = {
            "gate_proj": ("w13_weight", "w1"),
            "up_proj": ("w13_weight", "w3"),
            "down_proj": ("w2_weight", "w2"),
        }[proj]
        packed_name = f"{prefix}.{packed_weight}"

        target = targets.get(packed_name)
        if target is None:
            return None
        module_attrs = cast(
            Any,
            getattr(getattr(target, "weight_loader", None), "__self__", None)
            or self._direct_sparse_delta_named_module(packed_name.rsplit(".", 1)[0]),
        )
        if shard_id == "w3" and not module_attrs.moe_config.is_act_and_mul:
            shard_id = "w1"
        local_expert_id = int(
            module_attrs._map_global_expert_id_to_local_expert_id(global_expert_id)
        )
        if local_expert_id < 0:
            return _SparseDeltaTargetPlan(target=None)

        source_shape = tuple(item["shape"])
        target_shape = tuple(target.shape)
        shard_dim = 1 if shard_id == "w2" else 0
        if local_expert_id >= target_shape[0]:
            return None
        target_shard_dim = shard_dim + 1

        shard_size = target_shape[target_shard_dim]
        if shard_id in ("w1", "w3") and module_attrs.moe_config.is_act_and_mul:
            shard_size //= 2
        target_shard_offset = shard_size if shard_id == "w3" else 0
        if target_shape[target_shard_dim] < target_shard_offset + shard_size:
            return None
        tp_rank = int(module_attrs.tp_rank)
        shard_start = tp_rank * shard_size
        if source_shape[shard_dim] < shard_start:
            return _SparseDeltaTargetPlan(target=None)

        return self._make_sparse_delta_target_plan(
            target,
            source_shape=source_shape,
            source_to_target_dims=tuple(dim + 1 for dim in range(len(source_shape))),
            target_offset=(
                local_expert_id * target.stride(0)
                + target_shard_offset * target.stride(target_shard_dim)
            ),
            shard_dim=shard_dim,
            shard_start=shard_start,
            shard_size=min(shard_size, source_shape[shard_dim] - shard_start),
        )

    def _make_sparse_delta_target_plan(
        self,
        target: torch.Tensor,
        source_shape: tuple[int, ...],
        *,
        source_to_target_dims: tuple[int, ...] | None = None,
        target_offset: int = 0,
        shard_dim: int | None = None,
        shard_start: int = 0,
        shard_size: int = 0,
        segment_shards: tuple[tuple[int, int, int], ...] = (),
        log_delta_transform: bool = False,
    ) -> _SparseDeltaTargetPlan | None:
        if source_to_target_dims is None:
            source_to_target_dims = tuple(range(len(source_shape)))
        target_shape = tuple(target.shape)
        ignored_dim = (
            shard_dim if shard_dim is not None else 0 if segment_shards else -1
        )
        if len(source_to_target_dims) != len(source_shape) or any(
            target_dim >= len(target_shape)
            or (
                source_dim != ignored_dim
                and source_shape[source_dim] != target_shape[target_dim]
            )
            for source_dim, target_dim in enumerate(source_to_target_dims)
        ):
            return None
        identity = (
            shard_dim is None
            and target_offset == 0
            and not segment_shards
            and not log_delta_transform
            and source_to_target_dims == tuple(range(len(source_shape)))
            and source_shape == target_shape
        )
        return _SparseDeltaTargetPlan(
            target=target,
            source_shape=source_shape,
            source_strides=torch.empty(source_shape, device="meta").stride(),
            target_strides=tuple(
                target.stride(target_dim) for target_dim in source_to_target_dims
            ),
            target_offset=target_offset,
            shard_dim=shard_dim,
            shard_start=shard_start,
            shard_size=shard_size,
            segment_shards=segment_shards,
            log_delta_transform=log_delta_transform,
            identity=identity,
        )

    def _direct_sparse_delta_shard_plan(
        self,
        item: dict[str, Any],
        target: torch.Tensor,
        *,
        log_delta_transform: bool = False,
    ) -> _SparseDeltaTargetPlan | None:
        source_shape = tuple(item["shape"])
        target_shape = tuple(target.shape)
        if len(source_shape) != len(target_shape):
            return None

        candidate_dims = list(
            dict.fromkeys(
                dim % len(source_shape)
                for attr in ("output_dim", "input_dim")
                if isinstance(dim := getattr(target, attr, None), int)
            )
        )
        if not candidate_dims:
            candidate_dims = [
                dim
                for dim, (source_dim, target_dim) in enumerate(
                    zip(source_shape, target_shape, strict=True)
                )
                if source_dim != target_dim
            ]
            if len(candidate_dims) != 1:
                return None

        for shard_dim in candidate_dims:
            shard_size = target_shape[shard_dim]
            tp_size = int(getattr(target, "tp_size", 1))
            if tp_size <= 1:
                if shard_size <= 0 or source_shape[shard_dim] % shard_size:
                    continue
                tp_size = source_shape[shard_dim] // shard_size
                if tp_size <= 1:
                    continue
            if source_shape[shard_dim] > shard_size * tp_size:
                continue
            tp_rank = int(
                getattr(target, "tp_rank", self._direct_sparse_delta_tp_rank(tp_size))
            )
            plan = self._make_sparse_delta_target_plan(
                target=target,
                source_shape=source_shape,
                shard_dim=shard_dim,
                shard_start=tp_rank * shard_size,
                shard_size=shard_size,
                log_delta_transform=log_delta_transform,
            )
            if plan is not None:
                return plan
        return None

    def _local_sparse_delta_update_inputs(
        self,
        locations: torch.Tensor,
        values: torch.Tensor,
        plan: _SparseDeltaTargetPlan,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if plan.identity:
            return locations, values

        source_shape = plan.source_shape
        source_strides = plan.source_strides
        target_strides = plan.target_strides
        shard_dim = plan.shard_dim

        if source_strides == target_strides:
            if shard_dim is None:
                return locations + plan.target_offset, values
            if shard_dim == 0:
                shard_stride = source_strides[0]
                shard_coords = torch.div(locations, shard_stride, rounding_mode="floor")
                if plan.segment_shards:
                    mapped_locations = locations + plan.target_offset
                    keep = torch.zeros_like(locations, dtype=torch.bool)
                    for source_start, target_start, take in plan.segment_shards:
                        segment = (shard_coords >= source_start) & (
                            shard_coords < source_start + take
                        )
                        mapped_locations[segment] += (
                            target_start - source_start
                        ) * shard_stride
                        keep |= segment
                    return mapped_locations[keep], values[keep]
                shard_end = min(
                    plan.shard_start + plan.shard_size, source_shape[shard_dim]
                )
                keep = (shard_coords >= plan.shard_start) & (shard_coords < shard_end)
                return (
                    locations[keep]
                    + plan.target_offset
                    - plan.shard_start * shard_stride,
                    values[keep],
                )

        selected_locations = locations
        selected_values = values

        if shard_dim is not None:
            shard_coords = torch.div(
                locations,
                source_strides[shard_dim],
                rounding_mode="floor",
            ).remainder(source_shape[shard_dim])
            shard_end = min(
                plan.shard_start + plan.shard_size,
                source_shape[shard_dim],
            )
            keep = (shard_coords >= plan.shard_start) & (shard_coords < shard_end)
            selected_locations = locations[keep]
            selected_values = values[keep]
            if selected_locations.numel() == 0:
                return selected_locations, selected_values

        local_locations = torch.full_like(selected_locations, plan.target_offset)
        for dim, (source_stride, target_stride) in enumerate(
            zip(source_strides, target_strides, strict=True)
        ):
            coord = torch.div(
                selected_locations,
                source_stride,
                rounding_mode="floor",
            ).remainder(source_shape[dim])
            if dim == plan.shard_dim:
                coord = coord - plan.shard_start
            local_locations.add_(coord * target_stride)
        return local_locations, selected_values

    def _direct_sparse_delta_tp_rank(self, tp_size: int) -> int:
        if tp_size <= 1:
            return 0
        rank = int(getattr(self, "rank", 0))
        return rank % tp_size

    def _map_direct_sparse_delta_name(self, name: str) -> str | None:
        mapper = getattr(self.model_runner.model, "hf_to_vllm_mapper", None)
        if mapper is not None:
            name = cast(Any, mapper)._map_name(name)
            if name is None:
                return None
        if name.startswith("draft."):
            return None
        return name

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
                    self._process_weights_after_loading(self.model_config, self.device)
                    self.zmq_socket.send(IPCProtocol.ACK.value.encode())
                    break

                ipc_handle, list_keys, used_bytes = payload
                buffer = rebuild_cuda_tensor_from_ipc(ipc_handle, self.device.index)
                state_dict_info = self.state_dict_info
                if state_dict_info is None:
                    raise RuntimeError(
                        "state_dict_info is not prepared. "
                        "Call prepare_refit_info before loading weights."
                    )

                weight = None
                weights = []
                offset = 0
                for key in list_keys:
                    shape, dtype = state_dict_info[key]
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

    @wrap_with_nvtx_name(
        "vllm_internal_worker_extension/update_weights_from_serialized_sparse_payload"
    )
    def update_weights_from_serialized_sparse_payload(
        self,
        serialized_payload: bytes,
        synchronize: bool = True,
    ) -> dict[str, Any]:
        """Apply one serialized sparse-delta payload received from S3."""
        started = time.perf_counter()
        payload = cast(
            sparse_codec.TensorPayload,
            torch.load(
                io.BytesIO(serialized_payload),
                map_location="cpu",
                weights_only=True,
            ),
        )
        deserialize_s = time.perf_counter() - started
        result = self._apply_sparse_request(payload, synchronize=synchronize)
        result["receiver_deserialize_s"] = deserialize_s
        result["receiver_total_s"] = time.perf_counter() - started
        return result

    def _apply_sparse_request(
        self,
        payload: sparse_codec.TensorPayload,
        *,
        synchronize: bool,
    ) -> dict[str, Any]:
        locations, values, metadata = payload

        sparse_started = time.perf_counter()
        self._apply_sparse_weight_deltas((locations, values), metadata)
        sparse_apply_s = time.perf_counter() - sparse_started

        if synchronize and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        return {
            "ok": True,
            "receiver_sparse_apply_s": sparse_apply_s,
        }

    def synchronize_device(self) -> dict[str, Any]:
        """Synchronize this vLLM worker's CUDA device after deferred refit applies."""
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        return {"ok": True}

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
