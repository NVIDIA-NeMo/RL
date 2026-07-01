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
import asyncio
import gc
import re
import time
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import torch
import zmq

from nemo_rl.models.generation.vllm.refit_layout import (
    VllmExpertParamLayout,
    VllmWeightLayout,
    parse_hf_expert_weight,
)
from nemo_rl.models.policy.utils import (
    IPCProtocol,
    calculate_aligned_size,
    rebuild_cuda_tensor_from_ipc,
)
from nemo_rl.utils.nsys import wrap_with_nvtx_name
from nemo_rl.utils.packed_tensor import packed_broadcast_consumer

if TYPE_CHECKING:
    from nemo_rl.utils.checkpoint_engines.base import CheckpointEngine

try:
    import vllm  # noqa: F401
except ImportError:
    raise ImportError(
        "vLLM is not installed. Please check that the py_executable in the runtime_env of VllmGenerationWorker "
        "covers the vllm dependency. You may have to update nemo_rl/distributed/ray_actor_environment_registry.py. "
        "This error can also happen if the venv creation was aborted or errored out in the middle. In that case, "
        "please run at least once with the environment variable NRL_FORCE_REBUILD_VENVS=true set to force the rebuild of the environment."
    )


def maybe_preinit_nixl_for_vllm_worker(
    worker_wrapper: Any,
    *,
    backend_name: str,
    backend_init_params: dict[str, Any] | None = None,
) -> None:  # pragma: no cover
    from nemo_rl.utils.checkpoint_engines.nixl import preinit_nixl_agent

    vars(worker_wrapper)["_nrl_nixl_preinit_agent"] = preinit_nixl_agent(
        backend_name=backend_name, backend_init_params=backend_init_params
    )


def _global_rollout_rank(rank_prefix: int, rollout_world_size: int) -> int:
    rank = torch.distributed.get_rank()
    if torch.distributed.get_world_size() == rollout_world_size:
        # External vLLM DP uses one distributed group across all rollout ranks.
        return rank
    return rank_prefix + rank


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
    checkpoint_engine: "CheckpointEngine"

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
        rank = train_world_size + _global_rollout_rank(
            rank_prefix, world_size - train_world_size
        )

        self.model_update_group = StatelessProcessGroup(  # pyrefly: ignore[implicitly-defined-attribute]  This class does not define __init__ so assignments like this should be ignored
            master_address=ip, port=port, rank=rank, world_size=world_size
        )
        self.model_update_group.init_nccl_communicator(device=self.device)

    def init_checkpoint_engine(
        self, backend: str, bucket_size_bytes: int, engine_kwargs: dict[str, Any]
    ) -> None:  # pragma: no cover
        if getattr(self, "checkpoint_engine", None) is not None:
            return

        from nemo_rl.utils.checkpoint_engines.base import create_checkpoint_engine

        self.checkpoint_engine = create_checkpoint_engine(
            backend,
            bucket_size_bytes=bucket_size_bytes,
            engine_kwargs=engine_kwargs,
        )

    def prepare_checkpoint_engine(self) -> Any:  # pragma: no cover
        metadata = self.checkpoint_engine.prepare()
        if isinstance(metadata, dict):
            metadata = {**metadata, "rank": torch.distributed.get_rank()}
            if getattr(self.checkpoint_engine, "shard_hf_weights", False):
                metadata["weight_layout"] = self._checkpoint_engine_weight_layout()
        return metadata

    def init_checkpoint_engine_process_group(
        self,
        rank_prefix: int,
        train_world_size: int,
        rollout_world_size: int,
        metadata: list[Any],
    ) -> None:  # pragma: no cover
        self.checkpoint_engine.init_rollout_process_group(
            rollout_rank=_global_rollout_rank(rank_prefix, rollout_world_size),
            train_world_size=train_world_size,
            rollout_world_size=rollout_world_size,
            metadata=metadata,
        )

    def finalize_checkpoint_engine(self) -> None:  # pragma: no cover
        checkpoint_engine = getattr(self, "checkpoint_engine", None)
        if checkpoint_engine is None:
            return
        checkpoint_engine.finalize()

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

    def _is_sharded_refit_weight(self, name: str, tensor: torch.Tensor) -> bool:
        param_names = self._sharded_refit_param_names(name)
        if not param_names:
            return False

        state_dict_info = getattr(self, "state_dict_info", None)
        if state_dict_info is None or name not in state_dict_info:
            return False
        full_shape, _dtype = state_dict_info[name]
        return torch.Size(full_shape) != tensor.shape

    def _sharded_refit_param_names(self, name: str) -> list[str]:
        expert_weight = parse_hf_expert_weight(name)
        return [] if expert_weight is None else [expert_weight.parameter_name]

    @contextmanager
    def _vllm_sharded_weight_load_context(self, param_names: list[str]):
        params = self._get_named_parameters()
        sharded_params = [
            params[param_name] for param_name in param_names if param_name in params
        ]
        old_sharded_attrs = [
            (
                param,
                hasattr(param, "is_sharded_weight"),
                getattr(param, "is_sharded_weight", None),
            )
            for param in sharded_params
        ]

        patched_loaders = []
        try:
            from vllm.model_executor.layers.fused_moe.layer import FusedMoE

            patched_loaders.append((FusedMoE, FusedMoE._load_w13, FusedMoE._load_w2))
        except (ImportError, AttributeError):
            pass

        try:
            from vllm.model_executor.layers.fused_moe.routed_experts import (
                RoutedExperts,
            )

            patched_loaders.append(
                (RoutedExperts, RoutedExperts._load_w13, RoutedExperts._load_w2)
            )
        except (ImportError, AttributeError):
            pass

        try:
            for loader_cls, original_load_w13, original_load_w2 in patched_loaders:

                def load_w13_sharded(
                    module, *args, _original=original_load_w13, **kwargs
                ):
                    kwargs["load_full"] = True
                    return _original(module, *args, **kwargs)

                def load_w2_sharded(
                    module, *args, _original=original_load_w2, **kwargs
                ):
                    kwargs["load_full"] = True
                    return _original(module, *args, **kwargs)

                loader_cls._load_w13 = load_w13_sharded
                loader_cls._load_w2 = load_w2_sharded

            for param in sharded_params:
                param.is_sharded_weight = True
            yield
        finally:
            for param, had_attr, old_value in old_sharded_attrs:
                if had_attr:
                    param.is_sharded_weight = old_value
                elif hasattr(param, "is_sharded_weight"):
                    delattr(param, "is_sharded_weight")
            for loader_cls, original_load_w13, original_load_w2 in patched_loaders:
                loader_cls._load_w13 = original_load_w13
                loader_cls._load_w2 = original_load_w2

    def _get_named_parameters(self) -> dict[str, torch.nn.Parameter]:
        params = getattr(self, "_nrl_named_parameters", None)
        if params is None:
            params = dict(self.model_runner.model.named_parameters())
            self._nrl_named_parameters = params
        return params

    def _checkpoint_engine_weight_layout(self) -> VllmWeightLayout:
        from vllm.model_executor.models.utils import get_pp_missing_layer_names

        expert_params: dict[str, VllmExpertParamLayout] = {}
        for name, param in self._get_named_parameters().items():
            if not name.endswith((".w13_weight", ".w2_weight")):
                continue

            if bool(getattr(param, "is_transposed", False)):
                raise ValueError(
                    "Sharded NIXL HF refit requires canonical expert-weight "
                    f"orientation, but {name} is transposed."
                )

            weight_loader = getattr(param, "weight_loader", None)
            owner = getattr(weight_loader, "__self__", None)
            if owner is None:
                raise RuntimeError(
                    f"Could not inspect the vLLM expert weight loader for {name}."
                )

            quant_method = getattr(owner, "base_quant_method", None)
            backend = getattr(quant_method, "unquantized_backend", None)
            backend_name = getattr(backend, "name", None)
            if getattr(owner, "quant_config", None) is not None or backend_name not in {
                "TRITON",
                "BATCHED_TRITON",
            }:
                raise ValueError(
                    "Sharded NIXL HF refit requires canonical unquantized Triton "
                    f"expert weights, but {name} uses "
                    f"{type(quant_method).__name__}/{backend_name}. Set "
                    "policy.generation.vllm_kwargs.moe_backend=triton."
                )

            use_ep = bool(getattr(owner, "use_ep", False))
            local_expert_ids: list[int] | None = None
            if use_ep:
                if bool(getattr(owner, "enable_eplb", False)):
                    raise RuntimeError(
                        "Sharded refit does not support dynamic vLLM expert load "
                        "balancing because ownership can change after metadata exchange."
                    )
                expert_map = getattr(owner, "_expert_map", None)
                if expert_map is None:
                    raise RuntimeError(
                        f"vLLM reports EP for {name} without an expert ownership map."
                    )
                logical_num_experts = int(
                    getattr(owner, "logical_num_experts", expert_map.numel())
                )
                global_num_experts = int(
                    getattr(owner, "global_num_experts", logical_num_experts)
                )
                if global_num_experts != logical_num_experts:
                    raise RuntimeError(
                        "Sharded refit does not support redundant vLLM experts."
                    )
                local_expert_ids = [
                    expert_id
                    for expert_id, local_id in enumerate(
                        expert_map.detach().cpu().tolist()[:logical_num_experts]
                    )
                    if int(local_id) >= 0
                ]

            expert_params[name] = {
                "tp_rank": int(getattr(owner, "tp_rank", 0)),
                "tp_size": int(getattr(owner, "tp_size", 1)),
                "local_expert_ids": local_expert_ids,
            }

        return {
            "expert_params": expert_params,
            "missing_weight_prefixes": get_pp_missing_layer_names(
                self.model_runner.model
            ),
        }

    def _local_expert_id(self, param: torch.nn.Parameter, expert_id: int) -> int:
        weight_loader = getattr(param, "weight_loader", None)
        owner = getattr(weight_loader, "__self__", None)
        mapper = getattr(owner, "_map_global_expert_id_to_local_expert_id", None)
        if mapper is None:
            return expert_id
        return int(mapper(expert_id))

    def _copy_sharded_expert_group(
        self,
        param: torch.nn.Parameter,
        shard_id: str,
        items: list[tuple[int, torch.Tensor]],
    ) -> None:
        sorted_items = sorted(items, key=lambda item: item[0])
        expert_ids = [expert_id for expert_id, _tensor in sorted_items]
        loaded_weight = torch.stack(
            [tensor for _expert_id, tensor in sorted_items], dim=0
        )

        param_data = param.data
        if shard_id in {"w1", "w3"}:
            shard_size = param_data.shape[1] // 2
            start = 0 if shard_id == "w1" else shard_size
            target = param_data.narrow(1, start, shard_size)
        elif shard_id == "w2":
            target = param_data
        else:
            raise ValueError(f"Unexpected sharded expert shard_id: {shard_id}")

        if target.shape[1] < loaded_weight.shape[1]:
            raise ValueError(
                f"Sharded expert target shape {tuple(target.shape)} is smaller "
                f"than loaded weight shape {tuple(loaded_weight.shape)}"
            )
        target = target.narrow(1, 0, loaded_weight.shape[1])

        if target.shape[2] < loaded_weight.shape[2]:
            raise ValueError(
                f"Sharded expert target shape {tuple(target.shape)} is smaller "
                f"than loaded weight shape {tuple(loaded_weight.shape)}"
            )
        target = target.narrow(2, 0, loaded_weight.shape[2])

        with torch.no_grad():
            contiguous_expert_ids = list(
                range(expert_ids[0], expert_ids[0] + len(expert_ids))
            )
            if expert_ids == contiguous_expert_ids:
                target.narrow(0, expert_ids[0], len(expert_ids)).copy_(loaded_weight)
            else:
                index = torch.tensor(expert_ids, device=target.device)
                target.index_copy_(0, index, loaded_weight)

    def _load_sharded_expert_weight_groups(
        self, weights: list[tuple[str, torch.Tensor]]
    ) -> list[tuple[str, torch.Tensor]]:
        params = self._get_named_parameters()
        groups: dict[tuple[str, str], list[tuple[int, torch.Tensor]]] = {}
        remaining_weights: list[tuple[str, torch.Tensor]] = []

        for name, tensor in weights:
            expert_weight = parse_hf_expert_weight(name)
            if expert_weight is None:
                remaining_weights.append((name, tensor))
                continue

            mapped_name = expert_weight.parameter_name
            param = params.get(mapped_name)
            if param is None or param.data.ndim != 3 or tensor.ndim != 2:
                remaining_weights.append((name, tensor))
                continue

            owner = getattr(getattr(param, "weight_loader", None), "__self__", None)
            if not self._is_sharded_refit_weight(name, tensor) and not bool(
                getattr(owner, "use_ep", False)
            ):
                remaining_weights.append((name, tensor))
                continue

            local_expert_id = self._local_expert_id(param, expert_weight.expert_id)
            if local_expert_id == -1:
                continue

            groups.setdefault((mapped_name, expert_weight.shard_id), []).append(
                (local_expert_id, tensor)
            )

        for (mapped_name, shard_id), items in groups.items():
            self._copy_sharded_expert_group(params[mapped_name], shard_id, items)

        return remaining_weights

    def _with_sharded_weight_load_contexts(
        self, weights: list[tuple[str, torch.Tensor]]
    ) -> Iterator[tuple[str, torch.Tensor]]:
        for name, tensor in weights:
            if self._is_sharded_refit_weight(name, tensor):
                with self._vllm_sharded_weight_load_context(
                    self._sharded_refit_param_names(name)
                ):
                    yield name, tensor
            else:
                yield name, tensor

    def _use_sharded_hf_refit(self) -> bool:
        checkpoint_engine = getattr(self, "checkpoint_engine", None)
        return bool(getattr(checkpoint_engine, "shard_hf_weights", False))

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
        use_sharded_hf_refit = self._use_sharded_hf_refit()
        if fp8.is_fp8_model(self.model_runner.vllm_config):
            if use_sharded_hf_refit:
                raise ValueError(
                    "Sharded NIXL HF refit is not supported for FP8 vLLM models."
                )
            fp8.load_weights(policy_weights, self.model_runner)
        else:
            if use_sharded_hf_refit:
                remaining_policy_weights = self._load_sharded_expert_weight_groups(
                    policy_weights
                )
                if remaining_policy_weights:
                    self.model_runner.model.load_weights(
                        weights=self._with_sharded_weight_load_contexts(
                            remaining_policy_weights
                        )
                    )
            elif policy_weights:
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

    async def _update_weights_from_checkpoint_engine_async(
        self,
    ) -> bool:  # pragma: no cover
        loaded_tensors = 0
        loaded_bytes = 0
        loaded_batches = 0
        load_time = 0.0
        start_time = time.time()

        async for weight_batch in self.checkpoint_engine.receive_weight_batches():
            loaded_batches += 1
            loaded_tensors += len(weight_batch)
            loaded_bytes += sum(weight.nbytes for _name, weight in weight_batch)

            load_start = time.time()
            self._load_weights(weight_batch)
            torch.cuda.current_stream().synchronize()
            load_time += time.time() - load_start
            del weight_batch

        self._maybe_process_fp8_kv_cache()

        if self.checkpoint_engine.cleanup_after_load:
            gc.collect()
            torch.cuda.empty_cache()

        total_time = time.time() - start_time
        loaded_gib = loaded_bytes / (1024 * 1024 * 1024)
        print(
            "[vLLM refit] Loaded "
            f"{loaded_tensors} tensors in {loaded_batches} batches via checkpoint "
            f"engine; bytes={loaded_gib:.2f}GiB total={total_time:.2f}s "
            f"receive={max(total_time - load_time, 0.0):.2f}s load={load_time:.2f}s"
        )
        return True

    @wrap_with_nvtx_name(
        "vllm_internal_worker_extension/update_weights_from_checkpoint_engine"
    )
    def update_weights_from_checkpoint_engine(self) -> bool:  # pragma: no cover
        return asyncio.run(self._update_weights_from_checkpoint_engine_async())

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
