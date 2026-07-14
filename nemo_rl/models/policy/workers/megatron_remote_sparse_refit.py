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

"""Optional remote sparse-refit state owned by a Megatron policy worker."""

import re
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from functools import cache, partial
from typing import Any, cast

import torch

from nemo_rl.utils.weight_transfer_remote_sparse import (
    SparseDeltaStreamResult,
    init_sparse_delta_baseline_from_iterator,
    sparse_name_shard,
    stream_sparse_delta_payloads_via_s3_manifest,
)
from nemo_rl.utils.weight_transfer_sparse_codec import (
    DeltaCompressionTracker,
    SparseShardProjection,
)
from nemo_rl.utils.weight_transfer_zmq import stream_sparse_delta_payloads_via_zmq

_UNSUPPORTED = 0
_COLUMN = 1
_ROW = 2
_REPLICATED = 3
_DIRECT = 4
_GATED = 5


class MegatronRemoteSparseRefit:
    @classmethod
    def from_worker(cls, worker: Any) -> "MegatronRemoteSparseRefit":
        generation_config = worker.cfg.get("generation") or {}
        delta_config = generation_config.get("delta_compression")
        if generation_config.get("refit_transport") is None or not delta_config:
            raise RuntimeError("Remote sparse refit is not enabled for this worker.")
        return cls(worker, delta_config)

    def __init__(self, worker: Any, delta_config: Mapping[str, Any]) -> None:
        self._worker = worker
        self._delta_config = delta_config
        residual_config = dict(delta_config)
        if residual_config["encoding"] == "xor":
            residual_config["encoding"] = "overwrite"
        self._tracker = DeltaCompressionTracker(residual_config)
        self._policy_tracker: DeltaCompressionTracker | None = None
        self._local_tensors: list[tuple[str, torch.Tensor]] = []
        self._misc_local_tensors: list[tuple[str, torch.Tensor]] = []
        self._misc_conversion_tasks: list[Any] | None = None
        self._filter_misc_tasks = False

    @staticmethod
    @cache
    def _bridge_mapping_types() -> tuple[Any, dict[Any, int]]:
        # Bridge is optional outside Megatron workers, so keep these imports local.
        from megatron.bridge.models.conversion.param_mapping import (
            AutoMapping,
            ColumnParallelMapping,
            DirectMapping,
            GatedMLPMapping,
            ReplicatedMapping,
            RowParallelMapping,
        )

        return AutoMapping, {
            ColumnParallelMapping: _COLUMN,
            DirectMapping: _DIRECT,
            GatedMLPMapping: _GATED,
            ReplicatedMapping: _REPLICATED,
            RowParallelMapping: _ROW,
        }

    @staticmethod
    def _all_reduce_max(values: list[int]) -> list[int]:
        if not values or not torch.distributed.is_initialized():
            return values
        backend = str(torch.distributed.get_backend()).lower()
        device = (
            torch.device("cuda", torch.cuda.current_device())
            if backend.endswith("nccl")
            else torch.device("cpu")
        )
        reduced = torch.tensor(values, dtype=torch.int32, device=device)
        torch.distributed.all_reduce(reduced, op=torch.distributed.ReduceOp.MAX)
        return reduced.cpu().tolist()

    @classmethod
    def _local_mapping_kind(cls, task: Any) -> int:
        AutoMapping, mapping_kinds = cls._bridge_mapping_types()
        mapping = task.mapping
        if kind := mapping_kinds.get(type(mapping)):
            return kind
        if (
            type(mapping) is AutoMapping
            and mapping.permute_dims is None
            and task.megatron_module is not None
        ):
            return {
                "column": _COLUMN,
                "row": _ROW,
                "replicated": _REPLICATED,
            }.get(mapping._detect_parallelism_type(task.megatron_module), _UNSUPPORTED)
        return _UNSUPPORTED

    def _bridge_exports_are_identity(self) -> bool:
        bridge = getattr(self._worker, "megatron_bridge", None)
        if bridge is None:
            return True
        from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge

        model_bridge = bridge._model_bridge
        return (
            type(model_bridge).maybe_modify_converted_hf_weight
            is MegatronModelBridge.maybe_modify_converted_hf_weight
        )

    @staticmethod
    def _is_padded_or_tied_weight(task: Any) -> bool:
        hf_names = (
            task.mapping.hf_param.values()
            if isinstance(task.mapping.hf_param, dict)
            else (task.mapping.hf_param,)
        )
        return task.global_param_name.endswith(
            ("embedding.word_embeddings.weight", "output_layer.weight")
        ) or any(
            str(name).endswith(
                ("embed_tokens.weight", "embeddings.weight", "lm_head.weight")
            )
            for name in hf_names
        )

    @staticmethod
    def _owns_policy_local_task(task: Any, *, replicated: bool = False) -> bool:
        if not torch.distributed.is_initialized():
            return True
        from megatron.core import parallel_state

        if task.mapping.is_expert:
            # Expert-DP includes DP replicas and TP replicas when ETP < TP.
            replica_rank = parallel_state.get_expert_data_parallel_rank()
            replica_count = parallel_state.get_expert_data_parallel_world_size()
        else:
            replica_rank = parallel_state.get_data_parallel_rank(
                with_context_parallel=True
            )
            replica_count = parallel_state.get_data_parallel_world_size(
                with_context_parallel=True
            )
            if replicated:
                replica_rank = (
                    replica_rank * task.mapping.tp_size + task.mapping.tp_rank
                )
                replica_count *= task.mapping.tp_size
        return replica_rank == sparse_name_shard(task.global_param_name, replica_count)

    @classmethod
    def _task_ownership(cls, task: Any, kind: int) -> tuple[torch.Tensor | None, bool]:
        tensor = task.param_weight
        replicated = kind in (_DIRECT, _REPLICATED) or (
            kind == _ROW and tensor is not None and tensor.ndim == 1
        )
        return (
            tensor
            if tensor is not None
            and cls._owns_policy_local_task(task, replicated=replicated)
            else None,
            replicated,
        )

    def _policy_local_path_is_safe(self) -> bool:
        config = getattr(self._worker, "cfg", {})
        ddp_config = config.get("megatron_cfg", {}).get(
            "distributed_data_parallel_config", {}
        )
        fp8_cfg = getattr(self._worker, "fp8_cfg", None)
        return (
            config.get("quant_cfg") is None
            and not ddp_config.get("use_custom_fsdp", False)
            and (not fp8_cfg or not fp8_cfg.get("fp8_param", False))
        )

    @staticmethod
    def _canonical_hf_name(task: Any, name: str) -> str:
        mapping = task.mapping
        if not mapping.is_expert or mapping.ep_size == 1:
            return name

        match = re.search(r"(?<=\.experts\.)\d+(?=\.)", name)
        config = getattr(task.megatron_module, "config", None)
        num_experts = getattr(config, "num_moe_experts", None)
        if match is None or not isinstance(num_experts, int):
            raise ValueError(f"Cannot project expert parameter {name!r}.")
        if num_experts % mapping.ep_size:
            raise ValueError(
                f"Expert count {num_experts} is not divisible by EP size "
                f"{mapping.ep_size}."
            )
        experts_per_rank = num_experts // mapping.ep_size
        expert = int(match.group()) % experts_per_rank
        expert += experts_per_rank * mapping.ep_rank
        return f"{name[: match.start()]}{expert}{name[match.end() :]}"

    @staticmethod
    def _projection(
        name: str,
        tensor: torch.Tensor,
        *,
        shard_dim: int,
        shard_rank: int,
        shard_count: int,
    ) -> SparseShardProjection:
        if tensor.ndim <= shard_dim:
            raise ValueError(f"Cannot shard {name!r} on dimension {shard_dim}.")
        global_shape = list(tensor.shape)
        offsets = [0] * tensor.ndim
        global_shape[shard_dim] *= shard_count
        offsets[shard_dim] = tensor.shape[shard_dim] * shard_rank
        return SparseShardProjection(name, tuple(global_shape), tuple(offsets))

    @classmethod
    def _task_local_tensors(
        cls, task: Any, kind: int, *, identity_export: bool = True
    ) -> list[tuple[str, torch.Tensor, SparseShardProjection]] | None:
        mapping = task.mapping
        hf_param = mapping.hf_param
        if (
            kind == _UNSUPPORTED
            or not identity_export
            or getattr(mapping, "is_grouped_export", False)
            or getattr(mapping, "is_adapter", False)
            or cls._is_padded_or_tied_weight(task)
        ):
            return None
        if kind == _GATED:
            if not isinstance(hf_param, dict) or set(hf_param) != {
                "gate",
                "up",
            }:
                return None
        elif not isinstance(hf_param, str):
            return None
        tensor, replicated = cls._task_ownership(task, kind)
        if tensor is None:
            return []
        if kind == _GATED:
            gate, up = torch.chunk(tensor, 2, dim=0)
            return [
                (
                    f"{task.global_param_name}:{role}",
                    value,
                    cls._projection(
                        cls._canonical_hf_name(
                            task, str(cast(dict[str, Any], hf_param)[role])
                        ),
                        value,
                        shard_dim=0,
                        shard_rank=mapping.tp_rank,
                        shard_count=mapping.tp_size,
                    ),
                )
                for role, value in (("gate", gate), ("up", up))
            ]

        name = cls._canonical_hf_name(task, cast(str, hf_param))
        projection = (
            SparseShardProjection(name, tuple(tensor.shape), (0,) * tensor.ndim)
            if replicated
            else cls._projection(
                name,
                tensor,
                shard_dim=0 if kind == _COLUMN else 1,
                shard_rank=mapping.tp_rank,
                shard_count=mapping.tp_size,
            )
        )
        return [(task.global_param_name, tensor, projection)]

    def _prepare_paths(self) -> None:
        if self._misc_conversion_tasks is not None:
            return
        tasks = [
            task
            for task in self._worker.megatron_bridge.get_conversion_tasks(
                [self._worker.model]
            )
            if task is not None
        ]
        misc_tasks: list[Any] = []
        self._misc_conversion_tasks = misc_tasks
        if not tasks or not self._policy_local_path_is_safe():
            misc_tasks.extend(tasks)
            return
        kinds = self._all_reduce_max([self._local_mapping_kind(task) for task in tasks])
        identity_export = self._bridge_exports_are_identity()
        self._filter_misc_tasks = identity_export
        projections = {}
        for task, kind in zip(tasks, kinds, strict=True):
            local_tensors = self._task_local_tensors(
                task, kind, identity_export=identity_export
            )
            if local_tensors is not None:
                for key, tensor, projection in local_tensors:
                    if key in projections:
                        raise ValueError(
                            f"Duplicate policy-local sparse shard {key!r}."
                        )
                    projections[key] = projection
                    self._local_tensors.append((key, tensor))
                continue

            task_index = len(misc_tasks)
            misc_tasks.append(task)
            tensor, _ = self._task_ownership(task, kind)
            if tensor is not None:
                key = f"{task_index}:{task.global_param_name}"
                self._misc_local_tensors.append((key, tensor))

        self._policy_tracker = DeltaCompressionTracker(
            self._delta_config, projections=projections
        )

    def _iter_misc_params(
        self, conversion_tasks: list[Any] | None = None
    ) -> Iterable[tuple[str, torch.Tensor]]:
        tasks = (
            self._misc_conversion_tasks
            if conversion_tasks is None
            else conversion_tasks
        )
        return self._worker._iter_params_with_optional_kv_scales(conversion_tasks=tasks)

    def _changed_misc_tasks(self) -> tuple[list[Any], int, int]:
        assert self._misc_conversion_tasks is not None
        changed_keys: set[str] = set()
        changed = total = 0
        if self._misc_local_tensors:
            assert self._policy_tracker is not None
            changed_keys, changed, total = self._policy_tracker.prepare_change_summary(
                self._misc_local_tensors
            )
        flags = [0] * len(self._misc_conversion_tasks)
        for key in changed_keys:
            flags[int(key.partition(":")[0])] = 1
        flags = self._all_reduce_max(flags)
        if not any(flags):
            tasks = []
        elif not self._filter_misc_tasks:
            tasks = self._misc_conversion_tasks
        else:
            grouped_keys = {
                task.mapping.group_key
                for task, task_changed in zip(
                    self._misc_conversion_tasks, flags, strict=True
                )
                if task_changed and getattr(task.mapping, "is_grouped_export", False)
            }
            tasks = [
                task
                for task, task_changed in zip(
                    self._misc_conversion_tasks, flags, strict=True
                )
                if task_changed
                or (
                    getattr(task.mapping, "is_grouped_export", False)
                    and task.mapping.group_key in grouped_keys
                )
            ]
        return tasks, changed, total

    def initialize_baseline(
        self,
        *,
        shard_rank: int,
        shard_count: int,
        transport: str,
    ) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
        self._prepare_paths()
        snapshot = partial(
            init_sparse_delta_baseline_from_iterator,
            shard_rank=shard_rank,
            shard_count=shard_count,
            transport=transport,
        )
        policy_tracker = self._policy_tracker
        if policy_tracker is None:
            snapshot(self._iter_misc_params(), delta_tracker=self._tracker)
            return self.refit_info()

        with ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="nrl-refit-policy-local"
        ) as executor:
            local_future = executor.submit(
                snapshot,
                self._local_tensors + self._misc_local_tensors,
                delta_tracker=policy_tracker,
                partition="none",
            )
            snapshot(
                self._iter_misc_params(),
                delta_tracker=self._tracker,
                partition="names",
            )
            local_future.result()
        return self.refit_info()

    def refit_info(self) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
        info = {
            name: (tuple(tensor.shape), tensor.dtype)
            for name, tensor in self._tracker.baseline.items()
        }
        if self._policy_tracker is not None:
            for name, tensor in self._local_tensors:
                projection = self._policy_tracker.projections[name]
                info[projection.name] = (projection.global_shape, tensor.dtype)
        return info

    def stream(
        self,
        transport: str,
        targets: list[str],
        *,
        transfer_id: str,
        api_key_env_var: str | None,
        timeout_s: float,
        shard_rank: int,
        shard_count: int,
    ) -> SparseDeltaStreamResult:
        self._prepare_paths()
        streamer = {
            "s3": stream_sparse_delta_payloads_via_s3_manifest,
            "zmq": stream_sparse_delta_payloads_via_zmq,
        }[transport]
        send = partial(
            streamer,
            refit_targets=targets,
            api_key_env_var=api_key_env_var,
            timeout_s=timeout_s,
            shard_rank=shard_rank,
            shard_count=shard_count,
        )
        policy_tracker = self._policy_tracker
        if policy_tracker is None:
            result = send(
                self._iter_misc_params(),
                delta_tracker=self._tracker,
                transfer_id=transfer_id,
            )
        else:
            with ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="nrl-refit-policy-local"
            ) as executor:
                local_future = None
                if self._local_tensors:
                    local_future = executor.submit(
                        send,
                        self._local_tensors,
                        delta_tracker=policy_tracker,
                        transfer_id=f"{transfer_id}-local",
                        partition="none",
                    )
                changed_tasks, misc_changed, misc_total = self._changed_misc_tasks()
                misc_result = send(
                    self._iter_misc_params(changed_tasks),
                    delta_tracker=self._tracker,
                    transfer_id=f"{transfer_id}-misc",
                    partition="names",
                )
                local_result = (
                    local_future.result()
                    if local_future is not None
                    else {"payloads": 0, "changed_elements": 0, "total_elements": 0}
                )
            result = SparseDeltaStreamResult(
                payloads=int(local_result["payloads"]) + int(misc_result["payloads"]),
                changed_elements=int(local_result["changed_elements"]) + misc_changed,
                total_elements=int(local_result["total_elements"]) + misc_total,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return result

    def finish(self, succeeded: bool) -> None:
        for tracker in (self._tracker, self._policy_tracker):
            if tracker is None:
                continue
            if succeeded:
                tracker.on_sync_succeeded()
            else:
                tracker.on_sync_failed()
