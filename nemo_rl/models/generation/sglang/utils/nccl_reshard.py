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

"""BF16 SGLang receiver for the shared NCCL shard-to-shard refit transport."""

import base64
import json
import pickle
from collections.abc import Callable
from typing import Any, cast
from weakref import WeakKeyDictionary, ref

import torch

from nemo_rl.distributed.refit_watchdog import RefitAborted, RefitAbortWatchdog
from nemo_rl.utils.packed_tensor import packed_broadcast_consumer
from nemo_rl.weight_sync.nccl_reshard_utils import (
    _STR_TO_DTYPE,
    HFToLocalParamMap,
    LocalParamSpec,
    restore_refit_info_placements,
)

SGLANG_RESHARD_LOADER = (
    "nemo_rl.models.generation.sglang.utils.nccl_reshard.load_nccl_reshard"
)


def serialize_nccl_reshard_command(operation: str, payload: dict[str, Any]) -> str:
    """Inline control bytes so independent serving processes need no IPC handles."""
    control = torch.tensor(
        list(json.dumps(payload).encode()), dtype=torch.uint8, device="cpu"
    )
    return base64.b64encode(pickle.dumps([(operation, control)])).decode("ascii")


def build_sglang_refit_param_map(
    model: torch.nn.Module, refit_info: dict[str, Any], *, tp_size: int
) -> HFToLocalParamMap:
    """Map dense Qwen2 HF projections onto SGLang's live TP-local storage."""
    params = dict(model.named_parameters())
    specs: dict[str, LocalParamSpec] = {}
    for layer in refit_info["layer_names"]:
        for info in refit_info["per_layer_params"][layer]:
            name = info["name"]
            if ".experts." in name or info.get("grouped_expert_proj"):
                raise ValueError("SGLang NCCL reshard does not support MoE weights.")
            if _STR_TO_DTYPE[info["dtype"]] != torch.bfloat16:
                raise ValueError(f"SGLang NCCL reshard requires BF16: {name!r}.")
            shape = list(info["global_shape"])
            if name.endswith((".gate_proj.weight", ".up_proj.weight")):
                suffix = name.rsplit(".", 2)[1]
                target = params[name.replace(f".{suffix}.", ".gate_up_proj.")]
                if target.shape[0] % 2:
                    raise ValueError(f"Uneven fused gate/up storage for {name!r}.")
                half = target.shape[0] // 2
                offset = 0 if suffix == "gate_proj" else half
                local = target.data.narrow(0, offset, half)
                shard_dim = 0
            elif name.endswith(".down_proj.weight"):
                local = params[name].data
                shard_dim = 1
            else:
                raise ValueError(f"Unsupported SGLang bulk parameter {name!r}.")
            if len(shape) != 2 or shape[shard_dim] % tp_size:
                raise ValueError(f"Invalid TP{tp_size} shape for {name!r}: {shape}.")
            shape[shard_dim] //= tp_size
            if list(local.shape) != shape or not local.is_contiguous():
                raise ValueError(
                    f"SGLang destination {name!r}: expected contiguous {shape}, "
                    f"got {list(local.shape)}."
                )
            if local.dtype != torch.bfloat16:
                raise ValueError(f"SGLang destination {name!r} must store BF16.")
            specs[name] = LocalParamSpec(base=local, wire_dtype=torch.bfloat16)
    if not specs:
        raise ValueError("SGLang NCCL reshard requires at least one bulk parameter.")
    return HFToLocalParamMap(specs=specs)


class SGLangNcclReshardReceiver:
    """Own communicator and destination state for one SGLang GPU model."""

    def __init__(self, model: torch.nn.Module, *, tp_rank: int, tp_size: int):
        self._model_ref = ref(model)
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.model_update_group: Any = None
        self.pp_comm_groups: dict[int, Any] = {}
        self.refit_info: dict[str, Any] | None = None
        self.param_map = HFToLocalParamMap()

    @property
    def model(self) -> torch.nn.Module:
        """Return the live serving model without extending its lifetime."""
        model = self._model_ref()
        if model is None:
            raise RuntimeError("SGLang reshard serving model was released.")
        return model

    def init_collective(
        self, *, ip: str, port: int, world_size: int, train_world_size: int
    ) -> None:
        """Join the same miscellaneous-weight communicator as the trainer."""
        # NCCL is optional in CPU-only and driver environments.
        from nemo_rl.distributed.stateless_process_group import StatelessProcessGroup

        if self.model_update_group is not None:
            raise RuntimeError("SGLang reshard collective is already initialized.")
        group = StatelessProcessGroup(
            master_address=ip,
            port=port,
            rank=train_world_size + self.tp_rank,
            world_size=world_size,
        )
        group.init_nccl_communicator(device=torch.cuda.current_device())
        self.model_update_group = group

    def init_bulk(
        self,
        *,
        pp_ips: list[str],
        pp_ports: list[int],
        pp_size: int,
        train_ranks_per_stage: int,
        sub_world_size: int,
    ) -> None:
        """Join each training stage's bulk reshard communicator."""
        # NCCL is optional in CPU-only and driver environments.
        from nemo_rl.distributed.stateless_process_group import StatelessProcessGroup

        if self.pp_comm_groups:
            raise RuntimeError("SGLang bulk communicators are already initialized.")
        for stage in range(pp_size):
            group = StatelessProcessGroup(
                master_address=pp_ips[stage],
                port=pp_ports[stage],
                rank=train_ranks_per_stage + self.tp_rank,
                world_size=sub_world_size,
            )
            group.init_nccl_communicator(device=torch.cuda.current_device())
            self.pp_comm_groups[stage] = group

    def prepare(self, *, refit_info: dict[str, Any]) -> None:
        """Validate all live destination views before starting a transfer."""
        if self.model.config.model_type != "qwen2":
            raise ValueError("SGLang NCCL reshard currently supports dense Qwen2 only.")
        restored = restore_refit_info_placements(refit_info)
        mapping = build_sglang_refit_param_map(
            self.model, restored, tp_size=self.tp_size
        )
        self.param_map = mapping
        self.refit_info = restored

    @torch.no_grad()
    def refit(self, *, refit_timeout_s: float | None) -> None:
        """Receive bulk shards directly, then load full miscellaneous tensors."""
        # Loading the transport imports the optional NCCL bindings.
        from nemo_rl.weight_sync.xferdtensor import DTensorRef, xferdtensor

        refit_info = self.refit_info
        if refit_info is None or self.model_update_group is None:
            raise RuntimeError(
                "SGLang reshard requires communicator and metadata setup."
            )
        stream = torch.cuda.current_stream()
        groups = [self.model_update_group, *self.pp_comm_groups.values()]
        with RefitAbortWatchdog(groups, refit_timeout_s) as guard:
            for layer in refit_info["layer_names"]:
                for info in refit_info["per_layer_params"][layer]:
                    spec = self.param_map.specs[info["name"]]
                    ctx = spec.materialize()
                    xferdtensor(
                        None,
                        info["src_mesh_info"],
                        info["src_placements"],
                        DTensorRef(ctx.buf, info["global_shape"]),
                        info["dst_mesh_info"],
                        info["dst_placements"],
                        self.pp_comm_groups[info.get("pp_stage", 0)],
                        stream,
                    )
            stream.synchronize()
            misc = {
                name: (tuple(meta["shape"]), _STR_TO_DTYPE[meta["dtype"]])
                for name, meta in refit_info["misc_meta"].items()
            }
            if misc:
                packed_broadcast_consumer(
                    iterator=iter(misc.items()),
                    group=self.model_update_group,
                    src=0,
                    post_unpack_func=cast(
                        Callable[[list[tuple[str, torch.Tensor]]], None],
                        self.model.load_weights,
                    ),
                )
            torch.cuda.synchronize()
        if guard.fired:
            raise RefitAborted("SGLang NCCL reshard exceeded its refit deadline.")


_RECEIVERS: WeakKeyDictionary[torch.nn.Module, SGLangNcclReshardReceiver] = (
    WeakKeyDictionary()
)


def load_nccl_reshard(
    model: torch.nn.Module, named_tensors: list[tuple[str, torch.Tensor]]
) -> None:
    """Execute a reshard command through SGLang's registered custom-loader API."""
    # SGLang is installed only in the serving workers.
    from sglang.srt.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )

    if len(named_tensors) != 1:
        raise ValueError("SGLang reshard expects exactly one control tensor.")
    operation, control = named_tensors[0]
    if control.dtype != torch.uint8 or control.ndim != 1:
        raise ValueError(
            "SGLang reshard control must be a one-dimensional byte tensor."
        )
    payload = json.loads(bytes(control.cpu().tolist()))
    receiver = _RECEIVERS.get(model)
    if receiver is None:
        receiver = SGLangNcclReshardReceiver(
            model,
            tp_rank=get_tensor_model_parallel_rank(),
            tp_size=get_tensor_model_parallel_world_size(),
        )
        _RECEIVERS[model] = receiver
    operations = {
        "init_collective": receiver.init_collective,
        "init_bulk": receiver.init_bulk,
        "prepare": receiver.prepare,
        "refit": receiver.refit,
    }
    if operation not in operations:
        raise ValueError(f"Unknown SGLang reshard operation {operation!r}.")
    operations[operation](**payload)
