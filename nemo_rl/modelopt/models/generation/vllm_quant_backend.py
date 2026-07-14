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
import os
import re
import types
from contextlib import ExitStack, contextmanager
from typing import Any

import torch
import vllm  # noqa: F401
import zmq
from modelopt.torch.quantization.nn.modules.tensor_quantizer import TensorQuantizer

from nemo_rl.modelopt.utils import (
    MODELOPT_REAL_QUANT_ZMQ_TIMEOUT_MS,
    matches_quant_ignore_pattern,
)
from nemo_rl.models.generation.vllm.vllm_backend import VllmInternalWorkerExtension
from nemo_rl.models.policy.utils import (
    IPCProtocol,
    calculate_aligned_size,
    rebuild_cuda_tensor_from_ipc,
)
from nemo_rl.utils.packed_tensor import packed_broadcast_consumer

_FUSED_MODELOPT_MOE_SUFFIXES = {
    ".experts.w13_weight": "w13_weight",
    ".experts.w13_weight_scale": "w13_weight_scale",
    ".experts.w13_weight_scale_2": "w13_weight_scale_2",
    ".experts.up_proj": "up_proj.weight",
    ".experts.up_proj_scale": "up_proj.weight_scale",
    ".experts.up_proj_scale_2": "up_proj.weight_scale_2",
    ".experts.up_proj_input_scale": "up_proj.input_scale",
    ".experts.w2_weight": "down_proj.weight",
    ".experts.w2_weight_scale": "down_proj.weight_scale",
    ".experts.w2_weight_scale_2": "down_proj.weight_scale_2",
    ".experts.w13_input_scale": "w13_input_scale",
    ".experts.w2_input_scale": "w2_input_scale",
}
_FUSED_MODELOPT_MOE_DIRECT_SCALE_SUFFIXES = {
    ".experts.w13_weight_scale_2": "w13_weight_scale_2",
    ".experts.w2_weight_scale_2": "w2_weight_scale_2",
    ".experts.up_proj_scale_2": "w13_weight_scale_2",
    ".experts.up_proj_input_scale": "w13_input_scale",
    ".experts.w13_input_scale": "w13_input_scale",
    ".experts.w2_input_scale": "w2_input_scale",
}
_EXPERT_MODELOPT_SCALE_RE = re.compile(
    r"^(?P<prefix>.+)\.experts\.(?P<expert>\d+)\."
    r"(?P<projection>gate_proj|up_proj|down_proj)\."
    r"(?P<scale>input_scale|weight_scale_2)$"
)


def _format_refit_key_error(label: str, keys: set[str]) -> str:
    """Format a bounded refit-key diagnostic."""
    ordered = sorted(keys)
    suffix = " ..." if len(ordered) > 8 else ""
    return f"{label} ({len(ordered)}): {ordered[:8]}{suffix}"


def _match_fused_modelopt_moe_weight(name: str) -> tuple[str, str] | None:
    return next(
        (
            (suffix, target)
            for suffix, target in _FUSED_MODELOPT_MOE_SUFFIXES.items()
            if name.endswith(suffix)
        ),
        None,
    )


def _is_fused_modelopt_moe_weight(name: str) -> bool:
    return _match_fused_modelopt_moe_weight(name) is not None


def _w13_num_shards_from_state_dict_info(
    state_dict_info: dict[str, Any],
    *,
    require_input_scales: bool = False,
) -> dict[str, int]:
    """Validate complete fused-MoE families and resolve their W13 layout."""
    num_shards_by_prefix: dict[str, int] = {}
    input_shards_by_prefix: dict[str, int] = {}
    targets_by_prefix: dict[str, set[str]] = {}
    for name, (shape, _dtype) in state_dict_info.items():
        matched = _match_fused_modelopt_moe_weight(name)
        if matched is None:
            continue
        suffix, target = matched
        prefix = name[: -len(suffix)]
        if target.startswith("up_proj."):
            target = "w13_" + target.removeprefix("up_proj.")
        elif target.startswith("down_proj."):
            target = "w2_" + target.removeprefix("down_proj.")
        targets_by_prefix.setdefault(prefix, set()).add(target)
        if target == "w13_input_scale":
            if len(shape) == 1:
                input_shards = 1
            elif len(shape) == 2 and shape[1] in {1, 2}:
                input_shards = shape[1]
            else:
                raise ValueError(
                    f"Expected one or two W13 input scales per expert for {name}, "
                    f"got {tuple(shape)}"
                )
            input_shards_by_prefix[prefix] = input_shards
        if target != "w13_weight_scale_2":
            continue
        if len(shape) == 1:
            num_shards = 1
        elif len(shape) == 2 and shape[1] in {1, 2}:
            num_shards = shape[1]
        else:
            raise ValueError(
                f"Expected one or two W13 global scales per expert for {name}, "
                f"got {tuple(shape)}"
            )
        num_shards_by_prefix[prefix] = num_shards

    required_targets = {
        "w13_weight",
        "w13_weight_scale",
        "w13_weight_scale_2",
        "w2_weight",
        "w2_weight_scale",
        "w2_weight_scale_2",
    }
    if require_input_scales:
        required_targets.update({"w13_input_scale", "w2_input_scale"})
    for prefix, targets in targets_by_prefix.items():
        missing = required_targets - targets
        if missing:
            raise RuntimeError(
                f"Incomplete ModelOpt MoE export family for {prefix}: "
                f"missing {sorted(missing)}"
            )
    if set(num_shards_by_prefix) != set(targets_by_prefix):
        missing = set(targets_by_prefix) - set(num_shards_by_prefix)
        raise RuntimeError(
            "ModelOpt MoE export families are missing W13 global scales: "
            f"{sorted(missing)}"
        )
    if require_input_scales:
        mismatched = {
            prefix
            for prefix, num_shards in num_shards_by_prefix.items()
            if input_shards_by_prefix.get(prefix) != num_shards
        }
        if mismatched:
            raise RuntimeError(
                "ModelOpt MoE W13 input/global scale layouts disagree for: "
                f"{sorted(mismatched)}"
            )
    return num_shards_by_prefix


def _batch_fused_modelopt_moe_weights(
    weights: list[tuple[str, torch.Tensor]],
    *,
    w13_num_shards_by_prefix: dict[str, int],
) -> list[tuple[str, torch.Tensor]]:
    """Map fused ModelOpt payloads to vLLM per-projection checkpoint names.

    Large expert weights and block scales stay batched so vLLM can
    tensor-parallel-shard the full ``[E, ...]`` tensor at once.  Its scalar
    loader still requires an expert id, so only the tiny per-expert global
    scales are exposed as scalar views.
    """
    batched: list[tuple[str, torch.Tensor]] = []
    for name, tensor in weights:
        matched = _match_fused_modelopt_moe_weight(name)
        if matched is None:
            batched.append((name, tensor))
            continue

        suffix, target = matched
        prefix = name[: -len(suffix)]
        if tensor.ndim == 0:
            raise ValueError(
                f"Fused ModelOpt MoE tensor must have an expert dimension: {name}"
            )

        if target in {"w13_weight", "w13_weight_scale"}:
            target_suffix = "weight" if target == "w13_weight" else "weight_scale"
            if w13_num_shards_by_prefix.get(prefix) == 1:
                batched.append(
                    (
                        f"{prefix}.experts.0.up_proj.{target_suffix}",
                        tensor,
                    )
                )
                continue
            if tensor.ndim < 2 or tensor.shape[1] % 2 != 0:
                raise ValueError(
                    f"Expected fused gate/up tensor with an even projection "
                    f"dimension for {name}, got {tuple(tensor.shape)}"
                )
            gate, up = tensor.chunk(2, dim=1)
            batched.extend(
                (
                    f"{prefix}.experts.0.{projection}.{target_suffix}",
                    shard.contiguous(),
                )
                for projection, shard in (
                    ("gate_proj", gate),
                    ("up_proj", up),
                )
            )
            continue

        if target == "w13_input_scale":
            if tensor.ndim == 1:
                tensor = tensor[:, None]
            if tensor.ndim != 2 or tensor.shape[1] not in {1, 2}:
                raise ValueError(
                    f"Expected one or two W13 input scales per expert for {name}, "
                    f"got {tuple(tensor.shape)}"
                )
            if tensor.shape[1] == 1:
                batched.extend(
                    (
                        f"{prefix}.experts.{expert_id}.up_proj.input_scale",
                        expert_scale[0],
                    )
                    for expert_id, expert_scale in enumerate(tensor.unbind(0))
                )
                continue
            for expert_id, expert_scale in enumerate(tensor.unbind(0)):
                batched.append(
                    (
                        f"{prefix}.experts.{expert_id}.gate_proj.input_scale",
                        expert_scale[0],
                    )
                )
                batched.append(
                    (
                        f"{prefix}.experts.{expert_id}.up_proj.input_scale",
                        expert_scale[1],
                    )
                )
            continue

        if target == "up_proj.input_scale":
            if tensor.ndim == 2 and tensor.shape[1] == 1:
                tensor = tensor[:, 0]
            if tensor.ndim != 1:
                raise ValueError(
                    f"Expected one non-gated up-projection input scale per "
                    f"expert for {name}, got {tuple(tensor.shape)}"
                )
            batched.extend(
                (f"{prefix}.experts.{expert_id}.up_proj.input_scale", scale)
                for expert_id, scale in enumerate(tensor.unbind(0))
            )
            continue

        if target == "w2_input_scale":
            if tensor.ndim == 2 and tensor.shape[1] == 1:
                tensor = tensor[:, 0]
            if tensor.ndim != 1:
                raise ValueError(
                    f"Expected one down-projection input scale per expert for "
                    f"{name}, got {tuple(tensor.shape)}"
                )
            batched.extend(
                (f"{prefix}.experts.{expert_id}.down_proj.input_scale", scale)
                for expert_id, scale in enumerate(tensor.unbind(0))
            )
            continue

        if target == "w13_weight_scale_2":
            if tensor.ndim == 1:
                tensor = tensor[:, None]
            if tensor.ndim != 2 or tensor.shape[1] not in {1, 2}:
                raise ValueError(
                    f"Expected one or two W13 global scales per expert for {name}, "
                    f"got {tuple(tensor.shape)}"
                )
            if tensor.shape[1] == 1:
                batched.extend(
                    (
                        f"{prefix}.experts.{expert_id}.up_proj.weight_scale_2",
                        expert_scale[0],
                    )
                    for expert_id, expert_scale in enumerate(tensor.unbind(0))
                )
                continue
            for expert_id, expert_scale in enumerate(tensor.unbind(0)):
                batched.append(
                    (
                        f"{prefix}.experts.{expert_id}.gate_proj.weight_scale_2",
                        expert_scale[0],
                    )
                )
                batched.append(
                    (
                        f"{prefix}.experts.{expert_id}.up_proj.weight_scale_2",
                        expert_scale[1],
                    )
                )
            continue

        if not target.endswith("weight_scale_2"):
            batched.append((f"{prefix}.experts.0.{target}", tensor))
            continue

        if tensor.ndim == 1:
            expert_scales = tensor
        elif tensor.ndim == 2 and tensor.shape[1] == 1:
            expert_scales = tensor[:, 0]
        else:
            raise ValueError(
                f"Expected one global scale per expert for {name}, got "
                f"shape {tuple(tensor.shape)}"
            )

        batched.extend(
            (f"{prefix}.experts.{expert_id}.{target}", expert_scale)
            for expert_id, expert_scale in enumerate(expert_scales.unbind(0))
        )

    return batched


def _supports_batched_modelopt_moe_load(model: torch.nn.Module) -> bool:
    """Return whether every ModelOpt MoE layer owns the complete expert set."""
    found_modelopt_moe = False
    for module in model.modules():
        quant_method = getattr(module, "quant_method", None)
        quant_method = getattr(quant_method, "old_quant_method", quant_method)
        if quant_method.__class__.__name__ != "ModelOptNvFp4FusedMoE":
            continue
        found_modelopt_moe = True
        expert_map = getattr(
            module,
            "expert_map",
            getattr(module, "_expert_map", None),
        )
        if expert_map is not None:
            return False
        local_num_experts = getattr(module, "local_num_experts", None)
        global_num_experts = getattr(module, "global_num_experts", None)
        if (
            not isinstance(local_num_experts, int)
            or local_num_experts != global_num_experts
        ):
            return False
    return found_modelopt_moe


def _stash_fused_modelopt_moe_scales(
    extension: object,
    weights: list[tuple[str, torch.Tensor]],
) -> None:
    stash = getattr(extension, "_nrl_fused_moe_scales", None)
    if stash is None:
        stash = {}
        extension._nrl_fused_moe_scales = stash
    for name, tensor in weights:
        for suffix, attr in _FUSED_MODELOPT_MOE_DIRECT_SCALE_SUFFIXES.items():
            if name.endswith(suffix):
                prefix = name[: -len(suffix)]
                if (
                    suffix
                    in {
                        ".experts.up_proj_scale_2",
                        ".experts.up_proj_input_scale",
                    }
                    and tensor.ndim == 1
                ):
                    tensor = tensor[:, None]
                stash[(prefix, attr)] = tensor.detach().clone()
                break
        else:
            match = _EXPERT_MODELOPT_SCALE_RE.match(name)
            if match is None:
                continue
            projection = match.group("projection")
            scale = match.group("scale")
            prefix_name = "w2" if projection == "down_proj" else "w13"
            attr = f"{prefix_name}_{scale}"
            # ``up_proj`` is w3 for gated MoE and the sole w1 projection for
            # non-gated MoE.  The last column selects the right slot in both
            # vLLM layouts without coupling this transport code to the model.
            column = {"gate_proj": 0, "up_proj": -1, "down_proj": 0}[projection]
            stash[
                (
                    match.group("prefix"),
                    attr,
                    int(match.group("expert")),
                    column,
                )
            ] = tensor.detach().clone()


def _restore_fused_modelopt_moe_scales(extension: object) -> None:
    stash = getattr(extension, "_nrl_fused_moe_scales", {})
    if not stash:
        return

    model = extension.model_runner.model
    modules = dict(model.named_modules())
    mapper = getattr(model, "hf_to_vllm_mapper", None)
    for key, source in stash.items():
        prefix, attr, *index = key
        mapped_prefix = prefix
        if mapper is not None:
            mapped_prefixes = mapper.apply_list([prefix])
            if len(mapped_prefixes) != 1:
                raise RuntimeError(
                    f"Expected vLLM weight mapper to preserve ModelOpt MoE "
                    f"prefix {prefix}, got {mapped_prefixes}"
                )
            mapped_prefix = mapped_prefixes[0]

        candidates = []
        for name in (mapped_prefix, f"{mapped_prefix}.experts"):
            module = modules.get(name)
            if module is None:
                continue
            quant_method = getattr(
                getattr(module, "quant_method", None),
                "old_quant_method",
                getattr(module, "quant_method", None),
            )
            if quant_method.__class__.__name__ == "ModelOptNvFp4FusedMoE":
                candidates.append(module)
        if len(candidates) != 1:
            raise RuntimeError(
                f"Expected one vLLM ModelOpt MoE layer for {prefix} "
                f"(mapped to {mapped_prefix}), found {len(candidates)}"
            )
        destination = getattr(candidates[0], attr, None)
        if not isinstance(destination, torch.Tensor):
            raise RuntimeError(f"vLLM ModelOpt MoE layer is missing {prefix}.{attr}")
        source = source.to(device=destination.device, dtype=destination.dtype)
        if index:
            expert, column = index
            if attr in {"w13_input_scale", "w13_weight_scale_2"}:
                target = destination.data[expert, column]
            else:
                target = destination.data[expert]
            if source.numel() != 1:
                raise RuntimeError(
                    f"Expected scalar ModelOpt MoE {attr} for expert {expert}, "
                    f"got {tuple(source.shape)}"
                )
            target.copy_(source.reshape_as(target))
        else:
            if tuple(destination.shape) != tuple(source.shape):
                raise RuntimeError(
                    f"ModelOpt MoE {attr} shape mismatch for {prefix}: expected "
                    f"{tuple(destination.shape)}, got {tuple(source.shape)}"
                )
            destination.data.copy_(source)
    extension._nrl_fused_moe_scales = {}


if os.environ.get("VLLM_MODELOPT_REAL_QUANT", "0") == "1":
    from nemo_rl.modelopt.models.generation.vllm_modelopt_patch import (
        apply_modelopt_nvfp4_patches,
    )

    apply_modelopt_nvfp4_patches()


class VllmQuantInternalWorkerExtension(VllmInternalWorkerExtension):
    def maybe_init_zmq(self) -> None:
        """Use a longer timeout only for ModelOpt real-quant refits."""
        if not self._is_real_quant_model():
            super().maybe_init_zmq()
            return
        if not hasattr(self, "zmq_socket"):
            self.zmq_context = zmq.Context()
            self.zmq_socket = self.zmq_context.socket(zmq.REP)
            self.zmq_socket.setsockopt(zmq.SNDTIMEO, MODELOPT_REAL_QUANT_ZMQ_TIMEOUT_MS)
            self.zmq_socket.setsockopt(zmq.RCVTIMEO, MODELOPT_REAL_QUANT_ZMQ_TIMEOUT_MS)
            self.zmq_socket.setsockopt(zmq.LINGER, 0)
            self.zmq_socket.connect(self.get_zmq_address())

    def _is_real_quant_model(self) -> bool:
        return os.environ.get("VLLM_MODELOPT_REAL_QUANT", "0") == "1"

    def prepare_refit_info(self, state_dict_info: dict[str, Any]) -> None:
        super().prepare_refit_info(state_dict_info)
        if not self._is_real_quant_model():
            return
        quant_config = (
            self.model_runner.vllm_config.model_config.hf_config.quantization_config
        )
        self._nrl_w13_num_shards_by_prefix = _w13_num_shards_from_state_dict_info(
            state_dict_info,
            require_input_scales=quant_config.get("quant_mode") == "w4a4_nvfp4",
        )

    @contextmanager
    def _patch_named_parameters_to_include_buffers(self, model):
        """Temporarily patches model.named_parameters() to also yield input_quantizer buffers.

        Weights arrive pre-folded from the Megatron side, so only input_quantizer
        amax buffers need to be loaded. Weight quantizer buffers are skipped.
        """
        original_named_parameters = model.named_parameters
        # input_quantizer buffers we attached a weight_loader to and must
        # clean up on exit; pre-existing loaders (if any) are left untouched.
        patched_quantizer_buffers = []

        def input_amax_loader(param, loaded_weight, *args, **kwargs):
            param.copy_(torch.max(param, loaded_weight))

        def new_named_parameters(self, *args, **kwargs):
            yield from original_named_parameters(*args, **kwargs)
            for name, buf in self.named_buffers(*args, **kwargs):
                if "input_quantizer" not in name:
                    continue
                if not hasattr(buf, "weight_loader"):
                    buf.weight_loader = input_amax_loader
                    patched_quantizer_buffers.append(buf)
                yield name, buf

        model.named_parameters = types.MethodType(new_named_parameters, model)
        try:
            yield
        finally:
            model.named_parameters = original_named_parameters
            for buf in patched_quantizer_buffers:
                del buf.weight_loader

    def _load_weights(self, weights):
        """Load pre-folded weights and input_quantizer amax buffers.

        Weights arrive already folded from the Megatron side (weight_quantizer
        applied during export), so no fold_weight step is needed here.
        """
        if self._is_real_quant_model():
            quant_config = (
                self.model_runner.vllm_config.model_config.hf_config.quantization_config
            )
            ignore_patterns = quant_config.get("ignore", []) or []
            filtered = []
            for name, weight in weights:
                suffix = name.rsplit(".", 1)[-1]
                ignored = matches_quant_ignore_pattern(name, ignore_patterns)
                if ignored and suffix in {
                    "weight_scale",
                    "weight_scale_2",
                    "input_scale",
                }:
                    continue

                filtered.append((name, weight))
            _stash_fused_modelopt_moe_scales(self, filtered)
            if any(_is_fused_modelopt_moe_weight(name) for name, _ in filtered):
                supports_batched_moe = getattr(
                    self,
                    "_nrl_supports_batched_modelopt_moe_load",
                    None,
                )
                if supports_batched_moe is None:
                    supports_batched_moe = _supports_batched_modelopt_moe_load(
                        self.model_runner.model
                    )
                    self._nrl_supports_batched_modelopt_moe_load = supports_batched_moe
                if not supports_batched_moe:
                    raise RuntimeError(
                        "Fused ModelOpt MoE refits require all experts local; "
                        "vLLM expert parallelism is unsupported"
                    )
                weights = _batch_fused_modelopt_moe_weights(
                    filtered,
                    w13_num_shards_by_prefix=self._nrl_w13_num_shards_by_prefix,
                )
            else:
                weights = filtered
            if not weights:
                return None
            return super()._load_weights(weights)

        with ExitStack() as contexts:
            for _, child in self.model_runner.model.named_children():
                contexts.enter_context(
                    self._patch_named_parameters_to_include_buffers(child)
                )
            return super()._load_weights(weights)

    def update_weights_via_ipc_zmq(self) -> bool:
        """Receive and update weights through CUDA IPC."""
        if not self._is_real_quant_model():
            return super().update_weights_via_ipc_zmq()

        from nemo_rl.modelopt.models.generation.vllm_modelopt_patch import (
            modelopt_process_weights_after_loading,
            prepare_modelopt_for_weight_reload,
        )

        self._nrl_fused_moe_scales = {}
        prepare_modelopt_for_weight_reload(self.model_runner.model, device=self.device)
        self.maybe_init_zmq()
        expected_keys = set(self.state_dict_info)
        loaded_keys: set[str] = set()
        key_errors: list[str] = []
        buffer = None
        weight = None
        weights = None
        while True:
            payload = self.zmq_socket.recv_pyobj()

            if payload == IPCProtocol.COMPLETE:
                missing_keys = expected_keys - loaded_keys
                if missing_keys or key_errors:
                    details = list(key_errors)
                    if missing_keys:
                        details.append(
                            _format_refit_key_error("missing keys", missing_keys)
                        )
                    message = "ModelOpt real-quant refit rejected: " + "; ".join(
                        details
                    )
                    self._nrl_fused_moe_scales = {}
                    self.zmq_socket.send(IPCProtocol.ACK.value.encode())
                    raise RuntimeError(message)
                try:
                    _restore_fused_modelopt_moe_scales(self)
                    modelopt_process_weights_after_loading(self.model_runner.model)
                    torch.cuda.synchronize()
                except Exception as error:
                    self.zmq_socket.send(IPCProtocol.ACK.value.encode())
                    raise RuntimeError(
                        "ModelOpt real-quant refit post-processing failed"
                    ) from error
                self.zmq_socket.send(IPCProtocol.ACK.value.encode())
                break

            ipc_handle, list_keys, used_bytes = payload
            batch_keys: set[str] = set()
            duplicate_keys: set[str] = set()
            for key in list_keys:
                if key in batch_keys:
                    duplicate_keys.add(key)
                batch_keys.add(key)
            duplicate_keys.update(loaded_keys & batch_keys)
            unexpected_keys = batch_keys - expected_keys
            if duplicate_keys:
                key_errors.append(
                    _format_refit_key_error("duplicate keys", duplicate_keys)
                )
            if unexpected_keys:
                key_errors.append(
                    _format_refit_key_error("unexpected keys", unexpected_keys)
                )
            if key_errors:
                self.zmq_socket.send(IPCProtocol.ACK.value.encode())
                continue

            try:
                buffer = rebuild_cuda_tensor_from_ipc(ipc_handle, self.device.index)

                weights = []
                offset = 0
                for key in list_keys:
                    shape, dtype = self.state_dict_info[key]
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
                    "Offset is not equal to used bytes, usually indicate inaccurate "
                    "info like keys or cached dtype in state_dict_info"
                )

                self._load_weights(weights)
                torch.cuda.synchronize()
                loaded_keys.update(batch_keys)
            except Exception as error:
                message = f"{type(error).__name__}: {error}"
                if len(message) > 512:
                    message = message[:512] + " ..."
                key_errors.append(f"weight load failed: {message}")
            finally:
                # Match the base vLLM receiver: the loop variable is the final IPC
                # tensor view and must be dropped together with the list and base
                # buffer before ACK permits the sender to reuse or release storage.
                del weight, weights, buffer
                weight = None
                weights = None
                buffer = None
            self.zmq_socket.send(IPCProtocol.ACK.value.encode())

        self._maybe_process_fp8_kv_cache()
        gc.collect()
        torch.cuda.empty_cache()
        return True

    def update_weights_from_collective(self) -> bool:
        """Receive and update weights through collective communication."""
        if not self._is_real_quant_model():
            return super().update_weights_from_collective()

        from nemo_rl.modelopt.models.generation.vllm_modelopt_patch import (
            modelopt_process_weights_after_loading,
            prepare_modelopt_for_weight_reload,
        )

        assert self.state_dict_info is not None, (
            "state_dict_info is not prepared. "
            "Please call prepare_refit_info when initializing the worker."
        )

        self._nrl_fused_moe_scales = {}
        prepare_modelopt_for_weight_reload(self.model_runner.model, device=self.device)
        try:
            # Mirror the IPC refit ordering instead of reusing
            # super().update_weights_from_collective(): consume the broadcast
            # without vLLM's post-load conversion, restore the per-expert scales
            # stashed during _load_weights, and only then run the ModelOpt
            # conversion. The base override converts first, which rewrites the
            # scales out of their HF layout and makes the restore fail.
            packed_broadcast_consumer(
                iterator=iter(self.state_dict_info.items()),
                group=self.model_update_group,
                src=0,
                post_unpack_func=self._load_weights,
            )
            _restore_fused_modelopt_moe_scales(self)
            modelopt_process_weights_after_loading(self.model_runner.model)
            self._maybe_process_fp8_kv_cache()
        except Exception as error:
            self._nrl_fused_moe_scales = {}
            raise RuntimeError("ModelOpt real-quant collective refit failed") from error

        gc.collect()
        torch.cuda.empty_cache()
        return True

    def get_weight_snapshot(self, name: str) -> torch.Tensor:
        """Return a CPU copy of a named parameter for before/after comparison."""
        model = self.model_runner.model
        for n, p in model.named_parameters():
            if n == name:
                return p.detach().cpu().clone()
        raise KeyError(f"Parameter '{name}' not found in model")

    def get_quantizer_stats(self) -> dict:
        """Return summary statistics for all TensorQuantizer modules.

        Matches the interface of MegatronQuantPolicyWorker.get_quantizer_stats().
        """
        total = 0
        enabled = 0
        with_amax = 0
        positive_amax = 0
        model = self.model_runner.model
        for _, module in model.named_modules():
            if isinstance(module, TensorQuantizer):
                total += 1
                if module.is_enabled:
                    enabled += 1
                    if hasattr(module, "amax") and module.amax is not None:
                        with_amax += 1
                        if (module.amax > 0).all():
                            positive_amax += 1
        return {
            "total": total,
            "enabled": enabled,
            "with_amax": with_amax,
            "positive_amax": positive_amax,
        }
