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

"""Direct sparse-delta placement and application for vLLM workers."""

import io
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from math import prod
from typing import Any, cast

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from nemo_rl.utils import weight_transfer_sparse_codec as sparse_codec
from nemo_rl.utils.nsys import wrap_with_nvtx_name

_EXPERT_WEIGHT_RE = re.compile(r"\.experts\.\d+\.(?:gate|up|down)_proj\.weight$")


def _storage_key(tensor: torch.Tensor) -> int:
    return tensor.untyped_storage()._cdata


@dataclass(frozen=True)
class _SparseDeltaCopyPlan:
    target: torch.Tensor
    source_offset: int
    source_strides: tuple[int, ...]
    shape: tuple[int, ...]
    target_offset: int
    target_strides: tuple[int, ...]
    linear: bool


@dataclass(frozen=True)
class _SparseDeltaTargetPlan:
    copies: tuple[_SparseDeltaCopyPlan, ...] = ()
    log_delta_transform: bool = False
    identity: bool = False


class _SparseLoadTracer(TorchDispatchMode):
    """Capture the views copied by vLLM's native weight loaders."""

    def __init__(
        self,
        targets: list[torch.Tensor],
        sources: dict[str, torch.Tensor],
    ) -> None:
        super().__init__()
        self.copies: dict[str, list[_SparseDeltaCopyPlan]] = defaultdict(list)
        self.postprocessed: set[str] = set()
        self._sources = {_storage_key(tensor): name for name, tensor in sources.items()}
        self._targets: dict[int, list[torch.Tensor]] = defaultdict(list)
        for target in targets:
            if not target.is_contiguous():
                raise RuntimeError("Sparse delta targets must be contiguous.")
            self._targets[_storage_key(target)].append(target)
        self._last_source: dict[int, str] = {}

    def _target_for(self, view: torch.Tensor) -> torch.Tensor | None:
        candidates = self._targets.get(_storage_key(view), ())
        view_start = view.storage_offset()
        view_end = view_start + sum(
            (size - 1) * stride for size, stride in zip(view.shape, view.stride())
        )
        for target in candidates:
            start = target.storage_offset()
            if start <= view_start and view_end < start + target.numel():
                return target
        return None

    def __torch_dispatch__(
        self,
        func: Any,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        if func is not torch.ops.aten.copy_.default:
            return func(*args, **(kwargs or {}))

        destination, source = cast(tuple[torch.Tensor, torch.Tensor], args[:2])
        target = self._target_for(destination)
        if target is None:
            raise RuntimeError("vLLM loader copied outside a model parameter.")

        source_name = self._sources.get(_storage_key(source))
        target_key = id(target)
        if source_name is None:
            source_name = self._last_source.get(target_key)
            if source_name is None:
                raise RuntimeError("vLLM loader materialized an unsupported transform.")
            self.postprocessed.add(source_name)
            return destination

        if destination.shape != source.shape:
            raise RuntimeError("vLLM loader used an expanding copy.")
        shape = tuple(source.shape)
        source_strides = tuple(source.stride())
        target_strides = tuple(destination.stride())
        if any(size > 1 and stride <= 0 for size, stride in zip(shape, source_strides)):
            raise RuntimeError("vLLM loader used an unsupported source view.")
        contiguous = torch.empty(shape, device="meta").stride()
        self.copies[source_name].append(
            _SparseDeltaCopyPlan(
                target,
                int(source.storage_offset()),
                source_strides,
                shape,
                int(destination.storage_offset() - target.storage_offset()),
                target_strides,
                source_strides == target_strides == contiguous,
            )
        )
        self._last_source[target_key] = source_name
        return destination


class VllmSparseDeltaApplier:
    """Apply sparse HF deltas through plans derived from native vLLM loaders."""

    def __init__(self, model_runner: Any, device: torch.device) -> None:
        self.model_runner = model_runner
        self._cuda_device_index = device.index
        self._plan_cache: dict[str, _SparseDeltaTargetPlan] = {}
        self._verification: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self._verification_candidates = 0

    def _compile_plans(self, metadata: list[dict[str, Any]]) -> None:
        missing = {
            str(item["name"]): item
            for item in metadata
            if item["name"] not in self._plan_cache
        }
        if not missing:
            return

        model = self.model_runner.model
        targets = list(model.parameters()) + list(model.buffers())
        sources = {
            name: torch.empty(
                tuple(item["shape"]),
                dtype=sparse_codec.dtype_from_name(str(item["dtype"])),
                device="meta",
            )
            for name, item in missing.items()
        }
        tracer = _SparseLoadTracer(targets, sources)
        with torch.no_grad(), tracer:
            model.load_weights((name, sources[name]) for name in missing)

        for name, item in missing.items():
            source_shape = tuple(item["shape"])
            source_strides = tuple(sources[name].stride())
            copies = tuple(tracer.copies.get(name, ()))
            transformed = name in tracer.postprocessed
            log_transform = (
                transformed and ".mixer." in name and name.endswith((".A", ".A_log"))
            )
            if transformed and not log_transform:
                raise RuntimeError(
                    f"vLLM loader for {name!r} transforms weights and cannot apply deltas."
                )
            if not copies and not (
                name.startswith(("mtp.", "draft.")) or _EXPERT_WEIGHT_RE.search(name)
            ):
                raise RuntimeError(f"vLLM loader did not place {name!r}.")
            identity = (
                len(copies) == 1
                and not log_transform
                and copies[0].source_offset == 0
                and copies[0].shape == source_shape
                and copies[0].source_strides == source_strides
                and copies[0].target_offset == 0
                and copies[0].target_strides == source_strides
                and copies[0].target.numel() == prod(source_shape)
            )
            self._plan_cache[name] = _SparseDeltaTargetPlan(
                copies, log_transform, identity
            )

    @staticmethod
    def _map_copy(
        locations: torch.Tensor,
        values: torch.Tensor,
        copy: _SparseDeltaCopyPlan,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if copy.linear:
            end = copy.source_offset + prod(copy.shape)
            keep = (locations >= copy.source_offset) & (locations < end)
            return (
                locations[keep] + copy.target_offset - copy.source_offset,
                values[keep],
            )
        mapped = torch.full_like(locations, copy.target_offset)
        reconstructed = torch.full_like(locations, copy.source_offset)
        relative = locations - copy.source_offset
        for size, source_stride, target_stride in zip(
            copy.shape, copy.source_strides, copy.target_strides, strict=True
        ):
            coordinate = (
                torch.div(relative, source_stride, rounding_mode="floor").remainder(
                    size
                )
                if size > 1
                else torch.zeros_like(locations)
            )
            reconstructed.add_(coordinate * source_stride)
            mapped.add_(coordinate * target_stride)
        keep = reconstructed == locations
        return mapped[keep], values[keep]

    def _record_verification(
        self,
        item: dict[str, Any],
        plan: _SparseDeltaTargetPlan,
        operation: sparse_codec.SparseOperation,
        source_dtype: torch.dtype,
    ) -> None:
        sample_locations = item.get("verification_locations", [])
        self._verification_candidates += len(sample_locations)
        if not sample_locations or not plan.copies:
            return
        target = plan.copies[0].target
        locations = torch.tensor(sample_locations, device=target.device)
        value_dtype = sparse_codec.integer_dtype_for_element_size(source_dtype.itemsize)
        values = torch.tensor(
            item["verification_values"], device=target.device, dtype=value_dtype
        )
        for copy in plan.copies:
            mapped, selected = self._map_copy(locations, values, copy)
            if not mapped.numel():
                continue
            if operation == "xor":
                target_bits = self._integer_flat(copy.target)
                expected = target_bits.index_select(0, mapped).bitwise_xor(selected)
            else:
                _, replacement = self._overwrite_target_values(
                    copy.target,
                    selected,
                    source_dtype,
                    log_transform=plan.log_delta_transform,
                )
                expected = replacement.contiguous().view(
                    sparse_codec.integer_dtype_for_element_size(
                        copy.target.element_size()
                    )
                )
            self._verification.append((copy.target, mapped, expected))

    @staticmethod
    def _integer_flat(target: torch.Tensor) -> torch.Tensor:
        dtype = sparse_codec.integer_dtype_for_element_size(target.element_size())
        return target.data.view(dtype).view(-1)

    @staticmethod
    def _xor_target_mappings_overlap(plan: _SparseDeltaTargetPlan) -> bool:
        spans: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for copy in plan.copies:
            origin = int(copy.target.storage_offset()) + copy.target_offset
            extents = [
                (size - 1) * stride
                for size, stride in zip(copy.shape, copy.target_strides, strict=True)
            ]
            start = origin + sum(min(0, extent) for extent in extents)
            end = origin + sum(max(0, extent) for extent in extents)
            target_spans = spans[_storage_key(copy.target)]
            if any(
                start <= other_end and other_start <= end
                for other_start, other_end in target_spans
            ):
                return True
            target_spans.append((start, end))
        return False

    @classmethod
    def _overwrite_target_values(
        cls,
        target: torch.Tensor,
        values: torch.Tensor,
        source_dtype: torch.dtype,
        *,
        log_transform: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not log_transform and target.dtype == source_dtype:
            return cls._integer_flat(target), values
        source_values = values.contiguous().view(source_dtype)
        replacement = (
            -source_values.float().exp().to(target.dtype)
            if log_transform
            else source_values.to(target.dtype)
        )
        return target.data.view(-1), replacement

    def _apply_item(
        self,
        item: dict[str, Any],
        plan: _SparseDeltaTargetPlan,
        raw_locations: torch.Tensor,
        raw_value_groups: tuple[torch.Tensor, ...],
    ) -> None:
        operation = sparse_codec.sparse_operation(item["operation"])
        source_dtype = sparse_codec.dtype_from_name(str(item["dtype"]))
        if not plan.copies:
            self._record_verification(item, plan, operation, source_dtype)
            return
        first_target = plan.copies[0].target
        if operation == "xor" and plan.log_delta_transform:
            raise RuntimeError(f"XOR cannot apply transformed weight {item['name']!r}.")
        if operation == "xor" and any(
            copy.target.dtype != source_dtype for copy in plan.copies
        ):
            raise RuntimeError(
                f"XOR source and target dtypes differ for {item['name']!r}."
            )
        if operation == "xor" and self._xor_target_mappings_overlap(plan):
            raise RuntimeError(f"XOR target mappings overlap for {item['name']!r}.")
        value_start, value_end = int(item["value_start"]), int(item["value_end"])
        values = raw_value_groups[int(item["value_group"])][value_start:value_end]
        expected_dtype = sparse_codec.integer_dtype_for_element_size(
            source_dtype.itemsize
        )
        if values.dtype != expected_dtype:
            raise RuntimeError(
                f"Sparse values have the wrong dtype for {item['name']!r}."
            )
        values = values.to(device=first_target.device, non_blocking=True)
        self._record_verification(item, plan, operation, source_dtype)
        if plan.identity and item["index_encoding"] == "range":
            if operation == "xor":
                target = self._integer_flat(first_target)
                target.narrow(
                    0, int(item["range_start"]), value_end - value_start
                ).bitwise_xor_(values)
            else:
                target, replacement = self._overwrite_target_values(
                    first_target, values, source_dtype, log_transform=False
                )
                target.narrow(
                    0, int(item["range_start"]), value_end - value_start
                ).copy_(replacement)
            return

        locations = sparse_codec.sparse_locations_for_item(
            item, raw_locations, device=first_target.device
        )
        if plan.identity:
            if operation == "xor":
                target = self._integer_flat(first_target)
                current = target.index_select(0, locations)
                target.index_copy_(0, locations, current.bitwise_xor(values))
            else:
                target, replacement = self._overwrite_target_values(
                    first_target, values, source_dtype, log_transform=False
                )
                target.index_copy_(0, locations, replacement)
            return
        for copy in plan.copies:
            mapped, selected = self._map_copy(locations, values, copy)
            if not mapped.numel():
                continue
            if operation == "xor":
                target = self._integer_flat(copy.target)
                current = target.index_select(0, mapped)
                target.index_copy_(0, mapped, current.bitwise_xor(selected))
            else:
                target, replacement = self._overwrite_target_values(
                    copy.target,
                    selected,
                    source_dtype,
                    log_transform=plan.log_delta_transform,
                )
                target.index_copy_(0, mapped, replacement)

    def _apply_sparse_weight_deltas(
        self,
        payload_tensors: tuple[torch.Tensor, tuple[torch.Tensor, ...]],
        metadata: list[dict[str, Any]],
    ) -> None:
        self._compile_plans(metadata)
        raw_locations, raw_value_groups = payload_tensors
        with torch.no_grad():
            for item in metadata:
                self._apply_item(
                    item,
                    self._plan_cache[str(item["name"])],
                    raw_locations,
                    raw_value_groups,
                )

    @wrap_with_nvtx_name(
        "vllm_internal_worker_extension/update_weights_from_serialized_sparse_payload"
    )
    def update_weights_from_serialized_sparse_payload(
        self, *serialized_payloads: bytes
    ) -> dict[str, Any]:
        return self._load_and_apply_sparse_payloads(
            tuple(io.BytesIO(payload) for payload in serialized_payloads)
        )

    def _load_and_apply_sparse_payloads(
        self, sources: tuple[str | io.BytesIO, ...]
    ) -> dict[str, Any]:
        started = time.perf_counter()
        deserialize_s = sparse_apply_s = 0.0
        payloads = []
        for source in sources:
            item_started = time.perf_counter()
            payloads.append(
                cast(
                    sparse_codec.TensorPayload,
                    torch.load(source, map_location="cpu", weights_only=True),
                )
            )
            deserialize_s += time.perf_counter() - item_started

        item_started = time.perf_counter()
        self._compile_plans([item for _, _, metadata in payloads for item in metadata])
        plan_s = time.perf_counter() - item_started
        for locations, values, metadata in payloads:
            item_started = time.perf_counter()
            self._apply_sparse_weight_deltas((locations, values), metadata)
            sparse_apply_s += time.perf_counter() - item_started
        return {
            "ok": True,
            "receiver_deserialize_s": deserialize_s,
            "receiver_plan_s": plan_s,
            "receiver_sparse_apply_s": sparse_apply_s,
            "receiver_total_s": time.perf_counter() - started,
        }

    @wrap_with_nvtx_name(
        "vllm_internal_worker_extension/update_weights_from_sparse_payload_files"
    )
    def update_weights_from_sparse_payload_files(
        self, *payload_paths: str
    ) -> dict[str, Any]:
        return self._load_and_apply_sparse_payloads(payload_paths)

    def synchronize_device(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize(self._cuda_device_index)

    def finish_sparse_delta_refit(self) -> dict[str, Any]:
        """Synchronize and compare bounded producer samples with applied weights."""
        self.synchronize_device()
        verification, self._verification = self._verification, []
        candidates, self._verification_candidates = self._verification_candidates, 0
        if not verification:
            return {
                "ok": True,
                "verification_candidates": candidates,
                "verification_samples": 0,
                "verification_exact_mismatches": 0,
                "verification_mismatches": 0,
                "verification_abs_sum": 0.0,
                "verification_max_abs": 0.0,
            }

        with torch.no_grad():
            differences = []
            exact_mismatches = []
            mismatches = []
            samples = 0
            for target, locations, expected_bits in verification:
                integer_dtype = sparse_codec.integer_dtype_for_element_size(
                    target.element_size()
                )
                actual_bits = (
                    target.data.view(integer_dtype).view(-1).index_select(0, locations)
                )
                bit_mismatches = actual_bits.ne(expected_bits)
                actual = target.data.view(-1).index_select(0, locations).float()
                expected = expected_bits.view(target.dtype).float()
                difference = torch.where(
                    bit_mismatches, (actual - expected).abs(), torch.zeros_like(actual)
                )
                differences.append(torch.nan_to_num(difference, nan=float("inf")))
                exact_mismatches.append(bit_mismatches)
                mismatches.append(
                    bit_mismatches
                    & ~torch.isclose(actual, expected, rtol=1e-6, atol=1e-8)
                )
                samples += actual.numel()
            difference = torch.cat(differences)
            stats = torch.stack(
                (
                    difference.sum(),
                    difference.max(),
                    torch.cat(exact_mismatches).sum().float(),
                    torch.cat(mismatches).sum().float(),
                )
            ).cpu()
        return {
            "ok": True,
            "verification_candidates": candidates,
            "verification_samples": samples,
            "verification_exact_mismatches": int(stats[2]),
            "verification_mismatches": int(stats[3]),
            "verification_abs_sum": float(stats[0]),
            "verification_max_abs": float(stats[1]),
        }
