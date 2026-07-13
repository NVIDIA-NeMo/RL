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
from collections.abc import Mapping
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
        _types: Any,
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

    def sparse_delta_source_plans(
        self, metadata: list[dict[str, Any]]
    ) -> dict[str, sparse_codec.SparseSourcePlan]:
        """Describe which canonical source views this worker consumes."""
        self._compile_plans(metadata)
        return {
            name: sparse_codec.SparseSourcePlan(
                routes=tuple(
                    sparse_codec.SparseSourceRoute(
                        copy.source_offset,
                        copy.source_strides,
                        copy.shape,
                        copy.linear,
                    )
                    for copy in plan.copies
                ),
                identity=plan.identity,
            )
            for name, plan in (
                (str(item["name"]), self._plan_cache[str(item["name"])])
                for item in metadata
            )
        }

    def prewarm(
        self, state_dict_info: Mapping[str, tuple[tuple[int, ...], torch.dtype]]
    ) -> None:
        self._compile_plans(
            [
                {
                    "name": name,
                    "shape": shape,
                    "dtype": str(dtype).removeprefix("torch."),
                }
                for name, (shape, dtype) in state_dict_info.items()
            ]
        )

    @staticmethod
    def _map_copy(
        locations: torch.Tensor,
        values: torch.Tensor,
        copy: _SparseDeltaCopyPlan,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mapped, keep = sparse_codec.map_sparse_locations(
            locations,
            copy.source_offset,
            copy.source_strides,
            copy.shape,
            copy.linear,
            copy.target_offset,
            copy.target_strides,
        )
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

    def _apply_decoded_item(
        self,
        item: dict[str, Any],
        plan: _SparseDeltaTargetPlan,
        locations: torch.Tensor,
        values: torch.Tensor,
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
                target.narrow(0, int(item["range_start"]), values.numel()).bitwise_xor_(
                    values
                )
            else:
                target, replacement = self._overwrite_target_values(
                    first_target, values, source_dtype, log_transform=False
                )
                target.narrow(0, int(item["range_start"]), values.numel()).copy_(
                    replacement
                )
            return

        locations = locations.to(
            device=first_target.device, dtype=torch.int64, non_blocking=True
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

    def _apply_decoded_sparse_weight_deltas(
        self, decoded: list[sparse_codec.DecodedSparseItem]
    ) -> None:
        with torch.no_grad():
            for item, locations, values in decoded:
                self._apply_decoded_item(
                    item,
                    self._plan_cache[str(item["name"])],
                    locations,
                    values,
                )

    @wrap_with_nvtx_name(
        "vllm_internal_worker_extension/update_weights_from_decoded_sparse_payload"
    )
    def update_weights_from_decoded_sparse_payload(
        self, *serialized_payloads: bytes
    ) -> dict[str, Any]:
        return self._load_decoded_sparse_payloads(
            tuple(io.BytesIO(payload) for payload in serialized_payloads)
        )

    def _load_decoded_sparse_payloads(
        self, sources: tuple[str | io.BytesIO, ...]
    ) -> dict[str, Any]:
        started = time.perf_counter()
        deserialize_s = 0.0
        payloads: list[sparse_codec.DecodedSparsePayload] = []
        for source in sources:
            item_started = time.perf_counter()
            payloads.append(
                cast(
                    sparse_codec.DecodedSparsePayload,
                    torch.load(
                        source,
                        map_location="cpu",
                        weights_only=True,
                        mmap=isinstance(source, str),
                    ),
                )
            )
            deserialize_s += time.perf_counter() - item_started

        item_started = time.perf_counter()
        metadata = [item for _, _, items in payloads for item in items]
        source_plans = self.sparse_delta_source_plans(metadata)
        plan_s = time.perf_counter() - item_started
        partition_s = sparse_apply_s = 0.0
        for payload in payloads:
            item_started = time.perf_counter()
            selected = sparse_codec.partition_decoded_sparse_entries(
                sparse_codec.iter_decoded_sparse_payload(payload), source_plans
            )
            partition_s += time.perf_counter() - item_started
            item_started = time.perf_counter()
            self._apply_decoded_sparse_weight_deltas(selected)
            sparse_apply_s += time.perf_counter() - item_started
        return {
            "ok": True,
            "receiver_deserialize_s": deserialize_s,
            "receiver_plan_s": plan_s,
            "receiver_partition_s": partition_s,
            "receiver_sparse_apply_s": sparse_apply_s,
            "receiver_total_s": time.perf_counter() - started,
        }

    def update_weights_from_decoded_sparse_payload_files(
        self, *payload_paths: str
    ) -> dict[str, Any]:
        return self._load_decoded_sparse_payloads(payload_paths)

    def synchronize_device(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize(self._cuda_device_index)

    def finish_sparse_delta_refit(self) -> dict[str, Any]:
        """Synchronize and compare bounded producer samples with applied weights."""
        self.synchronize_device()
        verification, self._verification = self._verification, []
        candidates, self._verification_candidates = self._verification_candidates, 0
        samples = 0
        stats = [0.0] * 4
        if verification:
            with torch.no_grad():
                differences = []
                exact_mismatches = []
                mismatches = []
                for target, locations, expected_bits in verification:
                    integer_dtype = sparse_codec.integer_dtype_for_element_size(
                        target.element_size()
                    )
                    actual_bits = (
                        target.data.view(integer_dtype)
                        .view(-1)
                        .index_select(0, locations)
                    )
                    bit_mismatches = actual_bits.ne(expected_bits)
                    actual = target.data.view(-1).index_select(0, locations).float()
                    expected = expected_bits.view(target.dtype).float()
                    difference = torch.where(
                        bit_mismatches,
                        (actual - expected).abs(),
                        torch.zeros_like(actual),
                    )
                    differences.append(torch.nan_to_num(difference, nan=float("inf")))
                    exact_mismatches.append(bit_mismatches)
                    mismatches.append(
                        bit_mismatches
                        & ~torch.isclose(actual, expected, rtol=1e-6, atol=1e-8)
                    )
                    samples += actual.numel()
                difference = torch.cat(differences)
                stats = (
                    torch.stack(
                        (
                            difference.sum(),
                            difference.max(),
                            torch.cat(exact_mismatches).sum().float(),
                            torch.cat(mismatches).sum().float(),
                        )
                    )
                    .cpu()
                    .tolist()
                )
        return {
            "ok": True,
            "verification_candidates": candidates,
            "verification_samples": samples,
            "verification_exact_mismatches": int(stats[2]),
            "verification_mismatches": int(stats[3]),
            "verification_abs_sum": float(stats[0]),
            "verification_max_abs": float(stats[1]),
        }
