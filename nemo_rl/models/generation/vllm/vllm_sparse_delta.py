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

"""Apply canonical sparse updates through vLLM's native weight loaders."""

import io
import time
from collections.abc import Iterable, Iterator, Mapping
from math import prod
from typing import Any, cast

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from nemo_rl.utils import weight_transfer_sparse_codec as sparse_codec
from nemo_rl.utils.nsys import wrap_with_nvtx_name

_TensorViewKey = tuple[int, int, tuple[int, ...], tuple[int, ...]]
_LoaderWeight = tuple[str, torch.Tensor, sparse_codec.SparseOperation, int, int | None]


def _storage_key(tensor: torch.Tensor) -> int:
    return tensor.untyped_storage()._cdata


def _view_key(tensor: torch.Tensor) -> _TensorViewKey:
    return (
        _storage_key(tensor),
        int(tensor.storage_offset()),
        tuple(map(int, tensor.shape)),
        tuple(map(int, tensor.stride())),
    )


class _SparseWeightLoadMode(TorchDispatchMode):
    """Turn native loader copies into sparse XOR or overwrite."""

    def __init__(
        self,
        targets: set[int],
        verification: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> None:
        super().__init__()
        self._targets = targets
        self._verification = verification
        self._source_storage = 0
        self._operation: sparse_codec.SparseOperation = "overwrite"
        self._sample_limit = 0
        self._exact_sentinel: int | None = None
        self._active_masks: dict[_TensorViewKey, torch.Tensor] = {}
        self._verification_masks: dict[
            _TensorViewKey, tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._xor_spans: dict[int, list[tuple[int, int]]] = {}
        self.copies = 0

    def start(
        self,
        source: torch.Tensor,
        operation: sparse_codec.SparseOperation,
        sample_limit: int,
        exact_sentinel: int | None,
    ) -> None:
        self._source_storage = _storage_key(source)
        self._operation = operation
        self._sample_limit = sample_limit
        self._exact_sentinel = exact_sentinel
        self._active_masks.clear()
        self._verification_masks.clear()
        self._xor_spans.clear()
        self.copies = 0

    def _remember_changed(
        self, destination: torch.Tensor, changed: torch.Tensor
    ) -> None:
        if self._sample_limit <= 0:
            return
        view_key = _view_key(destination)
        previous = self._verification_masks.get(view_key)
        if previous is not None:
            changed = previous[1] | changed
        self._verification_masks[view_key] = (destination, changed)

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
        if _storage_key(destination) not in self._targets:
            return func(*args, **(kwargs or {}))

        self.copies += 1
        if self._operation == "overwrite":
            if not source.dtype.is_floating_point:
                raise RuntimeError("Sparse overwrite requires a floating-point loader.")
            source = source.expand_as(destination)
            view_key = _view_key(destination)
            if self._exact_sentinel is not None:
                if (
                    _storage_key(source) != self._source_storage
                    or source.dtype != destination.dtype
                ):
                    raise RuntimeError(
                        "Exact FP8 overwrite cannot pass through a transforming loader."
                    )
                source_bits = sparse_codec.integer_view(source)
                changed = source_bits.ne(self._exact_sentinel)
                destination_bits = sparse_codec.integer_view(destination)
                destination_bits.masked_scatter_(
                    changed, source_bits.masked_select(changed)
                )
                self._remember_changed(destination, changed)
                return destination
            changed = self._active_masks.get(view_key)
            if changed is None or _storage_key(source) == self._source_storage:
                changed = ~torch.isnan(source)
                self._active_masks[view_key] = changed
            destination.masked_scatter_(
                changed, source.masked_select(changed).to(destination.dtype)
            )
            self._remember_changed(destination, changed)
            return destination

        if _storage_key(source) != self._source_storage:
            raise RuntimeError(
                "XOR cannot pass through a native loader that transforms its input."
            )
        if source.dtype != destination.dtype:
            raise RuntimeError("XOR source and target dtypes must match.")
        source = source.expand_as(destination)
        origin = int(destination.storage_offset())
        extents = [
            (int(size) - 1) * int(stride)
            for size, stride in zip(
                destination.shape, destination.stride(), strict=True
            )
        ]
        span = (
            origin + sum(min(0, extent) for extent in extents),
            origin + sum(max(0, extent) for extent in extents),
        )
        spans = self._xor_spans.setdefault(_storage_key(destination), [])
        if any(span[0] <= other[1] and other[0] <= span[1] for other in spans):
            raise RuntimeError("XOR native loader produced overlapping target copies.")
        spans.append(span)
        destination_bits = sparse_codec.integer_view(destination)
        source_bits = sparse_codec.integer_view(source)
        changed = source_bits.ne(0)
        values = destination_bits.masked_select(changed).bitwise_xor(
            source_bits.masked_select(changed)
        )
        destination_bits.masked_scatter_(changed, values)
        self._remember_changed(destination, changed)
        return destination

    @torch.no_grad()
    def finish(self) -> None:
        """Record bounded target samples after the loader finishes transforms."""
        for target, changed in self._verification_masks.values():
            if self._sample_limit <= 0:
                break
            if not target.is_contiguous():
                continue
            locations = changed.reshape(-1).nonzero().reshape(-1)[: self._sample_limit]
            if locations.numel():
                target_bits = sparse_codec.integer_view(target).reshape(-1)
                self._verification.append(
                    (target, locations, target_bits.index_select(0, locations).clone())
                )
                self._sample_limit -= locations.numel()
        self._verification_masks.clear()


class VllmSparseDeltaApplier:
    """Own one dense GPU scratch buffer and delegate all placement to vLLM."""

    def __init__(self, model_runner: Any, device: torch.device) -> None:
        self.model_runner = model_runner
        self._cuda_device_index = device.index
        model = model_runner.model
        self._target_storages = {
            _storage_key(tensor) for tensor in (*model.parameters(), *model.buffers())
        }
        self._scratch = torch.empty(0, dtype=torch.uint8, device=device)
        self._skipped_names: set[str] = set()
        self._verification: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def prewarm(
        self, state_dict_info: Mapping[str, tuple[tuple[int, ...], torch.dtype]]
    ) -> None:
        """Reserve a reusable buffer for the largest canonical source tensor."""
        required = max(
            (prod(shape) * dtype.itemsize for shape, dtype in state_dict_info.values()),
            default=0,
        )
        if required > self._scratch.numel():
            self._scratch = torch.empty(
                required, dtype=torch.uint8, device=self._scratch.device
            )

    def discover_native_skips(
        self, state_dict_info: Mapping[str, tuple[tuple[int, ...], torch.dtype]]
    ) -> None:
        """Cache weights that the native loader explicitly skips on this rank."""
        pending = [
            (name, shape, dtype)
            for name, (shape, dtype) in state_dict_info.items()
            if dtype.is_floating_point
        ]
        if not pending:
            return

        def weights() -> Iterator[_LoaderWeight]:
            for name, shape, dtype in pending:
                source = self._source_tensor(
                    {"shape": shape, "dtype": str(dtype).removeprefix("torch.")}
                )
                source.fill_(float("nan"))
                exact_sentinel = (
                    int(sparse_codec.integer_view(source).reshape(-1)[0].item())
                    if source.element_size() == 1
                    else None
                )
                yield name, source, "overwrite", 0, exact_sentinel

        loaded, observations = self._load_weights(weights(), [])
        if len(observations) != len(pending):
            raise RuntimeError(
                "Native loader did not consume all sparse weight metadata."
            )
        self._validate_loader_report(loaded, observations, allow_unknown_skips=True)
        if loaded is not None:
            self._skipped_names.update(
                name for name, copies in observations if copies == 0
            )

    def _source_tensor(self, item: dict[str, Any]) -> torch.Tensor:
        shape = tuple(int(dim) for dim in item["shape"])
        dtype = sparse_codec.dtype_from_name(str(item["dtype"]))
        byte_count = prod(shape) * dtype.itemsize
        if byte_count > self._scratch.numel():
            self._scratch = torch.empty(
                byte_count, dtype=torch.uint8, device=self._scratch.device
            )
        return self._scratch[:byte_count].view(dtype).view(shape)

    @staticmethod
    def _scatter_values(
        source: torch.Tensor,
        item: dict[str, Any],
        locations: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        source_bits = sparse_codec.integer_view(source).reshape(-1)
        expected_dtype = sparse_codec.integer_dtype_for_element_size(
            source.element_size()
        )
        if values.dtype != expected_dtype:
            raise RuntimeError(
                f"Sparse values have the wrong dtype for {item['name']!r}."
            )
        values = values.to(device=source.device, non_blocking=True)
        if item["index_encoding"] == "range":
            source_bits.narrow(0, int(item["range_start"]), values.numel()).copy_(
                values
            )
            return
        source_bits.index_copy_(
            0,
            locations.to(device=source.device, dtype=torch.int64, non_blocking=True),
            values,
        )

    def _prepare_loader_weight(
        self,
        item: dict[str, Any],
        locations: torch.Tensor,
        values: torch.Tensor,
    ) -> _LoaderWeight:
        operation = sparse_codec.sparse_operation(item["operation"])
        source = self._source_tensor(item)
        exact_sentinel = None
        if operation == "xor":
            source.zero_()
        elif not source.dtype.is_floating_point:
            raise RuntimeError("Sparse overwrite requires a floating-point source.")
        else:
            source.fill_(float("nan"))
            if source.element_size() == 1:
                source_bits = sparse_codec.integer_view(source)
                exact_sentinel = int(source_bits.reshape(-1)[0].item())
                if bool(values.eq(exact_sentinel).any()):
                    exact_sentinel ^= 0x80
                    if bool(values.eq(exact_sentinel).any()):
                        raise RuntimeError(
                            "FP8 sparse overwrite exhausted its sentinel values."
                        )
                    source_bits.fill_(exact_sentinel)
        self._scatter_values(source, item, locations, values)

        return (
            str(item["name"]),
            source,
            operation,
            int(item.get("verification_samples", 0)),
            exact_sentinel,
        )

    def _load_weights(
        self,
        weights: Iterable[_LoaderWeight],
        verification: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> tuple[Any, list[tuple[str, int]]]:
        mode = _SparseWeightLoadMode(self._target_storages, verification)
        yielded_names: list[str] = []
        observations: list[tuple[str, int]] = []

        def observed_weights() -> Iterator[tuple[str, torch.Tensor]]:
            active = False
            for name, source, operation, sample_limit, exact_sentinel in weights:
                if active:
                    mode.finish()
                    observations.append((yielded_names[-1], mode.copies))
                mode.start(source, operation, sample_limit, exact_sentinel)
                active = True
                yielded_names.append(name)
                yield name, source
            if active:
                mode.finish()
                observations.append((yielded_names[-1], mode.copies))

        with torch.no_grad(), mode:
            loaded = self.model_runner.model.load_weights(observed_weights())
        if len(observations) != len(yielded_names):
            raise RuntimeError("Native loader did not consume all sparse weights.")
        return loaded, observations

    @staticmethod
    def _validate_loader_report(
        loaded: Any,
        observations: list[tuple[str, int]],
        *,
        allow_unknown_skips: bool,
    ) -> None:
        copied = sum(copies > 0 for _, copies in observations)
        if loaded is None:
            if not allow_unknown_skips and copied != len(observations):
                raise RuntimeError(
                    "Native loader did not report whether uncopied sparse weights "
                    "were skipped."
                )
        elif len(loaded) > copied:
            raise RuntimeError(
                "Native loader reported a loaded sparse weight without a supported "
                "target copy."
            )

    def _apply_decoded_items(
        self,
        items: Iterable[tuple[dict[str, Any], torch.Tensor, torch.Tensor]],
    ) -> None:
        def weights() -> Iterator[_LoaderWeight]:
            for item, locations, values in items:
                if str(item["name"]) in self._skipped_names:
                    continue
                yield self._prepare_loader_weight(item, locations, values)

        loaded, observations = self._load_weights(weights(), self._verification)
        self._validate_loader_report(loaded, observations, allow_unknown_skips=False)

    @wrap_with_nvtx_name(
        "vllm_internal_worker_extension/update_weights_from_decoded_sparse_payload"
    )
    def update_weights_from_decoded_sparse_payload(
        self, *payloads: bytes | str
    ) -> dict[str, Any]:
        return self._load_decoded_sparse_payloads(
            tuple(
                io.BytesIO(payload) if isinstance(payload, bytes) else payload
                for payload in payloads
            )
        )

    def _load_decoded_sparse_payloads(
        self, sources: tuple[str | io.BytesIO, ...]
    ) -> dict[str, Any]:
        started = time.perf_counter()
        deserialize_s = [0.0]

        def decoded_items() -> Iterator[sparse_codec.DecodedSparseItem]:
            for source in sources:
                item_started = time.perf_counter()
                payload = cast(
                    sparse_codec.DecodedSparsePayload,
                    torch.load(
                        source,
                        map_location="cpu",
                        weights_only=True,
                        mmap=isinstance(source, str),
                    ),
                )
                deserialize_s[0] += time.perf_counter() - item_started
                yield from sparse_codec.iter_decoded_sparse_payload(payload)

        item_started = time.perf_counter()
        self._apply_decoded_items(decoded_items())
        sparse_apply_s = time.perf_counter() - item_started
        return {
            "ok": True,
            "receiver_deserialize_s": deserialize_s[0],
            "receiver_sparse_apply_s": sparse_apply_s,
            "receiver_total_s": time.perf_counter() - started,
        }

    def synchronize_device(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize(self._cuda_device_index)

    def finish_sparse_delta_refit(self) -> dict[str, Any]:
        """Synchronize and compare bounded samples of target entries just changed."""
        self.synchronize_device()
        verification, self._verification = self._verification, []
        stats = (
            torch.zeros(4, device=verification[0][0].device) if verification else None
        )
        samples = 0
        with torch.no_grad():
            for target, locations, expected_bits in verification:
                actual_bits = (
                    sparse_codec.integer_view(target)
                    .reshape(-1)
                    .index_select(0, locations)
                )
                bit_mismatches = actual_bits.ne(expected_bits)
                actual = actual_bits.view(target.dtype).float()
                expected = expected_bits.view(target.dtype).float()
                difference = torch.nan_to_num(
                    torch.where(
                        bit_mismatches,
                        (actual - expected).abs(),
                        torch.zeros_like(actual),
                    ),
                    nan=float("inf"),
                )
                assert stats is not None
                stats[0] += difference.sum()
                stats[1] = torch.maximum(stats[1], difference.max())
                stats[2] += bit_mismatches.sum()
                stats[3] += (
                    bit_mismatches
                    & ~torch.isclose(actual, expected, rtol=1e-6, atol=1e-8)
                ).sum()
                samples += actual.numel()
        values = [0.0] * 4 if stats is None else stats.cpu().tolist()
        return {
            "ok": True,
            "verification_candidates": samples,
            "verification_samples": samples,
            "verification_exact_mismatches": int(values[2]),
            "verification_mismatches": int(values[3]),
            "verification_abs_sum": float(values[0]),
            "verification_max_abs": float(values[1]),
        }
