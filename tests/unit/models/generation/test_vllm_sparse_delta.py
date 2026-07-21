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

import io
import math
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from nemo_rl.models.generation.vllm.vllm_sparse_delta import (
    VllmSparseDeltaApplier,
)
from nemo_rl.utils.weight_transfer_sparse_codec import (
    SparseOperation,
    encode_sparse_infos,
)
from nemo_rl.utils.weight_transfer_sparse_codec import (
    integer_view as _bits,
)

_PACKED_LOADER_INFOS = [
    ("row_slice", (4, 2), [4, 5, 7], [1.0] * 3),
    ("column_slice", (2, 8), [4, 7, 12, 15], [2.0] * 4),
    ("split", (6, 2), [2, 5, 8, 11], [3.0, 3.0, 4.0, 4.0]),
]


class _NativeLoaderModel:
    def __init__(self, **targets: torch.Tensor) -> None:
        self.targets = targets
        self.load_calls = 0
        self.loaded_names: list[str] = []

    def parameters(self):
        return iter(self.targets.values())

    def buffers(self):
        return iter(())

    def load_weights(self, weights):
        self.load_calls += 1
        loaded = set()
        for name, source in weights:
            self.loaded_names.append(name)
            target = self.targets
            if name == "weight":
                target["identity"].copy_(source)
            elif name == "weight_scale_inv":
                target["scale"].copy_(source)
            elif name == "row_slice":
                target[name].copy_(source[2:4])
            elif name == "column_slice":
                target[name].copy_(source[:, 4:8])
            elif name == "split":
                target[name][:2].copy_(source[1:3])
                target[name][2:].copy_(source[4:6])
            elif name == "exp_transform":
                target[name].copy_(-torch.exp(source))
            else:
                continue
            loaded.add(name)
        return loaded


def _applier(model: Any) -> VllmSparseDeltaApplier:
    return VllmSparseDeltaApplier(
        SimpleNamespace(
            model=model,
            vllm_config=SimpleNamespace(),
        ),
        torch.device("cpu"),
    )


def _payload(
    name: str,
    tensor: torch.Tensor,
    locations: torch.Tensor | list[int],
    values: torch.Tensor,
    operation: SparseOperation = "overwrite",
) -> Any:
    return encode_sparse_infos(
        [(name, tensor, torch.as_tensor(locations), values, operation)]
    )


def _apply_payload(applier: VllmSparseDeltaApplier, payload: Any) -> None:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    applier.update_weights_from_decoded_sparse_payload(buffer.getvalue())


def test_sparse_discovery_reserves_largest_source_without_loading_weights() -> None:
    target = torch.zeros(8)
    applier = _applier(_NativeLoaderModel(identity=target))

    applier.discover_native_skips({"weight": ((8,), torch.float32)})

    assert applier._scratch.numel() == target.numel() * target.element_size()
    assert torch.equal(target, torch.zeros_like(target))


def test_sparse_discovery_caches_rank_local_native_loader_skips() -> None:
    target = torch.zeros(2)
    model = _NativeLoaderModel(identity=target)
    applier = _applier(model)
    info = {
        "weight": ((2,), torch.float32),
        "skipped": ((2,), torch.float32),
    }

    overwrite_names = applier.discover_native_skips(info)
    payload = _payload("skipped", target, [1], _bits(torch.tensor([3.0])))
    _apply_payload(applier, payload)

    assert model.loaded_names == ["weight", "skipped"]
    assert overwrite_names == set()
    assert torch.equal(target, torch.zeros_like(target))


def test_sparse_discovery_classifies_xor_unsafe_native_loads() -> None:
    model = _NativeLoaderModel(
        identity=torch.zeros(2),
        exp_transform=torch.zeros(2),
    )
    applier = _applier(model)
    info = {
        "weight": ((2,), torch.bfloat16),
        "exp_transform": ((2,), torch.float32),
    }

    assert applier.discover_native_skips(info) == {"weight", "exp_transform"}
    assert torch.equal(model.targets["identity"], torch.zeros(2))
    assert torch.equal(model.targets["exp_transform"], torch.zeros(2))


@pytest.mark.vllm
def test_backend_applies_decoded_sparse_payload_sources() -> None:
    from nemo_rl.models.generation.vllm.vllm_backend import (
        VllmInternalWorkerExtension,
    )

    ext = VllmInternalWorkerExtension.__new__(VllmInternalWorkerExtension)
    applier = MagicMock()
    applier.discover_native_skips.return_value = {"weight"}
    applier.update_weights_from_decoded_sparse_payload.return_value = {"ok": True}
    ext._get_sparse_delta_applier = MagicMock(return_value=applier)

    assert ext.prepare_sparse_delta_refit_info({"weight": ((8,), torch.float32)}) == [
        "weight"
    ]

    assert ext.update_weights_from_decoded_sparse_payload(b"payload") == {"ok": True}
    assert ext.update_weights_from_decoded_sparse_payload("first", "second") == {
        "ok": True
    }
    assert [
        item.args
        for item in applier.update_weights_from_decoded_sparse_payload.call_args_list
    ] == [(b"payload",), ("first", "second")]
    applier.discover_native_skips.assert_called_once_with(
        {"weight": ((8,), torch.float32)}
    )


def test_sparse_payload_batches_preserve_order(tmp_path) -> None:
    applier = _applier(_NativeLoaderModel(identity=torch.zeros(1)))
    payload_paths = [tmp_path / f"payload-{index}.pt" for index in range(3)]
    payloads = [
        _payload(
            f"weight-{index}",
            torch.empty(1),
            [0],
            _bits(torch.tensor([float(index)])),
        )
        for index in range(3)
    ]
    for path, payload in zip(payload_paths, payloads, strict=True):
        torch.save(payload, path)
    decoded_applied: list[Any] = []
    applier._apply_decoded_items = lambda items: decoded_applied.extend(
        item for item, _, _ in items
    )
    result = applier.update_weights_from_decoded_sparse_payload(
        *(path.read_bytes() for path in payload_paths)
    )
    decoded_result = applier.update_weights_from_decoded_sparse_payload(
        *(str(path) for path in reversed(payload_paths))
    )

    assert [item["name"] for item in decoded_applied] == [
        "weight-0",
        "weight-1",
        "weight-2",
        "weight-2",
        "weight-1",
        "weight-0",
    ]
    assert result["receiver_deserialize_s"] >= 0.0
    assert result["receiver_sparse_apply_s"] >= 0.0
    assert decoded_result["receiver_deserialize_s"] >= 0.0


def test_sparse_payload_batch_uses_one_streaming_native_loader_call() -> None:
    identity = torch.zeros(2)
    scale = torch.zeros(2)
    model = _NativeLoaderModel(identity=identity, scale=scale)
    serialized = []
    for payload in (
        _payload("weight", identity, [0], _bits(torch.tensor([2.0]))),
        _payload("weight_scale_inv", scale, [1], _bits(torch.tensor([3.0]))),
    ):
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        serialized.append(buffer.getvalue())

    _applier(model).update_weights_from_decoded_sparse_payload(*serialized)

    assert model.load_calls == 1
    assert torch.equal(identity, torch.tensor([2.0, 0.0]))
    assert torch.equal(scale, torch.tensor([0.0, 3.0]))


def test_compact_sparse_payload_decodes_locations_for_apply(tmp_path) -> None:
    target = torch.zeros(8)
    payload = _payload("weight", target, [1, 5], _bits(torch.tensor([2.0, 6.0])))
    path = tmp_path / "payload.pt"
    torch.save(payload, path)

    result = _applier(
        _NativeLoaderModel(identity=target)
    ).update_weights_from_decoded_sparse_payload(str(path))

    assert torch.equal(target, torch.tensor([0.0, 2.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0]))
    assert result["receiver_sparse_apply_s"] >= 0.0


def test_native_loaders_apply_sparse_views_and_transforms() -> None:
    targets = {
        "identity": torch.zeros(4),
        "row_slice": torch.zeros(2, 2),
        "column_slice": torch.zeros(2, 4),
        "split": torch.zeros(4, 2),
        "exp_transform": torch.tensor([-2.0, -4.0]),
    }
    infos = [
        ("weight", (4,), [1, 2], [1.0, 2.0]),
        *_PACKED_LOADER_INFOS,
        (
            "exp_transform",
            (2,),
            [0, 1],
            [math.log(3.0), math.log(2.0)],
        ),
    ]
    payload = encode_sparse_infos(
        [
            (
                name,
                torch.empty(shape),
                torch.tensor(locations),
                _bits(torch.tensor(values, dtype=torch.float32)),
                "overwrite",
            )
            for name, shape, locations, values in infos
        ]
    )
    payload[2][-1]["verification_samples"] = 2

    applier = _applier(_NativeLoaderModel(**targets))
    _apply_payload(applier, payload)
    verification = applier.finish_sparse_delta_refit()

    assert torch.equal(targets["identity"], torch.tensor([0.0, 1.0, 2.0, 0.0]))
    assert targets["row_slice"].view(-1).tolist() == [1.0, 1.0, 0.0, 1.0]
    assert targets["column_slice"].view(-1).tolist() == [2.0, 0.0, 0.0, 2.0] * 2
    assert targets["split"].view(-1).tolist() == [
        3.0,
        0.0,
        0.0,
        3.0,
        4.0,
        0.0,
        0.0,
        4.0,
    ]
    assert torch.allclose(targets["exp_transform"], torch.tensor([-3.0, -2.0]))
    assert verification["verification_candidates"] == 2
    assert verification["verification_samples"] == 2
    assert verification["verification_exact_mismatches"] == 0


def test_xor_applies_through_packed_native_loaders() -> None:
    targets = {
        "row_slice": torch.zeros(2, 2),
        "column_slice": torch.zeros(2, 4),
        "split": torch.zeros(4, 2),
    }
    payload = encode_sparse_infos(
        [
            (
                name,
                torch.empty(shape),
                torch.tensor(locations),
                _bits(torch.tensor(values)).bitwise_xor(
                    _bits(torch.zeros(len(values)))
                ),
                "xor",
            )
            for name, shape, locations, values in _PACKED_LOADER_INFOS
        ]
    )

    _apply_payload(_applier(_NativeLoaderModel(**targets)), payload)

    assert targets["row_slice"].view(-1).tolist() == [1.0, 1.0, 0.0, 1.0]
    assert targets["column_slice"].view(-1).tolist() == [2.0, 0.0, 0.0, 2.0] * 2
    assert targets["split"].view(-1).tolist() == [
        3.0,
        0.0,
        0.0,
        3.0,
        4.0,
        0.0,
        0.0,
        4.0,
    ]


def test_native_loader_explicit_skip_is_accepted() -> None:
    model = _NativeLoaderModel(identity=torch.zeros(1))
    payload = _payload("skipped", torch.empty(1), [0], _bits(torch.tensor([1.0])))

    _apply_payload(_applier(model), payload)

    assert torch.equal(model.targets["identity"], torch.zeros(1))


def test_native_loader_claim_without_copy_fails_closed() -> None:
    model = _NativeLoaderModel(identity=torch.zeros(1))
    model.load_weights = lambda weights: {name for name, _ in weights}
    payload = _payload("weight", torch.empty(1), [0], _bits(torch.tensor([1.0])))

    with pytest.raises(RuntimeError, match="without a supported target copy"):
        _apply_payload(_applier(model), payload)


def test_sparse_overwrite_preserves_unselected_transform_inputs() -> None:
    target = torch.tensor([-2.0, -4.0, -6.0, -8.0])
    payload = _payload(
        "exp_transform",
        target,
        [1],
        _bits(torch.tensor([math.log(3.0)])),
    )

    _apply_payload(_applier(_NativeLoaderModel(exp_transform=target)), payload)

    assert torch.allclose(target, torch.tensor([-2.0, -3.0, -6.0, -8.0]))


def test_unknown_sparse_operation_fails_closed() -> None:
    target = torch.zeros(1)
    payload = _payload("weight", target, [0], _bits(torch.tensor([1.0])))
    payload[2][0]["operation"] = "unknown"

    with pytest.raises(ValueError, match="Unsupported sparse-refit operation"):
        _apply_payload(_applier(_NativeLoaderModel(identity=target)), payload)


@pytest.mark.parametrize(
    ("initial", "verified_value", "exact_mismatches", "mismatches"),
    [(200.0, 4.0, 0, 0), (2.0, 4.0000005, 1, 0), (2.0, 5.0, 1, 1)],
)
def test_sparse_delta_verification_compares_replacement(
    initial: float,
    verified_value: float,
    exact_mismatches: int,
    mismatches: int,
) -> None:
    target = torch.tensor([1.0, initial, 3.0, initial])
    replacement = torch.tensor([initial + 4.0, initial + 4.0])
    payload = _payload("weight", target, [1, 3], _bits(replacement))
    payload[2][0]["verification_samples"] = 2
    applier = _applier(_NativeLoaderModel(identity=target))

    _apply_payload(applier, payload)
    target[[1, 3]] = initial + verified_value
    result = applier.finish_sparse_delta_refit()

    assert torch.equal(
        target,
        torch.tensor([1.0, initial + verified_value, 3.0, initial + verified_value]),
    )
    assert result["verification_candidates"] == 2
    assert result["verification_samples"] == 2
    assert result["verification_exact_mismatches"] == 2 * exact_mismatches
    assert result["verification_mismatches"] == 2 * mismatches


def test_fp8_weight_and_scale_use_exact_bit_overwrite() -> None:
    target = torch.tensor([0x38, 0x40, 0x48], dtype=torch.uint8).view(
        torch.float8_e4m3fn
    )
    scale = torch.tensor([1.0, 2.0])
    current = target.clone()
    current.view(torch.uint8)[1] = 0x41
    current.view(torch.uint8)[2] = 0x7F
    current_scale = scale.clone()
    current_scale[0] = 1.5
    payload = encode_sparse_infos(
        [
            (
                "weight",
                current,
                torch.tensor([1, 2]),
                current.view(torch.uint8)[1:3],
                "overwrite",
            ),
            (
                "weight_scale_inv",
                current_scale,
                torch.tensor([0]),
                current_scale.view(torch.int32)[:1],
                "overwrite",
            ),
        ]
    )
    payload[2][0]["verification_samples"] = 2
    payload[2][1]["verification_samples"] = 1
    applier = _applier(_NativeLoaderModel(identity=target, scale=scale))

    _apply_payload(applier, payload)
    _apply_payload(applier, payload)
    result = applier.finish_sparse_delta_refit()

    assert target.view(torch.uint8).tolist() == [0x38, 0x41, 0x7F]
    assert torch.equal(scale, current_scale)
    assert result["verification_samples"] == 6
    assert result["verification_exact_mismatches"] == 0
    assert result["verification_mismatches"] == 0
    assert result["verification_abs_sum"] == 0.0


def test_xor_applies_exact_bits_and_replay_reverts() -> None:
    baseline = torch.tensor([1.0, 2.0, 3.0])
    target = baseline.clone()
    current = torch.tensor([1.0, 5.0, -0.0])
    locations = torch.tensor([1, 2])
    xor_values = _bits(current)[locations].bitwise_xor(_bits(baseline)[locations])
    payload = encode_sparse_infos([("weight", current, locations, xor_values, "xor")])
    payload[2][0]["verification_samples"] = 2
    applier = _applier(_NativeLoaderModel(identity=target))

    _apply_payload(applier, payload)
    result = applier.finish_sparse_delta_refit()

    assert torch.equal(_bits(target), _bits(current))
    assert result["verification_exact_mismatches"] == 0
    assert result["verification_mismatches"] == 0

    _apply_payload(applier, payload)
    assert torch.equal(_bits(target), _bits(baseline))


def test_overwrite_casts_absolute_source_values() -> None:
    target = torch.zeros(2, dtype=torch.float16)
    source = torch.tensor([1.25, -2.5], dtype=torch.float32)
    payload = _payload("weight", source, [0, 1], _bits(source))

    _apply_payload(_applier(_NativeLoaderModel(identity=target)), payload)

    assert torch.equal(target, source.to(torch.float16))


@pytest.mark.parametrize(
    ("name", "source", "targets", "error"),
    [
        (
            "exp_transform",
            torch.tensor([math.log(2.0)]),
            {"exp_transform": torch.tensor([-1.0])},
            "without changing semantics",
        ),
        (
            "weight",
            torch.tensor([1.0], dtype=torch.float32),
            {"identity": torch.zeros(1, dtype=torch.float16)},
            "without changing semantics",
        ),
    ],
)
def test_xor_rejects_non_bitwise_compatible_targets(
    name: str,
    source: torch.Tensor,
    targets: dict[str, torch.Tensor],
    error: str,
) -> None:
    payload = _payload(name, source, [0], _bits(source), "xor")

    with pytest.raises(RuntimeError, match=error):
        _apply_payload(_applier(_NativeLoaderModel(**targets)), payload)


def test_xor_rejects_overlapping_native_loader_copies() -> None:
    class RepeatedCopyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(2))

        def load_weights(self, weights) -> None:
            for _, source in weights:
                self.weight.copy_(source)
                self.weight.copy_(source)

    source = torch.tensor([1.0, 2.0])
    payload = _payload(
        "weight",
        source,
        [0, 1],
        _bits(source).bitwise_xor(_bits(torch.zeros_like(source))),
        "xor",
    )

    with pytest.raises(RuntimeError, match="without changing semantics"):
        _apply_payload(_applier(RepeatedCopyModel()), payload)
