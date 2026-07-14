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
    decode_sparse_tensor_payload_for_staging,
    encode_sparse_infos,
    integer_dtype_for_element_size,
    iter_decoded_sparse_payload,
)


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
            elif name.endswith("self_attn.k_proj.weight"):
                target["qkv"][4:6].copy_(source[2:4])
            elif name.endswith("mlp.gate_proj.weight"):
                target["merged"][:4].copy_(source[4:8])
            elif name.endswith("mlp.up_proj.weight"):
                target["merged"][4:8].copy_(source[4:8])
            elif name.endswith("experts.3.gate_proj.weight"):
                target["w13"][1, :4].copy_(source[4:8])
            elif name.endswith("experts.3.down_proj.weight"):
                target["w2"][1].copy_(source[:, 4:8])
            elif name.endswith("mixer.in_proj.weight"):
                target["mamba"][:2].copy_(source[2:4])
                target["mamba"][2:].copy_(source[6:10])
            elif name.endswith(("mixer.A", "mixer.A_log")):
                target["a"].copy_(source)
                target["a"].copy_(-torch.exp(target["a"]))
            elif name == "transformed":
                target["identity"].copy_(source + 1)
            elif name == "raise_after_copy":
                target["identity"].copy_(source)
                raise RuntimeError("loader failed")
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


def _bits(values: torch.Tensor) -> torch.Tensor:
    return values.contiguous().view(
        integer_dtype_for_element_size(values.element_size())
    )


def _decode_staged(payload: Any) -> list[Any]:
    return list(
        iter_decoded_sparse_payload(decode_sparse_tensor_payload_for_staging(payload))
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
    for item, locations, values in _decode_staged(payload):
        applier._apply_decoded_item(item, locations, values)


def test_sparse_prewarm_reserves_largest_source_without_loading_weights() -> None:
    target = torch.zeros(8)
    applier = _applier(_NativeLoaderModel(identity=target))

    applier.prewarm({"weight": ((8,), torch.float32)})

    assert applier._scratch.numel() == target.numel() * target.element_size()
    assert torch.equal(target, torch.zeros_like(target))


@pytest.mark.vllm
def test_sparse_prewarm_caches_rank_local_native_loader_skips() -> None:
    target = torch.zeros(2)
    model = _NativeLoaderModel(identity=target)
    applier = _applier(model)
    info = {
        "weight": ((2,), torch.float32),
        "skipped": ((2,), torch.float32),
    }

    applier.prewarm(info)
    applier.discover_native_skips(info)
    payload = _payload("skipped", target, [1], _bits(torch.tensor([3.0])))
    _apply_payload(applier, payload)

    assert model.loaded_names == ["weight", "skipped"]
    assert torch.equal(target, torch.zeros_like(target))


@pytest.mark.vllm
def test_backend_applies_decoded_sparse_payload_files() -> None:
    from nemo_rl.models.generation.vllm.vllm_backend import (
        VllmInternalWorkerExtension,
    )

    ext = VllmInternalWorkerExtension.__new__(VllmInternalWorkerExtension)
    applier = MagicMock()
    applier.update_weights_from_decoded_sparse_payload.return_value = {"ok": True}
    applier.update_weights_from_decoded_sparse_payload_files.return_value = {"ok": True}
    ext._get_sparse_delta_applier = MagicMock(return_value=applier)

    ext.prepare_sparse_delta_refit_info({"weight": ((8,), torch.float32)})

    assert ext.update_weights_from_decoded_sparse_payload(b"payload") == {"ok": True}
    assert ext.update_weights_from_decoded_sparse_payload_files("first", "second") == {
        "ok": True
    }
    applier.update_weights_from_decoded_sparse_payload.assert_called_once_with(
        b"payload"
    )
    applier.update_weights_from_decoded_sparse_payload_files.assert_called_once_with(
        "first", "second"
    )
    applier.prewarm.assert_called_once_with({"weight": ((8,), torch.float32)})
    applier.discover_native_skips.assert_called_once_with(
        {"weight": ((8,), torch.float32)}
    )


@pytest.mark.vllm
def test_sparse_payload_batches_preserve_order(tmp_path) -> None:
    applier = _applier(_NativeLoaderModel(identity=torch.zeros(1)))
    decoded_paths = [tmp_path / f"decoded-{index}.pt" for index in range(3)]
    decoded_payloads = [
        (
            (
                torch.tensor([index], dtype=torch.int32),
                torch.empty(0, dtype=torch.int64),
            ),
            (torch.tensor([index]),),
            [
                {
                    "name": "weight",
                    "index": index,
                    "shape": (1,),
                    "dtype": "float32",
                    "operation": "overwrite",
                    "index_encoding": "range",
                    "range_start": 0,
                    "decoded_location_group": 0,
                    "decoded_location_start": 0,
                    "decoded_location_end": 1,
                    "value_group": 0,
                    "value_start": 0,
                    "value_end": 1,
                }
            ],
        )
        for index in range(3)
    ]
    for path, payload in zip(decoded_paths, decoded_payloads, strict=True):
        torch.save(payload, path)
    decoded_applied: list[Any] = []
    applier._apply_decoded_items = lambda items: decoded_applied.extend(
        item for item, _, _ in items
    )
    result = applier.update_weights_from_decoded_sparse_payload(
        *(path.read_bytes() for path in decoded_paths)
    )
    decoded_result = applier.update_weights_from_decoded_sparse_payload_files(
        *(str(path) for path in reversed(decoded_paths))
    )

    assert [item["index"] for item in decoded_applied] == [
        0,
        1,
        2,
        2,
        1,
        0,
    ]
    assert result["receiver_deserialize_s"] >= 0.0
    assert result["receiver_scratch_s"] >= 0.0
    assert result["receiver_sparse_apply_s"] >= 0.0
    assert decoded_result["receiver_deserialize_s"] >= 0.0


@pytest.mark.vllm
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
        torch.save(decode_sparse_tensor_payload_for_staging(payload), buffer)
        serialized.append(buffer.getvalue())

    _applier(model).update_weights_from_decoded_sparse_payload(*serialized)

    assert model.load_calls == 1
    assert torch.equal(identity, torch.tensor([2.0, 0.0]))
    assert torch.equal(scale, torch.tensor([0.0, 3.0]))


@pytest.mark.vllm
def test_decoded_sparse_payload_converts_compact_locations_for_apply(tmp_path) -> None:
    target = torch.zeros(8)
    payload = _payload("weight", target, [1, 5], _bits(torch.tensor([2.0, 6.0])))
    decoded = decode_sparse_tensor_payload_for_staging(payload)
    assert decoded[0][0].dtype == torch.int32
    path = tmp_path / "decoded.pt"
    torch.save(decoded, path)

    result = _applier(
        _NativeLoaderModel(identity=target)
    ).update_weights_from_decoded_sparse_payload_files(str(path))

    assert torch.equal(target, torch.tensor([0.0, 2.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0]))
    assert result["receiver_sparse_apply_s"] >= 0.0


@pytest.mark.vllm
def test_native_loaders_apply_sparse_overwrite_without_family_plans() -> None:
    targets = {
        "identity": torch.zeros(4),
        "qkv": torch.zeros(8, 2),
        "merged": torch.zeros(8, 2),
        "w13": torch.zeros(2, 4, 2),
        "w2": torch.zeros(2, 2, 4),
        "mamba": torch.zeros(6, 2),
        "a": torch.tensor([-2.0, -4.0]),
    }
    infos = [
        ("weight", (4,), [1, 2], [1.0, 2.0]),
        ("model.layers.0.self_attn.k_proj.weight", (4, 2), [0, 4, 5, 7], [1] * 4),
        ("model.layers.0.mlp.gate_proj.weight", (8, 2), [0, 8, 9, 15], [2] * 4),
        ("model.layers.0.mlp.up_proj.weight", (8, 2), [8, 15], [3] * 2),
        ("model.layers.0.mlp.experts.3.gate_proj.weight", (8, 2), [8, 15], [4] * 2),
        (
            "model.layers.0.mlp.experts.3.down_proj.weight",
            (2, 8),
            [3, 4, 7, 12, 15],
            [5] * 5,
        ),
        ("model.layers.0.mixer.in_proj.weight", (10, 2), [0, 4, 7, 12, 19], [6] * 5),
        (
            "backbone.layers.0.mixer.A_log",
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
    payload[2][7]["verification_samples"] = 2

    applier = _applier(_NativeLoaderModel(**targets))
    _apply_payload(applier, payload)
    verification = applier.finish_sparse_delta_refit()

    assert torch.equal(targets["identity"], torch.tensor([0.0, 1.0, 2.0, 0.0]))
    assert targets["qkv"].view(-1)[[8, 9, 11]].tolist() == [1.0, 1.0, 1.0]
    assert targets["merged"].view(-1)[[0, 1, 7, 8, 15]].tolist() == [2, 2, 2, 3, 3]
    assert targets["w13"].view(-1)[[8, 15]].tolist() == [4, 4]
    assert targets["w2"].view(-1)[[8, 11, 12, 15]].tolist() == [5, 5, 5, 5]
    assert targets["mamba"].view(-1)[[0, 3, 4, 11]].tolist() == [6, 6, 6, 6]
    assert torch.allclose(targets["a"], torch.tensor([-3.0, -2.0]))
    assert verification["verification_candidates"] == 2
    assert verification["verification_samples"] == 2
    assert verification["verification_exact_mismatches"] == 0


@pytest.mark.vllm
def test_xor_applies_through_packed_native_loaders() -> None:
    targets = {
        "qkv": torch.zeros(8, 2),
        "merged": torch.zeros(8, 2),
        "w13": torch.zeros(2, 4, 2),
        "mamba": torch.zeros(6, 2),
    }
    infos = [
        (
            "model.layers.0.self_attn.k_proj.weight",
            (4, 2),
            [0, 4, 5, 7],
            [1.0] * 4,
        ),
        (
            "model.layers.0.mlp.gate_proj.weight",
            (8, 2),
            [0, 8, 9, 15],
            [2.0] * 4,
        ),
        (
            "model.layers.0.mlp.up_proj.weight",
            (8, 2),
            [8, 15],
            [3.0] * 2,
        ),
        (
            "model.layers.0.mlp.experts.3.gate_proj.weight",
            (8, 2),
            [8, 15],
            [4.0] * 2,
        ),
        (
            "model.layers.0.mixer.in_proj.weight",
            (10, 2),
            [0, 4, 7, 12, 19],
            [6.0] * 5,
        ),
    ]
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
            for name, shape, locations, values in infos
        ]
    )

    _apply_payload(_applier(_NativeLoaderModel(**targets)), payload)

    assert targets["qkv"].view(-1)[[8, 9, 11]].tolist() == [1.0, 1.0, 1.0]
    assert targets["merged"].view(-1)[[0, 1, 7, 8, 15]].tolist() == [2, 2, 2, 3, 3]
    assert targets["w13"].view(-1)[[8, 15]].tolist() == [4.0, 4.0]
    assert targets["mamba"].view(-1)[[0, 3, 4, 11]].tolist() == [6, 6, 6, 6]


@pytest.mark.vllm
def test_native_loader_explicit_skip_is_accepted() -> None:
    model = _NativeLoaderModel(identity=torch.zeros(1))
    payload = _payload("skipped", torch.empty(1), [0], _bits(torch.tensor([1.0])))
    item, locations, values = _decode_staged(payload)[0]

    _applier(model)._apply_decoded_item(item, locations, values)

    assert torch.equal(model.targets["identity"], torch.zeros(1))


@pytest.mark.vllm
def test_native_loader_claim_without_copy_fails_closed() -> None:
    model = _NativeLoaderModel(identity=torch.zeros(1))
    model.load_weights = lambda weights: {name for name, _ in weights}
    payload = _payload("weight", torch.empty(1), [0], _bits(torch.tensor([1.0])))

    with pytest.raises(RuntimeError, match="without a supported target copy"):
        _apply_payload(_applier(model), payload)


@pytest.mark.vllm
def test_sparse_overwrite_preserves_unselected_transform_inputs() -> None:
    target = torch.tensor([-2.0, -4.0, -6.0, -8.0])
    payload = _payload(
        "backbone.layers.0.mixer.A_log",
        target,
        [1],
        _bits(torch.tensor([math.log(3.0)])),
    )

    _apply_payload(_applier(_NativeLoaderModel(a=target)), payload)

    assert torch.allclose(target, torch.tensor([-2.0, -3.0, -6.0, -8.0]))


@pytest.mark.vllm
def test_sparse_overwrite_rolls_back_loader_failure() -> None:
    target = torch.tensor([1.0, 2.0])
    payload = _payload("raise_after_copy", target, [1], _bits(torch.tensor([3.0])))

    with pytest.raises(RuntimeError, match="loader failed"):
        _apply_payload(_applier(_NativeLoaderModel(identity=target)), payload)

    assert torch.equal(target, torch.tensor([1.0, 2.0]))


@pytest.mark.vllm
def test_unknown_sparse_operation_fails_closed() -> None:
    target = torch.zeros(1)
    payload = _payload("weight", target, [0], _bits(torch.tensor([1.0])))
    payload[2][0]["operation"] = "unknown"

    with pytest.raises(ValueError, match="Unsupported sparse-refit operation"):
        _apply_payload(_applier(_NativeLoaderModel(identity=target)), payload)


@pytest.mark.vllm
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


@pytest.mark.vllm
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


@pytest.mark.vllm
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


@pytest.mark.vllm
def test_overwrite_casts_absolute_source_values() -> None:
    target = torch.zeros(2, dtype=torch.float16)
    source = torch.tensor([1.25, -2.5], dtype=torch.float32)
    payload = _payload("weight", source, [0, 1], _bits(source))

    _apply_payload(_applier(_NativeLoaderModel(identity=target)), payload)

    assert torch.equal(target, source.to(torch.float16))


@pytest.mark.vllm
@pytest.mark.parametrize(
    ("name", "source", "targets", "error"),
    [
        (
            "backbone.layers.0.mixer.A_log",
            torch.tensor([math.log(2.0)]),
            {"a": torch.tensor([-1.0])},
            "transforms its input",
        ),
        (
            "weight",
            torch.tensor([1.0], dtype=torch.float32),
            {"identity": torch.zeros(1, dtype=torch.float16)},
            "dtypes must match",
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


@pytest.mark.vllm
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

    with pytest.raises(RuntimeError, match="overlapping target copies"):
        _apply_payload(_applier(RepeatedCopyModel()), payload)
