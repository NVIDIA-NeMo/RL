# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the per-rank device-memory split.

The attribution is the part worth testing and it needs no GPU: the walk asks
tensors for their storage, so CPU tensors standing in for CUDA ones exercise
every rule except the device filter, which gets its own test.
"""

from types import SimpleNamespace

import pytest
import torch

from nemo_rl.telemetry.device_memory import (
    CATEGORIES,
    CATEGORY_GRADIENTS,
    CATEGORY_OPTIMIZER_STATE,
    CATEGORY_OTHER,
    CATEGORY_PARAMETERS,
    DEVICE_MEMORY_KEY,
    RL_MEMORY_CATEGORY_ATTR,
    _gradients,
    _parameters,
    _warn_if_chunks_are_unreachable,
    attribute_device_memory,
    record_device_memory,
)


class _Storage:
    def __init__(self, address, nbytes):
        self._address = address
        self._nbytes = nbytes

    def data_ptr(self):
        return self._address

    def nbytes(self):
        return self._nbytes


class _Tensor:
    """A tensor stripped to what the walk asks of it."""

    def __init__(self, address, nbytes, device="cuda"):
        self._storage = _Storage(address, nbytes)
        self.device = SimpleNamespace(type=device)

    def untyped_storage(self):
        return self._storage


def test_categories_partition_the_allocated_total():
    """The four sum to allocated, with the remainder landing in 'other'."""
    totals = attribute_device_memory(
        (
            (CATEGORY_PARAMETERS, [_Tensor(1, 400)]),
            (CATEGORY_GRADIENTS, [_Tensor(2, 300)]),
            (CATEGORY_OPTIMIZER_STATE, [_Tensor(3, 200)]),
        ),
        allocated=1000,
    )
    assert totals == {
        CATEGORY_PARAMETERS: 400,
        CATEGORY_GRADIENTS: 300,
        CATEGORY_OPTIMIZER_STATE: 200,
        CATEGORY_OTHER: 100,
    }
    assert sum(totals.values()) == 1000


def test_every_category_is_always_present():
    """A category that falls to zero reports zero rather than ending its series."""
    totals = attribute_device_memory([], allocated=512)
    assert set(totals) == set(CATEGORIES)
    assert totals[CATEGORY_OTHER] == 512


def test_one_storage_is_charged_once_to_the_earliest_category():
    """Views share storage; a gradient buffer must not be billed per parameter."""
    buffer = [_Tensor(7, 800), _Tensor(7, 800), _Tensor(7, 800)]
    totals = attribute_device_memory(((CATEGORY_GRADIENTS, buffer),), allocated=800)
    assert totals[CATEGORY_GRADIENTS] == 800
    assert totals[CATEGORY_OTHER] == 0

    # Reachable as both a parameter and optimizer state: an fp32 main param
    # aliasing a weight is a parameter, because parameters come first.
    shared = _Tensor(9, 600)
    totals = attribute_device_memory(
        (
            (CATEGORY_PARAMETERS, [shared]),
            (CATEGORY_OPTIMIZER_STATE, [shared]),
        ),
        allocated=600,
    )
    assert totals[CATEGORY_PARAMETERS] == 600
    assert totals[CATEGORY_OPTIMIZER_STATE] == 0


def test_host_resident_tensors_are_not_device_memory():
    """Optimizer state paged to the host must not be counted against the GPU."""
    totals = attribute_device_memory(
        (
            (
                CATEGORY_OPTIMIZER_STATE,
                [_Tensor(1, 999, device="cpu"), _Tensor(2, 50)],
            ),
        ),
        allocated=50,
    )
    assert totals[CATEGORY_OPTIMIZER_STATE] == 50


def test_missing_and_unreadable_tensors_are_skipped():
    """A parameter with no gradient yet, and anything that cannot answer."""

    class _Freed(_Tensor):
        def untyped_storage(self):
            raise RuntimeError("storage already freed")

    totals = attribute_device_memory(
        ((CATEGORY_GRADIENTS, [None, object(), _Freed(1, 10), _Tensor(2, 64)]),),
        allocated=64,
    )
    assert totals[CATEGORY_GRADIENTS] == 64


def test_residual_never_goes_negative():
    """Clamped, so an unexpected overshoot does not read as a negative gauge."""
    totals = attribute_device_memory(
        ((CATEGORY_PARAMETERS, [_Tensor(1, 900)]),), allocated=100
    )
    assert totals[CATEGORY_OTHER] == 0


def test_real_tensors_report_their_own_bytes():
    """Against torch itself, so the storage API is not only mocked."""
    weight = torch.zeros(256, dtype=torch.float32)
    totals = attribute_device_memory(((CATEGORY_PARAMETERS, [weight]),), allocated=0)
    # CPU tensors are skipped by the device filter, which is the point: the
    # walk never charges host memory to a device series.
    assert totals[CATEGORY_PARAMETERS] == 0


@pytest.mark.parametrize("size", [None, 1])
def test_no_virtual_pipeline_warning_when_every_chunk_is_reachable(size, monkeypatch):
    warned = []
    monkeypatch.setattr(
        "nemo_rl.telemetry.device_memory.warn_once",
        lambda key, message: warned.append(key),
    )
    model = SimpleNamespace(
        config=SimpleNamespace(virtual_pipeline_model_parallel_size=size)
    )
    _warn_if_chunks_are_unreachable(model)
    _warn_if_chunks_are_unreachable(SimpleNamespace())
    assert warned == []


def test_virtual_pipeline_undercount_is_logged_not_silent(monkeypatch):
    """Only chunk 0 is reachable, so the split is wrong in a plausible way."""
    warned = []
    monkeypatch.setattr(
        "nemo_rl.telemetry.device_memory.warn_once",
        lambda key, message: warned.append((key, message)),
    )
    model = SimpleNamespace(
        config=SimpleNamespace(virtual_pipeline_model_parallel_size=4)
    )
    _warn_if_chunks_are_unreachable(model)

    assert len(warned) == 1
    key, message = warned[0]
    assert key == "device_memory_virtual_pipeline"
    assert "4" in message and CATEGORY_OTHER in message


def test_parameters_accepts_a_chunk_list_or_a_single_module():
    """Megatron passes a list; the worker holds one module."""
    chunk = torch.nn.Linear(4, 4)
    assert len(_parameters(chunk)) == 2
    assert len(_parameters([chunk, torch.nn.Linear(4, 4)])) == 4
    # An entry that is not a module at all is skipped rather than raising.
    assert len(_parameters([chunk, object()])) == 2


def test_megatron_gradients_are_found_on_main_grad():
    """mcore DDP accumulates into main_grad and leaves .grad unset."""
    mcore_style = SimpleNamespace(grad=None, main_grad=_Tensor(1, 128))
    torch_style = SimpleNamespace(grad=_Tensor(2, 64), main_grad=None)
    totals = attribute_device_memory(
        ((CATEGORY_GRADIENTS, _gradients([mcore_style, torch_style])),),
        allocated=192,
    )
    assert totals[CATEGORY_GRADIENTS] == 192


def test_a_parameter_holding_both_grad_forms_is_billed_once():
    """.grad aliasing main_grad must not double the gradient category."""
    shared = _Tensor(5, 256)
    both = SimpleNamespace(grad=shared, main_grad=shared)
    totals = attribute_device_memory(
        ((CATEGORY_GRADIENTS, _gradients([both])),), allocated=256
    )
    assert totals[CATEGORY_GRADIENTS] == 256


@pytest.fixture
def opted_in(monkeypatch):
    """Switch the collection on, as a run config or a raw env var would."""
    monkeypatch.setenv("NEMO_RL_OTEL_DEVICE_MEMORY", "1")


def test_collection_is_off_until_asked_for(monkeypatch):
    """Default off, and the gate comes before the walk, not after it."""
    monkeypatch.delenv("NEMO_RL_OTEL_DEVICE_MEMORY", raising=False)
    walked = []
    monkeypatch.setattr(
        "nemo_rl.telemetry.device_memory.device_memory_bytes",
        lambda model, optimizer: walked.append(1) or dict.fromkeys(CATEGORIES, 8),
    )
    monkeypatch.setattr(
        "nemo_rl.telemetry.device_memory.record_rl_metrics",
        lambda values, attributes=None: pytest.fail("recorded while disabled"),
    )
    record_device_memory(object(), object())
    assert walked == []


@pytest.mark.parametrize("value", ["1", "true", "yes", "on"])
def test_the_env_var_alone_is_enough_to_switch_it_on(value, monkeypatch):
    """A worker only ever sees the setting as an env var."""
    monkeypatch.setenv("NEMO_RL_OTEL_DEVICE_MEMORY", value)
    recorded = []
    monkeypatch.setattr(
        "nemo_rl.telemetry.device_memory.device_memory_bytes",
        lambda model, optimizer: dict.fromkeys(CATEGORIES, 8),
    )
    monkeypatch.setattr(
        "nemo_rl.telemetry.device_memory.record_rl_metrics",
        lambda values, attributes=None: recorded.append(values),
    )
    record_device_memory(object(), object())
    assert len(recorded) == len(CATEGORIES)


def test_record_emits_one_point_per_category(opted_in, monkeypatch):
    """Each category is its own attribute set, carrying any caller attributes."""
    recorded = []
    monkeypatch.setattr(
        "nemo_rl.telemetry.device_memory.record_rl_metrics",
        lambda values, attributes=None: recorded.append((values, attributes)),
    )
    monkeypatch.setattr(
        "nemo_rl.telemetry.device_memory.device_memory_bytes",
        lambda model, optimizer: dict.fromkeys(CATEGORIES, 8),
    )
    record_device_memory(object(), object(), attributes={"rl.phase": "train"})

    assert len(recorded) == len(CATEGORIES)
    assert {a[RL_MEMORY_CATEGORY_ATTR] for _, a in recorded} == set(CATEGORIES)
    for values, attributes in recorded:
        assert values == {DEVICE_MEMORY_KEY: 8.0}
        assert attributes["rl.phase"] == "train"


def test_record_never_raises(opted_in, monkeypatch):
    """Telemetry must not be able to fail a training step."""
    monkeypatch.setattr(
        "nemo_rl.telemetry.device_memory.device_memory_bytes",
        lambda model, optimizer: (_ for _ in ()).throw(RuntimeError("no optimizer")),
    )
    record_device_memory(object(), object())


def test_record_never_raises_when_the_exporter_itself_fails(opted_in, monkeypatch):
    """A dead sink must not fail the step either, only the walk being guarded."""
    monkeypatch.setattr(
        "nemo_rl.telemetry.device_memory.device_memory_bytes",
        lambda model, optimizer: dict.fromkeys(CATEGORIES, 8),
    )
    monkeypatch.setattr(
        "nemo_rl.telemetry.device_memory.record_rl_metrics",
        lambda values, attributes=None: (_ for _ in ()).throw(RuntimeError("no meter")),
    )
    record_device_memory(object(), object())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_split_of_a_real_step_sums_to_allocated():
    """End to end on a device: a real model, a real optimizer, real storages."""
    from nemo_rl.telemetry.device_memory import device_memory_bytes

    model = torch.nn.Linear(64, 64).cuda()
    optimizer = torch.optim.AdamW(model.parameters())
    model(torch.randn(8, 64, device="cuda")).sum().backward()
    optimizer.step()

    totals = device_memory_bytes(model, optimizer)
    assert sum(totals.values()) == torch.cuda.memory_allocated()
    assert totals[CATEGORY_PARAMETERS] > 0
    assert totals[CATEGORY_GRADIENTS] > 0
    # Adam keeps two moments per parameter, so its state outweighs the weights.
    assert totals[CATEGORY_OPTIMIZER_STATE] > totals[CATEGORY_PARAMETERS]
