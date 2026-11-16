# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the per-rank device-memory split.

The attribution is the part worth testing and it needs no GPU: the walk asks
tensors for their storage, so CPU tensors standing in for CUDA ones exercise
every rule except the device filter, which gets its own test.
"""

import os
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
    _local_tensor,
    _optimizer_params,
    _optimizer_state_tensors,
    _parameters,
    _warn_if_chunks_are_unreachable,
    attribute_device_memory,
    device_memory_bytes,
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


def test_host_tensors_are_filtered_before_anything_is_charged():
    """A CPU tensor is real and has a storage; the device filter still skips it."""
    weight = torch.zeros(256, dtype=torch.float32)
    totals = attribute_device_memory(((CATEGORY_PARAMETERS, [weight]),), allocated=0)
    assert totals[CATEGORY_PARAMETERS] == 0


def test_real_views_share_one_storage_and_report_the_whole_buffer():
    """The premise the dedup rests on, checked against torch rather than assumed.

    A view reports its *buffer's* size and address, not its own slice, which is
    both why summing storages is right for a gradient bucket and why they have
    to be deduplicated. The device filter keeps real CPU tensors out of
    _storages, so this exercises the same calls directly.
    """
    buffer = torch.zeros(1024, dtype=torch.float32)
    slices = [buffer[:10], buffer[10:20], buffer[20:]]

    addresses, sizes = set(), set()
    for tensor in slices:
        storage = _local_tensor(tensor).untyped_storage()
        addresses.add(storage.data_ptr())
        sizes.add(storage.nbytes())

    # One buffer, however many views onto it.
    assert len(addresses) == 1
    assert addresses == {buffer.untyped_storage().data_ptr()}
    assert sizes == {4096}
    # What summing tensors instead of storages would have billed.
    assert sum(t.numel() * t.element_size() for t in slices) == 4096

    # And through the dedup: three views of one 4 KiB buffer cost 4 KiB.
    totals = attribute_device_memory(
        ((CATEGORY_GRADIENTS, [_Tensor(a, s) for a in addresses for s in sizes]),),
        allocated=4096,
    )
    assert totals[CATEGORY_GRADIENTS] == 4096


def test_the_fp32_main_params_own_gradients_are_counted():
    """mcore sets main_param.grad = model_param.main_grad.float().

    Against a bf16 grad buffer that is a whole extra fp32 shard, and walking
    only the model's parameters would leave every byte of it in `other`.
    """
    model_param = SimpleNamespace(grad=None, main_grad=_Tensor(1, 128))
    main_param = SimpleNamespace(grad=_Tensor(2, 256), main_grad=None)
    found = list(_gradients([model_param, main_param]))
    assert model_param.main_grad in found
    assert main_param.grad in found


def test_the_precision_aware_decoupled_grad_slot_is_counted():
    """The precision-aware optimizer reads decoupled_grad in place of .grad."""
    param = SimpleNamespace(grad=None, main_grad=None, decoupled_grad=_Tensor(1, 64))
    totals = attribute_device_memory(
        ((CATEGORY_GRADIENTS, _gradients([param])),), allocated=64
    )
    assert totals[CATEGORY_GRADIENTS] == 64


def test_a_real_dtensor_is_measured_as_its_local_shard():
    """FSDP2's parameters are DTensors, whose outer storage cannot be read.

    Reproduces the real failure rather than a stand-in: ``nbytes()`` answers
    plausibly but ``data_ptr()`` raises, so the walk's except clause would skip
    the tensor and charge every FSDP2 parameter to ``other``.
    """
    dist = pytest.importorskip("torch.distributed")
    if not dist.is_available():
        pytest.skip("torch.distributed unavailable")
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    already_up = dist.is_initialized()
    if not already_up:
        dist.init_process_group("gloo", store=dist.HashStore(), rank=0, world_size=1)
    try:
        dtensor = distribute_tensor(
            torch.zeros(64, 64), init_device_mesh("cpu", (1,)), [Shard(0)]
        )
        with pytest.raises(RuntimeError):
            dtensor.untyped_storage().data_ptr()

        local = _local_tensor(dtensor)
        assert local is not dtensor
        assert local.untyped_storage().nbytes() == 64 * 64 * 4
        assert local.untyped_storage().data_ptr() > 0
    finally:
        if not already_up:
            dist.destroy_process_group()


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


class _ProxyDict:
    """mcore's ChainedOptimizer.state: items(), but deliberately no values()."""

    def __init__(self, inner):
        self._inner = inner

    def items(self):
        for index, dict_ in enumerate(self._inner):
            for key, value in dict_.items():
                yield (index, key), value


def _optimizer(state=None, groups=None, chained=None):
    kwargs = {"state": state or {}, "param_groups": groups or []}
    if chained is not None:
        kwargs["chained_optimizers"] = chained
    return SimpleNamespace(**kwargs)


def test_optimizer_moments_are_collected():
    moments = {"exp_avg": _Tensor(1, 64), "exp_avg_sq": _Tensor(2, 64), "step": 7}
    found = list(_optimizer_state_tensors(_optimizer(state={"p": moments})))
    assert found.count(moments["exp_avg"]) == 1
    assert moments["exp_avg_sq"] in found


def test_the_fp32_main_copy_is_collected_from_param_groups():
    """mcore's distributed optimizer swaps main params into param_groups.

    Only the moments live in .state; the sharded fp32 copy they are moments of
    is in the group, so reading .state alone misses it entirely.
    """
    main_param = _Tensor(9, 4096)
    opt = _optimizer(
        state={"main": {"exp_avg": _Tensor(1, 2048)}},
        groups=[{"params": [main_param], "lr": 0.1}],
    )
    assert main_param in _optimizer_params(opt)
    # Not in .state -- that is exactly why reading it alone missed the copy.
    assert main_param not in list(_optimizer_state_tensors(opt))


def test_a_plain_optimizers_groups_do_not_double_count_the_model():
    """For torch optimizers param_groups hold the model's own parameters."""
    weight = _Tensor(3, 512)
    opt = _optimizer(groups=[{"params": [weight]}])
    totals = attribute_device_memory(
        (
            (CATEGORY_PARAMETERS, [weight]),
            (CATEGORY_OPTIMIZER_STATE, _optimizer_params(opt)),
        ),
        allocated=512,
    )
    assert totals[CATEGORY_PARAMETERS] == 512
    assert totals[CATEGORY_OPTIMIZER_STATE] == 0


def test_chained_optimizers_are_walked_through_their_links():
    first, second = _Tensor(1, 32), _Tensor(2, 32)
    chained = _optimizer(
        chained=[
            _optimizer(state={"a": {"exp_avg": first}}),
            _optimizer(groups=[{"params": [second]}]),
        ]
    )
    found = list(_optimizer_state_tensors(chained)) + _optimizer_params(chained)
    assert first in found and second in found


def test_an_empty_chained_optimizer_reports_nothing_rather_than_raising():
    """A rank with no trainable parameters gets an empty chain from mcore.

    Treating empty as "not chained" would fall back to the chained optimizer
    itself, whose .state is a ProxyDict with no values() -- losing the whole
    measurement on exactly the ranks with the least to report.
    """
    empty = SimpleNamespace(
        chained_optimizers=[],
        state=_ProxyDict([{"a": {"exp_avg": _Tensor(1, 8)}}]),
    )
    assert list(_optimizer_state_tensors(empty)) == []
    assert _optimizer_params(empty) == []


def test_a_state_map_without_values_is_skipped_not_raised():
    proxy = SimpleNamespace(state=_ProxyDict([{"a": {"exp_avg": _Tensor(1, 8)}}]))
    assert list(_optimizer_state_tensors(proxy)) == []


def test_device_memory_bytes_partitions_what_torch_reports(monkeypatch):
    """The three walks and the residual, assembled, without a GPU."""
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 1000)
    weight = _Tensor(1, 400)
    parameter = SimpleNamespace(grad=_Tensor(2, 300), main_grad=None)
    monkeypatch.setattr(
        "nemo_rl.telemetry.device_memory._parameters",
        lambda model: [weight, parameter],
    )
    totals = device_memory_bytes(
        object(), _optimizer(state={"p": {"exp_avg": _Tensor(3, 200)}})
    )
    assert totals == {
        CATEGORY_PARAMETERS: 400,
        CATEGORY_GRADIENTS: 300,
        CATEGORY_OPTIMIZER_STATE: 200,
        CATEGORY_OTHER: 100,
    }


def test_the_mcore_shape_lands_in_the_right_categories(monkeypatch):
    """A distributed-optimizer rank end to end, with every slot mcore uses.

    bf16 weight and grad buffer on the model side; fp32 main copy and its own
    fp32 grad on the optimizer side. Before the optimizer's parameters were
    walked, the last two were 768 bytes of `other`.
    """
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 2500)

    weight = _Tensor(1, 256)  # bf16 weight
    weight.grad = None  # mcore leaves .grad unset
    weight.main_grad = _Tensor(2, 256)  # the DDP bucket

    main_param = _Tensor(3, 1024)  # fp32 main copy, in param_groups
    main_param.grad = _Tensor(4, 512)  # its own fp32 grad shard
    main_param.main_grad = None

    monkeypatch.setattr(
        "nemo_rl.telemetry.device_memory._parameters", lambda model: [weight]
    )
    totals = device_memory_bytes(
        object(),
        _optimizer(
            state={"m": {"exp_avg": _Tensor(5, 128)}},
            groups=[{"params": [main_param]}],
        ),
    )
    assert totals[CATEGORY_PARAMETERS] == 256
    assert totals[CATEGORY_GRADIENTS] == 256 + 512  # bucket + fp32 shard
    assert totals[CATEGORY_OPTIMIZER_STATE] == 128 + 1024  # moment + main copy
    assert totals[CATEGORY_OTHER] == 2500 - 2176
    assert sum(totals.values()) == 2500


def test_the_config_field_reaches_a_worker_as_an_env_var():
    """The flag only matters if _config_to_env carries it across the Ray gap."""
    from nemo_rl.telemetry.config import TelemetryConfig
    from nemo_rl.telemetry.setup import _config_to_env, device_memory_requested

    assert TelemetryConfig().device_memory_enabled is False
    _config_to_env(TelemetryConfig(enabled=True, device_memory_enabled=True))
    assert os.environ["NEMO_RL_OTEL_DEVICE_MEMORY"] == "1"
    assert device_memory_requested() is True


def test_the_master_switch_turns_the_walk_off(monkeypatch):
    """telemetry.enabled=false must stop it, not merely drop its points.

    _config_to_env exports every field before init_telemetry_driver checks
    `enabled`, so the opt-in survives a disabled run; without this gate the
    model would be walked every step for points the recording path discards.
    """
    monkeypatch.setenv("NEMO_RL_OTEL_DEVICE_MEMORY", "1")
    monkeypatch.delenv("NEMO_RL_OTEL_ENABLED", raising=False)
    monkeypatch.setattr(
        "nemo_rl.telemetry.device_memory.device_memory_bytes",
        lambda model, optimizer: pytest.fail("walked while telemetry was off"),
    )
    record_device_memory(object(), object())


@pytest.fixture
def opted_in(monkeypatch):
    """Switch the collection on, as a run config or a raw env var would.

    Both switches: the collector honours the master switch as well as its own.
    """
    monkeypatch.setenv("NEMO_RL_OTEL_ENABLED", "1")
    monkeypatch.setenv("NEMO_RL_OTEL_DEVICE_MEMORY", "1")


def test_collection_is_off_until_asked_for(monkeypatch):
    """Default off, and the gate comes before the walk, not after it."""
    monkeypatch.setenv("NEMO_RL_OTEL_ENABLED", "1")
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
    monkeypatch.setenv("NEMO_RL_OTEL_ENABLED", "1")
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
