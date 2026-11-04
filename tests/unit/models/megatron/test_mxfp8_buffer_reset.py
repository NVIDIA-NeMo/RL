"""Poisoned transport storage, live sibling payloads, and accumulated gradients."""

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import torch

ROOT = Path(__file__).parents[4]
ENV = "NRL_MXFP8_RESET_SHARED_BUFFER"
Runtime = tuple[types.ModuleType, type, type, type]


def test_driver_forwards_export_without_mutating_config(
    runtime: Runtime, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, *_ = runtime
    monkeypatch.setenv(ENV, "1")
    config_env = {"KEEP": "value"}
    assert module.mxfp8_buffer_reset_env_vars(config_env) == {"KEEP": "value", ENV: "1"}
    assert config_env == {"KEEP": "value"}


def test_explicit_worker_setting_wins_over_export(
    runtime: Runtime, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, *_ = runtime
    monkeypatch.setenv(ENV, "1")
    assert module.mxfp8_buffer_reset_env_vars({ENV: "0"}) == {ENV: "0"}


def test_driver_rejects_invalid_export(
    runtime: Runtime, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, *_ = runtime
    monkeypatch.setenv(ENV, "yes")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        module.mxfp8_buffer_reset_env_vars(None)


def test_driver_leaves_unset_switch_absent(runtime: Runtime) -> None:
    module, *_ = runtime
    assert module.mxfp8_buffer_reset_env_vars(None) == {}


def test_cuda_reset_orders_synchronization_around_clear(
    runtime: Runtime, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, *_ = runtime
    events = []

    class Tensor:
        is_cuda = True
        device = torch.device("cuda:0")

        def zero_(self) -> None:
            events.append("clear")

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: events.append("sync"))
    module.reset_mxfp8_shared_buffer(Tensor())
    assert events == ["sync", "clear", "sync"]


def test_cuda_capture_rejected_before_clear(
    runtime: Runtime, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, *_ = runtime

    class Tensor:
        is_cuda = True

        def zero_(self) -> None:
            pytest.fail("captured buffer was cleared")

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="CUDA graph capture"):
        module.reset_mxfp8_shared_buffer(Tensor())


@pytest.fixture
def runtime(monkeypatch: pytest.MonkeyPatch) -> Runtime:
    spec = importlib.util.spec_from_file_location(
        "buffer_reset_under_test",
        ROOT / "nemo_rl/models/megatron/mxfp8_buffer_reset.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.delenv(ENV, raising=False)

    class Optimizer:
        def _copy_main_params_to_param_buffer(self) -> None:
            self.before_staging = self.buffers[0].shared_buffer.clone()
            self.buffers[0].param_data[: self.master.numel()].copy_(self.master)

    class Group:
        def _post_param_sync(self) -> None:
            for bucket in self.buckets:
                self.consumed.append(bucket.param_data.clone())

    class DDP:
        def zero_grad_buffer(self) -> None:
            # Deliberately leave storage poisoned: the extra reset must clean it.
            self.original_called = True

    for name, attrs in [
        (
            "megatron.core.distributed.distributed_data_parallel",
            {"DistributedDataParallel": DDP},
        ),
        (
            "megatron.core.distributed.param_and_grad_buffer",
            {"_ParamAndGradBucketGroup": Group},
        ),
        (
            "megatron.core.optimizer.distrib_optimizer",
            {"DistributedOptimizer": Optimizer},
        ),
        (
            "megatron.core.fp8_utils",
            {
                "is_mxfp8tensor": lambda p: p == "mxfp8",
                "is_grouped_mxfp8tensor": lambda p: p == "grouped_mxfp8",
            },
        ),
    ]:
        fake = types.ModuleType(name)
        fake.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, fake)
    return module, Optimizer, Group, DDP


def config(*, reuse: bool = True) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        reuse_grad_buf_for_mxfp8_param_ag=reuse, use_distributed_optimizer=True
    )


def group_instance(cls: type, bucket: Any, cfg: types.SimpleNamespace) -> Any:
    group = cls()
    group.ddp_config = cfg
    group.buckets = [bucket]
    group.param_gather_handle = None
    group.grad_reduce_handle = None
    group.consumed = []
    return group


def buffer_instance(*, grad_dtype: torch.dtype) -> types.SimpleNamespace:
    shared = torch.full((16,), float("nan"), dtype=grad_dtype)
    param = shared.view(torch.bfloat16)[:16]
    return types.SimpleNamespace(
        shared_buffer=shared,
        grad_data=shared,
        param_data=param,
        ddp_config=config(),
        params=["mxfp8"],
    )


@pytest.mark.parametrize("setting", [None, "0"])
def test_disabled_changes_no_methods(
    runtime: Runtime, monkeypatch: pytest.MonkeyPatch, setting: str | None
) -> None:
    module, optimizer, group, ddp = runtime
    original = (
        optimizer._copy_main_params_to_param_buffer,
        group._post_param_sync,
        ddp.zero_grad_buffer,
    )
    if setting is not None:
        monkeypatch.setenv(ENV, setting)
    assert module.install_mxfp8_buffer_reset() is False
    assert original == (
        optimizer._copy_main_params_to_param_buffer,
        group._post_param_sync,
        ddp.zero_grad_buffer,
    )


def test_invalid_switch_fails_before_install(
    runtime: Runtime, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, *_ = runtime
    monkeypatch.setenv(ENV, "true")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        module.install_mxfp8_buffer_reset()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_poison_removed_before_master_staging_and_master_payload_retained(
    runtime: Runtime, monkeypatch: pytest.MonkeyPatch, dtype: torch.dtype
) -> None:
    module, optimizer_class, group_class, _ = runtime
    monkeypatch.setenv(ENV, "1")
    assert module.install_mxfp8_buffer_reset()
    first = optimizer_class._copy_main_params_to_param_buffer
    assert module.install_mxfp8_buffer_reset()
    assert optimizer_class._copy_main_params_to_param_buffer is first
    buffer = buffer_instance(grad_dtype=dtype)
    bucket = types.SimpleNamespace(
        params={"mxfp8"}, param_data=buffer.param_data, grad_data=buffer.grad_data
    )
    group = group_instance(group_class, bucket, buffer.ddp_config)
    optimizer = optimizer_class()
    optimizer.config = buffer.ddp_config
    optimizer.is_stub_optimizer = False
    optimizer.buffers = [buffer]
    optimizer.per_model_bucket_groups = {0: [group]}
    optimizer.master = torch.arange(1, 9, dtype=torch.float32)
    optimizer._copy_main_params_to_param_buffer()
    assert torch.count_nonzero(optimizer.before_staging) == 0
    assert torch.equal(buffer.param_data[:8], optimizer.master.bfloat16())
    assert torch.count_nonzero(buffer.param_data[8:]) == 0
    group._post_param_sync()
    assert torch.equal(group.consumed[0][:8], optimizer.master.bfloat16())
    assert torch.count_nonzero(buffer.shared_buffer) == 0


def test_fp32_gradient_view_does_not_erase_sibling_bf16_payload(
    runtime: Runtime, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, _, group_class, _ = runtime
    monkeypatch.setenv(ENV, "1")
    module.install_mxfp8_buffer_reset()
    shared = torch.zeros(16, dtype=torch.float32)
    params = shared.view(torch.bfloat16)[:16]
    params[:8] = 2
    params[8:] = 7
    # The first bucket's full FP32 gradient view aliases BOTH BF16 payloads.
    bucket = types.SimpleNamespace(
        params={"mxfp8"}, param_data=params[:8], grad_data=shared[:8]
    )
    group = group_instance(group_class, bucket, config())
    group._post_param_sync()
    assert torch.equal(group.consumed[0], torch.full((8,), 2, dtype=torch.bfloat16))
    assert torch.count_nonzero(params[:8]) == 0
    assert torch.equal(params[8:], torch.full((8,), 7, dtype=torch.bfloat16))


@pytest.mark.parametrize("handle", ["param_gather_handle", "grad_reduce_handle"])
def test_inflight_collective_rejected_before_any_clear(
    runtime: Runtime, monkeypatch: pytest.MonkeyPatch, handle: str
) -> None:
    module, optimizer_class, group_class, _ = runtime
    monkeypatch.setenv(ENV, "1")
    module.install_mxfp8_buffer_reset()
    buffer = buffer_instance(grad_dtype=torch.bfloat16)
    buffer.shared_buffer.fill_(9)
    group = group_instance(group_class, None, config())
    setattr(group, handle, object())
    optimizer = optimizer_class()
    optimizer.config = config()
    optimizer.is_stub_optimizer = False
    optimizer.buffers = [buffer]
    optimizer.per_model_bucket_groups = {0: [group]}
    with pytest.raises(RuntimeError, match="outstanding"):
        optimizer._copy_main_params_to_param_buffer()
    assert torch.equal(buffer.shared_buffer, torch.full((16,), 9, dtype=torch.bfloat16))


def test_two_microbatches_retain_accumulated_gradients(
    runtime: Runtime, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, _, group_class, ddp_class = runtime
    monkeypatch.setenv(ENV, "1")
    module.install_mxfp8_buffer_reset()
    buffer = buffer_instance(grad_dtype=torch.float32)
    group = group_instance(group_class, None, config())
    ddp = ddp_class()
    ddp.ddp_config = config()
    ddp.buffers, ddp.expert_parallel_buffers = [buffer], []
    ddp.bucket_groups, ddp.expert_parallel_bucket_groups = [group], []
    ddp.zero_grad_buffer()
    assert ddp.original_called and torch.count_nonzero(buffer.shared_buffer) == 0
    weight = torch.arange(16, dtype=torch.float32).reshape(4, 4).requires_grad_()
    xs = [torch.tensor([[1.0, 2.0, 3.0, 4.0]]), torch.tensor([[4.0, 1.0, 2.0, 3.0]])]
    for x in xs:
        grad = torch.autograd.grad((x @ weight).square().sum(), weight)[0]
        buffer.grad_data.add_(grad.flatten())
    reference = torch.autograd.grad(
        sum((x @ weight).square().sum() for x in xs), weight
    )[0]
    assert torch.equal(buffer.grad_data, reference.flatten())


def test_mixed_bucket_is_rejected_without_erasing_model_weights(
    runtime: Runtime, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, _, group_class, _ = runtime
    monkeypatch.setenv(ENV, "1")
    module.install_mxfp8_buffer_reset()
    shared = torch.full((8,), 3, dtype=torch.bfloat16)
    bucket = types.SimpleNamespace(
        params={"mxfp8", "bf16"}, param_data=shared, grad_data=shared
    )
    group = group_instance(group_class, bucket, config())
    with pytest.raises(RuntimeError, match="mixed"):
        group._post_param_sync()
    assert torch.equal(shared, torch.full((8,), 3, dtype=torch.bfloat16))


def test_bf16_baseline_unchanged_when_switch_enabled(
    runtime: Runtime, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, _, group_class, _ = runtime
    monkeypatch.setenv(ENV, "1")
    module.install_mxfp8_buffer_reset()
    shared = torch.full((8,), 5, dtype=torch.bfloat16)
    bucket = types.SimpleNamespace(params={"bf16"}, param_data=shared, grad_data=shared)
    group = group_instance(group_class, bucket, config(reuse=False))
    group._post_param_sync()
    assert torch.equal(shared, torch.full((8,), 5, dtype=torch.bfloat16))
