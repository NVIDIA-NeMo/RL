import contextlib
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager

import pytest
import torch

snap_cache = pytest.importorskip(
    "nemo_rl.modelopt.models.policy.workers.snap_cache",
    reason="Requires ModelOpt",
)

pytestmark = pytest.mark.mcore


class _FakeQuantizer(torch.nn.Module):
    """Small deterministic quantizer used to exercise the cache lifecycle."""

    def __init__(
        self,
        *,
        offset: float = 0.25,
        enabled: bool = True,
        fail: bool = False,
    ) -> None:
        super().__init__()
        self.offset = offset
        self.is_enabled = enabled
        self.fail = fail
        self.calls = 0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        if self.fail:
            raise RuntimeError("snap failed")
        if not self.is_enabled:
            return inputs
        return inputs + self.offset


class _WeightModule(torch.nn.Module):
    """Module exposing the attributes discovered by the snapshot helpers."""

    def __init__(
        self,
        quantizer: _FakeQuantizer,
        *,
        activation_quantized: bool = False,
    ) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        self.weight_quantizer = quantizer
        if activation_quantized:
            self.input_quantizer = _FakeQuantizer()


class _WeightModel(torch.nn.Module):
    def __init__(self, *modules: _WeightModule) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(modules)


def _install_fake_tensor_quantizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Route the global forward shim through the deterministic test quantizer."""
    # _install_patch mutates this class directly; register its original method so
    # pytest restores it after the test along with the module globals below.
    monkeypatch.setattr(_FakeQuantizer, "forward", _FakeQuantizer.forward)
    monkeypatch.setattr(snap_cache, "TensorQuantizer", _FakeQuantizer)
    monkeypatch.setattr(snap_cache, "_patch_installed", False)


def test_weight_snap_cache_is_scoped_and_invalidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_tensor_quantizer(monkeypatch)
    cached_quantizer = _FakeQuantizer()
    other_quantizer = _FakeQuantizer()
    cached_module = _WeightModule(cached_quantizer)
    other_module = _WeightModule(other_quantizer)

    with snap_cache.weight_snap_cache(_WeightModel(cached_module)):
        with torch.no_grad():
            first = cached_quantizer(cached_module.weight)
            second = cached_quantizer(cached_module.weight)
            other_quantizer(other_module.weight)
            other_quantizer(other_module.weight)

        assert torch.equal(first, second)
        assert cached_quantizer.calls == 1
        assert other_quantizer.calls == 2

        with torch.enable_grad():
            cached_quantizer(cached_module.weight)
        assert cached_quantizer.calls == 2

        with torch.no_grad():
            cached_module.weight.add_(1.0)
            cached_quantizer(cached_module.weight)
        assert cached_quantizer.calls == 3

    with torch.no_grad():
        cached_quantizer(cached_module.weight)
    assert cached_quantizer.calls == 4
    assert not hasattr(cached_quantizer, "_nrl_cache_ok")
    assert not hasattr(cached_quantizer, "_nrl_snap_cache")


def test_materialized_snap_preserves_output_storage_and_restores_on_error() -> None:
    quantizer = _FakeQuantizer()
    module = _WeightModule(quantizer)
    model = _WeightModel(module)
    original_weight = module.weight.detach().clone()
    original_data_ptr = module.weight.data_ptr()
    inputs = torch.tensor([[2.0, -1.0]])
    snapped_weight = quantizer(original_weight)
    expected = torch.nn.functional.linear(inputs, snapped_weight)

    with pytest.raises(RuntimeError, match="body failed"):
        with snap_cache.materialized_weight_snap(model):
            assert module.weight.data_ptr() == original_data_ptr
            assert torch.equal(module.weight, snapped_weight)
            actual = torch.nn.functional.linear(inputs, module.weight)
            assert torch.equal(actual, expected)
            raise RuntimeError("body failed")

    assert module.weight.data_ptr() == original_data_ptr
    assert torch.equal(module.weight, original_weight)


def test_materialized_snap_rolls_back_partial_setup() -> None:
    first = _WeightModule(_FakeQuantizer())
    second = _WeightModule(_FakeQuantizer(fail=True))
    model = _WeightModel(first, second)
    original_first = first.weight.detach().clone()
    original_second = second.weight.detach().clone()

    with pytest.raises(RuntimeError, match="snap failed"):
        with snap_cache.materialized_weight_snap(model):
            pytest.fail("setup failure must prevent context entry")

    assert torch.equal(first.weight, original_first)
    assert torch.equal(second.weight, original_second)


def test_materialized_snap_refuses_activation_quantization_before_mutation() -> None:
    quantizer = _FakeQuantizer()
    module = _WeightModule(quantizer, activation_quantized=True)
    original_weight = module.weight.detach().clone()

    with pytest.raises(ValueError, match="requires weight-only quantization"):
        with snap_cache.materialized_weight_snap(_WeightModel(module)):
            pytest.fail("W4A4 must be rejected before context entry")

    assert quantizer.calls == 0
    assert torch.equal(module.weight, original_weight)


def test_shadow_forward_restores_class_after_error() -> None:
    class _Original:
        def forward(self) -> str:
            return "original"

    class _Wrapper:
        def forward(self) -> str:
            return "wrapped"

    class _Converted(_Wrapper, _Original):
        pass

    with pytest.raises(RuntimeError, match="body failed"):
        with contextlib.ExitStack() as cleanup:
            assert snap_cache._shadow_forward_after(
                _Converted,
                _Wrapper,
                cleanup,
            )
            assert _Converted.forward is _Original.forward
            raise RuntimeError("body failed")

    assert "forward" not in _Converted.__dict__
    assert _Converted().forward() == "wrapped"


@pytest.mark.parametrize(
    ("config", "expected_events"),
    [
        ({}, ["base"]),
        (
            {"quant_cache_frozen_weight_snap": True},
            ["cache_enter", "base", "cache_exit"],
        ),
        (
            {
                "quant_cache_frozen_weight_snap": True,
                "quant_materialize_frozen_weight_snap": True,
            },
            [
                "plain_enter",
                "materialize_enter",
                "base",
                "materialize_exit",
                "plain_exit",
            ],
        ),
    ],
)
def test_quant_worker_routes_logprobs_through_selected_snap_mode(
    monkeypatch: pytest.MonkeyPatch,
    config: dict[str, bool],
    expected_events: list[str],
) -> None:
    worker_module = pytest.importorskip(
        "nemo_rl.modelopt.models.policy.workers.megatron_quant_policy_worker",
        reason="Requires Megatron and Ray",
    )
    events: list[str] = []

    def recording_context(
        name: str,
    ) -> Callable[..., AbstractContextManager[None]]:
        @contextlib.contextmanager
        def manager(*args: object, **kwargs: object) -> Iterator[None]:
            events.append(f"{name}_enter")
            try:
                yield
            finally:
                events.append(f"{name}_exit")

        return manager

    def base_get_logprobs(
        self: object,
        *args: object,
        **kwargs: object,
    ) -> str:
        events.append("base")
        return "result"

    monkeypatch.setattr(
        snap_cache,
        "plain_module_attr_lookup",
        recording_context("plain"),
    )
    monkeypatch.setattr(
        snap_cache,
        "materialized_weight_snap",
        recording_context("materialize"),
    )
    monkeypatch.setattr(
        snap_cache,
        "weight_snap_cache",
        recording_context("cache"),
    )
    monkeypatch.setattr(
        worker_module.MegatronPolicyWorkerImpl,
        "get_logprobs",
        base_get_logprobs,
    )

    worker_class = (
        worker_module.MegatronQuantPolicyWorker.__ray_metadata__.modified_class
    )
    worker = object.__new__(worker_class)
    worker.cfg = config
    worker.model = object()
    worker.rank = 0

    assert worker.get_logprobs() == "result"
    assert events == expected_events
