"""Reference snapshot mapping must follow parameter identity, not name suffixes."""

from contextlib import contextmanager

import pytest
import torch

from nemo_rl.models.policy.reference_snapshot import borrowed_cpu_parameter_views


class Buffer:
    def __init__(self, owner: torch.nn.Parameter) -> None:
        self.owner = owner
        self.cpu = torch.empty_like(owner, device="cpu")
        self.active = False

    @contextmanager
    def borrow_cpu_param_snapshot(self):
        self.cpu.copy_(self.owner.detach())
        self.active = True
        try:
            yield {self.owner: self.cpu}
        finally:
            self.active = False


def test_identity_aliases_and_freshness() -> None:
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    model.alias = model.weight
    buffer = Buffer(model.weight)
    for value in (3.0, 7.0):
        with torch.no_grad():
            model.weight.fill_(value)
        with borrowed_cpu_parameter_views(model, [buffer]) as views:
            assert buffer.active
            assert set(views) == {"weight", "alias"}
            assert views["weight"].data_ptr() == buffer.cpu.data_ptr()
            assert views["alias"].data_ptr() == buffer.cpu.data_ptr()
            torch.testing.assert_close(views["weight"], torch.full_like(buffer.cpu, value))
        assert not buffer.active


def test_unsupported_buffer_falls_back_without_hiding_errors() -> None:
    model = torch.nn.Linear(2, 2, bias=False)

    class Unsupported:
        @contextmanager
        def borrow_cpu_param_snapshot(self):
            raise NotImplementedError("quantized storage")
            yield

    with borrowed_cpu_parameter_views(model, [object(), Unsupported()]) as views:
        assert views == {}

    class Broken:
        @contextmanager
        def borrow_cpu_param_snapshot(self):
            raise RuntimeError("invalid ownership")
            yield

    with pytest.raises(RuntimeError, match="invalid ownership"):
        with borrowed_cpu_parameter_views(model, [Broken()]):
            pass


def test_release_on_body_exception() -> None:
    model = torch.nn.Linear(2, 2, bias=False)
    buffer = Buffer(model.weight)
    with pytest.raises(ValueError, match="reference failure"):
        with borrowed_cpu_parameter_views(model, [buffer]):
            raise ValueError("reference failure")
    assert not buffer.active


def test_unknown_state_keys_are_not_guessed() -> None:
    model = torch.nn.Linear(2, 2, bias=False)
    unrelated_owner = torch.nn.Parameter(torch.ones_like(model.weight))
    with borrowed_cpu_parameter_views(model, [Buffer(unrelated_owner)]) as views:
        assert views == {}
