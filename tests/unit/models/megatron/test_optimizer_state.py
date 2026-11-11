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
"""CPU tensor checks for scoped precision-aware checkpoint restoration."""

import sys
from types import SimpleNamespace

import pytest
import torch

from nemo_rl.models.megatron.optimizer_state import precision_aware_state_loading


class FusedAdamFixture(torch.optim.Optimizer):
    def __init__(self, params: list[torch.Tensor]) -> None:
        super().__init__(params, {"lr": 3e-6, "betas": (0.9, 0.99)})
        self.master_weights = True
        self.store_param_remainders = True
        self.name_to_dtype_map = dict.fromkeys(
            ("exp_avg", "exp_avg_sq", "master_param"), torch.float32
        )

    def load_state_dict(self, state_dict: dict) -> None:
        raise RuntimeError("original TE loader")


@pytest.fixture
def checkpoint(monkeypatch: pytest.MonkeyPatch) -> tuple:
    monkeypatch.setitem(
        sys.modules,
        "transformer_engine.pytorch.optimizers",
        SimpleNamespace(FusedAdam=FusedAdamFixture),
    )
    params = [
        torch.nn.Parameter(torch.arange(4, dtype=torch.bfloat16)) for _ in range(2)
    ]
    inner = FusedAdamFixture(params)
    outer = SimpleNamespace(
        config=SimpleNamespace(
            use_precision_aware_optimizer=True, optimizer_cpu_offload=False
        ),
        ddp_config=SimpleNamespace(use_megatron_fsdp=False),
        optimizer=inner,
    )
    saved = inner.state_dict()
    saved["param_groups"][0]["step"] = 10
    saved["param_groups"][0]["lr"] = 7e-6
    saved["state"] = {
        i: {
            "exp_avg": torch.tensor([0.10001, 0.20002, 0.30003, 0.40004]),
            "exp_avg_sq": torch.arange(4, dtype=torch.float32),
            "master_param": torch.tensor([-32768, -1, 1, 32767], dtype=torch.int16),
        }
        for i in range(2)
    }
    return outer, inner, params, saved


def test_restore_preserves_values_metadata_and_storage(checkpoint: tuple) -> None:
    outer, inner, params, saved = checkpoint
    before = {
        i: {k: v.clone() for k, v in values.items()}
        for i, values in saved["state"].items()
    }
    with precision_aware_state_loading(outer):
        inner.load_state_dict(saved)
        assert inner.param_groups[0]["step"] == 10
        assert inner.param_groups[0]["lr"] == 7e-6
        for i, param in enumerate(params):
            assert inner.state[param] is not saved["state"][i]
            for key, tensor in inner.state[param].items():
                assert tensor is saved["state"][i][key]
                assert torch.equal(tensor, before[i][key])
    assert "load_state_dict" not in inner.__dict__
    with pytest.raises(RuntimeError, match="original TE loader"):
        inner.load_state_dict(saved)


@pytest.mark.parametrize(
    "variant",
    [
        "dtype",
        "shape",
        "extra",
        "missing",
        "groups",
        "duplicate",
        "remainder",
        "strided",
    ],
)
def test_incompatible_checkpoint_uses_original_loader(
    checkpoint: tuple, variant: str
) -> None:
    outer, inner, params, saved = checkpoint
    if variant == "dtype":
        saved["state"][0]["exp_avg"] = saved["state"][0]["exp_avg"].bfloat16()
    elif variant == "shape":
        saved["state"][0]["exp_avg"] = torch.ones(8)
    elif variant == "extra":
        saved["state"]["other"] = {"metadata": 1}
    elif variant == "missing":
        del saved["state"][0]["exp_avg_sq"]
    elif variant == "groups":
        saved["param_groups"] = []
    elif variant == "strided":
        saved["state"][0]["exp_avg"] = torch.ones(8)[::2]
    elif variant == "duplicate":
        saved["param_groups"][0]["params"] = [0, 0]
    else:
        saved["state"][0]["master_param"] = torch.ones(4)
    with pytest.raises(RuntimeError, match="original TE loader"):
        with precision_aware_state_loading(outer):
            inner.load_state_dict(saved)
    assert "load_state_dict" not in inner.__dict__
    assert not inner.state


@pytest.mark.parametrize(
    "variant",
    [
        "precision",
        "offload",
        "fsdp",
        "master",
        "remainder",
        "moment",
        "param",
        "pre_hook",
        "post_hook",
    ],
)
def test_incompatible_optimizer_is_not_modified(
    checkpoint: tuple, variant: str
) -> None:
    outer, inner, params, saved = checkpoint
    if variant == "precision":
        outer.config.use_precision_aware_optimizer = False
    elif variant == "offload":
        outer.config.optimizer_cpu_offload = True
    elif variant == "fsdp":
        outer.ddp_config.use_megatron_fsdp = True
    elif variant == "master":
        inner.master_weights = False
    elif variant == "remainder":
        inner.store_param_remainders = False
    elif variant == "moment":
        inner.name_to_dtype_map["exp_avg"] = torch.bfloat16
    elif variant == "param":
        params[0].data = params[0].data.float()
    elif variant == "pre_hook":
        inner.register_load_state_dict_pre_hook(lambda *args: None)
    else:
        inner.register_load_state_dict_post_hook(lambda *args: None)
    with precision_aware_state_loading(outer):
        assert "load_state_dict" not in inner.__dict__
    with pytest.raises(RuntimeError, match="original TE loader"):
        inner.load_state_dict(saved)


def test_context_failure_restores_loader(checkpoint: tuple) -> None:
    outer, inner, _, _ = checkpoint
    with pytest.raises(ValueError, match="checkpoint read failed"):
        with precision_aware_state_loading(SimpleNamespace(chained_optimizers=[outer])):
            assert "load_state_dict" in inner.__dict__
            raise ValueError("checkpoint read failed")
    assert "load_state_dict" not in inner.__dict__


def test_no_optimizer() -> None:
    with precision_aware_state_loading(None):
        pass


def test_hook_added_during_context_retains_original_loader(checkpoint: tuple) -> None:
    outer, inner, _, saved = checkpoint
    with precision_aware_state_loading(outer):
        handle = inner.register_load_state_dict_pre_hook(lambda *args: None)
        try:
            with pytest.raises(RuntimeError, match="original TE loader"):
                inner.load_state_dict(saved)
        finally:
            handle.remove()
    assert "load_state_dict" not in inner.__dict__


def test_existing_instance_override_is_preserved(checkpoint: tuple) -> None:
    outer, inner, _, saved = checkpoint

    def existing(state_dict: dict) -> None:
        raise LookupError("existing override")

    inner.load_state_dict = existing
    with precision_aware_state_loading(outer):
        assert inner.load_state_dict is existing
    assert inner.load_state_dict is existing
    with pytest.raises(LookupError, match="existing override"):
        inner.load_state_dict(saved)


def test_stub_optimizer_does_not_require_state() -> None:
    with precision_aware_state_loading(SimpleNamespace(is_stub_optimizer=True)):
        pass


def test_chained_optimizers_keep_separate_fallbacks(
    checkpoint: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    outer, inner, params, saved = checkpoint
    second = FusedAdamFixture(
        [torch.nn.Parameter(param.detach().clone()) for param in params]
    )
    second_outer = SimpleNamespace(
        config=outer.config, ddp_config=outer.ddp_config, optimizer=second
    )
    called = []

    def fallback(target: torch.optim.Optimizer, state_dict: dict) -> None:
        called.append(target)
        raise LookupError("fallback")

    monkeypatch.setattr(FusedAdamFixture, "load_state_dict", fallback)
    with precision_aware_state_loading(
        SimpleNamespace(chained_optimizers=[outer, second_outer])
    ):
        inner.load_state_dict(saved)
        second.load_state_dict(saved)
        for target in (inner, second):
            with pytest.raises(LookupError, match="fallback"):
                target.load_state_dict({"state": {}, "param_groups": []})
    assert called == [inner, second]
    assert "load_state_dict" not in inner.__dict__
    assert "load_state_dict" not in second.__dict__


def test_subclass_loader_is_not_replaced(checkpoint: tuple) -> None:
    outer, _, params, _ = checkpoint

    class CustomAdam(FusedAdamFixture):
        pass

    inner = CustomAdam(params)
    outer.optimizer = inner
    with precision_aware_state_loading(outer):
        assert "load_state_dict" not in inner.__dict__


def test_malformed_groups_retain_pytorch_validation(
    checkpoint: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    outer, inner, _, saved = checkpoint
    monkeypatch.setattr(
        FusedAdamFixture, "load_state_dict", torch.optim.Optimizer.load_state_dict
    )
    saved["param_groups"] = []
    with pytest.raises(ValueError, match="parameter groups"):
        with precision_aware_state_loading(outer):
            inner.load_state_dict(saved)
    assert "load_state_dict" not in inner.__dict__


def test_consumed_tensor_storage_remains_shared(checkpoint: tuple) -> None:
    outer, inner, params, saved = checkpoint
    with precision_aware_state_loading(outer):
        inner.load_state_dict(saved)
    inner.state[params[0]]["exp_avg"].add_(1)
    assert torch.equal(saved["state"][0]["exp_avg"], inner.state[params[0]]["exp_avg"])
    del saved["state"][0]["exp_avg"]
    assert "exp_avg" in inner.state[params[0]]
