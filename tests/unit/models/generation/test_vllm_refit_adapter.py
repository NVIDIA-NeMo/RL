from __future__ import annotations

from contextlib import AbstractContextManager
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

from nemo_rl.models.generation.vllm import refit_adapter


class _ConfigContext(AbstractContextManager[None]):
    def __init__(self, events: list[str]) -> None:
        self._events = events

    def __enter__(self) -> None:
        self._events.append("enter_config")

    def __exit__(self, *args: object) -> None:
        self._events.append("exit_config")


def _native_refit_info() -> dict[str, Any]:
    logical_name = "model.layers.0.mlp.down_proj.weight"
    return {
        "layer_names": ["model.layers.0"],
        "per_layer_params": {
            "model.layers.0": [
                {
                    "name": logical_name,
                    "components": [
                        {"role": "weight"},
                        {"role": "weight_scale"},
                    ],
                }
            ]
        },
    }


def _fake_importer(
    monkeypatch: pytest.MonkeyPatch,
    modules: dict[str, ModuleType],
) -> None:
    def import_module(name: str) -> ModuleType:
        if name not in modules:
            raise ModuleNotFoundError(name)
        return modules[name]

    monkeypatch.setattr(refit_adapter.importlib, "import_module", import_module)


def _make_adapter(
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
) -> tuple[refit_adapter.Vllm0251RefitAdapter, torch.nn.Parameter]:
    parameter = torch.nn.Parameter(torch.zeros(2, 2), requires_grad=False)

    def checkpoint_loader(
        target: torch.Tensor,
        loaded_weight: torch.Tensor,
    ) -> None:
        events.append(f"load:{int(loaded_weight.flatten()[0])}")
        target.copy_(loaded_weight)

    parameter.weight_loader = checkpoint_loader

    def initialize(model: SimpleNamespace) -> None:
        events.append("initialize")
        model.parameter.weight_loader = checkpoint_loader

    def finalize(_model: SimpleNamespace, _model_config: object) -> None:
        events.append("finalize")

    reload_module = ModuleType("vllm.model_executor.model_loader.reload")
    reload_module.initialize_layerwise_reload = initialize
    reload_module.finalize_layerwise_reload = finalize
    config_module = ModuleType("vllm.config")
    config_module.set_current_vllm_config = lambda _config: _ConfigContext(events)
    _fake_importer(
        monkeypatch,
        {
            "vllm.config": config_module,
            "vllm.model_executor.model_loader.reload": reload_module,
        },
    )
    model = SimpleNamespace(parameter=parameter)
    runner = SimpleNamespace(model=model, vllm_config=object())
    return (
        refit_adapter.Vllm0251RefitAdapter(
            model_runner=runner,
            model_config=object(),
            device=torch.device("cpu"),
        ),
        parameter,
    )


def test_0251_adapter_loads_each_component_through_wrapped_weight_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    adapter, parameter = _make_adapter(monkeypatch, events)
    logical_name = "model.layers.0.mlp.down_proj.weight"

    adapter.prepare(_native_refit_info())
    adapter.begin_update()
    adapter.load_component(
        logical_name=logical_name,
        role="weight",
        target=parameter,
        loaded_weight=torch.full((2, 2), 3.0),
    )
    adapter.load_component(
        logical_name=logical_name,
        role="weight_scale",
        target=parameter,
        loaded_weight=torch.full((2, 2), 4.0),
    )
    adapter.finish_update()

    assert events == [
        "enter_config",
        "initialize",
        "load:3",
        "load:4",
        "finalize",
        "exit_config",
    ]
    assert torch.equal(parameter, torch.full((2, 2), 4.0))


def test_0251_adapter_rejects_finalize_before_every_component_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    adapter, parameter = _make_adapter(monkeypatch, events)

    adapter.prepare(_native_refit_info())
    adapter.begin_update()
    adapter.load_component(
        logical_name="model.layers.0.mlp.down_proj.weight",
        role="weight",
        target=parameter,
        loaded_weight=torch.ones(2, 2),
    )

    with pytest.raises(RuntimeError, match="missing component loads"):
        adapter.finish_update()

    assert "finalize" not in events
    with pytest.raises(RuntimeError, match="worker is unusable"):
        adapter.begin_update()


def test_0251_adapter_fails_closed_after_loader_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, parameter = _make_adapter(monkeypatch, [])

    def failing_loader(_target: torch.Tensor, _loaded_weight: torch.Tensor) -> None:
        raise ValueError("load failed")

    adapter.prepare(_native_refit_info())
    adapter.begin_update()
    parameter.weight_loader = failing_loader

    with pytest.raises(ValueError, match="load failed"):
        adapter.load_component(
            logical_name="model.layers.0.mlp.down_proj.weight",
            role="weight",
            target=parameter,
            loaded_weight=torch.ones(2, 2),
        )

    with pytest.raises(RuntimeError, match="worker is unusable") as error:
        adapter.begin_update()
    assert isinstance(error.value.__cause__, ValueError)


def test_0251_adapter_allows_a_second_complete_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    adapter, parameter = _make_adapter(monkeypatch, events)
    logical_name = "model.layers.0.mlp.down_proj.weight"
    adapter.prepare(_native_refit_info())

    for value in (1.0, 2.0):
        adapter.begin_update()
        adapter.load_component(
            logical_name=logical_name,
            role="weight",
            target=parameter,
            loaded_weight=torch.full((2, 2), value),
        )
        adapter.load_component(
            logical_name=logical_name,
            role="weight_scale",
            target=parameter,
            loaded_weight=torch.full((2, 2), value),
        )
        adapter.finish_update()

    assert events.count("initialize") == 2
    assert events.count("finalize") == 2
    assert torch.equal(parameter, torch.full((2, 2), 2.0))


def test_factory_requires_layerwise_reload_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_importer(monkeypatch, {})
    runner = SimpleNamespace(model=SimpleNamespace(), vllm_config=object())

    with pytest.raises(RuntimeError, match="required layerwise reload"):
        refit_adapter.create_vllm_refit_adapter(
            model_runner=runner,
            model_config=object(),
            device=torch.device("cpu"),
        )


def test_capability_probe_records_later_engine_api_without_selecting_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reload_module = ModuleType("vllm.model_executor.model_loader.reload")
    reload_module.initialize_layerwise_reload = lambda _model: None
    reload_module.finalize_layerwise_reload = lambda _model, _config: None
    config_module = ModuleType("vllm.config")
    config_module.set_current_vllm_config = lambda _config: _ConfigContext([])
    factory_module = ModuleType("vllm.distributed.weight_transfer.factory")
    factory_module.WeightTransferEngineFactory = type(
        "WeightTransferEngineFactory",
        (),
        {"register_engine": staticmethod(lambda _name, _engine: None)},
    )
    factory_module.WeightTransferTrainerFactory = type(
        "WeightTransferTrainerFactory",
        (),
        {"register_engine": staticmethod(lambda _name, _engine: None)},
    )
    base_module = ModuleType("vllm.distributed.weight_transfer.base")
    base_module.WeightTransferEngine = type(
        "WeightTransferEngine",
        (),
        {
            "start_weight_update": lambda self: None,
            "update_weights": lambda self, _request: None,
            "finish_weight_update": lambda self: None,
        },
    )
    base_module.WeightTransferTrainer = type(
        "WeightTransferTrainer",
        (),
        {
            "trainer_init": lambda self: None,
            "send_weights": lambda self: None,
        },
    )
    _fake_importer(
        monkeypatch,
        {
            "vllm.config": config_module,
            "vllm.model_executor.model_loader.reload": reload_module,
            "vllm.distributed.weight_transfer.factory": factory_module,
            "vllm.distributed.weight_transfer.base": base_module,
        },
    )

    capabilities = refit_adapter.probe_vllm_refit_capabilities()
    assert capabilities == refit_adapter.VllmRefitCapabilities(
        layerwise_reload=True,
        weight_transfer_engine_registry=True,
        trainer_weight_transfer=True,
    )
    runner = SimpleNamespace(model=SimpleNamespace(), vllm_config=object())
    adapter = refit_adapter.create_vllm_refit_adapter(
        model_runner=runner,
        model_config=object(),
        device=torch.device("cpu"),
    )
    assert isinstance(adapter, refit_adapter.Vllm0251RefitAdapter)
