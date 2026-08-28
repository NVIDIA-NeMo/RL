from __future__ import annotations

import inspect
from contextlib import AbstractContextManager
from types import ModuleType, SimpleNamespace, TracebackType
from typing import Any

import pytest
import torch

from nemo_rl.models.generation.vllm import refit_adapter


class _ConfigContext(AbstractContextManager[None]):
    def __init__(
        self,
        events: list[str],
        exit_error: BaseException | None = None,
    ) -> None:
        self._events = events
        self._exit_error = exit_error
        self.exit_errors: list[BaseException | None] = []

    def __enter__(self) -> None:
        self._events.append("enter_config")

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, traceback
        self._events.append("exit_config")
        self.exit_errors.append(exc_value)
        if self._exit_error is not None:
            raise self._exit_error


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


def _native_binding_refit_info() -> dict[str, Any]:
    hidden_size = 32
    intermediate_size = 64
    num_experts = 4

    def parameter(
        name: str,
        shape: tuple[int, ...],
        *,
        grouped_expert_proj: str | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": name,
            "global_shape": shape,
            "dtype": "torch.float8_e4m3fn",
            "components": [
                {
                    "role": "weight",
                    "dtype": "torch.float8_e4m3fn",
                    "global_shape": shape,
                },
                {
                    "role": "weight_scale",
                    "dtype": "torch.uint8",
                    "global_shape": (*shape[:-1], shape[-1] // 32),
                },
            ],
        }
        if grouped_expert_proj is not None:
            result["grouped_expert_proj"] = grouped_expert_proj
        return result

    prefix = "model.layers.0.mlp"
    return {
        "gen_tp_size": 2,
        "layer_names": ["model.layers.0"],
        "per_layer_params": {
            "model.layers.0": [
                parameter(
                    f"{prefix}.gate_proj.weight", (intermediate_size, hidden_size)
                ),
                parameter(f"{prefix}.up_proj.weight", (intermediate_size, hidden_size)),
                parameter(
                    f"{prefix}.down_proj.weight", (hidden_size, intermediate_size)
                ),
                parameter(
                    f"{prefix}.experts.gate_proj.weight",
                    (num_experts, intermediate_size, hidden_size),
                    grouped_expert_proj="gate_proj",
                ),
                parameter(
                    f"{prefix}.experts.up_proj.weight",
                    (num_experts, intermediate_size, hidden_size),
                    grouped_expert_proj="up_proj",
                ),
                parameter(
                    f"{prefix}.experts.down_proj.weight",
                    (num_experts, hidden_size, intermediate_size),
                    grouped_expert_proj="down_proj",
                ),
            ]
        },
    }


class _BindingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([torch.nn.Module()])
        mlp = torch.nn.Module()
        self.model.layers[0].mlp = mlp
        mlp.gate_up_proj = torch.nn.Module()
        mlp.down_proj = torch.nn.Module()
        mlp.experts = torch.nn.Module()
        mlp.experts.routed_experts = torch.nn.Module()

        self._register_runtime_pair(
            mlp.gate_up_proj,
            value_shape=(64, 32),
            scale_shape=(64, 1),
        )
        self._register_runtime_pair(
            mlp.down_proj,
            value_shape=(32, 32),
            scale_shape=(32, 1),
        )
        self._register_runtime_pair(
            mlp.experts.routed_experts,
            value_name="w13_weight",
            value_shape=(2, 64, 32),
            scale_shape=(2, 64, 1),
        )
        self._register_runtime_pair(
            mlp.experts.routed_experts,
            value_name="w2_weight",
            value_shape=(2, 32, 32),
            scale_shape=(2, 32, 1),
        )

    @staticmethod
    def _register_runtime_pair(
        owner: torch.nn.Module,
        *,
        value_shape: tuple[int, ...],
        scale_shape: tuple[int, ...],
        value_name: str = "weight",
    ) -> None:
        value = torch.nn.Parameter(
            torch.zeros(value_shape, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        runtime_scale = torch.nn.Parameter(
            torch.zeros(scale_shape, dtype=torch.uint8),
            requires_grad=False,
        )
        checkpoint_scale = torch.nn.Parameter(
            torch.zeros(scale_shape, dtype=torch.uint8),
            requires_grad=False,
        )
        owner.register_parameter(value_name, value)
        owner.register_parameter(f"{value_name}_scale", runtime_scale)
        owner.register_parameter(
            f"{value_name}_scale_from_checkpoint", checkpoint_scale
        )
        for parameter in (value, runtime_scale, checkpoint_scale):
            parameter.weight_loader = lambda target, loaded_weight: target.copy_(
                loaded_weight
            )


def _make_binding_adapter(
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
) -> tuple[
    refit_adapter.Vllm0251RefitAdapter,
    _BindingModel,
    list[tuple[str, inspect.BoundArguments]],
]:
    model = _BindingModel()
    runtime_parameters = dict(model.named_parameters())
    retained_loads: list[tuple[str, inspect.BoundArguments]] = []

    def make_online_process_loader(owner: torch.nn.Module, parameter_name: str) -> Any:
        original_loader = getattr(owner, parameter_name).weight_loader
        signature = inspect.signature(original_loader)

        def online_process_loader(*args: Any, **kwargs: Any) -> None:
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            retained_loads.append((parameter_name, bound))
            original_loader(*bound.args, **bound.kwargs)

        online_process_loader.__name__ = "online_process_loader"
        online_process_loader.__wrapped__ = original_loader
        return online_process_loader

    def initialize(checkpoint_model: _BindingModel) -> None:
        events.append("initialize")
        for module_name, owner in checkpoint_model.named_modules():
            for parameter_name in tuple(owner._parameters):
                if parameter_name.endswith("_scale_from_checkpoint"):
                    owner._parameters.pop(parameter_name)
                    continue
                runtime = getattr(owner, parameter_name)
                replacement = torch.nn.Parameter(
                    torch.empty_like(runtime), requires_grad=False
                )

                def unsupported_checkpoint_loader(
                    _target: torch.Tensor,
                    _loaded_weight: torch.Tensor,
                    **_kwargs: Any,
                ) -> None:
                    raise AssertionError(
                        "the vLLM checkpoint loader must be replaced by the local bridge"
                    )

                replacement.weight_loader = unsupported_checkpoint_loader
                owner._parameters[parameter_name] = replacement
                replacement.weight_loader = make_online_process_loader(
                    owner, parameter_name
                )
                events.append(f"checkpoint:{module_name}.{parameter_name}")

    def finalize(checkpoint_model: _BindingModel, _model_config: object) -> None:
        events.append("finalize")
        checkpoint_parameters = dict(checkpoint_model.named_parameters())
        for runtime_name, runtime in runtime_parameters.items():
            if runtime_name.endswith("_scale_from_checkpoint"):
                active_name = runtime_name.removesuffix("_from_checkpoint")
            else:
                active_name = runtime_name
            runtime.copy_(checkpoint_parameters[active_name])

        for module_name, owner in checkpoint_model.named_modules():
            prefix = f"{module_name}." if module_name else ""
            for parameter_name in tuple(owner._parameters):
                owner._parameters.pop(parameter_name)
            owned_runtime = {
                name.removeprefix(prefix): parameter
                for name, parameter in runtime_parameters.items()
                if name.startswith(prefix) and "." not in name.removeprefix(prefix)
            }
            for parameter_name, parameter in owned_runtime.items():
                owner.register_parameter(parameter_name, parameter)

    reload_module = ModuleType("vllm.model_executor.model_loader.reload")
    reload_module.initialize_layerwise_reload = initialize
    reload_module.finalize_layerwise_reload = finalize
    layerwise_module = ModuleType("vllm.model_executor.model_loader.reload.layerwise")
    layerwise_module.make_online_process_loader = make_online_process_loader
    config_module = ModuleType("vllm.config")
    config_module.set_current_vllm_config = lambda _config: _ConfigContext(events)
    _fake_importer(
        monkeypatch,
        {
            "vllm.config": config_module,
            "vllm.model_executor.model_loader.reload": reload_module,
            "vllm.model_executor.model_loader.reload.layerwise": layerwise_module,
        },
    )
    runner = SimpleNamespace(model=model, vllm_config=object())
    return (
        refit_adapter.Vllm0251RefitAdapter(
            model_runner=runner,
            model_config=object(),
            device=torch.device("cpu"),
        ),
        model,
        retained_loads,
    )


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
    *,
    finalizer_error: BaseException | None = None,
    exit_error: BaseException | None = None,
) -> tuple[
    refit_adapter.Vllm0251RefitAdapter,
    torch.nn.Parameter,
    _ConfigContext,
]:
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
        if finalizer_error is not None:
            raise finalizer_error

    reload_module = ModuleType("vllm.model_executor.model_loader.reload")
    reload_module.initialize_layerwise_reload = initialize
    reload_module.finalize_layerwise_reload = finalize
    config_module = ModuleType("vllm.config")
    config_context = _ConfigContext(events, exit_error=exit_error)
    config_module.set_current_vllm_config = lambda _config: config_context
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
        config_context,
    )


def _assert_unusable_after_failure(
    adapter: refit_adapter.Vllm0251RefitAdapter,
    parameter: torch.nn.Parameter,
    failure: BaseException,
) -> None:
    logical_name = "model.layers.0.mlp.down_proj.weight"
    later_updates = (
        lambda: adapter.prepare(_native_refit_info()),
        adapter.begin_update,
        lambda: adapter.load_component(
            logical_name=logical_name,
            role="weight",
            target=parameter,
            loaded_weight=torch.ones(2, 2),
        ),
        adapter.finish_update,
    )
    for later_update in later_updates:
        with pytest.raises(RuntimeError, match="worker is unusable") as error:
            later_update()
        assert error.value.__cause__ is failure


def test_0251_adapter_loads_each_component_through_wrapped_weight_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    adapter, parameter, _config_context = _make_adapter(monkeypatch, events)
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
    adapter, parameter, _config_context = _make_adapter(monkeypatch, events)

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
    adapter, parameter, _config_context = _make_adapter(monkeypatch, [])

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
    adapter, parameter, _config_context = _make_adapter(monkeypatch, events)
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


def test_0251_adapter_passes_finalizer_failure_to_context_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    finalizer_error = RuntimeError("finalizer failed")
    adapter, parameter, config_context = _make_adapter(
        monkeypatch,
        events,
        finalizer_error=finalizer_error,
    )
    logical_name = "model.layers.0.mlp.down_proj.weight"

    adapter.prepare(_native_refit_info())
    adapter.begin_update()
    adapter.load_component(
        logical_name=logical_name,
        role="weight",
        target=parameter,
        loaded_weight=torch.ones(2, 2),
    )
    adapter.load_component(
        logical_name=logical_name,
        role="weight_scale",
        target=parameter,
        loaded_weight=torch.ones(2, 2),
    )

    with pytest.raises(RuntimeError, match="finalizer failed"):
        adapter.finish_update()

    assert events.count("finalize") == 1
    assert events.count("exit_config") == 1
    assert config_context.exit_errors == [finalizer_error]
    _assert_unusable_after_failure(adapter, parameter, finalizer_error)


def test_0251_adapter_poisoned_when_config_exit_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    exit_error = RuntimeError("config exit failed")
    adapter, parameter, config_context = _make_adapter(
        monkeypatch,
        events,
        exit_error=exit_error,
    )
    logical_name = "model.layers.0.mlp.down_proj.weight"

    adapter.prepare(_native_refit_info())
    adapter.begin_update()
    adapter.load_component(
        logical_name=logical_name,
        role="weight",
        target=parameter,
        loaded_weight=torch.ones(2, 2),
    )
    adapter.load_component(
        logical_name=logical_name,
        role="weight_scale",
        target=parameter,
        loaded_weight=torch.ones(2, 2),
    )

    with pytest.raises(RuntimeError, match="config exit failed"):
        adapter.finish_update()

    assert events.count("finalize") == 1
    assert events.count("exit_config") == 1
    assert config_context.exit_errors == [None]
    _assert_unusable_after_failure(adapter, parameter, exit_error)


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
    base_module.TrainerWeightTransferEngine = type(
        "TrainerWeightTransferEngine",
        (),
        {
            "trainer_init": classmethod(
                lambda cls, _init_info, *, client, source=None: None
            ),
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

    base_module.TrainerWeightTransferEngine = type(
        "TrainerWeightTransferEngine",
        (),
        {
            "trainer_init": classmethod(lambda cls, _init_info, client: None),
            "send_weights": lambda self: None,
        },
    )
    assert not refit_adapter.probe_vllm_refit_capabilities().trainer_weight_transfer


@pytest.mark.parametrize(
    ("logical_name", "role", "expected_shape"),
    [
        ("model.layers.0.mlp.gate_proj.weight", "weight", (32, 32)),
        ("model.layers.0.mlp.gate_proj.weight", "weight_scale", (32, 1)),
        ("model.layers.0.mlp.up_proj.weight", "weight", (32, 32)),
        ("model.layers.0.mlp.up_proj.weight", "weight_scale", (32, 1)),
        ("model.layers.0.mlp.down_proj.weight", "weight", (32, 32)),
        ("model.layers.0.mlp.down_proj.weight", "weight_scale", (32, 1)),
        ("model.layers.0.mlp.experts.gate_proj.weight", "weight", (2, 32, 32)),
        (
            "model.layers.0.mlp.experts.gate_proj.weight",
            "weight_scale",
            (2, 32, 1),
        ),
        ("model.layers.0.mlp.experts.up_proj.weight", "weight", (2, 32, 32)),
        (
            "model.layers.0.mlp.experts.up_proj.weight",
            "weight_scale",
            (2, 32, 1),
        ),
        ("model.layers.0.mlp.experts.down_proj.weight", "weight", (2, 32, 32)),
        (
            "model.layers.0.mlp.experts.down_proj.weight",
            "weight_scale",
            (2, 32, 1),
        ),
    ],
)
def test_0251_adapter_binds_dense_and_routed_checkpoint_components(
    monkeypatch: pytest.MonkeyPatch,
    logical_name: str,
    role: str,
    expected_shape: tuple[int, ...],
) -> None:
    adapter, _model, retained_loads = _make_binding_adapter(monkeypatch, [])
    adapter.prepare(_native_binding_refit_info())
    adapter.begin_update()

    spec = adapter.resolve_destination(logical_name=logical_name, role=role)
    ctx = (
        spec.pre(spec.base) if spec.pre is not None else SimpleNamespace(buf=spec.base)
    )

    assert tuple(ctx.buf.shape) == expected_shape
    assert ctx.buf.dtype == (torch.float8_e4m3fn if role == "weight" else torch.uint8)
    assert spec.post is not None
    ctx.buf.fill_(3 if role == "weight" else 7)
    spec.post(ctx)

    routed = ".experts." in logical_name
    expected_calls = expected_shape[0] if routed else 1
    assert len(retained_loads) == expected_calls
    for _parameter_name, bound in retained_loads:
        assert bound.arguments["logical_name"] == logical_name
        assert bound.arguments["role"] == role
        if routed:
            assert bound.arguments["weight_name"].endswith(
                "weight" if role == "weight" else "weight_scale"
            )
            assert bound.arguments["shard_id"] in {"w1", "w2", "w3"}
            assert isinstance(bound.arguments["expert_id"], int)
        elif logical_name.endswith(("gate_proj.weight", "up_proj.weight")):
            assert bound.arguments["loaded_shard_id"] in {0, 1}


@pytest.mark.parametrize(
    ("case", "error"),
    [
        ("missing_runtime_alias", "scale_from_checkpoint"),
        ("missing_runtime_scale", "runtime scale"),
        ("wrong_runtime_scale_shape", "shape"),
        ("wrong_runtime_scale_dtype", "torch.uint8"),
        ("missing_runtime_loader", "weight_loader"),
        ("wrong_component_role", "unsupported component role"),
        ("wrong_component_shape", "scale shape"),
        ("wrong_component_dtype", "weight_scale dtype"),
    ],
)
def test_0251_adapter_prepare_rejects_invalid_destinations_before_begin(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    error: str,
) -> None:
    adapter, model, _retained_loads = _make_binding_adapter(monkeypatch, [])
    refit_info = _native_binding_refit_info()
    parameters = dict(model.named_parameters())
    value_name = "model.layers.0.mlp.gate_up_proj.weight"
    runtime_scale_name = f"{value_name}_scale"
    alias_name = f"{runtime_scale_name}_from_checkpoint"

    if case == "missing_runtime_alias":
        model.model.layers[0].mlp.gate_up_proj._parameters.pop(
            "weight_scale_from_checkpoint"
        )
    elif case == "missing_runtime_scale":
        model.model.layers[0].mlp.gate_up_proj._parameters.pop("weight_scale")
    elif case == "wrong_runtime_scale_shape":
        parameters[alias_name].data = torch.empty(64, 2, dtype=torch.uint8)
    elif case == "wrong_runtime_scale_dtype":
        parameters[alias_name].data = torch.empty(64, 1, dtype=torch.float32)
    elif case == "missing_runtime_loader":
        del parameters[alias_name].weight_loader
    else:
        gate = refit_info["per_layer_params"]["model.layers.0"][0]
        scale = gate["components"][1]
        if case == "wrong_component_role":
            scale["role"] = "runtime_scale"
        elif case == "wrong_component_shape":
            scale["global_shape"] = (64, 2)
        elif case == "wrong_component_dtype":
            scale["dtype"] = "torch.float32"

    with pytest.raises((RuntimeError, ValueError), match=error):
        adapter.prepare(refit_info)

    assert runtime_scale_name not in getattr(adapter, "_active_scale_names", {})


def test_0251_adapter_rejects_missing_checkpoint_alias_and_loader_before_receive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for case in ("missing", "loader"):
        adapter, model, _retained_loads = _make_binding_adapter(monkeypatch, [])
        adapter.prepare(_native_binding_refit_info())
        adapter.begin_update()
        owner = model.model.layers[0].mlp.gate_up_proj
        if case == "missing":
            owner._parameters.pop("weight_scale")
        else:
            del owner.weight_scale.weight_loader

        with pytest.raises(
            (RuntimeError, ValueError), match="checkpoint|weight_loader"
        ):
            adapter.resolve_destination(
                logical_name="model.layers.0.mlp.gate_proj.weight",
                role="weight_scale",
            )


def test_0251_adapter_wrapped_loader_owns_received_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, _model, retained_loads = _make_binding_adapter(monkeypatch, [])
    adapter.prepare(_native_binding_refit_info())
    adapter.begin_update()
    spec = adapter.resolve_destination(
        logical_name="model.layers.0.mlp.gate_proj.weight",
        role="weight",
    )
    assert spec.pre is not None and spec.post is not None
    ctx = spec.pre(spec.base)
    ctx.buf.fill_(3)
    spec.post(ctx)
    retained = retained_loads[0][1].arguments["loaded_weight"]
    ctx.buf.fill_(5)

    assert torch.all(retained.float() == 3)
    assert retained.untyped_storage().data_ptr() != ctx.buf.untyped_storage().data_ptr()


def test_0251_adapter_repeated_refits_change_bytes_and_preserve_runtime_pointers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    adapter, model, _retained_loads = _make_binding_adapter(monkeypatch, events)
    refit_info = _native_binding_refit_info()
    adapter.prepare(refit_info)
    runtime_parameters = dict(model.named_parameters())
    value = runtime_parameters["model.layers.0.mlp.down_proj.weight"]
    scale = runtime_parameters["model.layers.0.mlp.down_proj.weight_scale"]
    value_pointer = value.data_ptr()
    scale_pointer = scale.data_ptr()
    snapshots: list[tuple[torch.Tensor, torch.Tensor]] = []

    for fill_value in (1, 2):
        adapter.begin_update()
        for param_info in refit_info["per_layer_params"]["model.layers.0"]:
            for component in param_info["components"]:
                role = component["role"]
                spec = adapter.resolve_destination(
                    logical_name=param_info["name"], role=role
                )
                assert spec.pre is not None and spec.post is not None
                ctx = spec.pre(spec.base)
                ctx.buf.fill_(fill_value if role == "weight" else fill_value + 4)
                spec.post(ctx)
        adapter.finish_update()
        snapshots.append((value.clone(), scale.clone()))
        assert value.data_ptr() == value_pointer
        assert scale.data_ptr() == scale_pointer

    assert not torch.equal(snapshots[0][0], snapshots[1][0])
    assert not torch.equal(snapshots[0][1], snapshots[1][1])
    assert events.count("initialize") == 2
    assert events.count("finalize") == 2
