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
import asyncio
import dataclasses
import inspect
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from nemo_rl.models.generation.megatron import dynamo_worker_mixin, megatron_worker
from nemo_rl.models.generation.megatron.dynamo_worker_mixin import (
    MegatronDynamoGenerationMixin,
    require_megatron_dynamo_integration,
)
from nemo_rl.models.generation.megatron.megatron_worker import MegatronGenerationMixin

pytestmark = pytest.mark.mcore


class _Worker(MegatronDynamoGenerationMixin, MegatronGenerationMixin):
    """MegatronPolicyWorkerImpl's generation bases, in the order it composes them."""


def _bare_worker(role, cls=_Worker):
    worker = object.__new__(cls)
    worker.cfg = {"generation": {}}
    if role is not None:
        worker.cfg["generation"]["_dynamo_role"] = role
    return worker


@pytest.fixture
def engine_modules(monkeypatch):
    """Fake the plain and the Dynamo handoff engine classes the hooks pick between."""
    plain = MagicMock(name="DynamicInferenceEngine")
    handoff = MagicMock(name="DynamoDynamicInferenceEngine")
    monkeypatch.setattr(
        "megatron.core.inference.engines.dynamic_engine.DynamicInferenceEngine", plain
    )
    for name in (
        "megatron.inference",
        "megatron.inference.integrations",
        "megatron.inference.integrations.dynamo",
    ):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    module = types.ModuleType("megatron.inference.integrations.dynamo.dynamic_engine")
    module.DynamoDynamicInferenceEngine = handoff
    monkeypatch.setitem(
        sys.modules, "megatron.inference.integrations.dynamo.dynamic_engine", module
    )
    monkeypatch.setattr(dynamo_worker_mixin, "_get_node_ip_local", lambda: "10.0.0.5")
    return plain, handoff


def test_plain_worker_builds_the_plain_engine(engine_modules):
    plain, handoff = engine_modules
    worker = _bare_worker(role=None, cls=MegatronGenerationMixin)
    controller, context = MagicMock(), MagicMock()

    assert worker._inference_config_overrides() == {}
    assert worker._coordinator_kwargs() == {}
    assert worker._build_dynamic_inference_engine(controller, context) is (
        plain.return_value
    )
    plain.assert_called_once_with(controller, context)
    handoff.assert_not_called()


# Every override of a MegatronGenerationMixin method on the Dynamo mixin. Each
# must fall back to the plain behaviour when the worker has no role, since every
# Megatron worker (training included) carries the mixin. An override added
# without that fallback fails test_every_override_of_the_plain_mixin_is_role_guarded
# until it is listed here and covered by test_no_role_hooks_behave_like_the_plain_worker.
_ROLE_GUARDED = {
    "_check_generation_requirements",
    "_inference_config_overrides",
    "_build_dynamic_inference_engine",
    "_coordinator_kwargs",
}


def test_every_override_of_the_plain_mixin_is_role_guarded():
    overrides = {
        name
        for name, value in vars(MegatronDynamoGenerationMixin).items()
        if callable(value) and hasattr(MegatronGenerationMixin, name)
    }
    assert overrides == _ROLE_GUARDED


def test_no_role_hooks_behave_like_the_plain_worker(engine_modules, monkeypatch):
    """A worker without a role (training, plain inference) gets the plain hooks."""
    plain, handoff = engine_modules
    checks = []
    monkeypatch.setattr(
        dynamo_worker_mixin,
        "require_megatron_dynamo_integration",
        lambda: checks.append("checked"),
    )
    worker = _bare_worker(role=None)
    controller, context = MagicMock(), MagicMock()

    assert worker._dynamo_role is None
    worker._check_generation_requirements(worker.cfg)
    assert checks == []
    assert worker._inference_config_overrides() == {}
    assert worker._coordinator_kwargs() == {}
    assert worker._build_dynamic_inference_engine(controller, context) is (
        plain.return_value
    )
    handoff.assert_not_called()


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_an_engine_role_runs_the_early_megatron_lm_check(monkeypatch, role):
    checks = []
    monkeypatch.setattr(
        dynamo_worker_mixin,
        "require_megatron_dynamo_integration",
        lambda: checks.append("checked"),
    )
    # The hook reads the constructor's config: self.cfg is not set yet.
    worker = object.__new__(_Worker)
    worker._check_generation_requirements({"generation": {"_dynamo_role": role}})
    assert checks == ["checked"]


def test_an_unknown_engine_role_is_refused():
    worker = _bare_worker(role="aggregated")
    with pytest.raises(ValueError, match="must be 'prefill' or 'decode'"):
        worker._dynamo_role
    with pytest.raises(ValueError, match="must be 'prefill' or 'decode'"):
        worker._check_generation_requirements(worker.cfg)


def _policy_worker_module():
    """``megatron_policy_worker``, which needs Megatron-Bridge (CI, not macOS)."""
    pytest.importorskip("megatron.bridge")
    from nemo_rl.models.policy.workers import megatron_policy_worker

    return megatron_policy_worker


def test_the_policy_worker_carries_the_dynamo_hooks_ahead_of_the_plain_ones():
    module = _policy_worker_module()
    mro = module.MegatronPolicyWorkerImpl.__mro__
    assert mro.index(MegatronDynamoGenerationMixin) < mro.index(MegatronGenerationMixin)
    assert not hasattr(module, "MegatronDynamoPolicyWorker")


def test_the_early_check_runs_before_the_worker_loads_anything(monkeypatch):
    """The first thing the constructor does, with the constructor's config."""
    module = _policy_worker_module()

    class Stop(Exception):
        pass

    seen = []

    class ExtensionWorker(module.MegatronPolicyWorkerImpl):
        def _check_generation_requirements(self, config):
            seen.append(config)
            raise Stop

    def must_not_run(*args, **kwargs):
        raise AssertionError("the worker started loading before the check")

    monkeypatch.setattr(module, "maybe_enable_msc", must_not_run)
    config = {"generation": {"_dynamo_role": "decode"}}
    with pytest.raises(Stop):
        ExtensionWorker(config, None, worker_sharding_annotations=None)
    assert seen == [config]


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_dynamo_worker_builds_the_handoff_engine(engine_modules, role):
    plain, handoff = engine_modules
    worker = _bare_worker(role=role)
    controller, context = MagicMock(), MagicMock()

    assert worker._dynamo_role == role
    # The peer role lives in other processes: NIXL carries the handoff and the
    # coordinator must listen on a routable address. The role reaches
    # Megatron-LM, which drops the Mamba prefix cache for decode.
    assert worker._inference_config_overrides() == {
        "kv_transport_backend": "nixl",
        "disaggregation_role": role,
    }
    assert worker._coordinator_kwargs() == {"hostname": "10.0.0.5"}

    engine = worker._build_dynamic_inference_engine(controller, context)

    assert engine is handoff.return_value
    handoff.assert_called_once_with(controller, context)
    engine.setup_kv_transfer.assert_called_once_with(
        role=role, require_handoff_claim=True
    )
    plain.assert_not_called()


def test_dynamo_worker_requires_the_megatron_dynamo_integration(monkeypatch):
    module = "megatron.inference.integrations.dynamo.dynamic_engine"
    monkeypatch.setitem(sys.modules, module, types.ModuleType(module))
    require_megatron_dynamo_integration()

    # None in sys.modules makes the import raise ImportError.
    monkeypatch.setitem(sys.modules, module, None)
    with pytest.raises(RuntimeError, match="megatron.inference.integrations.dynamo"):
        require_megatron_dynamo_integration()


def test_dynamo_worker_requires_an_inference_config_with_the_role(monkeypatch):
    """An older Megatron-LM fails at the early check, not at the first refit."""
    module = "megatron.inference.integrations.dynamo.dynamic_engine"
    monkeypatch.setitem(sys.modules, module, types.ModuleType(module))

    @dataclasses.dataclass
    class OldInferenceConfig:
        kv_transport_backend: str = "nixl"

    monkeypatch.setattr(
        "megatron.core.inference.config.InferenceConfig", OldInferenceConfig
    )
    with pytest.raises(RuntimeError, match="InferenceConfig has disaggregation_role"):
        require_megatron_dynamo_integration()


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_inference_config_accepts_every_dynamo_override(role):
    """The installed Megatron-LM takes every key the role hooks add."""
    from megatron.core.inference.config import InferenceConfig

    fields = {field.name for field in dataclasses.fields(InferenceConfig)}
    overrides = _bare_worker(role=role)._inference_config_overrides()
    assert set(overrides) <= fields


# ---------------------------------------------------------------------------
# The hooks at their real call sites in MegatronGenerationMixin
# ---------------------------------------------------------------------------


class _Stop(Exception):
    pass


_MCORE_GENERATION_CONFIG = {
    "buffer_size_gb": 1,
    "num_cuda_graphs": 1,
    "block_size_tokens": 16,
    "enable_chunked_prefill": False,
    "use_cuda_graphs_for_non_decode_steps": False,
    "max_tokens": 16,
    "max_model_len": 64,
    "kv_cache_management_mode": "persist",
    "materialize_only_last_token_logits": True,
    "num_speculative_tokens": 0,
    "enable_prefix_caching": False,
    "logprobs_mode": "processed_logprobs",
}

# (worker class, role): the plain mixin, the composite with no role (every
# Megatron worker now), and the composite as a prefill or decode engine.
_HOSTS = [
    pytest.param(MegatronGenerationMixin, None, id="plain_mixin"),
    pytest.param(_Worker, None, id="no_role"),
    pytest.param(_Worker, "prefill", id="prefill"),
    pytest.param(_Worker, "decode", id="decode"),
]


def _engine_ready_worker(monkeypatch, cls, role):
    """A worker whose _initialize_inference_engine runs to InferenceConfig on CPU."""
    monkeypatch.setattr(
        "megatron.core.utils.get_attr_wrapped_model", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        megatron_worker,
        "MambaInferenceStateConfig",
        SimpleNamespace(from_model=lambda _model: None),
    )
    monkeypatch.setattr(cls, "_gen_model", lambda self: SimpleNamespace(config=None))
    monkeypatch.setattr(cls, "_get_megatron_inference_wrapper_cls", lambda self: None)
    monkeypatch.setattr(
        cls,
        "_inference_model_and_media_parts",
        lambda self, _cls: (
            SimpleNamespace(
                config=SimpleNamespace(
                    params_dtype=torch.bfloat16,
                    inference_flashinfer_mxfp8_token_capacity=None,
                )
            ),
            None,
        ),
    )
    worker = _bare_worker(role, cls=cls)
    worker._inference_engine_initialized = False
    worker._router_replay_enabled = False
    return worker


def _expected_overrides(role):
    if role is None:
        return {}
    return {"kv_transport_backend": "nixl", "disaggregation_role": role}


@pytest.mark.parametrize(("cls", "role"), _HOSTS)
def test_role_overrides_reach_the_inference_config(monkeypatch, cls, role):
    seen = {}

    def record(**kwargs):
        seen.update(kwargs)
        raise _Stop

    monkeypatch.setattr(megatron_worker, "InferenceConfig", record)
    worker = _engine_ready_worker(monkeypatch, cls, role)

    with pytest.raises(_Stop):
        worker._initialize_inference_engine(dict(_MCORE_GENERATION_CONFIG))

    expected = _expected_overrides(role)
    assert {key: seen[key] for key in expected} == expected
    if role is None:
        assert "disaggregation_role" not in seen


@pytest.mark.parametrize(("cls", "role"), _HOSTS)
def test_the_installed_inference_config_takes_the_role_kwargs(monkeypatch, cls, role):
    """The real InferenceConfig is built; the call stops at the context after it."""
    monkeypatch.setattr(
        "megatron.core.inference.contexts.dynamic_context.DynamicInferenceContext",
        lambda *_a, **_k: (_ for _ in ()).throw(_Stop()),
    )
    worker = _engine_ready_worker(monkeypatch, cls, role)
    with pytest.raises(_Stop):
        worker._initialize_inference_engine(dict(_MCORE_GENERATION_CONFIG))


@pytest.mark.parametrize(("cls", "role"), _HOSTS)
def test_engine_build_goes_through_the_role_hook(
    monkeypatch, engine_modules, cls, role
):
    plain, handoff = engine_modules
    monkeypatch.setattr(megatron_worker, "InferenceConfig", lambda **kwargs: kwargs)
    for path, fake in (
        (
            "megatron.core.inference.contexts.dynamic_context.DynamicInferenceContext",
            lambda *_a: SimpleNamespace(max_requests=1),
        ),
        (
            "megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper.GPTInferenceWrapper",
            lambda *_a: "wrapped",
        ),
        (
            "megatron.core.inference.text_generation_controllers.text_generation_controller.TextGenerationController",
            lambda **_k: "controller",
        ),
    ):
        monkeypatch.setattr(path, fake)
    worker = _engine_ready_worker(monkeypatch, cls, role)
    worker.rank = 0
    worker.megatron_tokenizer = None

    worker._initialize_inference_engine(dict(_MCORE_GENERATION_CONFIG))

    if role is None:
        assert worker.dynamic_inference_engine is plain.return_value
        handoff.assert_not_called()
    else:
        assert worker.dynamic_inference_engine is handoff.return_value
        handoff.return_value.setup_kv_transfer.assert_called_once_with(
            role=role, require_handoff_claim=True
        )
        plain.assert_not_called()


@pytest.mark.parametrize(("cls", "role"), _HOSTS)
def test_coordinator_launch_gets_the_role_kwargs(monkeypatch, cls, role):
    monkeypatch.setattr(dynamo_worker_mixin, "_get_node_ip_local", lambda: "10.0.0.5")
    # Not rank zero, so no inference client is started.
    monkeypatch.setattr(torch.distributed, "get_rank", lambda *a, **k: 1)
    calls = []

    async def start_listening(**kwargs):
        calls.append(kwargs)
        return "tcp://coordinator"

    worker = _bare_worker(role, cls=cls)
    worker.dynamic_inference_engine = SimpleNamespace(
        start_listening_to_data_parallel_coordinator=start_listening
    )

    asyncio.run(worker._start_inference_coordinator())

    expected = {} if role is None else {"hostname": "10.0.0.5"}
    assert calls == [
        {
            "inference_coordinator_port": None,
            "launch_inference_coordinator": True,
            **expected,
        }
    ]


def test_installed_megatron_lm_serves_the_dynamo_worker_hooks():
    """The real Megatron-LM classes the hooks call take the arguments they pass."""
    from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine

    require_megatron_dynamo_integration()
    from megatron.inference.integrations.dynamo.dynamic_engine import (
        DynamoDynamicInferenceEngine,
    )

    setup = inspect.signature(DynamoDynamicInferenceEngine.setup_kv_transfer)
    assert {"role", "require_handoff_claim"} <= set(setup.parameters)
    listen = inspect.signature(
        DynamicInferenceEngine.start_listening_to_data_parallel_coordinator
    )
    assert "hostname" in listen.parameters
