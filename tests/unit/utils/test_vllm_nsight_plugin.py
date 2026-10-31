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
import logging
import sys
import types
from importlib.metadata import entry_points
from unittest.mock import patch

import pytest
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

# nemo_rl.utils.nsys reads NRL_NSYS_* once, at import. Import it here, before any test
# sets those variables, so its module constants never capture a test's environment.
import nemo_rl.utils.nsys  # noqa: F401
from nemo_rl.utils.vllm_nsight_plugin import (
    _WorkerGpuMetricsRay,
    register_nsight_inner_worker_profiling,
)

V2_MODULE = "vllm.v1.executor.ray_executor_v2"
PLUGIN_LOGGER = "nemo_rl.utils.vllm_nsight_plugin"


class _FakeRay:
    """Records the actor options a vLLM executor passes to ``ray``."""

    def __init__(self):
        self.options = []

    def remote(self, *args, **kwargs):
        if kwargs:  # v1 executor: ray.remote(**options)(cls)
            self.options.append(kwargs)
            return lambda cls: cls
        recorded = self.options  # v2 executor: ray.remote(cls).options(**options)
        return types.SimpleNamespace(options=lambda **kw: recorded.append(kw))

    def get(self, ref):
        return ref


@pytest.fixture
def fake_v2(monkeypatch):
    """Stub RayExecutorV2 (no GPU/vLLM needed) and enable NeMo-RL nsys profiling.

    The class is created per test and installed under its full module name, so the
    class-level patch never touches the real vLLM executor and nothing leaks.
    """

    class RayExecutorV2:
        def __init__(self, use_nsight=True, world_size=1):
            self.parallel_config = types.SimpleNamespace(
                ray_workers_use_nsight=use_nsight, world_size=world_size
            )

        def _build_runtime_env(self):
            return {"env_vars": {"KEEP": "1"}}

    module = types.ModuleType(V2_MODULE)
    module.RayExecutorV2 = RayExecutorV2
    module.ray = _FakeRay()
    RayExecutorV2.__module__ = V2_MODULE
    monkeypatch.setitem(sys.modules, V2_MODULE, module)
    monkeypatch.setenv("NRL_NSYS_WORKER_PATTERNS", "*vllm*")
    return RayExecutorV2


def test_entry_point_resolves():
    """vLLM finds the plugin only through installed metadata; a typo or stale install
    makes it silently absent (vLLM swallows plugin load errors)."""
    eps = [
        e
        for e in entry_points(group="vllm.general_plugins")
        if e.name == "nemo_rl_nsight_inner"
    ]
    assert len(eps) == 1, (
        f"expected exactly one 'nemo_rl_nsight_inner' vllm.general_plugins entry point, "
        f"found {eps}. The installed nemo-rl metadata is stale or missing: reinstall the "
        "project (e.g. `uv sync`) so [project.entry-points] from pyproject.toml is registered."
    )
    assert eps[0].load() is register_nsight_inner_worker_profiling


def test_noop_without_worker_patterns(fake_v2, monkeypatch):
    monkeypatch.delenv("NRL_NSYS_WORKER_PATTERNS")
    original = fake_v2._build_runtime_env
    register_nsight_inner_worker_profiling()
    assert fake_v2._build_runtime_env is original
    assert not hasattr(fake_v2, "_nrl_nsight_patched")


def test_installs_deferred_capture_config_once(fake_v2, caplog):
    with (
        patch("nemo_rl.utils.nsys.NRL_NSYS_PROFILE_STEP_RANGE", "1:3"),
        patch(
            "nemo_rl.utils.nsys.NRL_NSYS_EXTRA_OPTIONS",
            {"t": "cuda-sw,nvtx", "cuda-graph-trace": None},
        ),
        caplog.at_level(logging.INFO, logger=PLUGIN_LOGGER),
    ):
        register_nsight_inner_worker_profiling()
        patched = fake_v2._build_runtime_env
        register_nsight_inner_worker_profiling()  # second load must not re-wrap
        assert fake_v2._build_runtime_env is patched
        # The nsight config is built when the executor builds the runtime_env, so the
        # call must stay inside the patch() block.
        runtime_env = fake_v2()._build_runtime_env()
    assert runtime_env["env_vars"] == {"KEEP": "1"}  # the original runtime_env survives
    nsight = runtime_env["nsight"]
    assert nsight["o"] == "'vllm_tp_worker_1:3_%p'"
    assert nsight["capture-range"] == "cudaProfilerApi"
    assert nsight["capture-range-end"] == "stop"
    assert nsight["wait"] == "primary"
    assert nsight["t"] == "cuda-sw,nvtx"
    assert "cuda-graph-trace" not in nsight
    installs = [r for r in caplog.records if r.name == PLUGIN_LOGGER]
    assert len(installs) == 1, [r.getMessage() for r in installs]


def test_runtime_env_untouched_without_nsight(fake_v2):
    register_nsight_inner_worker_profiling()
    assert fake_v2(use_nsight=False)._build_runtime_env() == {"env_vars": {"KEEP": "1"}}


def test_missing_v2_executor_fails_loudly(monkeypatch):
    """No silent fallback: if vLLM moves RayExecutorV2, a profiled run fails at startup."""
    monkeypatch.setenv("NRL_NSYS_WORKER_PATTERNS", "*vllm*")
    monkeypatch.setitem(sys.modules, V2_MODULE, None)  # import now raises ImportError
    with pytest.raises(ImportError):
        register_nsight_inner_worker_profiling()


def test_bad_extra_options_fail_at_executor_startup_not_at_plugin_load(
    fake_v2, monkeypatch
):
    """A malformed NRL_NSYS_EXTRA_OPTIONS surfaces as the parser's ValueError (with
    guidance) when the executor builds the inner runtime_env, not at plugin load:
    vLLM runs plugins in every process and swallows only ``plugin.load()`` errors."""
    import nemo_rl.utils
    import nemo_rl.utils.nsys as nsys_module

    monkeypatch.setenv("NRL_NSYS_EXTRA_OPTIONS", "not-json")
    # nemo_rl.utils.nsys parses the variable at import time. Drop the cached module so the
    # plugin's lazy import re-parses it; monkeypatch restores the module and the package
    # attribute afterwards (the failed import leaves neither behind).
    monkeypatch.delitem(sys.modules, "nemo_rl.utils.nsys")
    monkeypatch.setattr(nemo_rl.utils, "nsys", nsys_module)

    register_nsight_inner_worker_profiling()
    assert "nemo_rl.utils.nsys" not in sys.modules  # helper import is deferred

    with pytest.raises(ValueError, match="Failed to parse NRL_NSYS_EXTRA_OPTIONS"):
        fake_v2()._build_runtime_env()


@pytest.mark.vllm
def test_real_ray_executor_v2_matches_stub_contract(monkeypatch):
    """The stub above mirrors private vLLM API; fail at the vLLM bump if it changes."""
    import inspect

    from vllm.v1.executor.ray_executor_v2 import RayExecutorV2

    assert "_build_runtime_env" in RayExecutorV2.__dict__
    assert list(inspect.signature(RayExecutorV2._build_runtime_env).parameters) == [
        "self"
    ]
    # Install on the real class; monkeypatch restores the method and removes the flag.
    monkeypatch.setattr(
        RayExecutorV2, "_build_runtime_env", RayExecutorV2._build_runtime_env
    )
    monkeypatch.setattr(RayExecutorV2, "_nrl_nsight_patched", False, raising=False)
    monkeypatch.setenv("NRL_NSYS_WORKER_PATTERNS", "*vllm*")
    original = RayExecutorV2._build_runtime_env
    register_nsight_inner_worker_profiling()
    assert RayExecutorV2._build_runtime_env is not original
    assert RayExecutorV2._nrl_nsight_patched is True


WORKER_GPU_METRICS = {"gpu-metrics-devices": "worker", "cuda-memory-usage": "true"}


def _strategy(bundle_index):
    return PlacementGroupSchedulingStrategy(
        placement_group="pg", placement_group_bundle_index=bundle_index
    )


def test_inner_workers_sample_their_own_gpu(fake_v2, monkeypatch):
    """v2: each inner worker created with the shared runtime_env samples its bundle's GPU."""
    # Non-colocated TP=2 engine: 2 x 0.5 GPU per inner worker = whole GPUs.
    monkeypatch.setenv("VLLM_RAY_PER_WORKER_GPUS", "0.5")
    module = sys.modules[V2_MODULE]
    fake_ray = module.ray
    with (
        patch("nemo_rl.utils.nsys.NRL_NSYS_EXTRA_OPTIONS", WORKER_GPU_METRICS),
        patch(
            "nemo_rl.distributed.virtual_cluster.get_bundle_gpu_id",
            side_effect=lambda pg, bundle: {4: 2, 5: 3}[bundle],
        ),
    ):
        register_nsight_inner_worker_profiling()
        runtime_env = fake_v2(world_size=2)._build_runtime_env()
        assert runtime_env["nsight"]["gpu-metrics-devices"] == "worker"
        assert isinstance(module.ray, _WorkerGpuMetricsRay)
        fake_v2(
            world_size=2
        )._build_runtime_env()  # second executor in the same process
        assert not isinstance(module.ray._ray, _WorkerGpuMetricsRay)  # no double wrap
        # v2 creates each inner worker with the one shared runtime_env.
        for bundle in (4, 5):
            module.ray.remote(object).options(
                runtime_env=runtime_env, scheduling_strategy=_strategy(bundle)
            )
    devices = [
        o["runtime_env"]["nsight"]["gpu-metrics-devices"] for o in fake_ray.options
    ]
    assert devices == ["2", "3"]
    assert all(
        o["runtime_env"]["nsight"]["cuda-memory-usage"] == "true"
        for o in fake_ray.options
    )
    # The shared runtime_env keeps the marker; each worker got its own copy.
    assert runtime_env["nsight"]["gpu-metrics-devices"] == "worker"


def test_colocated_inner_workers_do_not_sample(fake_v2, monkeypatch):
    """An engine sharing its GPUs with the training workers leaves the GPUs to them."""
    # Colocated TP=2 engine: 2 x 0.25 GPU per inner worker.
    monkeypatch.setenv("VLLM_RAY_PER_WORKER_GPUS", "0.25")
    module = sys.modules[V2_MODULE]
    fake_ray = module.ray
    with patch("nemo_rl.utils.nsys.NRL_NSYS_EXTRA_OPTIONS", WORKER_GPU_METRICS):
        register_nsight_inner_worker_profiling()
        nsight = fake_v2(world_size=2)._build_runtime_env()["nsight"]
    assert "gpu-metrics-devices" not in nsight
    assert nsight["cuda-memory-usage"] == "true"
    assert module.ray is fake_ray


def test_inherited_outer_config_gets_worker_gpu_metrics(fake_v2, monkeypatch):
    """Only the outer async worker profiled: inner workers inherit its config (``_nsight``
    in vLLM's ray_runtime_env) and still sample their own GPU."""
    monkeypatch.setenv("VLLM_RAY_PER_WORKER_GPUS", "0.5")
    module = sys.modules[V2_MODULE]
    outer = {"o": "'vllm_async_generation_worker_1:3_%p'", "t": "cuda-sw,nvtx"}
    monkeypatch.setattr(
        fake_v2, "_build_runtime_env", lambda self: {"_nsight": dict(outer)}
    )
    with patch("nemo_rl.utils.nsys.NRL_NSYS_EXTRA_OPTIONS", WORKER_GPU_METRICS):
        register_nsight_inner_worker_profiling()
        runtime_env = fake_v2(use_nsight=False, world_size=2)._build_runtime_env()
    assert runtime_env == {"nsight": {**outer, "gpu-metrics-devices": "worker"}}
    assert isinstance(module.ray, _WorkerGpuMetricsRay)
    with patch("nemo_rl.utils.nsys.NRL_NSYS_EXTRA_OPTIONS", {}):
        # Without the request the inherited config is left as it is.
        assert fake_v2(use_nsight=False, world_size=2)._build_runtime_env() == {
            "_nsight": outer
        }


def test_worker_gpu_metrics_ray_resolves_v1_options_and_delegates():
    """Only marked options are rewritten; unpinned workers do not sample; other calls pass through."""
    fake_ray = _FakeRay()
    wrapped = _WorkerGpuMetricsRay(fake_ray)
    shared = {"nsight": {"t": "cuda", "gpu-metrics-devices": "worker"}}
    with (
        patch("nemo_rl.utils.nsys.NRL_NSYS_EXTRA_OPTIONS", WORKER_GPU_METRICS),
        patch("nemo_rl.distributed.virtual_cluster.get_bundle_gpu_id", return_value=1),
    ):
        wrapped.remote(
            num_gpus=0.5, scheduling_strategy=_strategy(1), runtime_env=shared
        )
        # Not pinned to a bundle: the GPU is unknown, so no GPU metrics.
        wrapped.remote(num_gpus=0.5, runtime_env=shared)
        # Options without the marker pass through as they are.
        plain = {"runtime_env": {"nsight": {"t": "cuda"}}}
        wrapped.remote(**plain)
        # Ray's "any bundle" index: not pinned, so the GPU is unknown -> no GPU metrics.
        wrapped.remote(
            num_gpus=0.5, scheduling_strategy=_strategy(-1), runtime_env=shared
        )
    assert fake_ray.options[0]["runtime_env"]["nsight"] == {
        "t": "cuda",
        "gpu-metrics-devices": "1",
    }
    assert fake_ray.options[1]["runtime_env"]["nsight"] == {"t": "cuda"}
    assert fake_ray.options[2] == plain
    assert fake_ray.options[3]["runtime_env"]["nsight"] == {"t": "cuda"}
    assert wrapped.get("ref") == "ref"


def test_v1_executor_inner_workers_sample_their_own_gpu(monkeypatch):
    """v1 (in-process patch): each inner worker samples its bundle's GPU."""
    from nemo_rl.models.generation.vllm.vllm_worker import BaseVllmGenerationWorker

    v1_module_name = "vllm.v1.executor.ray_executor"

    class RayDistributedExecutor:
        def __init__(self):
            self.parallel_config = types.SimpleNamespace(world_size=2)

    RayDistributedExecutor.__module__ = v1_module_name
    module = types.ModuleType(v1_module_name)
    module.RayDistributedExecutor = RayDistributedExecutor
    module.ray = _FakeRay()
    fake_ray = module.ray
    monkeypatch.setitem(sys.modules, v1_module_name, module)
    monkeypatch.setenv("VLLM_RAY_PER_WORKER_GPUS", "0.5")
    with (
        patch("nemo_rl.utils.nsys.NRL_NSYS_PROFILE_STEP_RANGE", "1:3"),
        patch("nemo_rl.utils.nsys.NRL_NSYS_EXTRA_OPTIONS", WORKER_GPU_METRICS),
        patch(
            "nemo_rl.distributed.virtual_cluster.get_bundle_gpu_id",
            side_effect=lambda pg, bundle: bundle,
        ),
    ):
        BaseVllmGenerationWorker._patch_vllm_nsight_config()
        kwargs = RayDistributedExecutor()._configure_ray_workers_use_nsight({})
        # v1 creates each inner worker with ray.remote(**shared_kwargs)(cls).
        for bundle in (0, 1):
            module.ray.remote(
                num_gpus=0.5, scheduling_strategy=_strategy(bundle), **kwargs
            )
    nsight = kwargs["runtime_env"]["nsight"]
    assert nsight["capture-range-end"] == "repeat"
    assert nsight["o"] == "'vllm_tp_worker_1:3_%p'"
    devices = [
        o["runtime_env"]["nsight"]["gpu-metrics-devices"] for o in fake_ray.options
    ]
    assert devices == ["0", "1"]


@pytest.mark.vllm
@pytest.mark.parametrize(
    "module_name,method",
    [
        ("vllm.v1.executor.ray_executor_v2", "RayExecutorV2._init_executor"),
        ("vllm.v1.executor.ray_executor", "RayDistributedExecutor._init_workers_ray"),
    ],
)
def test_real_executors_create_workers_through_module_ray(module_name, method):
    """_WorkerGpuMetricsRay relies on vLLM creating each inner worker through the
    executor module's ``ray`` with a per-worker scheduling strategy and the shared
    runtime_env; fail at the vLLM bump if that changes."""
    import importlib
    import inspect

    module = importlib.import_module(module_name)
    cls_name, func_name = method.split(".")
    source = inspect.getsource(getattr(getattr(module, cls_name), func_name))
    assert hasattr(module, "ray")
    assert "ray.remote(" in source
    assert "PlacementGroupSchedulingStrategy(" in source
    assert "placement_group_bundle_index=" in source
