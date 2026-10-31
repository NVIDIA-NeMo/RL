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

from nemo_rl.utils.vllm_nsight_plugin import register_nsight_inner_worker_profiling

V2_MODULE = "vllm.v1.executor.ray_executor_v2"
PLUGIN_LOGGER = "nemo_rl.utils.vllm_nsight_plugin"


@pytest.fixture
def fake_v2(monkeypatch):
    """Stub RayExecutorV2 (no GPU/vLLM needed) and enable NeMo-RL nsys profiling.

    The class is created per test and installed under its full module name, so the
    class-level patch never touches the real vLLM executor and nothing leaks.
    """

    class RayExecutorV2:
        def __init__(self, use_nsight=True):
            self.parallel_config = types.SimpleNamespace(
                ray_workers_use_nsight=use_nsight
            )

        def _build_runtime_env(self):
            return {"env_vars": {"KEEP": "1"}}

    module = types.ModuleType(V2_MODULE)
    module.RayExecutorV2 = RayExecutorV2
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
