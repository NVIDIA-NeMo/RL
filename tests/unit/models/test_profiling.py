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
"""Four-phase capture contract shared by the GRPO driver and GPU workers."""

import inspect
import sys
import types
import uuid
from unittest.mock import MagicMock, patch

import pytest

from nemo_rl.models import profiling
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.models.profiling import (
    POLICY_PROFILER_METHODS,
    ROLLOUT_PROFILER_METHODS,
    GrpoCapture,
    WorkerCapture,
    actor_profiler_selected,
    load_profiler,
    profile_phase,
    profile_policy_method,
    profile_refit,
    selected_profiler_class,
    validate_rollout_profiler_topology,
)


@pytest.fixture
def four_phase(monkeypatch):
    for name in ("NTRACE_RUN_ID", "NEMO_RL_OTEL_RUN_ID", "NEMO_LENS_RUN_ID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NRL_NTRACE_FOUR_PHASE", "1")
    monkeypatch.setenv("NRL_POLICY_PROFILER_CLASS", "plugin.Policy")
    monkeypatch.setenv("NRL_ROLLOUT_PROFILER_CLASS", "plugin.Rollout")


@pytest.fixture
def plugin_module(monkeypatch):
    module = types.ModuleType("fake_profiler_plugin")

    class Plugin:
        def __init__(self, *, rank):
            self.rank = rank

    for method in ROLLOUT_PROFILER_METHODS:
        setattr(Plugin, method, lambda self, *args, **kwargs: None)

    class Incomplete:
        def __init__(self, *, rank):
            pass

        close = Plugin.close

    class Broken(Plugin):
        def __init__(self, *, rank):
            raise ValueError("bad rank")

    module.Plugin = Plugin
    module.Incomplete = Incomplete
    module.Broken = Broken
    module.not_a_class = object()
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


class Target:
    """A Ray-facing policy or rollout object that records capture commands."""

    def __init__(self, role, events, fail=None):
        self.role = role
        self.events = events
        self.fail = fail or {}

    def full_step_profile(self, command, **kwargs):
        self.events.append((self.role, command, kwargs))
        if command in self.fail:
            raise self.fail[command]


def commands(events):
    return [(role, command) for role, command, _ in events]


@pytest.mark.parametrize(
    "required", [POLICY_PROFILER_METHODS, ROLLOUT_PROFILER_METHODS]
)
def test_loader_constructs_profiler_with_rank(plugin_module, required):
    profiler = load_profiler(
        "fake_profiler_plugin.Plugin", rank=7, required_methods=required
    )

    assert isinstance(profiler, plugin_module.Plugin)
    assert profiler.rank == 7


@pytest.mark.parametrize(
    ("class_path", "error", "match"),
    [
        ("Plugin", ValueError, "fully qualified"),
        ("module.", ValueError, "fully qualified"),
        ("missing_profiler_plugin.Plugin", RuntimeError, "Could not import"),
        ("fake_profiler_plugin.Missing", RuntimeError, "does not resolve"),
        ("fake_profiler_plugin.not_a_class", RuntimeError, "does not resolve"),
        (
            "fake_profiler_plugin.Incomplete",
            RuntimeError,
            "begin_engine_initialization",
        ),
        ("fake_profiler_plugin.Broken", RuntimeError, "for rank 3"),
    ],
)
def test_loader_fails_clearly(plugin_module, class_path, error, match):
    with pytest.raises(error, match=match):
        load_profiler(class_path, rank=3, required_methods=ROLLOUT_PROFILER_METHODS)


@pytest.mark.parametrize(
    ("four_phase", "selector", "expected"),
    [("0", "", ""), ("1", "", ""), ("1", "plugin.Policy", "plugin.Policy")],
)
def test_selector_is_lazy_and_opt_in(monkeypatch, four_phase, selector, expected):
    monkeypatch.setenv("NRL_NTRACE_FOUR_PHASE", four_phase)
    monkeypatch.setenv("NRL_POLICY_PROFILER_CLASS", selector)
    with patch.object(profiling.importlib, "import_module") as importer:
        assert selected_profiler_class("NRL_POLICY_PROFILER_CLASS") == expected
    importer.assert_not_called()


def test_selector_without_four_phase_is_rejected(monkeypatch):
    monkeypatch.delenv("NRL_NTRACE_FOUR_PHASE", raising=False)
    monkeypatch.setenv("NRL_ROLLOUT_PROFILER_CLASS", "plugin.Rollout")

    with pytest.raises(ValueError, match="requires NRL_NTRACE_FOUR_PHASE=1"):
        selected_profiler_class("NRL_ROLLOUT_PROFILER_CLASS")


@pytest.mark.parametrize(
    ("driver", "actor_env_vars", "expected"),
    [
        ("plugin.Rollout", None, True),
        ("plugin.Rollout", {"NRL_ROLLOUT_PROFILER_CLASS": ""}, False),
        ("", {"NRL_ROLLOUT_PROFILER_CLASS": "plugin.Rollout"}, True),
        ("", {}, False),
    ],
)
def test_actor_settings_override_the_driver_selector(
    monkeypatch, driver, actor_env_vars, expected
):
    monkeypatch.setenv("NRL_ROLLOUT_PROFILER_CLASS", driver)

    assert (
        actor_profiler_selected("NRL_ROLLOUT_PROFILER_CLASS", actor_env_vars)
        is expected
    )


@pytest.mark.parametrize(("tp", "pp", "ep"), [(8, 2, 1), (8, 1, 2), (8, 1, 16)])
def test_rollout_topology_rejects_unsupported_layouts(tp, pp, ep):
    with pytest.raises(ValueError, match="Rollout profiling"):
        validate_rollout_profiler_topology(
            tensor_parallel_size=tp,
            pipeline_parallel_size=pp,
            expert_parallel_size=ep,
        )


@pytest.mark.parametrize(("tp", "ep"), [(1, 1), (8, 1), (8, 8)])
def test_rollout_topology_accepts_supported_layouts(tp, ep):
    validate_rollout_profiler_topology(
        tensor_parallel_size=tp, pipeline_parallel_size=1, expert_parallel_size=ep
    )


def test_worker_capture_keeps_phase_tokens_in_the_worker():
    profiler = MagicMock()
    token = object()
    profiler.begin_phase.return_value = token
    capture = WorkerCapture(profiler, lambda: "GPU-test")

    with pytest.raises(RuntimeError, match="not configured"):
        capture.dispatch("begin_step", step_id=1)

    capture.dispatch("configure_capture", run_id="run", role="policy")
    # Setup and warmup calls outside a step do not own capture windows.
    assert capture.dispatch("begin_phase", name="warmup", phase_slot="training") is None
    capture.dispatch("end_phase")
    capture.dispatch("begin_step", step_id=1, attempt=0, weight_version=0)
    assert capture.dispatch("begin_phase", name="train", phase_slot="training") is None
    with pytest.raises(RuntimeError, match="already open"):
        capture.dispatch("begin_phase", name="other", phase_slot="training")
    capture.dispatch("end_phase")
    capture.dispatch("finish_step")

    profiler.configure_capture.assert_called_once_with(
        device_uuid="GPU-test", run_id="run", role="policy"
    )
    profiler.begin_phase.assert_called_once_with(name="train", phase_slot="training")
    profiler.end_phase.assert_called_once_with(token)
    profiler.finish_step.assert_called_once_with()
    with pytest.raises(ValueError, match="Unknown"):
        capture.dispatch("resume")


def test_worker_capture_abort_clears_open_phase_even_if_profiler_fails():
    profiler = MagicMock()
    profiler.abort_step.side_effect = RuntimeError("save failed")
    capture = WorkerCapture(profiler, lambda: "GPU-test")
    capture.dispatch("configure_capture")
    capture.dispatch("begin_step", step_id=1)
    capture.dispatch("begin_phase", name="train", phase_slot="training")

    with pytest.raises(RuntimeError, match="save failed"):
        capture.dispatch("abort_step", reason="failed")

    assert (capture.step_open, capture.phase_open, capture.phase_token) == (
        False,
        False,
        None,
    )
    capture.dispatch("end_phase")
    profiler.end_phase.assert_not_called()


def test_profile_phase_is_inert_without_four_phase(monkeypatch):
    monkeypatch.setenv("NRL_NTRACE_FOUR_PHASE", "0")
    events = []

    with profile_phase((Target("policy", events),), name="x", phase_slot="refit"):
        pass

    assert events == []


@pytest.mark.parametrize("abort_fails", [False, True])
def test_profile_phase_aborts_entered_roles_and_keeps_the_workload_error(
    four_phase, abort_fails
):
    events = []
    abort_error = {"abort_step": RuntimeError("abort failed")} if abort_fails else {}
    policy = Target("policy", events, fail=abort_error)
    rollout = Target("rollout", events)

    with pytest.raises(KeyError, match="workload") as exc_info:
        with profile_phase((policy, rollout), name="weight_refit", phase_slot="refit"):
            raise KeyError("workload")

    assert commands(events) == [
        ("policy", "begin_phase"),
        ("rollout", "begin_phase"),
        ("rollout", "abort_step"),
        ("policy", "abort_step"),
    ]
    assert events[-1][2] == {"reason": "weight_refit_failed"}
    notes = getattr(exc_info.value, "__notes__", [])
    assert any("abort failed" in note for note in notes) is abort_fails


def test_refit_phase_spans_both_roles_with_weight_version(four_phase):
    events = []

    @profile_refit
    def refit(policy, generation, colocated, *, profile_weight_version=None):
        events.append(("driver", "refit", {"colocated": colocated}))
        return "metrics"

    policy, rollout = Target("policy", events), Target("rollout", events)
    assert refit(policy, rollout, True, profile_weight_version=4) == "metrics"

    assert commands(events) == [
        ("policy", "begin_phase"),
        ("rollout", "begin_phase"),
        ("driver", "refit"),
        ("rollout", "end_phase"),
        ("policy", "end_phase"),
    ]
    assert events[0][2] == {
        "name": "weight_refit",
        "phase_slot": "refit",
        "labels": {"weight_version": 4},
    }


@pytest.mark.parametrize(
    ("method", "name", "slot"),
    [
        ("train", "policy_training", "training"),
        ("prepare_for_training", "policy_training_onload", "training"),
        ("get_logprobs", "previous_policy_logprobs", "logprobs"),
        ("get_reference_policy_logprobs", "reference_policy_logprobs", "logprobs"),
        ("prepare_for_lp_inference", "policy_logprobs_onload", "logprobs"),
        ("finish_inference", "policy_logprobs_offload", "logprobs"),
    ],
)
def test_policy_operations_are_capture_phases(method, name, slot):
    closure = inspect.getclosurevars(getattr(Policy, method)).nonlocals
    assert (closure["name"], closure["phase_slot"]) == (name, slot)


def test_policy_method_phase_wraps_the_actual_call(four_phase):
    events = []

    class FakePolicy(Target):
        @profile_policy_method("policy_training", phase_slot="training")
        def train(self, data):
            self.events.append(("policy", "train", {"data": data}))
            return "result"

    assert FakePolicy("policy", events).train("batch") == "result"
    assert commands(events) == [
        ("policy", "begin_phase"),
        ("policy", "train"),
        ("policy", "end_phase"),
    ]


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({"NTRACE_RUN_ID": "explicit", "NEMO_RL_OTEL_RUN_ID": "otel"}, "explicit"),
        ({"NEMO_RL_OTEL_RUN_ID": " otel ", "NEMO_LENS_RUN_ID": "lens"}, "otel"),
        ({"NEMO_RL_OTEL_RUN_ID": " ", "NEMO_LENS_RUN_ID": " lens "}, "lens"),
        ({"NEMO_RL_OTEL_RUN_ID": "", "NEMO_LENS_RUN_ID": " "}, "generated"),
    ],
)
def test_capture_configures_both_roles_with_one_run_id(
    four_phase, monkeypatch, env, expected
):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    events = []
    with patch.object(uuid, "uuid4", return_value="generated"):
        GrpoCapture(
            Target("policy", events),
            Target("rollout", events),
            schedule="async",
            colocated=False,
        )

    assert events == [
        (
            role,
            "configure_capture",
            {
                "run_id": expected,
                "role": role,
                "schedule": "async",
                "colocated": False,
            },
        )
        for role in ("policy", "rollout")
    ]


def test_capture_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("NRL_NTRACE_FOUR_PHASE", raising=False)
    events = []
    capture = GrpoCapture(
        Target("policy", events),
        Target("rollout", events),
        schedule="sync",
        colocated=True,
    )

    with capture.step(step_id=1, weight_version=0):
        pass
    capture.begin_policy_step(step_id=1, weight_version=0)
    capture.finish_policy_step()
    capture.abort_all(RuntimeError("x"), reason="x")
    capture.close()

    assert capture.enabled is False
    assert events == []


def test_capture_requires_both_role_profilers(four_phase, monkeypatch):
    monkeypatch.setenv("NRL_ROLLOUT_PROFILER_CLASS", "")

    with pytest.raises(ValueError, match="both policy and rollout"):
        GrpoCapture(MagicMock(), MagicMock(), schedule="sync", colocated=True)


@pytest.mark.parametrize("fail_on", [None, "workload", ("rollout", "finish_step")])
def test_sync_step_owns_both_roles(four_phase, fail_on):
    events = []
    fail = {}
    if isinstance(fail_on, tuple):
        fail = {fail_on[1]: RuntimeError("finish failed")}
    policy = Target("policy", events)
    rollout = Target("rollout", events, fail=fail)
    capture = GrpoCapture(policy, rollout, schedule="sync", colocated=True)
    events.clear()

    def run_step():
        with capture.step(step_id=3, weight_version=2, attempt=1):
            if fail_on == "workload":
                raise KeyError("workload")

    if fail_on is None:
        run_step()
    else:
        with pytest.raises((KeyError, RuntimeError)):
            run_step()

    begin = {"step_id": 3, "attempt": 1, "weight_version": 2}
    assert events[:2] == [
        ("policy", "begin_step", begin),
        ("rollout", "begin_step", begin),
    ]
    if fail_on is None:
        assert commands(events[2:]) == [
            ("rollout", "finish_step"),
            ("policy", "finish_step"),
        ]
    else:
        assert commands(events[-2:]) == [
            ("rollout", "abort_step"),
            ("policy", "abort_step"),
        ]
        assert events[-1][2] == {"reason": "grpo_step_failed"}


def test_async_policy_step_retains_bootstrap_window(four_phase):
    events = []
    capture = GrpoCapture(
        Target("policy", events),
        Target("rollout", events),
        schedule="async",
        colocated=False,
    )
    events.clear()

    # The trainer opens step 1 before the initial refit; receiving the first
    # batch must not open it again.
    capture.begin_policy_step(step_id=1, weight_version=0)
    capture.begin_policy_step(step_id=1, weight_version=0)
    capture.finish_policy_step()
    capture.finish_policy_step()
    capture.begin_policy_step(step_id=2, weight_version=1)
    capture.abort_policy_step(reason="async_grpo_step_failed")
    capture.abort_policy_step(reason="async_grpo_step_failed")

    assert commands(events) == [
        ("policy", "begin_step"),
        ("policy", "finish_step"),
        ("policy", "begin_step"),
        ("policy", "abort_step"),
    ]


def test_setup_abort_and_close_report_every_role(four_phase):
    events = []
    policy = Target("policy", events, fail={"close": RuntimeError("policy close")})
    rollout = Target(
        "rollout",
        events,
        fail={
            "abort_step": RuntimeError("rollout abort"),
            "close": RuntimeError("rollout close"),
        },
    )
    capture = GrpoCapture(policy, rollout, schedule="async", colocated=False)
    setup_error = RuntimeError("refit failed")

    capture.abort_all(setup_error, reason="async_generation_setup_failed")
    with pytest.raises(RuntimeError, match="policy close") as exc_info:
        capture.close()

    assert any("rollout abort" in note for note in setup_error.__notes__)
    assert any("rollout close" in note for note in exc_info.value.__notes__)
    assert commands(events[-2:]) == [("policy", "close"), ("rollout", "close")]
