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

"""CUA baseline and FD update tests against explicit reference calculations."""

import copy
import json
import math
import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest
import torch
from cua_test_utils import create_optimizer
from logra.config import LoGRAConfig
from logra.cua import CuaConfig, TaskBaseline, fd_kl_step
from logra.optimizer import LoGRAOptimizer
from torch import Tensor, nn


def test_baseline_uses_previous_visits_and_masks_infrastructure_failures() -> None:
    baseline = TaskBaseline(CuaConfig())
    first = baseline.advantages(
        ["a", "a", "b"],
        ["v1", "v1", "v2"],
        torch.tensor([1.0, 0.0, float("nan")]),
        torch.tensor([True, True, False]),
    )
    # Both visits in this batch use the initial zero baseline.
    torch.testing.assert_close(first, torch.tensor([1.0, 0.0, 0.0]))
    assert baseline.counts == {("a", "v1"): (1.0, 2)}
    assert baseline.global_rate == pytest.approx(0.05)
    assert baseline.template_rates["v1"] == pytest.approx(0.095)
    second = baseline.advantages(
        ["a", "a"],
        ["v1", "v2"],
        torch.tensor([0.0, 1.0]),
        torch.ones(2, dtype=torch.bool),
    )
    torch.testing.assert_close(second, torch.tensor([-(1 + 2 * 0.095) / 4, 1 - 0.05]))


def test_baseline_checkpoint_has_no_live_aliases() -> None:
    original = TaskBaseline(CuaConfig())
    original.advantages(["a"], ["v1"], torch.ones(1), torch.ones(1, dtype=torch.bool))
    saved = original.state_dict()
    restored = TaskBaseline(CuaConfig())
    restored.load_state_dict(saved)
    args = (
        ["a", "b"],
        ["v1", "v2"],
        torch.tensor([0.0, 1.0]),
        torch.ones(2, dtype=torch.bool),
    )
    torch.testing.assert_close(original.advantages(*args), restored.advantages(*args))
    assert original.state_dict() == restored.state_dict()
    assert saved["counts"] == {("a", "v1"): (1.0, 1)}


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_trial_update_restores_weights_and_master_after_error(
    dtype: torch.dtype,
) -> None:
    net, optimizer = create_optimizer(dtype)
    optimizer.layers[0].sketch.fill_(0.2)
    optimizer.synchronize_and_clip(None, None)
    before = net[0].weight.detach().clone()
    state = copy.deepcopy(optimizer.state_dict())
    with pytest.raises(RuntimeError, match="probe failed"):
        with optimizer.trial_step(scale=3.0):
            assert not torch.equal(net[0].weight, before)
            raise RuntimeError("probe failed")
    torch.testing.assert_close(net[0].weight, before, rtol=0, atol=0)
    assert optimizer.update_count == 0
    for saved_id, values in state["sketch_optimizer"]["state"].items():
        for key, value in values.items():
            actual = optimizer.state_dict()["sketch_optimizer"]["state"][saved_id][key]
            torch.testing.assert_close(value, actual, rtol=0, atol=0)


def test_fd_step_matches_explicit_trial_kl_and_scaled_update() -> None:
    net, optimizer = create_optimizer()
    x = torch.tensor([[0.2, -0.4], [0.5, 0.9], [-0.1, 0.7]], requires_grad=True)
    net(x).square().sum().backward()
    norm = optimizer.synchronize_and_clip(None, None)
    layer = optimizer.layers[0]
    update = layer.sketch @ layer.projection
    before = net[0].weight.detach().clone()
    config = CuaConfig(max_update_ratio=0.2, kl_budget=1e-4)
    delta = update * (config.max_update_ratio * before.double().norm().item() / norm)
    reference = (x.detach() @ before.T).double().log_softmax(-1)
    trial = (x.detach() @ (before - delta).T).double().log_softmax(-1)
    expected_kl = (reference.exp() * (reference - trial)).sum(-1).mean().item()
    alpha = min(config.alpha_max, math.sqrt(config.kl_budget / expected_kl))

    def evaluate() -> Iterator[Tensor]:
        yield net(x.detach()).double().log_softmax(-1)

    metrics = fd_kl_step(
        optimizer, config, gradient_norm=norm, evaluate_logprobs=evaluate
    )
    assert metrics["kl_proposal"] == pytest.approx(expected_kl, rel=2e-5)
    assert metrics["kl_alpha"] == pytest.approx(alpha, rel=2e-5)
    assert metrics["kl_predicted"] == pytest.approx(config.kl_budget)
    torch.testing.assert_close(
        net[0].weight, before - delta * alpha, rtol=1e-5, atol=1e-7
    )
    realized = net(x.detach()).double().log_softmax(-1)
    actual_kl = (reference.exp() * (reference - realized)).sum(-1).mean().item()
    assert metrics["kl_realized"] == pytest.approx(actual_kl, rel=1e-8)
    assert optimizer.update_count == 1


def test_invalid_probe_restores_weights_without_committing() -> None:
    net, optimizer = create_optimizer()
    optimizer.layers[0].sketch.fill_(0.1)
    norm = optimizer.synchronize_and_clip(None, None)
    before = net[0].weight.detach().clone()

    def constant_probe() -> Iterator[Tensor]:
        yield torch.zeros(2, 3).log_softmax(-1)

    with pytest.raises(FloatingPointError, match="invalid KL"):
        fd_kl_step(
            optimizer, CuaConfig(), gradient_norm=norm, evaluate_logprobs=constant_probe
        )
    torch.testing.assert_close(net[0].weight, before, rtol=0, atol=0)
    assert optimizer.update_count == 0


def test_zero_gradient_does_not_probe_or_change_weights() -> None:
    net, optimizer = create_optimizer()
    norm = optimizer.synchronize_and_clip(None, None)
    before = net[0].weight.detach().clone()

    def unexpected_probe() -> Iterator[Tensor]:
        raise AssertionError("A zero direction must not run the model probe")
        yield torch.empty(0)

    metrics = fd_kl_step(
        optimizer, CuaConfig(), gradient_norm=norm, evaluate_logprobs=unexpected_probe
    )
    assert metrics["zero_gradient"] == 1.0
    torch.testing.assert_close(net[0].weight, before, rtol=0, atol=0)


def test_weight_only_restore_rebases_master_before_next_update() -> None:
    net, optimizer = create_optimizer(torch.bfloat16)
    with torch.no_grad():
        net[0].weight.fill_(0.75)
    optimizer.reset_master_weights()
    optimizer.layers[0].sketch.zero_()
    optimizer.synchronize_and_clip(None, None)
    optimizer.step()
    torch.testing.assert_close(net[0].weight, torch.full_like(net[0].weight, 0.75))


def test_evaluation_separates_macro_micro_and_binary_scores():
    from logra.osworld import DesktopTask, Episode, summarize_episodes

    tasks = [
        DesktopTask(str(i), str(i), domain, {})
        for i, domain in enumerate(["a", "a", "b"])
    ]
    episodes = [
        Episode(tasks[0], "v1", 1.0, True, None, []),
        Episode(tasks[1], "v1", 0.5, True, None, []),
        Episode(tasks[2], "v1", 0.0, False, "sandbox unavailable", []),
    ]
    metrics = summarize_episodes(episodes)
    assert metrics["score"] == pytest.approx(0.5)
    assert metrics["domain_macro_score"] == pytest.approx(0.375)
    assert metrics["success_rate"] == pytest.approx(1 / 3)
    assert metrics["infrastructure_failure_rate"] == pytest.approx(1 / 3)
    assert metrics["tasks"] == 3


def test_agent_keeps_historical_image_window_and_all_action_feedback(monkeypatch):
    import sys
    from pathlib import Path

    from logra.osworld import _DESKTOP, OSWorldConfig, _DesktopContext, make_agent

    class OriginalAgent:
        def __init__(self, **kwargs):
            self.observations = []
            self.kwargs = kwargs

        def predict(self, instruction, obs, **kwargs):
            self.observations.append(obs)
            return self.max_image_history_length, obs.copy(), kwargs["step_idx"]

    monkeypatch.setitem(
        sys.modules,
        "mm_agents.nvidia.nemotron_agent",
        SimpleNamespace(NemotronAgent=OriginalAgent),
    )
    monkeypatch.setitem(
        sys.modules,
        "desktop_env.execution_feedback",
        SimpleNamespace(render_feedback_batch=lambda values: "|".join(values)),
    )
    config = OSWorldConfig(
        **{
            name: Path("unused")
            for name in (
                "train_data",
                "validation_data",
                "train_task_base",
                "validation_task_base",
                "setup_cache",
                "template_capabilities",
                "task_requirements",
            )
        }
    )
    context = _DesktopContext(config, "v1")
    token = _DESKTOP.set(context)
    try:
        agent = make_agent(client_password="", log_context={})
        assert agent.kwargs["prompt_format_variant"] == "rfc0015_v1"
        lengths = []
        for step in range(18):
            context.feedback.extend([f"action-{step}-a", f"action-{step}-b"])
            length, observation, seen_step = agent.predict("task", {})
            lengths.append(length)
            assert (
                observation["execution_feedback_batch_text"]
                == f"action-{step}-a|action-{step}-b"
            )
            assert not context.feedback and seen_step == step
        assert lengths == list(range(1, 9)) + list(range(3, 11)) + [3, 4]
    finally:
        _DESKTOP.reset(token)


def test_frozen_embedding_allows_in_place_image_insertion() -> None:
    class ImageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(8, 4)
            self.q_proj = nn.Linear(4, 3, bias=False)

        def get_input_embeddings(self):
            return self.embedding

        def forward(self, ids):
            hidden = self.embedding(ids).view(-1, 4)
            hidden[1] = torch.arange(4, dtype=hidden.dtype)
            return self.q_proj(hidden)

    torch.manual_seed(12)
    model = ImageModel()
    reference = copy.deepcopy(model)
    ids = torch.tensor([[1, 2, 3]])
    reference(ids).square().sum().backward()
    opt = LoGRAOptimizer(
        model,
        torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0),
        LoGRAConfig(
            rank=2, optimizer="sgd", target_modules=["q_proj"], freeze_non_target=True
        ),
    )
    model(ids).square().sum().backward()
    layer = opt.layers[0]
    torch.testing.assert_close(
        layer.sketch, reference.q_proj.weight.grad @ layer.projection.T
    )
    assert model.embedding.weight.grad is None
    assert model.q_proj.weight.grad is None


def test_desktop_compatibility_filter_is_explicit(tmp_path) -> None:
    import json

    from logra.osworld import DesktopRollouts, DesktopTask, OSWorldConfig

    caps = tmp_path / "caps.json"
    needs = tmp_path / "needs.json"
    caps.write_text(
        json.dumps(
            {
                "templates": {
                    "osworld-slim": {"caps": {"browser": True}},
                    "osworld-v2": {"caps": {"browser": False}},
                }
            }
        )
    )
    needs.write_text(json.dumps([{"id": "task", "needs": ["browser"]}]))
    config = OSWorldConfig(
        train_data=tmp_path,
        validation_data=tmp_path,
        train_task_base=tmp_path,
        validation_task_base=tmp_path,
        setup_cache=tmp_path,
        template_capabilities=caps,
        task_requirements=needs,
    )
    rollouts = DesktopRollouts(
        config, None, "model", ["http://example.invalid/v1"], tmp_path
    )
    task = DesktopTask("task", "prompt", "chrome", {"id": "task"})
    assert rollouts.template_for(task, 0, False) == "osworld-v2"
    config.task_compatible_templates = True
    assert rollouts.template_for(task, 0, False) == "osworld-slim"
    assert rollouts.template_for(task, 0, True) == "osworld-v1"


def test_sandbox_credentials_do_not_survive_failed_rollouts(
    tmp_path, monkeypatch
) -> None:
    from logra.osworld import sandbox_credentials

    names = (
        "AENV_MODE",
        "AENV_URL",
        "AENV_API_KEY",
        "E2B_API_KEY",
        "E2B_API_URL",
        "E2B_SANDBOX_URL",
        "E2B_DOMAIN",
    )
    for name in names:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("E2B_API_URL", "previous-public-url")
    path = tmp_path / "credentials.json"
    path.write_text(
        json.dumps({"endpoint": "sandbox.example", "apikey": "test-only-key"})
    )
    with pytest.raises(RuntimeError, match="desktop failed"):
        with sandbox_credentials(path):
            assert os.environ["AENV_API_KEY"] == "test-only-key"
            assert os.environ["AENV_URL"] == "https://sandbox.example"
            # The SDK rewrites these globals while opening its local forwarder.
            os.environ["E2B_API_URL"] = "http://127.0.0.1:1234"
            raise RuntimeError("desktop failed")
    assert os.environ["E2B_API_URL"] == "previous-public-url"
    assert all(name not in os.environ for name in names if name != "E2B_API_URL")


@pytest.mark.parametrize(
    ("statuses", "valid", "score", "score_mode"),
    [
        ((500,), False, 0.0, "error_adjusted"),
        ((400,), True, 0.0, "error_adjusted"),
        ((500, 200), True, 0.75, "error_adjusted"),
        ((500,), False, 0.0, "verifier"),
        ((400,), True, 0.75, "verifier"),
        ((500, 200), True, 0.75, "verifier"),
    ],
)
def test_swallowed_model_errors_are_not_policy_failures(
    tmp_path, monkeypatch, statuses, valid, score, score_mode
) -> None:
    import sys
    from contextvars import ContextVar

    import requests
    from logra.osworld import DesktopRollouts, DesktopTask, OSWorldConfig, TurnRecorder

    responses = iter(statuses)

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, *args, **kwargs):
            response = requests.Response()
            response.status_code = next(responses)
            response._content = (
                b'{"error": "maximum model length exceeded"}'
                if response.status_code == 400
                else b'{"choices":[{"finish_reason":"stop","message":{"generation_token_ids":[1]}}]}'
            )
            return response

    def run_task(*args, messages_model_fn, **kwargs):
        # Match the real desktop agent: retry, then swallow the final HTTP error.
        for _ in statuses:
            try:
                messages_model_fn([], {"messages": []})
            except requests.HTTPError:
                pass
        return SimpleNamespace(
            runtime_eligible=True,
            evaluation_completed=True,
            runtime_admission_reason=None,
            score=0.75,
            error=None,
        )

    monkeypatch.setattr(requests, "Session", Session)
    monkeypatch.setattr(TurnRecorder, "record", lambda *args: {"content": "action"})
    monkeypatch.setitem(
        sys.modules,
        "desktop_env.providers.agentenv",
        SimpleNamespace(provider=SimpleNamespace(PRESET=ContextVar("test_preset"))),
    )
    monkeypatch.setitem(
        sys.modules,
        "responses_api_agents.osworld_agent.client",
        SimpleNamespace(run_osworld_task=run_task),
    )
    caps, needs = tmp_path / "caps.json", tmp_path / "needs.json"
    caps.write_text('{"templates":{}}')
    needs.write_text("[]")
    config = OSWorldConfig(
        train_data=tmp_path,
        validation_data=tmp_path,
        train_task_base=tmp_path,
        validation_task_base=tmp_path,
        setup_cache=tmp_path,
        template_capabilities=caps,
        task_requirements=needs,
        score_mode=score_mode,
    )
    rollouts = DesktopRollouts(
        config, None, "model", ["http://example.invalid/v1"], tmp_path
    )
    task = DesktopTask("task", "prompt", "chrome", {"id": "task"})
    episode = rollouts._episode(task, step=0, seed=42, index=0, evaluation=True)
    assert episode.valid is valid
    assert episode.score == score
    expected_adjusted = 0.0 if (not valid or statuses == (400,)) else 0.75
    assert episode.error_adjusted_score == expected_adjusted
    saved = json.loads((episode.artifact_dir / "episode.json").read_text())
    assert saved["score_mode"] == score_mode
    assert saved["score"] == score
    assert saved["verifier_score"] == 0.75
    assert saved["error_adjusted_score"] == expected_adjusted
    if not valid:
        assert episode.error == "Model service HTTPError: 500"
    else:
        assert episode.error is None


def test_generation_gate_finishes_turns_and_blocks_new_requests_during_refit() -> None:
    from logra.osworld import GenerationGate

    gate = GenerationGate()
    active, release, refitting, refitted, next_started = (Event() for _ in range(5))

    def active_request() -> None:
        with gate.request():
            active.set()
            assert release.wait(5)

    def refit() -> None:
        with gate.pause():
            refitting.set()
            assert refitted.wait(5)

    def next_request() -> None:
        with gate.request():
            next_started.set()

    with ThreadPoolExecutor(max_workers=3) as pool:
        current = pool.submit(active_request)
        assert active.wait(5)
        transfer = pool.submit(refit)
        assert not refitting.wait(0.05)
        release.set()
        current.result(timeout=5)
        assert refitting.wait(5)
        following = pool.submit(next_request)
        assert not next_started.wait(0.05)
        refitted.set()
        transfer.result(timeout=5)
        following.result(timeout=5)
        assert next_started.is_set()


def test_generation_gate_releases_after_request_and_refit_errors() -> None:
    from logra.osworld import GenerationGate

    gate = GenerationGate()
    with pytest.raises(ValueError, match="model failed"):
        with gate.request():
            raise ValueError("model failed")
    with pytest.raises(ValueError, match="transfer failed"):
        with gate.pause():
            raise ValueError("transfer failed")
    with gate.request():
        pass
    with gate.pause():
        pass


def test_prod4_prompt_baseline_matches_ungrouped_reference_after_resume() -> None:
    config = CuaConfig(baseline_prior=8, baseline_by_template=False)
    baseline = TaskBaseline(config)
    first = baseline.advantages(
        ["a", "a", "b"],
        ["v1", "v2", "v2"],
        torch.tensor([1.0, 0.0, float("nan")]),
        torch.tensor([True, True, False]),
    )
    torch.testing.assert_close(first, torch.tensor([1.0, 0.0, 0.0]))
    assert baseline.counts == {("a", ""): (1.0, 2)}
    assert baseline.global_rate == pytest.approx(0.05)
    assert baseline.template_rates == {}
    restored = TaskBaseline(config)
    restored.load_state_dict(baseline.state_dict())
    args = (
        ["a", "a", "b"],
        ["v2", "v1", "v2"],
        torch.tensor([0.0, 1.0, 1.0]),
        torch.ones(3, dtype=torch.bool),
    )
    expected = torch.tensor([-0.14, 0.86, 0.95])
    torch.testing.assert_close(baseline.advantages(*args), expected)
    torch.testing.assert_close(restored.advantages(*args), expected)
    assert restored.state_dict() == baseline.state_dict()


def test_prod4_cosine_schedule_matches_recorded_budgets_and_clamps() -> None:
    from logra.cua import kl_budget_at

    config = CuaConfig(kl_budget=5e-4, kl_budget_final=5e-5, kl_budget_steps=600)
    for step, expected in [
        (0, 5e-4),
        (300, 2.75e-4),
        (400, 1.625e-4),
        (600, 5e-5),
        (700, 5e-5),
    ]:
        assert kl_budget_at(config, step) == pytest.approx(expected)
    assert kl_budget_at(CuaConfig(), 400) == pytest.approx(5e-4)
    with pytest.raises(ValueError, match="nonnegative"):
        kl_budget_at(config, -1)


def test_fd_budget_uses_restored_optimizer_merge_count() -> None:
    net, optimizer = create_optimizer()
    optimizer.update_count = 400
    saved = copy.deepcopy(optimizer.state_dict())
    optimizer.update_count = 0
    optimizer.load_state_dict(saved)
    x = torch.tensor([[0.2, -0.4], [0.5, 0.9]], requires_grad=True)
    net(x).square().sum().backward()
    norm = optimizer.synchronize_and_clip(None, None)
    config = CuaConfig(
        max_update_ratio=0.2, kl_budget=5e-4, kl_budget_final=5e-5, kl_budget_steps=600
    )

    def probe() -> Iterator[Tensor]:
        yield net(x.detach()).double().log_softmax(-1)

    metrics = fd_kl_step(optimizer, config, gradient_norm=norm, evaluate_logprobs=probe)
    assert metrics["kl_budget"] == pytest.approx(1.625e-4)
    assert metrics["kl_predicted"] == pytest.approx(1.625e-4)
    assert metrics["kl_alpha"] == pytest.approx(
        math.sqrt(1.625e-4 / metrics["kl_proposal"])
    )
    assert optimizer.update_count == 401


def test_evaluation_reports_verifier_and_adjusted_scores_separately() -> None:
    from logra.osworld import DesktopTask, Episode, summarize_episodes

    task = DesktopTask("task", "prompt", "os", {"instruction": "do something"})
    episodes = [
        Episode(
            task,
            "v2",
            0.0,
            True,
            "context exhausted",
            [],
            verifier_score=1.0,
            context_exhausted=True,
        ),
        Episode(task, "v2", 0.5, True, None, [], verifier_score=0.5),
    ]
    metrics = summarize_episodes(episodes)
    assert metrics["score"] == 0.25
    assert metrics["success_rate"] == 0.0
    assert metrics["verifier_score"] == 0.75
    assert metrics["verifier_success_rate"] == 0.5
    assert metrics["score_override_rate"] == 0.5


def test_prod4_recipe_selects_only_v2_after_config_inheritance() -> None:
    from nemo_rl.utils.config import load_config

    recipe = Path(__file__).parents[2] / "configs" / "cua_prod4.yaml"
    config = load_config(recipe)
    assert dict(config.osworld.template_weights) == {"osworld-v2": 1.0}
    assert config.osworld.evaluation_template == "osworld-v2"
    assert config.cua.baseline_by_template is False
    assert config.cua.kl_budget_steps == 600
    assert config.osworld.score_mode == "verifier"
