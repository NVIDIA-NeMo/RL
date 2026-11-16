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

import io
import sys
import time
from argparse import Namespace
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from examples import run_grpo_single_controller
from nemo_rl.algorithms.grpo import GRPOConfig
from nemo_rl.algorithms.metric_utils import SetupTimingMetrics
from nemo_rl.algorithms.single_controller_utils.config import (
    AsyncRLConfig,
    MasterConfig,
    WatchdogConfig,
)
from nemo_rl.models.policy.draft_config import Eagle3DraftConfig
from nemo_rl.utils.logger import LoggerConfig


@pytest.fixture
def main_context(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    generation_config = {"backend": "vllm"}
    config = MasterConfig.model_construct(
        policy={
            "tokenizer": {},
            "generation": generation_config,
            "draft": Eagle3DraftConfig(enabled=False),
            "megatron_cfg": {"mtp_num_layers": 2},
        },
        data={},
        env={},
        data_plane={"enabled": True, "impl": "transfer_queue", "backend": "simple"},
        logger=LoggerConfig(log_dir="/tmp/logs"),
        checkpointing={"enabled": False},
        async_rl=AsyncRLConfig(),
        grpo=GRPOConfig(async_grpo=None),
    )
    configured_generation = {"backend": "vllm", "_mtp_weights_from_refit": True}
    configure_generation = MagicMock(return_value=configured_generation)
    materialize_video_config = MagicMock()
    actor = SimpleNamespace(run=SimpleNamespace(remote=MagicMock(return_value="run")))
    actor_args = SimpleNamespace(
        env_handles={},
        gen_handle=SimpleNamespace(shutdown=MagicMock()),
        trainer_handle=SimpleNamespace(shutdown=MagicMock()),
        value_handle=None,
    )
    setup_single_controller = MagicMock(return_value=(actor_args, SetupTimingMetrics()))
    ray_get = MagicMock(return_value={})
    # The driver now polls ping() around the run. Report the run as ready on the first
    # check so these tests keep exercising the same path they always did.
    ray_wait = MagicMock(side_effect=lambda refs, timeout=None: (list(refs), []))
    shutdown_environments = MagicMock()

    monkeypatch.setattr(
        run_grpo_single_controller,
        "parse_args",
        lambda: (Namespace(config="config.yaml"), []),
    )
    monkeypatch.setattr(run_grpo_single_controller, "load_config", lambda _: {})
    monkeypatch.setattr(
        run_grpo_single_controller.OmegaConf,
        "to_container",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(run_grpo_single_controller, "MasterConfig", lambda **_: config)
    monkeypatch.setattr(run_grpo_single_controller, "init_ray", lambda: None)
    monkeypatch.setattr(
        run_grpo_single_controller,
        "get_tokenizer",
        lambda _: "tokenizer",
    )
    monkeypatch.setattr(
        run_grpo_single_controller,
        "get_next_experiment_dir",
        lambda _: "/tmp/logs/0",
    )
    monkeypatch.setattr(
        run_grpo_single_controller,
        "configure_generation_config",
        configure_generation,
    )
    monkeypatch.setattr(
        run_grpo_single_controller,
        "materialize_vllm_video_config",
        materialize_video_config,
    )
    monkeypatch.setattr(
        run_grpo_single_controller,
        "setup_single_controller",
        setup_single_controller,
    )
    actor_remote = MagicMock(return_value=actor)
    monkeypatch.setattr(
        run_grpo_single_controller.SingleControllerActor,
        "remote",
        actor_remote,
    )
    # .options() returns a fresh actor class on the real thing, so route it back
    # to the same remote or the patch above is bypassed.
    actor_options = MagicMock(return_value=SimpleNamespace(remote=actor_remote))
    monkeypatch.setattr(
        run_grpo_single_controller.SingleControllerActor,
        "options",
        actor_options,
    )
    # No dedicated head in these tests, so the driver takes the unpinned path.
    monkeypatch.setattr(run_grpo_single_controller.ray, "cluster_resources", lambda: {})
    monkeypatch.setattr(run_grpo_single_controller.ray, "get", ray_get)
    monkeypatch.setattr(run_grpo_single_controller.ray, "wait", ray_wait)
    monkeypatch.setattr(
        run_grpo_single_controller,
        "shutdown_environments",
        shutdown_environments,
    )

    return SimpleNamespace(
        actor_args=actor_args,
        actor_options=actor_options,
        config=config,
        configure_generation=configure_generation,
        configured_generation=configured_generation,
        generation_config=generation_config,
        materialize_video_config=materialize_video_config,
        ray_get=ray_get,
        ray_wait=ray_wait,
        setup_single_controller=setup_single_controller,
        shutdown_environments=shutdown_environments,
    )


def test_cleanup_is_best_effort_and_preserves_run_error(
    main_context: SimpleNamespace,
    capsys: pytest.CaptureFixture[str],
) -> None:
    env_handles = {"nemo_gym": MagicMock()}
    generation = SimpleNamespace(
        shutdown=MagicMock(side_effect=RuntimeError("generation cleanup failed"))
    )
    trainer = SimpleNamespace(shutdown=MagicMock())
    main_context.actor_args.env_handles = env_handles
    main_context.actor_args.gen_handle = generation
    main_context.actor_args.trainer_handle = trainer

    def get(ref: object, timeout: float | None = None) -> None:
        del timeout
        if ref == "run":
            raise RuntimeError("training failed")
        return None

    main_context.ray_get.side_effect = get

    with pytest.raises(RuntimeError, match="training failed"):
        run_grpo_single_controller.main()

    # Env teardown semantics (bounded wait, kill fallback) belong to the shared
    # helper and are covered by its own tests; here the driver only has to reach
    # it before the generation and trainer handles.
    main_context.shutdown_environments.assert_called_once_with(env_handles)
    generation.shutdown.assert_called_once_with()
    trainer.shutdown.assert_called_once_with()
    output = capsys.readouterr().out
    assert "Generation shutdown failed: generation cleanup failed" in output


def test_a_dedicated_ray_head_claims_the_whole_unit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        run_grpo_single_controller.ray,
        "cluster_resources",
        lambda: {"ray_head": 1.0, "GPU": 8.0},
    )

    assert run_grpo_single_controller._controller_placement() == {
        "resources": {"ray_head": 1}
    }


def test_no_dedicated_head_leaves_the_controller_unpinned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ray.sub advertises ray_head only under DEDICATED_RAY_HEAD=1.

    Read from the live cluster rather than the env var, so a launcher that asked
    for a head it did not get falls back instead of pinning to nothing.
    """
    monkeypatch.setattr(
        run_grpo_single_controller.ray,
        "cluster_resources",
        lambda: {"GPU": 8.0},
    )

    assert run_grpo_single_controller._controller_placement() == {}


def test_main_applies_the_placement_when_launching_the_controller(
    main_context: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Overrides the fixture's headless cluster.
    monkeypatch.setattr(
        run_grpo_single_controller.ray,
        "cluster_resources",
        lambda: {"ray_head": 1.0},
    )

    run_grpo_single_controller.main()

    main_context.actor_options.assert_called_once_with(resources={"ray_head": 1})


def test_main_requests_no_resource_without_a_dedicated_head(
    main_context: SimpleNamespace,
) -> None:
    """The fixture's cluster has no head, which is every non-ray.sub run."""
    run_grpo_single_controller.main()

    main_context.actor_options.assert_called_once_with()


def _watch_a_controller_that_never_finishes(
    monkeypatch: pytest.MonkeyPatch, pongs: int
) -> None:
    """Drive the liveness watch with a run that never completes and ``pongs`` replies."""
    remaining = pongs

    def never_ready(refs: Any, timeout: float | None = None) -> tuple[list, list]:
        # Real ray.wait blocks for the interval; without that the watch spins hot
        # and never reaches stall_timeout_s.
        time.sleep(timeout or 0)
        return [], list(refs)

    def ping(_ref: Any, timeout: float | None = None) -> None:
        nonlocal remaining
        if remaining <= 0:
            raise TimeoutError("ping timed out")
        remaining -= 1

    monkeypatch.setattr(run_grpo_single_controller.ray, "wait", never_ready)
    monkeypatch.setattr(run_grpo_single_controller.ray, "get", ping)


def test_a_controller_that_never_answers_is_reported_as_unstarted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pin to a full head leaves the actor PENDING, which is not a stalled loop."""
    _watch_a_controller_that_never_finishes(monkeypatch, pongs=0)
    sc = SimpleNamespace(
        run=SimpleNamespace(remote=MagicMock(return_value="run")),
        ping=SimpleNamespace(remote=MagicMock(return_value="ping")),
    )

    with pytest.raises(RuntimeError, match="has not started") as excinfo:
        run_grpo_single_controller._run_with_controller_liveness_watch(
            sc, WatchdogConfig(interval_s=0.01, stall_timeout_s=0.02)
        )

    assert "the head has a free CPU" in str(excinfo.value)


def test_a_controller_that_answered_once_is_reported_as_stalled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once it has replied it was schedulable, so a later silence is the event loop."""
    _watch_a_controller_that_never_finishes(monkeypatch, pongs=1)
    sc = SimpleNamespace(
        run=SimpleNamespace(remote=MagicMock(return_value="run")),
        ping=SimpleNamespace(remote=MagicMock(return_value="ping")),
    )

    with pytest.raises(RuntimeError, match="event loop has been unresponsive"):
        run_grpo_single_controller._run_with_controller_liveness_watch(
            sc, WatchdogConfig(interval_s=0.01, stall_timeout_s=0.02)
        )


def test_main_configures_generation_for_trained_mtp(
    main_context: SimpleNamespace,
) -> None:
    run_grpo_single_controller.main()

    main_context.configure_generation.assert_called_once_with(
        main_context.generation_config,
        "tokenizer",
        has_refit_draft_weights=False,
        trains_mtp=True,
        draft_full_refit=False,
    )
    assert (
        main_context.config.policy["generation"] is main_context.configured_generation
    )


def test_main_materializes_video_contract_before_worker_setup(
    main_context: SimpleNamespace,
) -> None:
    run_grpo_single_controller.main()

    main_context.materialize_video_config.assert_called_once_with(
        main_context.config.policy, main_context.config.data
    )


def test_main_accepts_policy_without_draft_config(
    main_context: SimpleNamespace,
) -> None:
    main_context.config.policy.pop("draft")

    run_grpo_single_controller.main()

    main_context.configure_generation.assert_called_once_with(
        main_context.generation_config,
        "tokenizer",
        has_refit_draft_weights=False,
        trains_mtp=True,
        draft_full_refit=False,
    )


def test_main_preserves_generation_config_through_setup(
    main_context: SimpleNamespace,
) -> None:
    """main() forwards normalized generation config to setup_single_controller."""
    main_context.generation_config["vllm_cfg"] = {"refit_with_reload_api": True}
    main_context.configure_generation.side_effect = (
        lambda generation, *_args, **_kwargs: generation
    )

    run_grpo_single_controller.main()

    config_for_setup = main_context.setup_single_controller.call_args.args[0]
    assert (
        config_for_setup.policy["generation"]["vllm_cfg"]["refit_with_reload_api"]
        is True
    )


def test_main_passes_processor_for_vlm(
    main_context: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = SimpleNamespace(tokenizer="vlm-tokenizer")
    get_tokenizer = MagicMock(return_value=processor)
    setup_single_controller = MagicMock(
        return_value=(main_context.actor_args, SetupTimingMetrics())
    )
    main_context.config.policy["is_vlm"] = True
    monkeypatch.setattr(run_grpo_single_controller, "get_tokenizer", get_tokenizer)
    monkeypatch.setattr(
        run_grpo_single_controller,
        "setup_single_controller",
        setup_single_controller,
    )

    run_grpo_single_controller.main()

    get_tokenizer.assert_called_once_with(
        main_context.config.policy["tokenizer"], get_processor=True
    )
    setup_single_controller.assert_called_once_with(
        main_context.config, "vlm-tokenizer", processor=processor
    )


def test_main_line_buffers_driver_stdout(
    main_context: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forwarded actor prints must not sit in a block buffer when stdout is a file."""
    stdout = io.TextIOWrapper(io.BytesIO())
    monkeypatch.setattr(sys, "stdout", stdout)

    run_grpo_single_controller.main()

    assert stdout.line_buffering
