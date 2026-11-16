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

"""Driver-side launch path shared by the SingleController entrypoints.

``examples/run_grpo_single_controller.py`` (training) and
``examples/run_single_controller_simulator.py`` (simulator) parse their own
flags and hand over to :func:`run_single_controller`.
"""

import pprint
import sys
from typing import Any, Optional, cast

import ray
from omegaconf import OmegaConf
from transformers import PreTrainedTokenizerBase

from nemo_rl.algorithms.single_controller import SingleControllerActor
from nemo_rl.algorithms.single_controller_utils.config import (
    MasterConfig,
    is_ppo_run,
)
from nemo_rl.algorithms.single_controller_utils.setup import setup_single_controller
from nemo_rl.algorithms.single_controller_utils.simulator import (
    SIMULATOR_CONFIG_KEY,
    SimulatorConfig,
    apply_simulator_overrides,
    reject_online_quantization,
)
from nemo_rl.algorithms.single_controller_utils.watchdog import (
    run_with_controller_liveness_watch,
)
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data_plane.factory import maybe_configure_data_plane_env
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.nemo_gym import setup_nemo_gym_config
from nemo_rl.environments.utils import shutdown_environments
from nemo_rl.models.generation import (
    configure_generation_config,
    draft_full_refit_enabled,
    maybe_configure_engine_reaping_env,
)
from nemo_rl.models.generation.vllm.config import materialize_vllm_video_config
from nemo_rl.models.policy.draft_config import draft_refit_enabled
from nemo_rl.telemetry.instrumentation import setup_span, startup_span
from nemo_rl.telemetry.setup import init_telemetry_driver, shutdown_telemetry
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import get_next_experiment_dir
from nemo_rl.utils.outdated_config_checks import check_outdated_config


def controller_placement() -> dict[str, Any]:
    """Pin the controller to ray.sub's dedicated head (the ``ray_head`` resource) when the cluster has one.

    Returns:
        ``SingleControllerActor.options`` kwargs; empty when there is no ``ray_head``.
    """
    if ray.cluster_resources().get("ray_head", 0) >= 1:
        return {"resources": {"ray_head": 1}}
    return {}


def _load_configs(
    config_path: str,
    overrides: list[str],
    *,
    simulator_config_path: Optional[str],
) -> tuple[MasterConfig, Optional[SimulatorConfig]]:
    """Resolve the recipe, and the simulator YAML of a simulator run, with the overrides applied.

    Args:
        config_path: The SC recipe YAML.
        overrides: Hydra-style overrides; ``simulator.<key>=<value>`` addresses
            the simulator YAML, anything else the recipe.
        simulator_config_path: The simulator YAML, or None on a training run.

    Returns:
        The validated recipe, and the simulator config (None on a training run).
    """
    config = load_config(config_path)
    print(f"Loaded configuration from: {config_path}")
    if simulator_config_path is not None:
        if SIMULATOR_CONFIG_KEY in config:
            raise ValueError(
                f"{config_path} has a top-level `{SIMULATOR_CONFIG_KEY}` key, which a "
                "simulator run reserves for the simulator config; rename or drop it."
            )
        config[SIMULATOR_CONFIG_KEY] = load_config(simulator_config_path)
        print(f"Loaded simulator configuration from: {simulator_config_path}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    raw_config = cast(dict[str, Any], OmegaConf.to_container(config, resolve=True))
    simulator = None
    if simulator_config_path is not None:
        simulator = SimulatorConfig(**raw_config.pop(SIMULATOR_CONFIG_KEY))
        print(f"Simulator config: {simulator}")
    check_outdated_config(raw_config)
    if simulator is not None:
        # After the CLI overrides, so the simulator's node and step counts win over the recipe.
        apply_simulator_overrides(raw_config, simulator)
    master_config = MasterConfig(**raw_config)
    # Keep the processor, dataset and vLLM MediaIO on one frame-count contract.
    # Without this, vLLM's VideoMediaIO silently falls back to 32 frames even
    # when a video recipe requests (and caches) 64 frames.
    materialize_vllm_video_config(
        cast(dict[str, Any], master_config.policy),
        cast(dict[str, Any], master_config.data),
    )
    print("Applied CLI overrides")

    if is_ppo_run(master_config):
        legacy_async_block = "ppo.async_ppo"
        legacy_async = master_config.ppo.async_ppo
    else:
        legacy_async_block = "grpo.async_grpo"
        legacy_async = master_config.grpo.async_grpo
    if legacy_async is not None:
        raise ValueError(
            f"SC requires `{legacy_async_block}: null`; use `async_rl.*` instead. "
            "See docs/guides/single-controller.md#migrating-a-legacy-async-config."
        )

    dp_cfg = master_config.data_plane
    if not dp_cfg["enabled"]:
        raise ValueError(
            "The SingleController entrypoints require data_plane.enabled=true. "
            "Use examples/run_grpo.py for the legacy / sync paths."
        )
    return master_config, simulator


def _configure_generation(
    config: MasterConfig,
    tokenizer: PreTrainedTokenizerBase,
    *,
    simulator: Optional[SimulatorConfig],
) -> None:
    """Normalize ``config.policy["generation"]`` in place for the engines setup builds.

    Args:
        config: The validated recipe.
        tokenizer: Tokenizer used by the policy.
        simulator: The simulator config, or None on a training run.
    """
    assert config.policy["generation"] is not None, (
        "A generation config is required for SC-driven async GRPO"
    )
    if simulator is None:
        has_refit_draft_weights = draft_refit_enabled(config.policy.get("draft"))
        megatron_cfg = config.policy.get("megatron_cfg") or {}
        trains_mtp = bool(megatron_cfg.get("mtp_num_layers"))
        # Only used by Automodel draft co-training; SC supports only the
        # Megatron train backend today, so this is always False here.
        draft_full_refit = draft_full_refit_enabled(cast(dict[str, Any], config.policy))
    else:
        # First: configure_generation_config's nvfp4 check rejects such a recipe without saying why.
        reject_online_quantization(config.policy["generation"])
        # No refit follows, so the engine expects no draft or MTP weights from one.
        has_refit_draft_weights = trains_mtp = draft_full_refit = False
    config.policy["generation"] = configure_generation_config(
        config.policy["generation"],
        tokenizer,
        has_refit_draft_weights=has_refit_draft_weights,
        trains_mtp=trains_mtp,
        draft_full_refit=draft_full_refit,
    )


def run_single_controller(
    config_path: str,
    overrides: list[str],
    *,
    simulator_config_path: Optional[str] = None,
) -> None:
    """Load an SC recipe, bring up the SingleControllerActor and run it to completion.

    Args:
        config_path: The SC recipe YAML.
        overrides: Hydra-style overrides from the command line.
        simulator_config_path: The simulator YAML of a simulator run, whose
            trainer is a stand-in; None for a training run.
    """
    # The training loop runs in SingleControllerActor, whose flush=True prints only
    # reach this driver: Ray forwards actor output here without flushing. When the
    # driver's stdout is redirected (ray.sub -> ray-driver.log), Python
    # block-buffers it, so step and coordination logs would arrive in bursts or
    # only at exit. Line-buffer the driver so they show up as they are printed.
    sys.stdout.reconfigure(line_buffering=True)
    register_omegaconf_resolvers()
    config, simulator = _load_configs(
        config_path, overrides, simulator_config_path=simulator_config_path
    )

    print("Final config:")
    pprint.pprint(config)

    config.logger.log_dir = get_next_experiment_dir(config.logger.log_dir)
    print(f"📊 Using log directory: {config.logger.log_dir}")
    if config.checkpointing["enabled"]:
        print(
            f"📊 Using checkpoint directory: {config.checkpointing['checkpoint_dir']}"
        )

    # Must precede init_ray() so the resolved NEMO_RL_OTEL_* env is snapshotted
    # into the Ray runtime_env and inherited by every worker -- including the
    # SingleControllerActor, which is where this path's spans are opened. No-op
    # unless telemetry is enabled.
    init_telemetry_driver(config, algorithm="ppo" if is_ppo_run(config) else "grpo")

    # Startup is inside the try so shutdown_telemetry() below still runs when
    # it raises, which is when buffered spans are most worth having.
    actor_args = None
    try:
        # One root span, so init_ray() and setup_single_controller() phases land
        # in the same trace. Closed before the actor is launched, so the run's
        # own job and step spans stay separate traces.
        with startup_span():
            # Must precede init_ray() — see maybe_configure_data_plane_env's docstring.
            maybe_configure_data_plane_env(config.data_plane)
            maybe_configure_engine_reaping_env(
                config.async_rl.generation_fleet_health.enabled
            )
            # Opens rl.setup.ray_init itself, so no span here.
            init_ray()

            with setup_span("tokenizer"):
                processor = None
                if config.policy.get("is_vlm"):
                    processor = get_tokenizer(
                        config.policy["tokenizer"], get_processor=True
                    )
                    tokenizer = processor.tokenizer
                else:
                    tokenizer = get_tokenizer(config.policy["tokenizer"])
                _configure_generation(config, tokenizer, simulator=simulator)

            # Its own phase rather than part of the tokenizer block: it resolves
            # and can launch gym resource servers, so it is one of the phases
            # most likely to be the slow one.
            if bool(config.env.get("should_use_nemo_gym")):
                with setup_span("nemo_gym_config"):
                    setup_nemo_gym_config(config, tokenizer)

            # No child spans here: parallel init runs in threads, which do not
            # carry the OTel context. The breakdown comes from the
            # rl.setup.duration metric.
            with setup_span("workers"):
                actor_args, setup_timing_metrics = setup_single_controller(
                    config, tokenizer, processor=processor, simulator=simulator
                )

        placement = controller_placement()
        print(
            f"🚀 Launching SingleControllerActor (placement={placement or 'default'})"
        )
        sc = SingleControllerActor.options(**placement).remote(
            master_config=config,
            actor_args=actor_args,
            setup_timing_metrics=setup_timing_metrics,
        )
        result = run_with_controller_liveness_watch(sc, config.async_rl.stall_watchdog)
        print(f"SC run complete: {result}")
    finally:
        # None when setup raised before building them.
        if actor_args is not None:
            # Drain env actors before generation to avoid in-flight requests
            # during shutdown.
            shutdown_environments(actor_args.env_handles)

            teacher_worker_groups = (
                getattr(actor_args, "teacher_worker_groups", None) or {}
            )
            for teacher_alias, teacher in teacher_worker_groups.items():
                try:
                    teacher.shutdown()
                except Exception as e:
                    print(f"Teacher {teacher_alias!r} shutdown failed: {e}")

            for resource_name, resource in (
                ("Generation", actor_args.gen_handle),
                ("Trainer", actor_args.trainer_handle),
                ("Value", actor_args.value_handle),
            ):
                if resource is None:
                    continue
                try:
                    resource.shutdown()
                except Exception as e:
                    print(f"{resource_name} shutdown failed: {e}")

        # Last, and before cluster teardown: the OTel SDK's own atexit hook is
        # registered ahead of Ray's and so would otherwise run after it. The
        # actor flushes its own spans -- this covers the driver's side. No-op
        # when telemetry is inactive.
        shutdown_telemetry()
