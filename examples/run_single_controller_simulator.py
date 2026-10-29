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

"""SingleController simulator: run one side of the RL loop with the other mocked.

Keeps the SingleController's own pumps, TransferQueue data plane, sampler and
refit lifecycle, and swaps one side of the loop for a stand-in that uses no GPU:

  rollout_only    real generation and environments, ``cpu`` train backend
  train_only      real trainer fed by a ``cpu`` rollout backend (not implemented)
  train_rollout   both real: the ordinary run_grpo_single_controller.py run

Only ``rollout_only`` is implemented. It serves a recipe's rollouts exactly as
training would -- same engines, same Gym, same sampler, same pause/resume around
every refit -- while a CpuTrainer stands in for the train step and the weight
version advances without moving any weights. Mirrors run_grpo_single_controller.py
for config loading, so SC recipes apply as-is; the stand-in knobs are flags
because they describe this allocation and this experiment, not the recipe.
"""

import argparse
import os
import pprint
import sys

from omegaconf import OmegaConf

from nemo_rl.algorithms.single_controller import SingleControllerActor
from nemo_rl.algorithms.single_controller_utils import (
    MasterConfig,
    is_ppo_run,
    setup_single_controller,
)
from nemo_rl.algorithms.single_controller_utils.simulator import (
    SIMULATOR_MODES,
    TRAIN_BACKENDS,
    SimulatorSettings,
    apply_simulator_overrides,
    reject_online_quantization,
)
from nemo_rl.algorithms.single_controller_utils.utils import (
    run_with_controller_liveness_watch,
)
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data_plane.factory import maybe_configure_data_plane_env
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.nemo_gym import setup_nemo_gym_config
from nemo_rl.environments.utils import shutdown_environments
from nemo_rl.models.generation import (
    configure_generation_config,
    maybe_configure_engine_reaping_env,
)
from nemo_rl.telemetry.instrumentation import setup_span, startup_span
from nemo_rl.telemetry.setup import init_telemetry_driver, shutdown_telemetry
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import get_next_experiment_dir
from nemo_rl.utils.outdated_config_checks import check_outdated_config

# Drop examples/ from sys.path so examples/nemo_gym/ (no __init__.py) doesn't
# shadow the real nemo_gym package as a namespace package.
current_dir = os.path.dirname(os.path.abspath(__file__))
while current_dir in sys.path:
    sys.path.remove(current_dir)

# SimulatorSettings owns every default; the flags below only override it, so a
# flag left out is None here and the dataclass fills it in.
_DEFAULTS = SimulatorSettings()


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse the simulator flags; anything else is a Hydra-style config override."""
    parser = argparse.ArgumentParser(
        description="Run one side of the SingleController RL loop with the other mocked"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to an SC YAML recipe"
    )
    parser.add_argument(
        "--mode",
        choices=SIMULATOR_MODES,
        default=None,
        help="Which side of the loop is real; only rollout_only is implemented. "
        f"Default: {_DEFAULTS.mode}.",
    )
    parser.add_argument(
        "--train-backend",
        choices=TRAIN_BACKENDS,
        default=None,
        help="The trainer; rollout_only requires the cpu mock. "
        f"Default: {_DEFAULTS.train_backend}.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Steps to run (overrides the recipe's max_num_steps, which sizes a "
        "training run). Default: the recipe's.",
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=None,
        help="Inference nodes of this allocation (overrides the recipe's "
        "generation node count; every node serves generation). Default: the "
        "recipe's.",
    )
    parser.add_argument(
        "--train-time-s",
        type=float,
        default=None,
        help="Seconds the cpu train backend holds each step where policy.train() "
        "would run, so generation is paced as beside a real trainer. Ends early "
        f"once no rollout is in flight; the skipped seconds are logged. Default: "
        f"{_DEFAULTS.train_time_s}.",
    )
    parser.add_argument(
        "--refit-time-s",
        type=float,
        default=None,
        help="Seconds the mock weight transfer holds inside the refit pause, "
        f"where a real transfer would keep generation frozen. Default: "
        f"{_DEFAULTS.refit_time_s}.",
    )
    args, overrides = parser.parse_known_args()
    return args, overrides


def settings_from_args(args: argparse.Namespace) -> SimulatorSettings:
    """Turn the parsed flags into validated simulator settings.

    Only flags that were given are passed on, so the dataclass stays the one
    place the defaults live.
    """
    given = {
        name: value
        for name, value in (
            ("mode", args.mode),
            ("train_backend", args.train_backend),
            ("train_time_s", args.train_time_s),
            ("refit_time_s", args.refit_time_s),
            ("num_nodes", args.num_nodes),
            ("num_steps", args.num_steps),
        )
        if value is not None
    }
    return SimulatorSettings(**given)


def main() -> None:
    """Main entry point."""
    # Same reason as run_grpo_single_controller.py: the loop runs in the actor and
    # its prints reach this driver, whose stdout must not block-buffer them.
    sys.stdout.reconfigure(line_buffering=True)
    register_omegaconf_resolvers()
    args, overrides = parse_args()
    settings = settings_from_args(args)

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    config = OmegaConf.to_container(config, resolve=True)
    check_outdated_config(config)
    # After the CLI overrides, so the allocation's flags win over the recipe.
    apply_simulator_overrides(config, settings)
    config = MasterConfig(**config)
    print(f"Simulator settings: {settings}")

    if is_ppo_run(config):
        legacy_async_block, legacy_async = "ppo.async_ppo", config.ppo.async_ppo
    else:
        legacy_async_block, legacy_async = "grpo.async_grpo", config.grpo.async_grpo
    if legacy_async is not None:
        raise ValueError(
            f"SC requires `{legacy_async_block}: null`; use `async_rl.*` instead. "
            "See docs/guides/single-controller.md#migrating-a-legacy-async-config."
        )

    dp_cfg = config.data_plane
    if not dp_cfg["enabled"]:
        raise ValueError(
            "run_single_controller_simulator requires data_plane.enabled=true: "
            "the simulator runs the SingleController loop, which is built on the "
            "TransferQueue data plane."
        )

    print("Final config:")
    pprint.pprint(config)

    config.logger.log_dir = get_next_experiment_dir(config.logger.log_dir)
    print(f"📊 Using log directory: {config.logger.log_dir}")

    # Must precede init_ray() so the resolved NEMO_RL_OTEL_* env is snapshotted
    # into the Ray runtime_env and inherited by every worker, including the
    # SingleControllerActor. No-op unless telemetry is enabled.
    init_telemetry_driver(config, algorithm="ppo" if is_ppo_run(config) else "grpo")

    actor_args = None
    try:
        with startup_span():
            # Must precede init_ray() — see maybe_configure_data_plane_env's docstring.
            maybe_configure_data_plane_env(config.data_plane)
            maybe_configure_engine_reaping_env(
                config.async_rl.generation_fleet_health.enabled
            )
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
                assert config.policy["generation"] is not None, (
                    "A generation config is required for a simulator run"
                )
                # Before configure_generation_config, whose own nvfp4 check would
                # reject such a recipe first with a message that does not say why.
                reject_online_quantization(config.policy["generation"])
                # Unlike a training run there is no trainer to refit dummy weights:
                # is_eval=True makes the engine load the real checkpoint directly.
                config.policy["generation"] = configure_generation_config(
                    config.policy["generation"], tokenizer, is_eval=True
                )

            if bool(config.env.get("should_use_nemo_gym")):
                with setup_span("nemo_gym_config"):
                    setup_nemo_gym_config(config, tokenizer)

            with setup_span("workers"):
                actor_args, setup_timing_metrics = setup_single_controller(
                    config, tokenizer, processor=processor, simulator=settings
                )

        print("🚀 Launching SingleControllerActor (simulator)")
        sc = SingleControllerActor.remote(
            master_config=config,
            actor_args=actor_args,
            setup_timing_metrics=setup_timing_metrics,
        )
        result = run_with_controller_liveness_watch(sc, config.async_rl.stall_watchdog)
        print(f"Simulator run complete: {result}")
    finally:
        # None when setup raised before building them.
        if actor_args is not None:
            # Drain env actors before generation to avoid in-flight requests
            # during shutdown.
            shutdown_environments(actor_args.env_handles)
            for resource_name, resource in (
                ("Generation", actor_args.gen_handle),
                ("Trainer", actor_args.trainer_handle),
            ):
                if resource is None:
                    continue
                try:
                    resource.shutdown()
                except Exception as e:
                    print(f"{resource_name} shutdown failed: {e}")

        # Last, and before cluster teardown; no-op when telemetry is inactive.
        shutdown_telemetry()


if __name__ == "__main__":
    main()
