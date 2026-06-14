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

"""Async GRPO launcher driven by the SingleController actor.

Wires together :func:`setup_handle` (the four remote handles) and
:class:`SingleControllerActor` (which builds the six local components
internally via :func:`setup_single_controller_component`). Mirrors
``run_grpo.py`` for the synchronous path; everything before
``setup_handle`` is identical so the same YAML configs apply.

``data_plane.enabled=true`` is mandatory — SC is built on the
TransferQueue data plane.
"""

import argparse
import os
import pprint

import ray
from omegaconf import OmegaConf

from nemo_rl.algorithms.grpo import MasterConfig
from nemo_rl.algorithms.single_controller import (
    SingleControllerActor,
    SingleControllerConfig,
)
from nemo_rl.algorithms.single_controller_setup import setup_handle
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.utils import setup_response_data
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import get_next_experiment_dir


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run async GRPO training via SingleController"
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )
    args, overrides = parser.parse_known_args()
    return args, overrides


def _build_sc_config(master_config: MasterConfig) -> SingleControllerConfig:
    """Lift the SC-facing knobs out of the master config.

    Most fields fall back to dataclass defaults; the few that have a
    natural home in the existing config (max_train_steps,
    generations_per_prompt) are pulled from ``grpo.*``. SC-specific
    knobs (batch_selection_strategy, staleness, advantage_*) can be
    overridden via a ``single_controller:`` section on the master
    config — anything in there is forwarded as a kwarg to
    :class:`SingleControllerConfig`, with unknown keys parked under
    ``extra`` to avoid pydantic strictness.
    """
    grpo_config = master_config.grpo
    sc_overrides: dict = getattr(master_config, "single_controller", None) or {}

    base_kwargs: dict = {
        "max_train_steps": int(grpo_config["max_num_steps"]),
        "max_num_epochs": int(grpo_config.get("max_num_epochs") or 0) or None,
        "generations_per_prompt": int(grpo_config["num_generations_per_prompt"]),
    }
    base_kwargs.update(sc_overrides)
    known_fields = {f.name for f in SingleControllerConfig.__dataclass_fields__.values()}
    cfg_kwargs = {k: v for k, v in base_kwargs.items() if k in known_fields}
    extra = {k: v for k, v in base_kwargs.items() if k not in known_fields}
    if extra:
        cfg_kwargs["extra"] = extra
    return SingleControllerConfig(**cfg_kwargs)


def main() -> None:
    """Main entry point."""
    register_omegaconf_resolvers()
    args, overrides = parse_args()

    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__), "configs", "grpo_math_1B.yaml"
        )

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")

    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)

    config = OmegaConf.to_container(config, resolve=True)
    config = MasterConfig(**config)
    print("Applied CLI overrides")

    dp_cfg = config.data_plane or {}
    if not dp_cfg.get("enabled", False):
        raise ValueError(
            "run_grpo_single_controller requires data_plane.enabled=true. "
            "Use examples/run_grpo.py for the legacy / sync paths."
        )

    print("Final config:")
    pprint.pprint(config)

    config.logger["log_dir"] = get_next_experiment_dir(config.logger["log_dir"])
    print(f"📊 Using log directory: {config.logger['log_dir']}")
    if config.checkpointing["enabled"]:
        print(
            f"📊 Using checkpoint directory: {config.checkpointing['checkpoint_dir']}"
        )

    init_ray()

    tokenizer = get_tokenizer(config.policy["tokenizer"])
    assert config.policy["generation"] is not None, (
        "A generation config is required for SC-driven async GRPO"
    )
    has_refit_draft_weights = bool(config.policy["draft"]["enabled"])
    config.policy["generation"] = configure_generation_config(
        config.policy["generation"],
        tokenizer,
        has_refit_draft_weights=has_refit_draft_weights,
    )

    dataset, val_dataset, task_to_env, _val_task_to_env = setup_response_data(
        tokenizer, config.data, config.env
    )

    handles = setup_handle(
        config,
        tokenizer,
        dataset,
        val_dataset,
        env_handles=task_to_env,
    )

    sc_cfg = _build_sc_config(config)
    print("SingleController config:")
    pprint.pprint(sc_cfg)

    print("🚀 Launching SingleControllerActor")
    sc = SingleControllerActor.remote(
        cfg=sc_cfg,
        handles=handles,
        tokenizer=tokenizer,
    )
    result = ray.get(sc.run.remote())
    print(f"SC run complete: {result}")


if __name__ == "__main__":
    main()
