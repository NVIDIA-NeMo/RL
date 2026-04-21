#!/usr/bin/env python3
"""Stepwise GRPO setup diagnostic for local dataset runs."""

import argparse


def log(msg: str) -> None:
    print(f"[diag] {msg}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="examples/configs/grpo_math_1B.yaml",
        help="Base NeMo RL config path",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra-style overrides passed through to the config parser",
    )
    args = parser.parse_args()

    log("import OmegaConf")
    from omegaconf import OmegaConf

    log("import config utils")
    from nemo_rl.utils.config import (
        load_config,
        parse_hydra_overrides,
        register_omegaconf_resolvers,
    )

    register_omegaconf_resolvers()

    log(f"load config: {args.config}")
    config = load_config(args.config)
    if args.overrides:
        log(f"apply overrides: {len(args.overrides)}")
        config = parse_hydra_overrides(config, args.overrides)
    config = OmegaConf.to_container(config, resolve=True)
    log(f"config parsed; model={config['policy']['model_name']}")

    log("import grpo setup")
    from nemo_rl.algorithms.grpo import setup

    log("import tokenizer utils")
    from nemo_rl.algorithms.utils import get_tokenizer

    log("import data setup")
    from nemo_rl.data.utils import setup_response_data

    log("import ray init")
    from nemo_rl.distributed.virtual_cluster import init_ray

    log("import generation config")
    from nemo_rl.models.generation import configure_generation_config

    log("init ray")
    init_ray()
    log("ray ready")

    log("get tokenizer")
    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    log("tokenizer ready")

    config["policy"]["generation"] = configure_generation_config(
        config["policy"]["generation"], tokenizer
    )
    log("generation config ready")

    log("setup response data")
    dataset, val_dataset, _task_to_env, _val_task_to_env = setup_response_data(
        tokenizer, config["data"], config["env"]
    )
    train_len = len(dataset)
    val_len = len(val_dataset) if val_dataset is not None else None
    log(f"datasets ready; train={train_len} val={val_len}")

    log("setup trainer")
    setup(
        config,
        tokenizer,
        dataset,
        val_dataset,
    )
    log("setup complete")


if __name__ == "__main__":
    main()
