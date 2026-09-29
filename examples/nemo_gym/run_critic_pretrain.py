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
"""Offline PPO critic pretraining on stored rollout shards (stage B).

Reuses the production PPO config (value model, adv_estimator, value_loss_fn)
but initializes ONLY the value model — no policy, no vLLM, no gym. Rollout
shards come from examples/nemo_gym/run_swe_rollout_collection.py; knobs live in
a ``critic_pretrain:`` config block (see
``nemo_rl.algorithms.critic_pretrain.resolve_critic_pretrain_config``).
Launch via scripts/swe/ppo/critic_pretrain.sh.
"""

import argparse
import pprint

from omegaconf import OmegaConf

from nemo_rl.algorithms.critic_pretrain import critic_pretrain
from nemo_rl.algorithms.ppo import MasterConfig
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Offline PPO critic pretraining on stored rollouts"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    args, overrides = parser.parse_known_args()
    return args, overrides


def main() -> None:
    """Main entry point."""
    register_omegaconf_resolvers()
    args, overrides = parse_args()

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")
    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)
    config = OmegaConf.to_container(config, resolve=True)
    config = MasterConfig(**config)

    print("Final config:")
    pprint.pprint(config)

    tokenizer = get_tokenizer(config.policy["tokenizer"])

    init_ray()
    critic_pretrain(config, tokenizer)


if __name__ == "__main__":
    main()
