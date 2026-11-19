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

"""Offline PPO critic pretraining on SingleController rollout banks.

Takes the same YAML as the PPO run the critic will warm-start (so the value
model, privileged-critic, masking and GAE settings match), plus a
``critic_pretrain`` block naming the banks a ``rollout_collection`` run wrote.
Only the critic is built, on every node of ``cluster``. See
nemo_rl/algorithms/sc_critic_pretrain.py.
"""

import argparse
import os
import pprint
import sys

from omegaconf import OmegaConf

from nemo_rl.algorithms.sc_critic_pretrain import critic_pretrain
from nemo_rl.algorithms.single_controller_utils import MasterConfig
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data_plane.factory import maybe_configure_data_plane_env
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from nemo_rl.utils.logger import get_next_experiment_dir

# Drop examples/ from sys.path so examples/nemo_gym/ (no __init__.py) doesn't
# shadow the real nemo_gym package as a namespace package.
current_dir = os.path.dirname(os.path.abspath(__file__))
while current_dir in sys.path:
    sys.path.remove(current_dir)


def main() -> None:
    """Main entry point."""
    register_omegaconf_resolvers()
    parser = argparse.ArgumentParser(description="SingleController critic pretraining")
    parser.add_argument("--config", type=str, required=True, help="PPO YAML config")
    args, overrides = parser.parse_known_args()

    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")
    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)
    config = MasterConfig(**OmegaConf.to_container(config, resolve=True))
    if not config.data_plane.get("enabled", False):
        raise ValueError(
            "critic pretraining runs on the data plane: data_plane.enabled=true"
        )

    print("Final config:")
    pprint.pprint(config)
    config.logger["log_dir"] = get_next_experiment_dir(config.logger["log_dir"])
    print(f"📊 Using log directory: {config.logger['log_dir']}")

    # Must precede init_ray(); see maybe_configure_data_plane_env.
    maybe_configure_data_plane_env(config.data_plane)
    init_ray()
    tokenizer = get_tokenizer(config.policy["tokenizer"])
    critic_pretrain(config, tokenizer)


if __name__ == "__main__":
    main()
