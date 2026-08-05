#!/usr/bin/env python3
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
"""Backfill a W&B run's config from a NeMo-RL run directory.

SingleController runs launched before the `log_hyperparams` fix have an empty
Config tab. This rebuilds the resolved config exactly the way the entrypoint
does -- `load_config(--config)` then the CLI overrides, both read back out of
the run's `provenance.txt` -- and pushes it onto the existing W&B run.

Run from a networked shell (wandb.ai must be reachable). Use --no-project: a
plain `uv run` here syncs the whole NeMo-RL dependency tree (8k-line uv.lock --
torch, Megatron, vLLM) before running anything, which looks like a hang. This
script only needs three small packages:

    uv run --no-project --with hydra-core --with omegaconf --with wandb \
        ./wandb_backfill_config.py \
        --run-dir workspace/results/ultra-swe-sc-tq-zhiyul/runs/20260730-1539 \
        --wandb nvidia/nemorl-dataplane-zhiyul/0e4e72g8

Add --dry-run to print the reconstructed config without touching the run.
"""

import argparse
import shlex
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)


def config_from_provenance(run_dir: Path) -> dict[str, Any]:
    """Rebuild the resolved MasterConfig dict from a run's provenance.txt."""
    provenance = run_dir / "provenance.txt"
    command = ""
    for line in provenance.read_text().splitlines():
        if line.startswith("command: "):
            command = line[len("command: ") :]
            break
    if not command:
        raise SystemExit(f"no 'command:' line in {provenance}")

    argv = shlex.split(command)
    try:
        config_at = argv.index("--config")
    except ValueError:
        raise SystemExit(f"no --config in the command recorded in {provenance}")

    config_path = argv[config_at + 1]
    # Everything after the config path that looks like a hydra override. Env
    # assignments live before `uv run`, so they are never in this slice.
    overrides = [a for a in argv[config_at + 2 :] if "=" in a and not a.startswith("-")]

    # Configs use ${mul:...}/${div:...}/${max:...}; the entrypoints register these
    # on import, so resolve=True below fails without them.
    register_omegaconf_resolvers()

    config = load_config(config_path)
    if overrides:
        config = parse_hydra_overrides(config, overrides)
    return OmegaConf.to_container(config, resolve=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, required=True, help="run dir holding provenance.txt"
    )
    parser.add_argument("--wandb", required=True, help="entity/project/run_id")
    parser.add_argument(
        "--dry-run", action="store_true", help="print the config, do not upload"
    )
    args = parser.parse_args()

    config = config_from_provenance(args.run_dir)
    print(f"reconstructed {len(config)} top-level keys: {sorted(config)}")

    if args.dry_run:
        print(OmegaConf.to_yaml(OmegaConf.create(config)))
        return

    import wandb

    run = wandb.Api().run(args.wandb)
    run.config.update(config)
    run.update()
    print(f"updated config on {run.url}")


if __name__ == "__main__":
    main()
