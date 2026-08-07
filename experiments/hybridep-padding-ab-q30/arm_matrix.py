#!/usr/bin/env python3

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Literal


DEEPEP_17CF = "17cfb817bccec3a9c247013360cc550c2bac441e"
DEEPEP_F725 = "f725d29699f5bda9ba789456bb9579af69844685"
OFFICIAL_NEMO_RL = "ba473d47520472938482dae9a7f36414d034a110"
OFFICIAL_BRIDGE = "573e088c9c6740082c39744e03dc5b009e730ed4"
OFFICIAL_MCORE = "6513e3e23d6b5eda6a1c934990b15e804237732b"
OFFICIAL_BRANCH = "sna/hybridep-always-pad-uneven-20260805"
LEGACY_NEMO_RL = "d833180b9847daedafedaed6d7d1da6a013f14d0"
LEGACY_BRIDGE = "a68c7c893ea2c342660de0eef8a45032de8e9c89"
LEGACY_MCORE = "f812f5b3d20aa144c1762431ae77782f059dd9f9"
LEGACY_BRANCH = "sna/hybridep-legacy-prepad-q30-20260807"
CW_CONTAINER = "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/containers/nemo-rl-nightly-20260805/nemo_rl_nightly_20260805_15171871.sqsh"
CW_CONTAINER_SHA256 = "6623720fedcc82b31ab1f09f385590a5cf07751c35e5f6bf740a8b79c691b680"
CW_PREFLIGHT_MANIFEST_SHA256 = (
    "ab6797d70d846ae8a9734947f1cac99e1b0184fa7f2ac6c0e2643e77700649da"
)
QWEN30_RECIPE = "examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n8g.yaml"


@dataclass(frozen=True)
class ExperimentArm:
    name: str
    dispatcher: Literal["alltoall", "flex"]
    hybridep_backend: bool
    pad_uneven_dispatch_inputs: bool
    legacy_prepadding: bool
    deepep_commit: str | None
    source_profile: Literal["official", "legacy"]
    nemo_rl_commit: str
    bridge_commit: str
    mcore_commit: str
    source_branch: str
    container: str = CW_CONTAINER
    container_sha256: str = CW_CONTAINER_SHA256
    preflight_manifest_sha256: str = CW_PREFLIGHT_MANIFEST_SHA256
    recipe: str = QWEN30_RECIPE
    nodes: int = 4
    gpus_per_node: int = 8
    max_steps: int = 20


ARMS = (
    ExperimentArm(
        name="official-alltoall",
        dispatcher="alltoall",
        hybridep_backend=False,
        pad_uneven_dispatch_inputs=False,
        legacy_prepadding=False,
        deepep_commit=None,
        source_profile="official",
        nemo_rl_commit=OFFICIAL_NEMO_RL,
        bridge_commit=OFFICIAL_BRIDGE,
        mcore_commit=OFFICIAL_MCORE,
        source_branch=OFFICIAL_BRANCH,
    ),
    ExperimentArm(
        name="official-pr5008-17cf",
        dispatcher="flex",
        hybridep_backend=True,
        pad_uneven_dispatch_inputs=True,
        legacy_prepadding=False,
        deepep_commit=DEEPEP_17CF,
        source_profile="official",
        nemo_rl_commit=OFFICIAL_NEMO_RL,
        bridge_commit=OFFICIAL_BRIDGE,
        mcore_commit=OFFICIAL_MCORE,
        source_branch=OFFICIAL_BRANCH,
    ),
    ExperimentArm(
        name="official-pr5008-f725",
        dispatcher="flex",
        hybridep_backend=True,
        pad_uneven_dispatch_inputs=True,
        legacy_prepadding=False,
        deepep_commit=DEEPEP_F725,
        source_profile="official",
        nemo_rl_commit=OFFICIAL_NEMO_RL,
        bridge_commit=OFFICIAL_BRIDGE,
        mcore_commit=OFFICIAL_MCORE,
        source_branch=OFFICIAL_BRANCH,
    ),
    ExperimentArm(
        name="legacy-prepad-17cf",
        dispatcher="flex",
        hybridep_backend=True,
        pad_uneven_dispatch_inputs=False,
        legacy_prepadding=True,
        deepep_commit=DEEPEP_17CF,
        source_profile="legacy",
        nemo_rl_commit=LEGACY_NEMO_RL,
        bridge_commit=LEGACY_BRIDGE,
        mcore_commit=LEGACY_MCORE,
        source_branch=LEGACY_BRANCH,
    ),
)


def get_arm(name: str) -> ExperimentArm:
    for arm in ARMS:
        if arm.name == name:
            return arm
    available = ", ".join(arm.name for arm in ARMS)
    raise ValueError(f"unknown experiment arm {name!r}; choose one of: {available}")


def _as_tsv(arm: ExperimentArm) -> str:
    fields = (
        arm.name,
        arm.dispatcher,
        str(int(arm.hybridep_backend)),
        str(int(arm.pad_uneven_dispatch_inputs)),
        str(int(arm.legacy_prepadding)),
        arm.deepep_commit or "none",
        arm.source_profile,
        arm.nemo_rl_commit,
        arm.bridge_commit,
        arm.mcore_commit,
        arm.source_branch,
        arm.container,
        arm.container_sha256,
        arm.preflight_manifest_sha256,
        arm.recipe,
        str(arm.nodes),
        str(arm.gpus_per_node),
        str(arm.max_steps),
    )
    return "\t".join(fields)


def main() -> None:
    parser = argparse.ArgumentParser()
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--arm")
    selection.add_argument("--list", action="store_true")
    parser.add_argument("--format", choices=("json", "tsv"), default="json")
    args = parser.parse_args()

    if args.list:
        if args.format != "json":
            parser.error("--list supports only --format json")
        print(json.dumps([asdict(arm) for arm in ARMS], indent=2, sort_keys=True))
        return

    try:
        arm = get_arm(args.arm)
    except ValueError as error:
        parser.error(str(error))
    if args.format == "json":
        print(json.dumps(asdict(arm), indent=2, sort_keys=True))
    else:
        print(_as_tsv(arm))


if __name__ == "__main__":
    main()
