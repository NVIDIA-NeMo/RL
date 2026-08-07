#!/usr/bin/env python3

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Literal


DEEPEP_17CF = "17cfb817bccec3a9c247013360cc550c2bac441e"
DEEPEP_F725 = "f725d29699f5bda9ba789456bb9579af69844685"
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
    ),
    ExperimentArm(
        name="official-pr5008-17cf",
        dispatcher="flex",
        hybridep_backend=True,
        pad_uneven_dispatch_inputs=True,
        legacy_prepadding=False,
        deepep_commit=DEEPEP_17CF,
        source_profile="official",
    ),
    ExperimentArm(
        name="official-pr5008-f725",
        dispatcher="flex",
        hybridep_backend=True,
        pad_uneven_dispatch_inputs=True,
        legacy_prepadding=False,
        deepep_commit=DEEPEP_F725,
        source_profile="official",
    ),
    ExperimentArm(
        name="legacy-prepad-17cf",
        dispatcher="flex",
        hybridep_backend=True,
        pad_uneven_dispatch_inputs=False,
        legacy_prepadding=True,
        deepep_commit=DEEPEP_17CF,
        source_profile="legacy",
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
