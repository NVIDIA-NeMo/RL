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
"""Read-only manifest gate for all 34 games plus image-tool GRPO.

This validates data contracts, not server readiness or successful training.
It neither chooses mixture weights nor changes prompts, caps or image paths.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from tools.prepare_image_tools_grpo import validate_row

# Audited game identities from the existing visual-games-suite 34-game builder.
GYM_V_TRAIN = frozenset(
    """Algorithmic/AdditionTable-v0 Algorithmic/CoinSquareGame-v0
    Algorithmic/FaceRightWay-v0 Algorithmic/GridBFS-v0 Games/Crosswords-v0
    Games/FrozenLake-v0 Games/Minesweeper-v0 Games/PegJump-v0 Games/RushHour-v0
    Games/TowerOfHanoiMultiTurn-v0 Games/WordSearch-v0 Games/Wordle-v0
    Graphs/GridComponent-v0 Logic/BinarioNoAdjacencyRequirement-v0
    Logic/CampsitePuzzle-v0 Logic/SkyscraperPuzzle-v0 Puzzles/Jewel2-QA-v0
    Puzzles/KloBlocks-v0 Puzzles/Maze-QA-v0 Puzzles/TicTacToe-QA-v0
    Puzzles/WordSearch-QA-v0 Spatial/Empty-v0""".split()
)
GYM_V_VALIDATION = frozenset(
    """Algorithmic/LangtonAnt-QA-v0 Cognition/TransformResultPoly-v0
    Games/FifteenPuzzle-v0 Games/Sudoku-v0 Geometry/ConvexHull-v0
    Logic/StarBattle-QA-v0 Puzzles/Snake-QA-v0 Spatial/Unlock-v0""".split()
)
VISGYM_TRAIN = frozenset(
    f"{name}/easy"
    for name in """maze_2d maze_3d matchstick_equation matchstick_rotation
    sliding_block patch_reassembly mental_rotation_3d_cube jigsaw colorization
    mental_rotation_2d zoom_in_puzzle referring_dot_pointing""".split()
)


def read_rows(paths: list[Path]) -> list[dict]:
    """Read explicitly selected JSONL files without resolving image assets."""
    rows = []
    for path in paths:
        with path.open() as stream:
            for number, line in enumerate(stream, 1):
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{number}: expected an object")
                rows.append(row)
    return rows


def audit_mix(
    *, train: list[dict], validation: list[dict], max_output_tokens: int
) -> dict:
    """Require all game routes and explicit image GRPO rows; return counts only."""
    if type(max_output_tokens) is not int or max_output_tokens <= 0:
        raise ValueError("max_output_tokens must be a positive integer")
    counts = {"train": Counter(), "validation": Counter()}
    agents = Counter()
    task_ids: set[str] = set()
    for split, rows in (("train", train), ("validation", validation)):
        for number, row in enumerate(rows, 1):
            env = row.get("env_id")
            if not isinstance(env, str) or not env:
                raise ValueError(f"{split} row {number}: missing env_id")
            if split == "validation":
                if env not in GYM_V_VALIDATION:
                    raise ValueError(
                        f"Validation must contain only held-out Gym-V: {env}"
                    )
                agent = "gym_v_agent"
            elif env in GYM_V_TRAIN:
                agent = "gym_v_agent"
            elif env in VISGYM_TRAIN:
                agent = "visgym_agent"
            elif env.startswith("image_tools/") and env.removeprefix("image_tools/"):
                agent = "image_tools_simple_agent"
                validate_row(row, number)
                if (row.get("task_metadata") or {}).get("split") != "train":
                    raise ValueError(
                        "Image rows must come from the prepared training split"
                    )
            else:
                raise ValueError(f"Unknown or held-out training environment: {env}")
            if row.get("agent_ref") != {
                "name": agent,
                "type": "responses_api_agents",
            }:
                raise ValueError(f"{env}: explicit agent_ref must route to {agent}")
            if row.get("task_source") != agent:
                raise ValueError(f"{env}: task_source disagrees with agent_ref")
            params = row.get("responses_create_params")
            if not isinstance(params, dict):
                raise ValueError(f"{env}: missing responses_create_params")
            cap = params.get("max_output_tokens")
            if type(cap) is not int or not 0 < cap <= max_output_tokens:
                raise ValueError(f"{env}: invalid or excessive max_output_tokens")
            task_id = row.get("task_id")
            if not isinstance(task_id, str) or not task_id or task_id in task_ids:
                raise ValueError(f"{env}: missing or duplicate task_id")
            task_ids.add(task_id)
            counts[split][env] += 1
            if split == "train":
                agents[agent] += 1
    missing_train = (GYM_V_TRAIN | VISGYM_TRAIN) - counts["train"].keys()
    missing_validation = GYM_V_VALIDATION - counts["validation"].keys()
    if missing_train or missing_validation:
        raise ValueError(
            f"Missing games: train={sorted(missing_train)}, "
            f"validation={sorted(missing_validation)}"
        )
    if not agents["image_tools_simple_agent"]:
        raise ValueError("Training mixture contains no image-tool rows")
    total = sum(agents.values())
    return {
        "rows": {split: sum(values.values()) for split, values in counts.items()},
        "per_environment": {split: dict(values) for split, values in counts.items()},
        "train_agent_rows": dict(agents),
        "unweighted_row_fractions": {
            agent: count / total for agent, count in agents.items()
        },
        "max_output_tokens_ceiling": max_output_tokens,
        "runtime_verified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, action="append", required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--max-output-tokens", type=int, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            audit_mix(
                train=read_rows(args.train),
                validation=read_rows([args.validation]),
                max_output_tokens=args.max_output_tokens,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
