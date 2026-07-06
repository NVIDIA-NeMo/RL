# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Standalone 6x6 Sudoku (2x3 boxes) puzzle generator.

reasoning_gym only ships mini_sudoku (4x4) and sudoku (9x9), both hardcoded. 6x6
sits between them: ~28.2M valid complete grids (vs 288 for 4x4), too many to
memorize in a small model, so the task requires actual constraint solving -- yet
outputs stay short (36 cells). Same generation recipe as reasoning_gym's
mini_sudoku (random solved grid via shuffled backtracking, then uniqueness-checked
blank carving), emitting the SAME entry shape ({question, answer, metadata:{puzzle,
solution, ...}}) so the reasoning_gym env's sudoku_blanks reward and the
sudoku_answer_tag prompt (both metadata-driven) work unchanged.
"""

import copy
from random import Random
from typing import Any, Optional

N = 6
BOX_H, BOX_W = 2, 3  # 6x6 boxes are 2 rows tall, 3 columns wide


def _is_valid(board: list[list[int]], row: int, col: int, num: int) -> bool:
    if num in board[row]:
        return False
    if num in [board[i][col] for i in range(N)]:
        return False
    box_row, box_col = BOX_H * (row // BOX_H), BOX_W * (col // BOX_W)
    for i in range(box_row, box_row + BOX_H):
        for j in range(box_col, box_col + BOX_W):
            if board[i][j] == num:
                return False
    return True


def _find_empty(board: list[list[int]]):
    for i in range(N):
        for j in range(N):
            if board[i][j] == 0:
                return (i, j)
    return None


def _solve(board: list[list[int]], rng: Optional[Random] = None) -> bool:
    """Backtracking solve. With rng, tries candidates in shuffled order so a solve
    from an empty board yields a random complete grid."""
    empty = _find_empty(board)
    if not empty:
        return True
    row, col = empty
    nums = list(range(1, N + 1))
    if rng is not None:
        rng.shuffle(nums)
    for num in nums:
        if _is_valid(board, row, col, num):
            board[row][col] = num
            if _solve(board, rng):
                return True
            board[row][col] = 0
    return False


def _count_solutions(board: list[list[int]], limit: int = 2) -> int:
    """Count solutions up to `limit` (operate on a copy; may leave board dirty on
    early return, like reasoning_gym)."""
    empty = _find_empty(board)
    if not empty:
        return 1
    row, col = empty
    count = 0
    for num in range(1, N + 1):
        if _is_valid(board, row, col, num):
            board[row][col] = num
            count += _count_solutions(board, limit)
            if count >= limit:
                return count
            board[row][col] = 0
    return count


def _generate_solved_board(rng: Random) -> list[list[int]]:
    board = [[0] * N for _ in range(N)]
    if not _solve(board, rng):
        raise RuntimeError("failed to generate a solved 6x6 board")
    return board


def _create_puzzle(
    solved_board: list[list[int]], num_empty: int, rng: Random
) -> list[list[int]]:
    puzzle = [row[:] for row in solved_board]
    cells = [(i, j) for i in range(N) for j in range(N)]
    rng.shuffle(cells)
    num_removed = 0
    for i, j in cells:
        saved = puzzle[i][j]
        puzzle[i][j] = 0
        if _count_solutions(copy.deepcopy(puzzle)) > 1:
            puzzle[i][j] = saved  # removal breaks uniqueness -> keep the clue
        else:
            num_removed += 1
            if num_removed == num_empty:
                break
    return puzzle


def _board_to_string(board: list[list[int]]) -> str:
    return "\n".join(" ".join(str(x) if x != 0 else "_" for x in row) for row in board)


def generate_sudoku6x6(
    size: int = 5000,
    seed: int = 42,
    min_empty: int = 14,
    max_empty: int = 20,
) -> list[dict[str, Any]]:
    """Generate `size` 6x6 Sudoku entries. Each entry mirrors reasoning_gym's shape:
    {question, answer, metadata:{puzzle (0=blank), solution, num_empty, ...}}."""
    assert 0 <= min_empty <= max_empty <= N * N
    entries: list[dict[str, Any]] = []
    for idx in range(size):
        rng = Random(seed + idx)
        solved = _generate_solved_board(rng)
        num_empty = rng.randint(min_empty, max_empty)
        puzzle = _create_puzzle(solved, num_empty, rng)
        num_empty = sum(1 for row in puzzle for x in row if x == 0)
        question = (
            "In 6x6 Sudoku:\n"
            "- Each row must contain each number from 1-6 exactly once\n"
            "- Each column must contain each number 1-6 exactly once\n"
            "- Each 2x3 box (2 rows by 3 columns) must contain each number 1-6 exactly once\n"
            f"Solve this 6x6 Sudoku puzzle:\n{_board_to_string(puzzle)}\n"
        )
        entries.append(
            {
                "question": question,
                "answer": _board_to_string(solved),
                "metadata": {
                    "source_dataset": "sudoku6x6",
                    "source_index": idx,
                    "puzzle": puzzle,
                    "solution": solved,
                    "num_empty": num_empty,
                },
            }
        )
    return entries
