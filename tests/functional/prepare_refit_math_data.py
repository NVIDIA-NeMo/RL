"""Write deterministic arithmetic data for a refit smoke without a dataset download."""

import json
import sys
from pathlib import Path


def write_data(path: Path) -> None:
    rows = []
    for index in range(128):
        a = 11 + index % 79
        b = 7 + (index * 13) % 83
        if index % 3 == 0:
            problem, answer = f"Compute {a} + {b}.", a + b
        elif index % 3 == 1:
            problem, answer = f"Compute {a} * {b}.", a * b
        else:
            problem, answer = (
                f"Solve for x: {b} * x + {a} = {b * (index + 1) + a}.",
                index + 1,
            )
        rows.append(
            {
                "input": problem + " Give the final answer in \\boxed{}.",
                "output": str(answer),
            }
        )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


if __name__ == "__main__":
    write_data(Path(sys.argv[1]))
