"""Submit selected 20-step performance cells with a resumable ledger."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time


ARMS = (
    "bf16-bf16",
    "bf16-mxfp8",
    "mxfp8-false-mxfp8",
    "mxfp8-true-mxfp8",
)
MODELS = ("qwen30", "qwen235", "qwen35", "super")
MODES = ("sync", "async")


def write_ledger(path: Path, records: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(records, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--preflight-log", type=Path, required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--cases", nargs="+")
    args = parser.parse_args()

    if "32/32 configurations composed." not in args.preflight_log.read_text():
        raise SystemExit("Configuration preflight has not passed")

    root = Path(os.environ["REPO"])
    head = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != args.expected_sha:
        raise SystemExit("Repository differs from expected SHA")

    valid_cases = {
        f"{model}/{mode}/{arm}" for model in MODELS for mode in MODES for arm in ARMS
    }
    selected_cases = set(args.cases or valid_cases)
    if not selected_cases <= valid_cases:
        raise SystemExit(f"Unknown cases: {sorted(selected_cases - valid_cases)}")

    records = json.loads(args.ledger.read_text()) if args.ledger.exists() else []
    launcher = root / "experiments/precision_matrix_refresh_20260905/submit.sh"
    for model in MODELS:
        for mode in MODES:
            for arm in ARMS:
                case = f"{model}/{mode}/{arm}"
                if case not in selected_cases:
                    continue
                if any(row["case"] == case and row.get("job_id") for row in records):
                    continue

                env = dict(
                    os.environ,
                    MODEL=model,
                    MODE=mode,
                    ARM=arm,
                    MAX_STEPS="20",
                    EXPECTED_SOURCE_SHA=head,
                    PERFORMANCE_RECIPE="1",
                    TOPOLOGY="default",
                )
                row: dict[str, str] = {
                    "case": case,
                    "sha": head,
                    "run_group": env["RUN_GROUP"],
                }
                for action in ("test-only", "submit"):
                    env["ACTION"] = action
                    result = subprocess.run(
                        ["bash", str(launcher)],
                        env=env,
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                    )
                    row[action] = result.stdout
                    if result.returncode:
                        row["error"] = f"{action} exit {result.returncode}"
                        break
                    if action == "submit":
                        match = re.search(r"Submitted batch job (\d+)", result.stdout)
                        if not match:
                            raise SystemExit(
                                f"Ambiguous submission; inspect scheduler: {result.stdout}"
                            )
                        row["job_id"] = match.group(1)

                records.append(row)
                write_ledger(args.ledger, records)
                print(f"{case}: {row.get('job_id', row.get('error'))}", flush=True)
                if "error" in row:
                    raise SystemExit(row.get("submit", row["test-only"]))
                time.sleep(2)


if __name__ == "__main__":
    main()
