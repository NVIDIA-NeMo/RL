"""Submit supported 20-step arms and retain one resumable job ledger."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--preflight-log", type=Path, required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--cases", nargs="+", help="Only submit these model/mode/arm cases")
    args = parser.parse_args()
    performance = os.environ.get("PERFORMANCE_RECIPE") == "1"
    hybridep = os.environ.get("PERFORMANCE_HYBRIDEP") == "1"
    expected_count = 16 if hybridep else (32 if performance else 48)
    if f"{expected_count}/{expected_count} configurations composed." not in args.preflight_log.read_text():
        raise SystemExit("Configuration preflight has not passed")
    root = Path(os.environ["REPO"])
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if head != args.expected_sha:
        raise SystemExit("Repository differs from expected SHA")
    records = json.loads(args.ledger.read_text()) if args.ledger.exists() else []
    launcher = root / "experiments/precision_matrix_refresh_20260905/submit.sh"
    arms = ("bf16-bf16", "bf16-mxfp8", "mxfp8-false-mxfp8", "mxfp8-true-mxfp8")
    if not performance:
        arms += ("mxfp8-false-bf16", "mxfp8-true-bf16")
    models = ("qwen30", "qwen235", "qwen35", "super") if performance else ("qwen30", "qwen35", "lightning", "qwen235")
    if hybridep:
        models = ("qwen30", "super")
    valid_cases = {f"{model}/{mode}/{arm}" for model in models for mode in ("sync", "async") for arm in arms}
    if args.cases and not set(args.cases) <= valid_cases:
        raise SystemExit("Unknown matrix case requested")
    for arm in arms:
        for model in models:
            for mode in ("sync", "async"):
                case = f"{model}/{mode}/{arm}"
                if args.cases and case not in args.cases:
                    continue
                if any(row["case"] == case and row.get("job_id") for row in records):
                    continue
                env = dict(os.environ, MODEL=model, MODE=mode, ARM=arm,
                           MAX_STEPS="20", EXPECTED_SOURCE_SHA=head, TOPOLOGY="default")
                env.pop("CONFIG_OVERRIDE", None)
                row = {"case": case, "sha": head, "run_group": env["RUN_GROUP"]}
                for action in ("test-only", "submit"):
                    env["ACTION"] = action
                    result = subprocess.run(["bash", str(launcher)], env=env, text=True,
                                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                    row[action] = result.stdout
                    if result.returncode:
                        row["error"] = f"{action} exit {result.returncode}"
                        break
                    if action == "submit":
                        match = re.search(r"Submitted batch job (\d+)", result.stdout)
                        if not match:
                            raise SystemExit(f"Ambiguous submission; inspect scheduler before retry: {result.stdout}")
                        row["job_id"] = match.group(1)
                records.append(row)
                args.ledger.parent.mkdir(parents=True, exist_ok=True)
                temporary = args.ledger.with_suffix(".tmp")
                temporary.write_text(json.dumps(records, indent=2) + "\n")
                temporary.replace(args.ledger)
                print(f"{case}: {row.get('job_id', row.get('error'))}", flush=True)
                if "error" in row:
                    raise SystemExit(row["submit"] if "submit" in row else row["test-only"])
                time.sleep(2)


if __name__ == "__main__":
    main()
