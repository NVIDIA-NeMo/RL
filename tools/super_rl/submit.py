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

"""Call sbatch with a clean environment; scheduler test-only unless --submit."""

import argparse
import json
import os
from pathlib import Path
import subprocess
from collections.abc import Mapping, Sequence

BASE_ENV_KEYS = (
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "SHELL",
    "LANG",
    "LC_ALL",
    "TZ",
    "TMPDIR",
)
SCHEDULER_PREFIXES = ("SLURM_", "SBATCH_", "SRUN_", "PMI", "MPI", "OMPI_")


def submission_environment(
    caller: Mapping[str, str], *, pass_env: Sequence[str], slurm_conf: Path | None
) -> dict[str, str]:
    """Allow only identity/locale and explicitly selected workload variables."""
    result = {key: caller[key] for key in BASE_ENV_KEYS if key in caller}
    for key in pass_env:
        if key.startswith(SCHEDULER_PREFIXES):
            raise ValueError(f"Cannot inherit scheduler/MPI state: {key}")
        result[key] = caller[key]
    if slurm_conf is not None:
        if not slurm_conf.is_file():
            raise FileNotFoundError(slurm_conf)
        result["SLURM_CONF"] = str(slurm_conf.resolve())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit a job instead of scheduler test-only",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="Name of a workload variable to pass (not its value)",
    )
    parser.add_argument(
        "--slurm-conf", type=Path, help="Explicit site client config, if needed"
    )
    parser.add_argument(
        "--bootstrap-grace-minutes",
        type=int,
        help=(
            "Request an OccupiedIdleGPUsJobReaper startup grace for the first "
            "Slurm component (e.g. 75); opt-in and subject to site policy"
        ),
    )
    parser.add_argument(
        "sbatch_args",
        nargs=argparse.REMAINDER,
        help="-- <sbatch options> <batch script> [script args]",
    )
    args = parser.parse_args()
    batch_args = args.sbatch_args
    if batch_args[:1] == ["--"]:
        batch_args = batch_args[1:]
    if not batch_args:
        parser.error("A batch script and its sbatch options are required after --")
    env = submission_environment(
        os.environ, pass_env=args.env, slurm_conf=args.slurm_conf
    )
    command = ["sbatch"]
    if not args.submit:
        command.append("--test-only")
    if args.bootstrap_grace_minutes is not None:
        if args.bootstrap_grace_minutes <= 0:
            parser.error("--bootstrap-grace-minutes must be positive")
        # Later heterogeneous components may have their own serving comments.
        first_component = (
            batch_args[: batch_args.index(":")] if ":" in batch_args else batch_args
        )
        if any(
            arg == "--comment" or arg.startswith("--comment=")
            for arg in first_component
        ):
            parser.error(
                "Use either --bootstrap-grace-minutes or a first-component --comment"
            )
        comment = {
            "OccupiedIdleGPUsJobReaper": {
                "exemptIdleTimeMins": str(args.bootstrap_grace_minutes),
                "reason": "other",
                "description": "nemo-rl-run-bootstrap",
            }
        }
        command.append("--comment=" + json.dumps(comment, separators=(",", ":")))
    # Do not log environment values or duplicate submissions on timeout/error.
    subprocess.run([*command, *batch_args], env=env, check=True)


if __name__ == "__main__":
    main()
