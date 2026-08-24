from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shlex

from research.qwen3_8b_draft_cadence_200step.matrix import (
    CONTAINER,
    TARGET_SNAPSHOT,
    build_arms,
)


@dataclass(frozen=True, slots=True)
class StagedSource:
    archive: Path
    sha256: str
    allowed_signers: Path
    allowed_signers_sha256: str

    def __post_init__(self) -> None:
        for name, digest in (
            ("source archive", self.sha256),
            ("allowed signers", self.allowed_signers_sha256),
        ):
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"{name} SHA256 must be 64 lowercase hex digits")


def _quoted(value: str | Path) -> str:
    return shlex.quote(str(value))


def render_staged_array_script(
    *,
    staged: StagedSource,
    result_root: Path,
    expected_product_head: str,
    scratch_parent: Path = Path("/raid/scratch"),
    canary: bool = False,
    run_generation: str = "recovery1",
) -> str:
    if len(expected_product_head) != 40 or any(
        character not in "0123456789abcdef" for character in expected_product_head
    ):
        raise ValueError("expected product head must be 40 lowercase hex characters")
    if not run_generation or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789-"
        for character in run_generation
    ):
        raise ValueError(
            "run generation must use lowercase letters, digits, or hyphens"
        )
    arms = build_arms()
    cases = "\n".join(
        f"  {ordinal}) arm={arm.name} ;;" for ordinal, arm in enumerate(arms)
    )
    source_archive = _quoted(staged.archive)
    allowed_signers = _quoted(staged.allowed_signers)
    result_root_literal = _quoted(result_root)
    scratch_parent_literal = _quoted(scratch_parent)
    target_cache = _quoted(str(Path(TARGET_SNAPSHOT).parents[3]))
    container = _quoted(CONTAINER)
    if canary:
        quoted_result_dir_assignment = ""
        command = (
            'export COMMAND="cd ${quoted_source} && python3 -m '
            "research.qwen3_8b_draft_cadence_200step.launch preflight "
            "--arm ${arm} --source-root ${quoted_source} "
            f'--expected-product-head {expected_product_head}"'
        )
    else:
        quoted_result_dir_assignment = (
            "quoted_result_dir=$(printf '%q' \"${result_dir}\")"
        )
        command = (
            'export COMMAND="cd ${quoted_source} && bash '
            "research/qwen3_8b_draft_cadence_200step/run_arm.sh "
            "--arm ${arm} --result-dir ${quoted_result_dir} "
            f'--expected-product-head {expected_product_head}"'
        )
    return f"""#!/usr/bin/env bash
set -euo pipefail

on_error() {{
  local exit_code=$?
  trap - ERR
  printf 'STAGED_SOURCE_ERROR line=%s command=%q exit=%s\n' \\
    "${{BASH_LINENO[0]}}" "${{BASH_COMMAND}}" "${{exit_code}}" >&2
  exit "${{exit_code}}"
}}
trap on_error ERR

: "${{SLURM_JOB_ID:?SLURM_JOB_ID is required}}"
: "${{SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}}"
source_archive={source_archive}
expected_archive_sha256={staged.sha256}
actual_archive_sha256=$(sha256sum "${{source_archive}}")
actual_archive_sha256=${{actual_archive_sha256%% *}}
if [[ "${{actual_archive_sha256}}" != "${{expected_archive_sha256}}" ]]; then
  printf 'source archive digest mismatch: %s != %s\n' \
    "${{actual_archive_sha256}}" "${{expected_archive_sha256}}" >&2
  exit 65
fi

allowed_signers={allowed_signers}
expected_allowed_signers_sha256={staged.allowed_signers_sha256}
actual_allowed_signers_sha256=$(sha256sum "${{allowed_signers}}")
actual_allowed_signers_sha256=${{actual_allowed_signers_sha256%% *}}
if [[ "${{actual_allowed_signers_sha256}}" != "${{expected_allowed_signers_sha256}}" ]]; then
  printf 'allowed signers digest mismatch: %s != %s\n' \
    "${{actual_allowed_signers_sha256}}" "${{expected_allowed_signers_sha256}}" >&2
  exit 67
fi

restart_count=${{SLURM_RESTART_COUNT:-0}}
if ! [[ "${{restart_count}}" =~ ^[0-9]+$ ]]; then
  printf 'invalid Slurm restart count: %s\n' "${{restart_count}}" >&2
  exit 68
fi
scratch_root={scratch_parent_literal}/q8c200-${{SLURM_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}-r${{restart_count}}
source_root="${{scratch_root}}/source"
if [[ -e "${{scratch_root}}" ]]; then
  printf 'job-local scratch path already exists: %s\n' "${{scratch_root}}" >&2
  exit 66
fi
mkdir -p "${{source_root}}"
tar -xf "${{source_archive}}" -C "${{source_root}}"
[[ -f "${{source_root}}/ray.sub" ]]

case "${{SLURM_ARRAY_TASK_ID}}" in
{cases}
  *) printf 'invalid array task: %s\n' "${{SLURM_ARRAY_TASK_ID}}" >&2; exit 64 ;;
esac

result_root={result_root_literal}
result_dir="${{result_root}}/${{arm}}"
quoted_source=$(printf '%q' "${{source_root}}")
{quoted_result_dir_assignment}
export SLURM_SUBMIT_DIR="${{result_root}}"
export CONTAINER={container}
export MOUNTS="/lustre:/lustre,${{source_root}}:${{source_root}}"
export HF_HOME={target_cache}
export HF_DATASETS_CACHE="${{HF_HOME}}/cache"
export RAY_TMPDIR=/tmp
export TMPDIR=/tmp
export GPUS_PER_NODE=4
export WANDB_PROJECT=sna-specdec
export WANDB_RUN_ID="q8c200-${{arm}}-{expected_product_head[:8]}-{run_generation}"
export WANDB_RESUME=allow
export NRL_FORCE_REBUILD_VENVS=true
export BASE_LOG_DIR="${{result_dir}}/ray"
export GIT_CONFIG_COUNT=2
export GIT_CONFIG_KEY_0=gpg.format
export GIT_CONFIG_VALUE_0=ssh
export GIT_CONFIG_KEY_1=gpg.ssh.allowedSignersFile
export GIT_CONFIG_VALUE_1="${{allowed_signers}}"
{command}
cd "${{source_root}}"
exec bash "${{source_root}}/ray.sub"
"""


def build_staged_array_argv(
    *,
    script_path: Path,
    result_root: Path,
    account: str,
    test_only: bool,
    array: str = "0-12",
    job_suffix: str = "recovery1",
) -> tuple[str, ...]:
    if not str(script_path).startswith("/lustre/"):
        raise ValueError("staged array script must live under /lustre")
    if not str(result_root).startswith("/lustre/"):
        raise ValueError("result root must live under /lustre")
    if array not in {*(str(ordinal) for ordinal in range(13)), "0-12"}:
        raise ValueError("array must be a single arm or the complete 0-12 matrix")
    argv = [
        "sbatch",
        f"--array={array}",
        "--nodes=1",
        "--exclusive",
        f"--account={account}",
        f"--job-name={account}.q8c200-{job_suffix}",
        "--partition=batch",
        "--time=04:00:00" if array == "0-12" else "--time=00:20:00",
        "--gres=gpu:4",
        "--segment=1",
        f"--chdir={result_root}",
        f"--output={result_root}/scheduler-logs/q8c200-%A_%a.out",
        f"--error={result_root}/scheduler-logs/q8c200-%A_%a.err",
    ]
    if test_only:
        argv.append("--test-only")
    argv.append(str(script_path))
    return tuple(argv)
