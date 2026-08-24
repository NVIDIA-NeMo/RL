from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shlex
import subprocess
import tarfile
import tempfile
import unittest

from research.qwen3_8b_draft_cadence_200step.staged_launch import (
    StagedSource,
    build_staged_array_argv,
    render_staged_array_script,
)


class StagedLaunchContractTest(unittest.TestCase):
    def _archive(self, root: Path) -> StagedSource:
        source = root / "fixture-source"
        source.mkdir()
        ray_sub = source / "ray.sub"
        ray_sub.write_text(
            "#!/bin/bash\n"
            "set -euo pipefail\n"
            'printf \'pwd=%s\\n\' "$(pwd -P)" > "${STAGE_RECEIPT}"\n'
            'printf \'mounts=%s\\n\' "${MOUNTS}" >> "${STAGE_RECEIPT}"\n'
            'printf \'command=%s\\n\' "${COMMAND}" >> "${STAGE_RECEIPT}"\n'
            'printf \'submit_dir=%s\\n\' "${SLURM_SUBMIT_DIR:-}" >> "${STAGE_RECEIPT}"\n'
            'printf \'wandb_resume=%s\\n\' "${WANDB_RESUME:-}" >> "${STAGE_RECEIPT}"\n'
        )
        ray_sub.chmod(0o644)
        archive = root / "source.tar"
        with tarfile.open(archive, "w") as stream:
            stream.add(source, arcname=".")
        allowed_signers = root / "allowed-signers"
        allowed_signers.write_text("tester ssh-ed25519 AAAAC3NzaFixture\n")
        return StagedSource(
            archive=archive,
            sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
            allowed_signers=allowed_signers,
            allowed_signers_sha256=hashlib.sha256(
                allowed_signers.read_bytes()
            ).hexdigest(),
        )

    def test_rendered_script_extracts_and_mounts_job_local_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staged = self._archive(root)
            scratch = root / "scratch"
            result_root = root / "results it's-valid"
            result_root.mkdir()
            receipt = root / "receipt.txt"
            script = root / "run.sh"
            script.write_text(
                render_staged_array_script(
                    staged=staged,
                    result_root=result_root,
                    expected_product_head="a" * 40,
                    scratch_parent=scratch,
                    canary=True,
                )
            )
            environment = os.environ.copy()
            environment.update(
                {
                    "SLURM_JOB_ID": "123",
                    "SLURM_ARRAY_TASK_ID": "4",
                    "SLURM_RESTART_COUNT": "2",
                    "STAGE_RECEIPT": str(receipt),
                }
            )
            subprocess.run(("bash", str(script)), check=True, env=environment)
            lines = receipt.read_text().splitlines()
            expected_source = scratch / "q8c200-123_4-r2" / "source"
            self.assertEqual(lines[0], f"pwd={expected_source.resolve()}")
            self.assertEqual(
                lines[1], f"mounts=/lustre:/lustre,{expected_source}:{expected_source}"
            )
            self.assertIn(f"cd {expected_source}", lines[2])
            self.assertIn("launch preflight", lines[2])
            self.assertIn("launch compose-preflight", lines[2])
            self.assertIn("--arm dflash-fixed-10", lines[2])
            self.assertNotIn("run_arm.sh", lines[2])
            self.assertEqual(lines[3], f"submit_dir={result_root}")
            self.assertEqual(lines[4], "wandb_resume=allow")

    def test_rendered_script_rejects_tampered_allowed_signers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staged = self._archive(root)
            staged.allowed_signers.write_text("attacker ssh-ed25519 changed\n")
            script = root / "run.sh"
            script.write_text(
                render_staged_array_script(
                    staged=staged,
                    result_root=root / "results",
                    expected_product_head="a" * 40,
                    scratch_parent=root / "scratch",
                    canary=True,
                )
            )
            result = subprocess.run(
                ("bash", str(script)),
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "SLURM_JOB_ID": "123",
                    "SLURM_ARRAY_TASK_ID": "0",
                },
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("allowed signers digest mismatch", result.stderr)
            self.assertFalse((root / "scratch").exists())

    def test_rendered_full_script_runs_selected_arm_from_staged_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staged = self._archive(root)
            result_root = root / "results with;metacharacters"
            result_root.mkdir()
            receipt = root / "receipt.txt"
            script = root / "run.sh"
            script.write_text(
                render_staged_array_script(
                    staged=staged,
                    result_root=result_root,
                    expected_product_head="a" * 40,
                    scratch_parent=root / "scratch",
                    canary=False,
                )
            )
            subprocess.run(
                ("bash", str(script)),
                check=True,
                env={
                    **os.environ,
                    "SLURM_JOB_ID": "124",
                    "SLURM_ARRAY_TASK_ID": "6",
                    "STAGE_RECEIPT": str(receipt),
                },
            )
            command = receipt.read_text().splitlines()[2]
            self.assertIn("run_arm.sh", command)
            self.assertIn("--arm dflash-adaptive", command)
            command_argv = shlex.split(command.removeprefix("command="))
            result_index = command_argv.index("--result-dir")
            self.assertEqual(
                command_argv[result_index + 1],
                str(result_root / "dflash-adaptive"),
            )

    def test_rendered_script_rejects_tampered_archive_before_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staged = self._archive(root)
            staged.archive.write_bytes(staged.archive.read_bytes() + b"tampered")
            script = root / "run.sh"
            script.write_text(
                render_staged_array_script(
                    staged=staged,
                    result_root=root / "results",
                    expected_product_head="a" * 40,
                    scratch_parent=root / "scratch",
                    canary=True,
                )
            )
            result = subprocess.run(
                ("bash", str(script)),
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "SLURM_JOB_ID": "123",
                    "SLURM_ARRAY_TASK_ID": "0",
                },
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("source archive digest mismatch", result.stderr)
            self.assertFalse((root / "scratch").exists())

    def test_rendered_script_rejects_non_hex_product_head(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staged = self._archive(root)
            with self.assertRaisesRegex(ValueError, "40 lowercase hex"):
                render_staged_array_script(
                    staged=staged,
                    result_root=root / "results",
                    expected_product_head="a" * 39 + ";",
                    scratch_parent=root / "scratch",
                    canary=True,
                )

    def test_rendered_script_requires_readable_ray_sub_not_executable_mode(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rendered = render_staged_array_script(
                staged=self._archive(root),
                result_root=root / "results",
                expected_product_head="a" * 40,
                scratch_parent=root / "scratch",
                canary=True,
            )
            self.assertIn('[[ -f "${source_root}/ray.sub" ]]', rendered)
            self.assertNotIn('[[ -x "${source_root}/ray.sub" ]]', rendered)
            self.assertIn("STAGED_SOURCE_ERROR line=", rendered)

    def test_staged_array_argv_uses_lustre_chdir_and_segment_one(self) -> None:
        argv = build_staged_array_argv(
            script_path=Path("/lustre/results/staged-array.sh"),
            result_root=Path("/lustre/results/q8c200-recovery"),
            account="nemotron_n3_post",
            test_only=True,
        )
        self.assertEqual(argv[0], "sbatch")
        self.assertIn("--array=0-12", argv)
        self.assertIn("--segment=1", argv)
        self.assertIn("--chdir=/lustre/results/q8c200-recovery", argv)
        self.assertIn(
            "--error=/lustre/results/q8c200-recovery/scheduler-logs/q8c200-%A_%a.err",
            argv,
        )
        self.assertIn("--test-only", argv)
        self.assertLess(argv.index("--test-only"), len(argv) - 1)
        self.assertNotIn("/home/", "\n".join(argv))

    def test_staged_array_argv_supports_each_arm_test_only(self) -> None:
        for ordinal in range(13):
            with self.subTest(ordinal=ordinal):
                argv = build_staged_array_argv(
                    script_path=Path("/lustre/results/staged-array.sh"),
                    result_root=Path("/lustre/results/q8c200-recovery"),
                    account="nemotron_n3_post",
                    test_only=True,
                    array=str(ordinal),
                )
                self.assertIn(f"--array={ordinal}", argv)

        with self.assertRaisesRegex(ValueError, "single arm or the complete"):
            build_staged_array_argv(
                script_path=Path("/lustre/results/staged-array.sh"),
                result_root=Path("/lustre/results/q8c200-recovery"),
                account="nemotron_n3_post",
                test_only=True,
                array="13",
            )


if __name__ == "__main__":
    unittest.main()
