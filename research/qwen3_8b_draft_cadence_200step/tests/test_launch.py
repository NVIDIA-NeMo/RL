from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research.qwen3_8b_draft_cadence_200step.launch import (
    build_submission,
    initialize_run_identity,
    materialize_manifest,
    run_submission,
    validate_checkpoint_paths,
    validate_container,
    validate_runtime_source_root,
)
from research.qwen3_8b_draft_cadence_200step.matrix import build_arms


class LaunchContractTest(unittest.TestCase):
    def test_run_identity_initializes_after_ray_created_its_log_directory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result_dir = Path(directory) / "dflash-adaptive"
            (result_dir / "ray" / "123-logs").mkdir(parents=True)
            identity = initialize_run_identity(
                result_dir=result_dir,
                arm="dflash-adaptive",
                product_head="a" * 40,
                wandb_run_id="q8c200-dflash-adaptive-recovery1",
                slurm_job_id="123_6",
            )
            self.assertEqual(
                json.loads(identity.read_text()),
                {
                    "schema_version": 1,
                    "arm": "dflash-adaptive",
                    "product_head": "a" * 40,
                    "wandb_run_id": "q8c200-dflash-adaptive-recovery1",
                    "slurm_job_id": "123_6",
                },
            )
            with self.assertRaises(FileExistsError):
                initialize_run_identity(
                    result_dir=result_dir,
                    arm="dflash-adaptive",
                    product_head="a" * 40,
                    wandb_run_id="q8c200-dflash-adaptive-recovery1",
                    slurm_job_id="123_6",
                )

    def test_runtime_source_accepts_home_or_node_local_scratch_only(self) -> None:
        validate_runtime_source_root(Path("/home/sna/RL-cadence"))
        validate_runtime_source_root(Path("/raid/scratch/q8c200-123_4/source"))
        with self.assertRaisesRegex(ValueError, "source repository"):
            validate_runtime_source_root(Path("/lustre/results/source"))

    def test_manifest_is_exclusive_and_contains_all_literal_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = materialize_manifest(
                result_root=root,
                product_head="a" * 40,
                harness_head="b" * 40,
            )
            payload = json.loads(path.read_text())
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(len(payload["arms"]), 13)
            self.assertEqual(payload["product_head"], "a" * 40)
            self.assertEqual(payload["analysis_window"], [21, 200])
            self.assertEqual(payload["required_checkpoint_steps"], [50, 100, 150, 200])
            self.assertTrue((root / "scheduler-logs").is_dir())
            with self.assertRaises(FileExistsError):
                materialize_manifest(
                    result_root=root,
                    product_head="a" * 40,
                    harness_head="b" * 40,
                )

    def test_checkpoint_gate_rejects_missing_or_wrong_revision(self) -> None:
        arm = next(arm for arm in build_arms() if arm.name == "dflash-static")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / arm.target_revision
            drafter = root / arm.drafter_revision
            target.mkdir()
            drafter.mkdir()
            (target / "config.json").write_text("{}\n")
            (drafter / "config.json").write_text("{}\n")
            validate_checkpoint_paths(arm, target=target, drafter=drafter)
            with self.assertRaisesRegex(ValueError, "target revision"):
                validate_checkpoint_paths(arm, target=root / "wrong", drafter=drafter)

    def test_baseline_checkpoint_gate_still_requires_target_config(self) -> None:
        arm = next(arm for arm in build_arms() if arm.name == "baseline")
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / arm.target_revision
            target.mkdir()
            with self.assertRaisesRegex(ValueError, "config.json"):
                validate_checkpoint_paths(arm, target=target, drafter=None)

    def test_container_gate_requires_the_pinned_metadata_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "nemo_rl.sqsh"
            image.write_bytes(b"image")
            metadata = Path(str(image) + ".metadata.txt")
            metadata.write_text("sha256=wrong\n")
            with self.assertRaisesRegex(ValueError, "container digest"):
                validate_container(image)

    def test_container_gate_hashes_the_image_not_only_its_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "nemo_rl.sqsh"
            image.write_bytes(b"verified image")
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            Path(str(image) + ".metadata.txt").write_text(f"sha256={digest}\n")
            validate_container(image, expected_sha256=digest)
            image.write_bytes(b"tampered image")
            with self.assertRaisesRegex(ValueError, "image bytes"):
                validate_container(image, expected_sha256=digest)

    def test_run_script_invokes_the_resume_receipt_gate(self) -> None:
        script = (Path(__file__).parents[1] / "run_arm.sh").read_text()
        self.assertIn("launch resume-preflight", script)
        self.assertIn("launch adapt-native", script)
        self.assertIn("launch terminal-preflight", script)
        self.assertLess(
            script.index("uv run examples/run_grpo.py"),
            script.index("launch adapt-native"),
        )
        self.assertLess(
            script.index("launch adapt-native"),
            script.index("launch terminal-preflight"),
        )

    @patch("research.qwen3_8b_draft_cadence_200step.launch.subprocess.run")
    @patch(
        "research.qwen3_8b_draft_cadence_200step.launch.validate_source",
        side_effect=RuntimeError("incomplete product"),
    )
    def test_submission_preflights_product_before_sbatch(
        self, _validate_source, subprocess_run
    ) -> None:
        arm = next(arm for arm in build_arms() if arm.name == "dflash-adaptive")
        submission = build_submission(
            arm,
            remote_repo=Path("/home/sna/RL-cadence"),
            expected_product_head="a" * 40,
            result_root=Path("/lustre/results/q8-cadence"),
            account="nemotron_n3_post",
        )
        with self.assertRaisesRegex(RuntimeError, "incomplete product"):
            run_submission(submission)
        subprocess_run.assert_not_called()

    @patch(
        "research.qwen3_8b_draft_cadence_200step.launch.Path.exists", return_value=True
    )
    def test_sbatch_argv_uses_ray_sub_tmp_ray_and_no_shell_eval(self, _exists) -> None:
        arm = next(arm for arm in build_arms() if arm.name == "dspark-adaptive")
        submission = build_submission(
            arm,
            remote_repo=Path("/home/sna/RL-cadence"),
            expected_product_head="a" * 40,
            result_root=Path("/lustre/results/q8-cadence"),
            account="nemotron_n3_post",
        )
        argv = submission.argv
        joined = "\n".join(
            (
                *argv,
                *(f"{key}={value}" for key, value in submission.environment.items()),
            )
        )
        self.assertEqual(argv[0], "sbatch")
        self.assertIn("--test-only", argv)
        self.assertIn("--account=nemotron_n3_post", argv)
        self.assertIn("--nodes=1", argv)
        self.assertIn("--gres=gpu:4", argv)
        self.assertIn("--chdir=/home/sna/RL-cadence", argv)
        self.assertIn(
            "--output=/lustre/results/q8-cadence/scheduler-logs/q8c200-dspark-adaptive-%j.out",
            argv,
        )
        self.assertEqual(submission.environment["RAY_TMPDIR"], "/tmp")
        self.assertEqual(submission.environment["GPUS_PER_NODE"], "4")
        self.assertEqual(submission.environment["WANDB_PROJECT"], "sna-specdec")
        self.assertIn("research/qwen3_8b_draft_cadence_200step/run_arm.sh", joined)
        self.assertNotIn("eval", joined)
        self.assertTrue(argv[-1].endswith("/ray.sub"))


if __name__ == "__main__":
    unittest.main()
