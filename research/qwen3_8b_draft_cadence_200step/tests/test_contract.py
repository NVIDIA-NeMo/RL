from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from research.qwen3_8b_draft_cadence_200step.matrix import (
    ADAPTIVE_SCHEDULE,
    Arm,
    build_arms,
    render_hydra_overrides,
)
from research.qwen3_8b_draft_cadence_200step.receipts import (
    adapt_native_outputs,
    validate_adaptive_decisions,
    validate_arm_receipts,
    validate_resume_ready,
)


class MatrixContractTest(unittest.TestCase):
    def test_matrix_has_one_baseline_and_six_arms_per_drafter(self) -> None:
        arms = build_arms()
        self.assertEqual(len(arms), 13)
        self.assertEqual([arm.name for arm in arms].count("baseline"), 1)
        for drafter in ("dflash", "dspark"):
            self.assertEqual(
                [arm.cadence for arm in arms if arm.drafter == drafter],
                ["static", "always", "fixed-5", "fixed-10", "fixed-20", "adaptive"],
            )
        self.assertEqual(len({arm.wandb_name for arm in arms}), 13)
        self.assertEqual(
            next(arm.wandb_name for arm in arms if arm.name == "baseline"),
            "q8-cadence-200-baseline-nospec-seed42",
        )

    def test_all_arms_hold_the_science_contract_constant(self) -> None:
        arms = build_arms()
        held_constant = {
            (
                arm.max_steps,
                arm.seed,
                arm.output_sequence_length,
                arm.global_batch_size,
                arm.prompts_per_step,
                arm.generations_per_prompt,
                arm.nodes,
                arm.gpus_per_node,
                arm.tensor_parallel_size,
                arm.context_parallel_size,
                arm.dataset,
                arm.wandb_project,
            )
            for arm in arms
        }
        self.assertEqual(
            held_constant,
            {
                (
                    200,
                    42,
                    1024,
                    8,
                    2,
                    4,
                    1,
                    4,
                    2,
                    1,
                    "DAPOMath17K",
                    "sna-specdec",
                )
            },
        )

    def test_checkpoint_revisions_are_exact_and_method_specific(self) -> None:
        arms = build_arms()
        dflash = next(arm for arm in arms if arm.name == "dflash-static")
        dspark = next(arm for arm in arms if arm.name == "dspark-static")
        self.assertEqual(
            dflash.target_revision,
            "b968826d9c46dd6066d109eabc6255188de91218",
        )
        self.assertEqual(
            dflash.drafter_revision,
            "9b41424b7109f9c5413454f481b09a82b85333f4",
        )
        self.assertEqual(
            dspark.drafter_revision,
            "03326e5043815da1f81b109078b2889737c26017",
        )
        self.assertIn("models--z-lab--Qwen3-8B-DFlash-b16", dflash.drafter_snapshot)
        self.assertIn(
            "models--deepseek-ai--dspark_qwen3_8b_block7", dspark.drafter_snapshot
        )

    def test_schedule_overrides_encode_the_approved_treatments(self) -> None:
        arms = {arm.name: arm for arm in build_arms()}
        self.assertEqual(arms["dflash-static"].schedule["fixed_interval"], 201)
        self.assertEqual(arms["dflash-always"].schedule, {"mode": "always"})
        for interval in (5, 10, 20):
            schedule = arms[f"dspark-fixed-{interval}"].schedule
            self.assertEqual(
                schedule,
                {
                    "mode": "fixed",
                    "action": "sparse_update",
                    "fixed_interval": interval,
                },
            )
        self.assertEqual(arms["dflash-adaptive"].schedule, ADAPTIVE_SCHEDULE)
        self.assertEqual(arms["dspark-adaptive"].schedule, ADAPTIVE_SCHEDULE)

    def test_hydra_overrides_pin_runtime_evidence_and_cuda_graphs(self) -> None:
        arm = next(arm for arm in build_arms() if arm.name == "dflash-fixed-10")
        overrides = render_hydra_overrides(
            arm, result_dir="/lustre/result/dflash-fixed-10"
        )
        joined = "\n".join(overrides)
        for required in (
            "grpo.max_num_steps=200",
            "policy.generation.max_new_tokens=1024",
            "policy.train_global_batch_size=8",
            "checkpointing.save_period=50",
            "checkpointing.keep_top_k=null",
            "checkpointing.metric_name=null",
            "cadence_runtime.enabled=true",
            "cadence_runtime.required_checkpoint_steps=[50,100,150,200]",
            "policy.draft.optimizer.lr=5e-06",
            "policy.draft.optimizer.min_lr=5e-07",
            "policy.draft.optimizer.weight_decay=0.01",
            "policy.draft.update_schedule.fixed_interval=10",
            "++policy.generation.vllm_kwargs.compilation_config.cudagraph_mode=PIECEWISE",
            "logger.wandb.project=sna-specdec",
            "logger.wandb.entity=nvidia",
        ):
            self.assertIn(required, joined)
        self.assertIn(
            "++cadence_runtime.result_dir=/lustre/result/dflash-fixed-10", overrides
        )

    def test_all_fields_use_hydra_force_add_or_override(self) -> None:
        for arm in build_arms():
            with self.subTest(arm=arm.name):
                overrides = render_hydra_overrides(
                    arm, result_dir=f"/lustre/result/{arm.name}"
                )
                self.assertTrue(overrides)
                self.assertTrue(all(item.startswith("++") for item in overrides))

    def test_product_preflight_fails_closed_on_incomplete_source(self) -> None:
        arm = next(arm for arm in build_arms() if arm.name == "dflash-adaptive")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "nemo_rl/algorithms").mkdir(parents=True)
            (root / "nemo_rl/weight_sync").mkdir(parents=True)
            (root / "nemo_rl/models/policy").mkdir(parents=True)
            (root / "nemo_rl/algorithms/draft_cadence_runtime.py").write_text(
                "adaptive draft cadence requires selected-rollout acceptance provenance\n"
            )
            (root / "nemo_rl/algorithms/grpo_sync.py").write_text(
                "def helper_only(): pass\n"
            )
            (root / "nemo_rl/weight_sync/interfaces.py").write_text("")
            (root / "nemo_rl/models/policy/tq_policy.py").write_text("")
            with self.assertRaisesRegex(RuntimeError, "production cadence integration"):
                arm.validate_product_source(root)


class ReceiptContractTest(unittest.TestCase):
    def test_adaptive_replay_rejects_an_update_before_min_observations(self) -> None:
        with self.assertRaisesRegex(ValueError, "adaptive decision mismatch"):
            validate_adaptive_decisions(
                [
                    {
                        "decision_id": 1,
                        "global_step": 1,
                        "update_requested": True,
                        "draft_refit_requested": True,
                        "reason": "adaptive_degradation",
                        "forced": False,
                        "observed_acceptance": 0.5,
                        "applied_draft_version_before_step": 0,
                    }
                ]
            )

    def _write_success_receipts(self, root: Path, arm: Arm) -> None:
        ledger = []
        update_steps = (
            set(range(20, 201, 20))
            if arm.cadence == "adaptive"
            else set(arm.deterministic_update_steps())
        )
        applied_version = 0
        for step in () if arm.cadence == "baseline" else range(1, 201):
            requested = step in update_steps
            selected_version = applied_version
            if requested:
                applied_version = step
            ledger.append(
                {
                    "decision_id": step,
                    "global_step": step,
                    "update_requested": requested,
                    "update_attempted": requested,
                    "update_successful": requested,
                    "draft_refit_requested": requested,
                    "draft_refit_attempted": requested,
                    "draft_refit_successful": requested,
                    "reason": (
                        "always"
                        if arm.cadence == "always"
                        else (
                            "max_interval"
                            if arm.cadence == "adaptive" and requested
                            else ("fixed_interval" if requested else "none")
                        )
                    ),
                    "forced": arm.cadence == "adaptive" and requested,
                    "target_refit_attempted": True,
                    "target_refit_successful": True,
                    "selected_rollout_draft_version": selected_version,
                    "applied_draft_version_before_step": selected_version,
                    "applied_draft_version_after_step": applied_version,
                    "accepted_tokens": 50,
                    "draft_tokens": 100,
                    "observed_acceptance": 0.5 if arm.cadence == "adaptive" else None,
                }
            )
        (root / "decision-ledger.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in ledger)
        )
        for step in (50, 100, 150, 200):
            checkpoint = root / "checkpoints" / f"step_{step}"
            checkpoint.mkdir(parents=True)
            prefix_rows = [] if arm.cadence == "baseline" else ledger[:step]
            prefix_raw = "".join(json.dumps(row) + "\n" for row in prefix_rows)
            ledger_path = checkpoint / "draft-decision-ledger.jsonl"
            ledger_path.write_text(prefix_raw)
            ledger_digest = hashlib.sha256(prefix_raw.encode()).hexdigest()
            tree_digest = hashlib.sha256()
            tree_digest.update(b"draft-decision-ledger.jsonl\0")
            tree_digest.update(hashlib.sha256(prefix_raw.encode()).digest())
            high_water = 0 if arm.cadence == "baseline" else step
            (checkpoint / "cadence-checkpoint-receipt.json").write_text(
                json.dumps(
                    {
                        "successful": True,
                        "checkpoint_step": step,
                        "current_step": step,
                        "checkpoint_path": str(checkpoint.resolve()),
                        "checkpoint_tree_sha256": tree_digest.hexdigest(),
                        "last_decision_id": high_water,
                        "ledger_high_water": high_water,
                        "decision_ledger": {
                            "relative_path": ledger_path.name,
                            "size_bytes": len(prefix_raw.encode()),
                            "sha256": ledger_digest,
                            "first_decision_id": 1 if prefix_rows else None,
                            "last_decision_id": high_water,
                            "entry_count": len(prefix_rows),
                        },
                    }
                )
            )
        (root / "terminal.json").write_text(
            json.dumps(
                {
                    "terminal": True,
                    "exit_code": 0,
                    "requested_policy_steps": 200,
                    "completed_policy_steps": 200,
                    "attempted_updates": len(update_steps),
                    "successful_updates": len(update_steps),
                    "attempted_draft_refits": len(update_steps),
                    "successful_draft_refits": len(update_steps),
                    "successful_target_refits": 200,
                    "decision_count": 0 if arm.cadence == "baseline" else 200,
                    "skipped_updates": 0
                    if arm.cadence == "baseline"
                    else 200 - len(update_steps),
                    "forced_updates": len(update_steps)
                    if arm.cadence == "adaptive"
                    else 0,
                    "decision_reason_counts": {
                        "always": 200 if arm.cadence == "always" else 0,
                        "fixed_interval": len(update_steps)
                        if arm.cadence not in {"baseline", "always", "adaptive"}
                        else 0,
                        "none": (
                            0
                            if arm.cadence in {"baseline", "always"}
                            else 200 - len(update_steps)
                        ),
                        "adaptive_degradation": 0,
                        "adaptive_burst": 0,
                        "max_interval": len(update_steps)
                        if arm.cadence == "adaptive"
                        else 0,
                    },
                }
            )
        )
        (root / "runtime-evidence.json").write_text(
            json.dumps(
                {
                    "target_revision": arm.target_revision,
                    "drafter_revision": arm.drafter_revision,
                    "initial_draft_refit": None
                    if arm.drafter == "none"
                    else {
                        "attempted": True,
                        "successful": True,
                        "serving_version": 0,
                    },
                    "cuda_graph_mode": "PIECEWISE",
                    "cuda_graph_capture_sizes": [
                        1,
                        2,
                        4,
                        6,
                        8,
                        10,
                        12,
                        16,
                        18,
                        20,
                        24,
                        28,
                        30,
                        32,
                        36,
                        40,
                        42,
                        48,
                        50,
                        56,
                        60,
                        64,
                    ],
                    "step_1_complete": True,
                    "step_2_complete": True,
                }
            )
        )

    def test_fixed_receipts_require_exact_counts_versions_and_checkpoints(self) -> None:
        arm = next(arm for arm in build_arms() if arm.name == "dspark-fixed-10")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_success_receipts(root, arm)
            receipt = validate_arm_receipts(root, arm)
            self.assertEqual(receipt["successful_updates"], 20)
            self.assertEqual(receipt["successful_target_refits"], 200)
            self.assertEqual(receipt["decision_count"], 200)
            self.assertEqual(receipt["decision_reason_counts"]["fixed_interval"], 20)

    def test_missing_periodic_checkpoint_fails_closed(self) -> None:
        arm = next(arm for arm in build_arms() if arm.name == "dflash-fixed-5")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_success_receipts(root, arm)
            checkpoint = root / "checkpoints" / "step_150"
            (checkpoint / "cadence-checkpoint-receipt.json").unlink()
            (checkpoint / "draft-decision-ledger.jsonl").unlink()
            checkpoint.rmdir()
            with self.assertRaisesRegex(ValueError, "step_150"):
                validate_arm_receipts(root, arm)

    def test_native_checkpoint_receipt_aliases_are_accepted(self) -> None:
        arm = next(arm for arm in build_arms() if arm.name == "dspark-fixed-20")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_success_receipts(root, arm)
            for step in (50, 100, 150, 200):
                path = (
                    root
                    / "checkpoints"
                    / f"step_{step}"
                    / "cadence-checkpoint-receipt.json"
                )
                receipt = json.loads(path.read_text())
                receipt["current_step"] = receipt.pop("checkpoint_step")
                receipt["ledger_high_water"] = receipt.pop("last_decision_id")
                path.write_text(json.dumps(receipt))
            validate_arm_receipts(root, arm)

    def test_target_sync_or_version_provenance_failure_is_rejected(self) -> None:
        arm = next(arm for arm in build_arms() if arm.name == "dflash-always")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_success_receipts(root, arm)
            rows = [
                json.loads(line)
                for line in (root / "decision-ledger.jsonl").read_text().splitlines()
            ]
            rows[20]["target_refit_successful"] = False
            rows[21]["selected_rollout_draft_version"] = 999
            (root / "decision-ledger.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
            with self.assertRaisesRegex(ValueError, "target refit"):
                validate_arm_receipts(root, arm)

    def test_baseline_requires_neutral_empty_schedule_receipt(self) -> None:
        arm = next(arm for arm in build_arms() if arm.name == "baseline")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_success_receipts(root, arm)
            receipt = validate_arm_receipts(root, arm)
            self.assertEqual(receipt["successful_updates"], 0)
            self.assertEqual((root / "decision-ledger.jsonl").read_text(), "")

    def test_cuda_and_step2_runtime_evidence_are_mandatory(self) -> None:
        arm = next(arm for arm in build_arms() if arm.name == "dspark-fixed-20")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_success_receipts(root, arm)
            evidence = json.loads((root / "runtime-evidence.json").read_text())
            evidence["step_2_complete"] = False
            (root / "runtime-evidence.json").write_text(json.dumps(evidence))
            with self.assertRaisesRegex(ValueError, "Step 1/Step 2"):
                validate_arm_receipts(root, arm)

    def test_static_requires_a_successful_initial_draft_refit(self) -> None:
        arm = next(arm for arm in build_arms() if arm.name == "dflash-static")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_success_receipts(root, arm)
            evidence = json.loads((root / "runtime-evidence.json").read_text())
            evidence["initial_draft_refit"]["successful"] = False
            (root / "runtime-evidence.json").write_text(json.dumps(evidence))
            with self.assertRaisesRegex(ValueError, "initial draft refit"):
                validate_arm_receipts(root, arm)

    def test_adaptive_observation_is_bound_to_selected_rollout_counts(self) -> None:
        arm = next(arm for arm in build_arms() if arm.name == "dspark-adaptive")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_success_receipts(root, arm)
            validate_arm_receipts(root, arm)
            rows = [
                json.loads(line)
                for line in (root / "decision-ledger.jsonl").read_text().splitlines()
            ]
            rows[39]["observed_acceptance"] = 0.25
            (root / "decision-ledger.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
            with self.assertRaisesRegex(ValueError, "observed acceptance"):
                validate_arm_receipts(root, arm)

    def test_terminal_decision_reason_counters_must_match_the_ledger(self) -> None:
        arm = next(arm for arm in build_arms() if arm.name == "dflash-fixed-5")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_success_receipts(root, arm)
            terminal = json.loads((root / "terminal.json").read_text())
            terminal["decision_reason_counts"]["fixed_interval"] = 39
            (root / "terminal.json").write_text(json.dumps(terminal))
            with self.assertRaisesRegex(ValueError, "reason counters"):
                validate_arm_receipts(root, arm)

    def test_resume_gate_binds_identity_checkpoint_and_ledger_high_water(self) -> None:
        arm = next(arm for arm in build_arms() if arm.name == "dflash-fixed-10")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_success_receipts(root, arm)
            (root / "run-identity.json").write_text(
                json.dumps(
                    {
                        "arm": arm.name,
                        "product_head": "a" * 40,
                        "wandb_run_id": "stable-run",
                    }
                )
            )
            latest = validate_resume_ready(root, arm, product_head="a" * 40)
            self.assertEqual(latest.name, "step_200")
            receipt_path = latest / "cadence-checkpoint-receipt.json"
            receipt = json.loads(receipt_path.read_text())
            receipt["last_decision_id"] = 199
            receipt_path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "high-water"):
                validate_resume_ready(root, arm, product_head="a" * 40)
            receipt["last_decision_id"] = 200
            receipt_path.write_text(json.dumps(receipt))
            (latest / "draft-decision-ledger.jsonl").write_text("")
            with self.assertRaisesRegex(ValueError, "digest"):
                validate_resume_ready(root, arm, product_head="a" * 40)

    def _replace_normalized_with_native_outputs(
        self, root: Path, arm: Arm, *, product_head: str
    ) -> None:
        terminal = json.loads((root / "terminal.json").read_text())
        rows = [
            json.loads(line)
            for line in (root / "decision-ledger.jsonl").read_text().splitlines()
        ]
        final_receipt = json.loads(
            (
                root / "checkpoints" / "step_200" / "cadence-checkpoint-receipt.json"
            ).read_text()
        )
        final_receipt["completed_policy_steps"] = 200
        final_receipt_path = (
            root / "checkpoints" / "step_200" / "cadence-checkpoint-receipt.json"
        )
        final_receipt_path.write_text(json.dumps(final_receipt))
        (root / "checkpoint-runtime.json").write_text(json.dumps(final_receipt))
        schedule = {
            key: value
            for key, value in terminal.items()
            if key
            not in {
                "terminal",
                "exit_code",
                "requested_policy_steps",
                "completed_policy_steps",
            }
        }
        schedule["current_step"] = 200
        schedule["decision_rows"] = rows
        (root / "schedule-runtime.json").write_text(json.dumps(schedule))
        (root / "run-identity.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "arm": arm.name,
                    "product_head": product_head,
                    "wandb_run_id": "stable-run",
                }
            )
        )
        if arm.drafter != "none":
            identity = root / "initial-identity.json"
            identity.write_text('{"version":0}\n')
            (root / "initial-draft-apply.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "successful": True,
                        "serving_version": 0,
                        "snapshot_path": str(identity.resolve()),
                        "sha256": hashlib.sha256(identity.read_bytes()).hexdigest(),
                        "draft_model_sha256": "1" * 64,
                        "draft_optimizer_sha256": "2" * 64,
                    }
                )
            )
        (root / "terminal.json").unlink()
        (root / "runtime-evidence.json").unlink()
        (root / "decision-ledger.jsonl").unlink()

    def test_native_outputs_are_adapted_without_losing_science_or_versions(
        self,
    ) -> None:
        arm = next(arm for arm in build_arms() if arm.name == "dspark-fixed-10")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_success_receipts(root, arm)
            self._replace_normalized_with_native_outputs(
                root, arm, product_head="a" * 40
            )

            adapt_native_outputs(root, arm, product_head="a" * 40)

            terminal = validate_arm_receipts(root, arm)
            self.assertEqual(terminal["completed_policy_steps"], 200)
            rows = [
                json.loads(line)
                for line in (root / "decision-ledger.jsonl").read_text().splitlines()
            ]
            self.assertEqual(rows[9]["accepted_tokens"], 50)
            self.assertEqual(rows[9]["selected_rollout_draft_version"], 0)
            runtime = json.loads((root / "runtime-evidence.json").read_text())
            self.assertEqual(runtime["product_head"], "a" * 40)
            self.assertEqual(
                runtime["native_sources"]["schedule"], "schedule-runtime.json"
            )

    def test_native_adapter_fails_closed_before_writing_on_incomplete_science(
        self,
    ) -> None:
        arm = next(arm for arm in build_arms() if arm.name == "dflash-adaptive")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_success_receipts(root, arm)
            self._replace_normalized_with_native_outputs(
                root, arm, product_head="b" * 40
            )
            schedule_path = root / "schedule-runtime.json"
            schedule = json.loads(schedule_path.read_text())
            del schedule["decision_rows"][1]["accepted_tokens"]
            schedule_path.write_text(json.dumps(schedule))

            with self.assertRaisesRegex(ValueError, "acceptance|rollout counts"):
                adapt_native_outputs(root, arm, product_head="b" * 40)

            for name in (
                "decision-ledger.jsonl",
                "terminal.json",
                "runtime-evidence.json",
            ):
                self.assertFalse((root / name).exists())


if __name__ == "__main__":
    unittest.main()
