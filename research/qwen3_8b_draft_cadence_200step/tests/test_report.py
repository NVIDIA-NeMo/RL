from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from research.qwen3_8b_draft_cadence_200step.report import (
    summarize_history,
    terminal_report_fields,
)


class ReportContractTest(unittest.TestCase):
    def test_report_uses_closed_steps_21_through_200_and_logged_throughput(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.jsonl"
            rows = []
            for step in range(1, 201):
                rows.append(
                    {
                        "_step": step,
                        "performance/tokens_per_sec_per_gpu": float(step),
                        "performance/generation_tokens_per_sec_per_gpu": float(
                            step * 2
                        ),
                        "timing/train/total_step_time": 1000.0 / step,
                        "timing/train/generation": 500.0 / step,
                        "timing/train/prepare_for_generation/total": step / 100.0,
                        "train/vllm/spec_num_accepted_tokens": 25.0,
                        "train/vllm/spec_num_draft_tokens": 50.0,
                        "train/draft_schedule/update_requested": float(step % 10 == 0),
                        "train/draft_schedule/refit_requested": float(step % 10 == 0),
                    }
                )
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            summary = summarize_history(path)
            self.assertEqual(summary["window"], {"start": 21, "end": 200, "count": 180})
            self.assertEqual(summary["e2e_tps_per_gpu"], 110.5)
            self.assertEqual(summary["generation_tps_per_gpu"], 221.0)
            self.assertEqual(summary["acceptance_rate"], 0.5)
            self.assertEqual(summary["requested_updates"], 18)
            self.assertEqual(summary["requested_draft_refits"], 18)
            self.assertEqual(summary["window_requested_updates"], 18)
            self.assertEqual(summary["window_requested_draft_refits"], 18)
            self.assertAlmostEqual(summary["mean_total_refit_time_s"], 1.105)
            self.assertAlmostEqual(summary["total_refit_path_time_s"], 198.9)

    def test_terminal_schedule_counters_are_flattened_for_csv(self) -> None:
        fields = terminal_report_fields(
            {
                "decision_count": 200,
                "successful_updates": 20,
                "successful_draft_refits": 20,
                "skipped_updates": 180,
                "forced_updates": 2,
                "decision_reason_counts": {
                    "always": 0,
                    "fixed_interval": 0,
                    "none": 180,
                    "adaptive_degradation": 1,
                    "adaptive_burst": 17,
                    "max_interval": 2,
                },
            }
        )
        self.assertEqual(fields["run_successful_updates"], 20)
        self.assertEqual(fields["reason_adaptive_burst"], 17)
        self.assertEqual(fields["reason_max_interval"], 2)

    def test_missing_logged_throughput_is_not_reconstructed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.jsonl"
            path.write_text(
                "".join(
                    json.dumps(
                        {
                            "_step": step,
                            "timing/train/total_step_time": 1.0,
                            "train/mean_total_tokens_per_sample": 100.0,
                        }
                    )
                    + "\n"
                    for step in range(21, 201)
                )
            )
            with self.assertRaisesRegex(ValueError, "logged throughput"):
                summarize_history(path)

    def test_no_spec_baseline_does_not_require_acceptance_or_schedule_metrics(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.jsonl"
            path.write_text(
                "".join(
                    json.dumps(
                        {
                            "_step": step,
                            "performance/tokens_per_sec_per_gpu": 100.0,
                            "performance/generation_tokens_per_sec_per_gpu": 200.0,
                            "timing/train/total_step_time": 2.0,
                            "timing/train/generation": 1.0,
                            "timing/train/prepare_for_generation/total": 0.1,
                        }
                    )
                    + "\n"
                    for step in range(21, 201)
                )
            )
            summary = summarize_history(path, speculative=False)
            self.assertIsNone(summary["acceptance_rate"])
            self.assertEqual(summary["requested_updates"], 0)
            self.assertEqual(summary["requested_draft_refits"], 0)


if __name__ == "__main__":
    unittest.main()
