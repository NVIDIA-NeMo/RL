from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
RENDERER = EXPERIMENT_DIR / "report" / "render_summary_charts.py"


class SummaryChartTest(unittest.TestCase):
    def test_renderer_preserves_strict_ab_improvement_signs(self) -> None:
        spec = importlib.util.spec_from_file_location("render_summary_charts", RENDERER)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        rows = {
            (row["family"], row["model"], row["metric"]): row["improvement_pct"]
            for row in module.ROWS
        }
        self.assertAlmostEqual(rows[("Step time", "Qwen3-30B-A3B", "E2E")], 3.55)
        self.assertAlmostEqual(
            rows[("Step time", "Qwen3-235B-A22B", "Generation")], -0.83
        )
        self.assertAlmostEqual(rows[("Step time", "Nemotron3 Super", "Policy")], 44.00)
        self.assertAlmostEqual(rows[("Throughput", "Nemotron3 Super", "LogProb")], 8.90)

    def test_renderer_writes_png_and_pdf_for_both_metric_families(self) -> None:
        spec = importlib.util.spec_from_file_location("render_summary_charts", RENDERER)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            module.render_charts(output_dir)
            expected = {
                "step-time-improvement.png",
                "step-time-improvement.pdf",
                "throughput-improvement.png",
                "throughput-improvement.pdf",
            }
            self.assertEqual({path.name for path in output_dir.iterdir()}, expected)
            for path in output_dir.iterdir():
                self.assertGreater(path.stat().st_size, 1_000)


if __name__ == "__main__":
    unittest.main()
