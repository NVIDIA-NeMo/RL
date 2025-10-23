import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pytest


class BaseFunctionalTest:
    """Base class for functional deep learning model validation tests."""

    # Override this in subclasses to specify the data folder
    data_folder: str = None

    @pytest.fixture(scope="class")
    def golden_data(self):
        """Load golden reference data from JSON file."""
        if self.data_folder is None:
            raise ValueError("data_folder must be set in subclass")

        data_path = (
            Path(__file__).parent.parent / "data" / self.data_folder / "golden.json"
        )
        with open(data_path, "r") as f:
            return json.load(f)

    @pytest.fixture(scope="class")
    def experiment_data(self):
        """Load experiment data from JSON file."""
        if self.data_folder is None:
            raise ValueError("data_folder must be set in subclass")

        data_path = (
            Path(__file__).parent.parent / "data" / self.data_folder / "experiment.json"
        )
        with open(data_path, "r") as f:
            return json.load(f)

    def get_metric_arrays(
        self,
        golden_data: Dict[str, Any],
        experiment_data: Dict[str, Any],
        metric_key: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract arrays for a specific metric from both datasets."""
        if metric_key not in golden_data:
            raise KeyError(f"Metric '{metric_key}' not found in golden data")
        if metric_key not in experiment_data:
            raise KeyError(f"Metric '{metric_key}' not found in experiment data")

        ref_dict = golden_data[metric_key]
        exp_dict = experiment_data[metric_key]

        # Convert to arrays of values (sorted by timestep)
        ref_values = [ref_dict[str(k)] for k in sorted(map(int, ref_dict.keys()))]
        exp_values = [exp_dict[str(k)] for k in sorted(map(int, exp_dict.keys()))]

        return np.array(ref_values), np.array(exp_values)

    def validate_data_completeness(
        self, golden_data: Dict[str, Any], experiment_data: Dict[str, Any]
    ) -> None:
        """Validate that both datasets have the same metrics available."""
        golden_keys = set(golden_data.keys())
        experiment_keys = set(experiment_data.keys())

        # Check that all golden metrics exist in experiment
        missing_in_exp = golden_keys - experiment_keys
        extra_in_exp = experiment_keys - golden_keys

        if missing_in_exp:
            raise AssertionError(f"Missing metrics in experiment: {missing_in_exp}")
        if extra_in_exp:
            raise AssertionError(f"Extra metrics in experiment: {extra_in_exp}")

    # def validate_metric_length_consistency(self, golden_data: Dict[str, Any], experiment_data: Dict[str, Any],
    #                                      metric_key: str, tolerance: float = 0.2) -> None:
    #     """Validate that metric data has consistent length between golden and experiment."""
    #     ref, exp = self.get_metric_arrays(golden_data, experiment_data, metric_key)

    #     # Both arrays should have reasonable length
    #     if len(ref) == 0:
    #         raise AssertionError(f"Golden data for {metric_key} is empty")
    #     if len(exp) == 0:
    #         raise AssertionError(f"Experiment data for {metric_key} is empty")

    #     # Arrays should be similar in length (allow some variance for timing differences)
    #     length_ratio = len(exp) / len(ref)
    #     min_ratio, max_ratio = 1 - tolerance, 1 + tolerance

    #     if not (min_ratio <= length_ratio <= max_ratio):
    #         raise AssertionError(
    #             f"Length mismatch for {metric_key}: ref={len(ref)}, exp={len(exp)}, "
    #             f"ratio={length_ratio:.3f} (expected {min_ratio:.1f}-{max_ratio:.1f})"
    #         )
