import numpy as np
import pytest
from utils.assertions import check
from utils.base_test import BaseFunctionalTest
from utils.criterions import exact_match, threshold
from utils.mappers import AsFloat, BinaryAlign, Flatten
from utils.metric import Metric
from utils.reducers import PercentileDiffReducer, ResidualsReducer, WassersteinReducer


class TestSFT(BaseFunctionalTest):
    data_folder = "sft"

    @pytest.mark.functional
    @pytest.mark.parametrize(
        "metric_key",
        [
            "ray/node.0.gpu.0.mem_gb",
            "ray/node.0.gpu.1.mem_gb",
        ],
    )
    def test_memory_metrics_distribution(
        self, golden_data, experiment_data, metric_key
    ):
        ref, exp = self.get_metric_arrays(golden_data, experiment_data, metric_key)

        metric = Metric(
            preprocess_ref=[AsFloat()],
            preprocess_exp=[AsFloat()],
            reducer=ResidualsReducer(),
            id=f"memory_{metric_key.replace('/', '_')}",
        )

        criterion = threshold("max_abs_residual", np.mean(ref) * 0.05)
        check(metric, ref, exp, criterion)

    @pytest.mark.functional
    @pytest.mark.parametrize("metric_key", ["train/loss", "validation/val_loss"])
    def test_gpu_utilization_distribution(
        self, golden_data, experiment_data, metric_key
    ):
        ref, exp = self.get_metric_arrays(golden_data, experiment_data, metric_key)

        metric = Metric(
            preprocess_ref=[Flatten(), AsFloat()],
            preprocess_exp=[Flatten(), AsFloat()],
            joint_mappers=[BinaryAlign()],
            reducer=WassersteinReducer(),
            id=f"gpu_util_{metric_key.replace('/', '_')}",
        )

        criterion = threshold("distance", 15.0)
        check(metric, ref, exp, criterion)

    @pytest.mark.functional
    @pytest.mark.parametrize("metric_key", ["train/lr", "train/num_valid_samples"])
    def test_exact_match_metrics(self, golden_data, experiment_data, metric_key):
        ref, exp = self.get_metric_arrays(golden_data, experiment_data, metric_key)

        metric = Metric(
            preprocess_ref=[AsFloat()],
            preprocess_exp=[AsFloat()],
            reducer=ResidualsReducer(),
            id=f"exact_{metric_key.replace('/', '_')}",
        )

        criterion = exact_match()
        check(metric, ref, exp, criterion)

    @pytest.mark.functional
    @pytest.mark.parametrize("percentile", [50, 75, 90, 95])
    def test_timing_percentiles(self, golden_data, experiment_data, percentile):
        # exp data is based on SP, which should be faster than no SP (ref) run on all percentiles.
        # so we expect the "check" to fail
        ref, exp = self.get_metric_arrays(
            golden_data, experiment_data, "timing/train/total_step_time"
        )

        metric = Metric(
            preprocess_ref=[AsFloat()],
            preprocess_exp=[AsFloat()],
            reducer=PercentileDiffReducer(percentile),
            id=f"timing_p{percentile}",
        )

        criterion = threshold("diff", 1.0)
        with pytest.raises(AssertionError):
            check(metric, ref, exp, criterion)
