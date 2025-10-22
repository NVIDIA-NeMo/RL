# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import numpy as np
import pytest

from tests.test_suites.utils.assertions import check
from tests.test_suites.utils.base_test import BaseNeMoRLTest
from tests.test_suites.utils.criterions import exact_match, threshold
from tests.test_suites.utils.mappers import AsFloat, BinaryAlign, Flatten
from tests.test_suites.utils.metric import Metric
from tests.test_suites.utils.reducers import (
    PercentileDiffReducer,
    ResidualsReducer,
    WassersteinReducer,
)
from tests.test_suites.utils.types.base_config import NeMoRLTestConfig


class TestSftLlama321b1n8gFsdp2tp1V3(BaseNeMoRLTest):
    config = NeMoRLTestConfig(
        test_name="sft-llama3.2-1b-1n8g-fsdp2tp1.v3",
        algorithm="sft",
        model_class="llm",
        test_suites=["nightly", "akiswani"],  # TODO(ahmadki): test_suites
        time_limit_minutes=15,
        overrides={
            "sft.max_num_steps": 250,
        },
    )

    @pytest.mark.stage("validation")
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

    @pytest.mark.stage("validation")
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

    @pytest.mark.stage("validation")
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

    @pytest.mark.stage("validation")
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
