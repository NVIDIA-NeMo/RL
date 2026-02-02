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

import shutil
import tempfile
from unittest.mock import MagicMock, call, patch

import pytest
import torch

from nemo_rl.utils.logger import (
    Logger,
    MLflowLogger,
    RayGpuMonitorLogger,
    SwanlabLogger,
    TensorboardLogger,
    WandbLogger,
    flatten_dict,
    print_message_log_samples,
)


class TestFlattenDict:
    """Test the flatten_dict utility function."""

    def test_empty_dict(self):
        """Test flattening an empty dictionary."""
        assert flatten_dict({}) == {}

    def test_flat_dict(self):
        """Test flattening a dictionary that is already flat."""
        d = {"a": 1, "b": 2, "c": 3}
        assert flatten_dict(d) == d

    def test_nested_dict(self):
        """Test flattening a nested dictionary."""
        d = {"a": 1, "b": {"c": 2, "d": 3}, "e": {"f": {"g": 4}}}
        expected = {"a": 1, "b.c": 2, "b.d": 3, "e.f.g": 4}
        assert flatten_dict(d) == expected

    def test_custom_separator(self):
        """Test flattening with a custom separator."""
        d = {"a": 1, "b": {"c": 2, "d": 3}}
        expected = {"a": 1, "b_c": 2, "b_d": 3}
        assert flatten_dict(d, sep="_") == expected


class TestTensorboardLogger:
    """Test the TensorboardLogger class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for logs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @patch("nemo_rl.utils.logger.SummaryWriter")
    def test_init(self, mock_summary_writer, temp_dir):
        """Test initialization of TensorboardLogger."""
        cfg = {"log_dir": temp_dir}
        logger = TensorboardLogger(cfg, log_dir=temp_dir)

        # The log_dir is passed to SummaryWriter but not stored as an attribute
        mock_summary_writer.assert_called_once_with(log_dir=temp_dir)

    @patch("nemo_rl.utils.logger.SummaryWriter")
    def test_log_metrics(self, mock_summary_writer, temp_dir):
        """Test logging metrics to TensorboardLogger."""
        cfg = {"log_dir": temp_dir}
        logger = TensorboardLogger(cfg, log_dir=temp_dir)

        metrics = {"loss": 0.5, "accuracy": 0.8}
        step = 10
        logger.log_metrics(metrics, step)

        # Check that add_scalar was called for each metric
        mock_writer = mock_summary_writer.return_value
        assert mock_writer.add_scalar.call_count == 2
        mock_writer.add_scalar.assert_any_call("loss", 0.5, 10)
        mock_writer.add_scalar.assert_any_call("accuracy", 0.8, 10)

    @patch("nemo_rl.utils.logger.SummaryWriter")
    def test_log_metrics_with_prefix(self, mock_summary_writer, temp_dir):
        """Test logging metrics with a prefix to TensorboardLogger."""
        cfg = {"log_dir": temp_dir}
        logger = TensorboardLogger(cfg, log_dir=temp_dir)

        metrics = {"loss": 0.5, "accuracy": 0.8}
        step = 10
        prefix = "train"
        logger.log_metrics(metrics, step, prefix)

        # Check that add_scalar was called for each metric with prefix
        mock_writer = mock_summary_writer.return_value
        assert mock_writer.add_scalar.call_count == 2
        mock_writer.add_scalar.assert_any_call("train/loss", 0.5, 10)
        mock_writer.add_scalar.assert_any_call("train/accuracy", 0.8, 10)

    @patch("nemo_rl.utils.logger.SummaryWriter")
    def test_log_hyperparams(self, mock_summary_writer, temp_dir):
        """Test logging hyperparameters to TensorboardLogger."""
        cfg = {"log_dir": temp_dir}
        logger = TensorboardLogger(cfg, log_dir=temp_dir)

        params = {"lr": 0.001, "batch_size": 32, "model": {"hidden_size": 128}}
        logger.log_hyperparams(params)

        # Check that add_hparams was called with flattened params
        mock_writer = mock_summary_writer.return_value
        mock_writer.add_hparams.assert_called_once()
        # First argument should be flattened dict
        called_params = mock_writer.add_hparams.call_args[0][0]
        assert called_params == {
            "lr": 0.001,
            "batch_size": 32,
            "model.hidden_size": 128,
        }

    @patch("nemo_rl.utils.logger.SummaryWriter")
    def test_coerce_to_scalar_python_primitives(self, mock_summary_writer, temp_dir):
        """Test that Python primitives pass through unchanged."""
        cfg = {"log_dir": temp_dir}
        logger = TensorboardLogger(cfg, log_dir=temp_dir)

        assert logger._coerce_to_scalar(42) == 42
        assert logger._coerce_to_scalar(3.14) == 3.14
        assert logger._coerce_to_scalar(True) is True
        assert logger._coerce_to_scalar("hello") == "hello"

    @patch("nemo_rl.utils.logger.SummaryWriter")
    def test_coerce_to_scalar_numpy_types(self, mock_summary_writer, temp_dir):
        """Test that numpy scalar types are coerced to Python primitives."""
        import numpy as np

        cfg = {"log_dir": temp_dir}
        logger = TensorboardLogger(cfg, log_dir=temp_dir)

        # numpy scalar types
        assert logger._coerce_to_scalar(np.float32(1.5)) == 1.5
        assert logger._coerce_to_scalar(np.float64(2.5)) == 2.5
        assert logger._coerce_to_scalar(np.int32(10)) == 10
        assert logger._coerce_to_scalar(np.int64(20)) == 20
        assert logger._coerce_to_scalar(np.bool_(True)) is True

        # 0-d numpy arrays
        assert logger._coerce_to_scalar(np.array(3.14)) == 3.14
        # 1-element numpy arrays
        assert logger._coerce_to_scalar(np.array([42])) == 42

        # Multi-element arrays should return None
        assert logger._coerce_to_scalar(np.array([1, 2, 3])) is None

    @patch("nemo_rl.utils.logger.SummaryWriter")
    def test_coerce_to_scalar_torch_tensors(self, mock_summary_writer, temp_dir):
        """Test that torch scalar tensors are coerced to Python primitives."""
        cfg = {"log_dir": temp_dir}
        logger = TensorboardLogger(cfg, log_dir=temp_dir)

        # 0-d tensors
        assert logger._coerce_to_scalar(torch.tensor(3.14)) == pytest.approx(3.14)
        assert logger._coerce_to_scalar(torch.tensor(42)) == 42

        # 1-element tensors
        assert logger._coerce_to_scalar(torch.tensor([99])) == 99

        # Multi-element tensors should return None
        assert logger._coerce_to_scalar(torch.tensor([1, 2, 3])) is None

    @patch("nemo_rl.utils.logger.SummaryWriter")
    def test_coerce_to_scalar_incompatible_types(self, mock_summary_writer, temp_dir):
        """Test that incompatible types return None."""
        cfg = {"log_dir": temp_dir}
        logger = TensorboardLogger(cfg, log_dir=temp_dir)

        assert logger._coerce_to_scalar({"key": "value"}) is None
        assert logger._coerce_to_scalar([1, 2, 3]) is None
        assert logger._coerce_to_scalar(None) is None
        assert logger._coerce_to_scalar(object()) is None

    @patch("nemo_rl.utils.logger.SummaryWriter")
    def test_log_metrics_coerces_numpy_and_torch(self, mock_summary_writer, temp_dir):
        """Test that log_metrics correctly logs numpy/torch scalars."""
        import numpy as np

        cfg = {"log_dir": temp_dir}
        logger = TensorboardLogger(cfg, log_dir=temp_dir)

        metrics = {
            "python_float": 1.0,
            "numpy_float32": np.float32(2.0),
            "numpy_float64": np.float64(3.0),
            "torch_scalar": torch.tensor(4.0),
            "numpy_0d": np.array(5.0),
            "torch_1elem": torch.tensor([6.0]),
            "skip_list": [1, 2, 3],
            "skip_dict": {"a": 1},
            "skip_multi_tensor": torch.tensor([1.0, 2.0]),
        }
        logger.log_metrics(metrics, step=1)

        mock_writer = mock_summary_writer.return_value
        # Should log 6 scalars, skip 3 incompatible
        assert mock_writer.add_scalar.call_count == 6

        # Verify each scalar was logged with correct value
        calls = {c[0][0]: c[0][1] for c in mock_writer.add_scalar.call_args_list}
        assert calls["python_float"] == 1.0
        assert calls["numpy_float32"] == pytest.approx(2.0)
        assert calls["numpy_float64"] == pytest.approx(3.0)
        assert calls["torch_scalar"] == pytest.approx(4.0)
        assert calls["numpy_0d"] == pytest.approx(5.0)
        assert calls["torch_1elem"] == pytest.approx(6.0)


class TestWandbLogger:
    """Test the WandbLogger class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for logs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @patch("nemo_rl.utils.logger.wandb")
    def test_init_custom_config(self, mock_wandb, temp_dir):
        """Test initialization of WandbLogger with custom config."""
        cfg = {
            "project": "custom-project",
            "name": "custom-run",
            "entity": "custom-entity",
            "group": "custom-group",
            "tags": ["tag1", "tag2"],
        }
        WandbLogger(cfg, log_dir=temp_dir)

        mock_wandb.init.assert_called_once_with(
            project="custom-project",
            name="custom-run",
            entity="custom-entity",
            group="custom-group",
            tags=["tag1", "tag2"],
            dir=temp_dir,
        )

    @patch("nemo_rl.utils.logger.wandb")
    def test_log_metrics(self, mock_wandb):
        """Test logging metrics to WandbLogger."""
        cfg = {}
        logger = WandbLogger(cfg)

        metrics = {"loss": 0.5, "accuracy": 0.8}
        step = 10
        logger.log_metrics(metrics, step)

        # Check that log was called with metrics and step
        mock_run = mock_wandb.init.return_value
        mock_run.log.assert_called_once_with(metrics, step=step)

    @patch("nemo_rl.utils.logger.wandb")
    def test_log_metrics_with_prefix(self, mock_wandb):
        """Test logging metrics with a prefix to WandbLogger."""
        cfg = {}
        logger = WandbLogger(cfg)

        metrics = {"loss": 0.5, "accuracy": 0.8}
        step = 10
        prefix = "train"
        logger.log_metrics(metrics, step, prefix)

        # Check that log was called with prefixed metrics and step
        mock_run = mock_wandb.init.return_value
        expected_metrics = {"train/loss": 0.5, "train/accuracy": 0.8}
        mock_run.log.assert_called_once_with(expected_metrics, step=step)

    @patch("nemo_rl.utils.logger.wandb")
    def test_log_metrics_with_step_metric(self, mock_wandb):
        """Test logging metrics with a step metric to WandbLogger."""
        cfg = {}
        logger = WandbLogger(cfg)

        # Define step metric
        step_metric = "iteration"

        # Include the step metric in the metrics
        metrics = {"loss": 0.5, "accuracy": 0.8, "iteration": 15}
        step = 10  # This should be ignored when step_metric is provided

        logger.log_metrics(metrics, step, step_metric=step_metric)

        # Check that log was called with metrics and commit=False
        # When using step_metric, step should be ignored and commit=False should be used
        mock_run = mock_wandb.init.return_value
        mock_run.log.assert_called_once_with(metrics, commit=False)

    @patch("nemo_rl.utils.logger.wandb")
    def test_log_metrics_with_prefix_and_step_metric(self, mock_wandb):
        """Test logging metrics with both prefix and step metric."""
        cfg = {}
        logger = WandbLogger(cfg)

        # Define prefix and step metric
        prefix = "train"
        step_metric = "train/iteration"

        # Include the step metric in the metrics
        metrics = {"loss": 0.5, "accuracy": 0.8, "iteration": 15}
        step = 10  # This should be ignored when step_metric is provided

        logger.log_metrics(metrics, step, prefix=prefix, step_metric=step_metric)

        # Check that log was called with prefixed metrics and commit=False
        # The step_metric key gets prefixed based on the current implementation
        mock_run = mock_wandb.init.return_value
        expected_metrics = {
            "train/loss": 0.5,
            "train/accuracy": 0.8,
            "train/iteration": 15,
        }
        mock_run.log.assert_called_once_with(expected_metrics, commit=False)

    @patch("nemo_rl.utils.logger.wandb")
    def test_define_metric(self, mock_wandb):
        """Test defining a metric with a custom step metric."""
        cfg = {}
        logger = WandbLogger(cfg)

        # Define metric pattern and step metric
        logger.define_metric("ray/*", step_metric="ray/ray_step")

        # Check that define_metric was called
        mock_run = mock_wandb.init.return_value
        mock_run.define_metric.assert_called_once_with(
            "ray/*", step_metric="ray/ray_step"
        )

    @patch("nemo_rl.utils.logger.wandb")
    def test_log_hyperparams(self, mock_wandb):
        """Test logging hyperparameters to WandbLogger."""
        cfg = {}
        logger = WandbLogger(cfg)

        params = {"lr": 0.001, "batch_size": 32, "model": {"hidden_size": 128}}
        logger.log_hyperparams(params)

        # Check that config.update was called with params
        mock_run = mock_wandb.init.return_value
        mock_run.config.update.assert_called_once_with(params)


class TestSwanlabLogger:
    """Test the SwanlabLogger class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for logs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @patch("nemo_rl.utils.logger.swanlab")
    def test_init_custom_config(self, mock_swanlab, temp_dir):
        """Test initialization of SwanlabLogger with custom config."""
        cfg = {
            "project": "custom-project",
            "name": "custom-run",
            "entity": "custom-entity",
            "group": "custom-group",
            "tags": ["tag1", "tag2"],
        }
        SwanlabLogger(cfg, log_dir=temp_dir)

        mock_swanlab.init.assert_called_once_with(
            project="custom-project",
            name="custom-run",
            entity="custom-entity",
            group="custom-group",
            tags=["tag1", "tag2"],
            logdir=temp_dir,
        )

    @patch("nemo_rl.utils.logger.swanlab")
    def test_log_metrics(self, mock_swanlab):
        """Test logging metrics to SwanlabLogger."""
        cfg = {}
        logger = SwanlabLogger(cfg)

        metrics = {"loss": 0.5, "accuracy": 0.8}
        step = 10
        logger.log_metrics(metrics, step)

        # Check that log was called with metrics and step
        mock_run = mock_swanlab.init.return_value
        mock_run.log.assert_called_once_with(metrics, step=step)

    @patch("nemo_rl.utils.logger.swanlab")
    def test_log_metrics_with_prefix(self, mock_swanlab):
        """Test logging metrics with a prefix to SwanlabLogger."""
        cfg = {}
        logger = SwanlabLogger(cfg)

        metrics = {"loss": 0.5, "accuracy": 0.8}
        step = 10
        prefix = "train"
        logger.log_metrics(metrics, step, prefix)

        # Check that log was called with prefixed metrics and step
        mock_run = mock_swanlab.init.return_value
        expected_metrics = {"train/loss": 0.5, "train/accuracy": 0.8}
        mock_run.log.assert_called_once_with(expected_metrics, step=step)

    @patch("nemo_rl.utils.logger.swanlab")
    def test_log_metrics_with_step_metric(self, mock_swanlab):
        """Test logging metrics with a step metric to SwanlabLogger."""
        cfg = {}
        logger = SwanlabLogger(cfg)

        # Define step metric
        step_metric = "iteration"

        # Include the step metric in the metrics
        metrics = {"loss": 0.5, "accuracy": 0.8, "iteration": 15}
        step = 10  # This should be ignored when step_metric is provided

        logger.log_metrics(metrics, step, step_metric=step_metric)

        # Check that log was called with metrics
        mock_run = mock_swanlab.init.return_value
        mock_run.log.assert_called_once_with(metrics, step=step)

    @patch("nemo_rl.utils.logger.swanlab")
    def test_log_metrics_with_prefix_and_step_metric(self, mock_swanlab):
        """Test logging metrics with both prefix and step metric."""
        cfg = {}
        logger = SwanlabLogger(cfg)

        # Define prefix and step metric
        prefix = "train"
        step_metric = "train/iteration"

        # Include the step metric in the metrics
        metrics = {"loss": 0.5, "accuracy": 0.8, "iteration": 15}
        step = 10  # This should be ignored when step_metric is provided

        logger.log_metrics(metrics, step, prefix=prefix, step_metric=step_metric)

        # Check that log was called with prefixed metrics
        mock_run = mock_swanlab.init.return_value
        expected_metrics = {
            "train/loss": 0.5,
            "train/accuracy": 0.8,
            "train/iteration": 15,
        }
        mock_run.log.assert_called_once_with(expected_metrics, step=step)

    @patch("nemo_rl.utils.logger.swanlab")
    def test_log_hyperparams(self, mock_swanlab):
        """Test logging hyperparameters to SwanlabLogger."""
        cfg = {}
        logger = SwanlabLogger(cfg)

        params = {"lr": 0.001, "batch_size": 32, "model": {"hidden_size": 128}}
        logger.log_hyperparams(params)

        # Check that config.update was called with params
        mock_run = mock_swanlab.init.return_value
        mock_run.config.update.assert_called_once_with(params)


class TestMLflowLogger:
    """Test the MLflowLogger class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for logs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @patch("nemo_rl.utils.logger.mlflow")
    def test_init_basic_config(self, mock_mlflow, temp_dir):
        """Test initialization of MLflowLogger with basic config."""
        cfg = {
            "experiment_name": "test-experiment",
            "run_name": "test-run",
            "tracking_uri": None,
        }
        MLflowLogger(cfg, log_dir=temp_dir)

        mock_mlflow.set_experiment.assert_called_once_with("test-experiment")
        mock_mlflow.start_run.assert_called_once_with(run_name="test-run")

    @patch("nemo_rl.utils.logger.mlflow")
    def test_init_full_config(self, mock_mlflow, temp_dir):
        """Test initialization of MLflowLogger with full config."""
        cfg = {
            "experiment_name": "test-experiment",
            "run_name": "test-run",
            "tracking_uri": "http://localhost:5000",
        }
        MLflowLogger(cfg, log_dir=temp_dir)

        mock_mlflow.set_tracking_uri.assert_called_once_with("http://localhost:5000")
        mock_mlflow.set_experiment.assert_called_once_with("test-experiment")
        mock_mlflow.start_run.assert_called_once_with(run_name="test-run")

    @patch("nemo_rl.utils.logger.mlflow")
    def test_log_metrics(self, mock_mlflow, temp_dir):
        """Test logging metrics to MLflowLogger."""
        cfg = {
            "experiment_name": "test-experiment",
            "run_name": "test-run",
            "tracking_uri": None,
        }
        logger = MLflowLogger(cfg, log_dir=temp_dir)

        metrics = {"loss": 0.5, "accuracy": 0.8}
        step = 10
        logger.log_metrics(metrics, step)

        # Check that log_metric was called for each metric
        assert mock_mlflow.log_metric.call_count == 2
        mock_mlflow.log_metric.assert_any_call("loss", 0.5, step=10)
        mock_mlflow.log_metric.assert_any_call("accuracy", 0.8, step=10)

    @patch("nemo_rl.utils.logger.mlflow")
    def test_log_metrics_with_prefix(self, mock_mlflow, temp_dir):
        """Test logging metrics with a prefix to MLflowLogger."""
        cfg = {
            "experiment_name": "test-experiment",
            "run_name": "test-run",
            "tracking_uri": None,
        }
        logger = MLflowLogger(cfg, log_dir=temp_dir)

        metrics = {"loss": 0.5, "accuracy": 0.8}
        step = 10
        prefix = "train"
        logger.log_metrics(metrics, step, prefix)

        # Check that log_metric was called for each metric with prefix
        assert mock_mlflow.log_metric.call_count == 2
        mock_mlflow.log_metric.assert_any_call("train/loss", 0.5, step=10)
        mock_mlflow.log_metric.assert_any_call("train/accuracy", 0.8, step=10)

    @patch("nemo_rl.utils.logger.mlflow")
    def test_log_hyperparams(self, mock_mlflow, temp_dir):
        """Test logging hyperparameters to MLflowLogger."""
        cfg = {
            "experiment_name": "test-experiment",
            "run_name": "test-run",
            "tracking_uri": None,
        }
        logger = MLflowLogger(cfg, log_dir=temp_dir)

        params = {"lr": 0.001, "batch_size": 32, "model": {"hidden_size": 128}}
        logger.log_hyperparams(params)

        # Check that log_params was called with flattened params
        mock_mlflow.log_params.assert_called_once_with(
            {
                "lr": 0.001,
                "batch_size": 32,
                "model.hidden_size": 128,
            }
        )

    @patch("nemo_rl.utils.logger.mlflow")
    @patch("nemo_rl.utils.logger.plt")
    @patch("nemo_rl.utils.logger.os")
    def test_log_plot(self, mock_os, mock_plt, mock_mlflow, temp_dir):
        """Test logging plots to MLflowLogger."""
        import tempfile

        cfg = {
            "experiment_name": "test-experiment",
            "run_name": "test-run",
            "tracking_uri": None,
        }
        logger = MLflowLogger(cfg, log_dir=temp_dir)

        # Mock the figure
        mock_figure = mock_plt.Figure.return_value

        # Mock tempfile.NamedTemporaryFile
        mock_temp_file = type("MockTempFile", (), {"name": "/tmp/test.png"})()
        with patch.object(tempfile, "NamedTemporaryFile") as mock_tempfile:
            mock_tempfile.return_value.__enter__.return_value = mock_temp_file
            mock_tempfile.return_value.__exit__.return_value = None

            logger.log_plot(mock_figure, step=10, name="test_plot")

            # Check that figure was saved and logged as artifact
            mock_figure.savefig.assert_called_once_with(
                "/tmp/test.png", format="png", bbox_inches="tight"
            )
            mock_mlflow.log_artifact.assert_called_once_with(
                "/tmp/test.png", "plots/test_plot"
            )

    @patch("nemo_rl.utils.logger.mlflow")
    def test_cleanup(self, mock_mlflow, temp_dir):
        """Test cleanup when logger is destroyed."""
        cfg = {
            "experiment_name": "test-experiment",
            "run_name": "test-run",
            "tracking_uri": None,
        }
        logger = MLflowLogger(cfg, log_dir=temp_dir)

        # Trigger cleanup
        logger.__del__()

        # Check that end_run was called
        mock_mlflow.end_run.assert_called_once()

    @patch("nemo_rl.utils.logger.mlflow")
    def test_init_with_none_log_dir(self, mock_mlflow):
        """Test initialization with None log_dir uses server default artifact location."""
        cfg = {
            "experiment_name": "test-experiment",
            "run_name": "test-run",
            "tracking_uri": "http://localhost:5000",
        }
        mock_mlflow.get_experiment_by_name.return_value = None

        MLflowLogger(cfg, log_dir=None)

        # Verify create_experiment was called without artifact_location
        mock_mlflow.create_experiment.assert_called_once_with(name="test-experiment")
        mock_mlflow.start_run.assert_called_once_with(run_name="test-run")

    @patch("nemo_rl.utils.logger.mlflow")
    def test_init_with_custom_log_dir(self, mock_mlflow):
        """Test initialization with custom log_dir sets artifact_location."""
        cfg = {
            "experiment_name": "test-experiment",
            "run_name": "test-run",
            "tracking_uri": "http://localhost:5000",
        }
        mock_mlflow.get_experiment_by_name.return_value = None

        MLflowLogger(cfg, log_dir="/custom/path")

        # Verify create_experiment was called with artifact_location
        mock_mlflow.create_experiment.assert_called_once_with(
            name="test-experiment", artifact_location="/custom/path"
        )
        mock_mlflow.start_run.assert_called_once_with(run_name="test-run")

    @patch("nemo_rl.utils.logger.mlflow")
    def test_init_with_artifact_location_in_config(self, mock_mlflow):
        """Test initialization with artifact_location in config takes precedence over log_dir."""
        cfg = {
            "experiment_name": "test-experiment",
            "run_name": "test-run",
            "tracking_uri": "http://localhost:5000",
            "artifact_location": "/config/artifact/path",
        }
        mock_mlflow.get_experiment_by_name.return_value = None

        MLflowLogger(cfg, log_dir="/fallback/path")

        # Verify create_experiment was called with artifact_location from config
        mock_mlflow.create_experiment.assert_called_once_with(
            name=cfg["experiment_name"], artifact_location=cfg["artifact_location"]
        )
        mock_mlflow.set_tracking_uri.assert_called_once_with(cfg["tracking_uri"])
        mock_mlflow.start_run.assert_called_once_with(run_name=cfg["run_name"])

    @patch("nemo_rl.utils.logger.mlflow")
    def test_init_with_artifact_location_none_in_config(self, mock_mlflow):
        """Test initialization with artifact_location=None in config uses server default."""
        cfg = {
            "experiment_name": "test-experiment",
            "run_name": "test-run",
            "tracking_uri": "http://localhost:5000",
            "artifact_location": None,
        }
        mock_mlflow.get_experiment_by_name.return_value = None

        MLflowLogger(cfg, log_dir="/fallback/path")

        # Verify create_experiment was called without artifact_location
        # (None is explicitly set, so we don't pass it to MLflow)
        mock_mlflow.create_experiment.assert_called_once_with(
            name=cfg["experiment_name"], artifact_location=cfg["artifact_location"]
        )
        mock_mlflow.set_tracking_uri.assert_called_once_with(cfg["tracking_uri"])
        mock_mlflow.start_run.assert_called_once_with(run_name=cfg["run_name"])

    @patch("nemo_rl.utils.logger.mlflow")
    def test_init_without_artifact_location_uses_log_dir(self, mock_mlflow):
        """Test initialization without artifact_location in config uses log_dir."""
        cfg = {
            "experiment_name": "test-experiment",
            "run_name": "test-run",
            "tracking_uri": "http://localhost:5000",
        }
        mock_mlflow.get_experiment_by_name.return_value = None

        log_dir = "/fallback/path"
        MLflowLogger(cfg, log_dir=log_dir)

        # Verify create_experiment was called with log_dir as artifact_location
        mock_mlflow.create_experiment.assert_called_once_with(
            name=cfg["experiment_name"], artifact_location=log_dir
        )
        mock_mlflow.set_tracking_uri.assert_called_once_with(cfg["tracking_uri"])
        mock_mlflow.start_run.assert_called_once_with(run_name=cfg["run_name"])


class TestRayGpuMonitorLogger:
    """Test the RayGpuMonitorLogger class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for logs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_parent_logger(self):
        """Create a mock parent logger."""

        class MockLogger:
            def __init__(self):
                self.logged_metrics = []
                self.logged_steps = []
                self.logged_prefixes = []
                self.logged_step_metrics = []

            def log_metrics(self, metrics, step, prefix="", step_metric=None):
                self.logged_metrics.append(metrics)
                self.logged_steps.append(step)
                self.logged_prefixes.append(prefix)
                self.logged_step_metrics.append(step_metric)

        return MockLogger()

    @patch("nemo_rl.utils.logger.ray")
    def test_init(self, mock_ray):
        """Test initialization of RayGpuMonitorLogger."""
        # Mock ray.is_initialized to return True
        mock_ray.is_initialized.return_value = True

        # Initialize the monitor with standard settings
        monitor = RayGpuMonitorLogger(
            collection_interval=10.0,
            flush_interval=60.0,
            metric_prefix="test",
            step_metric="test/step",
            parent_logger=None,
        )

        # Verify initialization parameters
        assert monitor.collection_interval == 10.0
        assert monitor.flush_interval == 60.0
        assert monitor.metric_prefix == "test"
        assert monitor.step_metric == "test/step"
        assert monitor.parent_logger is None
        assert monitor.metrics_buffer == []
        assert monitor.is_running is False
        assert monitor.collection_thread is None

    @patch("nemo_rl.utils.logger.ray")
    @patch("nemo_rl.utils.logger.threading.Thread")
    def test_start(self, mock_thread, mock_ray):
        """Test start method of RayGpuMonitorLogger."""
        # Mock ray.is_initialized to return True
        mock_ray.is_initialized.return_value = True

        # Initialize the monitor
        monitor = RayGpuMonitorLogger(
            collection_interval=10.0,
            flush_interval=60.0,
            metric_prefix="test",
            step_metric="test/step",
            parent_logger=None,
        )

        # Start the monitor
        monitor.start()

        # Verify thread was created and started
        mock_thread.assert_called_once()
        mock_thread.return_value.start.assert_called_once()

        # Verify monitor state
        assert monitor.is_running is True
        assert monitor.collection_thread is mock_thread.return_value

    @patch("nemo_rl.utils.logger.ray")
    def test_start_ray_not_initialized(self, mock_ray):
        """Test start method when Ray is not initialized."""
        # Mock ray.is_initialized to return False
        mock_ray.is_initialized.return_value = False

        # Initialize the monitor
        monitor = RayGpuMonitorLogger(
            collection_interval=10.0,
            flush_interval=60.0,
            metric_prefix="test",
            step_metric="test/step",
            parent_logger=None,
        )

        # Starting should raise a ValueError
        with pytest.raises(ValueError):
            monitor.start()

    @patch("nemo_rl.utils.logger.ray")
    @patch("nemo_rl.utils.logger.threading.Thread")
    def test_stop(self, mock_thread, mock_ray):
        """Test stop method of RayGpuMonitorLogger."""
        # Mock ray.is_initialized to return True
        mock_ray.is_initialized.return_value = True

        # Initialize the monitor
        monitor = RayGpuMonitorLogger(
            collection_interval=10.0,
            flush_interval=60.0,
            metric_prefix="test",
            step_metric="test/step",
            parent_logger=None,
        )

        # Start the monitor
        monitor.start()

        # Create a spy for the flush method
        with patch.object(monitor, "flush") as mock_flush:
            # Stop the monitor
            monitor.stop()

            # Verify flush was called
            mock_flush.assert_called_once()

            # Verify monitor state
            assert monitor.is_running is False

    @patch("nemo_rl.utils.logger.ray")
    def test_parse_metric(self, mock_ray):
        """Test _parse_metric method."""
        # Mock ray.is_initialized to return True
        mock_ray.is_initialized.return_value = True

        # Initialize the monitor
        monitor = RayGpuMonitorLogger(
            collection_interval=10.0,
            flush_interval=60.0,
            metric_prefix="test",
            step_metric="test/step",
            parent_logger=None,
        )

        # Create a sample with GPU utilization metric
        from prometheus_client.samples import Sample

        utilization_sample = Sample(
            name="ray_node_gpus_utilization",
            labels={"GpuIndex": "0", "GpuDeviceName": "NVIDIA Test GPU"},
            value=75.5,
            timestamp=None,
            exemplar=None,
        )

        # Parse the sample
        result = monitor._parse_metric(utilization_sample, node_idx=1)

        # Verify the result
        assert result == {"node.1.gpu.0.util": 75.5}

        # Create a sample with GPU memory metric (in MB)
        memory_sample = Sample(
            name="ray_node_gram_used",
            labels={"GpuIndex": "0", "GpuDeviceName": "NVIDIA Test GPU"},
            value=80.0 * 1024,
            timestamp=None,
            exemplar=None,
        )

        # Parse the sample
        result = monitor._parse_metric(memory_sample, node_idx=1)

        # Verify the result
        assert result == {"node.1.gpu.0.mem_gb": 80.0}

        # Create a sample with system memory metric
        system_memory_sample = Sample(
            name="ray_node_mem_used",
            labels={"InstanceId": "n/a"},
            value=100.0 * 1024 * 1024 * 1024,
            timestamp=None,
            exemplar=None,
        )

        # Parse the sample
        result = monitor._parse_metric(system_memory_sample, node_idx=1)

        # Verify the result
        assert result == {"node.1.mem_gb": 100.0}

        # Create a sample with total system memory metric
        total_memory_sample = Sample(
            name="ray_node_mem_total",
            labels={"InstanceId": "n/a"},
            value=200.0 * 1024 * 1024 * 1024,
            timestamp=None,
            exemplar=None,
        )

        # Parse the sample
        result = monitor._parse_metric(total_memory_sample, node_idx=1)

        # Verify the result
        assert result == {"node.1.mem_total_gb": 200.0}

        # Test with an unexpected metric name
        other_sample = Sample(
            name="ray_node_cpu_utilization",
            labels={"GpuIndex": "0"},
            value=50.0,
            timestamp=None,
            exemplar=None,
        )

        # Parse the sample
        result = monitor._parse_metric(other_sample, node_idx=1)

        # Verify the result is empty
        assert result == {}

    @patch("nemo_rl.utils.logger.ray")
    @patch("nemo_rl.utils.logger.requests.get")
    def test_fetch_and_parse_metrics(self, mock_get, mock_ray):
        """Test _fetch_and_parse_metrics method."""
        # Mock ray.is_initialized to return True
        mock_ray.is_initialized.return_value = True

        # Set up mock response with Prometheus metrics
        mock_response = mock_get.return_value
        mock_response.status_code = 200
        # Simplified Prometheus format text with GPU metrics
        mock_response.text = f"""
# HELP ray_node_gpus_utilization GPU utilization
# TYPE ray_node_gpus_utilization gauge
ray_node_gpus_utilization{{GpuIndex="0",GpuDeviceName="NVIDIA Test GPU"}} 75.5
# HELP ray_node_gram_used GPU memory used
# TYPE ray_node_gram_used gauge
ray_node_gram_used{{GpuIndex="0",GpuDeviceName="NVIDIA Test GPU"}} {80.0 * 1024}
        """

        # Initialize the monitor
        monitor = RayGpuMonitorLogger(
            collection_interval=10.0,
            flush_interval=60.0,
            metric_prefix="test",
            step_metric="test/step",
            parent_logger=None,
        )

        # Mock the _parse_metric method to return expected values
        with patch.object(monitor, "_parse_metric") as mock_parse:
            mock_parse.side_effect = [
                {"node.2.gpu.0.util": 75.5},
                {"node.2.gpu.0.mem_gb": 80.0},
            ]

            # Call the method
            result = monitor._fetch_and_parse_metrics(
                node_idx=2,
                metric_address="test_ip:test_port",
                parser_fn=monitor._parse_metric,
            )

            # Verify request was made correctly
            mock_get.assert_called_once_with(
                "http://test_ip:test_port/metrics", timeout=5.0
            )

            # Verify parsing was done for both metrics
            assert mock_parse.call_count == 2

            # Verify the result combines both metrics
            assert result == {
                "node.2.gpu.0.util": 75.5,
                "node.2.gpu.0.mem_gb": 80.0,
            }

    @patch("nemo_rl.utils.logger.ray")
    def test_collect_metrics(self, mock_ray):
        """Test _collect_metrics method."""
        # Mock ray.is_initialized to return True
        mock_ray.is_initialized.return_value = True

        # Mock ray.nodes to return test nodes
        mock_ray.nodes.return_value = [
            {"NodeManagerAddress": "10.0.0.1", "MetricsExportPort": 8080},
            {"NodeManagerAddress": "10.0.0.2", "MetricsExportPort": 8080},
        ]

        # Initialize the monitor
        monitor = RayGpuMonitorLogger(
            collection_interval=10.0,
            flush_interval=60.0,
            metric_prefix="test",
            step_metric="test/step",
            parent_logger=None,
        )

        # Mock the _fetch_and_parse_metrics method
        with patch.object(monitor, "_fetch_and_parse_metrics") as mock_fetch:
            mock_fetch.side_effect = [
                {"node.0.gpu.0.util": 75.5, "node.0.gpu.0.mem_gb": 80.0},
                {"node.1.gpu.0.util": 50.0, "node.1.gpu.0.mem_gb": 20.0},
            ]

            # Call the method
            result = monitor._collect_metrics()

            # Verify _fetch_and_parse_metrics was called for each node
            assert mock_fetch.call_count == 2
            mock_fetch.assert_any_call(0, "10.0.0.1:8080", monitor._parse_metric)
            mock_fetch.assert_any_call(1, "10.0.0.2:8080", monitor._parse_metric)

            # Verify the result combines metrics from all nodes
            assert result == {
                "node.0.gpu.0.util": 75.5,
                "node.0.gpu.0.mem_gb": 80.0,
                "node.1.gpu.0.util": 50.0,
                "node.1.gpu.0.mem_gb": 20.0,
            }

    @patch("nemo_rl.utils.logger.ray")
    def test_flush_empty_buffer(self, mock_ray, mock_parent_logger):
        """Test flush method with empty buffer."""
        # Mock ray.is_initialized to return True
        mock_ray.is_initialized.return_value = True

        # Initialize the monitor with parent logger
        monitor = RayGpuMonitorLogger(
            collection_interval=10.0,
            flush_interval=60.0,
            metric_prefix="ray",
            step_metric="ray/ray_step",
            parent_logger=mock_parent_logger,
        )

        # Call flush with empty buffer
        monitor.flush()

        # Verify parent logger's log_metrics was not called
        assert len(mock_parent_logger.logged_metrics) == 0

    @patch("nemo_rl.utils.logger.ray")
    def test_flush(self, mock_ray, mock_parent_logger):
        """Test flush method with metrics in buffer."""
        # Mock ray.is_initialized to return True
        mock_ray.is_initialized.return_value = True

        # Initialize the monitor with parent logger
        monitor = RayGpuMonitorLogger(
            collection_interval=10.0,
            flush_interval=60.0,
            metric_prefix="ray",
            step_metric="ray/ray_step",
            parent_logger=mock_parent_logger,
        )

        # Add test metrics to buffer
        monitor.metrics_buffer = [
            {
                "step": 10,
                "metrics": {"node.0.gpu.0.util": 75.5, "node.0.gpu.0.mem_gb": 80.0},
            },
            {
                "step": 20,
                "metrics": {"node.0.gpu.0.util": 80.0, "node.0.gpu.0.mem_gb": 5120.0},
            },
        ]

        # Call flush
        monitor.flush()

        # Verify parent logger's log_metrics was called for each entry
        assert len(mock_parent_logger.logged_metrics) == 2

        # First metrics entry should include the step metric
        expected_first_metrics = {
            "node.0.gpu.0.util": 75.5,
            "node.0.gpu.0.mem_gb": 80.0,
            "ray/ray_step": 10,  # Step metric added
        }
        assert mock_parent_logger.logged_metrics[0] == expected_first_metrics
        assert mock_parent_logger.logged_steps[0] == 10
        assert mock_parent_logger.logged_prefixes[0] == "ray"
        assert mock_parent_logger.logged_step_metrics[0] == "ray/ray_step"

        # Second metrics entry should include the step metric
        expected_second_metrics = {
            "node.0.gpu.0.util": 80.0,
            "node.0.gpu.0.mem_gb": 5120.0,
            "ray/ray_step": 20,  # Step metric added
        }
        assert mock_parent_logger.logged_metrics[1] == expected_second_metrics
        assert mock_parent_logger.logged_steps[1] == 20
        assert mock_parent_logger.logged_prefixes[1] == "ray"
        assert mock_parent_logger.logged_step_metrics[1] == "ray/ray_step"

        # Verify buffer was cleared
        assert monitor.metrics_buffer == []

    @patch("nemo_rl.utils.logger.ray")
    def test_flush_with_custom_prefix(self, mock_ray, mock_parent_logger):
        """Test flush method with custom metric prefix."""
        # Mock ray.is_initialized to return True
        mock_ray.is_initialized.return_value = True

        # Initialize the monitor with parent logger and custom prefix
        custom_prefix = "custom_metrics"
        custom_step_metric = "custom_metrics/step"
        monitor = RayGpuMonitorLogger(
            collection_interval=10.0,
            flush_interval=60.0,
            metric_prefix=custom_prefix,
            step_metric=custom_step_metric,
            parent_logger=mock_parent_logger,
        )

        # Add test metrics to buffer
        monitor.metrics_buffer = [
            {
                "step": 15,
                "metrics": {"node.0.gpu.0.util": 60.0},
            }
        ]

        # Call flush
        monitor.flush()

        # Verify parent logger's log_metrics was called with the custom prefix
        assert len(mock_parent_logger.logged_metrics) == 1
        expected_metrics = {"node.0.gpu.0.util": 60.0, "custom_metrics/step": 15}
        assert mock_parent_logger.logged_metrics[0] == expected_metrics
        assert mock_parent_logger.logged_steps[0] == 15
        assert mock_parent_logger.logged_prefixes[0] == custom_prefix
        assert mock_parent_logger.logged_step_metrics[0] == custom_step_metric

    @patch("nemo_rl.utils.logger.ray")
    @patch("nemo_rl.utils.logger.time")
    def test_collection_loop(self, mock_time, mock_ray):
        """Test _collection_loop method (one iteration)."""
        # Mock ray.is_initialized to return True
        mock_ray.is_initialized.return_value = True

        # Set up time mocks for a single iteration
        mock_time.time.side_effect = [
            100.0,
            110.0,
            170.0,
            180.0,
        ]  # start_time, collection_time, flush_check_time, sleep_until

        # Initialize the monitor
        monitor = RayGpuMonitorLogger(
            collection_interval=10.0,
            flush_interval=60.0,
            metric_prefix="test",
            step_metric="test/step",
            parent_logger=None,
        )

        # Set start time and running flag
        monitor.start_time = 100.0
        monitor.is_running = True

        # Create a flag to only run one iteration
        monitor.iteration_done = False

        def side_effect():
            if not monitor.iteration_done:
                monitor.iteration_done = True
                return {"node.0.gpu.0.util": 75.5}
            else:
                monitor.is_running = False
                return {}

        # Mock _collect_metrics to return test metrics
        with patch.object(monitor, "_collect_metrics", side_effect=side_effect):
            # Mock flush method
            with patch.object(monitor, "flush") as mock_flush:
                # Run the collection loop (will stop after one iteration)
                monitor._collection_loop()

                # Verify monitor.metrics_buffer has the collected metrics
                assert len(monitor.metrics_buffer) == 1
                assert (
                    monitor.metrics_buffer[0]["step"] == 10
                )  # relative time (110 - 100)
                assert monitor.metrics_buffer[0]["metrics"] == {
                    "node.0.gpu.0.util": 75.5
                }

                # Verify flush was called (flush_interval elapsed)
                mock_flush.assert_called_once()

    @patch("nemo_rl.utils.logger.WandbLogger")
    @patch("nemo_rl.utils.logger.TensorboardLogger")
    @patch("nemo_rl.utils.logger.RayGpuMonitorLogger")
    def test_init_with_gpu_monitoring(
        self, mock_gpu_monitor, mock_tb_logger, mock_wandb_logger, temp_dir
    ):
        """Test initialization with GPU monitoring enabled."""
        cfg = {
            "wandb_enabled": True,
            "tensorboard_enabled": True,
            "mlflow_enabled": False,
            "swanlab_enabled": False,
            "monitor_gpus": True,
            "gpu_monitoring": {
                "collection_interval": 15.0,
                "flush_interval": 45.0,
            },
            "wandb": {"project": "test-project"},
            "tensorboard": {"log_dir": "test_logs"},
            "log_dir": temp_dir,
        }
        logger = Logger(cfg)

        # Check that regular loggers were initialized
        assert len(logger.loggers) == 2
        mock_wandb_logger.assert_called_once()
        mock_tb_logger.assert_called_once()

        # Check that GPU monitor was initialized with correct parameters
        mock_gpu_monitor.assert_called_once_with(
            collection_interval=15.0,
            flush_interval=45.0,
            metric_prefix="ray",
            step_metric="ray/ray_step",
            parent_logger=logger,
        )

        # Check that GPU monitor was started
        mock_gpu_instance = mock_gpu_monitor.return_value
        mock_gpu_instance.start.assert_called_once()

        # Check that wandb metrics are defined with the step metric
        mock_wandb_instance = mock_wandb_logger.return_value
        mock_wandb_instance.define_metric.assert_called_once_with(
            "ray/*", step_metric="ray/ray_step"
        )

    @patch("nemo_rl.utils.logger.WandbLogger")
    @patch("nemo_rl.utils.logger.TensorboardLogger")
    @patch("nemo_rl.utils.logger.RayGpuMonitorLogger")
    def test_gpu_monitoring_without_wandb(
        self, mock_gpu_monitor, mock_tb_logger, mock_wandb_logger, temp_dir
    ):
        """Test GPU monitoring initialization when wandb is disabled."""
        cfg = {
            "wandb_enabled": False,
            "tensorboard_enabled": True,
            "mlflow_enabled": False,
            "swanlab_enabled": False,
            "monitor_gpus": True,
            "gpu_monitoring": {
                "collection_interval": 15.0,
                "flush_interval": 45.0,
            },
            "tensorboard": {"log_dir": "test_logs"},
            "log_dir": temp_dir,
        }
        logger = Logger(cfg)

        # Check that only tensorboard logger was initialized
        assert len(logger.loggers) == 1
        mock_wandb_logger.assert_not_called()
        mock_tb_logger.assert_called_once()

        # Check that GPU monitor was initialized with correct parameters
        mock_gpu_monitor.assert_called_once_with(
            collection_interval=15.0,
            flush_interval=45.0,
            metric_prefix="ray",
            step_metric="ray/ray_step",
            parent_logger=logger,
        )

        # Since wandb is disabled, define_metric should not be called
        mock_wandb_instance = mock_wandb_logger.return_value
        assert not mock_wandb_instance.define_metric.called

    @patch("nemo_rl.utils.logger.WandbLogger")
    @patch("nemo_rl.utils.logger.TensorboardLogger")
    @patch("nemo_rl.utils.logger.RayGpuMonitorLogger")
    def test_gpu_monitoring_no_main_loggers(
        self, mock_gpu_monitor, mock_tb_logger, mock_wandb_logger, temp_dir
    ):
        """Test GPU monitoring initialization when no main loggers (wandb/tensorboard) are enabled."""
        cfg = {
            "wandb_enabled": False,
            "swanlab_enabled": False,
            "tensorboard_enabled": False,
            "mlflow_enabled": False,
            "monitor_gpus": True,
            "gpu_monitoring": {
                "collection_interval": 15.0,
                "flush_interval": 45.0,
            },
            "log_dir": temp_dir,
        }
        logger = Logger(cfg)

        # Check that regular loggers were NOT initialized
        assert len(logger.loggers) == 0
        mock_wandb_logger.assert_not_called()
        mock_tb_logger.assert_not_called()

        # Check that GPU monitor was initialized with correct parameters
        mock_gpu_monitor.assert_called_once_with(
            collection_interval=15.0,
            flush_interval=45.0,
            metric_prefix="ray",
            step_metric="ray/ray_step",
            parent_logger=logger,  # Logger instance is passed as parent
        )

        # Check that GPU monitor was started
        mock_gpu_instance = mock_gpu_monitor.return_value
        mock_gpu_instance.start.assert_called_once()

        # Since wandb is disabled, self.wandb_logger would be None,
        # and define_metric should not be called on it.
        # We access the mock_wandb_logger.return_value which is the mock object itself.
        mock_wandb_instance = mock_wandb_logger.return_value
        assert not mock_wandb_instance.define_metric.called


class TestLogger:
    """Test the main Logger class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for logs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @patch("nemo_rl.utils.logger.WandbLogger")
    @patch("nemo_rl.utils.logger.TensorboardLogger")
    def test_init_no_loggers(self, mock_tb_logger, mock_wandb_logger, temp_dir):
        """Test initialization with no loggers enabled."""
        cfg = {
            "wandb_enabled": False,
            "tensorboard_enabled": False,
            "mlflow_enabled": False,
            "swanlab_enabled": False,
            "monitor_gpus": False,
            "log_dir": temp_dir,
        }
        logger = Logger(cfg)

        assert len(logger.loggers) == 0
        mock_tb_logger.assert_not_called()
        mock_wandb_logger.assert_not_called()

    @patch("nemo_rl.utils.logger.WandbLogger")
    @patch("nemo_rl.utils.logger.TensorboardLogger")
    def test_init_wandb_only(self, mock_tb_logger, mock_wandb_logger, temp_dir):
        """Test initialization with only WandbLogger enabled."""
        cfg = {
            "wandb_enabled": True,
            "tensorboard_enabled": False,
            "mlflow_enabled": False,
            "swanlab_enabled": False,
            "monitor_gpus": False,
            "wandb": {"project": "test-project"},
            "log_dir": temp_dir,
        }
        logger = Logger(cfg)

        assert len(logger.loggers) == 1
        mock_wandb_logger.assert_called_once()
        wandb_cfg = mock_wandb_logger.call_args[0][0]
        assert wandb_cfg == {"project": "test-project"}
        mock_tb_logger.assert_not_called()

    @patch("nemo_rl.utils.logger.WandbLogger")
    @patch("nemo_rl.utils.logger.SwanlabLogger")
    @patch("nemo_rl.utils.logger.TensorboardLogger")
    def test_init_swanlab_only(self, mock_tb_logger, mock_swanlab_logger, temp_dir):
        """Test initialization with only SwanlabLogger enabled."""
        cfg = {
            "wandb_enabled": False,
            "tensorboard_enabled": False,
            "mlflow_enabled": False,
            "swanlab_enabled": True,
            "monitor_gpus": False,
            "swanlab": {"project": "test-project"},
            "log_dir": temp_dir,
        }
        logger = Logger(cfg)

        assert len(logger.loggers) == 1
        mock_swanlab_logger.assert_called_once()
        swanlab_cfg = mock_swanlab_logger.call_args[0][0]
        assert swanlab_cfg == {"project": "test-project"}
        mock_tb_logger.assert_not_called()

    @patch("nemo_rl.utils.logger.WandbLogger")
    @patch("nemo_rl.utils.logger.TensorboardLogger")
    def test_init_tensorboard_only(self, mock_tb_logger, mock_wandb_logger, temp_dir):
        """Test initialization with only TensorboardLogger enabled."""
        cfg = {
            "wandb_enabled": False,
            "tensorboard_enabled": True,
            "mlflow_enabled": False,
            "swanlab_enabled": False,
            "monitor_gpus": False,
            "tensorboard": {"log_dir": "test_logs"},
            "log_dir": temp_dir,
        }
        logger = Logger(cfg)

        assert len(logger.loggers) == 1
        mock_tb_logger.assert_called_once()
        tb_cfg = mock_tb_logger.call_args[0][0]
        assert tb_cfg == {"log_dir": "test_logs"}
        mock_wandb_logger.assert_not_called()

    @patch("nemo_rl.utils.logger.WandbLogger")
    @patch("nemo_rl.utils.logger.TensorboardLogger")
    def test_init_both_loggers(self, mock_tb_logger, mock_wandb_logger, temp_dir):
        """Test initialization with both loggers enabled."""
        cfg = {
            "wandb_enabled": True,
            "tensorboard_enabled": True,
            "mlflow_enabled": False,
            "swanlab_enabled": False,
            "monitor_gpus": False,
            "wandb": {"project": "test-project"},
            "tensorboard": {"log_dir": "test_logs"},
            "log_dir": temp_dir,
        }
        logger = Logger(cfg)

        assert len(logger.loggers) == 2
        mock_wandb_logger.assert_called_once()
        wandb_cfg = mock_wandb_logger.call_args[0][0]
        assert wandb_cfg == {"project": "test-project"}

        mock_tb_logger.assert_called_once()
        tb_cfg = mock_tb_logger.call_args[0][0]
        assert tb_cfg == {"log_dir": "test_logs"}

    @patch("nemo_rl.utils.logger.WandbLogger")
    @patch("nemo_rl.utils.logger.TensorboardLogger")
    def test_log_metrics(self, mock_tb_logger, mock_wandb_logger, temp_dir):
        """Test logging metrics to all enabled loggers."""
        cfg = {
            "wandb_enabled": True,
            "tensorboard_enabled": True,
            "mlflow_enabled": False,
            "swanlab_enabled": False,
            "monitor_gpus": False,
            "wandb": {"project": "test-project"},
            "tensorboard": {"log_dir": "test_logs"},
            "log_dir": temp_dir,
        }
        logger = Logger(cfg)

        # Create mock logger instances
        mock_wandb_instance = mock_wandb_logger.return_value
        mock_tb_instance = mock_tb_logger.return_value

        metrics = {"loss": 0.5, "accuracy": 0.8}
        step = 10
        logger.log_metrics(metrics, step)

        # Check that log_metrics was called on both loggers
        mock_wandb_instance.log_metrics.assert_called_once_with(
            metrics, step, "", None, False
        )
        mock_tb_instance.log_metrics.assert_called_once_with(
            metrics, step, "", None, False
        )

    @patch("nemo_rl.utils.logger.WandbLogger")
    @patch("nemo_rl.utils.logger.TensorboardLogger")
    def test_log_hyperparams(self, mock_tb_logger, mock_wandb_logger, temp_dir):
        """Test logging hyperparameters to all enabled loggers."""
        cfg = {
            "wandb_enabled": True,
            "tensorboard_enabled": True,
            "mlflow_enabled": False,
            "swanlab_enabled": False,
            "monitor_gpus": False,
            "wandb": {"project": "test-project"},
            "tensorboard": {"log_dir": "test_logs"},
            "log_dir": temp_dir,
        }
        logger = Logger(cfg)

        # Create mock logger instances
        mock_wandb_instance = mock_wandb_logger.return_value
        mock_tb_instance = mock_tb_logger.return_value

        params = {"lr": 0.001, "batch_size": 32}
        logger.log_hyperparams(params)

        # Check that log_hyperparams was called on both loggers
        mock_wandb_instance.log_hyperparams.assert_called_once_with(params)
        mock_tb_instance.log_hyperparams.assert_called_once_with(params)

    @patch("nemo_rl.utils.logger.WandbLogger")
    @patch("nemo_rl.utils.logger.TensorboardLogger")
    @patch("nemo_rl.utils.logger.RayGpuMonitorLogger")
    def test_init_with_gpu_monitoring(
        self, mock_gpu_monitor, mock_tb_logger, mock_wandb_logger, temp_dir
    ):
        """Test initialization with GPU monitoring enabled."""
        cfg = {
            "wandb_enabled": True,
            "tensorboard_enabled": True,
            "mlflow_enabled": False,
            "swanlab_enabled": False,
            "monitor_gpus": True,
            "gpu_monitoring": {
                "collection_interval": 15.0,
                "flush_interval": 45.0,
            },
            "wandb": {"project": "test-project"},
            "tensorboard": {"log_dir": "test_logs"},
            "log_dir": temp_dir,
        }
        logger = Logger(cfg)

        # Check that regular loggers were initialized
        assert len(logger.loggers) == 2
        mock_wandb_logger.assert_called_once()
        mock_tb_logger.assert_called_once()

        # Check that GPU monitor was initialized with correct parameters
        mock_gpu_monitor.assert_called_once_with(
            collection_interval=15.0,
            flush_interval=45.0,
            metric_prefix="ray",
            step_metric="ray/ray_step",
            parent_logger=logger,
        )

        # Check that GPU monitor was started
        mock_gpu_instance = mock_gpu_monitor.return_value
        mock_gpu_instance.start.assert_called_once()

        # Check that wandb metrics are defined with the step metric
        mock_wandb_instance = mock_wandb_logger.return_value
        mock_wandb_instance.define_metric.assert_called_once_with(
            "ray/*", step_metric="ray/ray_step"
        )

    @patch("nemo_rl.utils.logger.WandbLogger")
    @patch("nemo_rl.utils.logger.TensorboardLogger")
    def test_log_metrics_with_prefix_and_step_metric(
        self, mock_tb_logger, mock_wandb_logger, temp_dir
    ):
        """Test logging metrics with prefix and step_metric."""
        cfg = {
            "wandb_enabled": True,
            "tensorboard_enabled": True,
            "mlflow_enabled": False,
            "swanlab_enabled": False,
            "monitor_gpus": False,
            "wandb": {"project": "test-project"},
            "tensorboard": {"log_dir": "test_logs"},
            "log_dir": temp_dir,
        }
        logger = Logger(cfg)

        # Create mock logger instances
        mock_wandb_instance = mock_wandb_logger.return_value
        mock_tb_instance = mock_tb_logger.return_value

        # Create metrics with a step metric field
        metrics = {"loss": 0.5, "accuracy": 0.8, "iteration": 15}
        step = 10
        prefix = "train"
        step_metric = "iteration"

        # Log metrics with prefix and step_metric
        logger.log_metrics(metrics, step, prefix=prefix, step_metric=step_metric)

        # Check that log_metrics was called on both loggers with correct parameters
        mock_wandb_instance.log_metrics.assert_called_once_with(
            metrics, step, prefix, step_metric, False
        )
        mock_tb_instance.log_metrics.assert_called_once_with(
            metrics, step, prefix, step_metric, False
        )

    @patch("nemo_rl.utils.logger.WandbLogger")
    @patch("nemo_rl.utils.logger.TensorboardLogger")
    def test_log_plot_token_mult_prob_error(
        self, mock_tb_logger, mock_wandb_logger, temp_dir
    ):
        """Test logging token probability error plots."""
        cfg = {
            "wandb_enabled": True,
            "tensorboard_enabled": True,
            "mlflow_enabled": False,
            "swanlab_enabled": False,
            "monitor_gpus": False,
            "wandb": {"project": "test-project"},
            "tensorboard": {"log_dir": "test_logs"},
            "log_dir": temp_dir,
        }
        logger = Logger(cfg)

        # Create mock logger instances
        mock_wandb_instance = mock_wandb_logger.return_value
        mock_tb_instance = mock_tb_logger.return_value

        # Create sample data
        data = {
            "token_mask": torch.ones((1, 10)),
            "sample_mask": torch.ones(1),
            "generation_logprobs": torch.randn((1, 10)),
            "prev_logprobs": torch.randn((1, 10)),
            "prompt_lengths": torch.tensor([2]),
            "full_lengths": torch.tensor([8]),
        }
        step = 10
        name = "test_plot"

        # Log the plot
        logger.log_plot_token_mult_prob_error(data, step, name)

        # Check that log_plot was called on both loggers
        mock_wandb_instance.log_plot.assert_called_once()
        mock_tb_instance.log_plot.assert_called_once()

        # Verify the plot was created with correct data
        wandb_call_args = mock_wandb_instance.log_plot.call_args[0]
        fig = wandb_call_args[0]
        ax = fig.axes[0]

        # Check plot elements
        assert len(ax.lines) == 5  # Three lines for data + two points for max errors
        assert ax.get_xlabel() == "Token Position (starting from prompt end)"
        assert ax.get_ylabel() == "Log Probability/Difference"
        assert ax.get_legend() is not None  # Legend should exist

        # Verify the legend labels
        legend_texts = [text.get_text() for text in ax.get_legend().get_texts()]
        assert any("Max abs error" in text for text in legend_texts)
        assert any("Max rel error (prob)" in text for text in legend_texts)

    @patch("nemo_rl.utils.logger.WandbLogger")
    @patch("nemo_rl.utils.logger.TensorboardLogger")
    def test_init_mlflow_only(self, mock_tb_logger, mock_wandb_logger, temp_dir):
        """Test initialization with only MLflowLogger enabled."""
        cfg = {
            "wandb_enabled": False,
            "tensorboard_enabled": False,
            "mlflow_enabled": True,
            "swanlab_enabled": False,
            "monitor_gpus": False,
            "mlflow": {
                "experiment_name": "test-experiment",
                "tracking_uri": None,
                "run_name": "test-run",
            },
            "log_dir": temp_dir,
        }
        logger = Logger(cfg)

        assert len(logger.loggers) == 1
        mock_wandb_logger.assert_not_called()
        mock_tb_logger.assert_not_called()

    @patch("nemo_rl.utils.logger.WandbLogger")
    @patch("nemo_rl.utils.logger.TensorboardLogger")
    @patch("nemo_rl.utils.logger.MLflowLogger")
    @patch("nemo_rl.utils.logger.SwanlabLogger")
    def test_init_all_loggers(
        self,
        mock_swanlab_logger,
        mock_mlflow_logger,
        mock_tb_logger,
        mock_wandb_logger,
        temp_dir,
    ):
        """Test initialization with all loggers enabled."""
        cfg = {
            "wandb_enabled": True,
            "tensorboard_enabled": True,
            "mlflow_enabled": True,
            "swanlab_enabled": True,
            "monitor_gpus": False,
            "wandb": {"project": "test-project"},
            "swanlab": {"project": "test-project"},
            "tensorboard": {"log_dir": "test_logs"},
            "mlflow": {
                "experiment_name": "test-experiment",
                "tracking_uri": None,
                "run_name": "test-run",
            },
            "log_dir": temp_dir,
        }
        logger = Logger(cfg)

        assert len(logger.loggers) == 4
        mock_wandb_logger.assert_called_once()
        mock_tb_logger.assert_called_once()
        mock_mlflow_logger.assert_called_once()
        mock_swanlab_logger.assert_called_once()

    @patch("nemo_rl.utils.logger.WandbLogger")
    @patch("nemo_rl.utils.logger.TensorboardLogger")
    @patch("nemo_rl.utils.logger.MLflowLogger")
    @patch("nemo_rl.utils.logger.SwanlabLogger")
    def test_log_metrics_with_mlflow(
        self,
        mock_swanlab_logger,
        mock_mlflow_logger,
        mock_tb_logger,
        mock_wandb_logger,
        temp_dir,
    ):
        """Test logging metrics to all enabled loggers including MLflow."""
        cfg = {
            "wandb_enabled": True,
            "swanlab_enabled": True,
            "tensorboard_enabled": True,
            "mlflow_enabled": True,
            "monitor_gpus": False,
            "wandb": {"project": "test-project"},
            "swanlab": {"project": "test-project"},
            "tensorboard": {"log_dir": "test_logs"},
            "mlflow": {
                "experiment_name": "test-experiment",
                "tracking_uri": None,
                "run_name": "test-run",
            },
            "log_dir": temp_dir,
        }
        logger = Logger(cfg)

        # Create mock logger instances
        mock_wandb_instance = mock_wandb_logger.return_value
        mock_tb_instance = mock_tb_logger.return_value
        mock_mlflow_instance = mock_mlflow_logger.return_value
        mock_swanlab_instance = mock_swanlab_logger.return_value

        metrics = {"loss": 0.5, "accuracy": 0.8}
        step = 10
        logger.log_metrics(metrics, step)

        # Check that log_metrics was called on all loggers
        mock_wandb_instance.log_metrics.assert_called_once_with(
            metrics, step, "", None, False
        )
        mock_swanlab_instance.log_metrics.assert_called_once_with(
            metrics, step, "", None, False
        )
        mock_tb_instance.log_metrics.assert_called_once_with(
            metrics, step, "", None, False
        )
        mock_mlflow_instance.log_metrics.assert_called_once_with(
            metrics, step, "", None, False
        )

    @patch("nemo_rl.utils.logger.WandbLogger")
    @patch("nemo_rl.utils.logger.TensorboardLogger")
    @patch("nemo_rl.utils.logger.MLflowLogger")
    @patch("nemo_rl.utils.logger.SwanlabLogger")
    def test_log_hyperparams_with_mlflow(
        self,
        mock_swanlab_logger,
        mock_mlflow_logger,
        mock_tb_logger,
        mock_wandb_logger,
        temp_dir,
    ):
        """Test logging hyperparameters to all enabled loggers including MLflow."""
        cfg = {
            "wandb_enabled": True,
            "swanlab_enabled": True,
            "tensorboard_enabled": True,
            "mlflow_enabled": True,
            "monitor_gpus": False,
            "wandb": {"project": "test-project"},
            "swanlab": {"project": "test-project"},
            "tensorboard": {"log_dir": "test_logs"},
            "mlflow": {"experiment_name": "test-experiment"},
            "log_dir": temp_dir,
        }
        logger = Logger(cfg)

        # Create mock logger instances
        mock_wandb_instance = mock_wandb_logger.return_value
        mock_tb_instance = mock_tb_logger.return_value
        mock_mlflow_instance = mock_mlflow_logger.return_value
        mock_swanlab_instance = mock_swanlab_logger.return_value

        params = {"lr": 0.001, "batch_size": 32}
        logger.log_hyperparams(params)

        # Check that log_hyperparams was called on all loggers
        mock_wandb_instance.log_hyperparams.assert_called_once_with(params)
        mock_tb_instance.log_hyperparams.assert_called_once_with(params)
        mock_mlflow_instance.log_hyperparams.assert_called_once_with(params)
        mock_swanlab_instance.log_hyperparams.assert_called_once_with(params)

    def test_log_plot_per_worker_timeline_metrics_logs_expected_series(self):
        """Ensure per-worker and average plots are produced and logged."""
        logger = Logger.__new__(Logger)
        backend_logger = MagicMock()
        logger.loggers = [backend_logger]

        metrics = {
            0: [1, 2, 3],
            1: [2, 3, 4],
        }

        mock_fig_worker, mock_ax_worker = MagicMock(), MagicMock()
        mock_fig_avg, mock_ax_avg = MagicMock(), MagicMock()

        with (
            patch(
                "nemo_rl.utils.logger.plt.subplots",
                side_effect=[
                    (mock_fig_worker, mock_ax_worker),
                    (mock_fig_avg, mock_ax_avg),
                ],
            ) as mock_subplots,
            patch("nemo_rl.utils.logger.plt.close") as mock_close,
        ):
            logger.log_plot_per_worker_timeline_metrics(
                metrics,
                step=1,
                prefix="vllm",
                name="kv_cache",
                timeline_interval=0.5,
            )

        assert mock_subplots.call_count == 2
        expected_x = [0.0, 0.5, 1.0]
        mock_ax_worker.plot.assert_has_calls(
            [
                call(expected_x, [1.0, 2.0, 3.0], label="worker_0"),
                call(expected_x, [2.0, 3.0, 4.0], label="worker_1"),
            ],
            any_order=False,
        )

        avg_call = mock_ax_avg.plot.call_args_list[0]
        assert avg_call.args[0] == expected_x
        assert avg_call.args[1].tolist() == [1.5, 2.5, 3.5]
        assert avg_call.kwargs["label"] == "average"

        backend_logger.log_plot.assert_has_calls(
            [
                call(mock_fig_worker, 1, "vllm/per_worker_kv_cache"),
                call(mock_fig_avg, 1, "vllm/average_kv_cache"),
            ],
            any_order=False,
        )
        assert mock_close.call_args_list == [
            call(mock_fig_worker),
            call(mock_fig_avg),
        ]

    def test_log_plot_per_worker_timeline_metrics_requires_positive_interval(self):
        """timeline_interval must be positive."""
        logger = Logger.__new__(Logger)
        logger.loggers = [MagicMock()]

        with pytest.raises(ValueError):
            logger.log_plot_per_worker_timeline_metrics(
                metrics={0: [1, 2]},
                step=1,
                prefix="train",
                name="pending",
                timeline_interval=0.0,
            )

    def test_log_plot_per_worker_timeline_metrics_skips_when_no_data(self):
        """No plots should be produced when metrics are empty."""
        logger = Logger.__new__(Logger)
        backend_logger = MagicMock()
        logger.loggers = [backend_logger]

        with patch("nemo_rl.utils.logger.plt.subplots") as mock_subplots:
            logger.log_plot_per_worker_timeline_metrics(
                metrics={},
                step=1,
                prefix="train",
                name="pending",
                timeline_interval=1.0,
            )

        mock_subplots.assert_not_called()
        backend_logger.log_plot.assert_not_called()


def test_print_message_log_samples(capsys):
    """Test that print_message_log_samples displays full content correctly."""
    # Test message with full content (verifies our bug fix)
    message_logs = [
        [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 = 4"},
        ]
    ]
    rewards = [1.0]

    print_message_log_samples(message_logs, rewards, num_samples=1, step=0)

    captured = capsys.readouterr()
    # Verify content is displayed properly
    assert "What is 2+2?" in captured.out
    assert "2+2 = 4" in captured.out
    assert "Sample 1 | Reward: 1.0000" in captured.out
