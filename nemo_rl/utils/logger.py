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


import glob
import json
import logging
import os
import re
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Mapping, Optional, TypedDict

import ray
import requests
import torch
import wandb
from matplotlib import pyplot as plt
from prometheus_client.parser import text_string_to_metric_families
from prometheus_client.samples import Sample
from rich.box import ROUNDED
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from torch.utils.tensorboard import SummaryWriter

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

# Flag to track if rich logging has been configured
_rich_logging_configured = False


class WandbConfig(TypedDict):
    project: str
    name: str


class TensorboardConfig(TypedDict):
    log_dir: str


class GPUMonitoringConfig(TypedDict):
    collection_interval: int | float
    flush_interval: int | float


class LoggerConfig(TypedDict):
    log_dir: str
    wandb_enabled: bool
    tensorboard_enabled: bool
    wandb: WandbConfig
    tensorboard: TensorboardConfig
    monitor_gpus: bool
    gpu_monitoring: GPUMonitoringConfig


class LoggerInterface(ABC):
    """Abstract base class for logger backends."""

    @abstractmethod
    def log_metrics(
        self,
        metrics: dict[str, Any],
        step: int,
        prefix: Optional[str] = "",
        step_metric: Optional[str] = None,
    ) -> None:
        """Log a dictionary of metrics."""
        pass

    @abstractmethod
    def log_hyperparams(self, params: Mapping[str, Any]) -> None:
        """Log dictionary of hyperparameters."""
        pass


class TensorboardLogger(LoggerInterface):
    """Tensorboard logger backend."""

    def __init__(self, cfg: TensorboardConfig, log_dir: Optional[str] = None):
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"Initialized TensorboardLogger at {log_dir}")

    def log_metrics(
        self,
        metrics: dict[str, Any],
        step: int,
        prefix: Optional[str] = "",
        step_metric: Optional[str] = None,  # ignored in TensorBoard
    ) -> None:
        """Log metrics to Tensorboard.

        Args:
            metrics: Dict of metrics to log
            step: Global step value
            prefix: Optional prefix for metric names
            step_metric: Optional step metric name (ignored in TensorBoard)
        """
        for name, value in metrics.items():
            if prefix:
                name = f"{prefix}/{name}"
            self.writer.add_scalar(name, value, step)

    def log_hyperparams(self, params: Mapping[str, Any]) -> None:
        """Log hyperparameters to Tensorboard.

        Args:
            params: Dictionary of hyperparameters to log
        """
        # Flatten the params because add_hparams does not support nested dicts
        self.writer.add_hparams(flatten_dict(params), {})

    def log_plot(self, figure: plt.Figure, step: int, name: str) -> None:
        """Log a plot to Tensorboard.

        Args:
            plot_data: Dictionary of plot data
            step: Global step value
        """
        self.writer.add_figure(name, figure, step)


class WandbLogger(LoggerInterface):
    """Weights & Biases logger backend."""

    def __init__(self, cfg: WandbConfig, log_dir: Optional[str] = None):
        self.run = wandb.init(**cfg, dir=log_dir)
        self._log_code()
        self._log_diffs()
        print(
            f"Initialized WandbLogger for project {cfg.get('project')}, run {cfg.get('name')} at {log_dir}"
        )

    def _log_diffs(self):
        """Log git diffs to wandb.

        This function captures and logs two types of diffs:
        1. Uncommitted changes (working tree diff against HEAD)
        2. All changes (including uncommitted) against the main branch

        Each diff is saved as a text file in a wandb artifact.
        """
        try:
            branch_result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            current_branch = branch_result.stdout.strip()

            diff_artifact = wandb.Artifact(
                name=f"git-diffs-{self.run.project}-{self.run.id}", type="git-diffs"
            )

            # 1. Log uncommitted changes (working tree diff)
            uncommitted_result = subprocess.run(
                ["git", "diff", "HEAD"], capture_output=True, text=True, check=True
            )
            uncommitted_diff = uncommitted_result.stdout

            if uncommitted_diff:
                diff_path = os.path.join(
                    wandb.run.dir if wandb.run else ".", "uncommitted_changes_diff.txt"
                )
                with open(diff_path, "w") as f:
                    f.write(uncommitted_diff)

                # Add file to artifact
                diff_artifact.add_file(diff_path, name="uncommitted_changes_diff.txt")
                print("Logged uncommitted changes diff to wandb")
            else:
                print("No uncommitted changes found")

            # 2. Log diff against main branch (if current branch is not main)
            if current_branch != "main":
                # Log diff between main and working tree (includes uncommitted changes)
                working_diff_result = subprocess.run(
                    ["git", "diff", "main"], capture_output=True, text=True, check=True
                )
                working_diff = working_diff_result.stdout

                if working_diff:
                    # Save diff to a temporary file
                    diff_path = os.path.join(
                        wandb.run.dir if wandb.run else ".", "main_diff.txt"
                    )
                    with open(diff_path, "w") as f:
                        f.write(working_diff)

                    # Add file to artifact
                    diff_artifact.add_file(diff_path, name="main_diff.txt")
                    print("Logged diff against main branch")
                else:
                    print("No differences found between main and working tree")

            self.run.log_artifact(diff_artifact)

        except subprocess.CalledProcessError as e:
            print(f"Error during git operations: {e}")
        except Exception as e:
            print(f"Unexpected error during git diff logging: {e}")

    def _log_code(self):
        """Log code that is tracked by git to wandb.

        This function gets a list of all files tracked by git in the project root
        and manually uploads them to the current wandb run as an artifact.
        """
        try:
            result = subprocess.run(
                ["git", "ls-files"], capture_output=True, text=True, check=True
            )

            tracked_files = result.stdout.strip().split("\n")

            if not tracked_files:
                print(
                    "Warning: No git repository found. Wandb logs will not track code changes for reproducibility."
                )
                return

            code_artifact = wandb.Artifact(
                name=f"source-code-{self.run.project}", type="code"
            )

            for file_path in tracked_files:
                if os.path.isfile(file_path):
                    try:
                        code_artifact.add_file(file_path, name=file_path)
                    except Exception as e:
                        print(f"Error adding file {file_path}: {e}")

            self.run.log_artifact(code_artifact)
            print(f"Logged {len(tracked_files)} git-tracked files to wandb")

        except subprocess.CalledProcessError as e:
            print(f"Error getting git-tracked files: {e}")
        except Exception as e:
            print(f"Unexpected error during git code logging: {e}")

    def define_metric(
        self,
        name: str,
        step_metric: Optional[str] = None,
    ) -> None:
        """Define a metric with custom step metric.

        Args:
            name: Name of the metric or pattern (e.g. 'ray/*')
            step_metric: Optional name of the step metric to use
        """
        self.run.define_metric(name, step_metric=step_metric)

    def log_metrics(
        self,
        metrics: dict[str, Any],
        step: int,
        prefix: Optional[str] = "",
        step_metric: Optional[str] = None,
    ) -> None:
        """Log metrics to wandb.

        Args:
            metrics: Dict of metrics to log
            step: Global step value
            prefix: Optional prefix for metric names
            step_metric: Optional name of a field in metrics to use as step instead
                         of the provided step value
        """
        if prefix:
            metrics = {
                f"{prefix}/{k}" if k != step_metric else k: v
                for k, v in metrics.items()
            }

        # If step_metric is provided, use the corresponding value from metrics as step
        if step_metric and step_metric in metrics:
            # commit=False so the step does not get incremented
            self.run.log(metrics, commit=False)
        else:
            self.run.log(metrics, step=step)

    def log_hyperparams(self, params: Mapping[str, Any]) -> None:
        """Log hyperparameters to wandb.

        Args:
            params: Dict of hyperparameters to log
        """
        self.run.config.update(params)

    def log_plot(self, figure: plt.Figure, step: int, name: str) -> None:
        """Log a plot to wandb.

        Args:
            figure: Matplotlib figure to log
            step: Global step value
        """
        self.run.log({name: figure}, step=step)


class GpuMetricSnapshot(TypedDict):
    step: int
    metrics: dict[str, Any]


class RayGpuMonitorLogger:
    """Monitor GPU utilization across a Ray cluster and log metrics to a parent logger."""

    def __init__(
        self,
        collection_interval: int | float,
        flush_interval: int | float,
        metric_prefix: str,
        step_metric: str,
        parent_logger: Optional["Logger"] = None,
    ):
        """Initialize the GPU monitor.

        Args:
            collection_interval: Interval in seconds to collect GPU metrics
            flush_interval: Interval in seconds to flush metrics to parent logger
            step_metric: Name of the field to use as the step metric
            parent_logger: Logger to receive the collected metrics
        """
        self.collection_interval = collection_interval
        self.flush_interval = flush_interval
        self.metric_prefix = metric_prefix
        self.step_metric = step_metric
        self.parent_logger = parent_logger
        self.metrics_buffer: list[
            GpuMetricSnapshot
        ] = []  # Store metrics with timestamps
        self.last_flush_time = time.time()
        self.is_running = False
        self.collection_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

    def start(self) -> None:
        """Start the GPU monitoring thread."""
        if not ray.is_initialized():
            raise ValueError(
                "Ray must be initialized with nemo_rl.distributed.virtual_cluster.init_ray() before the GPU logging can begin."
            )

        if self.is_running:
            return

        self.start_time = time.time()
        self.is_running = True
        self.collection_thread = threading.Thread(
            target=self._collection_loop,
            daemon=True,  # Make this a daemon thread so it doesn't block program exit
        )
        self.collection_thread.start()
        print(
            f"GPU monitoring started with collection interval={self.collection_interval}s, flush interval={self.flush_interval}s"
        )

    def stop(self) -> None:
        """Stop the GPU monitoring thread."""
        self.is_running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=self.collection_interval * 2)

        # Final flush
        self.flush()
        print("GPU monitoring stopped")

    def _collection_loop(self) -> None:
        """Main collection loop that runs in a separate thread."""
        while self.is_running:
            try:
                collection_time = time.time()
                relative_time = collection_time - self.start_time

                # Collect metrics with timing information
                metrics = self._collect_metrics()
                if metrics:
                    with self.lock:
                        self.metrics_buffer.append(
                            {
                                "step": int(
                                    relative_time
                                ),  # Store the relative time as step
                                "metrics": metrics,
                            }
                        )

                # Check if it's time to flush
                current_time = time.time()
                if current_time - self.last_flush_time >= self.flush_interval:
                    self.flush()
                    self.last_flush_time = current_time

                time.sleep(self.collection_interval)
            except Exception as e:
                print(
                    f"Error in GPU monitoring collection loop or stopped abruptly: {e}"
                )
                time.sleep(self.collection_interval)  # Continue despite errors

    def _parse_metric(self, sample: Sample, node_idx: int) -> dict[str, Any]:
        """Parse a metric sample into a standardized format.

        Args:
            sample: Prometheus metric sample
            node_idx: Index of the node

        Returns:
            Dictionary with metric name and value
        """
        metric_name = sample.name
        labels = sample.labels
        value = sample.value

        if metric_name == "ray_node_gpus_utilization":
            index = labels["GpuIndex"]
            metric_name = f"node.{node_idx}.gpu.{index}.util"
        elif metric_name == "ray_node_gram_used":
            index = labels["GpuIndex"]
            metric_name = f"node.{node_idx}.gpu.{index}.mem_gb"
            # NOTE: It appears their docs say bytes, but it appears to be MB
            value /= 1024
        elif metric_name == "ray_node_mem_used":
            metric_name = f"node.{node_idx}.mem_gb"
            value /= 1024 * 1024 * 1024
        elif metric_name == "ray_node_mem_total":
            metric_name = f"node.{node_idx}.mem_total_gb"
            value /= 1024 * 1024 * 1024
        else:
            # Skip unexpected metrics
            return {}

        return {metric_name: value}

    def _parse_gpu_sku(self, sample: Sample, node_idx: int) -> dict[str, str]:
        """Parse a GPU metric sample into a standardized format.

        Args:
            sample: Prometheus metric sample
            node_idx: Index of the node

        Returns:
            Dictionary with metric name and value
        """
        # TODO: Consider plumbing {'GpuDeviceName': 'NVIDIA H100 80GB HBM3'}
        # Expected labels for GPU metrics
        expected_labels = ["GpuIndex", "GpuDeviceName"]
        for label in expected_labels:
            if label not in sample.labels:
                # This is probably a CPU node
                return {}

        metric_name = sample.name
        # Only return SKU if the metric is one of these which publish these metrics
        if (
            metric_name != "ray_node_gpus_utilization"
            and metric_name != "ray_node_gram_used"
        ):
            # Skip unexpected metrics
            return {}

        labels = sample.labels
        index = labels["GpuIndex"]
        value = labels["GpuDeviceName"]

        metric_name = f"node.{node_idx}.gpu.{index}.type"
        return {metric_name: value}

    def _collect_gpu_sku(self) -> dict[str, str]:
        """Collect GPU SKU from all Ray nodes.

        Note: This is an internal API and users are not expected to call this.

        Returns:
            Dictionary of SKU types on all Ray nodes
        """
        # TODO: We can re-use the same path for metrics because even though both utilization and memory metrics duplicate
        #       the GPU metadata information; since the metadata is the same for each node, we can overwrite it and expect them to
        #       be the same
        return self._collect(sku=True)

    def _collect_metrics(self) -> dict[str, Any]:
        """Collect GPU metrics from all Ray nodes.

        Returns:
            Dictionary of collected metrics
        """
        return self._collect(metrics=True)

    def _collect(self, metrics: bool = False, sku: bool = False) -> dict[str, Any]:
        """Collect GPU metrics from all Ray nodes.

        Returns:
            Dictionary of collected metrics
        """
        assert metrics ^ sku, (
            f"Must collect either metrics or sku, not both: {metrics=}, {sku=}"
        )
        parser_fn = self._parse_metric if metrics else self._parse_gpu_sku

        if not ray.is_initialized():
            print("Ray is not initialized. Cannot collect GPU metrics.")
            return {}

        try:
            nodes = ray.nodes()
            if not nodes:
                print("No Ray nodes found.")
                return {}

            # Use a dictionary to keep unique metric endpoints and maintain order
            unique_metric_addresses = {}
            for node in nodes:
                node_ip = node["NodeManagerAddress"]
                metrics_port = node.get("MetricsExportPort")
                if not metrics_port:
                    continue
                metrics_address = f"{node_ip}:{metrics_port}"
                unique_metric_addresses[metrics_address] = True

            # Process each node's metrics
            collected_metrics = {}
            for node_idx, metric_address in enumerate(unique_metric_addresses):
                metrics = self._fetch_and_parse_metrics(
                    node_idx, metric_address, parser_fn
                )
                collected_metrics.update(metrics)

            return collected_metrics

        except Exception as e:
            print(f"Error collecting GPU metrics: {e}")
            return {}

    def _fetch_and_parse_metrics(
        self, node_idx: int, metric_address: str, parser_fn: Callable
    ):
        """Fetch metrics from a node and parse GPU metrics.

        Args:
            node_idx: Index of the node
            metric_address: Address of the metrics endpoint

        Returns:
            Dictionary of GPU metrics
        """
        url = f"http://{metric_address}/metrics"

        try:
            response = requests.get(url, timeout=5.0)
            if response.status_code != 200:
                print(f"Error: Status code {response.status_code}")
                return {}

            metrics_text = response.text
            gpu_metrics = {}

            # Parse the Prometheus format
            for family in text_string_to_metric_families(metrics_text):
                for sample in family.samples:
                    metrics = parser_fn(sample, node_idx)
                    gpu_metrics.update(metrics)

            return gpu_metrics

        except Exception as e:
            print(f"Error fetching metrics from {metric_address}: {e}")
            return {}

    def flush(self) -> None:
        """Flush collected metrics to the parent logger."""
        with self.lock:
            if not self.metrics_buffer:
                return

            if self.parent_logger:
                # Log each set of metrics with its original step
                for entry in self.metrics_buffer:
                    step = entry["step"]
                    metrics = entry["metrics"]

                    # Add the step metric directly to metrics for use as step_metric
                    metrics[self.step_metric] = step

                    # Pass step_metric as the step_metric to use it as the step value in wandb
                    self.parent_logger.log_metrics(
                        metrics,
                        step=step,
                        prefix=self.metric_prefix,
                        step_metric=self.step_metric,
                    )

            # Clear buffer after logging
            self.metrics_buffer = []


class Logger(LoggerInterface):
    """Main logger class that delegates to multiple backend loggers."""

    def __init__(self, cfg: LoggerConfig):
        """Initialize the logger.

        Args:
            cfg: Config dict with the following keys:
                - wandb_enabled
                - tensorboard_enabled
                - wandb
                - tensorboard
                - monitor_gpus
                - gpu_collection_interval
                - gpu_flush_interval
        """
        self.loggers: list[LoggerInterface] = []
        self.wandb_logger = None

        self.base_log_dir = cfg["log_dir"]
        os.makedirs(self.base_log_dir, exist_ok=True)

        if cfg["wandb_enabled"]:
            wandb_log_dir = os.path.join(self.base_log_dir, "wandb")
            os.makedirs(wandb_log_dir, exist_ok=True)
            self.wandb_logger = WandbLogger(cfg["wandb"], log_dir=wandb_log_dir)
            self.loggers.append(self.wandb_logger)

        if cfg["tensorboard_enabled"]:
            tensorboard_log_dir = os.path.join(self.base_log_dir, "tensorboard")
            os.makedirs(tensorboard_log_dir, exist_ok=True)
            tensorboard_logger = TensorboardLogger(
                cfg["tensorboard"], log_dir=tensorboard_log_dir
            )
            self.loggers.append(tensorboard_logger)

        # Initialize GPU monitoring if requested
        self.gpu_monitor = None
        if cfg["monitor_gpus"]:
            metric_prefix = "ray"
            step_metric = f"{metric_prefix}/ray_step"
            if cfg["wandb_enabled"] and self.wandb_logger:
                self.wandb_logger.define_metric(
                    f"{metric_prefix}/*", step_metric=step_metric
                )

            self.gpu_monitor = RayGpuMonitorLogger(
                collection_interval=cfg["gpu_monitoring"]["collection_interval"],
                flush_interval=cfg["gpu_monitoring"]["flush_interval"],
                metric_prefix=metric_prefix,
                step_metric=step_metric,
                parent_logger=self,
            )
            self.gpu_monitor.start()

        if not self.loggers:
            print("No loggers initialized")

    def log_metrics(
        self,
        metrics: dict[str, Any],
        step: int,
        prefix: Optional[str] = "",
        step_metric: Optional[str] = None,
    ) -> None:
        """Log metrics to all enabled backends.

        Args:
            metrics: Dict of metrics to log
            step: Global step value
            prefix: Optional prefix for metric names
            step_metric: Optional name of a field in metrics to use as step instead
                         of the provided step value (currently only needed for wandb)
        """
        for logger in self.loggers:
            logger.log_metrics(metrics, step, prefix, step_metric)

    def log_hyperparams(self, params: Mapping[str, Any]) -> None:
        """Log hyperparameters to all enabled backends.

        Args:
            params: Dict of hyperparameters to log
        """
        for logger in self.loggers:
            logger.log_hyperparams(params)

    def log_batched_dict_as_jsonl(
        self, to_log: BatchedDataDict[Any] | dict[str, Any], filename: str
    ) -> None:
        """Log a list of dictionaries to a JSONL file.

        Args:
            to_log: BatchedDataDict to log
            filename: Filename to log to (within the log directory)
        """
        if not isinstance(to_log, BatchedDataDict):
            to_log = BatchedDataDict(to_log)

        # Create full path within log directory
        filepath = os.path.join(self.base_log_dir, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Write to JSONL file
        with open(filepath, "w") as f:
            for i, sample in enumerate(to_log.make_microbatch_iterator(1)):
                for key, value in sample.items():
                    if isinstance(value, torch.Tensor):
                        sample[key] = value.tolist()
                f.write(json.dumps({**sample, "idx": i}) + "\n")

        print(f"Logged data to {filepath}")

    def log_plot_token_mult_prob_error(
        self, data: dict[str, Any], step: int, name: str
    ) -> None:
        """Log a plot of log probability errors in samples.

        This function logs & plots the per-token log-probabilities and errors over the sequence
        for the sample with the highest multiplicative probability error in the batch.

        Args:
            log_data: Dictionary of log probability samples
            step: Global step value
            name: Name of the plot
        """
        # find the sample with the highest log probability error
        token_mask = data["token_mask"][:, 1:]
        sample_mask = data["sample_mask"]
        generation_logprobs = data["generation_logprobs"][:, 1:]
        prev_logprobs = data["prev_logprobs"][:, 1:]
        mask = token_mask * sample_mask.unsqueeze(-1)

        diff = (generation_logprobs - prev_logprobs).abs() * token_mask
        mask = token_mask * sample_mask.unsqueeze(-1)

        mult_prob_error = (torch.exp(diff) * mask).sum(dim=-1) / mask.sum(dim=-1)

        sample_idx = torch.argmax(mult_prob_error)
        sample_error = mult_prob_error[sample_idx]

        # plot the sample with the highest log probability error
        # offset by 1 token for next token prediction
        generation_start_idx, generation_end_idx = (
            data["prompt_lengths"][sample_idx] - 1,
            data["full_lengths"][sample_idx] - 1,
        )

        generation_logprob = generation_logprobs[
            sample_idx, generation_start_idx:generation_end_idx
        ]
        prev_logprob = (
            prev_logprobs[sample_idx, generation_start_idx:generation_end_idx]
            * mask[sample_idx, generation_start_idx:generation_end_idx]
        )
        diff_i = diff[sample_idx, generation_start_idx:generation_end_idx]

        # Find max absolute error token
        max_abs_error_idx = torch.argmax(diff_i).item()
        max_abs_error = diff_i[max_abs_error_idx].item()

        # Find max relative error token (ratio of probabilities)
        gen_prob = torch.exp(generation_logprob)
        prev_prob = torch.exp(prev_logprob)
        relative_error = torch.abs((gen_prob - prev_prob) / gen_prob)
        max_rel_error_idx = torch.argmax(relative_error).item()
        max_rel_error = relative_error[max_rel_error_idx].item()

        fig = plt.figure()
        step_idx = torch.arange(generation_start_idx, generation_end_idx)

        plt.plot(step_idx, generation_logprob, label="logprob (inference engine)")
        plt.plot(step_idx, prev_logprob, label="logprob (reference policy)")
        plt.plot(
            step_idx,
            diff_i,
            label=f"abs diff (token_mult_prob_error={sample_error:.2f})",
        )

        # Highlight max errors with points
        plt.plot(
            step_idx[max_abs_error_idx],
            diff_i[max_abs_error_idx],
            "ro",
            markersize=8,
            label=f"Max abs error: {max_abs_error:.4f}",
        )
        plt.plot(
            step_idx[max_rel_error_idx],
            diff_i[max_rel_error_idx],
            "bo",
            markersize=8,
            label=f"Max rel error (prob): {max_rel_error:.4f}",
        )

        plt.xlabel("Token Position (starting from prompt end)")
        plt.ylabel("Log Probability/Difference")
        plt.legend()
        plt.tight_layout()

        for logger in self.loggers:
            logger.log_plot(fig, step, name)

        plt.close(fig)

    def __del__(self) -> None:
        """Clean up resources when the logger is destroyed."""
        if self.gpu_monitor:
            self.gpu_monitor.stop()


def flatten_dict(d: Mapping[str, Any], sep: str = ".") -> dict[str, Any]:
    """Flatten a nested dictionary.

    Handles nested dictionaries and lists by creating keys with separators.
    For lists, the index is used as part of the key.

    Args:
        d: Dictionary to flatten
        sep: Separator to use between nested keys

    Returns:
        Flattened dictionary with compound keys

    Examples:
        ```{doctest}
        >>> from nemo_rl.utils.logger import flatten_dict
        >>> flatten_dict({"a": 1, "b": {"c": 2}})
        {'a': 1, 'b.c': 2}

        >>> flatten_dict({"a": [1, 2], "b": {"c": [3, 4]}})
        {'a.0': 1, 'a.1': 2, 'b.c.0': 3, 'b.c.1': 4}

        >>> flatten_dict({"a": [{"b": 1}, {"c": 2}]})
        {'a.0.b': 1, 'a.1.c': 2}
        ```
    """
    result: dict[str, Any] = {}

    def _flatten(d: Mapping[str, Any], parent_key: str = "") -> None:
        for key, value in d.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key

            if isinstance(value, dict):
                _flatten(value, new_key)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    list_key = f"{new_key}{sep}{i}"
                    if isinstance(item, dict):
                        _flatten(item, list_key)
                    else:
                        result[list_key] = item
            else:
                result[new_key] = value

    _flatten(d)
    return result


"""
Rich Console Logging Functionality
---------------------------------
Functions for setting up rich console logging and visualizing model outputs.
"""


def configure_rich_logging(
    level: str = "INFO", show_time: bool = True, show_path: bool = True
) -> None:
    """Configure rich logging for more visually appealing log output.

    Args:
        level: The logging level to use
        show_time: Whether to show timestamps in logs
        show_path: Whether to show file paths in logs
    """
    global _rich_logging_configured

    # Only configure if not already done
    if not _rich_logging_configured:
        # Configure logging with rich handler
        logging.basicConfig(
            level=level.upper(),
            format="%(message)s",
            datefmt="[%X]",
            handlers=[
                RichHandler(
                    rich_tracebacks=True,
                    show_time=show_time,
                    show_path=show_path,
                    markup=True,
                )
            ],
        )
        _rich_logging_configured = True


def print_message_log_samples(
    message_logs: list[LLMMessageLogType],
    rewards: list[float],
    num_samples: int = 5,
    step: int = 0,
) -> None:
    """Visualization for message logs and rewards using a more visual approach with emoji indicators and horizontal layout.

    Args:
        message_logs: List of message logs to sample from
        rewards: List of rewards corresponding to each message log
        num_samples: Number of samples to display (default: 5)
        step: Current training step (for display purposes)
    """
    # Make sure rich logging is configured before printing
    configure_rich_logging(level="INFO")

    if not message_logs or not rewards:
        print("⚠️  No message logs or rewards to display")
        return

    if num_samples <= 0:
        return

    if not message_logs:
        print("⚠️ No valid message logs to display")
        return

    assert len(message_logs) == len(rewards), (
        "Message logs and rewards must have the same length"
    )

    # Sample up to num_samples (or all if less)
    num_to_show = min(num_samples, len(message_logs))
    indices = list(range(len(message_logs)))

    # If we have more samples than needed, prioritize showing a mix of high and low rewards
    if len(indices) > num_to_show:
        # Sort indices by reward
        sorted_indices = sorted(indices, key=lambda i: rewards[i], reverse=True)
        # Take some from the top and some from the bottom
        half = num_to_show // 2
        indices = sorted_indices[:half] + sorted_indices[-half:]
        # If num_to_show is odd, add a middle sample
        if num_to_show % 2 == 1:
            middle_idx = len(sorted_indices) // 2
            indices.append(sorted_indices[middle_idx])
        indices = indices[:num_to_show]

    console = Console()

    # Header with step information
    console.rule(f"[bold bright_white on purple4]TRAINING STEP {step}")

    # Count the unique reward values
    all_rewards = rewards.copy()
    unique_rewards = sorted(set(all_rewards))
    reward_counts = {r: all_rewards.count(r) for r in unique_rewards}

    # Create a bar chart for discrete reward levels
    max_count = max(reward_counts.values()) if reward_counts else 1

    # Create discrete reward level visualization
    discrete_lines = []
    discrete_lines.append("[bold bright_white]Discrete Reward Levels:[/]")

    # Get emoji for each reward level
    def get_reward_emoji(reward: float) -> str:
        if reward >= 0.7:
            return "🔥"  # Excellent
        elif reward >= 0.3:
            return "✨"  # Good
        elif reward >= -0.5:
            return "🟠"  # Poor
        else:
            return "🔴"  # Very poor

    # Create a bar for each discrete reward level
    for reward in unique_rewards:
        count = reward_counts[reward]
        emoji = get_reward_emoji(reward)
        bar_len = int((count / max_count) * 20)

        # Choose different bar characters and colors
        if reward > 0.5:
            bar_char = "█"
            color = "bright_green"
        elif reward > 0:
            bar_char = "█"
            color = "green"
        elif reward == 0:
            bar_char = "▒"
            color = "bright_white"
        elif reward > -0.5:
            bar_char = "▓"
            color = "orange3"
        else:
            bar_char = "█"
            color = "red"

        bar = f"[{color}]{bar_char * bar_len}[/]"
        # Format with color based on reward value
        discrete_lines.append(
            f"{emoji} Reward [bold {color}]{reward:.4f}[/]: {bar} ({count} samples)"
        )

    # Create a summary panel
    avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0
    stats_text = (
        f"[bold]Batch Summary[/]\n"
        f"Total Samples: [bright_yellow]{len(all_rewards)}[/]\n"
        f"Avg Reward: [bright_blue]{avg_reward:.4f}[/]\n"
        f"Min: [orange3]{min(all_rewards):.4f}[/] | Max: [bright_green]{max(all_rewards):.4f}[/]\n\n"
        + "\n".join(discrete_lines)
    )

    stats_panel = Panel(
        stats_text,
        title="[bold purple4]Reward Statistics",
        border_style="purple4",
        box=ROUNDED,
    )

    # Display the stats panel
    console.print(stats_panel)

    # Display the samples with horizontal layout
    console.print("\n[bold bright_white]Sample Conversations[/]")

    # Helper function to safely render content that might have problematic markups
    def safe_render(content: str, role_color: str) -> str:
        # Fix common problematic patterns that might break Rich markup
        # Replace any standalone [/ without matching closing bracket
        content = content.replace("[/", "\\[/")
        # Replace any standalone [ that isn't followed by a valid tag with escaped version
        import re

        content = re.sub(r"\[(?![a-z_]+\s|/[a-z_]+\])", "\\[", content)
        return f"[{role_color}]{content}[/]"

    # Extract messages from a message log
    def extract_messages(message_log):
        def format_message(role, content):
            role = role.upper()
            if role == "SYSTEM":
                return f"[bold #8A2BE2]{role}:[/] {safe_render(content, '#8A2BE2')}"
            elif role == "USER":
                return f"[bold #4682B4]{role}:[/] {safe_render(content, '#4682B4')}"
            elif role == "ASSISTANT":
                return f"[bold #2E8B57]{role}:[/] {safe_render(content, '#2E8B57')}"
            else:
                return f"[bold]{role}:[/] {safe_render(content, 'bright_white')}"

        messages = []
        for msg in message_log:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                messages.append(format_message(msg["role"], msg["content"]))
        return messages

    for i, idx in enumerate(indices):
        message_log = message_logs[idx]
        reward = rewards[idx]

        # Extract messages from the sample
        message_parts = extract_messages(message_log)

        # Get reward emoji
        emoji = get_reward_emoji(reward)

        # Choose color based on reward
        if reward > 0.5:
            color = "bright_green"
        elif reward > 0:
            color = "green"
        elif reward == 0:
            color = "bright_white"
        elif reward > -0.5:
            color = "orange3"
        else:
            color = "red"

        content = "\n\n".join(message_parts)

        # If we have no content to display, show a placeholder
        if not content.strip():
            content = "[italic]No message content to display[/]"

        panel = Panel(
            content,
            title=f"[bold]{emoji} Sample {i + 1} | Reward: {reward:.4f}",
            border_style=color,
            box=ROUNDED,
        )

        console.print(panel)
        console.print("")  # Add some spacing

    console.rule("[bold bright_white on purple4]End of Samples")


def get_next_experiment_dir(base_log_dir: str) -> str:
    """Create a new experiment directory with an incremented ID.

    Args:
        base_log_dir (str): The base log directory path

    Returns:
        str: Path to the new experiment directory with incremented ID
    """
    # Check if the log directory already contains an experiment ID pattern (e.g., /exp_001/)
    pattern = re.compile(r"exp_(\d+)")
    next_exp_id = 1

    # Check for existing experiment directories
    existing_dirs = glob.glob(os.path.join(base_log_dir, "exp_*"))

    if existing_dirs:
        # Extract experiment IDs and find the maximum
        exp_ids = []
        for dir_path in existing_dirs:
            match = pattern.search(dir_path)
            if match:
                exp_ids.append(int(match.group(1)))

        if exp_ids:
            # Increment the highest experiment ID
            next_exp_id = max(exp_ids) + 1

    # Format the new log directory with the incremented experiment ID
    new_log_dir = os.path.join(base_log_dir, f"exp_{next_exp_id:03d}")

    # Create the new log directory
    os.makedirs(new_log_dir, exist_ok=True)

    return new_log_dir
