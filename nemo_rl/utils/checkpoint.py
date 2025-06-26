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
"""Checkpoint management utilities for the rl algorithm loop.

It handles logic at the algorithm level. Each RL Actor is expected to have its
own checkpoint saving function (called by the algorithm loop).
"""

import glob
import json
import os
import shutil
from pathlib import Path
from typing import Any, Optional, TypedDict, Union

import numpy as np
import torch
import yaml

PathLike = Union[str, "os.PathLike[Any]"]


class CheckpointingConfig(TypedDict):
    """Configuration for checkpoint management.

    Attributes:
    enabled (bool): Whether checkpointing is enabled.
    checkpoint_dir (PathLike): Directory where checkpoints will be saved.
    metric_name (str): Name of the metric to use for determining best checkpoints.
    higher_is_better (bool): Whether higher values of the metric indicate better performance.
    keep_top_k (Optional[int]): Number of best checkpoints to keep. If None, all checkpoints are kept.
    """

    enabled: bool
    checkpoint_dir: PathLike
    metric_name: str
    higher_is_better: bool
    save_period: int
    keep_top_k: Optional[int]


class CheckpointManager:
    """Manages model checkpoints during training.

    This class handles creating checkpoint dirs, saving training info, and
    configurations. It also provides utilities for keeping just the top-k checkpoints.
    The checkpointing structure looks like this:
    ```
    checkpoint_dir/
        step_0/
            training_info.json
            config.yaml
            policy.py (up to the algorithm loop to save here)
            policy_optimizer.py (up to the algorithm loop to save here)
            ...
        step_1/
            ...
    ```

    Attributes: Derived from the CheckpointingConfig.
    """

    def __init__(self, config: CheckpointingConfig):
        """Initialize the checkpoint manager.

        Args:
            config (CheckpointingConfig)
        """
        self.checkpoint_dir = Path(config["checkpoint_dir"])
        self.metric_name = config["metric_name"]
        self.higher_is_better = config["higher_is_better"]
        self.keep_top_k = config["keep_top_k"]

    def init_tmp_checkpoint(
        self,
        step: int,
        training_info: dict[str, Any],
        run_config: Optional[dict[str, Any]] = None,
    ) -> PathLike:
        """Initialize a temporary checkpoint directory.

        Creates a temporary directory for a new checkpoint and saves training info
        and configuration. The directory is named 'tmp_step_{step}' and will be renamed
        to 'step_{step}' when the checkpoint is completed.
        We do it this way to allow the algorithm loop to save any files it wants to save
        in a safe, temporary directory.

        Args:
            step (int): The training step number.
            training_info (dict[str, Any]): Dictionary containing training metrics and info.
            run_config (Optional[dict[str, Any]]): Optional configuration for the training run.

        Returns:
            PathLike: Path to the temporary checkpoint directory.
        """
        # create new step_{step} directory
        save_dir = self.checkpoint_dir / f"tmp_step_{step}"
        save_dir.mkdir(parents=True, exist_ok=True)

        # save training info
        with open(save_dir / "training_info.json", "w") as f:
            # make any numpy items serializable
            for k, v in training_info.items():
                if isinstance(v, torch.Tensor) or isinstance(v, np.ndarray):
                    training_info[k] = v.item()
            json.dump(training_info, f)

        # save config
        if run_config is not None:
            with open(save_dir / "config.yaml", "w") as f:
                yaml.safe_dump(run_config, f)

        return Path(os.path.abspath(save_dir))

    def finalize_checkpoint(self, checkpoint_path: PathLike) -> None:
        """Complete a checkpoint by moving it from temporary to permanent location.

        If a checkpoint at the target location already exists (i.e when resuming training),
        we override the old one.
        Also triggers cleanup of old checkpoints based on the keep_top_k setting.

        Args:
            checkpoint_path (PathLike): Path to the temporary checkpoint directory.
        """
        # rename tmp_step_{step} to step_{step}
        checkpoint_path = Path(checkpoint_path)
        to_checkpoint_path = (
            checkpoint_path.parent / f"step_{checkpoint_path.name.split('_')[2]}"
        )
        if to_checkpoint_path.exists():
            # if step_{step} exists, rename it to old_step_{step}, move tmp_step_{step} to step_{step}, then delete
            # we do this trickery to have a 'pseudo-atomic' checkpoint save
            old_checkpoint_path = (
                checkpoint_path.parent
                / f"old_step_{checkpoint_path.name.split('_')[2]}"
            )
            os.rename(to_checkpoint_path, old_checkpoint_path)
            os.rename(checkpoint_path, to_checkpoint_path)
            # delete old_step_{step}
            if old_checkpoint_path.exists():
                shutil.rmtree(old_checkpoint_path)
        else:
            os.rename(checkpoint_path, to_checkpoint_path)
        self.remove_old_checkpoints()

    def remove_old_checkpoints(self, exclude_latest: bool = True) -> None:
        """Remove checkpoints that are not in the top-k or latest based on the metric.

        If keep_top_k is set, this method removes all checkpoints except the top-k
        best ones based on the specified metric. The best checkpoints are determined
        by the metric value and the higher_is_better setting. When multiple checkpoints
        have the same metric value, more recent checkpoints (higher step numbers) are preferred.

        Args:
            exclude_latest (bool): Whether to exclude the latest checkpoint from deletion. (may result in K+1 checkpoints)
        """
        if self.keep_top_k is None:
            return
        checkpoint_history = _load_checkpoint_history(self.checkpoint_dir)
        latest_step = (
            max([step for step, _, _ in checkpoint_history])
            if checkpoint_history
            else None
        )
        # sort by metric value first, then by step number (for equal metrics, prefer more recent)
        if self.higher_is_better:
            # For higher_is_better=True: higher metric values first, then higher step numbers
            checkpoint_history.sort(
                key=lambda x: (x[2][self.metric_name], x[0]), reverse=True
            )
        else:
            # For higher_is_better=False: lower metric values first, then higher step numbers for equal values
            checkpoint_history.sort(key=lambda x: (x[2][self.metric_name], -x[0]))

        # remove checkpoints that are not in the top-k
        for checkpoint in checkpoint_history[self.keep_top_k :]:
            if exclude_latest and checkpoint[0] == latest_step:
                continue
            print(
                f"Removing checkpoint {checkpoint[1]} due to being outside top-{self.keep_top_k}, metric: {checkpoint[2][self.metric_name]}"
            )
            shutil.rmtree(checkpoint[1])

    def get_best_checkpoint_path(self) -> Optional[str]:
        """Get the path to the best checkpoint based on the metric.

        Returns the path to the checkpoint with the best metric value. If no checkpoints
        exist, returns None. If the metric isn't found, we warn and return the latest checkpoint.

        Returns:
            Optional[str]: Path to the best checkpoint, or None if no valid checkpoints exist.
        """
        checkpoint_history = _load_checkpoint_history(self.checkpoint_dir)
        if len(checkpoint_history) == 0:
            return None
        # sort by metric value
        if self.metric_name not in checkpoint_history[0][2]:
            print(
                f"WARNING:Metric {self.metric_name} not found in checkpoint history. Returning last"
            )
            return self.get_latest_checkpoint_path()

        checkpoint_history.sort(
            key=lambda x: x[2][self.metric_name], reverse=self.higher_is_better
        )
        return str(checkpoint_history[0][1])

    def get_latest_checkpoint_path(self) -> Optional[str]:
        """Get the path to the latest checkpoint.

        Returns the path to the checkpoint with the highest step number.

        Returns:
            Optional[str]: Path to the latest checkpoint, or None if no checkpoints exist.
        """
        # find checkpoint directory with highest step number
        step_dirs = glob.glob(str(self.checkpoint_dir / "step_*"))
        step_dirs.sort(key=lambda x: int(Path(x).name.split("_")[1]))
        if len(step_dirs) == 0:
            return None
        return str(step_dirs[-1])

    def load_training_info(
        self, checkpoint_path: Optional[PathLike] = None
    ) -> Optional[dict[str, Any]]:
        """Load the training info from a checkpoint.

        Args:
            checkpoint_path (Optional[PathLike]): Path to the checkpoint. If None,
                returns None.

        Returns:
            Optional[dict[str, Any]]: Dictionary containing the training info, or None if
                checkpoint_path is None.
        """
        if checkpoint_path is None:
            return None
        with open(Path(checkpoint_path) / "training_info.json", "r") as f:
            return json.load(f)


def _load_checkpoint_history(
    checkpoint_dir: Path,
) -> list[tuple[int, PathLike, dict[str, Any]]]:
    """Load the history of checkpoints and their metrics.

    Args:
        checkpoint_dir (Path): Directory containing the checkpoints.

    Returns:
        list[tuple[int, PathLike, dict[str, Any]]]: List of tuples containing
            (step_number, checkpoint_path, info) for each checkpoint.
    """
    checkpoint_history: list[tuple[int, PathLike, dict[str, Any]]] = []

    # Find all step directories
    step_dirs = glob.glob(str(checkpoint_dir / "step_*"))

    for step_dir in step_dirs:
        info_file = Path(step_dir) / "training_info.json"
        if info_file.exists():
            with open(info_file) as f:
                info: dict[str, Any] = json.load(f)
                step = int(Path(step_dir).name.split("_")[1])
                checkpoint_history.append((step, step_dir, info))

    return checkpoint_history
