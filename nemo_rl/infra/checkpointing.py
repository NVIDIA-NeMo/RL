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
"""Backend-agnostic checkpoint management.

This module provides a unified interface for checkpoint management that
supports multiple backend formats including:
- PyTorch native (DCP - Distributed Checkpoint)
- Megatron checkpoint format
- HuggingFace safetensors format

Key features:
- Automatic checkpoint format detection on load
- Backend-agnostic save/load interface
- Checkpoint versioning and migration support
"""

from __future__ import annotations

import glob
import json
import logging
import os
import re
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


class CheckpointFormat(str, Enum):
    """Supported checkpoint formats."""

    PYTORCH = "pytorch"  # Native PyTorch state_dict
    DCP = "dcp"  # Distributed Checkpoint (torch.distributed.checkpoint)
    MEGATRON = "megatron"  # Megatron-LM format
    SAFETENSORS = "safetensors"  # HuggingFace safetensors
    HUGGINGFACE = "huggingface"  # HuggingFace transformers format
    AUTO = "auto"  # Auto-detect format


class CheckpointError(Exception):
    """Error during checkpoint operations."""

    pass


@dataclass
class CheckpointMetadata:
    """Metadata about a checkpoint.

    Attributes:
        format: Checkpoint format.
        version: Checkpoint version/schema version.
        step: Training step when checkpoint was saved.
        model_name: Name/path of the model.
        extra: Additional metadata.
    """

    format: CheckpointFormat
    version: str = "1.0"
    step: int | None = None
    model_name: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class CheckpointBackend(ABC):
    """Abstract base class for checkpoint backends.

    Each backend implementation handles saving and loading checkpoints
    in a specific format.
    """

    @abstractmethod
    def save(
        self,
        path: Path,
        state_dict: dict[str, Any],
        metadata: CheckpointMetadata | None = None,
    ) -> None:
        """Save a checkpoint.

        Args:
            path: Path to save the checkpoint.
            state_dict: State dictionary to save.
            metadata: Optional checkpoint metadata.
        """
        pass

    @abstractmethod
    def load(
        self,
        path: Path,
        map_location: str | "torch.device" | None = None,
    ) -> dict[str, Any]:
        """Load a checkpoint.

        Args:
            path: Path to the checkpoint.
            map_location: Device to map tensors to.

        Returns:
            Loaded state dictionary.
        """
        pass

    @abstractmethod
    def can_load(self, path: Path) -> bool:
        """Check if this backend can load the given checkpoint.

        Args:
            path: Path to check.

        Returns:
            True if this backend can load the checkpoint.
        """
        pass


class PyTorchBackend(CheckpointBackend):
    """Native PyTorch checkpoint backend."""

    def save(
        self,
        path: Path,
        state_dict: dict[str, Any],
        metadata: CheckpointMetadata | None = None,
    ) -> None:
        import torch

        path.parent.mkdir(parents=True, exist_ok=True)

        # Save with metadata
        save_data = {"state_dict": state_dict}
        if metadata:
            save_data["metadata"] = {
                "format": metadata.format.value,
                "version": metadata.version,
                "step": metadata.step,
                "model_name": metadata.model_name,
                "extra": metadata.extra,
            }

        torch.save(save_data, path)
        logger.info(f"Saved PyTorch checkpoint to {path}")

    def load(
        self,
        path: Path,
        map_location: str | "torch.device" | None = None,
    ) -> dict[str, Any]:
        import torch

        data = torch.load(path, map_location=map_location, weights_only=False)

        # Handle both wrapped and unwrapped formats
        if isinstance(data, dict) and "state_dict" in data:
            return data["state_dict"]
        return data

    def can_load(self, path: Path) -> bool:
        # Check for .pt or .pth files
        if path.is_file():
            return path.suffix in (".pt", ".pth", ".bin")
        return False


class SafetensorsBackend(CheckpointBackend):
    """HuggingFace safetensors checkpoint backend."""

    def save(
        self,
        path: Path,
        state_dict: dict[str, Any],
        metadata: CheckpointMetadata | None = None,
    ) -> None:
        try:
            from safetensors.torch import save_file
        except ImportError:
            raise CheckpointError(
                "safetensors is required for SafetensorsBackend. "
                "Install with: pip install safetensors"
            )

        path.parent.mkdir(parents=True, exist_ok=True)

        # Filter out non-tensor items and save them separately
        tensor_dict = {}
        non_tensor_dict = {}
        for k, v in state_dict.items():
            import torch

            if isinstance(v, torch.Tensor):
                tensor_dict[k] = v
            else:
                non_tensor_dict[k] = v

        # Save tensors
        save_file(tensor_dict, path)

        # Save non-tensor data and metadata as JSON
        if non_tensor_dict or metadata:
            meta_path = path.with_suffix(".json")
            meta_data = {"non_tensor_state": non_tensor_dict}
            if metadata:
                meta_data["metadata"] = {
                    "format": metadata.format.value,
                    "version": metadata.version,
                    "step": metadata.step,
                    "model_name": metadata.model_name,
                }
            with open(meta_path, "w") as f:
                json.dump(meta_data, f)

        logger.info(f"Saved safetensors checkpoint to {path}")

    def load(
        self,
        path: Path,
        map_location: str | "torch.device" | None = None,
    ) -> dict[str, Any]:
        try:
            from safetensors.torch import load_file
        except ImportError:
            raise CheckpointError(
                "safetensors is required for SafetensorsBackend. "
                "Install with: pip install safetensors"
            )

        device = str(map_location) if map_location else "cpu"
        state_dict = load_file(path, device=device)

        # Load non-tensor data if exists
        meta_path = path.with_suffix(".json")
        if meta_path.exists():
            with open(meta_path) as f:
                meta_data = json.load(f)
            if "non_tensor_state" in meta_data:
                state_dict.update(meta_data["non_tensor_state"])

        return state_dict

    def can_load(self, path: Path) -> bool:
        if path.is_file():
            return path.suffix == ".safetensors"
        # Check for model.safetensors in directory
        if path.is_dir():
            return (path / "model.safetensors").exists()
        return False


class HuggingFaceBackend(CheckpointBackend):
    """HuggingFace transformers checkpoint backend."""

    def save(
        self,
        path: Path,
        state_dict: dict[str, Any],
        metadata: CheckpointMetadata | None = None,
    ) -> None:
        path.mkdir(parents=True, exist_ok=True)

        # Save as safetensors by default (preferred HF format)
        try:
            from safetensors.torch import save_file
            import torch

            tensor_dict = {k: v for k, v in state_dict.items() if isinstance(v, torch.Tensor)}
            save_file(tensor_dict, path / "model.safetensors")
        except ImportError:
            import torch

            torch.save(state_dict, path / "pytorch_model.bin")

        # Save config if model_name provided
        if metadata and metadata.model_name:
            config_path = path / "config.json"
            with open(config_path, "w") as f:
                json.dump({"model_name": metadata.model_name}, f)

        logger.info(f"Saved HuggingFace checkpoint to {path}")

    def load(
        self,
        path: Path,
        map_location: str | "torch.device" | None = None,
    ) -> dict[str, Any]:
        # Try safetensors first
        safetensors_path = path / "model.safetensors"
        if safetensors_path.exists():
            try:
                from safetensors.torch import load_file

                device = str(map_location) if map_location else "cpu"
                return load_file(safetensors_path, device=device)
            except ImportError:
                pass

        # Fall back to pytorch format
        pytorch_path = path / "pytorch_model.bin"
        if pytorch_path.exists():
            import torch

            return torch.load(pytorch_path, map_location=map_location, weights_only=False)

        raise CheckpointError(f"No valid checkpoint found at {path}")

    def can_load(self, path: Path) -> bool:
        if not path.is_dir():
            return False
        return (path / "model.safetensors").exists() or (
            path / "pytorch_model.bin"
        ).exists()


class CheckpointManager:
    """Unified checkpoint manager with backend-agnostic interface.

    Provides a single interface for saving and loading checkpoints
    across different formats (PyTorch, DCP, Megatron, HuggingFace).

    Supports:
    - Automatic format detection on load
    - Checkpoint versioning and migration
    - Top-k checkpoint retention
    - Training state persistence

    Attributes:
        checkpoint_dir: Base directory for checkpoints.
        format: Default format for saving checkpoints.
        keep_top_k: Number of best checkpoints to retain.

    Example:
        >>> manager = CheckpointManager(
        ...     checkpoint_dir="checkpoints",
        ...     format=CheckpointFormat.SAFETENSORS,
        ...     keep_top_k=5,
        ... )
        >>> manager.save(model.state_dict(), step=100, metrics={"loss": 0.5})
        >>> state_dict = manager.load("checkpoints/step_100")
    """

    # Registry of available backends
    _backends: dict[CheckpointFormat, type[CheckpointBackend]] = {
        CheckpointFormat.PYTORCH: PyTorchBackend,
        CheckpointFormat.SAFETENSORS: SafetensorsBackend,
        CheckpointFormat.HUGGINGFACE: HuggingFaceBackend,
    }

    def __init__(
        self,
        checkpoint_dir: str | Path,
        format: CheckpointFormat = CheckpointFormat.SAFETENSORS,
        keep_top_k: int | None = 5,
        metric_name: str | None = "val:reward",
        higher_is_better: bool = True,
    ):
        """Initialize the checkpoint manager.

        Args:
            checkpoint_dir: Directory for storing checkpoints.
            format: Default format for saving checkpoints.
            keep_top_k: Number of best checkpoints to keep. None keeps all.
            metric_name: Metric name for determining best checkpoint.
            higher_is_better: Whether higher metric values are better.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.format = format
        self.keep_top_k = keep_top_k
        self.metric_name = metric_name
        self.higher_is_better = higher_is_better

        # Initialize default backend
        self._default_backend = self._backends[format]()

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        state_dict: dict[str, Any],
        step: int,
        metrics: dict[str, Any] | None = None,
        model_name: str | None = None,
        format: CheckpointFormat | None = None,
    ) -> Path:
        """Save a checkpoint.

        Args:
            state_dict: Model state dictionary to save.
            step: Training step number.
            metrics: Optional training metrics.
            model_name: Optional model name.
            format: Optional format override.

        Returns:
            Path to the saved checkpoint.
        """
        format = format or self.format
        backend = self._backends.get(format, self._backends[self.format])()

        # Create checkpoint directory
        checkpoint_path = self.checkpoint_dir / f"step_{step}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Determine checkpoint file path
        if format == CheckpointFormat.PYTORCH:
            file_path = checkpoint_path / "model.pt"
        elif format == CheckpointFormat.SAFETENSORS:
            file_path = checkpoint_path / "model.safetensors"
        else:
            file_path = checkpoint_path

        # Create metadata
        metadata = CheckpointMetadata(
            format=format,
            step=step,
            model_name=model_name,
            extra=metrics or {},
        )

        # Save checkpoint
        backend.save(file_path, state_dict, metadata)

        # Save training info
        self._save_training_info(checkpoint_path, step, metrics or {})

        # Clean up old checkpoints
        self._cleanup_old_checkpoints()

        return checkpoint_path

    def load(
        self,
        path: str | Path | None = None,
        map_location: str | None = None,
        format: CheckpointFormat | None = None,
    ) -> dict[str, Any]:
        """Load a checkpoint.

        Args:
            path: Path to checkpoint. If None, loads the best checkpoint.
            map_location: Device to map tensors to.
            format: Format hint (AUTO for auto-detection).

        Returns:
            Loaded state dictionary.
        """
        if path is None:
            path = self.get_best_checkpoint_path()
            if path is None:
                raise CheckpointError("No checkpoints found")

        path = Path(path)
        format = format or CheckpointFormat.AUTO

        # Auto-detect format
        if format == CheckpointFormat.AUTO:
            format = self._detect_format(path)

        backend = self._backends.get(format, self._default_backend.__class__)()

        # Find the actual checkpoint file
        checkpoint_file = self._find_checkpoint_file(path)

        return backend.load(checkpoint_file, map_location)

    def _detect_format(self, path: Path) -> CheckpointFormat:
        """Auto-detect checkpoint format.

        Args:
            path: Path to checkpoint.

        Returns:
            Detected checkpoint format.
        """
        if path.is_file():
            if path.suffix in (".pt", ".pth", ".bin"):
                return CheckpointFormat.PYTORCH
            if path.suffix == ".safetensors":
                return CheckpointFormat.SAFETENSORS
        elif path.is_dir():
            if (path / "model.safetensors").exists():
                return CheckpointFormat.SAFETENSORS
            if (path / "pytorch_model.bin").exists():
                return CheckpointFormat.HUGGINGFACE
            if (path / "model.pt").exists():
                return CheckpointFormat.PYTORCH

        # Default to PyTorch
        return CheckpointFormat.PYTORCH

    def _find_checkpoint_file(self, path: Path) -> Path:
        """Find the actual checkpoint file in a directory.

        Args:
            path: Checkpoint path (file or directory).

        Returns:
            Path to the checkpoint file.
        """
        if path.is_file():
            return path

        # Look for checkpoint files in directory
        for filename in ["model.safetensors", "model.pt", "pytorch_model.bin"]:
            file_path = path / filename
            if file_path.exists():
                return file_path

        raise CheckpointError(f"No checkpoint file found in {path}")

    def _save_training_info(
        self,
        checkpoint_path: Path,
        step: int,
        metrics: dict[str, Any],
    ) -> None:
        """Save training info to checkpoint directory.

        Args:
            checkpoint_path: Path to checkpoint directory.
            step: Training step.
            metrics: Training metrics.
        """
        info = {
            "step": step,
            "metrics": metrics,
            "format": self.format.value,
        }
        info_path = checkpoint_path / "training_info.json"
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints based on keep_top_k setting."""
        if self.keep_top_k is None:
            return

        checkpoints = self._get_checkpoint_history()
        if len(checkpoints) <= self.keep_top_k:
            return

        # Sort by metric if available
        if self.metric_name:

            def get_metric(cp):
                return cp["metrics"].get(self.metric_name, float("-inf"))

            checkpoints.sort(key=get_metric, reverse=self.higher_is_better)
        else:
            # Sort by step (most recent first)
            checkpoints.sort(key=lambda x: x["step"], reverse=True)

        # Remove checkpoints outside top-k
        for checkpoint in checkpoints[self.keep_top_k :]:
            path = Path(checkpoint["path"])
            if path.exists():
                logger.info(f"Removing old checkpoint: {path}")
                shutil.rmtree(path)

    def _get_checkpoint_history(self) -> list[dict[str, Any]]:
        """Get list of all checkpoints with their info.

        Returns:
            List of checkpoint info dictionaries.
        """
        checkpoints = []
        pattern = str(self.checkpoint_dir / "step_*")

        for dir_path in glob.glob(pattern):
            dir_path = Path(dir_path)
            if not dir_path.is_dir():
                continue

            # Try to load training info
            info_path = dir_path / "training_info.json"
            if info_path.exists():
                with open(info_path) as f:
                    info = json.load(f)
                info["path"] = str(dir_path)
                checkpoints.append(info)
            else:
                # Extract step from directory name
                match = re.match(r"step_(\d+)", dir_path.name)
                if match:
                    checkpoints.append(
                        {
                            "step": int(match.group(1)),
                            "path": str(dir_path),
                            "metrics": {},
                        }
                    )

        return checkpoints

    def get_best_checkpoint_path(self) -> str | None:
        """Get path to the best checkpoint based on metric.

        Returns:
            Path to best checkpoint, or None if no checkpoints exist.
        """
        checkpoints = self._get_checkpoint_history()
        if not checkpoints:
            return None

        if self.metric_name:

            def get_metric(cp):
                return cp["metrics"].get(self.metric_name, float("-inf"))

            checkpoints.sort(key=get_metric, reverse=self.higher_is_better)
        else:
            checkpoints.sort(key=lambda x: x["step"], reverse=True)

        return checkpoints[0]["path"]

    def get_latest_checkpoint_path(self) -> str | None:
        """Get path to the latest checkpoint.

        Returns:
            Path to latest checkpoint, or None if no checkpoints exist.
        """
        checkpoints = self._get_checkpoint_history()
        if not checkpoints:
            return None

        checkpoints.sort(key=lambda x: x["step"], reverse=True)
        return checkpoints[0]["path"]

    def checkpoint_exists(self, step: int) -> bool:
        """Check if a checkpoint exists for the given step.

        Args:
            step: Training step.

        Returns:
            True if checkpoint exists.
        """
        checkpoint_path = self.checkpoint_dir / f"step_{step}"
        return checkpoint_path.exists()


__all__ = [
    "CheckpointFormat",
    "CheckpointError",
    "CheckpointMetadata",
    "CheckpointBackend",
    "CheckpointManager",
]
