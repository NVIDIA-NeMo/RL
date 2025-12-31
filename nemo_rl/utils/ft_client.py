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

"""NVIDIA Resiliency Extension (NVRX) Fault Tolerance Client wrapper."""

from contextlib import contextmanager
import logging
import os
import tempfile
import uuid
from typing import Any, Optional

logger = logging.getLogger(__name__)

import torch
import torch.distributed as dist

try:
    import nvidia_resiliency_ext.fault_tolerance as ft

    NVRX_AVAILABLE = True
except ImportError:
    ft = None
    NVRX_AVAILABLE = False


class FTClient:
    """Wrapper for NVRX RankMonitorClient."""

    def __init__(self, use_sections: bool = False):
        self._client: Optional[Any] = None
        self._enabled = False
        self._use_sections = use_sections

    @property
    def enabled(self) -> bool:
        """Return whether FT monitoring is currently enabled."""
        return self._enabled

    @property
    def use_sections(self) -> bool:
        """Return whether sections API is being used."""
        return self._use_sections

    def init_workload_monitoring(self, use_sections: bool = False) -> None:
        """Initialize FT monitoring.

        Call at start of training. This will connect to the FT launcher's
        rank monitor server if running under ft_launcher.

        Args:
            use_sections: If True, use the sections API for fine-grained
                timeout control. If False, use simple heartbeat API.
        """
        if not NVRX_AVAILABLE:
            logger.info("FT monitoring skipped - nvidia-resiliency-ext not installed")
            return

        if not dist.is_initialized():
            temp_dir = tempfile.gettempdir()
            init_file = os.path.join(temp_dir, f"ft_pg_init_{uuid.uuid4().hex[:8]}")
            dist.init_process_group(
                backend="gloo",
                rank=0,
                world_size=1,
                init_method=f"file://{init_file}",
            )

        self._use_sections = use_sections
        self._client = ft.RankMonitorClient()
        self._client.init_workload_monitoring()
        self._enabled = True
        logger.info("FT monitoring initialized successfully")

    def shutdown_workload_monitoring(self) -> None:
        """Shutdown FT monitoring.

        Call at end of training to cleanly disconnect from the rank monitor.
        """
        if self._client:
            self._client.shutdown_workload_monitoring()
            self._client = None
            self._enabled = False
            logger.info("FT monitoring shutdown complete")

    def send_heartbeat(self) -> None:
        """Send heartbeat signal.

        Call periodically in training loop to indicate the rank is alive.
        If using sections API, heartbeats are sent automatically when
        entering/exiting sections.
        """
        if self._client:
            self._client.send_heartbeat()

    def start_section(self, name: str) -> None:
        """Start a named section (for Sections API).

        Args:
            name: Name of the section (e.g., "generation", "training").
        """
        if self._client and self._use_sections:
            self._client.start_section(name)

    def end_section(self, name: str) -> None:
        """End a named section (for Sections API).

        Args:
            name: Name of the section to end.
        """
        if self._client and self._use_sections:
            self._client.end_section(name)

    @contextmanager
    def section(self, name: str):
        """Context manager for sections.

        Automatically starts and ends a named section. If sections API
        is not enabled, this is a no-op.

        Args:
            name: Name of the section.

        Yields:
            None
        """
        self.start_section(name)
        try:
            yield
        finally:
            self.end_section(name)

    def calculate_and_set_timeouts(self) -> None:
        """Auto-calculate timeouts from observed intervals.

        Call after completing one full epoch/iteration to let the FT
        system calculate appropriate timeouts based on observed timing.
        """
        if self._client:
            if self._use_sections:
                self._client.calculate_and_set_section_timeouts()
            self._client.calculate_and_set_hb_timeouts()
            logging.info("FT timeouts calculated and set successfully")

    def state_dict(self) -> dict:
        """Get state dict for checkpointing.

        Returns:
            Dictionary containing FT client state that should be saved
            with checkpoints to enable proper timeout restoration on restart.
        """
        if self._client:
            return self._client.state_dict()
        return {}

    def load_state_dict(self, state: dict) -> None:
        """Load state dict from checkpoint.

        Args:
            state: State dictionary previously returned by state_dict().
        """
        if self._client and state:
            self._client.load_state_dict(state)


# Global singleton instance
_ft_client: Optional[FTClient] = None


def get_ft_client() -> FTClient:
    """Get the global FT client instance.

    Returns a singleton FTClient instance. This ensures all parts of the
    training code use the same FT client.

    Returns:
        The global FTClient instance.
    """
    global _ft_client
    if _ft_client is None:
        _ft_client = FTClient(use_sections=True)
    return _ft_client


def is_ft_available() -> bool:
    """Check if NVRX fault tolerance is available.

    Returns:
        True if nvidia-resiliency-ext is installed, False otherwise.
    """
    return NVRX_AVAILABLE
