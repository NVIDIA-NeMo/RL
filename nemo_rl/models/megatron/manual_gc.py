# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Megatron-style manual garbage collection for Megatron training workers.

An automatic full Python GC lands on whichever rank crosses its threshold, and
every other rank then waits for it in the next collective. With
``megatron_cfg.manual_gc`` set, a worker disables automatic GC and every rank
collects together every ``megatron_cfg.manual_gc_interval`` optimizer steps.
"""

import gc
import os
from typing import Any, Optional


class ManualGCMixin:
    """Manual GC for Megatron workers that train across data-parallel ranks.

    The host class must set ``self.cfg`` (with a ``"megatron_cfg"`` entry) and make
    three calls:

    - ``self._manual_gc_interval_from_cfg()`` early in ``__init__``, to fail on a
      bad config before the expensive setup;
    - ``self._maybe_enable_manual_gc()`` as the last statement of ``__init__``,
      so model, optimizer and checkpoint setup run with automatic GC on;
    - ``self._maybe_run_manual_gc()`` once after every optimizer step.
    """

    cfg: Any
    # megatron_cfg.manual_gc_interval when manual GC is on, else None.
    _manual_gc_interval: Optional[int] = None
    _manual_gc_steps: int = 0

    def _manual_gc_interval_from_cfg(self) -> Optional[int]:
        """Return megatron_cfg.manual_gc_interval if manual GC is on, else None.

        Raises:
            ValueError: if manual_gc is True and manual_gc_interval is missing or not
                a positive number of optimizer steps. NeMo-RL has no equivalent of
                Megatron's ``manual_gc_eval`` collections, so a non-positive interval
                would disable automatic GC with no collection ever running.
        """
        megatron_cfg = self.cfg["megatron_cfg"]
        if not megatron_cfg.get("manual_gc"):
            return None
        interval = megatron_cfg.get("manual_gc_interval")
        if not isinstance(interval, int) or isinstance(interval, bool) or interval <= 0:
            raise ValueError(
                "megatron_cfg.manual_gc is True, so megatron_cfg.manual_gc_interval "
                f"must be a positive number of optimizer steps, got {interval!r}. "
                "Otherwise automatic GC is disabled with no collection ever running."
            )
        return interval

    def _maybe_enable_manual_gc(self) -> None:
        """Disable automatic Python GC when megatron_cfg.manual_gc is set.

        Mirrors Megatron's ``manual_gc``: collections then run on every rank at the
        same point (see ``_maybe_run_manual_gc``) instead of stalling the other
        ranks one at a time. Called at the end of ``__init__``, so setup runs with
        automatic GC on and the collection here frees its garbage. Forked children
        re-enable GC so subprocesses such as data workers keep the default behavior.
        """
        interval = self._manual_gc_interval_from_cfg()
        if interval is None:
            return
        gc.collect()
        gc.disable()
        os.register_at_fork(after_in_child=gc.enable)
        self._manual_gc_interval = interval

    def _maybe_run_manual_gc(self) -> None:
        """Collect every manual_gc_interval optimizer steps; every rank runs it in lockstep."""
        if not self._manual_gc_interval:
            return
        self._manual_gc_steps += 1
        if self._manual_gc_steps % self._manual_gc_interval == 0:
            gc.collect()
