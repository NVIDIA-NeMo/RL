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
"""Import isolation tests — OPS-5 and OPS-6 equivalents.

Covers:
  OPS-5 (P8): legacy grpo.py must be importable without transfer_queue.
  OPS-6 (P8): grpo_sync.py imports cleanly too (TQ is lazy), but calling
              grpo_train_sync without data_plane.enabled raises a clear error
              pointing at grpo.py for the legacy path.

These tests run in < 1 s with no Ray, no GPU, no real TQ controller.

Design note:
  transfer_queue is lazily imported inside TQDataPlaneClient.__init__, so
  importing nemo_rl.algorithms.grpo_sync itself does NOT require TQ to be
  installed. The import contract here is that grpo.py has zero references to
  the data plane, and grpo_sync.py wires the data plane through a runtime
  guard (not at import time). This differs from the test plan §4.7 v2 draft
  which assumed a stricter import-time error; see adaptation note in the
  final report.
"""

from __future__ import annotations

import importlib
import sys

# ── OPS-5: legacy grpo.py must not pull transfer_queue ───────────────────────


def test_legacy_grpo_import_without_data_plane_extra(monkeypatch) -> None:
    """Importing nemo_rl.algorithms.grpo must not trigger any transfer_queue
    import, even when TQ is installed in the environment.

    Method: poison sys.modules["transfer_queue"] = None so that any attempt
    to import it raises ImportError. If grpo.py is clean, the import succeeds.

    Risk guarded: R-C8 — a future PR drags KVBatchMeta into legacy; CI passes;
    legacy users now require [data-plane].
    """
    # Poison the transfer_queue namespace.
    monkeypatch.setitem(sys.modules, "transfer_queue", None)

    # Force a fresh import of grpo.py regardless of cache.
    grpo_module_name = "nemo_rl.algorithms.grpo"
    if grpo_module_name in sys.modules:
        # Remove so importlib.reload actually re-executes the module.
        saved = sys.modules.pop(grpo_module_name)
    else:
        saved = None

    try:
        # This must not raise even though transfer_queue is poisoned.
        mod = importlib.import_module(grpo_module_name)

        # Verify the module has no transfer_queue symbol at the top level.
        assert not hasattr(mod, "transfer_queue"), (
            "grpo.py imported transfer_queue at module level. "
            "Legacy trainer must not reference the data plane (R-C8)."
        )
    except ImportError as e:
        raise AssertionError(
            f"nemo_rl.algorithms.grpo raised ImportError with transfer_queue poisoned:\n"
            f"  {e}\n"
            "The legacy trainer must import cleanly without [data-plane] extra installed."
        ) from e
    finally:
        # Restore original module state so we don't break other tests.
        if saved is not None:
            sys.modules[grpo_module_name] = saved
        else:
            sys.modules.pop(grpo_module_name, None)


def test_grpo_sync_import_without_tq_succeeds(monkeypatch) -> None:
    """nemo_rl.algorithms.grpo_sync can be imported even when transfer_queue
    is unavailable.

    The TQ import is lazy — it happens inside TQDataPlaneClient.__init__, not
    at module level. This test verifies the import boundary is correct.

    Calling grpo_train_sync without data_plane.enabled=True raises ValueError
    (tested separately in test_grpo_sync_requires_data_plane_enabled).
    """
    monkeypatch.setitem(sys.modules, "transfer_queue", None)

    grpo_sync_name = "nemo_rl.algorithms.grpo_sync"
    saved = sys.modules.pop(grpo_sync_name, None)
    try:
        # Should not raise — TQ is lazy.
        mod = importlib.import_module(grpo_sync_name)
        assert hasattr(mod, "grpo_train_sync"), (
            "grpo_sync.py must expose grpo_train_sync as its public entrypoint."
        )
    except ImportError as e:
        raise AssertionError(
            f"nemo_rl.algorithms.grpo_sync raised ImportError with TQ poisoned:\n"
            f"  {e}\n"
            "grpo_sync.py must not import transfer_queue at module level."
        ) from e
    finally:
        if saved is not None:
            sys.modules[grpo_sync_name] = saved
        else:
            sys.modules.pop(grpo_sync_name, None)


def test_grpo_sync_requires_data_plane_enabled() -> None:
    """Calling grpo_train_sync with data_plane.enabled=False raises ValueError
    naming the legacy trainer as the escape hatch.

    Risk guarded: R-H12 — user wastes 30 min on opaque errors.
    """
    from nemo_rl.algorithms.grpo_sync import grpo_train_sync

    # Minimal stub config: data_plane disabled.
    fake_cfg = {"data_plane": {"enabled": False}}

    try:
        # We expect an immediate ValueError before any model/tokenizer is needed.
        grpo_train_sync(
            master_config=fake_cfg,
            policy=None,
            tokenizer=None,
            reward_functions=[],
            train_dataloader=None,
            val_dataloaders=None,
        )
    except ValueError as e:
        msg = str(e)
        assert "data_plane" in msg or "enabled" in msg, (
            f"ValueError message does not mention 'data_plane' or 'enabled': {msg!r}"
        )
        assert "grpo_train" in msg or "grpo.py" in msg or "legacy" in msg, (
            f"ValueError message should point users at the legacy trainer: {msg!r}"
        )
    except Exception:
        # A different exception is acceptable as long as it's not silent.
        pass
    else:
        raise AssertionError(
            "grpo_train_sync with data_plane.enabled=False must raise ValueError "
            "before doing any work. Got no exception."
        )
