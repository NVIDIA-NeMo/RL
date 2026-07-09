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
"""Unit tests for ``nemo_rl.data.utils.get_train_dataset_name`` and
``nemo_rl.data.utils.load_dataloader_state``.

These helpers underpin the dataset-swap-aware checkpoint resume logic: when
the saved ``dataset_name`` (read from the ``config.yaml`` written alongside
``train_dataloader.pt``) differs from the current run's ``dataset_name``, the
loader skips restoring ``samples_yielded`` so the new dataset starts from
index 0. They are pure Python and require no GPUs / Ray / models.
"""

import os
from typing import Any, Optional

import pytest
import torch
import yaml
from torchdata.stateful_dataloader import StatefulDataLoader

from nemo_rl.data.utils import get_train_dataset_name, load_dataloader_state

# ---------------------------------------------------------------------------
# Test fixtures / helpers
# ---------------------------------------------------------------------------


class _RangeDataset:
    """Tiny dataset returning integer-keyed strings so consumed samples are
    identifiable after a save/restore cycle."""

    def __init__(self, name: str, n: int):
        self.name = name
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> str:
        if idx < 0 or idx >= self.n:
            raise IndexError(f"{self.name}: idx={idx} out of range [0, {self.n})")
        return f"{self.name}-{idx:04d}"


def _id_collate(batch: list[Any]) -> list[Any]:
    return list(batch)


def _make_dl(name: str, n: int, batch_size: int = 1) -> StatefulDataLoader:
    return StatefulDataLoader(
        _RangeDataset(name, n),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_id_collate,
        drop_last=False,
    )


def _consume(dl: StatefulDataLoader, n_batches: int) -> list[Any]:
    """Consume ``n_batches`` batches and return what was yielded."""
    out = []
    it = iter(dl)
    for _ in range(n_batches):
        out.append(next(it))
    return out


def _write_checkpoint(
    dir_path: str,
    state: Any,
    saved_dataset_name: Optional[str] = None,
    write_config: bool = True,
    suffix: str = "",
) -> str:
    """Mimic what ``CheckpointManager.save_checkpoint`` writes for our purposes.

    Always writes ``train_dataloader{suffix}.pt`` as a raw state dict (the
    format the algorithms have always written). When ``write_config`` is true,
    also writes a sibling ``config.yaml`` carrying the saved ``dataset_name``
    under ``data.train.dataset_name``, exactly as the real checkpointer does
    via ``yaml.safe_dump(run_config.model_dump())``.
    """
    os.makedirs(dir_path, exist_ok=True)
    torch.save(state, os.path.join(dir_path, f"train_dataloader{suffix}.pt"))
    if write_config:
        cfg = {"data": {"train": {"dataset_name": saved_dataset_name}}}
        with open(os.path.join(dir_path, "config.yaml"), "w") as f:
            yaml.safe_dump(cfg, f)
    return dir_path


# ---------------------------------------------------------------------------
# get_train_dataset_name
# ---------------------------------------------------------------------------


class TestGetTrainDatasetName:
    """``get_train_dataset_name`` must tolerate every shape of
    ``data_config["train"]`` that the algorithms may produce, and must also
    work on the plain dict produced by ``yaml.safe_load(config.yaml)``."""

    def test_dict_shape_returns_name(self):
        # setup_preference_data (DPO/RM) leaves train as a dict.
        assert get_train_dataset_name({"train": {"dataset_name": "A"}}) == "A"

    def test_list_of_one_returns_name(self):
        # setup_response_data (GRPO/Distillation) and run_sft.setup_data
        # normalize a single-dataset dict into [dict].
        assert get_train_dataset_name({"train": [{"dataset_name": "A"}]}) == "A"

    def test_list_of_many_returns_first(self):
        # Multi-dataset concat (single dataloader, N datasets) — we use the
        # first dataset's name as a fingerprint so a swap is still detected.
        assert (
            get_train_dataset_name(
                {"train": [{"dataset_name": "A"}, {"dataset_name": "B"}]}
            )
            == "A"
        )

    def test_empty_list_returns_none(self):
        assert get_train_dataset_name({"train": []}) is None

    def test_none_train_returns_none(self):
        assert get_train_dataset_name({"train": None}) is None

    def test_missing_train_returns_none(self):
        assert get_train_dataset_name({}) is None

    def test_none_config_returns_none(self):
        assert get_train_dataset_name(None) is None

    def test_dict_without_dataset_name_returns_none(self):
        assert get_train_dataset_name({"train": {"foo": "bar"}}) is None

    def test_list_with_dict_missing_dataset_name_returns_none(self):
        assert get_train_dataset_name({"train": [{"foo": "bar"}]}) is None


# ---------------------------------------------------------------------------
# load_dataloader_state
# ---------------------------------------------------------------------------


class TestLoadDataloaderState:
    """End-to-end behavior of ``load_dataloader_state`` against
    ``StatefulDataLoader``: matching names restore state, mismatched names
    skip + warn, and checkpoints without a sibling ``config.yaml`` fall back
    to the historical raw-load behavior."""

    def test_match_restores_state(self, tmp_path):
        # Train on D1 for 5 batches, save state + config.yaml(dataset_name=D1).
        dl_save = _make_dl("D1", 20)
        _consume(dl_save, 5)
        _write_checkpoint(str(tmp_path), dl_save.state_dict(), saved_dataset_name="D1")

        # Resume on a fresh dataloader with the same name -> state restored.
        dl_new = _make_dl("D1", 20)
        load_dataloader_state(dl_new, str(tmp_path), {"train": {"dataset_name": "D1"}})
        # The next yielded sample should skip the 5 already consumed.
        assert next(iter(dl_new))[0] == "D1-0005"

    def test_mismatch_skips_state_and_warns(self, tmp_path, capsys):
        # Save under D1.
        dl_save = _make_dl("D1", 20)
        _consume(dl_save, 5)
        _write_checkpoint(str(tmp_path), dl_save.state_dict(), saved_dataset_name="D1")

        # Resume with a *different* dataset name -> swap detected.
        dl_new = _make_dl("D2", 20)
        load_dataloader_state(dl_new, str(tmp_path), {"train": {"dataset_name": "D2"}})
        captured = capsys.readouterr()
        assert "Dataset swap detected" in captured.out
        assert "'D1'" in captured.out and "'D2'" in captured.out

        # Because state was NOT restored, iteration starts from index 0.
        assert next(iter(dl_new))[0] == "D2-0000"

    def test_mismatch_with_short_new_dataset_does_not_crash(self, tmp_path):
        """Regression: when ``len(new_dataset) <= samples_yielded`` of the old
        dataset, the unguarded torchdata path crashes with ``StopIteration``
        during ``iter()`` construction. The swap-skip must make this safe."""
        dl_save = _make_dl("D1", 100)
        _consume(dl_save, 50)  # samples_yielded == 50
        _write_checkpoint(str(tmp_path), dl_save.state_dict(), saved_dataset_name="D1")

        # New dataset is shorter than 50 -- without the swap-skip this would
        # raise StopIteration on the first ``iter(dl_new)``.
        dl_new = _make_dl("D2_short", 5)
        load_dataloader_state(
            dl_new, str(tmp_path), {"train": {"dataset_name": "D2_short"}}
        )
        items = list(dl_new)
        assert items[0][0] == "D2_short-0000"
        assert len(items) == 5

    def test_no_config_yaml_falls_back_to_load(self, tmp_path):
        """Backward compat for very old checkpoints that somehow have no
        sibling ``config.yaml``: behave like the pre-fix code and just load
        the raw state. (Real nemo-rl checkpoints always write ``config.yaml``,
        so this branch mostly guards corrupt / hand-rolled dirs.)"""
        dl_save = _make_dl("D1", 20)
        _consume(dl_save, 5)
        _write_checkpoint(str(tmp_path), dl_save.state_dict(), write_config=False)

        dl_new = _make_dl("D1", 20)
        load_dataloader_state(dl_new, str(tmp_path), {"train": {"dataset_name": "D1"}})
        assert next(iter(dl_new))[0] == "D1-0005"

    def test_saved_name_none_loads_state(self, tmp_path):
        # config.yaml exists but has no dataset_name -> "can't determine" ->
        # load state to preserve legacy behavior.
        dl_save = _make_dl("D1", 20)
        _consume(dl_save, 5)
        _write_checkpoint(str(tmp_path), dl_save.state_dict(), saved_dataset_name=None)

        dl_new = _make_dl("D1", 20)
        load_dataloader_state(dl_new, str(tmp_path), {"train": {"dataset_name": "D1"}})
        assert next(iter(dl_new))[0] == "D1-0005"

    def test_current_name_none_loads_state(self, tmp_path):
        # Saved name known, current name unknown -> load state.
        dl_save = _make_dl("D1", 20)
        _consume(dl_save, 5)
        _write_checkpoint(str(tmp_path), dl_save.state_dict(), saved_dataset_name="D1")

        dl_new = _make_dl("D1", 20)
        load_dataloader_state(dl_new, str(tmp_path), {"train": None})
        assert next(iter(dl_new))[0] == "D1-0005"

    def test_suffix_routes_to_correct_file(self, tmp_path):
        """GRPO's multi-dataloader path uses ``suffix=f"_{task_name}"`` --
        verify the helper honors it. There is still just one ``config.yaml``
        per step dir, shared across all suffixed ``.pt`` files."""
        dl_save = _make_dl("D1", 20)
        _consume(dl_save, 5)
        _write_checkpoint(
            str(tmp_path),
            dl_save.state_dict(),
            saved_dataset_name="D1",
            suffix="_math",
        )

        dl_new = _make_dl("D1", 20)
        load_dataloader_state(
            dl_new,
            str(tmp_path),
            {"train": {"dataset_name": "D1"}},
            suffix="_math",
        )
        assert next(iter(dl_new))[0] == "D1-0005"

    def test_missing_file_raises(self, tmp_path):
        """No ``train_dataloader.pt`` at the expected path should propagate
        as an error (consistent with the pre-fix behavior)."""
        dl_new = _make_dl("D1", 20)
        with pytest.raises((FileNotFoundError, RuntimeError)):
            load_dataloader_state(
                dl_new, str(tmp_path), {"train": {"dataset_name": "D1"}}
            )
