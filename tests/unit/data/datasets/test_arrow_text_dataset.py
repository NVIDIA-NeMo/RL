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
"""PR #2508: ArrowTextDataset validation parity (D1) and config declaration (D2)."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc
import pytest

from nemo_rl.data import ResponseDatasetConfig


def _write_arrow_file(path: Path, rows: list[dict]) -> None:
    """Write a list-of-dict rows to ``path`` as a one-record-batch arrow IPC file."""
    table = pa.Table.from_pylist(rows)
    with ipc.new_file(str(path), table.schema) as writer:
        writer.write_table(table)


@pytest.fixture
def mixed_arrow_file(tmp_path: Path) -> str:
    """Arrow file with 3 valid strings, 1 empty string, 1 non-string row."""
    arrow_path = tmp_path / "mixed.arrow"
    _write_arrow_file(
        arrow_path,
        [
            {"text": "hello world"},
            {"text": "second sample with more chars"},
            {"text": ""},  # invalid: empty string
            {"text": "third valid sample"},
            {"text": None},  # invalid: non-string
        ],
    )
    return str(arrow_path)


# ---------------------------------------------------------------------------
# D1 — packed and non-packed paths must filter invalid text consistently.
# ---------------------------------------------------------------------------


def test_D1_unpacked_path_filters_invalid_rows(mixed_arrow_file):
    """Class A. ``characters_per_sample=None`` must filter out empty and
    non-string text rows just like the packed path's ``_pack_generator``
    does. Before the fix, only the packed path filters.
    """
    from nemo_rl.data.datasets.response_datasets.arrow_text_dataset import (
        ArrowTextDataset,
    )

    ds = ArrowTextDataset(
        data_files=mixed_arrow_file,
        text_key="text",
        characters_per_sample=None,
    )
    texts = [row["text"] for row in ds.dataset]
    # 3 valid rows total; empty string and None must be dropped.
    assert "" not in texts
    assert None not in texts
    assert len(texts) == 3, (
        f"Unpacked path failed to filter invalid rows: got {texts}. "
        "Reviewer flagged this: packed path drops them, unpacked path "
        "currently passes them through unchanged."
    )


def test_D1_packed_path_filters_invalid_rows(mixed_arrow_file):
    """Class A. Confirm the existing packed-path behavior so the harness
    catches a regression that *loosens* validation in the packed path
    to match a buggy unpacked path.
    """
    from nemo_rl.data.datasets.response_datasets.arrow_text_dataset import (
        ArrowTextDataset,
    )

    ds = ArrowTextDataset(
        data_files=mixed_arrow_file,
        text_key="text",
        characters_per_sample=10,
    )
    texts = [row["text"] for row in ds.dataset]
    # Each packed sample is a join of valid rows, but no packed sample
    # should be empty or None.
    assert all(isinstance(t, str) and t for t in texts), (
        f"Packed path emitted empty/None text: {texts}"
    )


def test_D1_packed_unpacked_emit_same_valid_row_set(mixed_arrow_file):
    """Class A. The union of substrings across packed samples must match
    the set of valid rows the unpacked path emits. Otherwise the two paths
    are loading different effective corpora.
    """
    from nemo_rl.data.datasets.response_datasets.arrow_text_dataset import (
        ArrowTextDataset,
    )

    ds_unpacked = ArrowTextDataset(
        data_files=mixed_arrow_file,
        text_key="text",
        characters_per_sample=None,
    )
    ds_packed = ArrowTextDataset(
        data_files=mixed_arrow_file,
        text_key="text",
        characters_per_sample=10,
    )
    unpacked_texts = {row["text"] for row in ds_unpacked.dataset}
    packed_blob = "\n".join(row["text"] for row in ds_packed.dataset)

    for t in unpacked_texts:
        assert t in packed_blob, (
            f"Valid row {t!r} present in unpacked dataset but missing from "
            f"packed corpus. The two code paths have drifted."
        )


# ---------------------------------------------------------------------------
# D2 — ResponseDatasetConfig must declare the keys ArrowTextDataset reads.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key",
    ["data_files", "text_key", "characters_per_sample"],
)
def test_D2_response_dataset_config_declares_arrow_text_keys(key: str):
    """Class B. Reviewer flagged that ArrowTextDataset reads undeclared
    config fields. After the fix, each must appear in
    ``ResponseDatasetConfig`` (as a required or optional key).
    """
    declared = set(ResponseDatasetConfig.__annotations__.keys())
    assert key in declared, (
        f"{key!r} is consumed by ArrowTextDataset but not declared on "
        f"ResponseDatasetConfig (declared keys: {sorted(declared)}). "
        f"Reviewer asked for explicit declaration."
    )
