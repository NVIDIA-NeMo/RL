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
# _pack_generator — packing semantics.
#
# Important: ``_pack_generator`` packs by **characters_per_sample**
# (character count, NOT token count or max_seq_length), emits a pack as
# soon as ``n >= characters_per_sample`` (so a pack can exceed the
# threshold by the size of the last appended row), and does NOT truncate
# any individual row. Empty-text filtering lives upstream in
# ``ArrowTextDataset`` — feeding empty rows directly into
# ``_pack_generator`` will still emit (empty) packs. The D1 tests above
# cover the upstream filtering.
# ---------------------------------------------------------------------------


from datasets import Dataset  # noqa: E402


def _packs_from(rows: list[dict], chars: int, task_name: str = "kd") -> list[dict]:
    from nemo_rl.data.datasets.response_datasets.arrow_text_dataset import (
        _pack_generator,
    )

    ds = Dataset.from_list(rows)
    return list(
        _pack_generator(
            raw=ds,
            text_key="text",
            characters_per_sample=chars,
            task_name=task_name,
        )
    )


class TestPackGenerator:
    def test_emits_when_threshold_met(self):
        # 3 rows of 4 chars each = 12 total; threshold 10. After row 3
        # (n=12 >= 10), one pack is emitted with all 3 rows joined.
        rows = [{"text": "aaaa"}, {"text": "bbbb"}, {"text": "cccc"}]
        packs = _packs_from(rows, chars=10)
        assert len(packs) == 1
        # Pack contains all three texts joined by newline.
        assert packs[0]["text"] == "aaaa\nbbbb\ncccc"
        # task_name forwarded into the emitted dict.
        assert packs[0]["task_name"] == "kd"
        # `messages` schema mirrors the source contract.
        assert packs[0]["messages"] == [
            {"role": "assistant", "content": "aaaa\nbbbb\ncccc"}
        ]

    def test_pack_may_exceed_threshold(self):
        # Two rows of 100 chars each, threshold 10 → first row alone
        # crosses the threshold, so each row becomes its own ~100-char
        # pack. The emitted pack content is ALLOWED to exceed
        # characters_per_sample.
        rows = [{"text": "x" * 100}, {"text": "y" * 100}]
        packs = _packs_from(rows, chars=10)
        assert len(packs) == 2
        assert len(packs[0]["text"]) == 100
        assert len(packs[1]["text"]) == 100

    def test_long_row_not_truncated(self):
        # A single row of 10k chars, threshold 10 → one pack containing
        # the full 10k chars. _pack_generator does NOT truncate rows.
        rows = [{"text": "z" * 10_000}]
        packs = _packs_from(rows, chars=10)
        assert len(packs) == 1
        assert len(packs[0]["text"]) == 10_000

    def test_trailing_partial_pack_emitted(self):
        # Three rows of 2 chars each (n=6 total) with threshold 10 — no
        # pack reaches the threshold mid-loop, so the for-loop emits
        # nothing. The trailing `if buf:` clause emits the accumulated
        # partial pack.
        rows = [{"text": "ab"}, {"text": "cd"}, {"text": "ef"}]
        packs = _packs_from(rows, chars=10)
        assert len(packs) == 1
        assert packs[0]["text"] == "ab\ncd\nef"

    def test_zero_rows_emits_zero_packs(self):
        # Truly empty input (no rows at all) → no packs, no raise.
        packs = _packs_from([], chars=10)
        assert packs == []

    def test_all_empty_text_rows_still_emit_pack(self):
        # _pack_generator does NOT filter empty strings — that's
        # ArrowTextDataset's job (covered by test_D1_*). Feeding empty
        # rows directly emits a (mostly-empty) trailing pack consisting
        # of newline separators only.
        rows = [{"text": ""}, {"text": ""}]
        packs = _packs_from(rows, chars=10)
        # Exactly one trailing partial pack, containing only the join
        # separator(s).
        assert len(packs) == 1
        assert packs[0]["text"] == "\n"

    def test_schema_version_does_not_change_output(self):
        # `schema_version` is `del`'d inside the function and only
        # affects the HF cache fingerprint — it must NOT alter the
        # emitted rows.
        rows = [{"text": "aaaa"}, {"text": "bbbb"}, {"text": "cccc"}]
        from nemo_rl.data.datasets.response_datasets.arrow_text_dataset import (
            _pack_generator,
        )

        ds = Dataset.from_list(rows)
        packs_v1 = list(
            _pack_generator(
                raw=ds,
                text_key="text",
                characters_per_sample=10,
                task_name="kd",
                schema_version="messages-v1",
            )
        )
        packs_v2 = list(
            _pack_generator(
                raw=ds,
                text_key="text",
                characters_per_sample=10,
                task_name="kd",
                schema_version="messages-v9999",
            )
        )
        assert packs_v1 == packs_v2

    def test_task_name_forwarded(self):
        rows = [{"text": "hello world"}]
        packs = _packs_from(rows, chars=5, task_name="my-task")
        assert packs[0]["task_name"] == "my-task"
