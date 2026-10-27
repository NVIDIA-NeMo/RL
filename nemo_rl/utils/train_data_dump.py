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

"""Stream untruncated training tensors without retaining a second step batch."""

import json
import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch


def json_safe_nonfinite(value: Any) -> Any:
    """Represent nonfinite diagnostic values as JSON null without mutating data."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe_nonfinite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe_nonfinite(item) for item in value]
    return value


class TrainDataDump:
    """Write chunks to a partial file, publishing only a completed optimizer step.

    Sequence columns are trimmed to input_lengths (padding only). Scalar
    columns are written per row as given, so variable-length values such as
    prompt_ids must be passed as jagged tensors rather than padded. Masked rows
    are retained. Values use the legacy train_data JSONL singleton-batch shape.
    A failed step leaves part files, never a completed-looking JSONL file.

    The advantage stage that produces these rows runs either in the controller
    or across a pool of Ray actors, so a step's rows are written by one writer
    per shard and merged on publish. Each writer owns
    ``train_data_step<N>.jsonl.part-<shard_id>`` and never touches another's,
    which is what makes concurrent shards safe; ``finish_step`` concatenates
    them in shard order and assigns ``idx`` across the merged file. Writers
    therefore omit ``idx`` -- only the merge sees every row, so only the merge
    can number them.

    Sharded writes assume the pool shares a filesystem with the controller,
    which is already true of the log dir the trainer checkpoints into.
    """

    def __init__(self, log_dir: str, *, shard_id: str = "0") -> None:
        self.log_dir = Path(log_dir)
        self.shard_id = shard_id
        self.step: int | None = None
        self.rows = 0

    def _part_path(self, step: int, shard_id: str) -> Path:
        return self.log_dir / f"train_data_step{step + 1}.jsonl.part-{shard_id}"

    def add_chunk(
        self,
        *,
        step: int,
        sample_ids: list[str],
        tags: list[dict[str, Any]] | None,
        input_lengths: torch.Tensor,
        sequences: dict[str, torch.Tensor],
        scalars: dict[str, torch.Tensor],
    ) -> None:
        rows = train_data_rows(
            step=step,
            sample_ids=sample_ids,
            tags=tags,
            input_lengths=input_lengths,
            sequences=sequences,
            scalars=scalars,
        )
        self.log_dir.mkdir(parents=True, exist_ok=True)
        part = self._part_path(step, self.shard_id)
        # Truncate on the first chunk of a step so a retried step replaces its
        # rows instead of appending a second copy to the previous attempt's.
        if self.step != step:
            self.rows = 0
        mode = "a" if self.step == step else "w"
        self.step = step
        with part.open(mode) as stream:
            for row in rows:
                stream.write(
                    json.dumps(json_safe_nonfinite(row), allow_nan=False) + "\n"
                )
                self.rows += 1

    def finish_step(self, step: int, expected_rows: int | None = None) -> None:
        """Merge every shard's part file and publish the step atomically.

        Called on the controller, which may not itself have written anything:
        with a pool the rows come from the actors, so the parts on disk are the
        only record of what the step produced.

        ``expected_rows`` is what the shards reported writing. A pool that does
        not share this filesystem with the controller still runs and still
        reports, and its parts are simply not here -- which would publish a
        short dump that looks complete. Checking the count is what turns that
        into a failure.
        """
        parts = sorted(self.log_dir.glob(f"train_data_step{step + 1}.jsonl.part-*"))
        if not parts:
            raise RuntimeError("Completed optimizer step has no training dump")
        partial = self.log_dir / f"train_data_step{step + 1}.jsonl.partial"
        rows = 0
        with partial.open("w") as merged:
            for part in parts:
                with part.open() as stream:
                    for line in stream:
                        # Splice idx onto the front rather than reparsing: the
                        # writers emit it nowhere, and a dump large enough to
                        # want sharding is large enough to not re-serialize.
                        merged.write(f'{{"idx": {rows}, {line[1:]}')
                        rows += 1
        if not rows:
            partial.unlink()
            raise RuntimeError("Completed optimizer step has no training dump")
        if expected_rows is not None and rows != expected_rows:
            partial.unlink()
            raise RuntimeError(
                f"Training dump merged {rows} row(s) from {len(parts)} shard(s) "
                f"but the stage reported writing {expected_rows}; the advantage "
                "pool must share this filesystem with the controller"
            )
        os.replace(partial, self.log_dir / f"train_data_step{step + 1}.jsonl")
        for part in parts:
            part.unlink()
        self.step = None
        self.rows = 0


def train_data_rows(
    *,
    step: int,
    sample_ids: list[str],
    tags: list[dict[str, Any]] | None,
    input_lengths: torch.Tensor,
    sequences: dict[str, torch.Tensor],
    scalars: dict[str, torch.Tensor],
) -> list[dict[str, Any]]:
    """One chunk's training-data rows in the legacy singleton-batch shape.

    Sequence columns are trimmed to each row's input length; scalars are kept as
    given. Shared by the JSONL dump and the W&B table, so both hold the same rows.
    """
    lengths = input_lengths.detach().cpu().reshape(-1).tolist()
    # unbind() splits dense and jagged columns alike into per-row tensors.
    columns = {
        k: list(v.detach().cpu().unbind()) for k, v in {**sequences, **scalars}.items()
    }
    if (
        len(lengths) != len(sample_ids)
        or any(len(v) != len(sample_ids) for v in columns.values())
        or (tags is not None and len(tags) != len(sample_ids))
    ):
        raise ValueError("Training dump column lengths do not match sample ids")
    for i, length in enumerate(lengths):
        if length < 0 or any(length > columns[k][i].shape[0] for k in sequences):
            raise ValueError("Training dump input length exceeds a sequence column")
    rows = []
    for i, length in enumerate(lengths):
        row = {
            "step": step + 1,
            "sample_id": [sample_ids[i]],
            "input_lengths": [length],
            "metadata": [tags[i] if tags is not None else {}],
        }
        for key in sequences:
            row[key] = [columns[key][i][:length].tolist()]
        for key in scalars:
            row[key] = [columns[key][i].tolist()]
        rows.append(row)
    return rows


_log = logging.getLogger(__name__)


class WandbTrainData:
    """Log this process's training-data rows to the controller's W&B run (shared mode).

    ``add_chunk`` takes ``TrainDataDump.add_chunk``'s arguments and returns at once;
    one background thread attaches to the run and logs each chunk as a table. Array
    cells are JSON strings: wandb.Table type-checks every element of a list cell.
    """

    def __init__(self, run: dict[str, Any], *, label: str) -> None:
        self._target = run  # id, project, entity of the controller's run
        self._label = label
        self._run: Any = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=label)

    def add_chunk(self, **chunk: Any) -> None:
        self._executor.submit(self._log, chunk)

    def _log(self, chunk: dict[str, Any]) -> None:
        try:
            # Deferred: wandb loads only in processes that log the table.
            import wandb

            if self._run is None:
                if wandb.run is not None and wandb.run.id == self._target["id"]:
                    self._run = wandb.run  # the controller, which owns the run
                else:
                    self._run = wandb.init(
                        **self._target,
                        settings=wandb.Settings(
                            mode="shared",
                            x_primary=False,
                            x_label=self._label,
                            x_update_finish_state=False,
                        ),
                    )
                self._run.define_metric(
                    "train/train_data", step_metric="train/train_data_step"
                )
            rows = train_data_rows(**chunk)
            columns = list(rows[0])
            table = wandb.Table(
                columns=columns,
                data=[[_cell(row[column]) for column in columns] for row in rows],
            )
            self._run.log(
                {"train/train_data": table, "train/train_data_step": rows[0]["step"]}
            )
        except Exception:  # noqa: BLE001 - a debug table must not stop training
            _log.warning("training-data rows not logged to W&B", exc_info=True)


def _cell(value: Any) -> Any:
    if isinstance(value, list) and len(value) == 1:
        value = value[0]  # the dump's singleton-batch shape
    return json.dumps(value) if isinstance(value, (list, dict)) else value
