# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""NemoGymDataset variant that defers the dataset read past run_start.

The MLPerf rules require that the clock starts before any part of the system
touches the dataset. The stock ``NemoGymDataset`` reads the whole JSONL into
memory at construction, which happens during untimed init. This drop-in
replacement takes only metadata before run_start — the record count, from a
newline scan whose page-cache footprint is dropped again afterwards — and
reads the rows on first access, inside the timed region.

Selected via the dotted-path form of ``data.*.dataset_name``
(``mlperf_lazy_dataset.LazyNemoGymDataset``); this file is importable because
the benchmark directory is on ``PYTHONPATH`` in the container.
"""

from __future__ import annotations

import os

from nemo_rl.data.datasets.raw_dataset import RawDataset


def _stat_identity(stat: os.stat_result) -> tuple[int, int, int, int, int]:
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


class _LazyJsonlRows:
    """Map-style rows over a JSONL file, read on first item access."""

    def __init__(self, data_path: str, task_name: str, repeat: int) -> None:
        self.data_path = data_path
        self.task_name = task_name
        self._rows: list[str] | None = None
        self._num_records, self._identity = self._scan_metadata(data_path)
        self._len = self._num_records * repeat

    @staticmethod
    def _scan_metadata(data_path: str) -> tuple[int, tuple[int, int, int, int, int]]:
        fd = os.open(data_path, os.O_RDONLY)
        try:
            count, tail = 0, b"\n"
            while chunk := os.read(fd, 1 << 20):
                count += chunk.count(b"\n")
                tail = chunk[-1:]
            if tail != b"\n":
                count += 1
            identity = _stat_identity(os.fstat(fd))
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            except OSError as error:
                print(
                    f"mlperf_lazy_dataset: page-cache eviction of {data_path} "
                    f"failed ({error}); the timed first read may hit pages "
                    f"warmed by the metadata scan",
                    flush=True,
                )
        finally:
            os.close(fd)
        return count, identity

    def _read_rows(self) -> list[str]:
        with open(self.data_path) as f:
            rows = f.readlines()
            if _stat_identity(os.fstat(f.fileno())) != self._identity:
                raise RuntimeError(
                    f"{self.data_path} changed between the metadata scan and "
                    f"the first read"
                )
        if len(rows) != self._num_records:
            raise RuntimeError(
                f"{self.data_path} record count changed between the metadata "
                f"scan ({self._num_records}) and the first read ({len(rows)})"
            )
        return rows

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> dict[str, str]:
        if not 0 <= idx < self._len:
            raise IndexError(idx)
        if self._rows is None:
            self._rows = self._read_rows()
        return {
            "extra_env_info": self._rows[idx % self._num_records],
            "task_name": self.task_name,
        }


class LazyNemoGymDataset(RawDataset):
    """Drop-in for ``NemoGymDataset`` whose rows load after run_start.

    ``agent_name_sources`` is an empty set rather than this file's identity:
    the sharded setup-time route-coverage scan would otherwise read and parse
    every row before run_start. With no sources to scan the check passes
    vacuously; a row naming an unhosted route still fails fast at its first
    dispatch (``NemoGymShardSet.shard_for_route``).
    """

    agent_name_sources: frozenset = frozenset()

    def __init__(self, data_path: str, repeat: int = 1, **kwargs) -> None:
        self.task_name = "-".join(data_path.split("/")[-2:]).split(".")[0]
        if self.task_name[0] == "-":
            self.task_name = self.task_name[1:]
        self.dataset = _LazyJsonlRows(data_path, self.task_name, repeat)
