#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Check that BIRD SQLite databases are readable the way the bird_sql Gym server reads them.

The server (Gym resources_servers/bird_sql/eval_utils.py) opens each query with a plain
``sqlite3.connect(db_path)`` from many threads at once and turns every exception into ``None``,
so a database that cannot be opened scores 0 silently (gold_execution_error). This script
repeats that access pattern -- concurrent threads in several processes -- and reports, per
database file, its journal mode (file header bytes 18-19: 1 = rollback journal, 2 = WAL),
how many queries succeeded, and every distinct error message.

Run it where training/eval runs: on a compute node, inside the training container, e.g.
  srun -A <account> -p <partition> -N1 -t 10 --container-image=<sqsh> \
       --container-mounts=/lustre:/lustre --no-container-mount-home \
       python3 scripts/check_bird_sqlite.py <db_dir> [<db_dir> ...]

Read-only queries only. Opening a WAL database may create <db>-shm / <db>-wal files next to it,
exactly as the server does.
"""

import argparse
import collections
import glob
import multiprocessing as mp
import os
import platform
import socket
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor


def journal_mode(path: str) -> str:
    with open(path, "rb") as f:
        header = f.read(20)
    mode = {(1, 1): "rollback", (2, 2): "WAL"}
    return mode.get((header[18], header[19]), f"unknown({header[18]},{header[19]})")


def one_query(path: str) -> str:
    """Mirror eval_utils.execute_sqlite: connect, run a read query, close. Return 'ok' or the error."""
    conn = None
    try:
        conn = sqlite3.connect(path)
        cur = conn.cursor()
        tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        if not tables:
            return "error: no tables"
        cur.execute(f'SELECT count(*) FROM "{tables[0]}"').fetchall()
        return "ok"
    except Exception as e:  # noqa: BLE001 - we want every failure mode
        return f"{type(e).__name__}: {e}"
    finally:
        if conn is not None:
            conn.close()


def worker(args: tuple[str, int, int]) -> collections.Counter:
    path, threads, queries = args
    with ThreadPoolExecutor(max_workers=threads) as pool:
        return collections.Counter(pool.map(one_query, [path] * queries))


def fstype(path: str) -> str:
    try:
        return subprocess.run(
            ["findmnt", "-T", path, "-no", "FSTYPE,OPTIONS"], capture_output=True, text=True, check=False
        ).stdout.strip() or "?"
    except FileNotFoundError:
        return "findmnt not available"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("db_dirs", nargs="+", help="directories searched recursively for *.sqlite")
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--threads", type=int, default=16, help="threads per process")
    parser.add_argument("--queries", type=int, default=64, help="queries per process")
    parser.add_argument("--only", default=None, help="substring filter on the path, e.g. card_games")
    args = parser.parse_args()

    print(f"host={socket.gethostname()} arch={platform.machine()} python={platform.python_version()} "
          f"sqlite={sqlite3.sqlite_version}")
    any_failed = False
    for db_dir in args.db_dirs:
        paths = sorted(glob.glob(os.path.join(db_dir, "**", "*.sqlite"), recursive=True))
        if args.only:
            paths = [p for p in paths if args.only in p]
        print(f"\n== {db_dir}  [{fstype(db_dir)}]  {len(paths)} files")
        for path in paths:
            with mp.get_context("spawn").Pool(args.processes) as pool:
                totals = sum(pool.map(worker, [(path, args.threads, args.queries)] * args.processes),
                             collections.Counter())
            ok, n = totals.pop("ok", 0), args.processes * args.queries
            status = "OK  " if ok == n else "FAIL"
            any_failed |= ok != n
            print(f"{status} {journal_mode(path):8s} {ok:4d}/{n}  {os.path.relpath(path, db_dir)}")
            for err, count in totals.most_common():
                print(f"       {count:4d} x {err}")
    print("\nRESULT:", "some databases FAILED" if any_failed else "all databases OK")
    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
