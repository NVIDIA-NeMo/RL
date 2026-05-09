"""Reproduce the TCPStore-rendezvous cascade for RL-412 fault-tolerant generation.

The bug: ``StatelessProcessGroup.__init__`` calls ``torch.distributed.TCPStore``
with ``is_master=(rank==0)`` and a 30s timeout. If one peer never connects, rank 0
times out → raises DistStoreError. EVERY other rank waiting on the same store
ALSO raises DistStoreError. Result: cascade-eviction of all gen workers.

This script spawns ``world_size`` child processes. A configurable "victim"
either (a) sleeps a long delay before constructing the TCPStore (b) never
constructs it at all (simulating a dead actor). Survivors record their
exception type, traceback, and elapsed time; the parent prints a per-rank
summary.

Modes:
  - vanilla:       Plain ``StatelessProcessGroup.__init__`` (production path).
                   Expected: cascade — rank 0 + survivors all raise DistStoreError.
  - sentinel_keys: Each rank sets ``rank_alive_{rank}`` BEFORE rendezvous so
                   rank 0 can identify which ranks are missing on timeout.
  - presence_then_rendezvous: Two-phase rendezvous: first an ephemeral
                              "presence" TCPStore (rank 0 polls num_keys to
                              identify ready ranks), then real TCPStore at
                              the smaller world.
"""
import argparse
import multiprocessing
import os
import socket
import sys
import time
import traceback


def find_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _install_self_watchdog(rank: int, deadline_s: float) -> None:
    """Force-exit after ``deadline_s`` so a hung child doesn't wedge the parent."""
    import signal

    def _handler(signum, frame):  # noqa: ARG001
        sys.stderr.write(f"[rank{rank}] SELF-WATCHDOG fired at {deadline_s}s\n")
        sys.stderr.flush()
        os._exit(99)

    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(int(deadline_s))


# =====================================================================
# Mode: vanilla — reproduce the production cascade.
# =====================================================================


def child_vanilla(rank: int, world_size: int, master_addr: str, master_port: int,
                  victim_rank: int, victim_delay_s: float, gpu_index: int,
                  result_q: multiprocessing.Queue) -> None:
    """Plain ``StatelessProcessGroup.__init__`` — the production cascade."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    sys.path.insert(0, "/opt/nemo-rl")
    sys.path.insert(0, "/work/src")
    _install_self_watchdog(rank, deadline_s=120.0)
    t0 = time.monotonic()
    print(f"[rank{rank}] starting", flush=True)
    try:
        # Victim never participates at all (simulates a Ray actor that died
        # before init_collective dispatch reached it).
        if rank == victim_rank:
            print(f"[rank{rank}] victim — sleeping {victim_delay_s}s, never joins TCPStore", flush=True)
            time.sleep(victim_delay_s)
            elapsed = time.monotonic() - t0
            result_q.put({"rank": rank, "status": "victim_absent",
                          "elapsed_s": round(elapsed, 2)})
            return

        from nemo_rl.distributed.stateless_process_group import StatelessProcessGroup

        print(f"[rank{rank}] entering StatelessProcessGroup.__init__", flush=True)
        pg = StatelessProcessGroup(master_addr, master_port, rank, world_size)
        print(f"[rank{rank}] TCPStore bound", flush=True)
        elapsed = time.monotonic() - t0
        result_q.put({"rank": rank, "status": "ok", "elapsed_s": round(elapsed, 2)})
    except BaseException as e:
        elapsed = time.monotonic() - t0
        tb = traceback.format_exc(limit=4)
        result_q.put({
            "rank": rank,
            "status": f"raised {type(e).__name__}",
            "msg": str(e)[:300],
            "tb_tail": tb.splitlines()[-3:] if tb else [],
            "elapsed_s": round(elapsed, 2),
        })


# =====================================================================
# Mode: sentinel_keys — each rank sets a presence sentinel, rank 0
# enumerates which keys exist on a probe to identify missing ranks.
# =====================================================================


def child_sentinel_keys(rank: int, world_size: int, master_addr: str, master_port: int,
                        victim_rank: int, victim_delay_s: float, gpu_index: int,
                        result_q: multiprocessing.Queue) -> None:
    """Two-step rendezvous: every rank publishes a presence key, rank 0 reads them."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    sys.path.insert(0, "/opt/nemo-rl")
    sys.path.insert(0, "/work/src")
    _install_self_watchdog(rank, deadline_s=180.0)
    t0 = time.monotonic()
    try:
        if rank == victim_rank:
            print(f"[rank{rank}] victim — sleeping {victim_delay_s}s, never joins TCPStore", flush=True)
            time.sleep(victim_delay_s)
            elapsed = time.monotonic() - t0
            result_q.put({"rank": rank, "status": "victim_absent",
                          "elapsed_s": round(elapsed, 2)})
            return

        # Use a small TCPStore for presence, then construct the StatelessProcessGroup.
        from datetime import timedelta
        import torch
        # Phase 1: ephemeral "presence" store at world_size=1 (no peer wait)
        # — every rank can connect freely.
        presence_port = master_port + 1
        # Rank 0 is the master; others are clients. world_size=1 means rank 0
        # doesn't block waiting for clients during construction.
        # NB: in PyTorch TCPStore, world_size=1 + is_master=True allows the
        # store to start immediately; clients can connect later via num_keys.
        presence_store = torch.distributed.TCPStore(
            host_name=master_addr,
            port=presence_port,
            world_size=-1,   # unbounded; clients added dynamically
            is_master=(rank == 0),
            timeout=timedelta(seconds=10),
            wait_for_workers=False,
        )
        # Every rank publishes its sentinel.
        presence_store.set(f"rank_alive_{rank}", b"1")
        print(f"[rank{rank}] published rank_alive_{rank}", flush=True)

        if rank == 0:
            # Poll until either (a) all ranks present (b) some short deadline expires.
            poll_deadline = time.monotonic() + 5.0
            present_ranks: list[int] = []
            while time.monotonic() < poll_deadline:
                present_ranks = []
                for r in range(world_size):
                    try:
                        v = presence_store.get(f"rank_alive_{r}")
                        if v:
                            present_ranks.append(r)
                    except Exception:
                        pass
                if len(present_ranks) == world_size:
                    break
                time.sleep(0.2)
            print(f"[rank0] presence={present_ranks} of {world_size}", flush=True)
            # Publish the resolved world.
            present_str = ",".join(str(r) for r in present_ranks)
            presence_store.set("present_ranks", present_str.encode())
            present_count = len(present_ranks)
            present_set = set(present_ranks)
        else:
            # Other ranks wait for rank 0 to publish the resolved world.
            presence_store.wait(["present_ranks"], timedelta(seconds=10))
            present_str = presence_store.get("present_ranks").decode()
            present_set = set(int(x) for x in present_str.split(",") if x)
            present_count = len(present_set)

        if rank not in present_set:
            # We didn't make it into the resolved world — surface that.
            result_q.put({"rank": rank, "status": "not_in_present_set",
                          "elapsed_s": round(time.monotonic() - t0, 2)})
            return

        # Phase 2: real TCPStore rendezvous at present_count (smaller world).
        # Re-rank ourselves: our new rank is our index in the sorted present list.
        sorted_present = sorted(present_set)
        new_rank = sorted_present.index(rank)
        new_world = present_count
        print(f"[rank{rank}] phase2 — old_rank={rank} new_rank={new_rank} new_world={new_world}",
              flush=True)

        from nemo_rl.distributed.stateless_process_group import StatelessProcessGroup
        pg = StatelessProcessGroup(master_addr, master_port, new_rank, new_world)
        print(f"[rank{rank}] phase2 TCPStore bound", flush=True)
        elapsed = time.monotonic() - t0
        result_q.put({
            "rank": rank, "status": "ok",
            "msg": f"phase2 world={new_world} (was {world_size})",
            "elapsed_s": round(elapsed, 2),
        })
    except BaseException as e:
        elapsed = time.monotonic() - t0
        tb = traceback.format_exc(limit=4)
        result_q.put({
            "rank": rank,
            "status": f"raised {type(e).__name__}",
            "msg": str(e)[:300],
            "tb_tail": tb.splitlines()[-3:] if tb else [],
            "elapsed_s": round(elapsed, 2),
        })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="vanilla",
                        choices=["vanilla", "sentinel_keys"])
    parser.add_argument("--victim", type=int, default=2)
    parser.add_argument("--delay", type=float, default=120.0,
                        help="Seconds the victim sleeps (must exceed 30s store timeout).")
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--watchdog-s", type=float, default=120.0)
    args = parser.parse_args()

    multiprocessing.set_start_method("spawn", force=True)
    port = find_free_port()
    master_addr = "127.0.0.1"

    target = {
        "vanilla": child_vanilla,
        "sentinel_keys": child_sentinel_keys,
    }[args.mode]

    print(
        f"=== mode={args.mode} world_size={args.world_size} victim={args.victim} "
        f"delay={args.delay}s port={port} ===",
        flush=True,
    )

    q = multiprocessing.Queue()
    procs: list[multiprocessing.Process] = []
    for r in range(args.world_size):
        p = multiprocessing.Process(target=target, args=(
            r, args.world_size, master_addr, port,
            args.victim, args.delay, r, q,
        ))
        p.start()
        procs.append(p)

    deadline = time.monotonic() + args.watchdog_s
    results: dict[int, dict] = {}
    while procs and time.monotonic() < deadline:
        try:
            while True:
                msg = q.get(timeout=0.5)
                results[msg["rank"]] = msg
        except Exception:
            pass
        for p in list(procs):
            if not p.is_alive():
                procs.remove(p)
        time.sleep(0.5)

    try:
        while True:
            msg = q.get(timeout=0.5)
            results[msg["rank"]] = msg
    except Exception:
        pass

    import signal as _signal
    for p in procs:
        try:
            os.kill(p.pid, _signal.SIGKILL)
        except Exception:
            pass

    print("=" * 60)
    print(f"RESULTS — mode={args.mode} victim={args.victim} delay={args.delay}s")
    print("-" * 60)
    for r in range(args.world_size):
        d = results.get(r)
        if d is None:
            print(f"  rank={r}: NO_RESULT (process died without reporting)")
        else:
            line = f"  rank={r}: status={d['status']}, elapsed_s={d.get('elapsed_s')}"
            if d.get("msg"):
                line += f", msg={d['msg'][:200]!r}"
            print(line)
            if d.get("tb_tail"):
                for ln in d["tb_tail"]:
                    print(f"      {ln}")
    print("=" * 60)


if __name__ == "__main__":
    main()
