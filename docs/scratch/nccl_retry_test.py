"""Retry test: after a failed init (peer died), can the survivors re-init at a SMALLER world size?

Mimics production: gen worker dies, router evicts it, surviving workers re-init
at world_size = world_size - 1.
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


def child(rank: int, world_size: int, master_addr: str, master_port: int,
          victim_rank: int, victim_delay_s: float, gpu_index: int,
          new_world_size: int, new_port: int,
          result_q: multiprocessing.Queue) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    sys.path.insert(0, "/opt/nemo-rl")
    sys.path.insert(0, "/work/src")
    import signal
    def _watchdog(signum, frame):  # noqa
        sys.stderr.write(f"[rank{rank}] WATCHDOG\n")
        os._exit(99)
    signal.signal(signal.SIGALRM, _watchdog)
    signal.alarm(120)

    t0 = time.monotonic()
    try:
        from nemo_rl.distributed.stateless_process_group import StatelessProcessGroup
        # Phase 1: full world size, with victim dying.
        pg1 = StatelessProcessGroup(master_addr, master_port, rank, world_size)
        if rank == victim_rank:
            import threading
            def _suicide():
                time.sleep(victim_delay_s)
                os._exit(137)
            threading.Thread(target=_suicide, daemon=True).start()
        try:
            pg1.init_nccl_communicator(device=0)
        except Exception as e:  # noqa
            phase1 = f"raised {type(e).__name__}: {str(e)[:120]}"
        else:
            phase1 = "ok (no failure detected!)"
        # Phase 2: surviving ranks re-init at smaller world. Skip the victim.
        if rank == victim_rank:
            return
        # Survivors get new ranks: pre-victim ranks keep their rank, post-victim
        # ranks shift down by 1.
        new_rank = rank if rank < victim_rank else rank - 1
        pg2 = StatelessProcessGroup(master_addr, new_port, new_rank, new_world_size)
        try:
            pg2.init_nccl_communicator(device=0)
            phase2 = f"ok (new_rank={new_rank}, new_ws={new_world_size})"
        except Exception as e:  # noqa
            phase2 = f"raised {type(e).__name__}: {str(e)[:120]}"
        elapsed = time.monotonic() - t0
        result_q.put({"rank": rank, "phase1": phase1, "phase2": phase2, "elapsed_s": round(elapsed, 2)})
    except BaseException as e:
        elapsed = time.monotonic() - t0
        result_q.put({
            "rank": rank,
            "phase1": "<unhandled>",
            "phase2": f"unhandled {type(e).__name__}: {str(e)[:200]}",
            "elapsed_s": round(elapsed, 2),
        })


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--world-size", type=int, default=4)
    p.add_argument("--victim", type=int, default=2)
    p.add_argument("--delay", type=float, default=0.5)
    args = p.parse_args()
    multiprocessing.set_start_method("spawn", force=True)
    port1 = find_free_port()
    port2 = find_free_port()
    new_world = args.world_size - 1

    print(f"=== retry test world={args.world_size} victim={args.victim} delay={args.delay}s "
          f"port1={port1} port2={port2} new_world={new_world} ===", flush=True)

    q = multiprocessing.Queue()
    procs = []
    for r in range(args.world_size):
        proc = multiprocessing.Process(target=child, args=(
            r, args.world_size, "127.0.0.1", port1,
            args.victim, args.delay, r,
            new_world, port2, q,
        ))
        proc.start()
        procs.append(proc)

    deadline = time.monotonic() + 100
    results = {}
    while procs and time.monotonic() < deadline:
        try:
            while True:
                m = q.get(timeout=0.5)
                results[m["rank"]] = m
        except Exception:
            pass
        procs = [p for p in procs if p.is_alive()]
        time.sleep(0.5)
    try:
        while True:
            m = q.get(timeout=1.0)
            results[m["rank"]] = m
    except Exception:
        pass
    for p in procs:
        try:
            p.kill()
        except Exception:
            pass

    print("=" * 60)
    print(f"RESULTS — retry, victim={args.victim}, delay={args.delay}s")
    for r in range(args.world_size):
        d = results.get(r)
        if d is None:
            print(f"  rank={r}: NO_RESULT")
        else:
            print(f"  rank={r}: phase1={d['phase1']}, phase2={d.get('phase2')}, elapsed={d.get('elapsed_s')}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
