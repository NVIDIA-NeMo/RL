"""Reproduce NCCL bootstrap peer-death cascade on 4 GPUs in one pod.

Each rank:
  - Sets CUDA_VISIBLE_DEVICES to its assigned physical GPU
  - Builds StatelessProcessGroup(master_address, port, rank, world_size=4)
  - Calls init_nccl_communicator(device=0)

A configurable "victim" rank sleeps for VICTIM_DELAY_S seconds and then
os._exit(137)'s, simulating SIGKILL during NCCL bootstrap. Survivors record
status (ok / raised <type> / sigabrt / hung).

Driver script: `python3 nccl_repro.py [--victim 2] [--delay 0.5] [--mode <mode>]`

Modes:
  - vanilla:   plain init_nccl_communicator (the production code path)
  - tryexcept: wrap init_nccl_communicator in try/except RuntimeError
  - timeout:   shorten StatelessProcessGroup timeout via NCCL_TIMEOUT env (informational)
  - nonblocking: use NCCLConfig(blocking=False) + poll get_async_error
  - subprocess:  run init in a child subprocess so peer death cannot abort the parent
"""
import argparse
import multiprocessing
import os
import signal
import socket
import sys
import time
import traceback
from datetime import timedelta


def find_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _install_self_watchdog(rank: int, deadline_s: float) -> None:
    """Self-watchdog: if this child is still alive past `deadline_s`, force-exit.

    Used so a hung child surfaces as a clean process termination instead of
    hanging the parent's join. SIGALRM handler raises so wrapping try/except
    can also catch it.
    """
    import signal
    def _handler(signum, frame):  # noqa: ARG001
        sys.stderr.write(f"[rank{rank}] SELF-WATCHDOG fired at {deadline_s}s\n")
        sys.stderr.flush()
        os._exit(99)
    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(int(deadline_s))


def child_vanilla(rank: int, world_size: int, master_addr: str, master_port: int,
                  victim_rank: int, victim_delay_s: float, gpu_index: int,
                  result_q: multiprocessing.Queue) -> None:
    """Plain init — this is the production behavior we are reproducing."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    sys.path.insert(0, "/opt/nemo-rl")
    sys.path.insert(0, "/work/src")
    _install_self_watchdog(rank, deadline_s=60.0)
    t0 = time.monotonic()
    print(f"[rank{rank}] starting", flush=True)
    try:
        from nemo_rl.distributed.stateless_process_group import StatelessProcessGroup
        # Victim joins TCPStore + starts NCCL bootstrap, then suicides AFTER
        # the configured delay so peers are mid-handshake.
        pg = StatelessProcessGroup(master_addr, master_port, rank, world_size)
        print(f"[rank{rank}] TCPStore bound", flush=True)
        if rank == victim_rank:
            # Schedule a deferred SIGKILL so it lands during NCCL bootstrap,
            # not before TCPStore rendezvous.
            import threading
            def _suicide():
                time.sleep(victim_delay_s)
                print(f"[rank{rank}] victim — SIGKILLing self after {victim_delay_s}s", flush=True)
                os._exit(137)
            threading.Thread(target=_suicide, daemon=True).start()
        print(f"[rank{rank}] entering init_nccl_communicator", flush=True)
        pg.init_nccl_communicator(device=0)
        print(f"[rank{rank}] init returned", flush=True)
        elapsed = time.monotonic() - t0
        result_q.put({"rank": rank, "status": "ok", "elapsed_s": round(elapsed, 2)})
        # Drain — keep the survivors alive briefly so the bootstrap fully closes.
        time.sleep(2.0)
        try:
            pg.destroy()
        except Exception:
            pass
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


def child_tryexcept(rank: int, world_size: int, master_addr: str, master_port: int,
                    victim_rank: int, victim_delay_s: float, gpu_index: int,
                    result_q: multiprocessing.Queue) -> None:
    """Try/except RuntimeError around init_nccl_communicator."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    sys.path.insert(0, "/opt/nemo-rl")
    sys.path.insert(0, "/work/src")
    _install_self_watchdog(rank, deadline_s=60.0)
    t0 = time.monotonic()
    try:
        from nemo_rl.distributed.stateless_process_group import StatelessProcessGroup
        pg = StatelessProcessGroup(master_addr, master_port, rank, world_size)
        if rank == victim_rank:
            import threading
            def _suicide():
                time.sleep(victim_delay_s)
                print(f"[rank{rank}] victim — SIGKILLing self after {victim_delay_s}s", flush=True)
                os._exit(137)
            threading.Thread(target=_suicide, daemon=True).start()
        try:
            pg.init_nccl_communicator(device=0)
            status = "ok"
            msg = ""
        except (RuntimeError, Exception) as inner:  # noqa: BLE001
            status = f"caught {type(inner).__name__}"
            msg = str(inner)[:300]
        elapsed = time.monotonic() - t0
        result_q.put({"rank": rank, "status": status, "msg": msg, "elapsed_s": round(elapsed, 2)})
    except BaseException as e:
        elapsed = time.monotonic() - t0
        tb = traceback.format_exc(limit=4)
        result_q.put({
            "rank": rank,
            "status": f"unhandled {type(e).__name__}",
            "msg": str(e)[:300],
            "tb_tail": tb.splitlines()[-3:] if tb else [],
            "elapsed_s": round(elapsed, 2),
        })


def child_nonblocking(rank: int, world_size: int, master_addr: str, master_port: int,
                      victim_rank: int, victim_delay_s: float, gpu_index: int,
                      result_q: multiprocessing.Queue,
                      poll_timeout_s: float = 30.0) -> None:
    """Non-blocking NCCL bootstrap with poll loop + abort on error."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    sys.path.insert(0, "/opt/nemo-rl")
    sys.path.insert(0, "/work/src")
    _install_self_watchdog(rank, deadline_s=60.0)
    t0 = time.monotonic()
    try:
        import torch
        from nccl.core.communicator import Communicator, NCCLConfig
        from nccl.core.utils import UniqueId, get_unique_id
        from nccl import bindings as _nccl_bindings

        # Mini-rendezvous via TCPStore
        from datetime import timedelta as _td
        store = torch.distributed.TCPStore(
            host_name=master_addr, port=master_port, world_size=world_size,
            is_master=(rank == 0), timeout=_td(seconds=30),
        )
        if rank == victim_rank:
            import threading
            def _suicide():
                time.sleep(victim_delay_s)
                print(f"[rank{rank}] victim — SIGKILLing self after {victim_delay_s}s", flush=True)
                os._exit(137)
            threading.Thread(target=_suicide, daemon=True).start()
        UNIQUE_ID_KEY = "nccl_unique_id"
        if rank == 0:
            uid = get_unique_id()
            store.set(UNIQUE_ID_KEY, uid.as_bytes)
        else:
            store.wait([UNIQUE_ID_KEY], _td(seconds=30))
            uid = UniqueId.from_bytes(store.get(UNIQUE_ID_KEY))

        cfg = NCCLConfig(blocking=False)

        with torch.cuda.device(0):
            comm = Communicator.init(nranks=world_size, rank=rank, unique_id=uid, config=cfg)
            # Poll until success or error
            deadline = time.monotonic() + poll_timeout_s
            last_state = None
            while True:
                state = comm.get_async_error()
                last_state = state
                # state is a Result; ncclSuccess == 0, ncclInProgress is positive int
                state_int = int(state) if hasattr(state, '__int__') else state
                if state_int == 0:
                    break
                # ncclInProgress is "in progress" — keep waiting
                # Anything else is an error
                # Check the enum from bindings
                if hasattr(_nccl_bindings, 'Result'):
                    success_v = int(_nccl_bindings.Result.success)
                    in_progress_v = int(_nccl_bindings.Result.in_progress) if hasattr(_nccl_bindings.Result, 'in_progress') else None
                    if state_int == success_v:
                        break
                    if in_progress_v is not None and state_int == in_progress_v:
                        if time.monotonic() > deadline:
                            try:
                                comm.abort()
                            except Exception:
                                pass
                            elapsed = time.monotonic() - t0
                            result_q.put({"rank": rank, "status": "timeout_during_init",
                                          "msg": f"state={state_int}", "elapsed_s": round(elapsed, 2)})
                            return
                        time.sleep(0.05)
                        continue
                    # error
                    err_msg = comm.get_last_error()
                    try:
                        comm.abort()
                    except Exception:
                        pass
                    elapsed = time.monotonic() - t0
                    result_q.put({"rank": rank, "status": "nccl_error_during_init",
                                  "msg": f"state={state_int} last_err={err_msg!r}",
                                  "elapsed_s": round(elapsed, 2)})
                    return
                else:
                    if time.monotonic() > deadline:
                        try:
                            comm.abort()
                        except Exception:
                            pass
                        result_q.put({"rank": rank, "status": "timeout_no_result_enum",
                                      "msg": f"state={state_int}", "elapsed_s": round(time.monotonic() - t0, 2)})
                        return
                    time.sleep(0.05)

            # Warmup broadcast: also poll non-blocking on this
            data = torch.ones(1, device=0) if rank == 0 else torch.zeros(1, device=0)
            stream = torch.cuda.current_stream()
            comm.broadcast(sendbuf=data, recvbuf=data, root=0, stream=int(stream.cuda_stream))
            # Poll for completion
            while True:
                state = comm.get_async_error()
                state_int = int(state)
                if state_int == 0:
                    break
                if hasattr(_nccl_bindings, 'Result') and hasattr(_nccl_bindings.Result, 'in_progress'):
                    if state_int == int(_nccl_bindings.Result.in_progress):
                        if time.monotonic() > deadline + 30:
                            err_msg = comm.get_last_error()
                            try:
                                comm.abort()
                            except Exception:
                                pass
                            result_q.put({"rank": rank, "status": "broadcast_timeout",
                                          "msg": f"last={err_msg!r}",
                                          "elapsed_s": round(time.monotonic() - t0, 2)})
                            return
                        time.sleep(0.05)
                        continue
                err_msg = comm.get_last_error()
                try:
                    comm.abort()
                except Exception:
                    pass
                result_q.put({"rank": rank, "status": "broadcast_error",
                              "msg": f"state={state_int} err={err_msg!r}",
                              "elapsed_s": round(time.monotonic() - t0, 2)})
                return
            torch.cuda.current_stream().synchronize()

            elapsed = time.monotonic() - t0
            result_q.put({"rank": rank, "status": "ok", "elapsed_s": round(elapsed, 2)})
            time.sleep(1.0)
            try:
                comm.abort()
            except Exception:
                pass
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


def child_destroy_kill(rank: int, world_size: int, master_addr: str, master_port: int,
                       victim_rank: int, victim_delay_s: float, gpu_index: int,
                       result_q: multiprocessing.Queue) -> None:
    """Bootstrap successfully, then victim dies as the group enters destroy()."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    sys.path.insert(0, "/opt/nemo-rl")
    sys.path.insert(0, "/work/src")
    _install_self_watchdog(rank, deadline_s=90.0)
    t0 = time.monotonic()
    print(f"[rank{rank}] destroy_kill starting", flush=True)
    try:
        from nemo_rl.distributed.stateless_process_group import StatelessProcessGroup
        pg = StatelessProcessGroup(master_addr, master_port, rank, world_size)
        pg.init_nccl_communicator(device=0)
        # All ranks have completed init. Now schedule victim's death AFTER
        # everyone reaches the rendezvous below. Use a very short delay so
        # the victim suicides ~10ms after entering destroy() — enough for
        # the survivors to also reach destroy() but not enough for them to
        # actually finish.
        # Synchronization barrier: a TCPStore key written by all ranks
        # ensures everyone is past init before victim arms its kill.
        pg.tcp_store.add("phase1_done", 1)
        # Wait until all 4 ranks have reported.
        while int(pg.tcp_store.get("phase1_done") or b"0") < world_size:
            time.sleep(0.01)
        if rank == victim_rank:
            import threading
            def _suicide():
                time.sleep(victim_delay_s)
                print(f"[rank{rank}] victim — SIGKILLing self at destroy entry", flush=True)
                os._exit(137)
            threading.Thread(target=_suicide, daemon=True).start()
        # Now everyone enters destroy().
        try:
            pg.destroy()
            status = "destroy_ok"
            msg = ""
        except Exception as inner:  # noqa: BLE001
            status = f"destroy_raised {type(inner).__name__}"
            msg = str(inner)[:300]
        elapsed = time.monotonic() - t0
        result_q.put({"rank": rank, "status": status, "msg": msg, "elapsed_s": round(elapsed, 2)})
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


def child_broadcast_kill(rank: int, world_size: int, master_addr: str, master_port: int,
                         victim_rank: int, victim_delay_s: float, gpu_index: int,
                         result_q: multiprocessing.Queue,
                         num_broadcasts: int = 200) -> None:
    """Bootstrap + warmup OK, then run several broadcasts; victim dies during one of them."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    sys.path.insert(0, "/opt/nemo-rl")
    sys.path.insert(0, "/work/src")
    _install_self_watchdog(rank, deadline_s=90.0)
    t0 = time.monotonic()
    print(f"[rank{rank}] broadcast_kill starting", flush=True)
    try:
        from nemo_rl.distributed.stateless_process_group import StatelessProcessGroup
        import torch
        pg = StatelessProcessGroup(master_addr, master_port, rank, world_size)
        pg.init_nccl_communicator(device=0)
        pg.tcp_store.add("init_done", 1)
        while int(pg.tcp_store.get("init_done") or b"0") < world_size:
            time.sleep(0.01)
        if rank == victim_rank:
            import threading
            def _suicide():
                time.sleep(victim_delay_s)
                print(f"[rank{rank}] victim — SIGKILLing self during broadcast loop", flush=True)
                os._exit(137)
            threading.Thread(target=_suicide, daemon=True).start()
        # Run several LARGE broadcasts (so each one takes long enough that
        # the victim has a real chance to die mid-broadcast). 64 MiB of
        # bf16 is ~32 MiB transfer per rank — measurable at NCCL-link speeds.
        completed = 0
        try:
            for i in range(num_broadcasts):
                if rank == 0:
                    data = torch.full((64 * 1024 * 1024,), float(i + 1), device=0, dtype=torch.float32)
                else:
                    data = torch.zeros(64 * 1024 * 1024, device=0, dtype=torch.float32)
                pg.broadcast(data, 0)
                torch.cuda.current_stream().synchronize()
                completed = i + 1
            status = f"all_{num_broadcasts}_ok"
            msg = ""
        except Exception as inner:  # noqa: BLE001
            status = f"bcast_raised {type(inner).__name__}"
            msg = f"completed={completed}/{num_broadcasts}: {str(inner)[:200]}"
        elapsed = time.monotonic() - t0
        result_q.put({"rank": rank, "status": status, "msg": msg, "elapsed_s": round(elapsed, 2)})
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


def child_subprocess(rank: int, world_size: int, master_addr: str, master_port: int,
                     victim_rank: int, victim_delay_s: float, gpu_index: int,
                     result_q: multiprocessing.Queue,
                     init_timeout_s: float = 60.0) -> None:
    """Run NCCL init in a subprocess so peer death can't abort the parent.

    The victim's helper child suicides itself with a deferred kill (after the
    configured delay) so the SUBPROCESS dies, simulating the production case
    where one Ray actor's NCCL helper dies. The parent (this Ray-actor proxy)
    must survive: we expect a non-zero exit on the victim and clean exit code 0
    on survivors. If the survivors' helpers SIGABRT, they'll exit non-zero but
    the parent process still survives — that's the whole point of subprocess
    isolation.
    """
    _install_self_watchdog(rank, deadline_s=80.0)
    t0 = time.monotonic()
    # Spawn a Python subprocess that does the actual init.
    import subprocess
    helper_path = "/work/src/nccl_init_helper.py"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    cmd = [sys.executable, helper_path,
           "--master-addr", master_addr,
           "--master-port", str(master_port),
           "--rank", str(rank),
           "--world-size", str(world_size)]
    if rank == victim_rank:
        # Helper schedules its own death after delay seconds.
        cmd += ["--suicide-after-s", str(victim_delay_s)]
    try:
        cp = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=init_timeout_s,
        )
        elapsed = time.monotonic() - t0
        if cp.returncode == 0:
            result_q.put({"rank": rank, "status": "ok",
                          "elapsed_s": round(elapsed, 2),
                          "child_stdout_tail": cp.stdout.splitlines()[-3:],
                          })
        else:
            # Surface non-zero as a clean Python-level outcome; PARENT survives.
            result_q.put({"rank": rank,
                          "status": f"child_exit_{cp.returncode}",
                          "msg": (cp.stderr or cp.stdout)[-400:],
                          "elapsed_s": round(elapsed, 2)})
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - t0
        result_q.put({"rank": rank, "status": "child_hung",
                      "elapsed_s": round(elapsed, 2)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="vanilla",
                        choices=["vanilla", "tryexcept", "nonblocking", "subprocess",
                                 "destroy_kill", "broadcast_kill"])
    parser.add_argument("--victim", type=int, default=2)
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--watchdog-s", type=float, default=120.0)
    args = parser.parse_args()

    multiprocessing.set_start_method("spawn", force=True)
    port = find_free_port()
    master_addr = "127.0.0.1"

    target = {
        "vanilla": child_vanilla,
        "tryexcept": child_tryexcept,
        "nonblocking": child_nonblocking,
        "subprocess": child_subprocess,
        "destroy_kill": child_destroy_kill,
        "broadcast_kill": child_broadcast_kill,
    }[args.mode]

    print(f"=== mode={args.mode} world_size={args.world_size} victim={args.victim} delay={args.delay}s port={port} ===", flush=True)

    q = multiprocessing.Queue()
    procs: list[multiprocessing.Process] = []
    for r in range(args.world_size):
        p = multiprocessing.Process(target=target, args=(
            r, args.world_size, master_addr, port,
            args.victim, args.delay, r, q,
        ))
        p.start()
        procs.append(p)

    # Wait with watchdog
    deadline = time.monotonic() + args.watchdog_s
    results: dict[int, dict] = {}
    while procs and time.monotonic() < deadline:
        # collect any results that are ready
        try:
            while True:
                msg = q.get(timeout=0.5)
                results[msg["rank"]] = msg
        except Exception:
            pass
        # check process status
        alive = []
        for p in procs:
            if p.is_alive():
                alive.append(p)
            else:
                rank = procs.index(p)  # noqa: not strictly index
        # Check if any procs have died without sending a result
        for p in list(procs):
            if not p.is_alive():
                # extract rank from name
                rank = int(p.name.split("-")[-1]) if "-" in p.name else -1
                # The ProcessName isn't tied to rank; we just track dead procs.
                procs.remove(p)
        time.sleep(0.5)

    # Drain final messages
    try:
        while True:
            msg = q.get(timeout=0.5)
            results[msg["rank"]] = msg
    except Exception:
        pass

    # Kill any survivors that are hung
    for p in procs:
        try:
            os.kill(p.pid, signal.SIGKILL)
            results[-p.pid] = {"rank": "?", "status": "HUNG_KILLED", "pid": p.pid}
        except Exception:
            pass

    # Report
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
