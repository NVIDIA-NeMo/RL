"""TP=2 cross-cluster NCCL init/broadcast/kill repro.

Mirrors the production pattern:
  - 1 train rank (rank 0) doing StatelessProcessGroup at world_size=3
  - 1 vLLM "leader" actor that delegates to 2 inner TP workers
    (mocked via multiprocessing.Process — same isolation model as
    vLLM's RayDistributedExecutor + multiproc TP fan-out)
  - inner TP workers compute rank = train_world_size + rank_prefix +
    local_rank and create their own StatelessProcessGroup at the
    cross-cluster world_size

Question we want to answer: when one inner TP worker dies during
init_nccl_communicator (or destroy(), or broadcast()), does the LEADER
PROCESS:
  (a) survive and surface a Python exception via the inner-worker
      result queue (good — we can catch and force-evict),
  (b) hang forever (caller's await blocks until watchdog),
  (c) DIE itself (e.g. SIGABRT cascade through CUDA driver context).

The production symptom is (c)-equivalent for Ray actors: the leader
"died unexpectedly... killed by ray.kill" message likely means the
underlying Ray worker process exited because something the leader
called (collective_rpc) raised a fatal/uncatchable error.

Run on a node with 3+ GPUs (1 train + 2 gen TP). No vLLM/Ray
dependency: we mock the leader→inner fan-out so we test purely the
NCCL bootstrap layer.

Usage:
  python docs/scratch/tp2_nccl_repro.py --mode tp2_init_clean
  python docs/scratch/tp2_nccl_repro.py --mode tp2_init_victim_dies_during_bootstrap
  python docs/scratch/tp2_nccl_repro.py --mode tp2_init_victim_dies_during_destroy
  python docs/scratch/tp2_nccl_repro.py --mode tp2_broadcast_victim_dies
  python docs/scratch/tp2_nccl_repro.py --mode tp2_collective_rpc_inner_raises
"""

import argparse
import multiprocessing
import os
import signal
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


def _install_self_watchdog(tag: str, deadline_s: float) -> None:
    def _handler(signum, frame):  # noqa: ARG001
        sys.stderr.write(f"[{tag}] SELF-WATCHDOG fired at {deadline_s}s\n")
        sys.stderr.flush()
        os._exit(99)

    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(int(deadline_s))


# ---------------------------------------------------------------------------
# Inner TP worker: a child process spawned by the leader. Equivalent to
# one of vLLM's two TP-internal workers when TP=2. Calls into NeMo-RL's
# StatelessProcessGroup at the cross-cluster rank.
# ---------------------------------------------------------------------------


def inner_tp_worker(
    cross_rank: int,
    cross_world_size: int,
    master_addr: str,
    master_port: int,
    gpu_index: int,
    role: str,  # "victim" or "survivor" or "raise_inner"
    delay_s: float,
    phase: str,  # "bootstrap", "destroy", "broadcast", "raise_pre_init"
    result_q: multiprocessing.Queue,
) -> None:
    """A single TP-internal vLLM worker (mocked).

    Lives in its own subprocess (multiprocessing.Process), exactly the
    way vLLM's MultiprocessingDistributedExecutor isolates its TP
    workers under the leader actor. The leader fans out a
    `collective_rpc("init_collective")` call which dispatches into this
    function on each TP rank.
    """
    tag = f"inner-cross_rank{cross_rank}-tp{cross_rank - 1}"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    sys.path.insert(0, "/opt/nemo-rl")
    sys.path.insert(0, "/work/src")
    _install_self_watchdog(tag, deadline_s=90.0)
    t0 = time.monotonic()

    try:
        if phase == "raise_pre_init" and role == "raise_inner":
            # Simulate inner worker raising synchronously during the
            # collective_rpc call body, BEFORE NCCL init even starts.
            raise RuntimeError(
                f"[{tag}] simulated inner-worker exception in init_collective body"
            )

        from nemo_rl.distributed.stateless_process_group import StatelessProcessGroup

        pg = StatelessProcessGroup(
            master_addr, master_port, cross_rank, cross_world_size
        )
        print(f"[{tag}] TCPStore bound", flush=True)

        if role == "victim" and phase == "bootstrap":
            import threading

            def _suicide():
                time.sleep(delay_s)
                print(f"[{tag}] victim — SIGKILL during NCCL bootstrap", flush=True)
                os._exit(137)

            threading.Thread(target=_suicide, daemon=True).start()

        print(f"[{tag}] entering init_nccl_communicator", flush=True)
        pg.init_nccl_communicator(device=0)
        print(f"[{tag}] init returned", flush=True)

        # Phase 1 barrier — everyone reached past init.
        pg.tcp_store.add("init_done", 1)
        deadline_b = time.monotonic() + 30.0
        while int(pg.tcp_store.get("init_done") or b"0") < cross_world_size:
            if time.monotonic() > deadline_b:
                raise RuntimeError(f"[{tag}] init_done barrier timeout")
            time.sleep(0.05)

        if role == "victim" and phase == "broadcast":
            import threading

            def _suicide_b():
                time.sleep(delay_s)
                print(f"[{tag}] victim — SIGKILL during broadcast loop", flush=True)
                os._exit(137)

            threading.Thread(target=_suicide_b, daemon=True).start()

        if phase == "broadcast":
            import torch

            completed = 0
            try:
                for i in range(50):
                    if cross_rank == 0:
                        # Train rank is the broadcast root in the cross-
                        # cluster setup; gen workers are receivers.
                        pass  # gen workers don't broadcast as root here
                    if cross_rank == 0:
                        data = torch.full(
                            (16 * 1024 * 1024,),
                            float(i + 1),
                            device=0,
                            dtype=torch.float32,
                        )
                    else:
                        data = torch.zeros(
                            16 * 1024 * 1024, device=0, dtype=torch.float32
                        )
                    pg.broadcast(data, 0)
                    torch.cuda.current_stream().synchronize()
                    completed = i + 1
                bcast_status = f"all_50_ok"
                bcast_msg = ""
            except Exception as e:  # noqa: BLE001
                bcast_status = f"bcast_raised {type(e).__name__}"
                bcast_msg = f"completed={completed}: {str(e)[:200]}"
            elapsed = time.monotonic() - t0
            result_q.put(
                {
                    "tag": tag,
                    "cross_rank": cross_rank,
                    "status": bcast_status,
                    "msg": bcast_msg,
                    "elapsed_s": round(elapsed, 2),
                }
            )
            time.sleep(1.0)
            try:
                pg.destroy()
            except Exception:
                pass
            return

        if role == "victim" and phase == "destroy":
            import threading

            def _suicide_d():
                time.sleep(delay_s)
                print(f"[{tag}] victim — SIGKILL at destroy", flush=True)
                os._exit(137)

            threading.Thread(target=_suicide_d, daemon=True).start()

        # Default phase=="bootstrap" or phase=="destroy" — do destroy and report.
        try:
            pg.destroy()
            destroy_status = "destroy_ok"
            destroy_msg = ""
        except Exception as e:  # noqa: BLE001
            destroy_status = f"destroy_raised {type(e).__name__}"
            destroy_msg = str(e)[:300]

        elapsed = time.monotonic() - t0
        result_q.put(
            {
                "tag": tag,
                "cross_rank": cross_rank,
                "status": "init_ok+" + destroy_status,
                "msg": destroy_msg,
                "elapsed_s": round(elapsed, 2),
            }
        )

    except BaseException as e:
        elapsed = time.monotonic() - t0
        tb = traceback.format_exc(limit=4)
        result_q.put(
            {
                "tag": tag,
                "cross_rank": cross_rank,
                "status": f"raised {type(e).__name__}",
                "msg": str(e)[:300],
                "tb_tail": tb.splitlines()[-3:] if tb else [],
                "elapsed_s": round(elapsed, 2),
            }
        )


# ---------------------------------------------------------------------------
# The leader process: this is what plays the role of the vLLM
# AsyncWorker (Ray actor). It spawns 2 inner TP workers and waits for
# them. The question is: if one inner dies, does the leader die or
# survive?
# ---------------------------------------------------------------------------


def leader_process(
    rank_prefix: int,
    train_world_size: int,
    cross_world_size: int,
    master_addr: str,
    master_port: int,
    gpu_indices: list,  # length 2: GPU index for tp_rank=0 and tp_rank=1
    phase: str,
    victim_local_rank: int,  # 0 or 1, which TP rank is the victim
    role_for_phase: str,  # "victim" / "raise_inner" / etc
    delay_s: float,
    leader_q: multiprocessing.Queue,
) -> None:
    """Leader = mock of VllmAsyncGenerationWorker.init_collective_async.

    In production this would be a Ray actor; here we use a plain process
    so the test runs without Ray. The fan-out via multiprocessing
    matches vLLM's MultiprocessingDistributedExecutor isolation: each TP
    worker is its own OS process.
    """
    tag = "leader"
    _install_self_watchdog(tag, deadline_s=120.0)
    t0 = time.monotonic()
    inner_q: multiprocessing.Queue = multiprocessing.Queue()
    inner_procs = []
    leader_status = "ok"
    leader_msg = ""
    inner_results: dict = {}

    try:
        for local_rank in range(2):
            cross_rank = train_world_size + rank_prefix + local_rank
            if local_rank == victim_local_rank:
                role = role_for_phase
            else:
                role = "survivor"
            p = multiprocessing.Process(
                target=inner_tp_worker,
                args=(
                    cross_rank,
                    cross_world_size,
                    master_addr,
                    master_port,
                    gpu_indices[local_rank],
                    role,
                    delay_s,
                    phase,
                    inner_q,
                ),
                name=f"inner-tp{local_rank}",
            )
            p.start()
            inner_procs.append(p)

        # === This is the moral equivalent of `await llm.collective_rpc()` ===
        # We block waiting for both inner workers. In real vLLM, this is
        # a synchronous fan-out internally. If one raises, vLLM raises
        # to the caller. If one OS-dies, behavior depends on vLLM's
        # executor implementation (which under multiproc TP is
        # signal-based — SIGCHLD / pipe-EOF on the worker IPC channel).
        deadline = time.monotonic() + 90.0
        for p in inner_procs:
            remaining = max(0.5, deadline - time.monotonic())
            p.join(timeout=remaining)
            if p.is_alive():
                leader_status = "inner_hung"
                leader_msg = f"{p.name} still alive past deadline"
                try:
                    os.kill(p.pid, signal.SIGKILL)
                except Exception:
                    pass

        # Drain inner result queue.
        for _ in range(2):
            try:
                msg = inner_q.get(timeout=2.0)
                inner_results[msg["tag"]] = msg
            except Exception:
                break

        # Look at exit codes — that's what surfaces "dead inner worker"
        # to the leader in real vLLM (worker IPC pipe closes, executor
        # raises EngineDeadError). Surface that as our leader_status.
        for p in inner_procs:
            ec = p.exitcode
            if ec is not None and ec != 0:
                if leader_status == "ok":
                    leader_status = f"inner_exited_nonzero({p.name}={ec})"

    except BaseException as e:
        leader_status = f"raised {type(e).__name__}"
        leader_msg = str(e)[:300]

    elapsed = time.monotonic() - t0
    leader_q.put(
        {
            "leader_status": leader_status,
            "leader_msg": leader_msg,
            "leader_elapsed_s": round(elapsed, 2),
            "leader_pid": os.getpid(),
            "inner_results": inner_results,
            "inner_exitcodes": {p.name: p.exitcode for p in inner_procs},
        }
    )


# ---------------------------------------------------------------------------
# The "train rank" — sibling to the leader. Stands in for the train side
# (RefitWorker / Megatron worker) at cross_rank=0.
# ---------------------------------------------------------------------------


def train_rank_process(
    cross_world_size: int,
    master_addr: str,
    master_port: int,
    gpu_index: int,
    phase: str,
    result_q: multiprocessing.Queue,
) -> None:
    tag = "train-cross_rank0"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    sys.path.insert(0, "/opt/nemo-rl")
    sys.path.insert(0, "/work/src")
    _install_self_watchdog(tag, deadline_s=120.0)
    t0 = time.monotonic()
    try:
        from nemo_rl.distributed.stateless_process_group import StatelessProcessGroup

        pg = StatelessProcessGroup(master_addr, master_port, 0, cross_world_size)
        pg.init_nccl_communicator(device=0)

        pg.tcp_store.add("init_done", 1)
        deadline_b = time.monotonic() + 30.0
        while int(pg.tcp_store.get("init_done") or b"0") < cross_world_size:
            if time.monotonic() > deadline_b:
                raise RuntimeError(f"[{tag}] init_done barrier timeout")
            time.sleep(0.05)

        if phase == "broadcast":
            import torch

            completed = 0
            try:
                for i in range(50):
                    data = torch.full(
                        (16 * 1024 * 1024,),
                        float(i + 1),
                        device=0,
                        dtype=torch.float32,
                    )
                    pg.broadcast(data, 0)
                    torch.cuda.current_stream().synchronize()
                    completed = i + 1
                status = "all_50_ok"
                msg = ""
            except Exception as e:  # noqa: BLE001
                status = f"bcast_raised {type(e).__name__}"
                msg = f"completed={completed}: {str(e)[:200]}"
            result_q.put(
                {
                    "tag": tag,
                    "status": status,
                    "msg": msg,
                    "elapsed_s": round(time.monotonic() - t0, 2),
                }
            )
            time.sleep(1.0)
            try:
                pg.destroy()
            except Exception:
                pass
            return

        try:
            pg.destroy()
            d_status = "destroy_ok"
            d_msg = ""
        except Exception as e:  # noqa: BLE001
            d_status = f"destroy_raised {type(e).__name__}"
            d_msg = str(e)[:300]

        result_q.put(
            {
                "tag": tag,
                "status": "init_ok+" + d_status,
                "msg": d_msg,
                "elapsed_s": round(time.monotonic() - t0, 2),
            }
        )
    except BaseException as e:
        tb = traceback.format_exc(limit=4)
        result_q.put(
            {
                "tag": tag,
                "status": f"raised {type(e).__name__}",
                "msg": str(e)[:300],
                "tb_tail": tb.splitlines()[-3:] if tb else [],
                "elapsed_s": round(time.monotonic() - t0, 2),
            }
        )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_mode(mode: str, watchdog_s: float = 180.0):
    multiprocessing.set_start_method("spawn", force=True)
    port = find_free_port()
    master_addr = "127.0.0.1"
    train_world_size = 1
    cross_world_size = 3  # 1 train + 2 gen TP

    # Pick GPU layout: train on GPU 0, leader's two TP workers on GPUs 1 & 2.
    gpu_train = 0
    gpu_inner = [1, 2]

    # Mode-specific behavior
    if mode == "tp2_init_clean":
        phase = "bootstrap"
        victim_local = -1  # no victim
        role_for_phase = "survivor"
        delay = 0.0
    elif mode == "tp2_init_victim_dies_during_bootstrap":
        phase = "bootstrap"
        victim_local = 1  # TP rank 1 dies
        role_for_phase = "victim"
        delay = 0.5
    elif mode == "tp2_init_victim_dies_during_destroy":
        phase = "destroy"
        victim_local = 1
        role_for_phase = "victim"
        delay = 0.0
    elif mode == "tp2_broadcast_victim_dies":
        phase = "broadcast"
        victim_local = 1
        role_for_phase = "victim"
        delay = 1.0
    elif mode == "tp2_collective_rpc_inner_raises":
        phase = "raise_pre_init"
        victim_local = 1
        role_for_phase = "raise_inner"
        delay = 0.0
    else:
        raise ValueError(f"unknown mode: {mode}")

    print(
        f"=== mode={mode} cross_world={cross_world_size} train_gpu={gpu_train} "
        f"inner_gpus={gpu_inner} victim_local={victim_local} delay={delay}s "
        f"port={port} ===",
        flush=True,
    )

    train_q: multiprocessing.Queue = multiprocessing.Queue()
    leader_q: multiprocessing.Queue = multiprocessing.Queue()

    train_p = multiprocessing.Process(
        target=train_rank_process,
        args=(
            cross_world_size,
            master_addr,
            port,
            gpu_train,
            phase if phase != "raise_pre_init" else "bootstrap",
            train_q,
        ),
        name="train",
    )
    leader_p = multiprocessing.Process(
        target=leader_process,
        args=(
            0,  # rank_prefix
            train_world_size,
            cross_world_size,
            master_addr,
            port,
            gpu_inner,
            phase,
            victim_local,
            role_for_phase,
            delay,
            leader_q,
        ),
        name="leader",
    )

    train_p.start()
    leader_p.start()

    deadline = time.monotonic() + watchdog_s
    while (
        train_p.is_alive() or leader_p.is_alive()
    ) and time.monotonic() < deadline:
        time.sleep(1.0)

    # Forced kill if anyone hung
    for p in (train_p, leader_p):
        if p.is_alive():
            try:
                os.kill(p.pid, signal.SIGKILL)
            except Exception:
                pass
            p.join(timeout=5.0)

    train_result = None
    leader_result = None
    try:
        train_result = train_q.get(timeout=2.0)
    except Exception:
        pass
    try:
        leader_result = leader_q.get(timeout=2.0)
    except Exception:
        pass

    print("=" * 70)
    print(f"RESULTS — mode={mode}")
    print("-" * 70)
    print(f"train process: exitcode={train_p.exitcode}")
    if train_result:
        print(
            f"  status={train_result['status']} elapsed={train_result.get('elapsed_s')}s"
        )
        if train_result.get("msg"):
            print(f"  msg={train_result['msg'][:200]!r}")
    else:
        print("  NO_RESULT (process died without reporting)")

    print(f"leader process: exitcode={leader_p.exitcode}")
    if leader_result:
        print(
            f"  leader_status={leader_result['leader_status']} "
            f"elapsed={leader_result.get('leader_elapsed_s')}s"
        )
        if leader_result.get("leader_msg"):
            print(f"  leader_msg={leader_result['leader_msg'][:200]!r}")
        print(f"  inner_exitcodes={leader_result.get('inner_exitcodes')}")
        for tag, ir in (leader_result.get("inner_results") or {}).items():
            print(
                f"  [{tag}] status={ir['status']} elapsed={ir.get('elapsed_s')}s"
                + (f" msg={ir['msg'][:150]!r}" if ir.get("msg") else "")
            )
    else:
        print("  NO_RESULT (leader died without reporting)")

    # Headline verdict
    print("-" * 70)
    if leader_p.exitcode == 0 and leader_result is not None:
        verdict = "LEADER_SURVIVED (good — caller can catch & evict)"
    elif leader_result is None:
        verdict = "LEADER_DIED_SILENT (bad — production-style cascade)"
    else:
        verdict = (
            f"LEADER_EXITED_NONZERO ec={leader_p.exitcode} "
            f"(reported={'yes' if leader_result else 'no'})"
        )
    print(f"VERDICT: {verdict}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="tp2_init_clean",
        choices=[
            "tp2_init_clean",
            "tp2_init_victim_dies_during_bootstrap",
            "tp2_init_victim_dies_during_destroy",
            "tp2_broadcast_victim_dies",
            "tp2_collective_rpc_inner_raises",
            "all",
        ],
    )
    parser.add_argument("--watchdog-s", type=float, default=180.0)
    args = parser.parse_args()

    if args.mode == "all":
        for m in [
            "tp2_init_clean",
            "tp2_init_victim_dies_during_bootstrap",
            "tp2_init_victim_dies_during_destroy",
            "tp2_broadcast_victim_dies",
            "tp2_collective_rpc_inner_raises",
        ]:
            run_mode(m, args.watchdog_s)
            print()
    else:
        run_mode(args.mode, args.watchdog_s)


if __name__ == "__main__":
    main()
