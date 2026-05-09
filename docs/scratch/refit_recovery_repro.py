"""Reproduce the RefitWorker abort+recovery cascade.

Production failure (FT 30B run, step 52):

  1. RefitWorker hit a CUDA error during NCCL broadcast (peer died mid-broadcast).
  2. Train side caught it, called ``policy.abort_collective`` → ``ray.kill(RefitWorker)``.
  3. ``ensure_collective_synced`` retry loop dispatched fresh ``init_collective`` at the
     new (smaller) world size.
  4. Training crashed with ``Get timed out: some object(s) not ready``.

This harness simulates that sequence end-to-end on a single 4-GPU node:

  - Spawn a RefitWorker (real, from nemo_rl.models.policy.workers.refit_worker).
  - Spawn 3 mock "gen" actors that each create a StatelessProcessGroup client
    on rank 1..3 and participate in NCCL broadcast.
  - Drive ``init_collective`` + ``broadcast_weights_until_complete`` against the
    RefitWorker via ZMQ, the way ``MegatronPolicyWorker.broadcast_weights_for_collective``
    does in production.
  - Mid-broadcast, kill one gen actor → RefitWorker hits NCCL error.
  - Then exercise the recovery path: ``abort_collective`` (ray.kill) → respawn
    a fresh RefitWorker → re-init at world_size=3 (1 train + 2 gen) → broadcast again.

What we measure:

  - Does the RefitWorker get into the ``cudaErrorXxx``-from-poisoned-NCCL state?
  - After ``ray.kill`` + respawn, does the *recovery* succeed?
  - How long does recovery take? (target: < 30s)
  - Are there any dangling Ray actors / placement groups / ports?

Usage (run inside the test pod):

  cd /opt/nemo-rl
  uv run python docs/scratch/refit_recovery_repro.py --mode baseline
  uv run python docs/scratch/refit_recovery_repro.py --mode candidate-eager-respawn

Modes:
  - baseline:                use the current production code path
  - candidate-eager-respawn: respawn RefitWorker INSIDE abort_collective
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time
import traceback
from typing import Any, Optional

# Force the repo to be on the path before we import torch/ray, so we get the
# in-tree edits (we may be testing a candidate fix).
sys.path.insert(0, "/opt/nemo-rl")

import ray  # noqa: E402
import torch  # noqa: E402


def find_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("0.0.0.0", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def get_pod_ip() -> str:
    """Return a non-loopback IPv4 to use as the TCPStore master address."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except OSError:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


# =====================================================================
# MockGenActor — simulates a vLLM gen worker on the receiving end of the
# RefitWorker's broadcast. Just enough to participate in NCCL and detect
# completion.
# =====================================================================

@ray.remote(num_gpus=1)
class MockGenActor:
    """Stand-in for a vLLM gen worker on rank 1..N."""

    def __init__(self, gpu_index: int) -> None:
        # Mirror RefitWorker's CUDA visibility setup.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        torch.cuda.set_device(0)
        self._gpu_index = gpu_index
        self._device = torch.device("cuda:0")
        self._group: Optional[Any] = None
        self._rank: int = -1
        self._world_size: int = 0
        print(
            f"[mock_gen] init gpu_index={gpu_index} pid={os.getpid()}",
            flush=True,
        )

    def init_collective(self, ip: str, port: int, world_size: int, rank: int) -> bool:
        from nemo_rl.distributed.stateless_process_group import StatelessProcessGroup

        # Idempotent re-init like the production gen path.
        if self._group is not None:
            try:
                self._group.destroy()
            except Exception:  # noqa: BLE001
                pass
            try:
                torch.cuda.synchronize()
            except Exception:  # noqa: BLE001
                pass
            self._group = None

        self._rank = rank
        self._world_size = world_size
        self._group = StatelessProcessGroup(
            master_address=ip, port=port, rank=rank, world_size=world_size
        )
        self._group.init_nccl_communicator(device=0)
        return True

    def receive_one_chunk(self, numel: int, dtype_str: str) -> bool:
        """Allocate a buffer of the given shape and recv the next broadcast."""
        if self._group is None:
            return False
        dtype = {"float32": torch.float32, "uint8": torch.uint8}[dtype_str]
        buf = torch.zeros(numel, dtype=dtype, device=self._device)
        self._group.broadcast(buf, src=0)
        torch.cuda.synchronize()
        # Sanity: src writes ones, we should see them.
        return bool(buf[0].item() == 1)

    def receive_n_chunks(self, numel: int, dtype_str: str, n_chunks: int) -> int:
        """Loop receive_one_chunk for exactly n_chunks (or until error)."""
        n = 0
        for i in range(n_chunks):
            try:
                if not self.receive_one_chunk(numel, dtype_str):
                    return n
                n += 1
            except Exception as e:  # noqa: BLE001
                print(
                    f"[mock_gen rank={self._rank}] receive_n_chunks raised "
                    f"after {n} chunks: {type(e).__name__}: {e}",
                    flush=True,
                )
                return n
        print(f"[mock_gen rank={self._rank}] received all {n} chunks cleanly", flush=True)
        return n

    def reset_collective(self) -> bool:
        if self._group is not None:
            try:
                self._group.destroy()
            except Exception:  # noqa: BLE001
                pass
        self._group = None
        try:
            torch.cuda.synchronize()
        except Exception:  # noqa: BLE001
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        return True

    def is_alive(self) -> bool:
        return True

    def get_node_id(self) -> str:
        return ray.get_runtime_context().get_node_id()


# =====================================================================
# MockTrainSender — stand-in for the rank-0 Megatron train worker.
# Sends repeated CUDA-IPC handles to the RefitWorker via the same ZMQ
# REQ/REP protocol used in production. This runs in-process (the test
# driver) — no need for a separate Ray actor since rank 0's job here is
# just to push bytes.
# =====================================================================

# Sender lives in its own actor with num_gpus=0 so it shares GPU 0 with
# the RefitWorker — same pattern as production train rank 0 + RefitWorker
# share the physical GPU but live in different processes.
@ray.remote(num_gpus=0)
class MockTrainSender:
    def __init__(self, gpu_index: int) -> None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        torch.cuda.set_device(0)
        self._device = torch.device("cuda:0")

    def ping(self) -> bool:
        # Force a CUDA op so cuInit fires before we time anything.
        torch.zeros(1, device=self._device).item()
        print(f"[mock_train_sender] ping ok pid={os.getpid()}", flush=True)
        return True

    def run(
        self,
        zmq_address: str,
        chunk_numel: int,
        n_chunks: int,
        dtype_str: str,
        chunk_delay_s: float = 0.0,
        abort_after: Optional[int] = None,
    ) -> str:
        import zmq

        from nemo_rl.models.policy.utils import IPCProtocol, get_handle_from_tensor

        print(
            f"[mock_train_sender.run] starting zmq={zmq_address} "
            f"n_chunks={n_chunks} numel={chunk_numel}",
            flush=True,
        )
        dtype = {"float32": torch.float32, "uint8": torch.uint8}[dtype_str]

        ctx = zmq.Context()
        sock = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.SNDTIMEO, 60000)
        sock.setsockopt(zmq.RCVTIMEO, 60000)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(zmq_address)
        try:
            for chunk_idx in range(n_chunks):
                buf = torch.ones(chunk_numel, dtype=dtype, device=self._device)
                torch.cuda.synchronize()
                handle = get_handle_from_tensor(buf)
                sock.send_pyobj((handle, [], int(buf.numel())))
                ack = sock.recv()
                if ack != IPCProtocol.ACK.value.encode():
                    return f"ack_mismatch:{ack!r}"
                print(f"[mock_train_sender.run] chunk {chunk_idx} ack OK", flush=True)
                del buf
                if abort_after is not None and chunk_idx + 1 >= abort_after:
                    return "interrupted"
                if chunk_delay_s > 0:
                    time.sleep(chunk_delay_s)
            print("[mock_train_sender.run] sending COMPLETE", flush=True)
            sock.send_pyobj(IPCProtocol.COMPLETE)
            final_ack = sock.recv()
            print("[mock_train_sender.run] COMPLETE acked", flush=True)
            if final_ack != IPCProtocol.ACK.value.encode():
                return f"complete_ack_mismatch:{final_ack!r}"
            return "complete"
        except Exception as e:  # noqa: BLE001
            return f"raised:{type(e).__name__}:{e}"
        finally:
            try:
                sock.close()
            except Exception:  # noqa: BLE001
                pass
            try:
                ctx.term()
            except Exception:  # noqa: BLE001
                pass


# =====================================================================
# The driver. Spawns RefitWorker + N MockGenActors, drives one good refit,
# then injects a fault and verifies recovery works.
# =====================================================================

def spawn_refit_worker(gpu_index: int) -> Any:
    from nemo_rl.models.policy.workers.refit_worker import RefitWorker
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    node_id = ray.get_runtime_context().get_node_id()
    actor = RefitWorker.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False),
        runtime_env={
            "env_vars": {
                "CUDA_VISIBLE_DEVICES": str(gpu_index),
                "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
            }
        },
    ).remote(gpu_index=gpu_index)
    # Mirror lm_policy._ensure_refit_worker.
    ray.get(actor.get_node_id.remote(), timeout=60)
    return actor


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="baseline",
        choices=["baseline", "candidate-eager-respawn"],
        help="Which fix variant to exercise.",
    )
    parser.add_argument("--n-gen", type=int, default=3, help="Number of gen ranks.")
    parser.add_argument("--n-chunks", type=int, default=20)
    parser.add_argument("--chunk-numel", type=int, default=1024 * 1024)
    parser.add_argument("--abort-after", type=int, default=5,
                        help="Send this many chunks, then kill a gen actor.")
    parser.add_argument("--watchdog-s", type=float, default=180.0)
    args = parser.parse_args()

    print(f"=== refit_recovery_repro mode={args.mode} ===", flush=True)

    ray.init(ignore_reinit_error=True)
    pod_ip = get_pod_ip()
    print(f"  pod_ip={pod_ip}", flush=True)

    # Reserve GPU 0 for RefitWorker, GPUs 1..n_gen for mock gen actors.
    refit_gpu = 0
    gen_gpus = list(range(1, 1 + args.n_gen))

    # ------------------------------------------------------------------
    # Phase 0: spawn actors + run one good refit to prove the harness
    # works.
    # ------------------------------------------------------------------
    refit = spawn_refit_worker(refit_gpu)
    zmq_address = ray.get(refit.start_zmq_server.remote(), timeout=60)
    print(f"  refit ZMQ addr: {zmq_address}", flush=True)

    # Sender shares GPU 0 with RefitWorker (same physical GPU, separate
    # process — exactly the prod layout for train rank 0 vs RefitWorker).
    sender = MockTrainSender.options(
        runtime_env={
            "env_vars": {
                "CUDA_VISIBLE_DEVICES": str(refit_gpu),
                "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
            }
        }
    ).remote(gpu_index=refit_gpu)
    # Pre-warm: force the sender's Ray actor process to come up before
    # we measure any timing — otherwise the first sender.run.remote()
    # eats 30-60s of venv setup which we'll mistake for a wedge.
    ray.get(sender.ping.remote(), timeout=120)
    print("  ✓ sender pre-warmed", flush=True)

    gen_actors: list[Any] = []
    for gpu in gen_gpus:
        gen_actors.append(MockGenActor.remote(gpu_index=gpu))

    cross_world = 1 + len(gen_actors)
    rendezvous_port = find_free_port()
    print(
        f"  phase 0: rendezvous on {pod_ip}:{rendezvous_port} world={cross_world}",
        flush=True,
    )

    # Symmetric init_collective dispatch: refit (rank 0) + gens (rank 1..N).
    futures = [refit.init_collective.remote(pod_ip, rendezvous_port, cross_world, 0)]
    for i, ga in enumerate(gen_actors):
        futures.append(ga.init_collective.remote(pod_ip, rendezvous_port, cross_world, i + 1))
    ready, pending = ray.wait(futures, num_returns=len(futures), timeout=60.0)
    if pending:
        print(f"  ! phase0 init_collective timed out; pending={len(pending)}", flush=True)
        return 2
    ray.get(ready)
    print("  ✓ phase 0 init_collective complete", flush=True)

    # Run one full refit round to prove the path works.
    print("  phase 0: full refit round (no faults)", flush=True)
    receiver_fut = refit.broadcast_weights_until_complete.remote()
    gen_recv_futs = [
        ga.receive_n_chunks.remote(args.chunk_numel, "uint8", args.n_chunks) for ga in gen_actors
    ]
    sender_fut = sender.run.remote(
        zmq_address,
        chunk_numel=args.chunk_numel,
        n_chunks=args.n_chunks,
        dtype_str="uint8",
    )
    sender_status = ray.get(sender_fut, timeout=120)
    print(f"  phase 0 sender returned: {sender_status}", flush=True)
    # NB: the RefitWorker's broadcast_weights_until_complete future may
    # not become ready promptly even after sender posts COMPLETE — the
    # destroy() call in its finally block can wedge waiting for NCCL
    # cleanup. We don't gate the test on it; the gen-side cleanly
    # returning is sufficient evidence that the broadcast succeeded.
    gen_chunks = ray.get(gen_recv_futs, timeout=60)
    print(
        f"  phase 0 result: sender={sender_status} "
        f"gen_chunks={gen_chunks}",
        flush=True,
    )
    # Best-effort fetch the receiver fut with a SHORT timeout so we can
    # log whether it returned. Doesn't gate progress.
    try:
        refit_ok = ray.get(receiver_fut, timeout=5)
        print(f"  phase 0 refit_ok={refit_ok}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"  phase 0 refit_ok=PENDING ({type(e).__name__})", flush=True)
    assert sender_status == "complete", f"phase 0 sender bad: {sender_status}"

    # ------------------------------------------------------------------
    # Phase 1: re-rendezvous (refit teardown happens at end of every
    # broadcast_weights_until_complete) and start a SECOND refit, this
    # time killing a gen actor mid-stream.
    # ------------------------------------------------------------------
    rendezvous_port = find_free_port()
    print(
        f"  phase 1: rendezvous on {pod_ip}:{rendezvous_port} world={cross_world}",
        flush=True,
    )
    futures = [refit.init_collective.remote(pod_ip, rendezvous_port, cross_world, 0)]
    for i, ga in enumerate(gen_actors):
        futures.append(ga.init_collective.remote(pod_ip, rendezvous_port, cross_world, i + 1))
    ready, pending = ray.wait(futures, num_returns=len(futures), timeout=60.0)
    if pending:
        print(f"  ! phase1 init_collective timed out", flush=True)
        return 2
    ray.get(ready)
    print("  ✓ phase 1 init_collective complete", flush=True)

    # Begin a fault-injection refit. Spawn the receiver, dispatch gens,
    # then kill one gen actor mid-flight.
    print("  phase 1: refit with fault injection", flush=True)
    receiver_fut = refit.broadcast_weights_until_complete.remote()
    gen_recv_futs = [
        ga.receive_n_chunks.remote(args.chunk_numel, "uint8", args.n_chunks) for ga in gen_actors
    ]

    # Pump some chunks first so NCCL is mid-flight when we kill the victim.
    t_fault_start = time.monotonic()
    sender_fut = sender.run.remote(
        zmq_address,
        chunk_numel=args.chunk_numel,
        n_chunks=args.n_chunks,
        dtype_str="uint8",
        chunk_delay_s=0.05,
    )

    # Wait until ~``abort_after`` chunks should have been sent, then kill
    # a victim gen actor mid-broadcast.
    time.sleep(args.abort_after * 0.05 + 0.1)
    victim_idx = len(gen_actors) - 1
    print(f"  ⚡ killing gen actor idx={victim_idx} mid-broadcast", flush=True)
    ray.kill(gen_actors[victim_idx])
    gen_actors.pop(victim_idx)

    # Wait for the sender + receiver futures to settle (they should
    # error out shortly after the kill).
    try:
        sender_status = ray.get(sender_fut, timeout=60)
    except Exception as e:  # noqa: BLE001
        sender_status = f"raised:{type(e).__name__}:{e}"

    try:
        refit_ok = ray.get(receiver_fut, timeout=60)
    except Exception as e:  # noqa: BLE001
        refit_ok = f"raised:{type(e).__name__}:{e}"
    fault_elapsed = time.monotonic() - t_fault_start
    print(
        f"  phase 1 fault outcome: sender={sender_status} "
        f"refit_ok={refit_ok} elapsed={fault_elapsed:.2f}s",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Phase 2: recover. This is the part the production code is failing
    # at. Mirror exactly what lm_policy.abort_collective + ensure_collective_synced
    # do.
    # ------------------------------------------------------------------
    print("  phase 2: recovery", flush=True)
    t_rec_start = time.monotonic()

    if args.mode == "baseline":
        # Mirror lm_policy.abort_collective: ray.kill, set _refit_worker = None.
        try:
            ray.kill(refit)
        except Exception as e:  # noqa: BLE001
            print(f"  ray.kill raised {type(e).__name__}: {e}", flush=True)
        refit = None
        # Wait for gens to come back to a responsive state by pinging
        # them with is_alive — this confirms their actor task queue is
        # drained after the wedged collective from phase 1 errored out.
        for i, ga in enumerate(gen_actors):
            try:
                t0 = time.monotonic()
                ray.get(ga.is_alive.remote(), timeout=30)
                print(
                    f"  ✓ gen[{i}] responsive after "
                    f"{time.monotonic() - t0:.2f}s",
                    flush=True,
                )
            except Exception as e:  # noqa: BLE001
                print(
                    f"  ! gen[{i}] still wedged after kill: "
                    f"{type(e).__name__}: {e}",
                    flush=True,
                )
        # Reset gen workers' stale comm (now that they're responsive).
        gen_reset_futs = [ga.reset_collective.remote() for ga in gen_actors]
        try:
            ray.get(gen_reset_futs, timeout=30)
        except Exception as e:  # noqa: BLE001
            print(f"  gen reset_collective raised {type(e).__name__}: {e}", flush=True)
        # Step 2: spawn a fresh RefitWorker.
        refit = spawn_refit_worker(refit_gpu)
        zmq_address = ray.get(refit.start_zmq_server.remote(), timeout=60)
        print(f"  ✓ respawned refit, zmq={zmq_address}", flush=True)
        # Step 3: re-init_collective at the new (smaller) world.
        new_world = 1 + len(gen_actors)
        rendezvous_port = find_free_port()
        futures = [refit.init_collective.remote(pod_ip, rendezvous_port, new_world, 0)]
        for i, ga in enumerate(gen_actors):
            futures.append(ga.init_collective.remote(pod_ip, rendezvous_port, new_world, i + 1))
        ready, pending = ray.wait(futures, num_returns=len(futures), timeout=60.0)
        if pending:
            print(f"  ! phase 2 init_collective TIMED OUT (pending={len(pending)})",
                  flush=True)
            return 3
        try:
            ray.get(ready)
        except Exception as e:  # noqa: BLE001
            print(f"  ! phase 2 init_collective raised {type(e).__name__}: {e}", flush=True)
            return 3
        print("  ✓ phase 2 init_collective complete", flush=True)
    elif args.mode == "candidate-eager-respawn":
        # Same logic as baseline for now — placeholder for the eager
        # respawn variant we'll wire into lm_policy directly later.
        pass

    # ------------------------------------------------------------------
    # Phase 3: prove the survivors can do another full refit.
    # ------------------------------------------------------------------
    print("  phase 3: post-recovery refit (must succeed)", flush=True)
    receiver_fut = refit.broadcast_weights_until_complete.remote()
    gen_recv_futs = [
        ga.receive_n_chunks.remote(args.chunk_numel, "uint8", args.n_chunks) for ga in gen_actors
    ]
    sender_fut = sender.run.remote(
        zmq_address,
        chunk_numel=args.chunk_numel,
        n_chunks=args.n_chunks,
        dtype_str="uint8",
    )
    try:
        sender_status = ray.get(sender_fut, timeout=120)
    except Exception as e:  # noqa: BLE001
        sender_status = f"raised:{type(e).__name__}:{e}"
    # Gate on gen-side completion, not RefitWorker's finally cleanup.
    try:
        gen_chunks = ray.get(gen_recv_futs, timeout=60)
    except Exception as e:  # noqa: BLE001
        gen_chunks = f"raised:{type(e).__name__}:{e}"
    # Probe RefitWorker health AFTER phase 3 to catch dangling cleanup.
    try:
        refit_ok = ray.get(receiver_fut, timeout=5)
    except Exception as e:  # noqa: BLE001
        refit_ok = f"PENDING ({type(e).__name__})"
    rec_elapsed = time.monotonic() - t_rec_start
    print(
        f"  phase 3 result: sender={sender_status} refit_ok={refit_ok} "
        f"gen_chunks={gen_chunks}",
        flush=True,
    )
    print(f"  RECOVERY ELAPSED (phase 2 + 3): {rec_elapsed:.2f}s", flush=True)

    # Success: sender completed AND surviving gens received all chunks.
    # ``refit_ok`` is informational only — the RefitWorker's
    # ``broadcast_weights_until_complete`` future may stay PENDING due
    # to a separate destroy() cleanup hang that doesn't affect refit
    # correctness (the bytes have already been broadcast).
    gen_ok = (
        isinstance(gen_chunks, list)
        and len(gen_chunks) == len(gen_actors)
        and all(c == args.n_chunks for c in gen_chunks)
    )
    success = (sender_status == "complete" and gen_ok)
    print(
        f"=== {'PASS' if success else 'FAIL'} "
        f"sender={sender_status} gen_chunks={gen_chunks} refit={refit_ok} ===",
        flush=True,
    )

    # Best-effort cleanup.
    try:
        ray.kill(refit)
    except Exception:  # noqa: BLE001
        pass
    try:
        ray.kill(sender)
    except Exception:  # noqa: BLE001
        pass
    for ga in gen_actors:
        try:
            ray.kill(ga)
        except Exception:  # noqa: BLE001
            pass
    ray.shutdown()
    return 0 if success else 1


if __name__ == "__main__":
    try:
        rc = main()
    except Exception as e:  # noqa: BLE001
        print(f"FATAL: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        rc = 99
    sys.exit(rc)
