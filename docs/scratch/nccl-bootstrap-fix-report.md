# NCCL bootstrap peer-death cascade — repro + fix

## Test rig

Single 4-GPU GB300 pod (`hemild-nccl-repro-pod`,
`nvcr.io/nvidian/nemo-rl:nightly`, `kai-scheduler`, no DRA — single-host
NCCL only). Driver script
`docs/scratch/nccl_repro.py` spawns 4 child
processes; victim rank schedules a deferred `os._exit(137)` after
entering `init_nccl_communicator`. Pod torn down after testing.

## Bug — confirmed

`StatelessProcessGroup.init_nccl_communicator` calls `Communicator.init`
in NCCL's default **blocking** mode. The underlying `ncclCommInitRank`
is a synchronous C call that ignores Python signals. When a peer dies
mid-rendezvous, all survivors **wedge indefinitely** (verified up to a
235s watchdog — Python `SIGALRM` handlers do not fire because the GIL
is held inside C).

Vanilla outcome (`world=4`, victim=rank 2, delay=0.5s):

| rank | outcome | elapsed |
| :--- | :--- | :--- |
| 0    | NO_RESULT (hung in `Communicator.init`) | >235s |
| 1    | NO_RESULT (hung in `Communicator.init`) | >235s |
| 2    | victim — `os._exit(137)`              | 0.5s  |
| 3    | NO_RESULT (hung in `Communicator.init`) | >235s |

`try/except RuntimeError` does not help — the call hangs, no exception
is raised. `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` does not apply to this
comm (uses `nccl-pyx` directly, not `torch.distributed`).

## Fix applied

`nemo_rl/distributed/stateless_process_group.py`:

- imports + `_NCCL_BOOTSTRAP_TIMEOUT` constant: lines 19–45
- non-blocking + raw-binding bootstrap in `init_nccl_communicator`:
  lines 107–122
- warmup async-error poll: lines 137–141
- new helper `_poll_raw_async`: lines 158–208
- new helper `_abort_raw_quietly`: lines 210–216

Switch to NCCL non-blocking mode for init (`NCCLConfig(blocking=False)`).
Bypass the high-level `Communicator.init` classmethod (its constructor
calls `comm_count` etc. on the still-in-progress comm and raises
`NCCLError(InvalidArgument)`). Call the raw binding directly, poll
`comm_get_async_error` in a 30s loop. On `Success` → wrap as
`Communicator(comm_ptr)`. On non-success / timeout → `comm_abort` and
raise `RuntimeError`.

```python
# Before:
self.nccl_communicator = Communicator.init(
    nranks=self.world_size, rank=self.rank, unique_id=unique_id,
)

# After:
cfg = NCCLConfig(blocking=False)
comm_ptr = _nccl_bindings.comm_init_rank_scalable(
    int(self.world_size), int(self.rank), 1, unique_id.ptr, cfg.ptr,
)
self._poll_raw_async(comm_ptr, phase="bootstrap",
                     timeout=_NCCL_BOOTSTRAP_TIMEOUT)
self.nccl_communicator = Communicator(comm_ptr)
```

## Outcomes with fix

Happy path (no victim):

| rank | status | elapsed |
| :--- | :----- | :------ |
| 0    | ok     | 5.18s   |
| 1    | ok     | 5.16s   |
| 2    | ok     | 5.17s   |
| 3    | ok     | 5.16s   |

Bootstrap kill (victim = rank 2, delay=0.5s):

| rank | status                                             | elapsed |
| :--- | :------------------------------------------------- | :------ |
| 0    | raised `RuntimeError("NCCL bootstrap timed out ... 30.0s")` | 32.74s |
| 1    | raised `RuntimeError("NCCL bootstrap timed out ... 30.0s")` | 32.73s |
| 2    | NO_RESULT (victim — `os._exit(137)`)               | 0.5s    |
| 3    | raised `RuntimeError("NCCL bootstrap timed out ... 30.0s")` | 32.75s |

Survivors stay alive as Python processes (would stay alive as Ray
actors). The callers in `RefitWorker.init_collective`
(`nemo_rl/models/policy/workers/refit_worker.py:124`) and
`vllm_backend.init_collective`
(`nemo_rl/models/generation/vllm/vllm_backend.py:55`) already wrap
`init_nccl_communicator` so the new `RuntimeError` propagates cleanly to
the router's per-worker eviction logic. Destroy-time peer death also
surfaces cleanly — the existing `destroy()` swallows comm errors;
verified with `--mode destroy_kill` (all survivors return `destroy_ok`
in 5s).

## What this fix doesn't cover

Peer death **during a steady-state broadcast** (post-init) is a separate
failure mode: NCCL's `comm_get_async_error` does not surface dead peers
fast enough for a Python-level poll to catch before
`cuda.synchronize()` wedges. Production already addresses this surface
via `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=60` and
`TORCH_NCCL_RETHROW_CUDA_ERRORS=0` (set in
`infra/nrl_k8s/examples/qwen3_4b_if_fault_tolerant.gb300.infra.yaml`
lines 73–76) plus the `RefitWorker.abort_collective` /
`destroy()` flow. Extending the non-blocking-poll treatment to
broadcast would require replacing every `cuda.synchronize` in
`packed_broadcast_consumer` / `packed_broadcast_producer` with a polled
wait — out of scope here.

## Considered alternatives

| approach | bootstrap-kill outcome | skipped because |
| :------- | :--------------------- | :-------------- |
| `try/except RuntimeError` around `init_nccl_communicator` | survivors hung | call doesn't raise; blocks |
| `NCCL_TIMEOUT` / `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` env vars | survivors hung | don't apply to `nccl-pyx` |
| Subprocess isolation around init | parent survives, child hangs (then killed at 60s) | works but bigger refactor; non-blocking gives faster, cleaner failure |
| **`NCCLConfig(blocking=False)` + raw bindings + poll** | **survivors raise `RuntimeError` in 30s** | adopted |
