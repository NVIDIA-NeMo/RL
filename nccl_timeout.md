# NCCL Timeout During CUDA Graph Warmup in MoE RL Training

## Symptom

After several successful GRPO training steps (anywhere from step 5 to step 20+), the job crashes with NCCL collective operation timeouts during the generation phase. The errors look like:

```
[Rank 5] Watchdog caught collective operation timeout:
  WorkNCCL(SeqNum=10707, OpType=COALESCED, ..., Timeout(ms)=600000)
  ran for 600030 milliseconds before timing out.
```

Key signatures:
- Different ranks report **different NCCL operation types** (ALLTOALL_BASE, REDUCE_SCATTER_BASE, ALLREDUCE, COALESCED) -- a collective mismatch
- The crash always happens during `cuda graph warmup` at the start of generation
- A new NCCL communicator is lazily initialized at the failing step (`NCCL version 2.27.5+cuda12.9` printed mid-warmup)
- Steps 1 through N-1 complete normally; the crash is non-deterministic

## Background: The Training-Inference Cycle

In the RL training loop, each step does:

```
generate() {
    _wake()          // resume inference engine (realloc KV cache, rebuild CUDA graphs)
    <submit requests, run generation>
    _sleep()         // suspend inference engine (dealloc KV cache, delete CUDA graphs)
}
```

With `static_kv_memory_pointers=false` and `kv_cache_management_mode=recompute`, every suspend/resume cycle **destroys and recreates CUDA graphs**. Graph warmup runs forward passes through the model, which for MoE models includes NCCL alltoall collectives across expert-parallel (EP) ranks. All EP/TP ranks must execute the same sequence of NCCL operations in lockstep during this warmup.

## Architecture: Two Communication Systems on One Event Loop

The `DynamicInferenceEngine` runs an async engine loop on a dedicated event loop thread. This single event loop handles two different communication systems:

| System | Purpose | Mechanism |
|--------|---------|-----------|
| **EP consensus** (`_ep_group_has_work`) | Coordinate EP ranks on work availability | Async ZMQ all-reduce |
| **CUDA graph warmup** (inside `resume()`) | Capture model forward passes into graphs | Blocking NCCL collectives |

Both run on the **same event loop thread**. This is the root of the problem.

## Root Cause

The engine loop has this structure (simplified from `run_engine_with_coordinator`):

```python
while True:
    self.schedule_requests()                          # read ZMQ messages
    ep_group_has_work = await self._ep_group_has_work(...)  # ZMQ all-reduce across EP ranks
    if not ep_group_has_work:
        if self.suspend_signal:
            self.suspend()      # no-op when already suspended
        else:
            self.resume()       # CUDA graph warmup -- blocks with NCCL!
        await asyncio.sleep(0.02)
```

When the coordinator sends `RESUME + UNPAUSE` to all engines, the signals arrive asynchronously. EP ranks process them at different times depending on ZMQ delivery and event loop scheduling. This leads to a **divergence**:

```
Rank A (received RESUME):  suspend_signal=False  -->  calls resume()  -->  NCCL alltoall BLOCKS event loop
Rank B (not yet received): suspend_signal=True   -->  calls suspend() (no-op)  -->  sleeps 20ms
```

On the next iteration, Rank B calls `_ep_group_has_work()` which does an async ZMQ all-reduce. This requires Rank A to respond. But Rank A's event loop is **blocked inside NCCL** (graph warmup forward pass). Rank A can never respond to ZMQ while NCCL is blocking its event loop.

**Deadlock: Rank A waits for Rank B in NCCL. Rank B waits for Rank A in ZMQ.**

After 10 minutes, the NCCL watchdog times out and kills the process.

### Why it's non-deterministic

The deadlock only occurs when at least one EP rank enters `resume()` before all other EP ranks have received the `RESUME` signal. When all ranks happen to process the signals within the same ~20ms engine loop cycle, they all enter `resume()` together and the warmup succeeds. This timing depends on ZMQ delivery, event loop scheduling, and OS thread scheduling -- hence the non-determinism.


### Implementation 1 (This causes delay of 25%)

```python
def _wake(self):
    # Phase 1: Unpause the engine loop (async, event loop stays free for ZMQ)
    asyncio.run_coroutine_threadsafe(self._unpause_engine(), self._inference_loop).result()

    # Phase 2: Synchronized resume on the main thread
    self._synchronized_resume()

async def _unpause_engine(self):
    # Send only UNPAUSE (not RESUME) -- keeps suspend_signal=True so the engine
    # loop never calls resume() on its own
    if torch.distributed.get_rank() == 0:
        self.inference_client.unpause_engines()
    await self.dynamic_inference_engine.running.wait()

def _synchronized_resume(self):
    engine = self.dynamic_inference_engine

    # Guard: replace suspend() with a no-op while we resume
    original_suspend = engine.suspend
    engine.suspend = lambda: None

    try:
        torch.distributed.barrier()        # all ranks ready
        engine.resume()                     # CUDA graph warmup (NCCL collectives)
        engine.suspend_signal = False       # let engine loop transition to normal mode
        torch.distributed.barrier()        # all ranks done
    finally:
        engine.suspend = original_suspend
```

### Why this works

**No event-loop blocking.** The NCCL barriers and `resume()` run on the main thread. The event loop thread continues running the engine loop, freely handling ZMQ communication for EP consensus. No rank's ZMQ is ever starved.

**No RESUME signal divergence.** We never send the `RESUME` header to the coordinator. Instead, we send only `UNPAUSE` (which restarts the engine loop) and keep `suspend_signal=True`. The engine loop sees `suspend_signal=True`, calls `suspend()` (no-op since already suspended), and idles. It never calls `resume()` on its own. We control exactly when `resume()` happens -- after the barrier on the main thread.

**No thread-safety race.** When `resume()` runs on the main thread, it sets `is_suspended=False`. Without the guard, the engine loop's next `suspend()` call (on the event loop thread) would read `is_suspended=False`, enter the suspend body, and **deallocate buffers while the main thread is still creating CUDA graphs**. The `suspend()` guard (replacing it with `lambda: None`) prevents this. The guard is removed only after `suspend_signal=False` is set, so the engine loop transitions to calling `resume()` (which is a no-op since we already resumed) instead of `suspend()`.

### Thread interaction timeline

```
Main Thread                              Event Loop Thread (engine loop)
-----------                              --------------------------------
_unpause_engine() ----sends UNPAUSE---->
                                         schedule_requests(): reads UNPAUSE
                                         _ep_group_has_work(): ZMQ all-reduce
                                         suspend_signal=True -> suspend() [no-op]
                                         asyncio.sleep(0.02)

barrier() ............all ranks sync.... (ZMQ continues running freely)

engine.suspend = no-op                   suspend() -> no-op [guarded]
engine.resume()                          (ZMQ continues, no GPU conflict)
  -> reinitialize buffers
  -> create_cuda_graphs() [NCCL]
engine.suspend_signal = False            (ZMQ continues)

barrier() ............all ranks done....

engine.suspend = original                suspend_signal=False -> resume() [no-op]
                                         (engine is ready for requests)
```

## Affected Configuration

This bug affects MoE models using the megatron generation backend with:
- `moe_token_dispatcher_type=alltoall` (NCCL alltoall inside CUDA graph warmup)
- `static_kv_memory_pointers=false` (CUDA graphs deleted/recreated each cycle)
- `kv_cache_management_mode=recompute` (full dealloc on suspend)
- `num_cuda_graphs > 0`
- Expert parallelism (EP) > 1

Dense models or configs with `static_kv_memory_pointers=true` are not affected because CUDA graphs are not recreated on resume.

### Implementation 2 
In dynamic_engine.py you set asyncio.sleep(0) instead of 0.02. This works 