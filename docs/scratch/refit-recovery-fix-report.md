# RefitWorker abort+recovery fix (FT 30B step 52)

## (a) Repro outcome

Built `docs/scratch/refit_recovery_repro.py` — drives a real `RefitWorker`
plus 3 `MockGenActor`s through a 4-phase sequence on a 4-GPU GB300 pod
(`docs/scratch/refit_recovery_pod.yaml`): clean refit → fault refit (kill
gen mid-broadcast) → abort+respawn+reinit → post-recovery refit.

Reproduces the **exact** production error (`GetTimeoutError: Get timed
out: some object(s) not ready`) and pinpoints two distinct bugs:

- **Bug A (the cascade that killed step 52):** after `ray.kill(refit_worker)`,
  surviving gen actors stay wedged in their pending NCCL `broadcast`
  (kernel waits for the dead peer; non-blocking init bounds rendezvous
  but not in-flight collectives). Their actor task queues are blocked.
  The next `init_collective.remote()` queues **behind** the wedge; the
  `ray.wait(..., timeout=45s)` rendezvous expires before the wedged op
  errors out (~60 s). All 6 retries fail, run dies.

- **Bug B (orthogonal cleanup hang):** `broadcast_weights_until_complete.finally`
  → `StatelessProcessGroup.destroy()` → `torch.cuda.synchronize()` /
  `comm.abort()` hangs even on the happy 4-rank path. Out of scope.

## (b) Call-graph trace of the timeout

```
grpo.py:1303  policy.broadcast_weights_for_collective(...)   # dispatches receiver_future
grpo.py:1306  policy_generation.update_weights_from_collective()
grpo.py:1309  ray.get(futures_train)         # ◀ NO TIMEOUT — surfaces as Ray-internal
                                               "Get timed out: some object(s) not ready"
              └── peer dies → cudaError on RefitWorker sync (refit_worker.py:290)
              ▼ except → grpo.py:1377 policy.abort_collective()  # ray.kill, set None
              ▼ grpo.py:1376 ensure_collective_synced(policy)
              ▼   ↳ policy.init_collective(...)  # spawns fresh RefitWorker
              ▼   ↳ self.init_collective(...)    # POST gen /init_collective
              ▼   ↳ ray.wait(all_futs, 45s)      # ◀ HERE: gen futs queued behind wedged
                                                 # broadcast on each gen actor → all 6 attempts fail
```

## (c) Per-candidate outcome

| Candidate                                 | Recovers? | Latency  | Notes                                                                       |
| ----------------------------------------- | --------- | -------- | --------------------------------------------------------------------------- |
| A: longer `ray.get` timeout               | No        | n/a      | Just shifts the wedge; gens still queue init_collective behind broadcast.   |
| B: eager respawn in `abort_collective`    | No        | n/a      | RefitWorker respawn (~5 s) was never the bottleneck.                        |
| **C: bounded `ray.get` + gen `reset_collective` + RefitWorker preflight `is_alive`** | **Yes** | **~13 s** | Survivors fully recover; clean post-recovery refit completes. |
| D: classify CUDA error severity           | n/a       | n/a      | `cudaErrorPeerAccessFailed` from peer SIGKILL is recoverable; gating blocks legit retries. |

## (d) The applied fix (no commit, no push)

Three surgical edits in the train-side abort/retry path:

```diff
# nemo_rl/algorithms/grpo.py — bound ray.get + force gen reset
-                    ray.get(futures_train)
-                    results = ray.get(futures_inference)
+                    ray.get(futures_train, timeout=600)
+                    results = ray.get(futures_inference, timeout=600)
                     ...
                     if hasattr(policy, "abort_collective"):
                         try: ray.get(policy.abort_collective(), timeout=30)
                         ...
+                    if hasattr(policy_generation, "reset_collective"):
+                        try:
+                            reset_futs = policy_generation.reset_collective()
+                            ray.wait(list(reset_futs), timeout=15.0)
+                        except Exception as reset_e:
+                            print(f"! gen reset_collective raised "
+                                  f"{type(reset_e).__name__}: {reset_e}")

# nemo_rl/models/policy/lm_policy.py — sync-kill + preflight is_alive
@@ abort_collective @@
+            stale_handle = self._refit_worker
-            ray.kill(self._refit_worker)
+            ray.kill(stale_handle)
+            try: ray.get(stale_handle.is_alive.remote(), timeout=10)
+            except Exception: pass

@@ broadcast_weights_for_collective @@
+        try: ray.get(self._refit_worker.is_alive.remote(), timeout=10)
+        except Exception as e:
+            raise RuntimeError(f"RefitWorker liveness failed: {e}") from e
```

**Why C works:** the bounded `ray.get` converts the Ray-internal wedge
into a clean `GetTimeoutError` the retry loop catches. `reset_collective`
gives the gen-side reset a head-start while gens finish unwedging
(~60 s); by the next `ensure_collective_synced` attempt the gens are
responsive. The `is_alive` preflight makes a stale RefitWorker handle
fail fast instead of wedging mid-broadcast.

Verified: phase 1 fault → phase 2 recovery → phase 3 clean refit,
**12-14 s end-to-end**, well under the 30 s target.

Files touched:
- `nemo_rl/algorithms/grpo.py`
- `nemo_rl/models/policy/lm_policy.py`

## (e) What's still uncovered

- **Bug B (destroy hang)** in `StatelessProcessGroup.destroy()` repros
  on every refit in the harness. Production masks it (inner `try` returns
  `True` before `finally` runs; sender doesn't gate on receiver fut) but
  it leaves the actor "busy" until the wedge clears, so the new
  `is_alive` preflight could spuriously timeout on back-to-back refits.
  Follow-up: bound `comm.abort()` in destroy or drop the in-finally
  destroy (next `init_collective` destroys it anyway).
- Harness uses raw `StatelessProcessGroup` gens, not vLLM. Recovery
  on the gen side may surface vLLM-specific issues (cudagraph cache,
  kv-cache reset) not modeled here. Recommended: rerun 30B FT with a
  fault injector at step 5 and confirm `broadcast_attempts==2` works.
- MNNVL needed `NCCL_MNNVL_ENABLE=0` on the test pod (no IMEX channel
  via DRA); production has it.

Pod cleaned up: `kubectl delete pod hemild-refit-recovery-pod` ✓
