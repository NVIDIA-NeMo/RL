# FP8 Memory Leak Fix — Ablation Results

Bisecting Anna Shors's two-commit fix (`620f90ee9` + `0fe5dd57d`) on branch `ashors/zhiyul/fp_memory_cleanup`. Repro per [issue #2003](https://github.com/NVIDIA-NeMo/RL/issues/2003): 8-node Qwen3-30B-A3B-Base, FP8, `local_qwen_fp8_8node.sh`, 5 steps.

All edits live in `nemo_rl/models/policy/workers/megatron_policy_worker.py`.

## The four fixes

### Fix A — Null out TE base global CUDA tensors

In `_clear_fp8_caches` (gated by `force_clear_fp8_caches=true`):

```python
import transformer_engine.pytorch.module.base as te_base
te_base._cublas_workspace = None
te_base._multi_stream_cublas_workspace = []
te_base._dummy_wgrads = {}
```

Targets persistent cuBLAS scratch buffers held by TE module-level globals (~33 MB).

### Fix B — Clear Megatron's persistent all-gather scratch buffer

In `_offload_after_refit`:

```python
try:
    from megatron.core.parallel_state import get_global_memory_buffer
    get_global_memory_buffer().buffer.clear()
except Exception:
    pass
```

Targets the all-gather scratch buffer that grows to the largest sequence length seen and never frees on its own (~200 MB per Anna's snapshot).

### Fix C — Clear RotaryEmbedding's `@lru_cache(maxsize=32)`

In `_offload_after_refit`:

```python
try:
    from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
    RotaryEmbedding.forward.cache_clear()
except Exception:
    pass
```

Each cache entry is a GPU tensor (concatenated sin/cos embedding). With training + logprob runs at different sequence lengths, the cache fills (32 entries × ~1.6 MB) and the tensors anchor large CUDA segments — claimed ~5.46 GB *reserved*.

### Fix D — Null out MoE token dispatcher reference cycle

In `_offload_after_refit`:

```python
try:
    for module in self.model.modules():
        if not hasattr(module, "token_dispatcher"):
            continue
        dispatcher = module.token_dispatcher
        if dispatcher is None:
            continue
        for attr in (
            "probs",
            "routing_map",
            "reversed_local_input_permutation_mapping",
            "local_probs",
            "local_map",
        ):
            if isinstance(getattr(dispatcher, attr, None), torch.Tensor):
                setattr(dispatcher, attr, None)
except Exception:
    pass
```

`MoETokenDispatcher` is a plain Python class (not `nn.Module`) so `self.model.modules()` skips it; access via the `token_dispatcher` attribute on `MoELayer`. With `recompute_mlp=True` + FP8, `te_checkpoint`'s backward stores `dispatcher.probs = routing_probs` under `enable_grad`, creating a reference cycle:
`_CheckpointFunctionBackward → ctx → run_function=mlp → mlp.token_dispatcher.probs → probs.grad_fn → … → _CheckpointFunctionBackward`.
Nulling the attrs breaks the cycle, freeing both the routing tensors (~175 MB) and the te_checkpoint ctx-saved layernorm outputs (~576 MB) → ~750 MB total.

## Methodology

- **V0** (leaky baseline): Anna's two commits reverted, FP8 default config.
- **V1** (full fix): HEAD with all four fixes active.
- **V1-X** (leave-one-out): all four fixes active *except* fix X. Gates added inline:
  ```python
  if os.getenv("NRL_FP8_FIX_A", "1") == "1":
      ...  # fix A block
  ```

`NRL_FORCE_REBUILD_VENVS=true` was set on every run because the v0.5.0 container's per-worker venv fingerprint had drifted from the branch's `pyproject.toml`. No other workarounds were needed.

## Results

`GPU Memory after refit complete` lines, sampled rank pairs, GB allocated / GB reserved:

| Step | V0 (no fix) | V1 (full fix) | V1-A (TE off) | V1-B (gmem off) | V1-C (rotary off) | V1-D (MoE off) |
|---|---|---|---|---|---|---|
| 1 | 3.78 / 4.21 | 2.80 / 3.11 | 4.90 / 5.45 | 4.98 / 5.54 | 5.46 / 6.07 | 3.71 / 4.12 |
| 2 | 8.92 / **21.50** | 0.09 / 0.18 | 0.24 / 0.42 | 0.18 / 0.41 | 0.10 / 0.88 | 0.12 / 0.34 |
| 3 | 8.23 / **25.22** | 0.09 / 0.21 | 0.24 / 0.41 | 0.18 / 0.38 | 0.11 / 1.34 | 0.53 / **9.66** ⚡ |
| 4 | 8.95 / **25.35** | 0.09 / 0.18 | 0.24 / 0.59 | 0.18 / 0.60 | 0.12 / 2.62 | 0.12 / 0.38 |
| 5 | similar | 0.09 / 0.16 | 0.24 / 0.41 | 0.18 / 0.53 | 0.12 / **3.04** | 0.15 / 0.41 |

V1-D step 3's 9.66 GB is rank 32; other ranks at the same step were 0.34 GB. The MoE recompute reference cycle produces non-deterministic per-rank fragmentation.

Wall time: V0 = 22:43, V1 = 22:38. **No perf regression** from the fix (vs +33% in the issue's "Expandable" config).

## Verdicts

| Fix | Step 5 reserved when removed | Load-bearing? | Action |
|---|---|---|---|
| A — TE globals | 0.41 GB (flat) | No | Drop |
| B — gmem buffer | 0.53 GB (flat) | No | Drop |
| C — RotaryEmbedding | **3.04 GB, monotonic ↗** | Yes | Keep |
| D — MoE dispatcher | 0.41 GB avg, **9.66 GB tail spike** | Yes (worst rank) | Keep |

## Recommended minimal merge

Keep only fixes **C** (RotaryEmbedding LRU clear) and **D** (MoE dispatcher null-out). Drop fixes A and B and all debug instrumentation:
- `_iter_params_call_count` debug counter (~lines 2148–2151)
- `_record_memory_history` / `_dump_snapshot` / `_offload_after_refit_count` block (~lines 2430–2447)

Final diff is roughly half the size of Anna's full commit while preserving the entire memory benefit observed in this 5-step window.

**Caveat**: 5 steps may not fully expose A/B if their effect compounds over many more steps. If conservatism is preferred, keep all four — V1 baseline is 0.16 GB reserved either way. The decision is between "minimal diff" and "belt-and-suspenders".

## Job IDs (for reference)

- V0: 11607373 (COMPLETED 22:43)
- V1: 11610040 (COMPLETED 22:38)
- V1-A: 11610868
- V1-B: 11610869 (port-conflict failure) → 11611549 (retry, COMPLETED)
- V1-C: 11610870
- V1-D: 11610872

Container: `nvcr.io#nvidia/nemo-rl:v0.5.0.squashfs`. Driver: `local_qwen_fp8_8node.sh` with `FP8=true [FORCE_CLEAR_FP8_CACHES=true]`.
