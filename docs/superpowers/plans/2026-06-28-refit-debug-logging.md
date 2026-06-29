# Refit Debug Logging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in, bounded refit diagnostics that locate the first dtype or value divergence between the DTensor policy and vLLM.

**Architecture:** A dependency-light utility module owns environment gating, representative-name selection, bounded fingerprints, and aggregate stats. The DTensor worker wraps the exact tensors consumed by IPC/collective refit, while the vLLM extension observes reconstructed inputs, accumulates `load_weights` results, and inspects exact-name destination parameters after loading.

**Tech Stack:** Python 3.13, PyTorch, pytest, OmegaConf/NeMo-RL configuration.

## Global Constraints

- Enable diagnostics only when `NRL_REFIT_DEBUG=1` (truthy aliases are accepted by the helper).
- Every diagnostic line starts with `[REFIT_DEBUG]`.
- Never print a complete tensor or copy a complete large tensor to CPU.
- Do not alter refit tensor values, order, generator consumption, or synchronization.
- Cover colocated IPC/ZMQ and non-colocated collective refit.
- Preserve unrelated changes in the dirty worktree.

---

### Task 1: Shared refit diagnostics

**Files:**
- Create: `nemo_rl/utils/refit_debug.py`
- Create: `tests/unit/utils/test_refit_debug.py`

**Interfaces:**
- Produces: `refit_debug_enabled() -> bool`
- Produces: `select_refit_debug_names(names: Iterable[str]) -> dict[str, str]`
- Produces: `tensor_fingerprint(tensor: torch.Tensor, max_samples: int = 16) -> str`
- Produces: `RefitDebugStats.observe_tensor(name, tensor)`, `observe_metadata(name, shape, dtype)`, `observe_loaded(names)`, and `format()`
- Produces: `debug_refit_tensors(iterator, phase, selected_names, rank, stats)` without changing yielded objects
- Produces: `log_refit_destinations(model, selected_names, rank)` for exact destination matches and explicit mapped/unresolved markers

- [ ] **Step 1: Write failing helper tests**

Create tests that require the public interfaces and assert the user-visible behavior:

```python
def test_select_refit_debug_names_is_bounded_and_deterministic():
    names = [
        "model.layers.4.mlp.gate.weight",
        "model.layers.0.mlp.gate.weight",
        "model.layers.0.mlp.e_score_correction_bias",
        "model.layers.0.mlp.experts.0.w1.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.embed_tokens.weight",
        "lm_head.weight",
    ]
    selected = select_refit_debug_names(reversed(names))
    assert selected["router_gate"] == "model.layers.0.mlp.gate.weight"
    assert selected["routing_bias"].endswith("e_score_correction_bias")
    assert len(selected) == 6


def test_debug_refit_tensors_preserves_identity_and_logs_selected(capsys):
    tensor = torch.arange(8, dtype=torch.float32)
    stats = RefitDebugStats()
    output = list(
        debug_refit_tensors(
            iter([("model.layers.0.mlp.gate.weight", tensor)]),
            phase="policy_payload_ipc",
            selected_names={"router_gate": "model.layers.0.mlp.gate.weight"},
            rank="policy:0",
            stats=stats,
        )
    )
    assert output[0][1] is tensor
    assert "[REFIT_DEBUG]" in capsys.readouterr().out
```

Also test disabled logging, stable/non-mutating fingerprints, fp32/bf16 byte summaries, loaded-name aggregation, exact destination logging, and unresolved mapped labels.

- [ ] **Step 2: Run the tests and confirm the expected RED state**

Run:

```bash
/home/larkz/.local/bin/uv run pytest tests/unit/utils/test_refit_debug.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'nemo_rl.utils.refit_debug'`.

- [ ] **Step 3: Implement the utility module**

Implement deterministic category patterns, bounded evenly spaced sampling, a short digest over at most 16 float32 sample values, sampled finite/non-finite statistics, and aggregate dtype counts/bytes. The wrapper must be structurally equivalent to:

```python
def debug_refit_tensors(iterator, *, phase, selected_names, rank, stats):
    for name, tensor in iterator:
        stats.observe_tensor(name, tensor)
        if name in selected_names.values():
            category = next(k for k, v in selected_names.items() if v == name)
            print(
                f"[REFIT_DEBUG] phase={phase} rank={rank} category={category} "
                f"name={name} shape={list(tensor.shape)} dtype={tensor.dtype} "
                f"device={tensor.device} {tensor_fingerprint(tensor)}"
            )
        yield name, tensor
```

`tensor_fingerprint` must return a marker rather than raising if sampling fails, so logging cannot hide a real refit exception.

- [ ] **Step 4: Run helper tests to confirm GREEN**

Run the Task 1 pytest command again.

Expected: all tests in `test_refit_debug.py` pass.

### Task 2: Wire diagnostics into policy and vLLM refit

**Files:**
- Modify: `nemo_rl/models/policy/workers/dtensor_policy_worker_v2.py:1490-1638`
- Modify: `nemo_rl/models/generation/vllm/vllm_backend.py:98-350`
- Modify: `tests/unit/models/policy/test_dtensor_worker_v2.py`
- Create: `tests/unit/models/generation/test_vllm_backend_debug.py`

**Interfaces:**
- Consumes: all Task 1 helper interfaces.
- Produces: four observable boundaries: `policy_metadata`, `policy_payload_ipc` or `policy_payload_collective`, `vllm_incoming`, and `vllm_destination`.
- Produces: one policy payload summary and one vLLM refit summary per update call.

- [ ] **Step 1: Add failing policy and vLLM behavior tests**

Policy tests exercise the shared wrapper around a real `dtensor_params_generator` result and verify that an fp32 tensor remains fp32 in the emitted payload log when the base dtype is bf16.

The vLLM-marked test constructs `VllmInternalWorkerExtension` with a small fake `model_runner.model` whose `load_weights` copies tensors and returns a set:

```python
result = extension._load_weights(
    [("model.layers.0.mlp.gate.weight", source_tensor)]
)
assert result == {"model.layers.0.mlp.gate.weight"}
assert extension._refit_debug_stats.loaded_count == 1
assert torch.equal(fake_model.gate, source_tensor)
```

Expected RED: `_load_weights` currently returns `None` and does not update refit debug stats.

- [ ] **Step 2: Run focused tests and confirm RED**

Run:

```bash
/home/larkz/.local/bin/uv run pytest tests/unit/models/policy/test_dtensor_worker_v2.py -k 'refit_debug or preserves_fp32' --automodel-only -q
/home/larkz/.local/bin/uv run pytest tests/unit/models/generation/test_vllm_backend_debug.py --vllm-only -q
```

Expected: the new assertions fail for missing boundary logging/stats and the missing `_load_weights` return value.

- [ ] **Step 3: Instrument DTensor metadata and payloads**

In `prepare_refit_info`, preserve the current `(shape, transfer_dtype)` schema, compute representative names from the completed metadata, and emit:

```python
print(
    f"[REFIT_DEBUG] phase=policy_metadata rank=policy:{self.rank} "
    f"category={category} name={name} shape={list(shape)} "
    f"source_dtype={source_dtype} transfer_dtype={target_dtype} "
    f"metadata_dtype={metadata_dtype}"
)
```

Wrap `self._all_params_generator()` with `debug_refit_tensors` before passing it to `stream_weights_via_ipc_zmq_impl` or `packed_broadcast_producer`. Emit the aggregate stats after the generator is exhausted.

- [ ] **Step 4: Instrument vLLM receive/load/destination boundaries**

`prepare_refit_info` selects the same representative incoming names. Each update method initializes fresh `RefitDebugStats`; `_load_weights` observes and logs its reconstructed input list before calling the actual loader, records a returned set when available, and returns that result unchanged. At successful update completion, print the aggregate summary and call:

```python
log_refit_destinations(
    self.model_runner.model,
    self._refit_debug_names,
    rank=f"vllm:{refit_debug_rank()}",
)
```

Exact-name parameters produce fingerprints. QKV/expert packed names without an exact destination match produce `status=mapped_or_unresolved`, not a false mismatch.

- [ ] **Step 5: Run focused tests to confirm GREEN**

Run both Task 2 pytest commands.

Expected: all selected tests pass or dependency-specific suites report only their pre-existing environment skip.

### Task 3: Enable and verify the 5-layer smoke recipe

**Files:**
- Modify: `exp/grpo-m3-5layers.yaml`

**Interfaces:**
- Consumes: `NRL_REFIT_DEBUG` in both Ray worker environments.
- Produces: a smoke run in which policy and vLLM processes both emit the four diagnostic boundaries.

- [ ] **Step 1: Enable the flag in the untracked smoke config**

Add:

```yaml
  dtensor_cfg:
    env_vars:
      PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True
      NRL_REFIT_DEBUG: "1"
```

and:

```yaml
    vllm_cfg:
      env_vars:
        NRL_REFIT_DEBUG: "1"
```

- [ ] **Step 2: Validate the merged config**

Run:

```bash
/home/larkz/.local/bin/uv run python -c 'from nemo_rl.utils.config import load_config; c=load_config("exp/grpo-m3-5layers.yaml"); assert c.policy.dtensor_cfg.env_vars.NRL_REFIT_DEBUG == "1"; assert c.policy.generation.vllm_cfg.env_vars.NRL_REFIT_DEBUG == "1"'
```

Expected: exit status 0.

- [ ] **Step 3: Run formatting and focused regression tests**

Run:

```bash
/home/larkz/.local/bin/uv run python -m ruff check nemo_rl/utils/refit_debug.py nemo_rl/models/policy/workers/dtensor_policy_worker_v2.py nemo_rl/models/generation/vllm/vllm_backend.py tests/unit/utils/test_refit_debug.py tests/unit/models/generation/test_vllm_backend_debug.py
/home/larkz/.local/bin/uv run pytest tests/unit/utils/test_refit_debug.py -q
git diff --check
```

Expected: commands exit 0 with no new lint or whitespace failures.

- [ ] **Step 4: Inspect the final diff and prepare run instructions**

Confirm only the plan, utility, focused tests, two refit integration files, and `exp/grpo-m3-5layers.yaml` changed for this task. The handoff must include the exact launcher command and a log extraction command using `rg '\[REFIT_DEBUG\]|Generation KL Error' <job>-logs/ray-driver.log`.
