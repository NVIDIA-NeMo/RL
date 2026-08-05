# HybridEP Uneven Dispatch Safety Design

## Objective

Enable Megatron-LM's upstream uneven-dispatch protection whenever NeMo-RL selects the HybridEP flex dispatcher, then validate HybridEP with THD sequence packing on H100 and GB200 without PR #2964's NeMo-level input pre-padding or custom Megatron forks.

## Context

NeMo-RL main pins Megatron-Bridge `573e088c9c6740082c39744e03dc5b009e730ed4`, whose Megatron-LM pin `6513e3e23d6b5eda6a1c934990b15e804237732b` contains Megatron-LM PR #5008 (`81770cb015eab05785ecd540ba929d1400a52f67`). PR #5008 adds `moe_hybridep_pad_uneven_dispatch_inputs`, which pads HybridEP dispatcher inputs to the ETP×EP group maximum and a 64-token boundary, then trims the combined output to each rank's original token count.

The MCore option defaults to `False`. NeMo-RL workloads can produce variable token counts through rollout and packing, so NeMo-RL will adopt a correctness-first policy: HybridEP always enables this protection.

## Production Change

In `nemo_rl/models/megatron/setup.py::_apply_moe_config`, set:

```python
model_cfg.moe_hybridep_pad_uneven_dispatch_inputs = True
```

when `moe_flex_dispatcher_backend == "hybridep"`.

The setting is derived internally. No new YAML key is exposed, and activation does not depend on `sequence_packing.enabled`.

## Scope Boundaries

This workstream will not modify:

- `nemo_rl/models/megatron/data.py` or `train.py`
- `.gitmodules` or the Megatron-Bridge/Megatron-LM gitlinks
- `worker_groups.py` or `venvs.py`
- HybridEP topology selection
- canonical performance recipes

HybridEP dependency installation remains the responsibility of the existing dependency workstream. Cluster validation may overlay the pinned HybridEP wheel at commit `17cfb817bccec3a9c247013360cc550c2bac441e`, but that overlay is not a production-code change in this branch.

## Tests

Add focused unit coverage in `tests/unit/models/megatron/test_megatron_setup.py`:

- HybridEP backend sets `moe_hybridep_pad_uneven_dispatch_inputs` to `True`.
- A non-HybridEP backend preserves the model configuration's existing value.

The test must fail before the production change and pass afterward.

## Cluster Validation

Use Qwen3-30B-A3B with THD sequence packing for 20 training steps:

- CW H100: existing 4-node × 8-GPU performance topology.
- OCI-HSG GB200: existing 4-node × 4-GPU performance topology.

Both runs must use upstream Megatron-Bridge/Megatron-LM pins and HybridEP commit `17cfb817bccec3a9c247013360cc550c2bac441e`.

Success requires:

- the effective MCore flag is `True`;
- THD sequence packing and HybridEP flex dispatch are active;
- all 20 steps complete without hang, illegal memory access, OOM, or non-finite training metrics;
- logs record code, submodule, container, dependency, hardware, and job provenance.

## Performance Interpretation

Always enabling the flag can add a per-layer token-count collective and alignment padding even when ranks already receive equal token counts. H100 and GB200 results will therefore record step-time and throughput alongside correctness. Any measurable regression is a trade-off of the always-on safety policy, not evidence that NeMo-level pre-padding should be restored.
