# Nano3 W4A4 NaN-amax debug log

Job 157621 (W4A4 NVFP4 default + SFT loss, branch `mxin/moe-mamba-sft`,
container `nemo-rl-moe-mamba-py3.13.13-latest.sqsh`) failed in 4:10
with:

```
File "modelopt/torch/quantization/plugins/vllm.py", line 469
    A = self.w13_input_quantizer(A)
File "modelopt/torch/quantization/calib/max.py", line 69
    assert not torch.any(torch.isnan(local_amax)), ...
AssertionError: detected nan values in amax. nan in original tensor: True
```

Working hypothesis from observation: the assertion fires inside
`mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)`, on
`MaxCalibrator.collect`. *If calibration is actually running real data
through the model*, the activation reaching `w13_input_quantizer` cannot
legitimately be NaN — so either:

A. `calibrate_loop` is feeding random/uninitialized data instead of the
   real `quant_calib_data`.
B. The calibration is invoked before model weights are initialized
   (e.g., on a CUDA-graph dummy buffer left from `compile_or_warm_up_model`).
C. Some upstream layer in the BF16-kept set is producing NaN under the
   actual calibration input — a real numerical bug, not a calibration
   wiring bug.

We rule them out one at a time by reading code, not by guessing or
patching.

## Step 1 — Identify the calibration entry point

Stack frame at `vllm_quant_patch.py:72`:

```
compile_or_warm_up_model
  → _fakequant_run_prolog_worker(self)
    → mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)
```

So calibration is owned by `_fakequant_run_prolog_worker` in
`nemo_rl/modelopt/models/generation/vllm_quant_patch.py`. We need to
read that function to know what `calibrate_loop` actually does, what
data it feeds, and at what point in vLLM's lifecycle it runs.

## Step 2 — Read `_fakequant_run_prolog_worker`

`nemo_rl/modelopt/models/generation/vllm_quant_patch.py`:

```python
def _fakequant_run_prolog_worker(self) -> None:
    def calibrate_loop(model: Any = None) -> None:
        self.model_runner._dummy_run(1, skip_eplb=True, remove_lora=False)

    quant_cfg = resolve_quant_cfg(os.environ["VLLM_QUANT_CFG"])
    model = self.model_runner.model
    if hasattr(model, "unwrap"):
        model = model.unwrap()
    with disable_compilation(model):
        mtq.quantize(model, quant_cfg, forward_loop=calibrate_loop)
    ...
    # we are using dummy data for calibration, we expect the amax is loaded from the actor
    for name, module in model.named_modules():
        if isinstance(module, TensorQuantizer) and module.is_enabled:
            module._is_active = True
            if module.amax is not None:
                module.amax.fill_(-1.0)
            if name.endswith("weight_quantizer"):
                module.disable()
```

Key facts:
- `forward_loop=calibrate_loop` is `self.model_runner._dummy_run(1, ...)`,
  i.e. vLLM's internal dummy forward. **Calibration uses dummy data,
  not real `quant_calib_data`.**
- After calibration, every enabled quantizer's `amax` is overwritten
  with `-1.0` sentinel. The comment is explicit: *"we are using dummy
  data for calibration, we expect the amax is loaded from the actor"*.
- Conclusion: the prolog's only job is to **attach quantizer modules**
  to the vLLM graph. The numerical amax values it produces are
  thrown away. Real amax is delivered later by the Megatron policy
  worker via the QAT/refit weight-and-amax transfer.

## Step 3 — Read `MaxCalibrator.collect` and assertion provenance

`modelopt/torch/quantization/calib/max.py:53-77` (current):

```python
@torch.no_grad()
def collect(self, x):
    reduce_axis = quant_utils.convert_quantization_axis_to_reduce_axis(x, self._axis)
    local_amax = quant_utils.reduce_amax(x, axis=reduce_axis).detach()
    if x.device.type == "meta":
        self._calib_amax = local_amax
        return
    assert not torch.any(torch.isnan(local_amax)), (
        f"detected nan values in amax. nan in original tensor: {torch.any(torch.isnan(x))}"
    )
    ...
```

`git blame` on lines 65-77:
- 2025-04-15 (`9cb36cb43c3`): added the `meta`-device shortcut.
- 2026-02-17 (`590f9fc6622`, "Mamba MOE Quant Configs + Fix Export
  Bug" PR #882, Jenny Chen): **added the NaN-amax assertion**.

So the NaN assertion has existed since 2026-02-17. The assertion
fires on the *input tensor* x containing NaN: `reduce_amax(NaN) = NaN`,
then the assert blows. The diagnostic message confirms it
("`nan in original tensor: True`").

## Step 4 — Compare with the dirty branch's W4A4 run

The dirty branch's `mxin/moe-mamba-dirty` was used for the prior
W4A4 real run on **2026-05-04**, slurm job 156103. Its driver log:

```
.../156103-logs/ray-driver.log:14030
  model.layers.1.mixer.experts.w13_input_quantizer
  TensorQuantizer(... amax=0.0003 calibrator=MaxCalibrator quant)
```

That run **successfully calibrated** `w13_input_quantizer` with a
finite `amax=0.0003`. It did NOT crash.

The dirty run used:
- The **same** modelopt source (shared `MODELOPT_SOURCE_PATH`).
- The **same** `_fakequant_run_prolog_worker` calibration path (with
  the same NaN-amax assertion already in place since 2026-02-17).
- The **same** Nemotron-H-30B-A3B-BF16 weights.
- The **same** `NANO3_NVFP4_CFG` (semantically equivalent — `mtq.NVFP4_DEFAULT_CFG`
  + Nano3 BF16 layer disables).
- An **older container**: `nemo-rl-qarl-kv-cache-v3-latest.sqsh`
  (Python 3.12, older vLLM).

Job 157621 used the **new container**:
`nemo-rl-moe-mamba-py3.13.13-latest.sqsh` (Python 3.13.13, newer vLLM).

The traceback at the failure also shows new vLLM internals:
`vllm/model_executor/layers/fused_moe/modular_kernel.py:1753, 1537, 1375`,
`unquantized_fused_moe_method.py:328 forward_cuda`. The "modular kernel"
fused-MoE framework was introduced in newer vLLM and was not in the old
container's vLLM (different `_fused_experts` path).

**Conclusion**: the old container's `_dummy_run` produced finite
activations through the Nemotron-H MoE expert path; the new container's
`_dummy_run` produces NaN activations along the same path. This is a
behavioural change in vLLM's dummy-warmup forward — not in modelopt or
in our quant config.

## Step 5 — Localize the NaN source with a targeted print

Confirming Step 4 requires knowing **which expert call** first sees NaN
input `A` in the new vLLM. The `_invoke_fused_moe_quantized_function`
in `modelopt/torch/quantization/plugins/vllm.py` is where ModelOpt
intercepts the fused-MoE kernel; it gets `A` exactly once per expert
matmul invocation.

Patched at line 466 (DEBUG: clearly marked, to be reverted) to log
NaN/Inf/min/max of `A` if either NaN or Inf appears in it, with the
`w13` vs `w2` stage tag. If even the first MoE layer's first call is
NaN, the upstream chain (mamba/attention/layer-norm) is the source. If
NaN appears only at deeper layers, the source is between MoE layers.

Re-running the W4A4 job with this probe and reading the `[DBG-W4A4]`
lines from the driver log.

## Step 6 — Fix and verify

(to fill in once the probe identifies the upstream module)
