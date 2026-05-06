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

(to fill in)

## Step 3 — Read `mtq.quantize` calibration ordering

(to fill in)

## Step 4 — Identify upstream NaN source vs calibration wiring bug

(to fill in)

## Step 5 — Fix and verify

(to fill in)
