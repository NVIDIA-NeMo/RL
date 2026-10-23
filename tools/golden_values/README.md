# Golden values for model bring-up

Check that NeMo-RL's Megatron training path trains a model the way plain
Megatron-Core does. Record a few optimizer steps with Megatron-Bridge and
Megatron-Core alone, then replay them through NeMo-RL and compare.

This catches bugs that train without errors. In September 2026, NeMo-RL installed
Megatron's gradient-sync callbacks on a config that Nemotron Omni models had
copied. Gradients were never reduced across GPUs, yet loss and reward still went
up (RL #3789, Megatron-Bridge #6067). A golden check shows that kind of bug as a
grad norm and weight updates that disagree with the reference.

## What a golden file holds

For a fixed, seeded token batch and N steps (default 2):

- **Per step:** the loss (global token mean) and the grad norm before clipping.
- **Per weight**, before the first step and after each step: a fingerprint of the
  full tensor under its HuggingFace name. The fingerprint is the sum, the sum of
  squares, and two signed projections. Weight names and fingerprints come from
  the Megatron-Bridge export, so they don't depend on the parallel layout: you
  can record at TP=4 and check at TP=8 with EP=16.
- **Settings and versions**, enough to rerun: the model, the optimizer, the
  provider overrides, the scheduler's model config, git SHAs, package versions
  and the GPU.

`nemo_rl/models/megatron/golden_values.py` holds the shared code: the batch, the
fingerprints, the file format and the comparison.

## Record (Megatron-Core reference)

`record_mcore_goldens.py` uses no NeMo-RL training code. It does the following:

1. Builds the model from the Megatron-Bridge provider with the HF weights.
2. Installs the gradient-sync callbacks on `get_model_config(model)`, as
   Megatron-Bridge's own `setup()` does.
3. Runs Megatron-Core's forward-backward schedule and the distributed optimizer.

Launch one process per GPU:

```bash
torchrun --nproc-per-node 4 tools/golden_values/record_mcore_goldens.py \
    --hf-model /path/to/hf_checkpoint --tp 2 --ep 2 \
    --global-batch-size 8 --seq-length 256 --steps 2 \
    --out goldens/my-model.json.gz
```

Each golden run turns these off: dropout, the MTP loss (the MTP layers stay),
the MoE auxiliary and z losses, and the expert-bias update. Their randomness or scaling differs between trainers
for reasons unrelated to gradient sync. Change one with
`--override FIELD=JSON` (the value is parsed as JSON).

The optimizer settings are deliberately unlike training. Megatron's distributed
optimizer only supports Adam. Adam's first step moves a weight by lr·g/(|g|+eps).
With the usual tiny eps that is lr·sign(g), so a near-zero gradient element moves
by a full ±lr in a direction set by numerical noise, and that noise differs
between parallel layouts.

- **eps:** golden runs use eps 1e-2, so the step is about (lr/eps)·g,
  proportional to the gradient.
- **lr:** by default it's chosen from the weight count so that a clipped step
  (`--clip-grad 1`) moves the weights by an RMS of 2e-3 (`--update-rms`). That is
  well above bf16 rounding of typical weights, for any model size.
- **Steps:** the default is one step. A second step (`--steps 2`) also checks the
  updated weights' loss. On a small random-weight model it amplifies numerical
  noise, so compare it loosely.

The batch is text only, which was enough to show the Omni bug.

## Check (NeMo-RL)

`check_nemo_rl_goldens.py` does the following:

1. Rebuilds the batch and verifies its hash against the golden file.
2. Starts NeMo-RL's Megatron `Policy` from `examples/configs/sft.yaml`, with the
   golden file's optimizer settings and overrides.
3. Trains the same number of steps with the NLL loss.
4. Compares the results with the golden file. It exits 1 on a mismatch and
   prints the worst relative error for each metric.

```bash
uv run python tools/golden_values/check_nemo_rl_goldens.py \
    --golden goldens/my-model.json.gz --hf-model /path/to/hf_checkpoint \
    --tp 4 --ep 2 --gpus-per-node 8
```

It also prints model-config fields that differ between the two runs, which is the
first place to look when a check fails.

## Tolerances

The default tolerances (relative) are stored in the golden file:

- initial weights: 1e-6. Both runs load the same checkpoint, so the weights
  before the first step match exactly; they did across layouts for both the
  truncated and the full Super 3.5 VL. A mismatch means the check run loaded a
  different model. NeMo-RL reuses its HF-to-Megatron conversion under
  `$NRL_MEGATRON_CHECKPOINT_DIR` (default `$HF_HOME/nemo_rl`), keyed by the
  checkpoint path, so after rewriting a checkpoint at the same path, point
  `NRL_MEGATRON_CHECKPOINT_DIR` somewhere fresh.
- loss: 5e-3
- grad norm: 5e-2
- weight update: 5e-2, size-weighted over all tensors in a step
- weight update per tensor: 0.5

The truncated Super 3.5 VL was recorded on 1 GPU and checked one step at a time,
over three runs on 2026-10-20:

| Layout | Loss | Grad norm | Updates | Worst tensor |
| --- | --- | --- | --- | --- |
| DP=2 | 3e-5 | 1e-4 | 1-2% | 9-20% |
| TP=2 with sequence parallelism | 2e-4 | 2e-4 | 4-5% | 22-38% |

The TP=2 noise is MoE routing flipping on bf16 near-ties under the different
numerics. The CI test therefore gives TP=2 looser weight tolerances.

The full public Nemotron 3.5 Super VL 120B
(`nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16`) was recorded and checked
on 8 GB200 nodes at TP=2, EP=16 (DP=16) on 2026-10-21. NeMo-RL came within 9e-5
(loss), 5e-3 (grad norm), 3% (updates) and 32% (worst tensor) of the reference,
which passes the defaults. The record took 10 minutes and the check 8.5. The golden
file is 9.3 MB compressed.

For a new model:

1. Record twice at two different layouts.
2. Run the checker on each file against the other's run to see the noise floor.
3. Set tolerances a few times above it with `--tolerance METRIC=VALUE`.

A missing gradient reduction moves the grad norm and the updates by far more than
layout noise.

## A small checkpoint with the real architecture

`make_truncated_checkpoint.py` copies a checkpoint's config, remote code and
tokenizer, edits the config, and saves random weights through Megatron-Bridge.
The weights have exactly the tensor names the bridge loads; the model's own HF
code doesn't always reproduce them. CI uses it to run a
production architecture on two GPUs. For example, a 4-layer, 8-expert Nemotron
3.5 Super VL:

```bash
torchrun --nproc-per-node 1 tools/golden_values/make_truncated_checkpoint.py \
    --hf-model nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16 --out /tmp/super-vl-tiny \
    --set 'llm_config.layers_block_type=["mamba","moe","attention","moe"]' \
    --set llm_config.n_routed_experts=8 --set llm_config.num_experts_per_tok=2
```

Keep the MTP block: the Megatron-Bridge Super 3.5 VL bridge requires exactly one.
Golden runs keep it and zero its loss instead.

## Bringing up a new model

1. Make sure Megatron-Bridge's `AutoBridge` loads the HF checkpoint.
2. Record goldens at a layout that fits.
3. Run the checker at the layout NeMo-RL recipes use.
4. Keep the golden file next to the checkpoint and link it from the model's
   recipe or bring-up notes.
