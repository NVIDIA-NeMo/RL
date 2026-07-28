# Nemotron3 Super Sync HybridEP A/B Design

## Goal

Measure HybridEP against the recipe-default `alltoall` dispatcher for the
Nemotron3 Super 120B-A12B synchronous 32-node performance recipe on OCI-HSG.
The comparison must preserve the default recipe as the baseline and expose the
HybridEP arm through a separate, reusable configuration.

## Fixed Runtime

- NeMo-RL branch:
  `sna/qwen30-pr2964-f725-pin-oci-20260727`.
- NeMo-RL root:
  `ee21a7e23531a38b60acd4dfbdd6e8dab67179a8` before this config-only change.
- Megatron-Bridge:
  `483749cb773415f7608525838607dcefc62e4307`.
- Megatron-LM:
  `4d04e7625c5e84f984a9f01aef58cb006b0aa7ac`.
- DeepEP:
  `f725d29699f5bda9ba789456bb9579af69844685`.
- Container SHA256:
  `5e9f6066897057d8701e0722a5023c08a997f10f4eff61340c249ed73f7c33fc`.
- Hardware: 32 OCI-HSG nodes, 4 GB200 GPUs per node, 128 GPUs total,
  segment size 8.
- Both arms run 20 GRPO steps with checkpointing and W&B disabled,
  TensorBoard enabled, `NCCL_NVLS_ENABLE=0`, and padding telemetry disabled.

## Configurations

The baseline remains the existing, unmodified recipe:

`examples/configs/recipes/llm/performance/grpo-nemotron3-super-120BA12B-32n4g.yaml`

Add a separate HybridEP recipe:

`examples/configs/recipes/llm/performance/grpo-nemotron3-super-120BA12B-32n4g-hybridep.yaml`

The new recipe inherits the baseline recipe and changes only:

```yaml
policy:
  megatron_cfg:
    moe_token_dispatcher_type: flex
    moe_flex_dispatcher_backend: hybridep
    moe_hybridep_num_sms: 32
```

The model, tokenizer, batch sizes, sequence packing, parallelism, optimizer,
generation, validation cadence, cluster shape, and seed remain inherited from
the baseline.

## Reusable Profiles

Add two model profiles under
`scripts/experiments/oci-hsg/hybridep/models/`:

- `nemotron3-super-120ba12b-32n4g-sync.env` selects the unchanged baseline
  recipe.
- `nemotron3-super-120ba12b-32n4g-sync-hybridep.env` selects the explicit
  HybridEP recipe.

Both profiles define 32 nodes, 4 GPUs per node, segment size 8, 20 steps,
four-hour wall time, exact DeepEP f725, and `NCCL_NVLS_ENABLE=0`.

Both launches use `DISPATCHER_MODE=recipe`. This prevents the launcher from
adding hidden dispatcher overrides and makes the selected YAML file the only
dispatcher-control surface.

## Verification

Before submission:

1. Assert that the existing baseline recipe is unchanged.
2. Resolve both Hydra configurations and verify that the relevant dispatcher
   values are `alltoall` for baseline and `flex` plus `hybridep` plus 32 SMs
   for the HybridEP arm.
3. Verify that all non-dispatcher resolved configuration fields match after
   excluding expected run-name and checkpoint-path labels.
4. Verify both profile files select the intended recipe and identical cluster,
   step, time, DeepEP, and NCCL settings.
5. Run shell syntax, launcher regression tests, `uv lock --check`, and
   `git diff --check`.
6. Commit and push all source changes before submission.

## Submission

Submit both arms through the existing reusable launcher:

- baseline: baseline profile with `DISPATCHER_MODE=recipe`;
- HybridEP: HybridEP profile with `DISPATCHER_MODE=recipe`.

The launcher must pull the pushed branch, initialize recursive submodules,
select the highest user-level FairShare account, pass `sbatch --test-only`,
and record exact provenance before each real submission. Monitor both jobs for
at least five minutes and scan for actor loss, timeout, NCCL, CUDA, OOM,
NaN/Inf, and model-config failures.

## Metrics and Reporting

Success requires `COMPLETED 0:0`, 20/20 steps, and no fatal signature.
Compare matched steps 5–20:

- mean and median total step time;
- ratio-of-sums E2E tokens/second/GPU;
- policy-training time and tokens/second/GPU;
- policy/reference LogProb time and tokens/second/GPU;
- generation time and tokens/second/GPU;
- reward and generation KL error;
- validation accuracy and response length at steps 10 and 20.

Record job IDs, node lists, commits, config paths, terminal states, absolute
metrics, percentage changes, and short-run numerical caveats in the existing
secret-free HTML report. Twenty steps and two validation checkpoints establish
smoke-level consistency only, not long-horizon convergence.
