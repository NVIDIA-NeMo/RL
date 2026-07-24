# Multi-LoRA SFT recipe (standalone, 1 node x 8 GPU)

Train several LoRA adapters **concurrently** on one base model
(Nemotron-3-Nano-30B-A3B), each adapter on its own dataset, with per-adapter
optimizers — and verify the result is equivalent to training each adapter alone.

Everything runs from this checkout: `nemo_rl/models/multi_lora/` carries the
implementation, `patches/automodel/` carries the four NeMo-Automodel
integration files the launcher overlays into the container, and
`examples/configs/recipes/multi_lora/ray.sub` is the cluster-adapted
(pmi2) Ray bootstrap. No external checkout is imported; `scripts/audit_multi_lora_standalone.py`
fails closed if that ever regresses.

## Configs

```
base.yaml                  shared config (validated battery values)
parity100_<mode>_<who>.yaml  2-8 line overlays, mode={noclip,clip1},
                             who={multi,single_a..d}
smoke_10step_multi.yaml    10-step functional check
```

`base.yaml` is byte-equivalent to the configuration validated by the
100-step parity battery; overlays only change run identity + the knob under
test. A new experiment is:

```yaml
defaults: base.yaml
run_name: my_experiment        # -> results/my_experiment_{ckpt,logs}
policy:
  max_grad_norm: 1.0           # example knob
```

Key knobs (see header comment in `base.yaml`): `policy.max_grad_norm`,
`multi_lora.enabled`, `single_dataset`, per-adapter LoRA/optimizer settings
under `multi_lora.adapters[]`.

## Running

Single run:

```bash
sbatch --export=ALL,CFG=examples/configs/recipes/multi_lora/smoke_10step_multi.yaml \
    -J ml-smoke examples/configs/recipes/multi_lora/sft_8gpu_native.slurm
```

Full 10-run parity battery (needs canonical exact-init LoRA shards so every
run starts from identical adapter weights):

```bash
scripts/submit_parity_battery.sh          # CANON=<shards dir> to override
```

Success is `SFT_DONE_OK` in the run's `ray-head.log` (`sacct` state is not
reliable on this cluster). Per-adapter loss traces land in
`results/<job>/diag_loss_trace/rank_*.jsonl`.

## Verifying equivalence

```bash
python scripts/analyze_pairwise_loss.py \
    results/<multi-run> results/<single_a> results/<single_b> \
    results/<single_c> results/<single_d> \
    --output-prefix results/parity100_noclip --title "no clipping"
python scripts/plot_parity_loss_differences.py --prefix parity100
```

The analyzer joins trace rows by (optimizer step, DP rank), requires exact
input SHA-256 + token-count alignment, and never sums losses across adapters.
The plots show per-adapter single-vs-multi curves plus an explicit |delta|
panel (0.1 yardstick).

## Launcher environment

`sft_8gpu_native.slurm` exports the validated runtime env (deterministic seed,
NCCL Ring, cuBLAS workspace pin). `NOUSNET_*` variables are the multi-LoRA
module's own knobs (names kept from the original campaign so runs stay
comparable):

```
NOUSNET_DIAG_ENABLED/_LOSS_TRACE/_TRACE_ONLY/_LORA_STEP/_WHO
                          per-adapter loss tracing (battery turns these on)
NOUSNET_INIT_EXPORT_DIR   write canonical LoRA init shards, then exit-equivalent
NOUSNET_INIT_IMPORT_DIR   read canonical shards (all runs of a battery)
NOUSNET_INIT_IMPORT_SLOT  which shard slot a single-LoRA run imports (0-3)
NOUSNET_PER_ADAPTER_GRAD_CLIP  clip each adapter's grads independently (multi)
NOUSNET_FORCE_PAD_TO      pad every microbatch to a fixed length (parity)
NOUSNET_DETERMINISTIC(_SEED)   torch deterministic algorithms + seed
```

Unit tests: `pytest tests/unit/models/multi_lora/ -q` (CPU-only, no GPU
needed for the routing/data/trace tests).
