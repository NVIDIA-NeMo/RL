<style>
body, main, article, .markdown-body, .rendered_html, .jp-RenderedHTMLCommon, .jp-MarkdownOutput { max-width: none !important; width: 98% !important; }
table { width: 100% !important; }
</style>

# POLARIS buffered-rollout smoke

`run.sbatch` runs the native `examples/run_grpo.py` twice: update 1 with a
checkpoint deadline, then resume through update 5. `config.yaml` contains the
same parameters as attempt 004, inheriting the repository's Megatron config.
There is no environment installation, import audit, config-generation script,
or post-training validation in the launcher.

The fixed runtime is the existing DFW AMD64 `nemo-rl-v0.8.0.rc4` image and the
already-prepared Lustre `dependencies/nemo-rl-rc4-amd64/mcore/mcore-venv`.
Both commands use `uv run --frozen --no-sync --extra mcore`. Worker roles reuse
the image's prebuilt environments. No setup work runs in the GPU allocation.

Parameters remain POLARIS/Qwen3-1.7B, Omega per-sequence/inverse-length,
8 prompts × 8 responses, global batch 64, rollout multiplier 3, five updates,
temperature 1.0, and token limits 39936 generated / 40960 total / 1025 input.
The allocation is DFW interactive, one exclusive 8-H100 node, 128 CPUs, two hours,
no requeue. Training uses BF16, TP2/CP2/DP2, microbatch 1, sequence packing,
fused packed loss, padding alignment 8 and activation checkpointing. Generation
uses four TP2 replicas with Inductor, FULL_AND_PIECEWISE graphs, the existing
capture sizes, prefix caching, asynchronous scheduling and chunked prefill.
W&B and evaluation are off; TensorBoard and response logs remain enabled.

## Launch

Stage the clean checkout including its initialized submodules on DFW Lustre.
The launcher accepts two absolute paths: a new run directory and that source
directory. The run directory's parent and scheduler log directory must exist.
For a future authorized run, submit held with an explicit scheduler log:

```bash
sbatch --hold --output="$attempt/logs/slurm-%j.out" \
  "$source_dir/smoke_tests/polaris_buffered_rollouts/run.sbatch" \
  "$attempt/run" "$source_dir"
```

Use a fresh child of the task's `smoke_tests/native-resume/attempts/` hierarchy.
Register the submitted job with the existing central `job_monitoring/workflow.py`
monitor and verify monitoring before releasing the hold through shared scheduler
control. The model, POLARIS data, image and Lustre venv are shared assets.

The launcher writes `fresh.log`, `resume.log`, native checkpoints and training
logs under the run directory. The native checkpoint includes the resolved config.
Existing attempt snapshots and artifacts remain unchanged.

## Separate result check

`validate.py --output /absolute/run/path` is a standalone CPU result check;
it is not run by `run.sbatch`. It checks all five checkpoints, the 128 saved
responses/rewards consumed by resumed updates, finite/nonzero gradients, buffer
ages, generation counts, and compilation/graph-capture evidence. It reads the
native final checkpoint's config and has no import-audit dependency.
Run it with the prepared environment and source mount on CPU after training:

```bash
uv run --frozen --no-sync --extra mcore python -B -u \
  smoke_tests/polaris_buffered_rollouts/validate.py --output "$run_dir"
```

A successful training job alone is not the overall smoke completion gate. The
central monitor must require the separate successful result check and all jobs
terminal before notifying overall success. Do not reuse a training-only
`completion_conditions` gate from an older launcher version.

Retain checkpoints, logs and final validation receipts until authorized cleanup.
Local checks use the main task's `tests/launcher-simplification/` under `~/data`;
compare the native resolved config with attempt 004 and check shell syntax/lint.
This smoke does not validate production W&B, throughput or long-run convergence.
