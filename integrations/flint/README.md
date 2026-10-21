# Flint support for NeMo RL (private integration)

This branch collects the changes used to run Flint GRPO with Automodel and
SGLang. It is based on RL commit `e65b8b8b808e078b2217d2058d7e0fe5ee120799`,
not a claim of compatibility with newer upstream revisions. The private `main`
branch adds one administrative commit removing inherited GitHub workflows;
upstream history is retained.

## Implementation

The changes under `nemo_rl/` preserve model-declared FP32 parameters during
refit, support Automodel's native gate dtype, handle the state-dict adapter's
mapping interface, and provide SGLang startup, router lifetime, and KV-budget
compatibility. The tokenizer compatibility guard supports the verified
Transformers 5.18 training environment alongside the 5.12.1 serving environment.

`patches/` contains the changes to external dependencies, not copies of their
private source trees:

* `sglang-flint-refit.patch`: Flint plugin refit support and dtype handling.
  Apply to the original Flint SGLang plugin checkout, not to SGLang itself.
* `automodel-sink-checkpoint.patch`: supply a caller activation to the FP32 sink
  holder so FSDP activation-checkpoint recomputation has a nonempty input tuple.
* `sglang-hybrid-kv-offload.patch`: propagate memory-saver configuration into
  both hybrid KV pool leaves. The JSON manifest records exact source hashes.

The separate Automodel model implementation remains a dependency:
<https://github.com/snowmanwwg/Automodel-private-20261017/pull/1>.
It is not replaced or duplicated by this RL PR.

Pinned dependencies from the tested image:

| Component | Revision |
| --- | --- |
| Automodel | `d7cdb35446e2f44f232f0005a4bb5ecda3ce85c6` |
| SGLang/Miles | `3003d70f680d41c59d1b7acbf65cb47795dfd19e` |
| Custom image | `flint-nemorl-terry-20261019-arm64-native-01.sqsh` |

## Recipes and operational tools

The four-layer and full 49-layer smoke recipes live under
`examples/configs/recipes/llm/`. The two eight-node DAPO recipes preserve the
original 4096/3072-token experiment and the new 5120/4096-token experiment.
The latter disables reward shaping, enables overlong filtering and W&B, and
starts from original weights. Both use 32 prompts x 8 responses, DAPO-Math-17k,
AIME2024 every five steps, two retained checkpoints, and a 50-step target.
A W&B run ID must be unique per experiment and stable across its restarts;
the already-running experiment's ID is intentionally not reused in the recipe.

`validation/` archives the actual launch, preflight, checkpoint, refit, and
failure-guard code from each workspace. These are deployment-specific tools,
not portable install scripts: they refer to the tested OCI paths and container
mounts (`/flint-work`, `/flint-run`, `/flint-model`). They also rely on source
baselines captured in the existing read-only smoke workspace. Those private
source captures, model weights, datasets, credentials, logs, and generated
reports are intentionally not included. Do not execute an archived submission
script unchanged: stage a new run directory, configure paths and a unique W&B
ID, and inspect its expected-resume and stop-marker guards first. The previous
experiment was explicitly stopped and must not be restarted.

The new run's `apply_runtime_overlays.py` additionally records the router
`router_policy`/`policy` compatibility shim. Its launcher scopes
`CUDA_LAUNCH_BLOCKING=1` to SGLang, uses a fatal-process guard, and prevents W&B
source/diff artifact uploads. The scripts preserve four-hour regular-batch
windows and submit one dependent successor only after finalized checkpoints.

## Validation and limitations

CPU compatibility tests are in `tests/compat/`. Regression tests for checkpoint
windows, runtime overlays, fatal guards, checkpoint audits, and step-zero tensor
capture are archived beside their production helpers. Run the CPU compatibility
suite without importing the root GPU fixture stack:

```bash
uv run --no-project --with pytest --with torch --with 'pydantic>=2' --with packaging --with 'transformers==5.18.0' \
  python -m pytest --confcutdir=tests/compat tests/compat -q
```

Historical GPU evidence established four-layer training/refit and full 49-layer
training plus optimizer resume. The earlier long run recorded 48 steps and was
stopped by the user; its latest durable checkpoint is step 45. It did not finish
50 steps. The new 4096-generation run is separate and was only in startup when
this change collection was prepared. No new GPU rerun is implied by this PR.

Known limits remain: full-logits parity is unverified; ordinary parameter
payload finiteness and optimizer-moment equality were not exhaustively audited.
The FlashInfer TRT-LLM serving failure was intermittent and is not claimed fixed.
The old checkpoint-save exception coincided with quota exhaustion, but causality
was not established. Do not interpret successful smoke tests as resolving those
issues.

`source-manifest.json` maps collected files to their original workspaces. It
contains paths and hashes only. Generated validation output is kept outside Git.
GitHub Actions is disabled in this private repository because upstream workflows
require NVIDIA infrastructure and the private model assets.

Fresh packaging checks: 30 CPU compatibility tests passed with Transformers
5.18.0; 36 runtime-tool tests passed; 10 step-zero tensor audit tests passed.
Python parsing, shell syntax, credential-pattern scanning, and Git whitespace
checks passed. GPU training was not rerun as part of packaging this PR.
