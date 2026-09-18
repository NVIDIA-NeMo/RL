# Super RL maintenance branch and launch contract

The deliverable is **this NeMo-RL repository**, on the PR 4136 maintenance
branch, based on the fixed PR3941 commit
`ca06137460b7e2edcaf6f1fd67ddbda5ddd6b8e2`. It is not a Research Factory run
directory, a pipeline fork, or a collection of job-specific overlays.
Keep the existing problem-by-problem commits; do not squash them while they
are under review. See [the fix index](super-rl-stability.md) for their scope.

For the regular s120 experiment, see [the reviewed gold-alignment changes](super-rl-gold-alignment.md),
including context/serving budgets, output penalties, explicit startup grace and
the unvalidated CP4/EP16 topology candidate.

**Current status: configuration separation and read-only checks, not a turnkey
training release.** Kimi's reward/cumulative-budget integration and Gym's judge
failure contract remain implementation blockers. Submodule closure and native
ARM/x86 validation remain unverified. Smoke/submission are not exposed by the
new entrypoint. No running job has been modified or restarted.

## Ownership boundaries

| Layer | Location | Owns | Must not contain |
| --- | --- | --- | --- |
| Training implementation | `nemo_rl/` | Reusable algorithms and correctness fixes | Cluster branches, private paths or experiment-specific reward constants |
| Preparation/runtime utilities | `tools/super_rl/` | Source/config checks, CCC staging, helper prebuild, clean submission | Copied run directories or silent data/environment fallbacks |
| Experiment delta | `training_configs/super_rl/experiments/` | Kimi budgets, reward rule, sample/step budget, age and replay selection | Accounts, images, model/data paths or secrets |
| Judge configuration | `training_configs/judges/` | Provider/served-model reference, judge settings and resource bindings | Secret values or automatic judge substitution |
| Site/hardware profiles | `training_configs/super_rl/profiles/` | Hardware, image family hints, container cwd and filesystem requirements | Personal accounts/paths, scientific hyperparameters or assumed current capacity |
| User configuration | External YAML or ignored `super-rl.local.yaml` | Account/site scheduling, requested role counts, artifact paths/digests, mounts, interpreter and secret **names** | Secret values; do not commit this file |
| Evidence | Run-owned directory, outside source | Immutable resolved config, source/image/data identities, preflight/smoke/checkpoint receipts | Mutations to the submitted source snapshot |

Image **family/architecture compatibility** belongs to the profile. The actual
image's location and digest belong to the user's artifact configuration until
a team-readable immutable release image exists. A filename hint does not pin
image bytes. Similarly, profiles specify filesystem requirements; users supply
actual source/target mounts, including logical and backing paths where needed.
Do not invent a verified image digest or copy an ARM image into the H100 profile.

AWS-CMH is four-GPU GB300/aarch64, OCI-HSG four-GPU GB200/aarch64. `h100` is
an eight-GPU x86_64 **hardware adapter**, not a scheduler site. Users must name
their actual site and derive an appropriate TP/CP/EP topology. Historical H100
pipeline validation does not certify this PR3941 maintenance branch. All three
profiles remain `certified: false`; changing that flag cannot enable submission.

## Kimi experiment delta

`experiments/kimi_s25.yaml` preserves the selected experiment parameters:

- low budget `1.25 × baseline_output_tokens`; high `2 × baseline_output_tokens`;
- over-budget reward **replacement** `-0.5`, not an additive penalty;
- policy generation ceiling **102,400 tokens**, separate from total context;
- 256 prompts × 16 generations; 150 steps, corresponding to 38,400 prompt
  consumptions in an ordered pass (the actual dataset still needs validation);
- async age 2, in-flight weight updates, compact-compatible packing/worker
  flags, and no cross-Ray-runtime replay-buffer restore;
- reasoning enabled; no prompt rewrite, TIR-to-CoT conversion or SciCode removal.

The intended budget rule clamps each multiplier-derived budget to the policy
ceiling. `max` means no extra effort-specific reduction, not infinite generation.
Multi-turn accounting must count assistant output cumulatively, including
reasoning; it must not reset the 100K allowance at each tool call. The YAML does
not implement this rule. `GRPOConfig` currently rejects enabled effort until
the implementation is ported and tested, preventing silent ordinary-GRPO runs.

This is a **delta**, not a complete recipe to pass to `run_grpo.py`. Do not
combine it with an arbitrary baseline and declare it runnable. A complete
repository-owned Super recipe still needs reviewed model architecture/MTP,
optimizer/loss, context/packing limits, role placement and all-route Gym catalog.
Compose through NeMo-RL's existing `defaults` mechanism once that recipe exists;
do not build a second generic configuration framework in the launcher.

The existing hosted DeepSeek files remain an explicit option. The user template
selects `self_hosted`; this selects an intent, **not** a provisioned endpoint.
Local judge serving, model/prompt/reasoning parity, resource accounting and
failure tests are still required. Never silently fall back to hosted serving
or map an unavailable verdict to reward zero.

## Read-only entrypoint

Use an already prepared environment with OmegaConf and Pydantic; this stage
does not need to import Ray, Torch or Gym. From the repository root, with that
environment already available to `uv`:

```bash
uv run --no-sync tools/super_rl/launch.py \
  --profile aws-cmh --user /absolute/path/to/my-super-rl.yaml --check
```

The `--check` flag is optional: this is always a read-only check. Use `oci-hsg`
or `h100` to select the corresponding profile. `--experiment` can select
another Kimi delta; the default is `experiments/kimi_s25.yaml`. `--user`
defaults to the Git-ignored `super-rl.local.yaml` at the repository root. Start
from `training_configs/super_rl/user.example.yaml`; unresolved `???` values
fail. Only the example belongs in Git. Resolve cluster paths on the cluster, not
by pretending a workstation can inspect remote Lustre.

**Scope of the experiment check.** `experiment_errors` validates Kimi effort
deltas only, and `read_yaml` rejects every `${...}` interpolation so that no
secret is ever resolved. The regular recipe
`experiments/regular_s120_smoke.yaml` is built from `${oc.env:SUPER_RL_*}`
interpolations and carries no effort block, so passing it to `--experiment`
fails by design. That recipe is validated by
`tests/unit/tools/test_regular_recipe_assets.py` and, at startup, by the
trainer's configuration schema. Conversely, `kimi_s25.yaml` passes this checker
but `GRPOConfig` rejects it at startup until the effort port lands; the checker
confirms the delta's internal consistency, not that it can run today.

## Regular smoke environment contract

Super RL uses **W&B only**, for smoke and full training alike:
`logger.wandb_enabled: true`, `logger.tensorboard_enabled: false`,
`logger.mlflow_enabled: false`, and `logger.swanlab_enabled: false`.
The CP4/EP16 candidate inherits these settings; the Kimi delta also sets them
explicitly to override an inherited logger configuration. Keep them unchanged
when deriving a full-training config or preparing a smoke driver. Provide W&B
credentials privately to the driver, and verify scalar delivery to the intended
run before certifying a smoke. Do not disable W&B or fall back to TensorBoard
when authentication fails. Historical TensorBoard-only smoke results do not
certify W&B delivery. Other NeMo-RL recipes retain their own logger choices.

`regular_s120_smoke.yaml` and `local_deepseek_v4_flash.yaml` read every private
or site-specific input from the environment of the process that loads the
config (the driver inside the training container). Nothing under
`tools/super_rl/` exports these yet; `submit.py` only forwards the names it is
given with `--env`. Set all required variables before launch, or OmegaConf fails
at load with a missing-variable error.

| Variable | Read by | Default |
| --- | --- | --- |
| `SUPER_RL_ROOT` | `checkpointing.checkpoint_dir`, `logger.log_dir`, Gym log/cache/results dirs (`<root>/smoke/...`) | required |
| `SUPER_RL_MODEL` | `policy.model_name`, `policy.tokenizer.name` | required |
| `SUPER_RL_DATA` | `data.train.data_path`, `data.validation.data_path` | required |
| `SUPER_RL_PARSER` | `policy.generation.vllm_cfg.reasoning_parser_plugin` | required |
| `SUPER_RL_SCICODE_HDF5` | `scicode` resource `test_data_fpath` | required |
| `SUPER_RL_CCC_METADATA` | `competitive_coding_challenges` resource `test_file` | required |
| `CCC_SHARED_TEMP_DIR` | `competitive_coding_challenges` resource `shared_dir`; must be visible to every sandbox node | required |
| `SUPER_RL_RUN_NAME` | `logger.wandb.name` | required |
| `SUPER_RL_JUDGE_URL` | `deepseek_v4_flash_judge_model` `base_url` in `local_deepseek_v4_flash.yaml` | required |
| `NEMO_SKILLS_SANDBOX_HOST` | `ns_tools` resource `sandbox_host` | `127.0.0.1` |
| `NEMO_SKILLS_SANDBOX_PORT` | `ns_tools` resource `sandbox_port` | `6000` |

The same fields appear under different names in `user.example.yaml`
(`model`, `train_data`, `ccc_metadata`, `scicode_hdf5`, `work_root`) because the
launcher does not compose recipes yet; keep the two consistent by hand until
step 4 of the acceptance order below wires them together.

The smoke also fixes the cluster shape and container layout of the two
four-GPU profiles: `cluster.gpus_per_node: 4`, 48 generation nodes, 64 nodes in
total, `segment_size: 4`, the judge fragment at
`/opt/nemo-rl/training_configs/super_rl/local_deepseek_v4_flash.yaml` (the
profiles' `container_workdir`), and `uv_venv_dir: /opt/train_gym_venvs`. Those
values belong to the site/profile layer; an `h100` run needs its own experiment
file with an eight-GPU topology rather than edits to this one.

The checker validates source ancestry and pinned submodule checkouts, literal
configuration, scalar budget/batch consistency, declared image architecture,
host-path existence/access, secret **presence**, and basic mount consistency.
It reports total requested nodes/GPUs, including separately declared judge
nodes. It does not allocate those roles. It never evaluates secret environment
interpolations, prints configuration values, fetches Git refs, installs packages,
hashes large files, contacts Slurm, or writes a resolved YAML/run directory.

Exit `0` means only that these static checks passed. The JSON always reports
`submission_supported: false` and names the remaining native/release gates.
Exit `1` means a static check failed. `--submit` is rejected. The existing
low-level `submit.py` remains available to independently reviewed launchers;
it is **not** a bypass that completes this new Super launch contract.

## Remaining implementation and acceptance order

1. Resolve accessible immutable submodules, preserving pins unless a separately
   reviewed dependency commit deliberately changes them. A branch name or
   mutable cluster copy is not a substitute for Git provenance.
2. Port Kimi reward shaping and **all** cumulative-budget paths, then replace
   the effort rejection guard. Test CCC, CoT, TIR/ns_tools, SciCode and
   equivalence judging, including negative and malformed/error responses.
3. Port the judge failure contract and integrate explicit self-hosted serving.
   Exercise bounded queueing/retry, a slow request and replica loss. Retries
   must not discard the first valid negative verdict.
4. Add the complete base recipe and runtime adapter. Reuse `ray.sub`,
   `submit.py`, CCC staging and helper prebuild; do not copy historical
   `run_driver_vN.sh` scripts. Freeze the resolved source/config/images/data,
   propagate worker settings, and check model shards, every route, mounts,
   CPU affinity, account/QoS/topology, quota, singleton ownership and secrets.
5. Expose explicit smoke and production actions on this **same entrypoint**.
   Default stays non-submitting. Do not automatically scale after scheduler
   acceptance. An age-2 smoke needs at least three optimizer targets, all-route
   rewards, update/refit, finalized checkpoint and distributed reload. Periodic
   checkpoint/resume certification and performance acceptance precede production.

Keep future implementation commits separate: Kimi shaping; cumulative budgets;
judge failures; each genuinely distinct runtime fix; complete recipe/launcher;
then native certification evidence per profile. Store evidence of unresolved
numerical/throughput problems in the guide instead of merging speculative fixes.

## Validation of the configuration/check stage

`tests/unit/tools/test_super_rl_launch.py`: **32 passed** in the lightweight
preparation environment. Tests cover profile separation, private-input schema,
unfilled/interpolated/malformed YAML, declared image architecture, mounted
checkout identity, explicit H100 site, Kimi cap/batch consistency, reasoning-on
and compact flags, no fetching/submission, and secret-value redaction.

Combined with the prior lightweight stability suite: **102 passed, 1 skipped**.
The skip still requires the native sandbox worker. The five new `GRPOConfig`
guard cases in `tests/unit/algorithms/test_grpo.py` could not collect locally
because Ray is absent; they are not counted as passed. Ruff and whitespace
checks passed. Native imports, GPUs, Slurm, image digests, service calls,
throughput and full configuration/route composition were not tested here.
