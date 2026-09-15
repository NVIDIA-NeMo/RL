# Super RL maintenance branch and launch contract

The deliverable is **this NeMo-RL repository**, on
`jcxu/pr3941-stable-super-rl`, based on the fixed PR3941 commit
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
another Kimi delta; the default is `experiments/kimi_s25.yaml`. Start from
`training_configs/super_rl/user.example.yaml`; unresolved `???` values fail.
Only the example belongs in Git. Resolve cluster paths on the cluster, not by
pretending a workstation can inspect remote Lustre.

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
