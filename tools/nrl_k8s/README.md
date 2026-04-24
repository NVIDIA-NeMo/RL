# nrl-k8s

A config-driven launcher that runs a NeMo-RL recipe on any Kubernetes cluster
with the KubeRay operator installed. One YAML pair captures *what to train*
(the recipe) and *where to run it* (the infra); the CLI brings up every
RayCluster, submits any long-running daemons (gen / gym servers), and kicks
off the training job in a single step.

## Prerequisites

The CLI delegates to `kubectl`, the Kubernetes Python client, and Ray's Job
Submission SDK. Before your first run make sure the following are in place:

- **`kubectl`** on your `PATH`, pointed at the cluster you want to deploy to.
  `kubectl auth can-i create rayclusters -n <namespace>` must return `yes`.
- **KubeRay operator** v1.2+ installed on the target cluster. The CLI applies
  `ray.io/v1` `RayCluster` custom resources and polls `.status.state`.
- **Ray dashboard** must be reachable from your laptop for job submission.
  `nrl-k8s` opens a port-forward via the `kubectl-ray` plugin (or plain
  `kubectl port-forward` as a fallback) — `submit.portForward` picks which.
- **AWS EKS, p5.48xlarge**: use an image that bundles the
  `aws-ofi-nccl` plugin (the `nvcr.io/nvidian/nemo-rl:nightly` image shipped
  with this repo does). The EFA device plugin must be installed in the
  cluster so pods can request `vpc.amazonaws.com/efa`.

## Install

`nrl-k8s` installs as a standalone CLI from this repo. Use [uv](https://docs.astral.sh/uv/)
for both setup and development — it's what the project is tested with.

### End-user install (global `nrl-k8s` binary)

```bash
uv tool install ./tools/nrl_k8s
nrl-k8s --version
```

`uv tool install` drops the CLI in `~/.local/bin` (on `PATH`) inside its
own isolated environment, so it never clashes with whatever your project
venv has pinned. Upgrade with `uv tool upgrade nrl-k8s` after a git pull,
or `uv tool install --reinstall ./tools/nrl_k8s`.

### Development (editable install + tests)

```bash
# from the repo root
cd tools/nrl_k8s
uv venv                                 # creates .venv/
source .venv/bin/activate
uv pip install -e ".[test]"             # editable install + test extras
pytest                                  # 9 test modules, ~100 tests
```

Or run commands without activating the venv:

```bash
uv run --directory tools/nrl_k8s -- pytest
uv run --directory tools/nrl_k8s -- nrl-k8s --help
```

The package depends on `click`, `omegaconf`, `pydantic`, `kubernetes`,
`ray[default]`, and `tenacity`. It does *not* require the full `nemo_rl`
package to be importable on your laptop — the CLI stages a working_dir for
Ray's Job SDK and runs the training entrypoint inside the cluster image.

## Quick start

Three canonical flows ship with working recipes under
`tools/nrl_k8s/examples/`. All three train Qwen3-4B with GRPO on the
`instruction_following` gym; they differ in how many RayClusters the run
occupies and where generation/gym live.

| variant | RayClusters | generation | gym |
|---|---|---|---|
| `qwen3_4b_if_single` | 1 | colocated in training cluster | local Ray actor in training cluster |
| `qwen3_4b_if_gym_disagg` | 2 | colocated in training cluster | dedicated RayCluster + HTTP daemon |
| `qwen3_4b_if_full_disagg` | 3 | dedicated RayCluster + HTTP daemon | dedicated RayCluster + HTTP daemon |

### `qwen3_4b_if_single` — everything on one RayCluster

Simplest shape. One GPU RayCluster hosts training + colocated vLLM, and
nemo_gym runs as a local Ray actor pinned to the worker node. No HTTP
between roles, no endpoint-registry rendezvous.

```bash
nrl-k8s run \
    tools/nrl_k8s/examples/qwen3_4b_if_single.yaml \
    --infra tools/nrl_k8s/examples/qwen3_4b_if_single.infra.yaml \
    --follow
```

### `qwen3_4b_if_gym_disagg` — gym on its own cluster

One GPU RayCluster hosts training and colocated vLLM; a CPU-only
RayCluster runs the gym rollout server.

```bash
nrl-k8s run \
    tools/nrl_k8s/examples/qwen3_4b_if_gym_disagg.yaml \
    --infra tools/nrl_k8s/examples/qwen3_4b_if_gym_disagg.infra.yaml \
    --follow
```

`run` applies both RayCluster manifests in order, submits the gym daemon
once its cluster is `Ready`, then submits the training Ray Job against the
training cluster and tails its logs.

```bash
nrl-k8s status \
    tools/nrl_k8s/examples/qwen3_4b_if_gym_disagg.yaml \
    --infra tools/nrl_k8s/examples/qwen3_4b_if_gym_disagg.infra.yaml
```

### `qwen3_4b_if_full_disagg` — generation + gym + training on separate clusters

Three RayClusters, one per role. Training streams generation requests to
the standalone generation server, which lives on its own GPUs:

```bash
nrl-k8s run \
    tools/nrl_k8s/examples/qwen3_4b_if_full_disagg.yaml \
    --infra tools/nrl_k8s/examples/qwen3_4b_if_full_disagg.infra.yaml \
    --follow
```

`run` walks the three roles in order: `generation` first (vLLM has to be
serving before training opens sockets to it), then `gym` (publishes
`gym_head_server` into the endpoint-registry ConfigMap), then `training`.
Once the training Ray Job is submitted its auto-generated ID is printed and
`--follow` tails its logs via a port-forward to the training dashboard.

```bash
nrl-k8s status \
    tools/nrl_k8s/examples/qwen3_4b_if_full_disagg.yaml \
    --infra tools/nrl_k8s/examples/qwen3_4b_if_full_disagg.infra.yaml
```

`DAEMON` is populated for `generation` and `gym`; `training` shows `—`
because its jobs are short-lived and auto-named, so look them up with
`nrl-k8s job list --role training`.

## Config layout

Each run is two files: a recipe and an infra.

- **`<recipe>.yaml`** — pure NeMo-RL config. Everything the training
  entrypoint (`examples/nemo_gym/run_grpo_nemo_gym.py` in the examples)
  expects: `policy`, `grpo`, `data`, `logger`, etc. Inherits from
  `examples/configs/recipes/**` via a standard `defaults:` field so it stays
  short. Portable across clusters.
- **`<recipe>.infra.yaml`** — K8s-only. Namespace, container image, the
  inline RayCluster spec for each role, daemon entrypoints, the training
  entrypoint, and where Ray should upload code from. Validated against
  `nrl_k8s.schema.InfraConfig` (see `tools/nrl_k8s/src/nrl_k8s/schema.py`).

You can also bundle the two in one file — put an `infra:` top-level key on
the recipe and omit `--infra`. The split is preferred for anything you plan
to share, because the recipe itself then has no environmental assumptions.

### `defaults:` inheritance

Recipes support a `defaults:` field (same semantics as NeMo-RL's own
loader). Point it at a parent recipe path relative to the file itself:

```yaml
# tools/nrl_k8s/examples/qwen3_4b_if_full_disagg.yaml
defaults: ../../../examples/nemo_gym/grpo_qwen3_4b_instruct_k8s_base.yaml

grpo:
  max_num_steps: 200         # override one field from the parent
```

The parent is loaded first; the child's keys are then merged on top. Chains
work — the parent can itself have a `defaults:`. See
`tools/nrl_k8s/src/nrl_k8s/config.py:165` for the walker.

Infra files also honour `defaults:` (via the same walker), so a team can
keep a `defaults.infra.yaml` with shared node selectors, image, and
namespace and point each per-run infra at it.

### Override priority

Four layers stack low-to-high (last wins):

1. Shipped defaults: `tools/nrl_k8s/src/nrl_k8s/defaults/defaults.example.yaml`
2. User defaults: `~/.config/nrl-k8s/defaults.yaml` (optional; can be
   repointed with `NRL_K8S_DEFAULTS=/path/to/file.yaml`)
3. The infra file (via `--infra`) *or* the recipe's `infra:` block. Not both.
4. Hydra-style CLI overrides: `infra.scheduler.queue=team-a`,
   `grpo.max_num_steps=10`.

`infra.*` overrides target the infra layer; everything else targets the
recipe. See `tools/nrl_k8s/src/nrl_k8s/config.py:102` for the partition
logic.

## Command reference

Every command takes the recipe path first, then positional Hydra overrides,
then flags. Pass `--infra <path>` when recipe and infra are split.

### `nrl-k8s check`

Load and validate a recipe/infra pair. Default mode prints a one-page
summary (namespace, image, per-role head/worker sizing, daemon ids, full
training entrypoint body). Pass `-o <file>` to write the fully-resolved
`InfraConfig` + recipe + rendered RayCluster manifests to disk instead —
the format picks up from the extension (`.yaml` / `.json`).

```bash
# summary
nrl-k8s check \
    tools/nrl_k8s/examples/qwen3_4b_if_full_disagg.yaml \
    --infra tools/nrl_k8s/examples/qwen3_4b_if_full_disagg.infra.yaml

# full bundle for diffs / kubectl apply --dry-run piping
nrl-k8s check ... -o /tmp/bundle.yaml
```

Replaces the older `validate` + `plan` commands (`validate` stays as a
hidden deprecation alias that routes to `check`). To render just a single
role's manifest, pipe from the bundle file, or use
`nrl-k8s cluster up --dry-run --role <role>`.

### `nrl-k8s cluster up --role {generation,gym,training}`

Apply the RayCluster manifest for a role, wait for `state=ready`, and
submit the role's daemon if declared. `--dry-run` prints the exact
manifest that would be applied and exits without hitting the API server:

```bash
nrl-k8s cluster up \
    tools/nrl_k8s/examples/qwen3_4b_if_full_disagg.yaml \
    --infra tools/nrl_k8s/examples/qwen3_4b_if_full_disagg.infra.yaml \
    --role training --dry-run
```

### `nrl-k8s launch`

Submit a training Ray Job against an **already-up** training cluster. Does
not bring up generation or gym. Use this when `nrl-k8s run` has already
stood things up and you just want to rerun training after editing code.

```bash
nrl-k8s launch \
    tools/nrl_k8s/examples/qwen3_4b_if_full_disagg.yaml \
    --infra tools/nrl_k8s/examples/qwen3_4b_if_full_disagg.infra.yaml \
    --follow --replace
```

Flags: `--repo-root <path>` (defaults to `cwd`; the source tree Ray packages
into `working_dir`), `--follow`, `--replace` (see below).

### `nrl-k8s run`

Do the full sequence: apply each RayCluster, submit each role's daemon
(for generation / gym), then submit the training Ray Job. Same flags as
`launch`. Safe to re-run idempotently on healthy clusters — already-running
daemons are skipped unless `--replace` is passed.

### `nrl-k8s cluster up --role <role>`

Apply one RayCluster manifest and, once `Ready`, submit its daemon if the
recipe has one.

```bash
nrl-k8s cluster up \
    tools/nrl_k8s/examples/qwen3_4b_if_full_disagg.yaml \
    --infra tools/nrl_k8s/examples/qwen3_4b_if_full_disagg.infra.yaml \
    --role gym
```

Flags: `--wait/--no-wait`, `--timeout <seconds>` (default 900).

### `nrl-k8s cluster down`

Delete a RayCluster by role (resolved from the recipe) or by name.

```bash
nrl-k8s cluster down \
    tools/nrl_k8s/examples/qwen3_4b_if_full_disagg.yaml \
    --infra tools/nrl_k8s/examples/qwen3_4b_if_full_disagg.infra.yaml \
    --role gym
```

Flags: `--role <role>` or `--name <raycluster-name>`, `--wait/--no-wait`.

### `nrl-k8s cluster list -n <namespace>`

List RayClusters in a namespace with their `.status.state`.

```bash
nrl-k8s cluster list -n nemo-rl-testing
```

### `nrl-k8s status`

One-line-per-role summary of every cluster in the recipe: RayCluster state,
head pod phase, worker pod phases, daemon submission id and Ray Job status.
See the Quick start output above for the exact format.

### `nrl-k8s logs --role <role>`

Stream logs for a role. With `--source auto` (default) the CLI picks the
daemon's Ray Job when the role has one, else the head pod's container
logs. Override with `--source {daemon,head,worker}`.

```bash
nrl-k8s logs tools/nrl_k8s/examples/qwen3_4b_if_full_disagg.yaml \
    --infra tools/nrl_k8s/examples/qwen3_4b_if_full_disagg.infra.yaml \
    --role generation -f --tail 500
```

### `nrl-k8s job list --role <role>`

List Ray Jobs currently on the role's RayCluster (via its dashboard).

```bash
nrl-k8s job list \
    tools/nrl_k8s/examples/qwen3_4b_if_full_disagg.yaml \
    --infra tools/nrl_k8s/examples/qwen3_4b_if_full_disagg.infra.yaml \
    --role training
```

### `nrl-k8s job logs <submission_id> --role <role>`

Tail logs for a specific Ray Job submission by id on a role's cluster.
Equivalent to `ray job logs --follow <id>` with the dashboard port-forward
auto-managed.

### `nrl-k8s job stop <submission_id> --role <role>`

Stop a Ray Job by submission id. Useful for clearing a stuck training job
before a re-run (though `launch --replace` does this automatically).

### `nrl-k8s dev` / `nrl-k8s dashboard` / `nrl-k8s doctor`

Not yet implemented — stubs print `not yet implemented (phase: ...)` and
exit `2`.

## `--replace` semantics

Both `nrl-k8s launch` and `nrl-k8s run` accept `--replace`. It performs
three idempotency-relevant actions before submitting:

1. **Endpoint registry reset.** The CLI parses the gym daemon's
   `--job-id` flag (see `tools/nrl_k8s/src/nrl_k8s/orchestrate.py:231`) and
   deletes the `nemo-rl-endpoints-<job-id>` ConfigMap. Without this the new
   gym or training publishes alongside stale keys from a prior failed run,
   and the rendezvous picks up stragglers. See the recipes guide for the
   registry's role.
2. **Stop running Ray Jobs.** On every cluster touched, any Ray Job in
   state `RUNNING` is stopped and the CLI blocks until it reaches a
   terminal state (capped at 60 s). This applies to the daemons
   (if their `submissionId` matches) and to every running job on the
   training cluster.
3. **Suffix daemon submissionIds.** Ray refuses to reuse a submissionId
   after the job has terminated, so `--replace` appends `-<unix-ts>` to
   `infra.clusters.<role>.daemon.submissionId` when resubmitting. Your
   configured id stays the same for the *next* run — the suffix only
   affects this submission.

`--replace` does **not** delete RayCluster custom resources; clusters stay
up so a re-run doesn't pay the image pull + pod scheduling cost again. Use
`nrl-k8s cluster down` for that.

## Troubleshooting

### Slow `working_dir` upload (or `RuntimeError: size over 100 MiB`)

Ray's Job SDK caps `working_dir` at 100 MiB. `infra.launch.rayUploadPaths`
(and the per-daemon `rayUploadPaths` on each cluster) exists to narrow what
you ship. The disagg example lists individual files under
`resources_servers/instruction_following/` so the 87 MiB `train.jsonl`
isn't included (see `qwen3_4b_if_full_disagg.infra.yaml:229`). If uploads are slow,
`nrl-k8s validate ... --show-recipe` won't help — instead `ls -lh` the
staged tmpdir by running `nrl-k8s launch --follow` and inspecting the log
line `[training] staging working_dir ...` (look at
`tools/nrl_k8s/src/nrl_k8s/workdir.py` for defaults).

### "expired token" / Kubernetes SSO errors

`nrl-k8s` uses the same kubeconfig your `kubectl` does. If you see
`TokenRequest: Unauthorized` mid-run, refresh SSO in a separate shell:

```bash
aws sso login --profile <profile>
kubectl auth whoami
```

Then re-run the same command — state lives on the cluster, not on your
laptop, so a re-run on an existing RayCluster just reconnects.

### Gym daemon stuck on `RUNNING` but training hangs

Gym's standalone server publishes `gym_head_server` into the
`nemo-rl-endpoints-<job-id>` ConfigMap, and training publishes
`vllm_base_urls` (disagg writes from the gen server; single-cluster writes
from training itself once colocated vLLM spawns). If either side is stale
from a prior run, the rendezvous deadlocks. Fix:

```bash
kubectl -n <namespace> delete configmap nemo-rl-endpoints-<job-id>
# or simply: nrl-k8s run ... --replace
```

### GPU OOM in colocated mode

Colocated vLLM (single-cluster) shares GPUs with the training backend.
`policy.generation.vllm_cfg.gpu_memory_utilization=0.45` in
`qwen3_4b_if_gym_disagg.yaml` leaves 55 % for training state — if you push
context length or batch size, drop it further (e.g. `0.35`) or halve
`policy.max_total_sequence_length`. The disagg pair doesn't have this
problem because generation lives on its own GPUs.

Note also that colocated runs are **incompatible with**
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (vLLM's CuMemAllocator
asserts). The disagg entrypoint sets it, the single-cluster entrypoint
does not — see the comment in `qwen3_4b_if_gym_disagg.infra.yaml:76`.

### Stale endpoint registry between runs

Symptoms: a fresh `nrl-k8s run` reports the training job submitted but
immediately logs "connection refused" to a URL that belongs to a pod from
a previous run. Always use `--replace` after a failed run; it wipes the
ConfigMap as described under `--replace` semantics.

