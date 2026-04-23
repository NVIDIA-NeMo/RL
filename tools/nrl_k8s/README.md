# nrl-k8s

> [!WARNING]
> These instructions are in active development and should not be relied on as stable yet.
> APIs, manifests, and tooling may change without notice.

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
    --wait
```

### `qwen3_4b_if_gym_disagg` — gym on its own cluster

One GPU RayCluster hosts training and colocated vLLM; a CPU-only
RayCluster runs the gym rollout server.

```bash
nrl-k8s run \
    tools/nrl_k8s/examples/qwen3_4b_if_gym_disagg.yaml \
    --infra tools/nrl_k8s/examples/qwen3_4b_if_gym_disagg.infra.yaml \
    --raycluster --wait
```

`--raycluster` is required for disaggregated runs (multiple clusters).
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
    --raycluster --wait
```

`run --raycluster` walks the three roles in order: `generation` first (vLLM has to be
serving before training opens sockets to it), then `gym` (publishes
`gym_head_server` into the endpoint-registry ConfigMap), then `training`.
Once the training Ray Job is submitted its auto-generated ID is printed and
`--wait` tails its logs until the job reaches a terminal state.

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

### `nrl-k8s run`

Submit a recipe to the cluster. Defaults to ephemeral **RayJob mode**
(`--rayjob`): KubeRay creates the RayCluster, submits the entrypoint,
polls until terminal, then tears down the cluster automatically. DRA
resources (ComputeDomain, RoCE ResourceClaimTemplate) are auto-created
before the job and auto-deleted after it finishes.

Pass `--raycluster` for **long-lived mode**: idempotently bring up each
declared RayCluster, submit daemons (generation / gym), then submit
training. Clusters stay up for subsequent runs. Use `--replace` to stop
running jobs before resubmitting, `--recreate` to delete + re-apply
drifted clusters.

Flags: `--wait/--no-wait`, `--dry-run` (RayJob mode only),
`--replace`, `--recreate`, `--skip-daemons` (long-lived mode only),
`--run-id <tag>`, `--mode {interactive, batch}`,
`--code-source {upload, image, lustre}`, `--code-path <path>`.

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
before a re-run (though `run --raycluster --replace` does this automatically).

### `nrl-k8s dev`

Lightweight dev pod for setup tasks (cloning repos, downloading models,
debugging). The pod runs on a CPU node with the shared workspace PVC
mounted.

#### `nrl-k8s dev setup-secrets`

Create or update your user secrets (tokens + SSH key). Required before
first `dev connect`:

```bash
nrl-k8s dev setup-secrets \
    HF_TOKEN=hf_xxx WANDB_API_KEY=key_yyy \
    --ssh-key ~/.ssh/id_ed25519
```

First-time usage requires `HF_TOKEN`, `WANDB_API_KEY`, and `--ssh-key`.
Subsequent runs accept any subset to update individual keys.

#### `nrl-k8s dev connect`

Create the dev pod (if it doesn't exist) and exec into it:

```bash
nrl-k8s dev connect
```

Lands you in `/mnt/rl-workspace/<username>` with your tokens as env vars
and SSH key at `/root/.ssh/`. The pod stays running after you exit — reconnect
with `dev connect` again.

#### `nrl-k8s dev stop`

Delete the dev pod:

```bash
nrl-k8s dev stop
```

### `nrl-k8s doctor`

Not yet implemented.

## Modes: interactive vs batch

`nrl-k8s run --raycluster` takes `--mode {interactive, batch}`. The flag is
a macro — it flips a coherent set of defaults that a researcher would
otherwise pick individually.

| dimension | `--mode interactive` (default) | `--mode batch` |
|---|---|---|
| Submitter transport | port-forward + Ray Job SDK | `kubectl exec` + `nohup` on head pod |
| Code source | `upload` (zipped working_dir via Ray SDK) | `image` (baked at `launch.codePath`) |
| Foreground wait | yes — tails logs, exits on terminal state | no — returns as soon as nohup fires |
| Laptop on critical path? | yes, for run lifetime | no |
| Typical use | dev iteration | long production runs |

Each piece is independently overridable: `--submitter portForward`,
`--code-source {upload, image, lustre}`, `--code-path /opt/nemo-rl`,
`--run-id <tag>`, `--wait` / `--no-wait`.

You can also set the mode in the infra YAML so researchers don't have
to remember the flag every run:

```yaml
# recipe.infra.yaml
launch:
  runMode: batch            # flips defaults; --mode on CLI still wins
  codeSource: image
  codePath: /opt/nemo-rl
submit:
  submitter: exec
```

### Batch submission walkthrough

The canonical example is `qwen3_4b_if_gym_disagg.yaml` paired with its
production infra variant. Same recipe as the dev example above, but
submission goes through `kubectl exec` and code comes from
`/opt/nemo-rl` inside the container instead of a laptop upload.

```bash
RECIPE=tools/nrl_k8s/examples/qwen3_4b_if_gym_disagg.yaml
INFRA=tools/nrl_k8s/examples/qwen3_4b_if_gym_disagg.prod.infra.yaml

# Admin, once: bring up the two RayClusters.
nrl-k8s cluster up "$RECIPE" --infra "$INFRA" --role gym
nrl-k8s cluster up "$RECIPE" --infra "$INFRA" --role training

# Researcher, per run — returns in seconds, laptop can close.
RUN_ID=qwen3-4b-gym-disagg-$(date +%Y%m%d-%H%M%S)
nrl-k8s run "$RECIPE" --infra "$INFRA" --raycluster --run-id "$RUN_ID"
# run id:  qwen3-4b-gym-disagg-20260421-103012
# kind:    exec
# cluster: raycluster-gym-disagg-qwen3-4b  (ns=nemo-rl-testing)
# pod:     raycluster-gym-disagg-qwen3-4b-head-xyz42
# tmp:     /tmp/nrl-qwen3-4b-gym-disagg-20260421-103012
# follow:  nrl-k8s job logs $RUN_ID <recipe> --role training -f

# Observe from any laptop, any time — the handle is cached under
# ~/.cache/nrl-k8s/runs/<run-id>.json.
nrl-k8s job logs "$RUN_ID" "$RECIPE" --infra "$INFRA" --role training -f
nrl-k8s job stop "$RUN_ID" "$RECIPE" --infra "$INFRA" --role training
```

The prod infra declares `submit.submitter: exec` + `launch.runMode:
batch` + `launch.codeSource: image`, so `--mode batch` is implicit.
The dev infra (`qwen3_4b_if_gym_disagg.infra.yaml`) keeps the
port-forward + upload path, letting `run --raycluster` default to
foreground log tailing for dev iteration.

The exec submitter writes a launcher script onto the head pod, runs it
under `nohup` + `disown`, captures the PID and (on exit) the exitcode.
`job logs` streams the driver's stdout via `kubectl exec tail -F`;
`job stop` sends SIGTERM via `kubectl exec kill` (SIGKILL with
`--force`).

Because the driver calls `ray.init(address="auto")` and registers with
the cluster, the run also shows up under `ray job list` on the head
dashboard and is reachable with `ray job logs` — a bonus if you prefer
the Ray CLI for log tailing.

### Env vars and `$NRL_K8S_RUN_ID`

Both submitters inject the resolved run id as `$NRL_K8S_RUN_ID`
(alongside anything in `infra.launch.env`) by prepending shell
`export` lines to the entrypoint. Recipes can reference it:

```yaml
launch:
  entrypoint: |
    set -eu
    cd /opt/nemo-rl
    python -u examples/nemo_gym/run_grpo_nemo_gym.py \
      --config examples/nemo_gym/grpo_qwen3_4b_instruct_k8s_base.yaml \
      logger.wandb.name=my-run-$NRL_K8S_RUN_ID
```

Inlining envs as `export` (rather than `runtime_env.env_vars`) avoids
Ray's "Failed to merge the Job's runtime env" error, which triggers
when the head pod's captured `os.environ` conflicts with a submission's
declared env. The shell path is portable; the Ray path isn't.

### Viewing the Ray dashboard

```
nrl-k8s cluster dashboard <cluster-name> [-n <namespace>]
```

`<cluster-name>` is what `nrl-k8s cluster list` / `kubectl get
rayclusters` shows. Namespace defaults to the current kube context.
Port-forwards `svc/<cluster-name>-head-svc:8265` to `localhost:8265`
and opens the URL in the default browser. Ctrl+C kills the forward.

**Blank dashboard?** The Ray dashboard serves its frontend from
`ray/dashboard/client/build/static/`. If those JS/CSS files are
symlinks into the uv cache (`/root/.cache/uv/archive-v0/...`),
aiohttp's `follow_symlinks=False` default 404s every request — the
browser renders an empty page.

The command's default `--fix` detects this and reinstalls
`ray[default]==<current-version>` in copy mode on the head pod
(`uv pip install --reinstall --link-mode=copy`). Idempotent: skipped
when the assets are already real files; ~30s when it runs. Pass
`--no-fix` on images that were already built correctly.

**The permanent fix is in the image build.** Set
`UV_LINK_MODE=copy` *before* the `uv pip install` that brings in
`ray[default]`:

```Dockerfile
ENV UV_LINK_MODE=copy
RUN uv pip install "ray[default]==2.54.0" ...
```

Or scope it to just the ray install:

```Dockerfile
RUN uv pip install --link-mode=copy "ray[default]==2.54.0"
```

Setting `UV_LINK_MODE=copy` only at pod runtime has no effect on an
already-populated venv — existing symlinks stay symlinks until the
package is reinstalled (which is exactly what `--fix` does on demand).

### Entrypoint shell

Ray's Job submission API runs the entrypoint through `/bin/dash` by
default, which does not support `set -o pipefail` or bash arrays. If
your entrypoint needs bash-only syntax, either:

- use `set -eu` instead of `set -euo pipefail`, or
- wrap the body: `exec bash -c '<your entrypoint>'`.

The exec submitter runs the launcher under `bash` directly, so
bash-specific features work there regardless.

### Code sources

`launch.codeSource` picks what's on disk inside the pod at run time:

| value | behaviour | when to use |
|---|---|---|
| `upload` (default) | Stage + upload `working_dir` via Ray Job SDK (100 MiB cap) | dev iteration with `--mode interactive` |
| `image` | Code baked into the container at `launch.codePath` (default `/opt/nemo-rl`) | production from a frozen image tag |
| `lustre` | Code pre-staged on a Lustre mount at `launch.codePath` | production with per-run snapshots without rebuilding |

For `image` and `lustre`, the entrypoint is responsible for `cd`ing
into `codePath` and `source`ing any env (`3rdparty/vllm/nemo-rl.env`,
etc.) — the CLI does not inject `cd` for you.

## `--replace` semantics

`nrl-k8s run --raycluster` accepts `--replace`. It performs three
idempotency-relevant actions before submitting:

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
`nrl-k8s check` won't help — instead `ls -lh` the
staged tmpdir by running `nrl-k8s run --raycluster --wait` and inspecting the log
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

