# Writing `nrl-k8s` recipes

A `nrl-k8s` run is two YAML files. Together they tell the CLI
*what* to train (the recipe) and *where* to run it (the infra). This guide
covers the split, every infra field, and how to port a recipe from one
cluster to another. For the CLI command surface, see the README.

## Recipe vs. infra

### Recipe file (`<run>.yaml`)

Pure NeMo-RL config — exactly what a training entrypoint like
`examples/nemo_gym/run_grpo_nemo_gym.py` expects. Typical keys: `cluster`,
`policy`, `grpo`, `data`, `logger`, `checkpointing`. Inherits from a
parent recipe via `defaults:`. Nothing K8s-specific lives here.

```yaml
# tools/nrl_k8s/examples/qwen3_4b_if_full_disagg.yaml
defaults: ../../../examples/nemo_gym/grpo_qwen3_4b_instruct_k8s_base.yaml

cluster:
  gpus_per_node: 8
  num_nodes: 1
grpo:
  num_prompts_per_step: 32
  max_num_steps: 200
policy:
  train_global_batch_size: 512
  max_total_sequence_length: 16384
```

The CLI stages the *merged* recipe (after `defaults:` resolution) as
`nrl_k8s_run.yaml` at the `working_dir` root before submitting the job, so
the entrypoint can reference it by the constant name
(`--config nrl_k8s_run.yaml`).

### Infra file (`<run>.infra.yaml`)

K8s-only — the pydantic `InfraConfig` body, defined in
`tools/nrl_k8s/src/nrl_k8s/schema.py`. The recipe and infra files are
loaded independently and merged by the CLI.

```yaml
# tools/nrl_k8s/examples/qwen3_4b_if_full_disagg.infra.yaml (excerpt)
namespace: nemo-rl-testing
image: nvcr.io/nvidian/nemo-rl:nightly
imagePullSecrets: [nvidia-ngcuser-pull-secret, ngc-registry-secret]
serviceAccount: nemo-rl-endpoint-registry
launch:
  mode: attach
  ...
clusters:
  generation: { ... }
  gym: { ... }
  training: { ... }
```

You can also put infra into the recipe as a top-level `infra:` key and omit
`--infra`. Don't do both — the loader refuses
(`tools/nrl_k8s/src/nrl_k8s/config.py:202`).

## Infra fields

Every field in the sections below corresponds to a pydantic model in
`schema.py`. The source is the authoritative reference; what follows is
grouped by concern with one example per area.

### Namespace, image, pull secrets

```yaml
namespace: nemo-rl-testing             # required
image: nvcr.io/nvidian/nemo-rl:nightly # required; patched onto every container
imagePullSecrets: [my-registry-secret] # attached to every pod template
rayVersion: "2.52.0"                   # optional — defaults to the image default
serviceAccount: nemo-rl-endpoint-registry  # set per pod when non-null
labels: {team: nemo-rl}                # merged into every RayCluster's metadata
annotations: {}
```

`image` is the one field you'll change most often when moving between
clusters. `serviceAccount` is required when anything in the run talks to the
Kubernetes API (e.g. `K8sEndpointRegistry` publishing into a ConfigMap,
used by gym/training rendezvous).

### Scheduler

```yaml
scheduler:
  kind: kai            # "kai" | "kueue" | "default"
  queue: priority-team # required when kind != "default"
```

`kai`/`kueue` trigger a scheduler-specific annotation/label patch on each
RayCluster. `default` leaves the Kubernetes default scheduler alone.

### Placement

```yaml
placement:
  nodeSelector:
    gpu-wrangler.nvidia.com/lease: nemo-rl-testing
  tolerations:
    - {key: platform.nvidia.com/gpu, operator: Equal, value: "true", effect: NoSchedule}
  affinity: null   # rare — raw-dict passthrough to pod spec
```

Node selectors and tolerations here apply to every pod template; the
examples duplicate them via YAML anchors inside each `clusters.<role>.spec`
for per-cluster flexibility.

### Networking

```yaml
networking:
  hostNetwork: false          # see note under the worked example below
  gloo_socket_ifname: enp71s0
  nccl_socket_ifname: enp71s0
  nccl_ib_disable: false
  nccl_net: OFI               # "Socket" | "IB" | "OFI"
  extra_env: {FI_PROVIDER: efa}
```

These become pod-level env vars on containers the CLI templates. Most real
recipes leave `networking:` at its defaults and bake the env into the
container image instead — setting them in pod env *and* `launch.env` at the
same time triggers Ray's runtime-env merge conflict (see the inline comments
in `qwen3_4b_if_full_disagg.infra.yaml`).

### Workspace, HF cache, checkpoints

```yaml
workspace:
  kind: rayUpload         # "lustre" | "pvc" | "hostPath" | "rayUpload" | "auto"
  pvcName: null           # required when kind in {lustre, pvc}
  mountPath: /mnt/nemo-rl
  repoSubdir: workdirs
  size: null              # only used when kind=lustre and PVC needs creating
  hostPath: null          # kind=hostPath only (dev/kind only)
hf_cache:
  kind: none              # "lustre" | "pvc" | "emptyDir" | "none"
  pvcName: null
  mountPath: /root/.cache/huggingface
checkpoints:
  kind: none              # "lustre" | "pvc" | "none"
  pvcName: null
  mountPath: /mnt/nemo-rl/checkpoints
```

`workspace.kind=rayUpload` is the default and the one the shipped examples
use — the CLI packages code into a tmpdir and Ray uploads it to the
cluster's GCS. PVC-backed kinds exist for larger repos or when you want
checkpoints to persist beyond the RayCluster lifetime; today they're
wired at the manifest level but not exercised by the example recipes.

### Submit (how the CLI gets a job in)

```yaml
submit:
  kind: sdk                       # "sdk" (Ray Job SDK) | "rayjob" (RayJob CRD)
  portForward: auto               # "kubectl-ray-plugin" | "kubectl-port-forward" | "auto"
  devPod: auto                    # "auto" | "required" | "skip" (not yet wired)
  localDashboardPort: 18265       # avoids collision with `kubectl-ray session`
```

### Launch

```yaml
launch:
  mode: attach                    # "single" | "rayjob" | "attach" | "bringup"
  attach:
    generation: raycluster-generation-qwen3-4b
    gym: raycluster-gym-qwen3-4b
    training: raycluster-rl-qwen3-4b
  peerWatcher: false              # inject the peer-watcher sidecar for failure cascades
  entrypoint: |                   # the training command; see below
    export ...
    python -u examples/nemo_gym/run_grpo_nemo_gym.py --config nrl_k8s_run.yaml ...
  env: {}                         # runtime_env.env_vars for the training job
  rayUploadPaths:                 # repo-relative paths to include in working_dir
    - nemo_rl
    - examples
    - infra/examples
    - 3rdparty/Gym-workspace/Gym/nemo_gym
    - ...
```

`entrypoint` is required for `launch` / `run`. The CLI stages the merged
recipe as `nrl_k8s_run.yaml` at the `working_dir` root; the entrypoint
references it by that fixed name.

Keep `rayUploadPaths` narrow — Ray's Job SDK caps the upload at 100 MiB.
List individual files inside heavy directories (see the example gym config)
when a full tree would include unneeded data. `null` means "use the built-in
default" from `tools/nrl_k8s/src/nrl_k8s/workdir.py`.

### Clusters and daemons

```yaml
clusters:
  generation:
    name: raycluster-generation-qwen3-4b
    labels: {disagg.nemo-rl/cluster: generation-qwen3-4b}
    annotations: {}
    daemon:
      submissionId: qwen3-4b-generation-server-v7
      entrypoint: |
        python -u examples/run_standalone_generation_server.py ...
      env: {}                     # runtime_env.env_vars for the daemon job
      healthCheckUrl: null        # optional — CLI polls before returning
      healthCheckTimeoutS: 300
      rayUploadPaths: [nemo_rl, examples, infra/examples]
    spec:
      # Full RayCluster .spec — headGroupSpec, workerGroupSpecs, etc.
      # Free-form dict; no pydantic schema.
      rayVersion: "2.52.0"
      headGroupSpec: { ... }
      workerGroupSpecs: [ ... ]
  gym:      { ... }
  training: { ... }
```

`spec` is the RayCluster `.spec` body passed straight through to the
Kubernetes API, wrapped by the CLI in the `apiVersion: ray.io/v1` +
`kind: RayCluster` + `metadata` envelope. Cross-cutting fields (`image`,
`imagePullSecrets`, `serviceAccountName`) are patched from the top-level
keys, so you don't repeat them. See
`tools/nrl_k8s/src/nrl_k8s/manifest.py`.

### Resources

```yaml
resources:
  training:
    head:   {cpu: "8", memory: "32Gi", gpu: null}
    worker: {cpu: "96", memory: "768Gi", gpu: 8}
  generation: { ... }
  gym:        { ... }
```

The `resources:` block is an escape hatch for when you want the CLI to
derive sensible container `resources:` per role instead of specifying
them inline in `spec`. The shipped examples set container resources
directly inside `spec` (via YAML anchors) for maximum clarity, so the
`resources:` block stays at defaults.

## Writing a fresh recipe from scratch

Suppose you want to run a simple colocated SFT-like scenario: one GPU
RayCluster, one training Ray Job, no gym, no generation server. Here's the
minimum pair.

### Recipe (`examples/sft_llama3_1b.yaml`)

```yaml
defaults: ../../../examples/configs/sft_llama3.1_1b.yaml

cluster:
  gpus_per_node: 8
  num_nodes: 1
policy:
  train_global_batch_size: 128
  max_total_sequence_length: 4096
checkpointing:
  save_period: 500
logger:
  wandb_enabled: true
  wandb: {entity: nvidia, project: nrl-k8s-smoke, name: sft-llama3-1b}
```

### Infra (`examples/sft_llama3_1b.infra.yaml`)

```yaml
namespace: nemo-rl-testing
image: nvcr.io/nvidian/nemo-rl:nightly
imagePullSecrets: [nvidia-ngcuser-pull-secret]
serviceAccount: nemo-rl-endpoint-registry

launch:
  mode: attach
  attach:
    training: raycluster-sft-llama3-1b
  peerWatcher: false
  rayUploadPaths:
    - nemo_rl
    - examples
    - infra/examples
  entrypoint: |
    export GLOO_SOCKET_IFNAME=enp71s0
    export NCCL_SOCKET_IFNAME=enp71s0
    export FI_PROVIDER=efa
    python -u examples/run_sft.py --config nrl_k8s_run.yaml

clusters:
  training:
    name: raycluster-sft-llama3-1b
    spec:
      rayVersion: "2.52.0"
      headGroupSpec:
        rayStartParams: {dashboard-host: "0.0.0.0", num-gpus: "0"}
        template:
          spec:
            nodeSelector: {gpu-wrangler.nvidia.com/lease: nemo-rl-testing}
            tolerations:
              - {key: gpu-wrangler.nvidia.com/lease, operator: Equal, value: nemo-rl-testing, effect: NoSchedule}
            containers:
              - name: ray-head
                resources: {limits: {cpu: "8", memory: "32Gi"}}
                ports:
                  - {containerPort: 6379,  name: gcs-server}
                  - {containerPort: 8265,  name: dashboard}
                  - {containerPort: 10001, name: client}
      workerGroupSpecs:
        - groupName: gpu-workers
          replicas: 1
          minReplicas: 1
          maxReplicas: 1
          rayStartParams: {num-gpus: "8"}
          template:
            spec:
              hostNetwork: true
              dnsPolicy: ClusterFirstWithHostNet
              nodeSelector: {gpu-wrangler.nvidia.com/lease: nemo-rl-testing}
              tolerations:
                - {key: gpu-wrangler.nvidia.com/lease, operator: Equal, value: nemo-rl-testing, effect: NoSchedule}
              containers:
                - name: ray-worker
                  resources:
                    limits: {cpu: "96", memory: "768Gi", nvidia.com/gpu: "8", vpc.amazonaws.com/efa: "32"}
```

Launch:

```bash
nrl-k8s run examples/sft_llama3_1b.yaml \
    --infra examples/sft_llama3_1b.infra.yaml --follow
```

Note what we did *not* declare: no `generation:`, no `gym:`, no daemon,
no `launch.env`. The `run` loop skips undeclared roles (see
`orchestrate.py:218`) so a training-only recipe flows straight through.

## Adapting an existing recipe to a new cluster

Assume you want to take `qwen3_4b_if_full_disagg.yaml` + `.infra.yaml` and run them on
a different cluster. Typical changes, in order of likelihood:

1. **`namespace`** — the one you have permission in.
2. **`image`** and **`imagePullSecrets`** — point at the registry
   reachable from the new cluster. The image must ship the NCCL/EFA plugin
   your network needs.
3. **`serviceAccount`** — the SA that can read/write
   `nemo-rl-endpoints-*` ConfigMaps in the new namespace.
4. **`placement.nodeSelector`** and **`tolerations`** — match your node
   labels. In the shipped example these are inlined inside each
   `clusters.<role>.spec` via anchors, so update both places.
5. **Worker resources** — change
   `clusters.<role>.spec.workerGroupSpecs[].template.spec.containers[].resources`
   to match your instance type. On non-EFA clusters, drop
   `vpc.amazonaws.com/efa` and set `hostNetwork: false`.
6. **Environment variables** in the `entrypoint` blocks —
   `GLOO_SOCKET_IFNAME`, `NCCL_SOCKET_IFNAME`, `FI_PROVIDER` are specific
   to AWS p5.48xlarge. On GCP A3 they're different; on baremetal, drop
   them entirely if the image already sets sensible defaults.
7. **`launch.attach.*` names** and **`clusters.<role>.name`** — bump these
   if you're running the same recipe side-by-side with an existing run
   (e.g. append `-v2`). The `submissionId` of each daemon should change too
   so Ray accepts the new submission.

Things that should *not* change when porting:

- The recipe file. The recipe is cluster-agnostic by design; if you're
  editing `policy:`, `grpo:`, etc. to port it, something's off.
- The training `entrypoint`'s python command. Env vars and data paths may
  differ; the `python -u examples/.../entry.py --config nrl_k8s_run.yaml
  ...` line stays.
- `launch.rayUploadPaths` — governed by what the entrypoint imports, not
  by the target cluster.

## Endpoint registry ConfigMap

Disaggregated and single-cluster-colocated runs both need the gym and
training halves to find each other. They rendezvous via a Kubernetes
ConfigMap named `nemo-rl-endpoints-<job_id>` in the namespace. Keys:

- `gym_head_server` — published by the gym standalone server once its HTTP
  listener is up.
- `vllm_base_urls` — published by the generation side (the standalone
  gen server in disagg mode; colocated vLLM in single-cluster mode) so
  training and gym know where to send generation requests.

`<job_id>` comes from the `--job-id` flag on the gym entrypoint (see
`qwen3_4b_if_full_disagg.infra.yaml:221`). The CLI does not have a separate config key
for it — that would just duplicate what's in the entrypoint.

### How `disagg_job_id` is inferred

`--replace` wipes the ConfigMap so a new run rendezvouses on fresh keys.
The CLI finds the ConfigMap's name by parsing `--job-id <value>` from the
**gym daemon entrypoint** string
(`tools/nrl_k8s/src/nrl_k8s/orchestrate.py:228`). If your recipe has no
gym cluster or the gym daemon entrypoint doesn't include `--job-id`, the
reset is skipped silently. Keep the flag on one line and quote-free for
the regex (`--job-id my-id` or `--job-id=my-id`).

The same `job_id` is fed into the training entrypoint via
`+env.disagg_job_id=<id>` (see `qwen3_4b_if_full_disagg.infra.yaml:116`) so training's
`K8sEndpointRegistry` publishes and reads from the same ConfigMap.

### What `--replace` cleans up

As covered in the README:

1. `nemo-rl-endpoints-<job_id>` ConfigMap deleted.
2. Running Ray Jobs on any touched cluster stopped (daemons and training).
3. Daemon `submissionId` suffixed with a unix timestamp for the
   resubmission — Ray can't reuse IDs even after a terminal state.

It does *not* delete RayClusters or PVCs; use `nrl-k8s cluster down` for
that.
