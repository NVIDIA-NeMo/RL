# Onboarding a new Kubernetes cluster

Checklist for going from `kubectl config use-context <new-cluster>` to
`nrl-k8s launch` landing training. Walk through in order; each step
ends with a verify-it-worked probe.

---

## 1. Baseline cluster capabilities

Required — refuses to install onto clusters that lack these.

```bash
# KubeRay operator
kubectl get crd rayclusters.ray.io -o jsonpath='{.metadata.name}' && echo " ✓"
kubectl -n kuberay-system get deploy kuberay-operator \
    -o jsonpath='{.status.conditions[?(@.type=="Available")].status}'
# → True

# GPU device plugin (at least one node reports nvidia.com/gpu in allocatable)
kubectl get nodes \
    -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.allocatable.nvidia\.com/gpu}{"\n"}{end}'
# → one or more rows with non-empty GPU counts
```

If KubeRay isn't present, install it. The operator helm chart we use
is tracked in this repo under `infra/helm/`; on a fresh cluster:

```bash
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm upgrade --install kuberay-operator kuberay/kuberay-operator \
    --namespace kuberay-system --create-namespace \
    --version 1.5.1
```

If your GPU nodes don't yet advertise `nvidia.com/gpu`, install the
NVIDIA device plugin (`kubectl apply -f
https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.15.0/nvidia-device-plugin.yml`)
and re-run the `kubectl get nodes` check.

---

## 2. Create the working namespace

```bash
NS=nemo-rl-testing   # pick whatever fits your org
kubectl create namespace $NS
kubectl config set-context --current --namespace=$NS
```

`nrl-k8s` reads the active namespace from your kube context by
default, so setting it on the context saves you from `-n $NS` on
every call.

---

## 3. RBAC — the endpoint-registry ServiceAccount

Disaggregated runs rendezvous on a ConfigMap called
`nemo-rl-endpoints-<job-id>`, written and read by the training +
generation + gym pods via the Kubernetes Python client. Each pod
needs `get / list / watch / create / update / patch` on ConfigMaps
in its namespace, and (if the peer-watcher sidecar is enabled)
`get / delete` on `rayclusters.ray.io` too.

Apply the bundled RBAC manifest:

```bash
kubectl apply -n $NS -f - <<'EOF'
apiVersion: v1
kind: ServiceAccount
metadata:
  name: nemo-rl-endpoint-registry
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: nemo-rl-endpoint-registry
rules:
  - apiGroups: [""]
    resources: [configmaps]
    verbs: [get, list, watch, create, update, patch, delete]
  - apiGroups: [ray.io]
    resources: [rayclusters]
    verbs: [get, list, watch, delete]
EOF
kubectl create rolebinding -n $NS nemo-rl-endpoint-registry \
    --role=nemo-rl-endpoint-registry \
    --serviceaccount=$NS:nemo-rl-endpoint-registry
```

Verify:

```bash
kubectl auth can-i create configmaps --as=system:serviceaccount:$NS:nemo-rl-endpoint-registry
kubectl auth can-i delete rayclusters.ray.io --as=system:serviceaccount:$NS:nemo-rl-endpoint-registry
# → yes, yes
```

---

## 4. Image pull secrets (private registries)

If your image lives in a private registry (e.g. `nvcr.io/nvidian/...`),
create a Docker-config secret:

```bash
kubectl create secret docker-registry ngc-registry-secret \
    --docker-server=nvcr.io \
    --docker-username='$oauthtoken' \
    --docker-password="$NGC_API_KEY" \
    -n $NS
```

The recipe's `infra.imagePullSecrets` field references these by name:

```yaml
imagePullSecrets: [ngc-registry-secret]
```

---

## 5. W&B secret (optional)

If your recipe logs to Weights & Biases, pods need `WANDB_API_KEY`.
We consume it via a `secretKeyRef` rather than embedding the key in
YAML — so rotate/replace the secret without touching any recipe:

```bash
kubectl create secret generic wandb-api-key \
    --from-literal=WANDB_API_KEY="$WANDB_API_KEY" \
    -n $NS
```

The existing example infra files reference `secretKeyRef: {name:
wandb-api-key, key: WANDB_API_KEY}` on the training head container.

---

## 6. Node pool — labels, taints, GPU resources

RayCluster specs pin workers via `nodeSelector` + `tolerations`. The
bundled examples use two keys that are NVIDIA-internal (`gpu-wrangler.nvidia.com/lease`
and `platform.nvidia.com/gpu`); you'll almost certainly want to swap
those for whatever your GPU node pool advertises.

```bash
# Inspect an existing GPU node:
kubectl get node <gpu-node> -o jsonpath='{.metadata.labels}' | python3 -m json.tool
kubectl get node <gpu-node> -o jsonpath='{.spec.taints}'    | python3 -m json.tool
```

Then update `clusters.training.spec.*.template.spec.nodeSelector`
and `.tolerations` in your recipe's `.infra.yaml` to match. The
examples use YAML anchors (`&shared_node_selector`) so the selector
lives in one place per file.

The GPU worker's resource request must include:

```yaml
resources:
  limits:
    cpu: "176"                 # leave headroom for daemonsets
    memory: "1800Gi"           # ditto
    nvidia.com/gpu: "8"        # must equal rayStartParams.num-gpus
    vpc.amazonaws.com/efa: "32"  # AWS p5 only; drop on other clouds
```

On AWS EKS with p5 instances you also need the EFA device plugin
installed in the cluster (`https://github.com/aws/eks-charts/tree/main/stable/aws-efa-k8s-device-plugin`).
On non-AWS clouds, remove the EFA line entirely.

---

## 7. Sanity-check with `nrl-k8s`

Install the CLI if you haven't already:

```bash
uv tool install ./tools/nrl_k8s
# or for editable dev: cd tools/nrl_k8s && uv pip install -e ".[test]"
nrl-k8s --version
```

Load-test a recipe without hitting the API server:

```bash
nrl-k8s check \
    tools/nrl_k8s/examples/qwen3_4b_if_single.yaml \
    --infra tools/nrl_k8s/examples/qwen3_4b_if_single.infra.yaml
```

This validates schema, resolves the full infra, and prints a
one-page summary. No cluster calls.

---

## 8. Bring up a RayCluster

```bash
nrl-k8s cluster up \
    tools/nrl_k8s/examples/qwen3_4b_if_single.yaml \
    --infra tools/nrl_k8s/examples/qwen3_4b_if_single.infra.yaml \
    --role training
```

The command applies the RayCluster CR and waits for
`.status.state == ready`. Typical times: 1–2 min image pull +
scheduling, then steady-state.

Check the head pod came up:

```bash
nrl-k8s cluster list
# → raycluster-single-qwen3-4b    ready
kubectl get pods -n $NS \
    -l ray.io/cluster=raycluster-single-qwen3-4b
# → both head + worker showing Running
```

---

## 9. Browse the Ray dashboard

```bash
nrl-k8s cluster dashboard raycluster-single-qwen3-4b
```

Port-forwards `localhost:8265` and opens your browser. First run
auto-fixes a known uv-symlink issue on the head pod (~30 s one-time
reinstall of `ray[default]` in copy mode). Pass `--no-fix` if your
image was built with `ENV UV_LINK_MODE=copy` already.

If the dashboard still renders blank, see the "Blank dashboard"
section in `README.md` — the permanent fix is in the image build.

---

## 10. Submit a dry run

Pick a mode up front:

- **interactive** (default): laptop stays attached, tails logs, exits
  on terminal state. Good for "does my recipe even start." Uploads a
  `working_dir` via Ray's SDK (100 MiB cap).
- **batch**: `kubectl exec` + `nohup` on the training head pod. Code
  must already be on disk in the pod (`launch.codeSource=image` or
  `lustre`). Returns in <30 s and the laptop can disconnect. Use this
  for real production runs.

Dev iteration:

```bash
nrl-k8s run \
    tools/nrl_k8s/examples/qwen3_4b_if_single.yaml \
    --infra tools/nrl_k8s/examples/qwen3_4b_if_single.infra.yaml \
    --mode interactive  grpo.max_num_steps=2
```

Production (using the prod-mode variant of the gym-disagg example):

```bash
nrl-k8s launch \
    tools/nrl_k8s/examples/qwen3_4b_if_gym_disagg.yaml \
    --infra tools/nrl_k8s/examples/qwen3_4b_if_gym_disagg.prod.infra.yaml \
    --run-id smoke-$(date +%s)
```

Observability is transport-aware:

```bash
nrl-k8s job logs <run-id>  <recipe> --infra <infra.yaml> --role training -f
nrl-k8s job stop <run-id>  <recipe> --infra <infra.yaml> --role training
```

---

## 11. Clean up

```bash
# Stop a specific run:
nrl-k8s job stop <run-id> <recipe> --infra <infra.yaml> --role training

# Tear down the RayCluster when you're done:
nrl-k8s cluster down <recipe> --infra <infra.yaml> --role training
# or by name:
nrl-k8s cluster down <recipe> --name raycluster-single-qwen3-4b
```

---

## Per-cluster fixture summary

| Thing | Where it lives | Source of truth |
|---|---|---|
| KubeRay operator | `kuberay-system` namespace | `infra/helm/helmfile.yaml` |
| EFA / GPU device plugins | `kube-system` namespace | `infra/helm/` |
| Your working namespace | cluster root | `kubectl create namespace` |
| `nemo-rl-endpoint-registry` SA + Role | per-namespace | RBAC snippet in §3 |
| `ngc-registry-secret` pull secret | per-namespace | `kubectl create secret docker-registry` |
| `wandb-api-key` secret | per-namespace | `kubectl create secret generic` |
| Node pool selectors / tolerations | per-cluster node labels | update `.infra.yaml` `_shared` anchors |

The `.infra.yaml` file is the single place per recipe where
cluster-specific values live. Once you've adapted the examples to
your cluster, commit them somewhere your team can share (a team
`infra/` repo, or the `nrl_k8s/examples/` dir if you contribute
back).
