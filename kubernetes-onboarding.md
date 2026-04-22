# Kubernetes Onboarding

## Prerequisites

### Install kubectl

```bash
# macOS
brew install kubectl

# Linux (amd64)
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl && sudo mv kubectl /usr/local/bin/
```

### Install kubectx and kubens

`kubectx` switches between clusters; `kubens` switches the default namespace
for the current context.

```bash
# macOS
brew install kubectx

# Linux
sudo apt install kubectx
# or from source:
git clone https://github.com/ahmetb/kubectx ~/.kubectx
ln -s ~/.kubectx/kubectx ~/.local/bin/kubectx
ln -s ~/.kubectx/kubens ~/.local/bin/kubens
```

### Install nrl-k8s

The `nrl-k8s` CLI launches NeMo-RL training jobs on Kubernetes. It lives
under `tools/nrl_k8s/` and will be available on `main` soon. Until then,
install from the `hemil/k8s-infra-cp` branch:

```bash
uv tool install "nrl-k8s @ git+https://github.com/NVIDIA-NeMo/RL.git@hemil/k8s-infra-cp#subdirectory=tools/nrl_k8s"
nrl-k8s --version
```

If you have the repo checked out locally, you can install from whatever
branch you're on:

```bash
uv tool install ./tools/nrl_k8s
```

In both cases, reinstall to pick up changes:

```bash
uv tool install --reinstall ./tools/nrl_k8s
```

### Install AWS CLI v2

Required for EKS clusters (nemo-ci-h100, aws-cmh).

```bash
# macOS
brew install awscli

# Linux
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip && sudo ./aws/install
```

---

## 1. aws-cmh (GB300)

### 1.1 Requesting access

Request access to the AWS account via one of the following DLs:

- **Admins:** [access-aws-nemo-rl-dev-admin](https://dlrequest/GroupID/Groups/Properties?identity=MzI2NjViN2ViMjdkNGQ0ZGEwMWYxYjhiMmMzN2E0NGJ8Z3JvdXA=)
- **Users:** [access-aws-nemo-rl-dev-engineer](https://dlrequest/GroupID/Groups/Properties?identity=MDVmMzI0OTc5ZDhmNDk5ZWI3MjlkY2E1ZjAxOWVkMDZ8Z3JvdXA=)

> **Note:** After your DL request is approved, permissions sync on the hour,
> so you may need to wait up to an hour before you can log in.

### 1.2 Logging in

Authenticate with AWS using `nvsec`:

```bash
nvsec aws auth
```

If you're on a remote machine (e.g. SSH'd into a devbox), use the
no-browser flag:

```bash
nvsec aws auth --no-browser
```

This will print a URL like:

```
https://awscloud.nvidia.com/cli-login?redirect_uri=http://localhost:53682/callback&state=...&client=nvsec
```

Note the port number in the URL (e.g. `53682`). Before opening the URL,
set up a port forward from your local machine so the callback can reach
the remote host:

```bash
ssh -L 53682:localhost:53682 <the-host-you-ran-nvsec-auth-from>
```

Then open the URL in your local browser. The auth should complete in your
terminal.

### 1.3 Selecting the AWS profile

List available accounts:

```bash
nvsec aws list
```

You should see `nemo-rl-dev` listed:

```
Available roles (3):

  nemo-rl-dev
  Account: 942195279341 | MPA: DGX_CLOUD_MPA
    0) CS-Admin
    1) CS-Engineer-942195279341

  NeMo_Megatron
  Account: 766267172432 | MPA: DGX_CLOUD_MPA
    2) CS-Admin
```

Configure credentials for your access level. When prompted for a profile
name, press Enter to accept the default:

```bash
nvsec aws configure 0   # CS-Admin
nvsec aws configure 1   # CS-Engineer
```

### 1.4 Setting up your kubeconfig

Add the EKS cluster to your kubeconfig:

```bash
aws eks update-kubeconfig \
  --name ltqlfcnzyr-dgxc-k8s-aws-use2-prod \
  --region us-east-2
```

The command creates a context with a long ARN name. Create a friendly alias:

```bash
kubectx aws-cmh=arn:aws:eks:us-east-2:942195279341:cluster/ltqlfcnzyr-dgxc-k8s-aws-use2-prod
```

Switch to the context:

```bash
kubectx aws-cmh
```

Verify access:

```bash
kubectl get nodes            # should list a bunch of nodes
kubectl auth can-i create rayclusters  # should print "yes"
```

### 1.5 Shared workspace (PVC)

The `default` namespace has a shared FSx Lustre PVC called `rl-workspace`
(1.2 TiB, ReadWriteMany). All pods across all nodes can read and write to
it simultaneously.

The PVC already exists — you don't need to create it. If it ever needs to
be recreated:

```bash
kubectl apply -f - <<'EOF'
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: rl-workspace
  namespace: default
spec:
  accessModes:
    - ReadWriteMany
  storageClassName: dgxc-enterprise-file
  resources:
    requests:
      storage: 1200Gi
EOF
```

> **Note:** FSx Lustre has a minimum size of 1.2 TiB. Provisioning takes
> 5–15 minutes while AWS creates the Lustre filesystem. Watch progress with:
>
> ```bash
> kubectl get pvc rl-workspace -w
> ```
>
> It will show `Pending` until the filesystem is ready, then flip to `Bound`.
> If you see `ProvisioningFailed` / `DeadlineExceeded` events, that's normal
> — the CSI driver retries automatically.

Verify:

```bash
kubectl get pvc rl-workspace  # should show Bound, 1200Gi, RWX, dgxc-enterprise-file
```

To resize (only works after the PVC is `Bound` — if you try while
`Pending`, you'll get):

```
The PersistentVolumeClaim "rl-workspace" is invalid: spec: Forbidden: spec is immutable after creation except resources.requests and volumeAttributesClassName for bound claims
```

Once bound, resize with:

```bash
kubectl patch pvc rl-workspace -p '{"spec":{"resources":{"requests":{"storage":"2400Gi"}}}}'
```

FSx Lustre grows in increments of 2.4 TiB.

This PVC is shared across the team. To avoid collisions, organize by
username:

```
/mnt/rl-workspace/
├── terryk/
│   ├── data/
│   ├── checkpoints/
│   └── hf-cache/
├── hemild/
│   └── ...
└── shared/
    └── models/
```

You can create additional PVCs if needed, but data cannot be shared across
different PVCs. To deduplicate things like HuggingFace model downloads, it's
easier to start with this shared PVC and only create a separate one if you
need isolation.

The PVC uses `reclaimPolicy: Retain`, so data persists even if pods are
deleted. Do not delete the PVC itself — it's shared across the team.

## 2. nemo-ci-h100

### 2.1 Requesting access

TODO

### 2.2 Logging in

Authenticate with AWS SSO using the `megatron` profile:

```bash
aws sso login --profile megatron
```

This will attempt to open your browser. If you're on a remote machine (e.g.
SSH'd into a devbox), the browser won't open and you'll see something like:

```
$ aws sso login --profile megatron
Attempting to open your default browser. If the browser does not open, open the following URL.
If you are unable to open the URL on this device, run this command again with the '--use-device-code' option.

https://oidc.us-east-2.amazonaws.com/authorize?response_type=code&client_id=...&redirect_uri=http%3A%2F%2F127.0.0.1%3A33977%2Foauth%2Fcallback&state=...
```

Open that URL in your local browser. It will redirect to a `127.0.0.1` callback
URL, which will fail because the redirect targets the remote machine. Note the
port number in the redirect URL (e.g. `33977`) and set up a port forward from
your local machine:

```bash
ssh -L 33977:localhost:33977 <the-host-you-ran-aws-login>
```

Then refresh the page in your browser. You should see an AWS page that says:

> Your credentials have been shared successfully and can be used until your
> session expires. You can now close this tab.

Back in your terminal, the login will complete:

```
Successfully logged into Start URL: https://nv-h100.awsapps.com/start
```

Verify the session is active:

```bash
aws sts get-caller-identity --profile megatron
```

You should see something like:

```json
{
    "UserId": "AROA3E2IVEZIKPBBBR34J:terryk@nvidia.com",
    "Account": "766267172432",
    "Arn": "arn:aws:sts::766267172432:assumed-role/AWSReservedSSO_CS-Admin_a7cbef6db22f1b0b/terryk@nvidia.com"
}
```

You'll need to re-run `aws sso login` whenever your SSO token expires
(typically every 8–12 hours).

### 2.3 Setting up your kubeconfig

Add the EKS cluster to your kubeconfig and create a friendly context name:

```bash
aws eks update-kubeconfig \
  --region us-east-1 \
  --name geyydnzzhv-dgxc-k8s-aws-use1-prod \
  --profile megatron \
  --alias nemo-ci-h100
```

Switch to the context and set the default namespace:

```bash
kubectx nemo-ci-h100
kubens nemo-rl-testing
```

Verify access:

```bash
kubectl get nodes            # should list a bunch of nodes
kubectl auth can-i create rayclusters  # should print "yes"
```

### 2.4 Shared workspace (PVC)

The `nemo-rl-testing` namespace has a shared EFS PVC called `rl-workspace`
(100 GiB, ReadWriteMany). It's backed by AWS EFS so all pods across all
nodes can read and write to it simultaneously.

The PVC already exists — you don't need to create it. If it ever needs to
be recreated:

```bash
kubectl apply -f - <<'EOF'
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: rl-workspace
  namespace: nemo-rl-testing
spec:
  accessModes:
    - ReadWriteMany
  storageClassName: efs-rwx
  resources:
    requests:
      storage: 100Gi
EOF
```

Verify:

```bash
kubectl get pvc rl-workspace  # should show Bound, 100Gi, RWX, efs-rwx
```

Use this PVC for training data, model caches, checkpoints, and code. To
avoid collisions between users, organize by username:

```
/mnt/rl-workspace/
├── terryk/
│   ├── data/           # training datasets
│   ├── checkpoints/    # training checkpoints
│   └── hf-cache/       # HuggingFace model cache
├── hemild/
│   └── ...
└── shared/
    └── models/         # models everyone uses
```

To get a shell on the head node and set up your data:

```bash
kubectl exec -it <head-pod-name> -- bash

# Inside the pod:
mkdir -p /mnt/rl-workspace/$(whoami)/{data,checkpoints,hf-cache}
```

The PVC uses `reclaimPolicy: Retain`, so data persists even if pods are
deleted. Do not delete the PVC itself — it's shared across the team.

### 2.5 Running a monolithic RayCluster

This deploys a single RayCluster with training, colocated vLLM generation,
and gym all on one node.

Validate the config:

```bash
nrl-k8s check \
  tools/nrl_k8s/examples/qwen3_4b_if_single.yaml \
  --infra tools/nrl_k8s/examples/qwen3_4b_if_single.infra.yaml
```

Bring up the cluster:

```bash
nrl-k8s cluster up \
  tools/nrl_k8s/examples/qwen3_4b_if_single.yaml \
  --infra tools/nrl_k8s/examples/qwen3_4b_if_single.infra.yaml \
  --role training
```

Submit training and tail logs. This uploads your local code (the paths
listed in `rayUploadPaths` in the infra YAML) to the Ray cluster as a
working directory, then runs the entrypoint on the cluster:

```bash
nrl-k8s launch \
  tools/nrl_k8s/examples/qwen3_4b_if_single.yaml \
  --infra tools/nrl_k8s/examples/qwen3_4b_if_single.infra.yaml \
  --follow
```

Or do everything in one step (cluster up + job submit):

```bash
nrl-k8s run \
  tools/nrl_k8s/examples/qwen3_4b_if_single.yaml \
  --infra tools/nrl_k8s/examples/qwen3_4b_if_single.infra.yaml \
  --follow
```

Check status:

```bash
nrl-k8s status \
  tools/nrl_k8s/examples/qwen3_4b_if_single.yaml \
  --infra tools/nrl_k8s/examples/qwen3_4b_if_single.infra.yaml
```

List and inspect jobs:

```bash
nrl-k8s job list \
  tools/nrl_k8s/examples/qwen3_4b_if_single.yaml \
  --infra tools/nrl_k8s/examples/qwen3_4b_if_single.infra.yaml \
  --role training

nrl-k8s job logs <submission_id> \
  tools/nrl_k8s/examples/qwen3_4b_if_single.yaml \
  --infra tools/nrl_k8s/examples/qwen3_4b_if_single.infra.yaml \
  --role training
```

Tear down when done:

```bash
nrl-k8s cluster down \
  tools/nrl_k8s/examples/qwen3_4b_if_single.yaml \
  --infra tools/nrl_k8s/examples/qwen3_4b_if_single.infra.yaml \
  --role training
```

## 3. kind

TODO

## 4. Cursor setup

TODO

## 5. Claude Code setup

TODO
