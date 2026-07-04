# Ray-managed Dynamo on Slurm

These assets run NeMo-RL training and a fixed Dynamo vLLM fleet in one
`ray.sub` allocation. They assume the workspace root:

```text
/lustre/fsw/portfolios/coreai/users/jothomson/nemo-rl-dynamo-slurm-new
```

Build the linux/amd64 squashfs on a Slurm compute node:

```bash
cd /lustre/fsw/portfolios/coreai/users/jothomson/nemo-rl-dynamo-slurm-new/RL
sbatch \
  --output=/lustre/fsw/portfolios/coreai/users/jothomson/nemo-rl-dynamo-slurm-new/logs/build-%j.out \
  examples/slurm/dynamo/build_sqsh.sub
```

The build overlays this branch on the NeMo-RL release environment, refreshes
its locked driver and Ray actor environments, installs Dynamo commit
`59358c26d0aeed19300706462b63ada25a0a6d7c` in `/opt/dynamo_venv`, validates
representative Qwen and Nemotron argument lists, and writes:

```text
images/nemo-rl-dynamo-slurm.sqsh
```

Submit the one-node two-step smoke, the dummy-weight refit verifier, or the
two-node acceptance run with:

```bash
examples/slurm/dynamo/launch.sh one-node
examples/slurm/dynamo/launch.sh refit-verifier
examples/slurm/dynamo/launch.sh two-node
```

`one-node` allocates eight physical H100s but claims two Ray GPUs: one for the
DTensor policy and one for Dynamo. `two-node` claims eight training GPUs on one
node and launches eight fixed TP1 Dynamo workers on the other. Logs, model and
build caches, results, and Ray logs remain below the workspace root.

The Nemotron fixture is an argument-contract test input, not a runnable
training recipe. It preserves the DGD parser names (`nemotron_deci` and
`nemotron_nano`), Mamba cache dtype, and advanced batching arguments without
making those model-specific choices defaults for other models.

## HSG Nemotron Nano SWE R1

The aarch64 HSG build starts from Ruit's Gym-enabled image, overlays this
checkout (including Gym `eddd5e98`), and installs the same pinned Dynamo stack
without replacing the proven NeMo/Gym environment:

```bash
ROOT=/lustre/fsw/portfolios/coreai/users/jothomson/nemo-rl-dynamo-slurm-swe
cd "${ROOT}/RL"
sbatch --output="${ROOT}/logs/build-hsg-%j.out" \
  examples/slurm/dynamo/build_sqsh_hsg.sub
```

The build validates aarch64 etcd/NATS, Dynamo 1.3.0, vLLM 0.23.0, the exact
TP4 Nemotron Nano parser/compilation argv, and that Gym imports resolve to the
overlaid `eddd5e98` source. It writes:

```text
images/nemo-rl-dynamo-swe-hsg.sqsh
```

Submit the fixed two-train-node plus one-inference-node, one-step E2E with:

```bash
examples/swe_bench/run_grpo_nano_v3_5_swe_dynamo_hsg_e2e.sh
```

The launcher uses account `coreai_tritoninference_triton3`, keeps caches,
results, and logs below the HSG workspace root, and disables W&B for acceptance.
