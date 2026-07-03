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
