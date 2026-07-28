# In-allocation external GenRM pool

These helpers reserve part of one Slurm allocation for independent vLLM
servers while NeMo RL uses the remaining nodes for its Ray cluster. The
servers are exposed through one OpenAI-compatible load-balancer URL.

The topology is intentionally opt-in:

```text
one Slurm allocation
├── GenRM nodes: independent private Ray/vLLM replicas
└── NeMo RL nodes: one Ray cluster started by ray.sub
```

`run_in_allocation.sh` performs the lifecycle:

1. Split the allocation into GenRM and NeMo RL node lists.
2. Start each GenRM replica in its own private Ray cluster.
3. Start the load balancer and wait for every backend to become healthy.
4. Replace `__GENRM_BASE_URL__` in `COMMAND` with the load-balancer URL.
5. Export the remaining nodes as `RAY_NODELIST` and start `ray.sub`.
6. Stop the whole job if any required service exits unexpectedly.

## Required launcher environment

The recipe-specific launcher is responsible for setting:

| Variable | Purpose |
| --- | --- |
| `BASE_LOG_DIR` | Parent directory for `<job-id>-logs`. |
| `COMMAND` | NeMo RL command containing `__GENRM_BASE_URL__`. |
| `CONTAINER` | NeMo RL container used by `ray.sub` and the load balancer. |
| `GENRM_CONTAINER` | Container used by the native vLLM replicas. |
| `GENRM_MODEL` | Model path passed to vLLM. |
| `GENRM_TOOLS_DIR_HOST` | Shared-filesystem path to this directory. |
| `GENRM_VLLM_PYTHON` | Python executable from the vLLM worker environment. |
| `MOUNTS` | Mounts for the NeMo RL container; mount this directory at `/opt/nemo-rl/tools/external_genrm`. |
| `NUM_GENRM_NODES` | Nodes reserved for the external replicas. |

The usual Slurm variables (`SLURM_JOB_ID`, `SLURM_JOB_NODELIST`,
`SLURM_JOB_ACCOUNT`, `SLURM_JOB_PARTITION`, and `SLURM_SUBMIT_DIR`) must also
be present, so this script is submitted with `sbatch` rather than run on a
login node.

Optional variables include `GENRM_REPLICAS`,
`GENRM_TENSOR_PARALLEL_SIZE`, `GENRM_REASONING_PARSER`,
`GENRM_VLLM_PORT`, `GENRM_LB_PORT`, `GENRM_STARTUP_TIMEOUT`, and `RAY_SUB`.

The default topology is eight TP=8 replicas on four-GPU nodes, requiring 16
reserved nodes. A launcher must keep the replica count, tensor parallel size,
GPU count per node, and `NUM_GENRM_NODES` consistent.
