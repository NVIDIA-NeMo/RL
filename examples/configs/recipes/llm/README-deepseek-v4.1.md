# DeepSeek V4.1 DAPO

The colocated DAPO recipe supports BF16 compute, frozen host Engram,
MXFP8 compressed KV, router replay, CPU parameter offload and BF16CPUAdamW.
See the YAML for topology, batch sizes, evaluation and checkpoint settings.
The checkpoint deadline is an application setting, not a scheduler time limit.

From the `code/` root, inside the matching distributed CUDA environment:

```bash
export DS41_FULL_CHECKPOINT=/path/to/verified/full-checkpoint
bash examples/run_deepseek_v41_stream_adam.sh
```

The launcher accepts ordinary configuration overrides. Dataset loading uses
DAPO/AIME datasets; no `DS41_SMOKE_DATA` file is required. Cluster allocation,
containers, Ray startup, dataset cache and W&B credentials are provided by
the deployment environment.

AutoModel selects the GPU-gradient/norm/streaming-update lifecycle from
`BF16CPUAdamW`. Local CPU checks do not establish GPU performance or convergence;
validate the complete configuration in the target environment.

The non-colocated smoke recipe remains available but is not selected by the launcher.
