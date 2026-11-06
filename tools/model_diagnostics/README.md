# Rollout checkpoint benchmarks

The generation-only proxy exercises real prefix capture, staging and TransferQueue
snapshots without a trainer or Gym agent/resource graph. It measures per-engine
checkpoint overhead; it does not predict the full 88-node Gym preparation or
controller recovery-state cost.

## Runtime and dependencies

Use the Gym submodule pinned by the RL checkout:

```bash
git submodule update --init 3rdparty/Gym-workspace/Gym
```

The runtime includes incremental engine token/logprob reads, incremental prefix
identities with legacy restore readers, route chunk selection, direct batch
encoding, native route staging, and an optional separate TQ writer actor. Durable
staging rows remain version 2; cumulative active-prefix identities can be version
3. Restore needs the matching RL and Gym readers.

The tested stack used Python 3.13.14, vLLM 0.29.0 and TransferQueue 0.1.11 on
four-GPU nodes. The branch's TQ controller compatibility patch installs at
bootstrap. Keep the branch's dependency pins and use a compatible image.
Native route staging incorporates upstream RL commit
`681ca0463c771b413896ff6cca075b8de5da5d18`.

## Full 88-node training run

Recipe:
`examples/nemo_gym/nemotron-3.5-nano/configs/rlvr_sc_bf16_prefix_88n_writer_actor.yaml`.

It uses 32 training nodes, 32 TP=4 generation engines, four Gym nodes and 20
external judge nodes. The workload has a 131072-rollout ceiling, `max_num_seqs=512`,
256 SimpleStorage units, router replay, 300-second snapshots and three retained
trainer anchors. The pipeline has a 2 GiB payload budget, 256 MiB batch limit and
four writers. Each generation engine creates one CPU writer actor on its own node.

Configure the destination cluster using a private site overlay based on
`examples/nemo_gym/nemotron-3.5-nano/scripts/env_site_template.sh`. Supply that
cluster's image, shared filesystem, model/data/judge paths and credentials; do not
copy credentials into the repository. The existing launcher validates these inputs.

From the RL repository root, with the site variables and W&B credentials exported:

```bash
EXP_NAME=nano35-prefix-writer-88n-test \
SITE_OVERLAY=/shared/private/site.sh \
CONFIG_PATH_HOST="$PWD/examples/nemo_gym/nemotron-3.5-nano/configs/rlvr_sc_bf16_prefix_88n_writer_actor.yaml" \
NUM_TRAIN_NODES=32 NUM_GEN_NODES=32 NUM_GYM_NODES=4 \
GENRM_REPLICAS=8 NL2BASH_REPLICAS=4 \
WALLTIME=1:30:00 DRY_RUN=1 \
bash examples/nemo_gym/nemotron-3.5-nano/scripts/launch_nano35_bf16_009_prefix_checkpoint.sh
```

Inspect the generated command and mounts, then repeat with `DRY_RUN=0`. Set the
account, partition, QoS, container and results path through the destination's site
profile. Use a new `EXP_NAME` for a fresh comparison; reusing a checkpoint lineage
can resume prior state. The site profile must support the launcher's heterogeneous
Slurm allocation and external-service layout.

## Four-node generation-only proxy

Allocate one driver/load node, one TQ storage node and two four-GPU generation
nodes. All Ray nodes need the same RL/Gym source mounts and compatible worker
venvs. Run the driver within that allocation.

`--reference-config` takes an expanded YAML, rather than a defaults-only recipe.
Prepare it with the repository config loader and resolve interpolations after
setting the destination model path and required environment variables:

```python
import os
from omegaconf import OmegaConf
from nemo_rl.utils.config import load_config

config = load_config("examples/nemo_gym/nemotron-3.5-nano/configs/rlvr_sc_bf16_prefix_88n_writer_actor.yaml")
config.policy.model_name = os.environ["MODEL_PATH"]
OmegaConf.save(config, "/shared/reference.yaml", resolve=True)
```

Run that preparation with `uv run` in the allocation, where `SLURM_JOB_ID` is set.
The following command reproduces the workload used in the matched writer-actor
benchmarks. Set W&B credentials in the environment and select a fresh output path.

```bash
uv run --active --no-project python tools/model_diagnostics/prefix_cut_benchmark.py \
  --reference-config /shared/reference.yaml --output /shared/results/prefix-writer-test \
  --cut-pipeline --prefix-writer-actor --slots 4 4 \
  --router-replay --no-heap-census --no-compact-output \
  --frontend-timings --checkpoint-latency-diagnostics \
  --packed-capture-logprobs --direct-prefix-batch-encoding --output-chunk-size 32 \
  --outstanding 630 --storage-units 16 --inventories 39 \
  --warmup-seconds 60 --rounds 2 --interval-seconds 0 \
  --capture-token-targets 2450792 8912407 \
  --capture-new-token-targets 0 6461615 \
  --max-output-tokens 65536 --target-wait-seconds 1800 \
  --token-volume-tolerance 0.1 --cut-budget 300
```

Add `--dry-run` to inspect the plan. Use `--no-router-replay` for the token-only
comparison. Omit `--prefix-writer-actor` to keep PUTs in the generation process.
`--slots` controls pipeline write concurrency in this mode and also bounds writer
actor concurrency; preparation remains on one thread.

For a turn-level proxy, use `--checkpoint-level turn` without prefix pipeline,
writer-actor, direct-prefix-encoding or token-target options. It saves completed
model calls and records unfinished calls for regeneration. It does not capture
Gym agent state and does not wait for unfinished calls to finish.

## Results and interpretation

The two writer-actor runs committed both snapshots with no failed write batches:

| Router replay | Job | Snapshot 1 | Snapshot 2 |
|---|---|---:|---:|
| Off | 5232711 | 17.6 s | 32.9 s |
| On | 5233626 | 16.8 s | 28.0 s |

Snapshot 2 matched both cumulative and new-token targets within 10% on both
engines. It processed approximately 6.5–7.0 million new token positions per engine.
Snapshot 1 overshot targets and is a less precise comparison. These single trials
do not establish that router replay is intrinsically faster.

Outputs include per-round timing JSON, token-target observations, cut receipts,
verified snapshot manifests and W&B metrics. Compare actual saved token volumes,
route payload, prefix count and per-engine load alongside latency. Writer-actor
logs separate unpack/PUT work from the sender's pack, handoff and reply waits.
The queue/buffer budgets cover staging payload, not total process RSS.
