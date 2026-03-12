# Interactive Ray Cluster on Slurm

> [!TIP]
> A key advantage of running interactively is the ability to execute **multiple multi-node jobs without re-queuing** in the Slurm job queue. During a debugging session you can submit, cancel, and re-submit NeMo RL jobs directly from the head-node shell — paying the queue wait time only once.

`ray-interactive.sub` starts a Ray cluster on your Slurm allocation, waits until all workers have connected, then prints connection instructions and **idles indefinitely**. You attach to the head node at any time and submit jobs manually.

This is the recommended approach when:

- Iterating quickly on configs or debugging failures
- Running a series of ablations back-to-back without re-queuing
- Exploring the cluster interactively with `ray status` or custom scripts

---

## 1. Submit the Job

```sh
# Run from the root of the NeMo RL repo
NUM_NODES=2   # Total nodes (head + workers share the allocation)

CONTAINER=YOUR_CONTAINER \
MOUNTS="$PWD:$PWD" \
sbatch \
    --nodes=${NUM_NODES} \
    --account=YOUR_ACCOUNT \
    --job-name=YOUR_JOBNAME \
    --partition=YOUR_PARTITION \
    --time=4:0:0 \
    --gres=gpu:8 \
    ray-interactive.sub
```

> [!TIP]
> Depending on your Slurm cluster configuration, you may or may not need to include `--gres=gpu:8`. For GB200 systems with 4 GPUs per node, use `--gres=gpu:4`.

> [!NOTE]
> All environment variables supported by `ray.sub` (e.g. `GPUS_PER_NODE`, `SETUP_COMMAND`, `UV_CACHE_DIR_OVERRIDE`, `RAY_LOG_SYNC_FREQUENCY`, `SANDBOX_CONTAINER`) are also supported by `ray-interactive.sub`. See the [cluster documentation](../cluster.md#slurm-environment-variables) for the full list.

Upon successful submission, Slurm will print the job ID:

```text
Submitted batch job 1980204
```

---

## 2. Wait for the Cluster to Come Up

The SLURM job stdout (`slurm-<JOB_ID>.out` by default) streams startup progress. Watch it with:

```sh
tail -f slurm-1980204.out
```

The head-node Ray log is also useful if you want to see `ray start` output in detail:

```sh
tail -f 1980204-logs/ray-head.log
```

Once all worker nodes have connected you will see the ready banner in the job stdout:

```text
============================================================
 Ray cluster is READY (job 1980204)
 Head node : gpu-node-001 (10.0.0.1)
 Ray address: 10.0.0.1:9900
 Dashboard  : http://10.0.0.1:8265
 Logs       : /path/to/submit/dir/1980204-logs
============================================================

Attach to the head node with:
  bash /path/to/submit/dir/1980204-attach.sh

Or run a one-shot command on the head node:
  COMMAND='python my_script.py' bash /path/to/submit/dir/1980204-attach.sh

Attach to worker N (1-indexed):
  bash /path/to/submit/dir/1980204-attach.sh <N>
```

---

## 3. Attach to the Head Node

Once the banner appears, an attach script is written to your submission directory. Run it from any login node:

```sh
bash 1980204-attach.sh
```

This opens an **interactive PTY shell** inside the container running on the Ray head node. From there you can submit jobs, inspect the cluster, or run debugging commands.

---

## 4. Submit Jobs from the Interactive Shell

Inside the head-node shell, submit jobs exactly as you would via `COMMAND=` in a batched run:

```sh
# Verify the cluster is healthy first
ray status

# Run a NeMo RL training job
uv run ./examples/run_grpo.py --config examples/configs/grpo_math_1B.yaml

# Run another job after the first finishes — no re-queuing needed
uv run ./examples/run_grpo.py --config examples/configs/grpo_math_1B_run2.yaml
```

> [!TIP]
> Because the Ray cluster stays alive between runs, subsequent jobs skip the cluster startup overhead (typically 2–3 minutes) and begin nearly instantly.

---

## 5. Attach Script Usage Reference

The generated `<JOB_ID>-attach.sh` supports several modes, all run from **outside** the container (on a login node):

| Command | Effect |
|---|---|
| `bash <JOB_ID>-attach.sh` | Interactive shell on the **head node** |
| `bash <JOB_ID>-attach.sh <N>` | Interactive shell on **worker N** (1-indexed) |
| `COMMAND='...' bash <JOB_ID>-attach.sh` | Run a one-shot command on the head node (non-interactive) |
| `COMMAND='...' bash <JOB_ID>-attach.sh <N>` | Run a one-shot command on worker N (non-interactive) |

Examples:

```sh
# Verify cluster health non-interactively
COMMAND="ray status" bash 1980204-attach.sh

# Inspect GPU usage on worker 1
COMMAND="nvidia-smi" bash 1980204-attach.sh 1

# Open an interactive shell on worker 2
bash 1980204-attach.sh 2
```

---

## 6. Graceful Shutdown

The cluster runs until the Slurm time limit expires. To terminate it early:

```sh
touch 1980204-logs/ENDED
```

All node sidecars detect this file within ~60 seconds and terminate cleanly. The Slurm job will then exit.

Alternatively, cancel the job via Slurm:

```sh
scancel 1980204
```

---

## 7. Smoke Test Checklist

Use this checklist to confirm the cluster is working correctly before submitting a real training run:

- [ ] Job transitions to `RUNNING` state: `squeue -j 1980204`
- [ ] Head log shows `ray start --head` completing: `tail -f 1980204-logs/ray-head.log`
- [ ] Worker logs show `ray start --address` completing: `tail -f 1980204-logs/ray-worker-1.log`
- [ ] Job stdout shows the "Ray cluster is READY" banner: `tail -f slurm-1980204.out`
- [ ] Attach script is present: `ls 1980204-attach.sh`
- [ ] Attach script opens a shell: `bash 1980204-attach.sh`
- [ ] `ray status` inside the shell shows all nodes and correct `worker_units` count
- [ ] A minimal Ray task runs successfully:

```python
import ray
ray.init(address="auto")

@ray.remote
def hello():
    return "hello from Ray"

print(ray.get(hello.remote()))
ray.shutdown()
```

- [ ] `touch 1980204-logs/ENDED` terminates the job cleanly
