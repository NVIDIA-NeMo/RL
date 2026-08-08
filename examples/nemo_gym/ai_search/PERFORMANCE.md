# AI-search performance report

These measurements answer two different questions:

1. How fast is the search path compared with straightforward alternatives on
   the same hardware and vectors?
2. Where does time go in a real NeMo RL rollout and GRPO update?

They do **not** compare model quality with published search-agent systems.

## Vector-search and RTX diagnostic setup

- Date: 2026-08-08
- GPU: NVIDIA RTX 6000D, 85,651 MiB, driver 595.58.03
- Python 3.13.14
- PyTorch 2.11.0+cu130, vLLM 0.25.1
- cuVS 26.06, CuPy 14.1.1
- Dense vectors: 384-dimensional normalized float32
- Search: top 10; latency includes host-to-device query transfer and result
  transfer, but not index build
- Reported latency: median of measured repetitions after warm-up
- NeMo RL base commit: `0e687e6d07623d780a4174310e92382ce738a8a2`
- NeMo Gym commit: `473f446f71ec7c1243eb1517fe2440a2b37fe68b`

The generated vectors are clustered around random centers and queries are noisy
copies of corpus rows. This is closer to semantic neighborhoods than isotropic
random vectors, but it is still synthetic. NumPy exhaustive search supplies
ground truth for recall.

## Vector-index comparison

### 100,000 vectors

| Backend | Batch 1 p50 | Batch 32 p50 | Batch 32 queries/s | Recall@10 |
| --- | ---: | ---: | ---: | ---: |
| NumPy CPU exact | 24.609 ms | 68.081 ms | 460 | 1.000 |
| FAISS CPU FlatIP | 4.816 ms | 16.512 ms | 1,928 | 1.000 |
| PyTorch CUDA matmul/top-k | **0.235 ms** | **0.331 ms** | **95,980** | 1.000 |
| cuVS brute force | 0.383 ms | 0.676 ms | 47,365 | 1.000 |
| cuVS CAGRA | 0.694 ms | 0.789 ms | 40,621 | 1.000 |

At this size, CAGRA's graph traversal overhead is larger than exhaustive GPU
search. The example therefore defaults to cuVS brute force for its tiny corpus.
The direct PyTorch baseline is fastest here; it is intentionally included so
the report does not assume a library call must win every small exact-search
microbenchmark.

### 1,000,000 vectors

| Backend | Batch 1 p50 | Batch 32 p50 | Batch 32 queries/s | Recall@10 |
| --- | ---: | ---: | ---: | ---: |
| NumPy CPU exact | 232.361 ms | 620.017 ms | 52 | 1.000 |
| FAISS CPU FlatIP | 17.648 ms | 144.697 ms | 195 | 1.000 |
| PyTorch CUDA matmul/top-k | 1.332 ms | 2.114 ms | 15,129 | 1.000 |
| cuVS brute force | 1.610 ms | 2.630 ms | 12,163 | 1.000 |
| cuVS CAGRA | **0.712 ms** | **0.847 ms** | **37,864** | 1.000 |

At one million rows, CAGRA is 2.26 times faster than cuVS brute force for one
query and 3.11 times faster for a batch of 32 on this workload. Its measured
recall is 1.0 with `itopk_size=256`; this is not a promise for another embedding
distribution. An isotropic stress workload produced lower CAGRA recall, which
is why the profiling script always reports recall beside speed.

## Actual E5 -> cuVS -> document pipeline

The full local path used the checked-in 32-document corpus, E5-small-v2 on the
GPU, cuVS brute force, and top 3 retrieval.

| Query batch | E5 encode | cuVS index | Fetch text | Direct wall | Direct queries/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.454 ms | 0.374 ms | 0.012 ms | 3.847 ms | 260 |
| 4 | 4.490 ms | 0.386 ms | 0.024 ms | 4.909 ms | 815 |
| 8 | 4.648 ms | 0.387 ms | 0.035 ms | 5.084 ms | 1,574 |

The encoder dominates this small-index path. A repeated query hit the in-memory
embedding cache and reduced total retrieval from a cold first-call encode of
75.66 ms to 0.351 ms. The first number includes model/CUDA warm-up and should
not be used as steady-state latency.

The asynchronous search batcher was compared with issuing one-query calls
serially:

| Concurrent queries | Serial queries/s | Microbatch queries/s | Speedup |
| --- | ---: | ---: | ---: |
| 1 | 251 | 157 | 0.63x |
| 4 | 256 | 539 | 2.11x |
| 8 | 259 | 1,056 | 4.08x |

The configured 2 ms collection window hurts a lone query but helps concurrent
rollouts. It should be reduced for latency-sensitive serving and retained or
tuned for high-throughput training.

## 7B one-step full-parameter B300 run

The default Qwen2.5-7B-Instruct recipe completed one full-parameter DTensor
GRPO step on one NVIDIA B300 SXM6 AC (275,040 MiB, driver 595.58.03). The run
used two prompts and four trajectories per prompt. Slurm job `3543660`
completed successfully in 13 minutes 1 second, including cold environment
installation and process startup.

| Stage | Time |
| --- | ---: |
| One-time NeMo RL setup | 250.11 s |
| Initial four-question validation | 16.38 s |
| Eight training rollouts / generation | 2.34 s |
| Policy and reference log probabilities | 23.26 s |
| Policy training | 14.90 s |
| Reward calculation | 0.0035 s |
| Post-update four-question validation | 0.88 s |
| Local `/tmp` full checkpoint | 164.83 s |
| Reported training-step total | 287.28 s |

NeMo RL reported 17.47 end-to-end tokens/s/GPU, 2,140.33 generation
tokens/s/GPU, 336.79 policy-training tokens/s/GPU, and 215.73 policy plus
reference-logprob tokens/s/GPU. One-second external sampling observed a peak
of 155,472 MiB (151.83 GiB) of GPU memory. Slurm reported a maximum resident
host-memory footprint of 264,028,472 KiB (251.80 GiB), essentially the entire
257,852 MiB allocation. The job completed, but a longer run should request
substantially more host-memory headroom.

The eight trajectories made nine search calls. All 17 assistant messages had
valid tool-call and thinking format, and no search failed. Every trajectory
received the same maximum reward of 1.4, however, so GRPO produced zero
advantages, zero loss, and zero gradient norm. This validates every code path
through full-parameter optimizer execution and checkpointing, but the toy step
did not provide a useful reinforcement-learning signal.

Checkpointing consumed 57.4% of the reported step; the portion of the step
excluding checkpointing was 122.45 seconds. Keep checkpoints on local storage
and reduce save frequency for throughput runs.

The following numbers contrast the full-parameter B300 run with the RTX LoRA
diagnostic below. They are useful for locating bottlenecks, not for claiming a
hardware speedup, because the update and checkpoint workloads differ.

| Metric | B300 full parameter | RTX 6000D LoRA |
| --- | ---: | ---: |
| Generation tokens/s/GPU | **2,140.33** | 1,885.67 |
| Policy + reference logprob tokens/s/GPU | **215.73** | 156.39 |
| Policy-training tokens/s/GPU | 336.79 | **1,966.77** |
| End-to-end tokens/s/GPU | 17.47 | **76.44** |
| Checkpoint time | 164.83 s | **0.31 s** |

## 7B one-step LoRA diagnostic

The current Qwen2.5-7B-Instruct path was checked on the same RTX 6000D with
DTensor V2 LoRA rank 64, two prompts, and four trajectories per prompt. This
was a smaller-GPU pipeline diagnostic; the checked-in B300 recipe remains
full-parameter training.

| Stage | Time |
| --- | ---: |
| One-time setup | 138.85 s |
| Initial four-question validation | 2.44 s |
| Eight training rollouts / generation | 2.75 s |
| Policy and reference log probabilities | 33.19 s |
| Policy training | 2.64 s |
| Reward calculation | 0.0066 s |
| Post-update four-question validation | 1.84 s |
| Local `/tmp` adapter checkpoint | 0.31 s |
| Reported training-step total | 67.89 s |

NeMo RL reported 76.44 end-to-end tokens/s/GPU, 1,885.67 generation
tokens/s/GPU, 1,966.77 policy-training tokens/s/GPU, and 156.39 policy plus
reference-logprob tokens/s/GPU. Peak sampled GPU memory was 58.81 GiB, peak
sampled host memory was 183.42 GiB, and the adapter checkpoint was 666 MiB.
The eight trajectories made nine search calls; all 17 assistant messages had
valid tool-call and thinking format, and no search failed.

All eight toy trajectories received the same maximum reward, so their
group-relative advantages, loss, and gradient norm were zero. The run proves
that 7B model loading, rollout, cuVS search, reward, log-probability, backward,
optimizer, validation, and checkpoint code paths execute; it does **not** prove
that this step changed the policy or improved quality. A learning test needs a
larger, harder corpus that produces reward variation.

A full-parameter CPU-offload attempt reached policy training but was killed at
the node's host-memory limit: total use reached 242.01 of 251.40 GiB and the
policy worker alone used 193.58 GiB. This was not a GPU OOM. It demonstrates
that CPU offload is not a substitute for the target B300-class GPU on this
host.

## Historical 1.5B one-step GRPO run

This measurement predates the example's switch to a 7B default. The end-to-end
smoke run used Qwen2.5-1.5B on one RTX 6000D, two prompts, four
trajectories per prompt, and a maximum of five Gym agent steps. It performed
real vLLM generation, structured search calls, reward calculation, reference
and policy log-probability calculation, backpropagation, optimizer update,
post-update validation, and checkpointing.

| Stage | Time |
| --- | ---: |
| One-time setup | 102.19 s |
| Initial four-question validation | 4.83 s |
| Eight training rollouts / generation | 3.13 s |
| Policy and reference log probabilities | 5.63 s |
| Policy training | 2.35 s |
| Reward calculation | 0.0036 s |
| Post-update four-question validation | 2.81 s |
| Local `/tmp` checkpoint | 23.04 s |
| Reported training-step total | 52.07 s |

Throughput reported by NeMo RL was 1,462 generated tokens/s/GPU, 1,945 policy
training tokens/s/GPU, and 87.9 end-to-end tokens/s/GPU when checkpoint and all
step overhead were included. Peak sampled GPU memory was 40.37 GiB and peak
host memory was 53.68 GiB.

The eight trajectories made four successful search calls in total; invalid tool
call rate and search error count were both zero. This proves that the complete
training path works. It does not prove learning quality: the dataset is tiny,
the generation is stochastic, and only one optimizer step was run. The apparent
validation reward change from 0.075 to 0.241 must not be interpreted as a model
improvement.

Storage materially changes the overall result. Saving the same class of
checkpoint to node-local `/tmp` took 23.04 s. An earlier run saving to the shared
filesystem took 206.57 s, dominating that step. Training runs should keep hot
checkpoints on local NVMe and copy durable artifacts asynchronously when policy
allows.

## What can and cannot be compared

The vector backends above are a valid same-process, same-vector, same-hardware
comparison. The serial-versus-microbatch result is also a controlled comparison.

An overall seconds-per-step comparison with Search-R1, ZeroSearch, ReSearch, or
ASearcher would not be valid from their published logs: model sizes, token
counts, corpora, search services, rollout lengths, GPU types, and checkpoint
policies differ. Reproducing each system with an identical model, prompt set,
token budget, and hardware is separate benchmark work. This report therefore
shows the NeMo RL end-to-end breakdown and does not manufacture a cross-project
speedup number.

The compact machine-readable result used by this report is in
`profiling/results/rtx6000d_20260808.json`. Re-run
`profile_ai_search.sh` on the target corpus and hardware before making a
deployment decision.
