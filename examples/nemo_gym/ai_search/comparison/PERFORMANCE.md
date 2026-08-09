# Aligned AI-search training performance

## Result

NeMo RL completed the lowest-latency optimizer step, but it did not have the
highest end-to-end output-token throughput. Its measured core step was 1.63x
faster than Search-R1 and 1.70x faster than ZeroSearch. The NeMo Gym agent also
generated substantially shorter trajectories, however, so the normalized
end-to-end rates were 77.54, 81.83, and 87.61 model-generated tokens/s for
NeMo RL, Search-R1, and ZeroSearch respectively.

The implementations have different dominant costs:

- Search-R1 and ZeroSearch spend about 94% of the core step in multi-turn
  rollout generation. Search, reward, advantage calculation, logprobs, and the
  actor update are not material bottlenecks for this workload.
- NeMo RL's rollout path is 4.8-5.2x faster in output tokens/s, but synchronizing
  the updated DTensor policy back to four vLLM workers takes 7.45 s, or 44.0% of
  its core step. Training preparation and logprob preparation add another 23.8%.
- The shared BM25 service is not a bottleneck. Measured trainer-side search is
  0.11% of a Search-R1 step and 0.39% of a ZeroSearch step. NeMo includes HTTP
  search in generation; the shared server spent only 20.18 ms on all 89 queries
  across NeMo's three measured steps.

## Controlled workload

All formal runs used the same physical node and the following contract:

| Item | Value |
| --- | --- |
| Hardware | 4 x NVIDIA A100-SXM4-80GB; driver 595.58.03 |
| Policy | `Qwen/Qwen2.5-3B-Instruct`, BF16 full-parameter GRPO |
| Data | The same 32-train/8-validation synthetic two-hop questions and answers |
| Retrieval | The same 160-document deterministic BM25 service, top-3 |
| Optimizer step | 4 prompts x 4 trajectories = 16 samples |
| Agent budget | At most 3 model calls: 2 searchable rounds and 1 answer-only call |
| Sampling | `temperature=1.0`, `top_p=0.95`, at most 256 new tokens per call |
| Reward | Normalized answer exact match |
| Measurement | Step 1 warm-up; arithmetic mean of core steps 2-4 |

The runs were sequential on node `ro-prod-01-80gb`. Search-R1 and ZeroSearch
ran in Slurm job `3555954`; the clean NeMo core-step run was job `3555955`.
Checkpoint and validation costs are excluded from the core-step percentages.

## End-to-end comparison

Output tokens count only tokens sampled by the policy. Search observations
injected into the response sequence are excluded. Throughput is the aggregate
for the four-GPU job, not a per-GPU number.

| Implementation | Core step (s) | Output tokens/sample | Rollout output tok/s | E2E output tok/s | Mean exact match | Peak GPU/rank | Peak host |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NeMo RL AI-search | **16.94** | 82.06 | **450.74** | 77.54 | 0.917 | **42.21 GiB** | 168.09 GiB |
| Search-R1 | 27.53 | 140.35 | 87.27 | 81.83 | 0.292 | 51.83 GiB | **99.21 GiB** |
| ZeroSearch | 28.77 | 157.58 | 93.29 | **87.61** | 0.292 | 51.98 GiB | 99.05 GiB |

NeMo RL's wall-time advantage is real for this exact workload: the same batch
reaches an optimizer update 10.59 s earlier than Search-R1 and 11.82 s earlier
than ZeroSearch. It is not a framework-only speedup. The structured-tool agent
produced 42-48% fewer model tokens per trajectory than the tag-based veRL
agents, and NeMo uses a much newer serving stack. The output-token columns are
therefore required beside wall time.

## Core-step breakdown

Each cell is mean seconds followed by share of the measured core step. Nested
timers are counted once: veRL generation excludes its search child timer, and
NeMo's aggregate policy-plus-reference timer is replaced by its two leaf
timers.

| Stage | NeMo RL | Search-R1 | ZeroSearch |
| --- | ---: | ---: | ---: |
| Generation/agent loop | 2.937 / 17.33% | 25.804 / 93.72% | 26.902 / 93.51% |
| Search | included above | 0.031 / 0.11% | 0.112 / 0.39% |
| Policy logprob | 0.263 / 1.55% | 0.171 / 0.62% | 0.161 / 0.56% |
| Reference logprob | 0.720 / 4.25% | 0.169 / 0.61% | 0.171 / 0.59% |
| Reward + advantage | 0.010 / 0.06% | 0.006 / 0.02% | 0.007 / 0.02% |
| Actor backward/optimizer | 1.440 / 8.50% | 1.351 / 4.91% | 1.412 / 4.91% |
| Weight sync to generation | **7.447 / 43.95%** | included in generation boundary | included in generation boundary |
| Logprob-mode preparation | 1.673 / 9.88% | included in framework boundaries | included in framework boundaries |
| Training-mode preparation | 2.367 / 13.97% | included in framework boundaries | included in framework boundaries |
| Other | 0.087 / 0.51% | 0.002 / 0.01% | 0.002 / 0.01% |

The NeMo categories expose state transitions more finely than the older veRL
forks. Consequently, the table identifies each implementation's actionable
bottleneck but should not be interpreted as identical profiler call stacks.

## Retrieval and checkpoint costs

Search-R1 batches a search round into one HTTP request. ZeroSearch retains its
native one-request-per-query path, explaining why its client-side search timer
is about 3.6x larger, although both remain negligible.

| Implementation | Measured search calls/step | Measured client search (ms/step) | Shared-server evidence |
| --- | ---: | ---: | --- |
| NeMo RL | 29.67 queries | included in generation | 31 requests / 89 queries / 20.18 ms total |
| Search-R1 | 31.00 queries | 31.0 | 12 requests / 155 queries / 32.06 ms over its formal train-and-validation stage |
| ZeroSearch | 30.33 queries | 112.3 | 138 requests / 138 queries / 38.28 ms over its formal train-and-validation stage |

Checkpointing was measured separately because it occurred after the fourth
core step. Search-R1 took 33.23 s and ZeroSearch 32.80 s. A separate NeMo
diagnostic on the same node and workload took 40.33 s; its initial and final
eight-sample validations took 2.55 s and 1.83 s. The post-run directories were
approximately 13 GiB for each veRL fork and 36 GiB for NeMo, but their saved
state and checkpoint formats differ, so these numbers are operational costs,
not serializer microbenchmarks.

## Bottlenecks and next experiments

For NeMo RL, the first optimization target is the 7.45 s policy-to-vLLM weight
sync. Reducing refit fan-out, avoiding a full per-step transfer when possible,
or overlapping transfer with trainer preparation has a much larger ceiling
than optimizing the 10 ms reward or the BM25 service. The next targets are the
2.37 s training transition and 1.67 s logprob transition.

For Search-R1 and ZeroSearch, generation is the only first-order bottleneck.
The actor update is already about 1.4 s, and the combined measured search,
reward, advantage, and two logprob passes are below 0.5 s. Improvements should
focus on the older vLLM path, turn scheduling, prefix reuse, and batching across
agent rounds. Batching ZeroSearch retrieval would improve cleanliness and
tail latency but cannot materially change total training time on this corpus.

## Interpretation limits

- This compares complete runnable implementations, not trainer kernels in
  isolation. NeMo used Python 3.13, Torch 2.11, vLLM 0.25.1, and DTensor;
  Search-R1 and ZeroSearch used their compatible Python 3.10, Torch 2.4,
  vLLM 0.6.3, FSDP-era stack.
- Native agent syntax was preserved. NeMo used structured function calls;
  Search-R1 and ZeroSearch used `<search>` and `<answer>` tags. The semantic
  questions, answers, corpus, limits, model, reward, and hardware were aligned.
- Search-R1 and ZeroSearch are both veRL derivatives. They are useful agent-loop
  comparisons, not three independent training frameworks.
- Three measured steps and a 160-document synthetic corpus are sufficient for
  performance diagnosis, not a quality or scaling claim. Reward is reported to
  show the runs were non-degenerate, not to rank model quality.

## Reproducibility

The final benchmark commits are:

- NeMo RL: `dea8dd0610ca7dedf0e5d91f0d5dbbe433efa5e1`
- Search-R1: `2d0e225716fe3ccc071c9d020f5561548fdefc54`
- ZeroSearch: `5a09dbdc517bd6a9f4f459058de56b59359cc03f`

The Search-R1/ZeroSearch campaign manifest captured their preceding committed
HEADs because the final alignment patches were committed after submission; the
working-tree code executed by job `3555954` is the content subsequently fixed
in the commits above. The resolved configs and logs confirm five veRL loop
indices for four optimizer updates, two searchable rounds, and an answer-only
final rollout.

Machine-readable, per-step and aggregate results are in
[`a100_sxm4_4x_20260809.json`](a100_sxm4_4x_20260809.json).
Raw logs and one-second resource samples are retained under
`session/20260808_013543/aligned-campaign/job-3555954` and
`session/20260808_013543/aligned-campaign/job-3555955`.
