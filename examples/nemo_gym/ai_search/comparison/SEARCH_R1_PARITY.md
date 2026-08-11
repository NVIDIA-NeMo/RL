# Search-R1 training parity target

This target reproduces the Search-R1 paper-v5 NQ/HotpotQA GRPO experiment in
NeMo RL while retaining NeMo RL's trainer and NeMo Gym's service boundary. The
reference paper is [Search-R1 v5](https://arxiv.org/html/2503.09516v5), and the
reference implementation is Search-R1 commit
`2d0e225716fe3ccc071c9d020f5561548fdefc54`.

The experiment hypothesis is: with the official questions, retrieval corpus,
rollout protocol, outcome reward, and effective GRPO batches held fixed, NeMo
RL should learn to emit useful search actions and improve exact-match reward
over 500 optimizer steps.

## Fixed parity contract

| Item | NeMo RL target | Search-R1 reference |
| --- | --- | --- |
| Policy | `Qwen/Qwen2.5-7B` base | Qwen2.5-7B base |
| Training data | Official NQ + HotpotQA train split | `PeterJinGo/nq_hotpotqa_train` |
| Retrieval | 2018 Wikipedia E5 index, top 3 | Same |
| Action syntax | Case-sensitive `<search>` / `<answer>` | Same |
| Action budget | Four executable actions | Four |
| Observation | `<information>Doc N(Title: ...) ...</information>` | Same |
| Reward | Search-R1 normalized exact match only | Same |
| Observation loss | Masked because observations are user-role messages | Masked |
| GRPO group | Five rollouts per prompt | Five |
| Advantage | Group mean and sample standard deviation, no LOO | Same |
| Prompt batch | 512 prompts, 2,560 rollouts | Same |
| Optimizer mini-batch | 256 rollouts, ten updates per outer step | Public worker execution |
| Sequence/action limit | 4,096 / 500 tokens | Same |
| Sampling | Temperature 1.0, top-p 1.0 | Same |
| Optimization | AdamW, LR 1e-6, 28.5% linear warmup | Same |
| KL / clipping | KL 0.001, ratio clip 0.2 | Same |
| Run | 500 steps on 8 GPUs, save every 100 and validate every 50 | Paper plus public script |

The NeMo recipe is
`examples/nemo_gym/ai_search/grpo_qwen2_5_7b_search_r1.yaml`. The official
Parquet files are converted without changing the supplied prompt text by
`prepare_search_r1_data.py`.

## Deliberate choices where the references disagree

- The paper says 500 steps, while the repository's current top-level GRPO
  script says 1,005. This target uses the paper's 500-step result protocol.
- The paper describes a four-action budget. The implementation performs four
  executable action rounds and, if no answer was produced, one final generation
  with search disabled. The NeMo agent follows that implementation behavior.
- The paper does not report an entropy bonus, while the upstream veRL config
  inherited by the repository has an entropy coefficient of 0.001. The NeMo
  target follows the reported paper objective and does not add a trainable
  entropy bonus.
- The paper reports total/mini/micro batches of 512/256/64 and five responses
  per prompt. In the public Agent path, `n_agent=5` expands the 512 prompts to
  2,560 trajectories, while the FSDP worker shards the configured 256/64 over
  eight ranks and does not multiply them by `n_agent`. The equivalent NeMo
  settings are therefore a global train batch of 256 and local micro-batch of
  8, producing ten optimizer updates per outer step. A short system preflight
  may lower only the local micro-batch size if memory requires it; such a run is
  not a parity result.

This is protocol parity, not framework identity. Search-R1 uses its veRL fork;
this target intentionally uses NeMo RL for optimization, Ray orchestration,
weight synchronization, and checkpointing.

For a multi-node run, start the E5 retrieval server on its dedicated GPU and
export `AI_SEARCH_RETRIEVER_URL=http://<retriever-host>:8000/retrieve` in the
training allocation. If the variable is unset, the systems preflight keeps its
local default of `http://127.0.0.1:8000/retrieve`.
