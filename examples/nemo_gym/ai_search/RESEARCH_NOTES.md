# Search-agent design notes

This example borrows patterns from representative open search-agent systems,
then expresses them through NeMo Gym and NeMo RL instead of copying another
trainer. The comparison was performed against the repositories linked below on
2026-08-08.

| Project | Useful idea | What this example does |
| --- | --- | --- |
| [Search-R1](https://github.com/PeterGriffinJin/Search-R1) | Interleave reasoning and multiple searches; keep retrieval behind a service boundary; mask retrieved tokens from the policy loss. | Uses a structured search tool, permits repeated searches, keeps retrieval in a Gym resources server, and trains only model-generated tokens. |
| [R1-Searcher](https://github.com/RUCAIBox/R1-Searcher) | First teach tool-call format, then optimize search usefulness. | Supplies a strict tool schema and format reward. A production run should add a format/SFT warm-up if the base model rarely calls tools. |
| [ZeroSearch](https://github.com/Alibaba-NLP/ZeroSearch) | Replace costly live search with a controllable simulator and curriculum when appropriate. | Uses a deterministic local corpus, so rollout cost and document freshness are controlled. It does not add an LLM search simulator. |
| [ReSearch](https://github.com/Agent-RL/ReSearch) | Treat retrieval and other tools as external services and score long, multi-hop behavior. | Separates the Gym agent, resources server, retriever, and verifier. The toy task is short-horizon, but the interfaces do not assume one search. |
| [ASearcher](https://github.com/inclusionAI/ASearcher) | Async rollout avoids making a whole batch wait for its longest search trajectory. | Microbatches concurrent search calls asynchronously. Full trainer-level asynchronous rollout is left to NeMo RL's async recipes. |
| [SimpleTIR](https://github.com/ltzheng/SimpleTIR) | Invalid or empty tool trajectories can destabilize multi-turn RL and should be detected or filtered. | NeMo RL masks flagged samples; the run logs invalid tool calls, malformed thinking, search errors, and natural termination. |
| [Tongyi DeepResearch](https://github.com/Alibaba-NLP/DeepResearch) | Long-horizon search benefits from on-policy updates, leave-one-out advantages, and explicit stabilization. | Uses on-policy GRPO, a leave-one-out baseline, a reference-policy KL penalty, and a bounded search budget. |

## Decisions

### Structured tools instead of parsing `<search>` text

The model receives an OpenAI Responses API function definition and emits a
structured function call. NeMo Gym dispatches it. This avoids relying on a
regular expression to find search text inside free-form model output and lets
the server validate `query` and `top_k` independently.

### Local retrieval instead of a third-party API

The user asked for private indexes. The search service therefore has no API key,
network dependency, or per-query fee. The same service boundary can later wrap
BM25/Lucene, a company search endpoint, or an online provider without changing
GRPO.

### Reward components instead of exact match alone

An exact-answer reward is cheap but ambiguous: the model can improve it by
becoming terse without improving search. This verifier reports token-F1,
supporting-document recall, format, efficiency, duplicate queries, errors, and
latency separately. GRPO receives their weighted scalar sum; the individual
components remain visible for diagnosis and can be used by a multi-reward
algorithm later.

### cuVS brute force and CAGRA

Exact brute force is the default for the tiny example and is a sound baseline
for small collections. CAGRA is implemented for larger collections. The checked
in profiler measures both latency and recall because approximate-index speed is
not meaningful without search quality.

### Microbatch concurrent searches

Four rollouts for the same question often reach the search tool at nearly the
same time. The resources server waits up to 2 ms and combines those calls into
one E5/cuVS batch. On the measured RTX 6000D workload, eight concurrent queries
were 4.08 times faster than issuing eight one-query searches serially. The wait
hurts an isolated single query, so it remains configurable.

## Ideas deliberately not copied

- Public web-search APIs: they make training dependent on credentials, rate
  limits, cost, and a changing index.
- LLM-as-judge as the only reward: it is slower and harder to reproduce than
  rule-based QA/retrieval checks for this example.
- Unbounded long-horizon search: it is valuable for deep research but makes a
  one-GPU example expensive and increases reward-hacking surface area.
- Search simulation by another LLM: useful when real retrieval is costly, but
  unnecessary for a local cuVS index.
- Reported headline accuracy from other projects: models, data, retrievers,
  turn budgets, and hardware differ, so those numbers are not a performance
  baseline for this implementation.

## Production follow-ups

1. Add a cold-start set of valid search trajectories if tool-use rate is low.
2. Replace the toy corpus with a versioned train corpus and a disjoint held-out
   test corpus.
3. Implement the organization's inverted-index adapter behind `SearchProvider`
   and evaluate hybrid fusion/reranking.
4. Tune CAGRA against measured recall on real embeddings.
5. Scale asynchronous rollout and retriever workers independently when long-tail
   tool latency starts idling generation GPUs.
6. Run multiple seeds and compare answer quality, retrieval quality, and format
   validity before claiming a training improvement.
