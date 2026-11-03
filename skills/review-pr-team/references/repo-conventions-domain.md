# Repo conventions: domain

NeMo-RL conventions and domain facts a reviewer needs.

- **A direct dep with a `[tool.uv.sources]` entry may be the carrier of the git pin for its extra's fork; `uv lock` and diff before approving removal.** (#3235)
- **Before "backends run different versions of P", map each pin in `pyproject.toml` to its extra.** (#4607)
- **Telemetry metrics are declared beside the function that produces the key; NeMo-RL owns its attribute names.** (#4052) Details: [telemetry-metrics-declared-beside-producer](details/telemetry-metrics-declared-beside-producer.md).
- **A user guide section documents only what changes for the reader, as tables of new and renamed keys.** (#4068)
- **A research project extends core via a `*Impl` subclass + `worker_extension_cls_fqn`, as `research/template_project/` shows.** (#3408) Details: [research-project-extension-points](details/research-project-extension-points.md).
- **For logprob-source PRs, map each loss term to curr/prev/generation and check every consumer of `exp(prev − gen)` for silent no-ops.** (#3295) Details: [logprob-reuse-three-tensor-analysis](details/logprob-reuse-three-tensor-analysis.md).
- **The Megatron EP group spans `ETP*EP` per pipeline stage, which can exceed `TP*PP*CP`; size NVLink domains with `max(TP*CP, ETP*EP)`.** Details: [megatron-ep-overlaps-but-spans-wider](details/megatron-ep-overlaps-but-spans-wider.md).
- **In refit/weight-sync name checks, say whether the function gets HF or Megatron names.** DeepSeek MTP has `mtp.` only on the Megatron side (#3619). Details: [name-domain-hf-vs-megatron](details/name-domain-hf-vs-megatron.md).
