# Repo conventions: config and domain

NeMo-RL conventions and domain facts a reviewer needs.

- **Never suggest `.get(key, default)` for a required key or invariant; a KeyError there is the feature, at most ask for a clearer message.** (#3308, #3652)
- **An "add it to the exemplar YAML" ask must also name the `tests/unit/reference_configs/` twin, and one exemplar is enough.** (#3798)
- **Check recipe-name `tpN/cpN/epN` tokens against the resolved `defaults:` chain's training config.** (#4701)
- **A direct dep with a `[tool.uv.sources]` entry may be the carrier of the git pin for its extra's fork; `uv lock` and diff before approving removal.** (#3235)
- **Before "backends run different versions of P", map each pin in `pyproject.toml` to its extra.** (#4607)
- **Telemetry metrics are declared beside the function that produces the key; NeMo-RL owns its attribute names.** (#4052)
- **Flag 5+ line comment blocks inside function bodies (not on declarations), listing each with a one-line replacement.** Five were already wrong before merge (#4052).
- **A user guide section documents only what changes for the reader, as tables of new and renamed keys.** (#4068)
- **A research project extends core via a `*Impl` subclass + `worker_extension_cls_fqn`, as `research/template_project/` shows.** (#3408)
- **For logprob-source PRs, map each loss term to curr/prev/generation and check every consumer of `exp(prev − gen)` for silent no-ops.** (#3295)
- **The Megatron EP group spans `ETP*EP` per pipeline stage, which can exceed `TP*PP*CP`; size NVLink domains with `max(TP*CP, ETP*EP)`.**
- **In refit/weight-sync name checks, say whether the function gets HF or Megatron names.** DeepSeek MTP has `mtp.` only on the Megatron side (#3619).
