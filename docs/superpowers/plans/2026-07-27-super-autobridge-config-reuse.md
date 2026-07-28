# Super AutoBridge Config Reuse Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reuse the already validated Hugging Face config in the pinned
Megatron-Bridge line, then prove Nemotron3 Super 120B can complete the exact
32n4g HybridEP recipe for 20 GRPO steps.

**Architecture:** Backport only the three-line config-reuse behavior from
upstream Megatron-Bridge commit `327bcc73`, with an identity-based unit test
that fails if lazy config access triggers a second load. Publish the Bridge
commit on a dedicated `seonjinn/Megatron-Bridge` branch, advance only the
NeMo-RL Bridge submodule pointer, run a one-node OCI-HSG preflight, and then
submit the unchanged 128-GPU Super profile.

**Tech Stack:** Python 3.13, pytest, Hugging Face Transformers,
Megatron-Bridge, NeMo-RL, Git submodules, Bash, SLURM, Ray, OCI-HSG GB200.

## Global Constraints

- Base NeMo-RL branch:
  `sna/qwen30-pr2964-f725-pin-oci-20260727`.
- Base Megatron-Bridge commit:
  `45e4e4be2591186ac795eea4205c44089b45fcfd`.
- Keep Megatron-LM pinned to
  `4d04e7625c5e84f984a9f01aef58cb006b0aa7ac`.
- Keep source-native DeepEP pinned to
  `f725d29699f5bda9ba789456bb9579af69844685`.
- Keep the nightly image SHA256
  `5e9f6066897057d8701e0722a5023c08a997f10f4eff61340c249ed73f7c33fc`.
- Do not cherry-pick the complete upstream 73-file deprecation commit.
- Do not alter the Super recipe, model, topology, parallelism, or 20-step gate.
- Run `sbatch --test-only`, choose the highest current FairShare account, and
  monitor the submitted job for at least five minutes.

---

### Task 1: Backport Config Reuse with TDD

**Files:**
- Modify:
  `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/tests/unit_tests/models/test_auto_bridge.py`
- Modify:
  `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src/megatron/bridge/models/conversion/auto_bridge.py`

**Interfaces:**
- Consumes:
  `AutoBridge.from_hf_pretrained(path: Union[str, Path], **kwargs) -> AutoBridge`
- Produces: a returned `AutoBridge` whose
  `hf_pretrained.config is validated_config`.

- [ ] **Step 1: Create the Bridge branch**

```bash
git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge \
  switch -c sna/super-autobridge-config-reuse-20260727 \
  45e4e4be2591186ac795eea4205c44089b45fcfd
```

- [ ] **Step 2: Write the failing identity test**

Add this test beside the existing `from_hf_pretrained` tests:

```python
def test_from_hf_pretrained_reuses_validated_config(self):
    class NoReloadPreTrainedCausalLM(PreTrainedCausalLM):
        def __init__(self):
            pass

        def _load_config(self):
            raise AssertionError("validated config was loaded a second time")

    validated_config = Mock(spec=PretrainedConfig)
    validated_config.architectures = ["GPT2LMHeadModel"]
    lazy_wrapper = NoReloadPreTrainedCausalLM()

    with (
        patch(
            "megatron.bridge.models.conversion.auto_bridge.safe_load_config_with_retry",
            return_value=validated_config,
        ),
        patch(
            "megatron.bridge.models.conversion.auto_bridge.PreTrainedCausalLM.from_pretrained",
            return_value=lazy_wrapper,
        ),
        patch.object(AutoBridge, "_validate_config"),
    ):
        bridge = AutoBridge.from_hf_pretrained("gpt2", trust_remote_code=True)

    assert bridge.hf_pretrained.config is validated_config
```

- [ ] **Step 3: Run the test and verify RED**

```bash
cd 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
uv run pytest -q \
  tests/unit_tests/models/test_auto_bridge.py::TestAutoBridge::test_from_hf_pretrained_reuses_validated_config
```

Expected: FAIL with
`AssertionError: validated config was loaded a second time`.

- [ ] **Step 4: Apply the minimal upstream behavior**

Replace the direct return in `AutoBridge.from_hf_pretrained` with:

```python
hf_pretrained = PreTrainedCausalLM.from_pretrained(path, **kwargs)
hf_pretrained.config = config
return cls(hf_pretrained)
```

Keep the existing `try`/`except` and loader arguments unchanged.

- [ ] **Step 5: Run focused and adjacent tests**

```bash
uv run pytest -q \
  tests/unit_tests/models/test_auto_bridge.py::TestAutoBridge::test_from_hf_pretrained_reuses_validated_config \
  tests/unit_tests/models/test_auto_bridge.py::TestAutoBridge::test_from_hf_pretrained_with_model_id \
  tests/unit_tests/models/test_auto_bridge.py::TestAutoBridge::test_from_pretrained_with_additional_kwargs \
  tests/unit_tests/models/test_auto_bridge.py::TestAutoBridge::test_kwargs_passed_through
```

Expected: 4 passed.

- [ ] **Step 6: Commit and push the Bridge branch**

```bash
git add \
  src/megatron/bridge/models/conversion/auto_bridge.py \
  tests/unit_tests/models/test_auto_bridge.py
git commit -s -m "fix: reuse validated config in AutoBridge"
git push -u origin sna/super-autobridge-config-reuse-20260727
```

### Task 2: Advance the NeMo-RL Submodule Pointer

**Files:**
- Modify: `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge`

**Interfaces:**
- Consumes: the pushed Bridge commit from Task 1.
- Produces: a recursively reproducible NeMo-RL checkout with the same
  Megatron-LM pin.

- [ ] **Step 1: Verify recursive commit invariants**

```bash
git -C 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge rev-parse HEAD
git -C \
  3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM \
  rev-parse HEAD
```

Expected: the first command prints the new Bridge commit and the second prints
`4d04e7625c5e84f984a9f01aef58cb006b0aa7ac`.

- [ ] **Step 2: Run NeMo-RL launcher regressions**

```bash
python3 -m pytest -q --confcutdir=tests/unit/tools \
  tests/unit/tools/test_hybridep_submit_grpo.py
bash -n scripts/experiments/oci-hsg/hybridep/submit_grpo.sh
uv lock --check
git diff --check
```

Expected: all tests pass, shell syntax passes, 515 packages resolve, and the
diff check is clean.

- [ ] **Step 3: Commit and push only the pointer**

```bash
git add 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
git commit -s -m "fix: pin Bridge AutoBridge config reuse"
git push fork sna/qwen30-pr2964-f725-pin-oci-20260727
```

### Task 3: Run the OCI-HSG Preflight

**Files:**
- Read:
  `scripts/experiments/oci-hsg/hybridep/models/nemotron3-super-120ba12b-32n4g-async-1off.env`
- Create remotely under the run artifact directory: `submission.env`,
  the job-specific SLURM output, and Ray logs.

**Interfaces:**
- Consumes: the pushed NeMo-RL and Bridge commits.
- Produces: evidence that the actual mcore worker environment reuses the
  validated Nemotron-H config without a second load.

- [ ] **Step 1: Refresh the clean OCI checkout**

```bash
ssh oci-hsg-cs-001-vscode-02 '
  cd /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/nemo-rl-qwen30-hybridep-oci-20260727
  git pull --ff-only
  git submodule sync --recursive
  git submodule update --init --recursive
  git status --short
'
```

Expected: the pushed NeMo-RL commit and a clean recursive checkout.

- [ ] **Step 2: Submit a one-node refit/config preflight**

Use the same nightly and mcore environment as job `5641462`, loading
`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` at exact snapshot
`d51eab0d1f979ebc26b546e634a04f450d99158e`. The test must construct
`AutoBridge.from_hf_pretrained`, access `_causal_lm_architecture`, and assert
that `bridge.hf_pretrained.config` is the same object returned by the initial
safe config load.

Before submission:

```bash
fairshare_account="$(
  sshare -a --user=sna -o Account,User,FairShare -n -P |
    awk -F'|' '$2 == "sna" && $3 + 0 > best {best = $3 + 0; account = $1} END {print account}'
)"
preflight_script="$PWD/scripts/experiments/oci-hsg/hybridep/preflight_super_autobridge.sub"
sbatch --test-only --nodes=1 --gres=gpu:4 --segment=1 \
  --partition=batch --account="${fairshare_account}" "${preflight_script}"
```

Expected: test-only accepted, then `COMPLETED 0:0` with the expected
`NemotronHForCausalLM` and `NemotronHBridge` mapping.

### Task 4: Run Super 120B for 20 Steps

**Files:**
- Read:
  `examples/configs/recipes/llm/performance/grpo-nemotron3-super-120BA12B-32n4g-async-1off.yaml`
- Read:
  `scripts/experiments/oci-hsg/hybridep/models/nemotron3-super-120ba12b-32n4g-async-1off.env`

**Interfaces:**
- Consumes: successful Task 3 preflight.
- Produces: terminal 20-step Super evidence and performance metrics.

- [ ] **Step 1: Query FairShare**

```bash
ssh oci-hsg-cs-001-vscode-02 \
  "sshare -a --user=sna -o Account,User,FairShare -n -P"
```

Select the highest user-level FairShare row at submission time.

- [ ] **Step 2: Submit through the reusable launcher**

```bash
ssh oci-hsg-cs-001-vscode-02 '
  cd /lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/nemo-rl-qwen30-hybridep-oci-20260727
  RUN_NAME=nemotron3-super-120ba12b-32n4g-async-1off-hybridep-f725-configreuse \
  RUN_SUFFIX=configreuse \
  WANDB_ENABLED=False \
  scripts/experiments/oci-hsg/hybridep/submit_grpo.sh \
    scripts/experiments/oci-hsg/hybridep/models/nemotron3-super-120ba12b-32n4g-async-1off.env
'
```

The launcher must report an accepted `sbatch --test-only`, 32 nodes × 4 GPUs,
segment 8, `NCCL_NVLS_ENABLE=0`, highest FairShare account, and one canonical
job ID.

- [ ] **Step 3: Monitor for at least five minutes and past refit metadata**

```bash
ssh oci-hsg-cs-001-vscode-02 '
  super_run_root=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/nemo-rl-qwen30-hybridep-oci-20260727/exp_logs/hybridep/nemotron3-super-120ba12b-32n4g-async-1off/nemotron3-super-120ba12b-32n4g-async-1off-hybridep-f725-configreuse
  source "${super_run_root}/submission.env"
  squeue -j "${job_id}" -o "%i|%T|%R|%S|%M"
  tail -n 120 "${super_run_root}/slurm-${job_id}.out"
'
```

Check for `No architectures found`, traceback, HybridEP error, NCCL timeout,
CUDA illegal address, OOM, rank loss, NaN, or Inf. Do not dump full logs.

- [ ] **Step 4: Verify the terminal gate**

```bash
ssh oci-hsg-cs-001-vscode-02 '
  super_run_root=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/nemo-rl-qwen30-hybridep-oci-20260727/exp_logs/hybridep/nemotron3-super-120ba12b-32n4g-async-1off/nemotron3-super-120ba12b-32n4g-async-1off-hybridep-f725-configreuse
  source "${super_run_root}/submission.env"
  sacct -j "${job_id}" -X -n -P \
    --format=JobID,State,ExitCode,Elapsed,NodeList
'
```

Expected: `COMPLETED|0:0`, 20 completed training steps, and no fatal error.

### Task 5: Report and Publish

**Files:**
- Modify:
  `/Users/sna/nemo-rl_release_perf_investigator/public/qwen3-30ba3b-hybridep-oci-20260727/index.html`

**Interfaces:**
- Consumes: preflight and full-run evidence.
- Produces: the validated, secret-free HTML status report.

- [ ] **Step 1: Add the exact Bridge commit, job status, step count, elapsed
  time, failure scan, and Super performance metrics**

- [ ] **Step 2: Validate HTML and scan for credentials**

```bash
python3 - <<'PY'
from html.parser import HTMLParser
from pathlib import Path

page = Path(
    "/Users/sna/nemo-rl_release_perf_investigator/"
    "public/qwen3-30ba3b-hybridep-oci-20260727/index.html"
)
HTMLParser().feed(page.read_text())
print("HTMLParser: PASS")
PY
rg -n --ignore-case \
  'hf_[A-Za-z0-9]{20,}|api[_-]?key[[:space:]]*[=:]|bearer[[:space:]]+[A-Za-z0-9._-]{16,}' \
  /Users/sna/nemo-rl_release_perf_investigator/public/qwen3-30ba3b-hybridep-oci-20260727/index.html
git -C /Users/sna/nemo-rl_release_perf_investigator diff --check
```

Expected: HTML parser passes, credential scan returns no match, and diff check
passes.

- [ ] **Step 3: Commit and push the report**

```bash
git -C /Users/sna/nemo-rl_release_perf_investigator add \
  public/qwen3-30ba3b-hybridep-oci-20260727/index.html
git -C /Users/sna/nemo-rl_release_perf_investigator commit -s \
  -m "report: add Super config-reuse rerun"
git -C /Users/sna/nemo-rl_release_perf_investigator push
```
