# Qwen3-30B-A3B Thinking SWE trajectory-collection rollout-only benchmark

This experiment compares a matched no-speculation baseline with DFlash K5/K7
and DSpark K5/K7 on SWE-Bench Verified. It inherits the native Thinking SWE
workload through PR #3733's trajectory-collection path and five checked-in YAML
overlays. It is not the PR #3243 generation-only eval benchmark. It changes no
full-workload semantics outside the speculative configuration and its derived
CUDA Graph capture sizes.

No GPU result is recorded yet. PR #3733 explicitly did not validate the 30B
runtime/performance path; a successful benchmark canary is experiment evidence,
not retroactive proof attached to either PR.

## Contract

- Source history contains exact PR #3733 head
  `b580dd8927b88c996470d315e74d57bf0cb4090e`.
- The authoritative SWE base, PR overlay, and PR launcher are unchanged from
  that head.
- Target, DFlash, DSpark, data, source commit, and container bytes are verified
  before scheduling.
- Output and state are bound beneath the approved user Lustre `experiments`
  prefix; alternate state directories are rejected.
- Baseline is the same rollout-only overlay with `speculative_config=null`.
- Scheduler topology is two nodes, four GPUs per node, segment size one; the
  inherited generation resources use one node at TP2.
- W&B routes to `nvidia/sna-specdec`.
- Every arm must pass the identical `sbatch --test-only` contract before its
  exclusive pre-submission reservation can be created.
- Full submission remains locked until baseline and DFlash K5 canaries complete
  successfully.

## OCI-HSG sequence

Run from a clean recursive checkout under `/home/sna` at the signed experiment
commit. Use an output root on the user's Lustre allocation. Replace the shell
variables with the FairShare-selected account and compute-verified immutable
container identity.

Run `preflight` on an allocated compute node: it streams the target, draft, and
container bytes to SHA256 and must not perform those heavy reads on a login
node. The manifest intentionally fails closed while any expected digest is a
placeholder.

```bash
experiment=experiments/qwen3_30ba3b_swe_rollout_pr3733
source_commit=$(git rev-parse HEAD)
container=/lustre/path/to/immutable-nemo-rl.sqsh
container_sha256=<64-lowercase-hex>
output_root=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/experiments/q30-swe-pr3733
state_dir=${output_root}/state
account=<fairshare-selected-account>

python3 ${experiment}/benchmark.py preflight \
  --repo-root "$PWD" \
  --source-commit "${source_commit}" \
  --container "${container}" \
  --container-sha256 "${container_sha256}" \
  --record "${state_dir}/preflight.json"

python3 ${experiment}/benchmark.py materialize-canary \
  --destination "${output_root}/inputs/swebench_verified_first1.jsonl" \
  --record "${state_dir}/canary-input.json"

python3 ${experiment}/benchmark.py plan \
  --profile canary \
  --source-commit "${source_commit}" \
  --container "${container}" \
  --container-sha256 "${container_sha256}" \
  --output-root "${output_root}" \
  --preflight-record "${state_dir}/preflight.json" \
  --canary-record "${state_dir}/canary-input.json" \
  > "${output_root}/canary-plan.json"

for arm in baseline dflash_k5; do
  python3 ${experiment}/submit.py test-only \
    --plan "${output_root}/canary-plan.json" --arm "${arm}" \
    --repo-root "$PWD" --state-dir "${state_dir}" \
    --preflight-record "${state_dir}/preflight.json" \
    --account "${account}" --partition batch --time 04:00:00
done
```

The canary bounds the dataset to its canonical first prompt and preserves the
native trajectory-collection validation sampling contract of one generation.

Inspect the two test-only records before changing the explicit action to
`submit`. Collect the returned job IDs and monitor them with one filtered query
per minute over five minutes:

```bash
python3 ${experiment}/submit.py monitor \
  --job-id <baseline-job-id> --job-id <dflash-k5-job-id> \
  --state-dir "${state_dir}" \
  --campaign-id <campaign-id-from-plan> --profile canary
```

After both jobs terminate successfully, their job wrappers create matched
completion records. `benchmark.py unlock-full --state-dir "${state_dir}"
--campaign-id <campaign-id-from-plan>` must return `full-unlocked` before a full
plan can pass scheduler submission gates.

## Local verification

The repository lock targets Linux, so the authoritative locked test command is
run inside the selected OCI container. A macOS-safe dependency-free fallback is:

```bash
pytest --noconftest tests/unit/test_qwen3_30ba3b_swe_rollout_benchmark.py -q
ruff check experiments/qwen3_30ba3b_swe_rollout_pr3733 \
  tests/unit/test_qwen3_30ba3b_swe_rollout_benchmark.py
```
