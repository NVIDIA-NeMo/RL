#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Checks that parse_results.py reports the error a human would point at.
#
# The signature is what lands in the PR comment, so picking the wrong line sends
# the author after the wrong bug. Signatures are found by scanning a failed test's
# output backwards, which once made interpreter-shutdown noise outrank the real
# exception: three genuine `AttributeError: 'NoneType' ... 'tolist'` failures were
# all reported as `PythonFinalizationError: preexec_fn not supported`.
#
#   bash .agents/nemo-rl-testing-agent/tests/test_parse_results.sh

set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARSER="${here}/../scripts/parse_results.py"
work="$(mktemp -d)"
trap 'rm -rf "${work}"' EXIT

failures=0

check_signature() { # description, test_name, expected_substring
  local desc="$1" name="$2" want="$3" got
  got="$(python3 - "$work/out.json" "$name" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
for test in payload["tests"]:
    if test["name"] == sys.argv[2]:
        print(test.get("error_signature") or "")
        break
else:
    print("<test not found>")
PY
)"
  if [[ "${got}" == *"${want}"* ]]; then
    echo "ok: ${desc}"
  else
    echo "FAIL: ${desc}"
    echo "    wanted substring: ${want}"
    echo "    got:              ${got}"
    failures=$((failures + 1))
  fi
}

# A realistic failure: the true exception, then Ray/interpreter teardown noise.
cat > "${work}/suite.log" <<'LOG'
mcore_sha=15245041e771c84820438b99cd8bc82ebf7ec594
===NRLTA_SUITE_BEGIN suite=L1_Functional_Tests_Megatron_4 tests=3===
===NRLTA_TEST_BEGIN name=sync_generation suite=L1_Functional_Tests_Megatron_4 cmd=bash x.sh===
cuda graph warmup - [688]: 16 P + 672 D:  58%|#####     | 398/690 [00:31<00:12, 23.17it/s]
Traceback (most recent call last):
  File "/opt/nemo-rl/nemo_rl/models/generation/megatron/megatron_worker.py", line 560, in _parse_result_to_batched_data_dict
    tokens = result[i].prompt_tokens.tolist() + result[i].generated_tokens
AttributeError: 'NoneType' object has no attribute 'tolist'
Exception ignored in: <function Policy.__del__ at 0xfffdd89cb2e0>
Traceback (most recent call last):
  File "/opt/nemo-rl/nemo_rl/models/policy/lm_policy.py", line 100, in __del__
PythonFinalizationError: preexec_fn not supported at interpreter shutdown
===NRLTA_TEST_END name=sync_generation rc=1 secs=177===
===NRLTA_TEST_BEGIN name=metric_regression suite=L1_Functional_Tests_Megatron_4 cmd=bash y.sh===
AssertionError: max(data["train/token_mult_prob_error"]) < 1.05
PythonFinalizationError: preexec_fn not supported at interpreter shutdown
===NRLTA_TEST_END name=metric_regression rc=1 secs=88===
===NRLTA_TEST_BEGIN name=metric_table suite=L1_Functional_Tests_Megatron_4 cmd=bash m.sh===
🔹 train/token_mult_prob_error - 3 steps
│ PASS      │ max(data["train/truncation_rate"]) < 1.05                           │ 0.5                      │ ok                                   │
│ FAIL      │ median(data["train/token_mult_prob_error"]) < 1.1                   │ 3.370945930480957        │ median(data["train/token_mult_prob_error"]) < 1.1 (condition evaluated to False)  │
===NRLTA_TEST_END name=metric_table rc=1 secs=322===
===NRLTA_TEST_BEGIN name=healthy suite=L1_Functional_Tests_Megatron_4 cmd=bash z.sh===
all good
===NRLTA_TEST_END name=healthy rc=0 secs=42===
===NRLTA_TEST_BEGIN name=async_generation suite=L1_Functional_Tests_Megatron_4 cmd=bash a.sh===
(MegatronPolicyWorker[rank=0] pid=405303) [Rank 0] Completed 1 requests
Error generating response for sample 7: ray::MegatronPolicyWorker.generate_async() (pid=405303)
  File "/opt/nemo-rl/nemo_rl/models/generation/megatron/megatron_worker.py", line 655, in _generate_single_item
    output = self._parse_result_to_batched_data_dict(datum, result)
AttributeError: 'NoneType' object has no attribute 'tolist'
Error generating response for sample 5: ray::MegatronPolicyWorker.generate_async() (pid=405303)
AttributeError: 'NoneType' object has no attribute 'tolist'
Training step 1 complete
===NRLTA_TEST_END name=async_generation rc=0 secs=228===
===NRLTA_SUITE_END suite=L1_Functional_Tests_Megatron_4 overall_rc=1===
LOG

uv run --script "${PARSER}" "${work}/suite.log" --out "${work}/out.json" >/dev/null || {
  echo "FAIL: parser exited non-zero"
  exit 1
}

check_signature "reports the real exception, not interpreter-shutdown noise" \
  sync_generation "AttributeError: 'NoneType' object has no attribute 'tolist'"
check_signature "prefers an AssertionError over trailing noise" \
  metric_regression 'AssertionError: max(data["train/token_mult_prob_error"]) < 1.05'
check_signature "leaves a passing test without a signature" healthy ""
# A numerics regression raises nothing; it is a row in a rich table. Without this
# the failure these suites exist to catch arrives with an empty signature, and
# the known-issues registry cannot recognise it on the next PR.
check_signature "names a failed metric check that raised no exception" \
  metric_table 'metric check failed: median(data["train/token_mult_prob_error"]) < 1.1'

detail="$(python3 - "${work}/out.json" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
for test in payload["tests"]:
    if test["name"] == "metric_table":
        print(test.get("metric_failure", ""))
PY
)"
want_detail='median(data["train/token_mult_prob_error"]) < 1.1 (measured 3.370945930480957)'
if [[ "${detail}" == "${want_detail}" ]]; then
  echo "ok: keeps the measured value out of the signature but in the report"
else
  echo "FAIL: expected metric detail '${want_detail}', got '${detail}'"
  failures=$((failures + 1))
fi
check_signature "names the exception a green test swallowed" \
  async_generation "AttributeError: 'NoneType' object has no attribute 'tolist'"

statuses="$(python3 - "${work}/out.json" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
print(",".join(f"{t['name']}={t['status']}" for t in payload["tests"]))
PY
)"
want="sync_generation=fail,metric_regression=fail,metric_table=fail,healthy=pass,async_generation=pass (suspect)"
if [[ "${statuses}" == "${want}" ]]; then
  echo "ok: rc=0 with swallowed sample errors is flagged suspect, clean rc=0 is not"
else
  echo "FAIL: unexpected statuses"
  echo "    wanted: ${want}"
  echo "    got:    ${statuses}"
  failures=$((failures + 1))
fi

swallowed="$(python3 - "${work}/out.json" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1]))
for test in payload["tests"]:
    if test["name"] == "async_generation":
        print(test.get("swallowed_errors", 0))
PY
)"
if [[ "${swallowed}" == "2" ]]; then
  echo "ok: counts every swallowed sample error"
else
  echo "FAIL: expected 2 swallowed errors, got '${swallowed}'"
  failures=$((failures + 1))
fi

echo
if [[ "${failures}" -eq 0 ]]; then
  echo "parse_results: all checks passed"
else
  echo "parse_results: ${failures} check(s) failed"
fi
exit $(( failures > 0 ))
