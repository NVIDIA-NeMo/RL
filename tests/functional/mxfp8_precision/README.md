# Nano MXFP8 precision and resume diagnostics

These are functional probes of real Megatron/Transformer Engine workers. CPU
checker tests are separate and do not constitute a successful Nano GPU run.
Production model files and checkpoints must be mounted read-only; every output,
FP32 master bank, gradient reference and checkpoint copy belongs to the audit.
The functional probes intentionally synchronize CUDA and copy diagnostics to CPU.
They are not throughput benchmarks and their checkpoints are not training inputs.

## Fixed teacher-forced FP8 storage comparison

Run `run_precision_probe.py` from this directory on an eight-GPU Ray cluster,
using the production Nano config and the two existing TE recipe files. Supply
`--worker-python` with the actual Megatron backend interpreter. For example:

```bash
PYTHONPATH=$PWD/tests/functional/mxfp8_precision:$PYTHONPATH \
uv run --no-sync tests/functional/mxfp8_precision/run_precision_probe.py \
  --config /opt/pipeline-config/rlvr_sc_mxfp8_lmhead_fp32_r3_fp8_param.yaml \
  --model /read-only/path/to/nano-checkpoint/hf \
  --recipe-bf16-param /opt/pipeline-config/te_mxfp.yaml \
  --recipe-fp8-param /opt/pipeline-config/te_mxfp_fp8_param.yaml \
  --worker-python /opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python \
  --output /audit/results
```

The output directory must not exist. The default topology is trainer TP4/EP8/CP1
on two four-GPU nodes; this is reduced from production TP4/EP16/CP4. All model
layers, routed experts, FP32 LM head and configured MTP layers remain present. The
probe uses an NLL objective on fixed, tokenized text instead of the production
reward objective. Generation and sampled router replay are disabled, so the
router operates on identical teacher-forced inputs in each storage mode.

The first worker lifetime reconstructs full FP32 parameters by semantic name
from optimizer shards and writes a canonical bank. Expert values receive a
deterministic sub-BF16-ULP perturbation once in that bank, so the test exercises
FP32 rounding on both sides of BF16 midpoints rather than only values already
representable in BF16. The second loads that bank,
sets the corresponding owned shards despite possible bucket-layout changes,
and reconstructs again to require exact equality. Both publish those masters
through the actual parameter-staging/all-gather implementation. A failed
FP32 or staging equality check stops the run rather than comparing different
models under the same name.

For each storage setting, the probe captures every vocabulary-sharded logit
and every `main_grad` tensor from a combined fixed batch, a repeat of that batch,
and two streaming chunks with an intervening production logprob/preparation
detour. It compares full gradients, not selected elements or gradient norms.
The controlled version fixes `max_sequences_per_bin=1` so combined and split
passes use the same microbatch geometry and activation-quantization grouping.
An additional `chunked` case splits the batch without a logprob detour. Compare
`chunked` versus `combined` to assess chunking and auxiliary-loss scaling, and
`streamed` versus `chunked` to isolate the inference detour. Training logits are
aligned by explicit prompt IDs after packing, including the streamed comparison.
Named model buffers and Megatron's RNG tracker are restored from a common
pre-forward state for each case. `initial-state.json` records those controls.
`--fresh-process-control` repeats the FP8/non-native configuration in a fresh
worker lifetime, separating process/runtime variability from a refit-flag effect.
The report records chunk/microbatch counts, valid-token/sequence sums, and the
MTP loss weight. Captured gradients are still before finish-time normalization;
if only the chunking control fails, investigate loss/auxiliary scaling rather
than attributing it to the logprob buffer detour.

### Gradient accumulation order control

Run with `--gradient-order-focus` on the eight-GPU trainer to exclude refit,
rollout and fresh-process comparisons. The production packer previously yielded
`[3,2,1,0]` for the combined batch and `[1,0,3,2]` for two chunks. Aligning saved
logits after forward did not align backward accumulation, and the recipe uses
BF16 gradient buffers (`grad_reduce_in_fp32=false`). The controlled path now
sorts single-sequence bins by prompt ID and preserves their padding lengths,
giving `[0,1,2,3]` in every combined/split comparison. Production packing code
is unchanged.

`combined_packer_order` / `chunked_packer_order` deliberately preserve the old
orders as an order-sensitivity control. Their differences are diagnostic and
do not count as failure of the matched-order tests. `gradient-prefixes.pt`
records selected accumulated gradients before each microbatch and after the
last, and `probe.json` records the actual gradient-buffer dtypes. Add
`--fp32-gradient-buffers` in a separate run to change gradient accumulation to
FP32 while retaining the same model compute precision and fixed masters.
Before/after gradient hashes detect any modification by that detour. Both cases
run `begin_train_step` and `train_microbatch`, then `abort_train_step`;
`optimizer.step()` is forbidden and FP32 masters must remain unchanged.

The separate stale-storage test doubles only private routed-expert FP32 master
shards. Its negative control observes that the old storage is unchanged before
publication. After publication, storage and forward logprobs must change.
Stored MXFP8 codes/scales are compared with fresh quantization of reconstructed
FP32 masters cast to BF16, including columnwise storage when materialized.
Original masters are restored in a `finally` block.

The defaults `atol=rtol=0.001` are explicit diagnostic thresholds, not established
Nano acceptance criteria. Every tensor's maximum error, relative L2 error and
cosine are recorded. Review cross-mode errors together with repeat and streaming
controls; a threshold failure alone does not prove a particular buffer defect.
Full reference tensors require substantial private scratch space.

## Initialization, shared buffers, reduction and ordinary-refit audit

`run_boundary_probe.py` uses four fresh worker lifetimes: BF16 storage, MXFP8
storage, and a genuine optimizer-checkpoint resume for each. The default trainer
topology is TP4/EP16/CP4 on sixteen GPUs, with four additional GPUs for vLLM.
Global data parallelism, sequence length and batch size remain smaller than the
production run. All comparisons use fixed synthetic teacher-forced text and an
NLL objective, rather than sampled RL rewards. Native MXFP8 refit stays disabled.

The probe records naturally loaded FP32 masters before installing a common bank.
It performs one private seed update per storage mode to initialize optimizer
state, then installs the same semantic FP32 bank in both modes. Audited forwards
stop inside the real `finish_train_step` immediately before `optimizer.step`,
capturing raw gradients and reconstructed owned gradients after normalization,
DP/CP/TP finalization and reduction. No audited optimizer update occurs. Resume
snapshots verify loaded masters before replacing or republishing them.

Shared-buffer observers check named spans and gathered BF16 contents before
`_post_param_sync`, fresh row/column quantization afterward, buffer clearing,
and zero gradient buffers at each parameter's first contribution. A separate
stale-storage check changes only private expert masters, checks both cached
orientations, and restores the original masters. First staging/refresh failures
retain tensors and event records. As with other observers in this directory,
hashing synchronizes GPU work: a passing observed run cannot exclude every
timing race in an uninstrumented throughput run.

A separate reduction probe supplies identical semantic local gradients in the
two actual buffer layouts. The real communication path is compared with an FP32
reference collective after matching production's original-dtype scaling. This
separates reduction roundoff from differences already present in backward.

`boundary_sitecustomize.py` is copied to the isolated launcher's hooks directory;
it is never installed in production. It captures refit wires, logical vLLM
expert weights before padding/permutation, execution tensors and full sampler
logits with input-token/position fingerprints. MXFP8 encodings are non-unique:
different FP8 bytes and E8M0 scales can represent identical values. Therefore
the audit fingerprints decoded logical weights separately from encoded pairs.
Missing sampler ranks or missing forward context cannot establish equivalence.

`result.json` reports completed capture and numerical diagnostics, rather than
declaring every boundary equivalent. Review natural-initialization differences,
pre/post-reduction errors, synthetic-input controls, effective refit weights and
same-context logits separately. The private launcher pins the experiment source
commit and records hashes of the source archive and probe files.

## Parameter storage and actual refit matrix

Add `--refit-matrix` to the command above and provide a third four-GPU node for
the vLLM sampler. `--nodes=2` still describes the eight-GPU trainer. Three fresh
trainer/sampler lifetimes use the same canonical FP32 master bank:

| `fp8_param` | `native_mxfp8_refit` | Test |
| --- | --- | --- |
| false | false | BF16 parameter storage, ordinary refit |
| true | false | MXFP8 parameter storage, ordinary refit |
| true | true | MXFP8 parameter storage, native refit |

Native refit requires MXFP8 training storage. The unsupported `(false, true)`
combination is excluded from this matrix.
Each supported case runs the teacher-forced and buffer probes, followed by the
actual production weight synchronizer, NCCL reshard transfer and two greedy
rollout batches. Every trainer rank must report the expected native-export
selection and unchanged FP32 masters/expert storage across refit. The sampler
uses eager execution and disabled startup autotuning, as in the resume audit.

`refit-matrix.json` compares within-case repeatability, BF16 versus MXFP8 storage
using ordinary refit, and ordinary versus native refit using MXFP8 storage.
Asynchronous outputs are reordered by original prompt index, with duplicate or
missing rows treated as errors. Generated logprobs are compared only when tokens
and lengths match, since divergent trajectories have different conditioning.
Cross-mode rollout equality is a diagnostic expectation rather than a guarantee
of the precision formats. Training logits/gradients remain a separate comparison
before generation or any optimizer update. These greedy rollout checks complement
the sampled clean/clean/poison resume diagnostics below.

The vLLM async API receives exactly one input sample per call. The probe submits
these calls sequentially and collects each call's local output index zero into
the original prompt order before padding/combining the results. This avoids
passing a multi-row batch to the single-sample API and keeps repeat scheduling
controlled. The regression suite checks four distinct prompts plus missing,
duplicate, wrongly indexed and multi-row responses.

## Resume rollout localization

`rollout_diagnostics.py` is an optional observer installed by the isolated resume
audit loader, before Ray captures trainer methods. It records:

1. All trainer parameters, physical FP8 caches/scales, BF16 parameters and buffers.
2. Bulk transfer tensors on the sender and receiver, with refit ordinal, mesh,
   dtype, shape and exact hashes.
3. Sampler parameters/buffers after refit, global RNG states and engine config.
4. Actual tokenized request inputs and effective per-request sampling parameters.
5. Raw `compute_logits` outputs for the first 256 calls and generated tokens/logprobs.

Run independent `clean_a`, `clean_b`, and `poison` lifetimes from independent
copies of the **same audit checkpoint**. The first two are the clean/clean
repeatability control. Keep the initial cohort's weights fixed: admit only the
two initial prompt groups before its first update. Both native-refit modes can
be run separately; never compare different refit modes as a poisoning control.

`analyze_rollout_boundaries.py /audit` reports the first differing stage and
nested value, with coverage counts. Missing captures fail closed. Refits after
the first are retained as raw diagnostics but excluded from the initial-cohort
comparison. RNG state differences are recorded separately from model hashes;
unused parameters and physical scale padding can differ harmlessly, so inspect
the first differing names instead of assuming every hash difference is causal.
A clean/clean difference blocks attribution to poisoning. A difference only in
the clean/poison pair is a candidate boundary to investigate, not proof of cause.
The captured `.pt` logits allow inspection of the first divergent decode step.
New logits also include model-forward token IDs and positions when the runner
exposes named inputs. Missing context is marked explicitly; raw batch ordinal
alone does not establish that two logits have identical conditioning.
The LM-head input fingerprint separates a different final hidden state from
differences introduced by the logits projection itself.
The private launcher supports `NANO_AUDIT_SERIAL_GREEDY=1` for a separate control:
one active prompt/sequence and greedy request sampling, while training retains
temperature 1. This control localizes scheduling/sampling effects; it does not
replace the concurrent sampled audit or change production defaults.
The serial control runs the analyzer with `--require-forward-context`, so
missing token/position capture cannot produce a successful equivalence result.
`--initial-sequential-requests=8` limits serial logit comparison to the first
cohort, using its recorded generation lengths. Extra requests admitted before
the next refit stay in raw diagnostics instead of changing the compared window.
The one-active-sequence control must supply one logit row per generated token;
missing coverage is an incomplete result, not equivalence.

Version 3 snapshots compare the logical unswizzled MXFP8 scale rectangles:
rowwise `[M, ceil(K/32)]` and columnwise `[ceil(M/32), K]`. Full allocation hashes
remain in `raw_storage` for inspection but do not decide semantic equivalence.
Used columnwise caches remain part of the comparison, and unknown/swizzled scale
layouts fail closed. The older run's whole-allocation hashes cannot establish
that its mismatch was only padding; new snapshots are required.
Version 2 assumed the wrong column-scale axes and must not be used to conclude
that a column cache is stale. The corrected geometry was checked against
`MXFP8Quantizer.get_scale_shape` extracted from the actual test image.

`run_column_cache_probe.py --output /audit/result.json` is a small GPU check in
the Megatron backend interpreter. It exercises real TE in-place refresh versus
fresh quantization, injects different values into scale padding to reproduce the
old false failure, and verifies that changing a used scale still fails. The Nano
stale-storage report records column data and scale comparisons separately.

The analyzer now reports every stage's comparison as well as the first
difference. New request/output and sampling records carry a token-input key;
within the initial cohort they are compared as multisets rather than completion
order, preserving duplicates and their counts. The raw event order remains on
disk. First-logit snapshots also report finite checks, top-two token IDs, and
their margin to help assess whether a small numerical error can change decoding.

Worker-local batches now use the real `shard_by_batch_size` packing path for
training and logprobs. A driver preflight exercises full and split batches and
checks that no row is lost or duplicated before model workers are allocated.

Run checker regressions without GPU dependencies:

```bash
PYTHONPATH=tests/functional/mxfp8_precision uv run --group test python -m pytest \
  tests/functional/mxfp8_precision/test_probe_diagnostics.py
```
