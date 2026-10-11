# Prefix checkpoint performance integration

This branch is a reproducible baseline for training and checkpoint tests. The
upstream PRs remain independent; the experimental source copies in pipeline have
not been changed or removed.

## Sources

| Component | Source |
| --- | --- |
| Prefix recovery and batched prefix PUT / coalesced GET | RL #4508, `7eb9a0184637393dda852ddbafed663f58842fa9` |
| Captured non-streaming OpenAI logprob-object bypass | RL #4634, `1cfe1e3389dda2fef9c7858da623fcc157331494` |
| Owned NumPy staging tensors and byte buffers | RL #4654, `218dc6f8599821f8021aa3a92900bc52e75f964d`, `d75f626859dcf8923fa5e62d7bc0a6e0b4aaae21`, `e77ffea08f42e465e0fbbe3399a2aa2f0803ec2f` |
| Bounded background prefix cleanup | RL #4655, `6f488cb1bd7d12dca7570c2485c1751ed9fe2de8` |
| Bulk Gym column encoding and single staging digest | Gym #4213, net changes through `f0e84c288c42e5161dabba865921790a95b5eac7` |

The Gym branch starts at the checkpoint stack's pin
`120d516f09cfd0377dbed6828247ab1606e9e686`, rather than Gym main. Its integration
branch is `amahishi/ckpt-perf-integration-gym` and its pinned commit is
`41f82e2f78ed0647cf360016da29c9bca6fb1830`.

The Gym branch also preserves the existing replay-completion fix from #3882,
the local late-model-reply parking fix, and the stateless/replay declarations
used by the current checkpoint dataset. These are separate commits. The resource
declarations cover abstention, calendar, equivalence_rule, ether0,
format_verification, instruction_following, inverse_if, jailbreak_detection,
mcqa, multichallenge, nvarc, reasoning_gym, structured_outputs and terminus_judge.
Workplace Assistant retains its existing exported-session implementation.

## Integration adjustments

- #4508 moves encoding into `TQTokenSink._encode_record`. The #4654 conversion
  is applied there, so terminal rows and prefix batches share the optimization.
- The checkpoint Gym stack completes a call through `build_prefix_record`.
  Both that builder and `build_generation_chunk_record` use the validated
  `StagedCallRecord.from_components` path from #4213. Local construction hashes
  once; fetched/deserialized records still verify their supplied digest.
- The serving regression tests exercise the logprob bypass with prefix cuts
  both enabled and disabled, using actual capture-state objects.

## Select this checkout

From pipeline:

```bash
export NEMO_RL_ROOT="$PWD/nemo_rl_integration"
export NEMO_GYM_ROOT="$NEMO_RL_ROOT/3rdparty/Gym-workspace/Gym"
```

The existing training launcher mounts Gym from the selected RL checkout.
Continue using the existing site overlay and container. Use a fresh experiment
name for the first save/restore test. No job was submitted while creating this
branch, and existing launcher/config defaults were not changed.

Background cleanup remains disabled by default. To enable it in a test config:

```yaml
rollout_recovery:
  target_level: prefix
  generation_prefix_cleanup:
    enabled: true
    batch_size: 32
    max_pending: 128
    wait_seconds: 0.01
```

It queues only obsolete chunk keys after terminal-row acknowledgement; the
snapshot fence pauses new delete batches and drains any active delete batch.

## Validation

Local diagnostic environment: Python 3.12.3, owned NumPy 2.4.6 conversion,
preinstalled Torch/Ray/TQ dependencies. Production Python 3.13/container/GPU
validation is still required.

| Suite | Result |
| --- | --- |
| Gym staging, agent checkpoint races, replay completion, resources | 183 passed |
| vLLM hosting, logprob bypass, serving wiring, prefix read coalescer | 95 passed, 3 optional tests skipped |
| Background cleanup and tensor wire values/buffer lifetime | 28 passed |
| SingleController setup and checkpoint coordination | 217 passed |

Total: **523 passed, 3 skipped**. Focused Ruff lint/format and whitespace checks
passed. A local CPU-only Simple TQ test passed Gym golden vectors, five ragged
prefix rows and five terminal rows, validated fetched digests, and confirmed
prefix-only deletion leaves canonical rows readable. This is transport validation,
not a GPU generation or full training save/restore result.

## Follow-ups

Incremental prefix caches, restore chunk-encoding reuse, terminal PUT batching,
the TQ physical-core cache workaround and extended checkpoint/TQ telemetry are
not in this baseline. Extract each as a separate follow-up commit with a matched
A/B test. Router replay and multimodal prefix extensions remain separate.

The current pipeline `benchmarks/rollout_checkpoint` harness imports experimental
cleanup/telemetry interfaces from `nemo_rl_refresh`. Its compatibility adapter
must be updated for this branch before using the existing large-scale matrices;
setting a terminal-batching option alone cannot add the missing implementation.

For updates, refresh a replacement integration branch from a known #4508 head
and reapply the component commits. Preserve the tested branch/commit rather than
rebasing a branch used by an active run. Record RL SHA, Gym SHA, container and
resolved config together for each run.
