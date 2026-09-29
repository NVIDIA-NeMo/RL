# CPU checkpoint workload

This harness runs the real Single Controller, ready-first sampler, Gym agents,
token capture, row assembly, and SimpleStorage checkpoint path with no GPUs.
It replaces policy, generation, and refit computation through factories in
`checkpoint.yaml`. Recovery stays in the production code.

`Policy` updates a small tensor from the ordered training batch. `Generation`
produces stable tokens and weight-dependent logprobs. `CopyRefit` copies the
tensor and checks the receiver's acknowledgement. Each generation call uses
the weights it received when that call started.

To replace a component, point its `factory` at `module:Class`. The class accepts
its validated nested `Config` model plus any dependencies: refit receives
`policy` and `generation`. The protocols in `components.py`, `servers.py`, and
`runtime.py` describe the methods the adapters consume. A replacement need not
inherit the default mock. Mixing real and fake components may need another
adapter; this harness does not promise GPU backend compatibility.

## Checkpoint scenario

| Prompt | Siblings | Turns per sibling | Seconds per turn |
| --- | ---: | ---: | --- |
| P1 | 3 | 6 | 11, 11, 11 |
| P2 | 3 | 2 | 8, 8, 8 |
| P3 | 3 | 7 | 2, 7, 7 |
| P4 | 3 | 2 | 3, 3, 3 |

Each training step takes two seconds and consumes one complete group. The
expected order is P4, P2, P3, P1. The controller checkpoints after steps 2 and 4
and shuts down normally. The test removes step 4, starts a fresh Ray runtime
and capture ledger, and lets normal discovery select step 2.

At the step-2 cut, P3's first sibling is complete. Its other siblings each
have three saved model calls; P1's siblings each have two. Restore should reuse
the completed sibling and issue exactly 20 new calls. The test compares row
order, tokens, masks, rewards, and saved logprob prefixes. Gym's counter checks
that every tool call ran exactly once, including a pending tool call whose
model response was already saved.

Timing uses elapsed sleeps, with no event gates. Startup and checkpoint work
add overhead. The nominal workload is about 114 seconds across both runs;
the complete test takes longer. A machine that misses the expected cut must
fail the timing/count assertions, not retry until it passes.

## Running

Initialize the pinned Gym submodule and use a CPU environment containing the
repository's runtime dependencies plus Gym. Run:

```bash
uv run pytest tests/mock_stack -q
```

The harness keeps the production HTTP capture configuration but replaces
construction of the GPU handles. The real controller runs in the test process;
Gym, storage, and row assembly use Ray actors. Gym subprocesses use the same
Python environment as the test. The checkpoint test has a 300-second timeout.

Local macOS development used Python 3.13.14 and torch 2.10.0 because the
repository's torch 2.11.0 dependency resolution required a Linux-only NCCL
wheel. That validation does not cover the pinned Linux environment.
The complete 24-test suite passed locally in 257 seconds; the checkpoint test
itself took 179 seconds, excluding setup and teardown.

## Recorded timelines

Each `run()` writes `output/traces/<run-id>.jsonl` and returns its path as
`Run.trace_path`. These are standard OpenTelemetry span JSON records, exported
locally through the SDK with NeMo-RL's Lens `span_cm` helper. No collector or
GPU is required. Each run has its own provider; other tests' telemetry settings
are untouched. Keep pytest output with `--basetemp=/tmp/cpu-checkpoint-run`.

The trace records generation (prompt, sibling, turn, requested delay), policy
training, initial/step refits, policy restore, and controller checkpoint saves.
`test.checkpoint.prepare_commit` covers parking participants and exporting
state. `test.checkpoint.release` starts after snapshot publication; it is an
upper bound on publication time, not the exact instant of the atomic rename.
Generation spans measure mock computation, not the subsequent capture write
or final Gym tool call. Trace timing alone does not prove saved state; the
checkpoint assertions check that state separately.

The parent branch includes checkpoint metrics (#3925) but predates Single
Controller tracing (#4052) and its Lens upgrade. `tracing.py` observes the
existing test adapters and controller methods so this PR stays limited to
`tests/`. It does not claim distributed traces inside Ray or Gym workers.
The wrappers also observe replacement implementations through the same
methods; replacements do not need a new telemetry interface.

For a plot, read each JSON line, group generation spans by `test.prompt` and
`test.sibling`, and draw `start_time`–`end_time` intervals labeled `test.turn`.
Align each run to its earliest generation start to exclude startup. Draw
training, refit, and checkpoint spans on separate lanes. Compare counts and
ordering exactly; display clock differences rather than requiring equal
wall times. Preserve both original and restored trace files.
