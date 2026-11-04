# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Run the real pinned output-processing function with and without timing."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from nemo_rl.models.generation.vllm.frontend_timings import FrontendStageTimings


def native_processor():
    source = Path("/tmp/nano35-vllm029-output-processor.py")
    if not source.exists():
        pytest.skip("Provide the installed vLLM 0.29 output_processor source")
    tree = ast.parse(source.read_text())
    node = next(
        method
        for cls in tree.body
        if isinstance(cls, ast.ClassDef) and cls.name == "OutputProcessor"
        for method in cls.body
        if isinstance(method, ast.FunctionDef) and method.name == "process_outputs"
    )
    namespace = {
        "OutputProcessorOutput": SimpleNamespace,
        "FinishReason": SimpleNamespace(STOP="stop"),
    }
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[])),
            str(source),
            "exec",
        ),
        namespace,
    )

    class Processor:
        process_outputs = namespace["process_outputs"]

        def __init__(self):
            self.request_states = {}
            self.events = []
            self.tracing_enabled = False

        def _update_stats_from_output(self, *args):
            self.events.append("stats")

        def _update_stats_from_finished(self, *args):
            self.events.append("finished_stats")

        def _finish_request(self, state):
            self.events.append("cleanup")
            self.request_states.pop(state.request_id)

    return Processor()


def add_request(processor, *, stop=False):
    tokens, logprobs, delivered = [], [], []

    def detokenize(new_ids, finished):
        tokens.extend(new_ids)
        return "stop" if stop else None

    def make_output(*args):
        processor.events.append("make_output")
        return ("result", tuple(tokens))

    state = SimpleNamespace(
        request_id="call",
        routed_experts_chunks=[],
        is_prefilling=False,
        detokenizer=SimpleNamespace(update=detokenize),
        logprobs_processor=SimpleNamespace(
            update_from_output=lambda out: logprobs.append(out.new_logprobs)
        ),
        sampling_mask_chunks=[],
        make_request_output=make_output,
        streaming_input=False,
        queue=SimpleNamespace(put=delivered.append),
    )
    processor.request_states["call"] = state
    return state, tokens, logprobs, delivered


def output():
    return SimpleNamespace(
        request_id="call",
        new_token_ids=[4, 5],
        pooling_output=None,
        finish_reason=None,
        stop_reason=None,
        kv_transfer_params=None,
        ec_transfer_params=None,
        routed_experts="route_chunk",
        spec_decode_metrics=None,
        new_sampling_mask=None,
        new_logprobs=[-0.2, -0.3],
        finished=False,
    )


@pytest.mark.parametrize("stop", [False, True])
def test_native_stage_instrumentation_preserves_outputs_and_cleanup(stop):
    baseline = native_processor()
    observed = native_processor()
    left = add_request(baseline, stop=stop)
    right = add_request(observed, stop=stop)
    timings = FrontendStageTimings(observed, sample_every=1)
    expected = baseline.process_outputs([output()], 1.0)
    actual = observed.process_outputs([output()], 1.0)
    assert vars(actual) == vars(expected)
    assert observed.events == baseline.events
    assert right[1:] == left[1:]
    assert right[0].routed_experts_chunks == left[0].routed_experts_chunks
    assert observed.request_states.keys() == baseline.request_states.keys()
    stats = timings.stats()
    assert stats["sampled_calls"] == 1
    assert stats["sampled_request_updates"] == 1
    assert stats["detokenize_stop_calls"] == 1
    assert stats["logprobs_calls"] == 1
    assert stats["router_collect_calls"] == 1
    assert stats["completion_delivery_calls"] == 1
    # The stage includes the finish-condition check even on unfinished requests.
    assert stats["finish_cleanup_calls"] == 1


def test_sampling_preserves_outer_publication_wrapper():
    processor = native_processor()
    add_request(processor)
    timings = FrontendStageTimings(processor, sample_every=2)
    inner = processor.process_outputs
    publications = []

    def published(*args, **kwargs):
        publications.append("begin")
        result = inner(*args, **kwargs)
        publications.append("end")
        return result

    processor.process_outputs = published
    for _ in range(3):
        processor.process_outputs([output()])
    assert publications == ["begin", "end"] * 3
    assert timings.stats()["process_calls"] == 3
    assert timings.stats()["sampled_calls"] == 2
    assert timings.stats()["logprobs_calls"] == 2


def test_mismatched_upstream_source_fails_before_patching():
    class Changed:
        def process_outputs(self, outputs):
            return outputs

    processor = Changed()
    original = processor.process_outputs
    with pytest.raises(RuntimeError, match="Unverified vLLM"):
        FrontendStageTimings(processor, sample_every=64)
    assert processor.process_outputs == original


def test_wire_decoder_timing_preserves_payload_and_errors():
    processor = native_processor()
    timings = FrontendStageTimings(processor, sample_every=64)

    def decode(frames):
        if frames is None:
            raise ValueError("invalid message")
        return frames

    decoder = SimpleNamespace(decode=decode)
    timings.observe_decoder(decoder)
    frames = [b"payload"]
    assert decoder.decode(frames) is frames
    with pytest.raises(ValueError, match="invalid message"):
        decoder.decode(None)
    assert timings.stats()["wire_decode_calls"] == 2
