"""Check score alignment, first-failure selection, and observer buffer ownership."""

import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

source = (
    Path(__file__).parents[4] / "tools/model_diagnostics/native_vllm_context_rescore.py"
)
spec = importlib.util.spec_from_file_location("native_context_rescore", source)
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


@pytest.mark.parametrize(
    "decode,fresh,start,threshold,maximum,first",
    [
        ([-0.1, -0.2], [-0.11, -0.18], 0, 5.0, 0.02, None),
        ([-0.00435], [-23.18], 0, 5.0, 23.17565, 0),
        ([0.0, 0.0, 0.0], [-9.0, -6.0, -20.0], 1, 5.0, 20.0, 1),
        ([0.0, 0.0], [-9.0, -0.1], 1, 5.0, 0.1, None),
        ([0.0], [-5.0], 0, 5.0, 5.0, None),
        ([], [], 0, 5.0, 0.0, None),
        ([math.nan], [-0.1], 0, 5.0, math.inf, 0),
        ([-0.1], [-math.inf], 0, 5.0, math.inf, 0),
    ],
)
def test_checked_interval_and_first_mismatch(
    decode, fresh, start, threshold, maximum, first
):
    actual_maximum, actual_first = diagnostic.compare_scores(
        decode, fresh, start=start, threshold=threshold
    )
    assert actual_maximum == pytest.approx(maximum)
    assert actual_first == first


@pytest.mark.parametrize(
    "decode,fresh,start", [([0.0], [], 0), ([0.0], [0.0], 2), ([0.0], [0.0], -1)]
)
def test_misaligned_scores_rejected(decode, fresh, start):
    with pytest.raises(ValueError):
        diagnostic.compare_scores(decode, fresh, start=start, threshold=5.0)


def _worker():
    batch = SimpleNamespace(
        req_ids=["native-main-a", "native-main-b"],
        num_draft_tokens=0,
        num_reqs=2,
        idx_mapping=torch.tensor([2, 0]),
        logits_indices=torch.tensor([1, 3]),
        positions=torch.tensor([9, 10, 19, 20]),
        input_ids=torch.tensor([43, 44, 32, 33]),
        seq_lens=torch.tensor([11, 21]),
    )
    all_ids = torch.zeros(3, 32, dtype=torch.int64)
    all_ids[2, 10], all_ids[0, 20] = 44, 33
    output = SimpleNamespace(
        sampler_output=SimpleNamespace(
            sampled_token_ids=torch.tensor([[45], [34]]),
            logprobs_tensors=SimpleNamespace(
                logprob_token_ids=torch.tensor([[45, 99], [34, 99]]),
                logprobs=torch.tensor([[-0.02, -0.1], [-0.05, -0.2]]),
            ),
        ),
        num_sampled_tokens=torch.tensor([1, 1]),
    )
    runner = SimpleNamespace(
        prepare_inputs=lambda: batch,
        sample_tokens=lambda: output,
        execute_model_state=SimpleNamespace(input_batch=batch),
        model_state=SimpleNamespace(_mamba_state_idx_gpu=torch.tensor([10, 11, 12])),
        req_states=SimpleNamespace(
            last_sampled_tokens=torch.tensor([[33], [0], [44]]),
            all_token_ids=SimpleNamespace(gpu=all_ids),
            num_computed_tokens=SimpleNamespace(gpu=torch.tensor([20, 0, 10])),
        ),
    )
    return SimpleNamespace(model_runner=runner), batch, output


def test_observer_owns_input_and_sampler_buffers(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    worker, batch, output = _worker()
    diagnostic.install_worker_trace(worker, capacity=2)
    assert worker.model_runner.prepare_inputs() is batch
    assert worker.model_runner.sample_tokens() is output
    # Simulate producer reuse after the asynchronous output was constructed.
    batch.input_ids.fill_(255)
    output.sampler_output.sampled_token_ids.fill_(256)
    output.sampler_output.logprobs_tensors.logprobs.fill_(-999)
    trace = diagnostic.read_worker_trace(worker, "native-main-a", 11)
    row = trace["records"][0]
    assert row["forward_position"] == 10
    assert row["input_token"] == row["last_sampled_token"] == 44
    assert row["all_token_ids_at_position"] == 44
    assert row["request_slot"] == 2
    assert row["sampled_token_ids"] == [45]
    assert row["logprobs"][0] == pytest.approx(-0.02)
    assert row["mamba_state_indices_after_forward"] == 12


def test_observer_keeps_bounded_history(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    worker, _, _ = _worker()
    diagnostic.install_worker_trace(worker, capacity=2)
    for _ in range(3):
        worker.model_runner.prepare_inputs()
        worker.model_runner.sample_tokens()
    trace = diagnostic.read_worker_trace(worker, "native-main-a", 11)
    assert trace["retained_steps"] == 2
    assert [row["step"] for row in trace["records"]] == [1, 2]


def test_other_ranks_do_not_record(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    worker, _, _ = _worker()
    diagnostic.install_worker_trace(worker, capacity=2)
    assert not hasattr(worker, "_native_context_trace")
    assert diagnostic.read_worker_trace(worker, "native-main-a", 11)["records"] == []


def test_wrong_runner_rejected(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    with pytest.raises(RuntimeError, match="V2"):
        diagnostic.install_worker_trace(
            SimpleNamespace(model_runner=SimpleNamespace()), capacity=2
        )
