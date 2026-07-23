"""Tests for compact pairwise loss tracing."""

from __future__ import annotations

import json

import torch

from nemo_rl.models.multi_lora.diag import append_loss_trace


def test_append_loss_trace_records_aligned_loss_and_hash(tmp_path, monkeypatch):
    monkeypatch.setenv("NOUSNET_DIAG_ENABLED", "1")
    monkeypatch.setenv("NOUSNET_DIAG_LOSS_TRACE", "1")
    monkeypatch.setenv("NOUSNET_RUN_DIR", str(tmp_path))
    ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    logp = torch.tensor([[-1.0, -3.0]])
    mask = torch.tensor([[1.0, 0.0]])

    append_loss_trace(
        step=2,
        rank=1,
        who="single_a",
        input_ids=ids,
        token_logprobs=logp,
        loss_mask=mask,
    )

    path = tmp_path / "diag_loss_trace" / "rank_01.jsonl"
    row = json.loads(path.read_text().strip())
    assert row["step"] == 2
    assert row["rank"] == 1
    assert row["who"] == "single_a"
    assert row["loss"] == 1.0
    assert row["num_tokens"] == 1
    assert len(row["input_sha256"]) == 64
    assert row["input_shape"] == [1, 3]


def test_append_loss_trace_is_opt_in(tmp_path, monkeypatch):
    monkeypatch.setenv("NOUSNET_DIAG_ENABLED", "1")
    monkeypatch.delenv("NOUSNET_DIAG_LOSS_TRACE", raising=False)
    monkeypatch.setenv("NOUSNET_RUN_DIR", str(tmp_path))
    append_loss_trace(
        step=1,
        rank=0,
        who="single_a",
        input_ids=torch.ones(1, 2, dtype=torch.long),
        token_logprobs=torch.zeros(1, 1),
        loss_mask=torch.ones(1, 1),
    )
    assert not (tmp_path / "diag_loss_trace").exists()
