# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest
import ray
import torch
from fastapi.responses import JSONResponse
from tensordict import TensorDict

from nemo_rl.data_plane.adapters.noop import NoOpDataPlaneClient
from nemo_rl.data_plane.schema import ROLLOUT_STAGING_FIELDS
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.experience.rollout_writer import (
    CursorConflictError,
    CursorFailedError,
    DuplicateRequestError,
    FinalizationManifest,
    RolloutContext,
    RolloutContextError,
    RolloutCursorStateMachine,
    RolloutCursorRegistry,
    TurnRecord,
    assemble_staged_batch,
    build_staging_delta,
    derive_request_nonce,
    encoded_response_payload_sizes,
    extract_generation_token_info,
    hash_token_ids,
    mint_rollout_context,
    mint_rollout_id,
    strip_direct_response_logprobs,
    validate_rollout_context,
)
from nemo_rl.experience.sync_rollout_actor import _attach_rollout_contexts
from nemo_rl.models.generation.vllm.vllm_worker_async import (
    VllmAsyncGenerationWorkerImpl,
)


def test_context_round_trip_tamper_and_expiry() -> None:
    secret = b"unit-test-secret"
    context = mint_rollout_context(
        sample_id="sample-1",
        group_id="group-1",
        weight_version=7,
        secret=secret,
        ttl_s=10,
        now=100,
    )
    decoded = RolloutContext.from_dict(context.to_dict())
    validate_rollout_context(decoded, secret=secret, now=105)

    with pytest.raises(RolloutContextError, match="signature"):
        validate_rollout_context(
            replace(decoded, sample_id="sample-2"), secret=secret, now=105
        )
    with pytest.raises(RolloutContextError, match="expired"):
        validate_rollout_context(decoded, secret=secret, now=111)


def test_rollout_context_attachment_preserves_existing_metadata() -> None:
    secret = b"context-secret"
    original = BatchedDataDict(
        {
            "extra_env_info": [
                {
                    "responses_create_params": {
                        "metadata": {
                            "extra_body": '{"seed":7}',
                            "request_tag": "keep",
                        }
                    }
                },
                {"responses_create_params": {}},
            ]
        }
    )
    attached = _attach_rollout_contexts(
        original,
        rollout_ids=["sample-0", "sample-1"],
        group_ids=["group", "group"],
        weight_version=3,
        secret=secret,
        ttl_s=60,
    )
    assert (
        original["extra_env_info"][0]["responses_create_params"]["metadata"][
            "extra_body"
        ]
        == '{"seed":7}'
    )
    for index, row in enumerate(attached["extra_env_info"]):
        metadata = row["responses_create_params"]["metadata"]
        extra_body = json.loads(metadata["extra_body"])
        context = RolloutContext.from_dict(extra_body["nemo_rl_rollout_context"])
        validate_rollout_context(context, secret=secret)
        assert context.sample_id == f"sample-{index}"
        # Migration alias pair: the bare rollout_id rides alongside the signed
        # context and must equal its sample_id.
        assert extra_body["nemo_rl_rollout_id"] == context.sample_id
    assert (
        attached["extra_env_info"][0]["responses_create_params"]["metadata"][
            "request_tag"
        ]
        == "keep"
    )


def test_mint_rollout_id_is_opaque_url_safe_and_unique() -> None:
    minted = {mint_rollout_id() for _ in range(64)}
    assert len(minted) == 64
    for rollout_id in minted:
        # 128 random bits as lowercase hex; valid under Gym's rollout-id
        # charset (^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$) so the same string
        # can ride a /ng-rollout/{rid}/v1 base URL and a TQ staging key.
        assert len(rollout_id) == 32
        assert all(c in "0123456789abcdef" for c in rollout_id)


def test_request_nonce_and_prefix_hash_are_stable_and_order_sensitive() -> None:
    assert derive_request_nonce("sample", [1, 2, 3]) == derive_request_nonce(
        "sample", [1, 2, 3]
    )
    assert derive_request_nonce("sample", [1, 2, 3]) != derive_request_nonce(
        "sample", [1, 3, 2]
    )
    assert hash_token_ids([1, 2]) != hash_token_ids([2, 1])


def test_staging_delta_slices_prompt_and_places_masks_and_logprobs() -> None:
    token_ids, token_mask, logprobs = build_staging_delta(
        prompt_token_ids=[1, 2, 3, 4],
        generated_token_ids=[5, 6],
        generated_logprobs=[-0.1, -0.2],
        prev_len=2,
    )
    assert token_ids == [3, 4, 5, 6]
    assert token_mask == [0.0, 0.0, 1.0, 1.0]
    assert logprobs == [0.0, 0.0, -0.1, -0.2]

    with pytest.raises(ValueError, match="lengths differ"):
        build_staging_delta(
            prompt_token_ids=[1],
            generated_token_ids=[2],
            generated_logprobs=[],
            prev_len=1,
        )


@pytest.mark.parametrize(
    "choice",
    [
        {
            "message": {
                "generation_token_ids": [3, 4],
                "generation_log_probs": [-0.1, -0.2],
            }
        },
        {
            "message": {},
            "logprobs": {
                "content": [
                    {"token": "token_id:3", "logprob": -0.1},
                    {"token": "token_id:4", "logprob": -0.2},
                ]
            },
        },
    ],
)
def test_extract_generation_token_info_supports_vllm_response_shapes(choice) -> None:
    token_ids, logprobs = extract_generation_token_info(choice)
    assert token_ids == [3, 4]
    assert logprobs == [-0.1, -0.2]


def test_direct_response_strips_logprobs_but_retains_token_ids() -> None:
    response = {
        "choices": [
            {
                "logprobs": {"content": [{"logprob": -0.1}]},
                "message": {
                    "prompt_token_ids": [1, 2],
                    "generation_token_ids": [3],
                    "generation_log_probs": [-0.1],
                },
            }
        ]
    }
    stripped = strip_direct_response_logprobs(response)
    assert stripped["choices"][0]["message"]["generation_token_ids"] == [3]
    assert "logprobs" not in stripped["choices"][0]
    assert "generation_log_probs" not in stripped["choices"][0]["message"]


def test_encoded_response_payload_sizes_separate_tokens_and_logprobs() -> None:
    payload = {
        "id": "response",
        "choices": [
            {
                "message": {
                    "content": "answer",
                    "prompt_token_ids": [1, 2],
                    "generation_token_ids": [3, 4],
                    "generation_log_probs": [-0.1, -0.2],
                },
                "logprobs": {"top_logprobs": [{"x": -0.1}]},
            }
        ],
    }

    sizes = encoded_response_payload_sizes(payload)

    assert sizes["encoded_response_bytes"] == len(
        json.dumps(
            payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode("utf-8")
    )
    assert (
        sizes["encoded_response_base_bytes"]
        < sizes["encoded_response_bytes_without_token_ids"]
        < sizes["encoded_response_bytes"]
    )
    assert (
        sizes["encoded_response_base_bytes"]
        < sizes["encoded_response_bytes_without_logprobs"]
        < sizes["encoded_response_bytes"]
    )


def test_encoded_response_payload_sizes_count_tokenize_tokens() -> None:
    payload = {
        "count": 3,
        "max_model_len": 4096,
        "tokens": [101, 102, 103],
        "token_strs": None,
    }

    sizes = encoded_response_payload_sizes(payload)

    assert (
        sizes["encoded_response_bytes_without_logprobs"]
        == sizes["encoded_response_bytes"]
    )
    assert (
        sizes["encoded_response_bytes_without_token_ids"]
        == sizes["encoded_response_base_bytes"]
    )
    assert sizes["encoded_response_base_bytes"] < sizes["encoded_response_bytes"]


def test_rollout_http_metrics_include_tokenize_request_and_response() -> None:
    worker = object.__new__(VllmAsyncGenerationWorkerImpl)
    worker._rollout_metrics_enabled = True
    worker._rollout_metrics_lock = threading.Lock()
    worker._rollout_dp_client = None
    worker.clear_rollout_transport_metrics()

    request_body = b'{"model":"test","messages":[]}'
    response_payload = {
        "count": 2,
        "max_model_len": 4096,
        "tokens": [101, 102],
        "token_strs": None,
    }
    response = JSONResponse(content=response_payload)
    worker._record_http_request("tokenize", len(request_body))
    worker._record_http_response(
        "tokenize", response_payload, response, time.perf_counter()
    )

    metrics = worker._rollout_transport_metrics
    assert metrics["http_request_count"] == 1
    assert metrics["http_response_count"] == 1
    assert metrics["encoded_request_bytes"] == len(request_body)
    assert metrics["encoded_response_bytes"] == len(response.body)
    assert metrics["tokenize_request_count"] == 1
    assert metrics["tokenize_response_count"] == 1
    assert metrics["tokenize_encoded_request_bytes"] == len(request_body)
    assert metrics["tokenize_encoded_response_bytes"] == len(response.body)
    assert (
        metrics["tokenize_encoded_response_base_bytes"]
        < metrics["tokenize_encoded_response_bytes"]
    )
    assert len(metrics["tokenize_request_ms"]) == 1
    assert metrics["last_response_completed_monotonic_s"] > 0


def test_cursor_reserve_commit_and_duplicate_completed_request() -> None:
    state = RolloutCursorStateMachine(lease_ttl_s=5, cursor_ttl_s=30)
    reservation = state.reserve_turn("sample", "nonce-0", now=1)
    same = state.reserve_turn("sample", "nonce-0", now=2)
    assert same == reservation

    state.commit_turn(
        "sample",
        reservation.lease,
        staging_key="sample/t0",
        new_len=3,
        new_hash=hash_token_ids([1, 2, 3]),
        group_id="group",
        weight_version=7,
        now=3,
    )
    manifest = state.get_finalization_manifest("sample")
    assert manifest.committed_length == 3
    assert manifest.turns[0].staging_key == "sample/t0"

    with pytest.raises(DuplicateRequestError):
        state.reserve_turn("sample", "nonce-0", now=4)
    with pytest.raises(CursorFailedError, match="duplicate_request"):
        state.reserve_turn("sample", "nonce-1", now=5)


def test_cursor_rejects_concurrent_request() -> None:
    state = RolloutCursorStateMachine(lease_ttl_s=5, cursor_ttl_s=30)
    state.reserve_turn("sample", "nonce-0", now=1)
    with pytest.raises(CursorConflictError, match="active turn"):
        state.reserve_turn("sample", "nonce-1", now=2)
    with pytest.raises(CursorFailedError, match="concurrent_request"):
        state.reserve_turn("sample", "nonce-0", now=3)


def test_cursor_expired_lease_reissues_only_for_same_nonce() -> None:
    state = RolloutCursorStateMachine(lease_ttl_s=5, cursor_ttl_s=30)
    first = state.reserve_turn("sample", "nonce-0", now=1)
    retry = state.reserve_turn("sample", "nonce-0", now=7)
    assert retry.turn == first.turn
    assert retry.lease != first.lease
    with pytest.raises(CursorConflictError):
        state.commit_turn(
            "sample",
            first.lease,
            staging_key="sample/t0",
            new_len=1,
            new_hash=hash_token_ids([1]),
            group_id="group",
            weight_version=7,
            now=8,
        )


def test_cursor_failure_and_ttl_cleanup() -> None:
    state = RolloutCursorStateMachine(lease_ttl_s=5, cursor_ttl_s=10)
    reservation = state.reserve_turn("sample", "nonce-0", now=1)
    state.fail_turn("sample", reservation.lease, reason="write_failed", now=2)
    assert state.get_finalization_manifest("sample").failure_reason == "write_failed"
    assert state.expire_stale(now=13) == ["sample"]


@pytest.mark.asyncio
async def test_worker_rejects_echoed_prefix_that_differs_from_committed_cursor() -> (
    None
):
    cursor = RolloutCursorRegistry.remote(lease_ttl_s=10, cursor_ttl_s=100)
    committed_ids = [1, 2, 3]
    first = ray.get(cursor.reserve_turn.remote("sample", "first-request"))
    ray.get(
        cursor.commit_turn.remote(
            "sample",
            first.lease,
            staging_key="sample/t0",
            new_len=len(committed_ids),
            new_hash=hash_token_ids(committed_ids),
            group_id="group",
            weight_version=4,
        )
    )

    secret = b"prefix-check-secret"
    context = mint_rollout_context(
        sample_id="sample",
        group_id="group",
        weight_version=4,
        secret=secret,
        ttl_s=60,
    )
    worker = object.__new__(VllmAsyncGenerationWorkerImpl)
    worker._rollout_writer_cfg = SimpleNamespace(enabled=True)
    worker._rollout_writer_secret = secret
    worker._rollout_cursor = cursor
    worker._rollout_requests = {}
    worker._rollout_prompt_tokens = {}
    request = SimpleNamespace(
        nemo_rl_rollout_context=context.to_dict(),
        required_prefix_token_ids=committed_ids,
        stream=False,
    )

    class _Tokenizer:
        @staticmethod
        def decode(token_ids):
            return " ".join(map(str, token_ids))

    await worker._prepare_rollout_request(
        request,
        prompt_token_ids=[1, 2, 9, 10],
        tokenizer=_Tokenizer(),
    )

    manifest = ray.get(cursor.get_finalization_manifest.remote("sample"))
    assert manifest.failure_reason is not None
    assert "prefix_mismatch:first_divergent_token=2" in manifest.failure_reason
    assert id(request) not in worker._rollout_requests


@pytest.mark.asyncio
async def test_worker_rejects_mismatched_rollout_id_alias_pair() -> None:
    cursor = RolloutCursorRegistry.remote(lease_ttl_s=10, cursor_ttl_s=100)
    secret = b"alias-check-secret"
    context = mint_rollout_context(
        sample_id="sample",
        group_id="group",
        weight_version=4,
        secret=secret,
        ttl_s=60,
    )
    worker = object.__new__(VllmAsyncGenerationWorkerImpl)
    worker._rollout_writer_cfg = SimpleNamespace(enabled=True)
    worker._rollout_writer_secret = secret
    worker._rollout_cursor = cursor
    worker._rollout_requests = {}
    worker._rollout_prompt_tokens = {}
    request = SimpleNamespace(
        nemo_rl_rollout_context=context.to_dict(),
        nemo_rl_rollout_id="a-different-rollout",
        required_prefix_token_ids=None,
        stream=False,
    )

    await worker._prepare_rollout_request(request, prompt_token_ids=[1, 2, 3])

    # The request is served uncollected: no request state is retained and no
    # cursor was ever created (rejection precedes the reservation).
    assert id(request) not in worker._rollout_requests
    with pytest.raises(ray.exceptions.RayTaskError, match="unknown rollout sample"):
        ray.get(cursor.get_finalization_manifest.remote("sample"))


def _gateway_worker(cursor, *, accept_gateway_identity=True, weight_version=5):
    worker = object.__new__(VllmAsyncGenerationWorkerImpl)
    worker._rollout_writer_cfg = SimpleNamespace(
        enabled=True,
        accept_gateway_identity=accept_gateway_identity,
        staging_partition="staging",
    )
    worker._rollout_writer_secret = None
    worker._rollout_cursor = cursor
    worker._rollout_requests = {}
    worker._rollout_prompt_tokens = {}
    worker._rollout_response_tokens = {}
    worker._rollout_weight_version = weight_version
    return worker


def _chat_response(generated_ids, generated_logprobs):
    payload = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "hi",
                    "generation_token_ids": list(generated_ids),
                    "generation_log_probs": list(generated_logprobs),
                }
            }
        ]
    }
    return SimpleNamespace(model_dump=lambda: payload)


@pytest.mark.asyncio
async def test_worker_stages_gateway_identified_call_without_token_echo() -> None:
    """A trusted-gateway call carries {rollout_id, call_id}, no signed context
    and no prefix echo: the worker verifies the committed prefix by hash,
    stages under <rollout_id>/<call_id>, and never echoes tokens back."""
    cursor = RolloutCursorRegistry.remote(lease_ttl_s=10, cursor_ttl_s=100)
    worker = _gateway_worker(cursor)
    client = NoOpDataPlaneClient()
    client.register_partition(
        "staging", list(ROLLOUT_STAGING_FIELDS), 10, consumer_tasks=[]
    )
    worker._rollout_dp_client = client

    # Turn 0: prompt [1,2,3], generates [4,5].
    first = SimpleNamespace(
        nemo_rl_rollout_context=None,
        nemo_rl_rollout_id="rid",
        nemo_rl_call_id="c1",
        required_prefix_token_ids=None,
        stream=False,
    )
    await worker._prepare_rollout_request(first, prompt_token_ids=[1, 2, 3])
    assert id(first) in worker._rollout_requests
    await worker._stage_rollout_response(first, _chat_response([4, 5], [-0.1, -0.2]))
    # No token echo for gateway-identified calls.
    assert id(first) not in worker._rollout_response_tokens

    # Turn 1: rendered prompt extends the committed sequence; no echo needed.
    second = SimpleNamespace(
        nemo_rl_rollout_context=None,
        nemo_rl_rollout_id="rid",
        nemo_rl_call_id="c2",
        required_prefix_token_ids=None,
        stream=False,
    )
    await worker._prepare_rollout_request(second, prompt_token_ids=[1, 2, 3, 4, 5, 6])
    await worker._stage_rollout_response(second, _chat_response([7], [-0.3]))

    manifest = ray.get(cursor.get_finalization_manifest.remote("rid"))
    assert manifest.failure_reason is None
    assert [t.staging_key for t in manifest.turns] == ["rid/c1", "rid/c2"]
    assert all(t.weight_version == 5 for t in manifest.turns)
    assert all(t.group_id == "" for t in manifest.turns)
    assert manifest.committed_length == 7


@pytest.mark.asyncio
async def test_worker_rejects_gateway_identity_when_not_accepted() -> None:
    cursor = RolloutCursorRegistry.remote(lease_ttl_s=10, cursor_ttl_s=100)
    worker = _gateway_worker(cursor, accept_gateway_identity=False)
    request = SimpleNamespace(
        nemo_rl_rollout_id="rid",
        nemo_rl_call_id="c1",
        nemo_rl_rollout_context=None,
        required_prefix_token_ids=None,
        stream=False,
    )
    await worker._prepare_rollout_request(request, prompt_token_ids=[1, 2])
    assert id(request) not in worker._rollout_requests
    with pytest.raises(ray.exceptions.RayTaskError, match="unknown rollout sample"):
        ray.get(cursor.get_finalization_manifest.remote("rid"))


@pytest.mark.asyncio
async def test_worker_fails_gateway_call_when_rendered_prompt_diverges() -> None:
    """Rendering that no longer reproduces the committed prefix (harness
    compaction / history rewrite) fails the linear cursor by hash."""
    cursor = RolloutCursorRegistry.remote(lease_ttl_s=10, cursor_ttl_s=100)
    committed = [1, 2, 3]
    first = ray.get(cursor.reserve_turn.remote("rid", "call:c1"))
    ray.get(
        cursor.commit_turn.remote(
            "rid",
            first.lease,
            staging_key="rid/c1",
            new_len=len(committed),
            new_hash=hash_token_ids(committed),
            group_id="",
            weight_version=5,
        )
    )
    worker = _gateway_worker(cursor)
    request = SimpleNamespace(
        nemo_rl_rollout_context=None,
        nemo_rl_rollout_id="rid",
        nemo_rl_call_id="c2",
        required_prefix_token_ids=None,
        stream=False,
    )

    class _Tokenizer:
        @staticmethod
        def decode(token_ids):
            return " ".join(map(str, token_ids))

    await worker._prepare_rollout_request(
        request, prompt_token_ids=[1, 9, 9, 9], tokenizer=_Tokenizer()
    )
    assert id(request) not in worker._rollout_requests
    manifest = ray.get(cursor.get_finalization_manifest.remote("rid"))
    assert manifest.failure_reason is not None
    assert "prefix_mismatch" in manifest.failure_reason


def test_finalizer_assembles_valid_row_and_masks_missing_row() -> None:
    client = NoOpDataPlaneClient()
    client.register_partition(
        "staging", list(ROLLOUT_STAGING_FIELDS), 10, consumer_tasks=[]
    )
    cursor = RolloutCursorRegistry.remote(lease_ttl_s=10, cursor_ttl_s=100)

    reservation = ray.get(cursor.reserve_turn.remote("valid", "nonce"))
    ids = [1, 2, 3, 4]
    client.put_samples(
        ["valid/t0"],
        "staging",
        fields=TensorDict(
            {
                "token_ids_delta": torch.tensor([ids]),
                "token_mask_delta": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
                "generation_logprobs_delta": torch.tensor([[0.0, 0.0, -0.1, -0.2]]),
            },
            batch_size=[1],
        ),
    )
    ray.get(
        cursor.commit_turn.remote(
            "valid",
            reservation.lease,
            staging_key="valid/t0",
            new_len=4,
            new_hash=hash_token_ids(ids),
            group_id="group-valid",
            weight_version=3,
        )
    )
    ray.get(cursor.fail_sample.remote("missing", reason="missing_turn"))

    legacy_bulk = BatchedDataDict(
        {
            "input_ids": torch.zeros((2, 6), dtype=torch.int64),
            "input_lengths": torch.tensor([4, 4]),
            "generation_logprobs": torch.zeros((2, 6)),
            "token_mask": torch.zeros((2, 6)),
            "sample_mask": torch.ones(2),
        }
    )
    legacy_carry = BatchedDataDict(
        {
            "total_reward": torch.tensor([1.0, 9.0]),
            "loss_multiplier": torch.ones(2),
            "trajectory_valid_mask": torch.ones(2),
            "input_lengths": torch.tensor([4, 4]),
            "response_token_lengths": torch.tensor([2, 2]),
        }
    )
    finalized = assemble_staged_batch(
        dp_client=client,
        cursor=cursor,
        staging_partition="staging",
        sample_ids=["valid", "missing"],
        group_ids=["group-valid", "group-missing"],
        weight_version=3,
        legacy_bulk=legacy_bulk,
        legacy_carry=legacy_carry,
        pad_token_id=0,
    )
    assert finalized.bulk_batch["input_ids"][0, :4].tolist() == ids
    assert finalized.bulk_batch["sample_mask"].tolist() == [1.0, 0.0]
    assert finalized.driver_carry["trajectory_valid_mask"].tolist() == [1.0, 0.0]
    assert finalized.driver_carry["total_reward"].tolist() == [1.0, 0.0]


def test_finalizer_times_out_missing_committed_staging_row() -> None:
    client = NoOpDataPlaneClient()
    client.register_partition(
        "staging-missing", list(ROLLOUT_STAGING_FIELDS), 2, consumer_tasks=[]
    )
    cursor = RolloutCursorRegistry.remote(lease_ttl_s=10, cursor_ttl_s=100)
    reservation = ray.get(cursor.reserve_turn.remote("sample", "nonce"))
    ray.get(
        cursor.commit_turn.remote(
            "sample",
            reservation.lease,
            staging_key="sample/t0",
            new_len=2,
            new_hash=hash_token_ids([1, 2]),
            group_id="group",
            weight_version=1,
        )
    )
    legacy_bulk = BatchedDataDict(
        {
            "input_ids": torch.zeros((1, 4), dtype=torch.int64),
            "input_lengths": torch.tensor([2]),
            "generation_logprobs": torch.zeros((1, 4)),
            "token_mask": torch.zeros((1, 4)),
            "sample_mask": torch.ones(1),
        }
    )
    legacy_carry = BatchedDataDict(
        {
            "total_reward": torch.ones(1),
            "loss_multiplier": torch.ones(1),
            "trajectory_valid_mask": torch.ones(1),
            "input_lengths": torch.tensor([2]),
            "response_token_lengths": torch.tensor([1]),
        }
    )
    finalized = assemble_staged_batch(
        dp_client=client,
        cursor=cursor,
        staging_partition="staging-missing",
        sample_ids=["sample"],
        group_ids=["group"],
        weight_version=1,
        legacy_bulk=legacy_bulk,
        legacy_carry=legacy_carry,
        pad_token_id=0,
        finalize_timeout_s=0.01,
        poll_interval_s=0.001,
    )
    assert finalized.bulk_batch["sample_mask"].tolist() == [0.0]
    assert finalized.manifest_rows[0]["rejection_reason"].startswith(
        "missing_staging_row"
    )
    assert finalized.staging_keys == ("sample/t0",)


@pytest.mark.parametrize(
    ("corruption", "expected_reason"),
    [
        ("hash", "staging_hash_mismatch"),
        ("mask", "invalid_token_mask"),
        ("logprob", "non_finite_generation_logprob"),
        ("identity", "identity_or_weight_version_mismatch"),
        ("weight_version", "identity_or_weight_version_mismatch"),
    ],
)
def test_finalizer_rejects_corrupt_staging_rows(
    corruption: str, expected_reason: str
) -> None:
    client = NoOpDataPlaneClient()
    partition = f"staging-corrupt-{corruption}"
    client.register_partition(
        partition, list(ROLLOUT_STAGING_FIELDS), 2, consumer_tasks=[]
    )
    cursor = RolloutCursorRegistry.remote(lease_ttl_s=10, cursor_ttl_s=100)
    reservation = ray.get(cursor.reserve_turn.remote("sample", "nonce"))
    delta_ids = [1, 2] if corruption != "hash" else [1, 9]
    delta_mask = [0.0, 1.0] if corruption != "mask" else [0.0, 2.0]
    delta_logprobs = [0.0, -0.1] if corruption != "logprob" else [0.0, float("nan")]
    client.put_samples(
        ["sample/t0"],
        partition,
        fields=TensorDict(
            {
                "token_ids_delta": torch.tensor([delta_ids]),
                "token_mask_delta": torch.tensor([delta_mask]),
                "generation_logprobs_delta": torch.tensor([delta_logprobs]),
            },
            batch_size=[1],
        ),
    )
    ray.get(
        cursor.commit_turn.remote(
            "sample",
            reservation.lease,
            staging_key="sample/t0",
            new_len=2,
            new_hash=hash_token_ids([1, 2]),
            group_id="wrong-group" if corruption == "identity" else "group",
            weight_version=2 if corruption == "weight_version" else 1,
        )
    )
    legacy_bulk = BatchedDataDict(
        {
            "input_ids": torch.zeros((1, 4), dtype=torch.int64),
            "input_lengths": torch.tensor([2]),
            "generation_logprobs": torch.zeros((1, 4)),
            "token_mask": torch.zeros((1, 4)),
            "sample_mask": torch.ones(1),
        }
    )
    legacy_carry = BatchedDataDict(
        {
            "total_reward": torch.tensor([7.0]),
            "loss_multiplier": torch.ones(1),
            "trajectory_valid_mask": torch.ones(1),
            "input_lengths": torch.tensor([2]),
            "response_token_lengths": torch.tensor([1]),
        }
    )

    finalized = assemble_staged_batch(
        dp_client=client,
        cursor=cursor,
        staging_partition=partition,
        sample_ids=["sample"],
        group_ids=["group"],
        weight_version=1,
        legacy_bulk=legacy_bulk,
        legacy_carry=legacy_carry,
        pad_token_id=0,
    )

    assert expected_reason in finalized.manifest_rows[0]["rejection_reason"]
    assert finalized.bulk_batch["sample_mask"].tolist() == [0.0]
    assert finalized.driver_carry["trajectory_valid_mask"].tolist() == [0.0]
    assert finalized.driver_carry["total_reward"].tolist() == [0.0]


class _RemoteManifestMethod:
    def __init__(self, manifest: FinalizationManifest) -> None:
        self.manifest = manifest

    def remote(self, _sample_id: str):
        return ray.put(self.manifest)


class _ManifestCursor:
    def __init__(self, manifest: FinalizationManifest) -> None:
        self.get_finalization_manifest = _RemoteManifestMethod(manifest)


@pytest.mark.parametrize("turn_numbers", [(1, 0), (0, 0)])
def test_finalizer_rejects_reordered_or_duplicated_turn_manifest(
    turn_numbers: tuple[int, int],
) -> None:
    turns = tuple(
        TurnRecord(
            turn=turn,
            request_nonce=f"nonce-{index}",
            lease=f"lease-{index}",
            prev_len=index,
            prev_hash=hash_token_ids(list(range(index))),
            state="committed",
            updated_at=1.0,
            staging_key=f"sample/t{index}",
            new_len=index + 1,
            new_hash=hash_token_ids(list(range(index + 1))),
            group_id="group",
            weight_version=1,
        )
        for index, turn in enumerate(turn_numbers)
    )
    cursor = _ManifestCursor(
        FinalizationManifest(
            sample_id="sample",
            committed_length=2,
            prefix_hash=turns[-1].new_hash or "",
            turns=turns,
            failure_reason=None,
        )
    )
    client = NoOpDataPlaneClient()
    client.register_partition(
        "staging-manifest", list(ROLLOUT_STAGING_FIELDS), 2, consumer_tasks=[]
    )
    client.put_samples(
        ["sample/t0"],
        "staging-manifest",
        fields=TensorDict(
            {
                "token_ids_delta": torch.tensor([[0]]),
                "token_mask_delta": torch.tensor([[0.0]]),
                "generation_logprobs_delta": torch.tensor([[0.0]]),
            },
            batch_size=[1],
        ),
    )
    legacy_bulk = BatchedDataDict(
        {
            "input_ids": torch.zeros((1, 4), dtype=torch.int64),
            "input_lengths": torch.tensor([2]),
            "generation_logprobs": torch.zeros((1, 4)),
            "token_mask": torch.zeros((1, 4)),
            "sample_mask": torch.ones(1),
        }
    )
    legacy_carry = BatchedDataDict(
        {
            "total_reward": torch.ones(1),
            "loss_multiplier": torch.ones(1),
            "trajectory_valid_mask": torch.ones(1),
            "input_lengths": torch.tensor([2]),
            "response_token_lengths": torch.tensor([1]),
        }
    )

    finalized = assemble_staged_batch(
        dp_client=client,
        cursor=cursor,
        staging_partition="staging-manifest",
        sample_ids=["sample"],
        group_ids=["group"],
        weight_version=1,
        legacy_bulk=legacy_bulk,
        legacy_carry=legacy_carry,
        pad_token_id=0,
    )

    assert finalized.manifest_rows[0]["rejection_reason"] == "non_contiguous_turns"
    assert finalized.bulk_batch["sample_mask"].tolist() == [0.0]
