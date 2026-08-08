# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reward and parsing tests for the AI-search resources server."""

from unittest.mock import MagicMock

import pytest

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.ai_search.app import (
    AISearchResourcesServer,
    AISearchResourcesServerConfig,
    AISearchVerifyRequest,
    _parse_final_answer,
    _token_f1,
)
from resources_servers.ai_search.retrieval.config import (
    EncoderConfig,
    IndexConfig,
    RewardConfig,
    SearchRuntimeConfig,
)


def _make_response(text: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="response",
        created_at=0.0,
        model="policy",
        object="response",
        output=[
            NeMoGymResponseOutputMessage(
                id="message",
                content=[
                    NeMoGymResponseOutputText(
                        annotations=[], type="output_text", text=text
                    )
                ],
                role="assistant",
                status="completed",
                type="message",
            )
        ],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    )


def test_parse_final_answer_prefers_required_format() -> None:
    answer, valid = _parse_final_answer(
        "I searched twice.\nFinal Answer: polished cerulean quartz"
    )
    assert answer == "polished cerulean quartz"
    assert valid is True


def test_answer_tag_is_accepted_but_format_is_invalid() -> None:
    answer, valid = _parse_final_answer("<answer>Selka</answer>")
    assert answer == "Selka"
    assert valid is False


def test_token_f1_allows_semantically_matching_short_answer() -> None:
    assert _token_f1("the silver larch", "silver larch") == 1.0
    assert _token_f1("Rhea", "Rhea Coil") == pytest.approx(2.0 / 3.0)


@pytest.mark.asyncio
async def test_verifier_combines_decomposed_reward(tmp_path) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    embeddings_path = tmp_path / "embeddings.npy"
    corpus_path.write_text(
        '{"id":"d1","title":"Mara Voss","text":"Mara Voss was born in Selka."}\n',
        encoding="utf-8",
    )

    encoder = EncoderConfig(
        kind="hash",
        model_name="deterministic-hash",
        device="cpu",
        dtype="float32",
        batch_size=8,
        max_length=64,
        dimension=64,
        query_prefix="",
        passage_prefix="",
        normalize=True,
        trust_remote_code=False,
    )
    from resources_servers.ai_search.retrieval.corpus import (
        JsonlDocumentStore,
        sha256_file,
    )
    from resources_servers.ai_search.retrieval.encoder import (
        HashEncoder,
        encoder_manifest,
    )

    import json
    import numpy as np

    store = JsonlDocumentStore(corpus_path)
    values = HashEncoder(encoder).encode_passages(store.passages())
    np.save(embeddings_path, values)
    embeddings_path.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "corpus_sha256": sha256_file(corpus_path),
                "documents": 1,
                "dimension": 64,
                "encoder": encoder_manifest(encoder),
            }
        ),
        encoding="utf-8",
    )

    runtime = SearchRuntimeConfig(
        corpus_path=corpus_path,
        embeddings_path=embeddings_path,
        verify_artifact_hash=True,
        encoder=encoder,
        index=IndexConfig(
            kind="numpy",
            algorithm="brute_force",
            metric="cosine",
            serialized_index_path=None,
            save_built_index=False,
            graph_degree=8,
            intermediate_graph_degree=16,
            build_algorithm="ivf_pq",
            search_width=1,
            itopk_size=16,
        ),
        default_top_k=1,
        max_top_k=1,
        max_query_chars=128,
        max_passage_chars=512,
        max_search_calls=2,
        batch_max_size=8,
        batch_wait_ms=0.0,
        query_cache_size=8,
        include_scores=True,
        reward=RewardConfig(
            answer_weight=1.0,
            retrieval_weight=0.25,
            format_weight=0.1,
            efficiency_weight=0.05,
            answer_threshold_for_efficiency=0.5,
        ),
    )
    server = AISearchResourcesServer(
        config=AISearchResourcesServerConfig(
            host="127.0.0.1",
            port=8080,
            entrypoint="app.py",
            name="ai_search",
            search=runtime,
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    server._session_metrics["session"] = {
        "calls": [
            {
                "query": "Mara Voss birthplace",
                "retrieved_doc_ids": ["d1"],
                "queue_ms": 1.0,
                "encode_ms": 2.0,
                "index_ms": 3.0,
                "fetch_ms": 4.0,
                "total_ms": 9.0,
                "cache_hits": 0,
                "cache_misses": 1,
            }
        ]
    }
    from resources_servers.ai_search.app import _SessionMetrics

    server._session_metrics["session"] = _SessionMetrics.model_validate(
        server._session_metrics["session"]
    )
    request = MagicMock()
    request.session = {"session_id": "session"}
    result = await server.verify(
        request,
        AISearchVerifyRequest(
            question="Where was Mara Voss born?",
            answers=["Selka"],
            supporting_doc_ids=["d1"],
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[{"role": "user", "content": "Where was Mara Voss born?"}]
            ),
            response=_make_response("Final Answer: Selka"),
        ),
    )

    assert result.reward == pytest.approx(1.4)
    assert result.reward_components == {
        "answer": 1.0,
        "efficiency": 0.05,
        "format": 0.1,
        "retrieval": 0.25,
    }
    assert result.exact_match == 1.0
    assert result.retrieval_recall == 1.0
    assert result.num_search_calls == 1
    server._provider.close()
