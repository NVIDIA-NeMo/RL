from unittest.mock import Mock
from pathlib import Path

from resources_servers.ai_search.retrieval.config import (
    EncoderConfig,
    IndexConfig,
    RewardConfig,
    SearchRuntimeConfig,
)
from resources_servers.ai_search.retrieval.http import HttpSearchProvider


def test_http_provider_converts_search_r1_response() -> None:
    config = SearchRuntimeConfig(
        provider="http",
        http_url="http://127.0.0.1:8100/retrieve",
        corpus_path=Path("unused-corpus.jsonl"),
        embeddings_path=Path("unused-embeddings.npy"),
        verify_artifact_hash=False,
        encoder=EncoderConfig(
            kind="hash",
            model_name="unused",
            device="cpu",
            dtype="float32",
            batch_size=1,
            max_length=8,
            dimension=8,
            query_prefix="",
            passage_prefix="",
            normalize=True,
            trust_remote_code=False,
        ),
        index=IndexConfig(
            kind="numpy",
            algorithm="brute_force",
            metric="cosine",
            serialized_index_path=None,
            save_built_index=False,
            graph_degree=2,
            intermediate_graph_degree=2,
            build_algorithm="ivf_pq",
            search_width=1,
            itopk_size=2,
        ),
        default_top_k=1,
        max_top_k=1,
        max_query_chars=128,
        max_passage_chars=512,
        max_search_calls=2,
        batch_max_size=8,
        batch_wait_ms=0.0,
        query_cache_size=0,
        include_scores=True,
        reward=RewardConfig(
            answer_weight=1.0,
            retrieval_weight=0.0,
            format_weight=0.0,
            efficiency_weight=0.0,
            answer_threshold_for_efficiency=0.5,
        ),
    )
    provider = HttpSearchProvider(config)
    response = Mock()
    response.json.return_value = {
        "result": [
            [
                {
                    "document": {
                        "id": "d1",
                        "title": "Title",
                        "text": "Passage",
                        "contents": "Title\nPassage",
                    },
                    "score": 2.5,
                }
            ]
        ]
    }
    response.raise_for_status.return_value = None
    provider._session.post = Mock(return_value=response)

    results = provider.search_batch(["query"], top_k=1)

    assert results[0].hits[0].document.id == "d1"
    assert results[0].hits[0].score == 2.5
    assert results[0].timings.batch_size == 1
    provider.close()
