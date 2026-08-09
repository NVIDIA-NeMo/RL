# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validated, user-facing configuration for the AI-search retriever."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class EncoderConfig(BaseModel):
    """Configuration for turning text into dense vectors."""

    kind: Literal["e5", "hash"]
    model_name: str
    device: str
    dtype: Literal["float16", "bfloat16", "float32"]
    batch_size: int = Field(ge=1)
    max_length: int = Field(ge=8)
    dimension: int = Field(ge=1)
    query_prefix: str
    passage_prefix: str
    normalize: bool
    trust_remote_code: bool


class IndexConfig(BaseModel):
    """Configuration for the vector index.

    ``numpy`` exists as a correctness baseline and CPU fallback for tests. The
    production example uses ``cuvs``.
    """

    kind: Literal["cuvs", "numpy"]
    algorithm: Literal["brute_force", "cagra"]
    metric: Literal["cosine", "sqeuclidean"]
    serialized_index_path: Path | None
    save_built_index: bool
    graph_degree: int = Field(ge=2)
    intermediate_graph_degree: int = Field(ge=2)
    build_algorithm: Literal["ivf_pq", "nn_descent", "iterative_cagra_search", "ace"]
    search_width: int = Field(ge=1)
    itopk_size: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_backend_options(self) -> "IndexConfig":
        if self.kind == "numpy" and self.algorithm != "brute_force":
            raise ValueError("The numpy reference backend only supports brute_force")
        if self.algorithm == "cagra" and self.metric == "cosine":
            # cuVS CAGRA supports cosine, but normalized vectors with squared
            # Euclidean distance are faster and preserve the same ordering.
            raise ValueError(
                "CAGRA must use sqeuclidean; normalize embeddings to preserve cosine ordering"
            )
        if self.intermediate_graph_degree < self.graph_degree:
            raise ValueError(
                "intermediate_graph_degree must be greater than or equal to graph_degree"
            )
        return self


class RewardConfig(BaseModel):
    """Weights used to combine independent, observable reward components."""

    answer_metric: Literal["token_f1", "exact_match"] = "token_f1"
    answer_weight: float = Field(ge=0.0)
    retrieval_weight: float = Field(ge=0.0)
    format_weight: float = Field(ge=0.0)
    efficiency_weight: float = Field(ge=0.0)
    answer_threshold_for_efficiency: float = Field(ge=0.0, le=1.0)


class SearchRuntimeConfig(BaseModel):
    """Configuration shared by the server, engine, and asynchronous batcher."""

    provider: Literal["dense", "http"] = "dense"
    http_url: str | None = None
    http_timeout_s: float = Field(default=10.0, gt=0.0)
    corpus_path: Path
    embeddings_path: Path
    verify_artifact_hash: bool
    encoder: EncoderConfig
    index: IndexConfig
    default_top_k: int = Field(ge=1)
    max_top_k: int = Field(ge=1)
    max_query_chars: int = Field(ge=1)
    max_passage_chars: int = Field(ge=1)
    max_search_calls: int = Field(ge=1)
    batch_max_size: int = Field(ge=1)
    batch_wait_ms: float = Field(ge=0.0)
    query_cache_size: int = Field(ge=0)
    include_scores: bool
    reward: RewardConfig

    @model_validator(mode="after")
    def validate_limits(self) -> "SearchRuntimeConfig":
        if self.default_top_k > self.max_top_k:
            raise ValueError("default_top_k cannot be greater than max_top_k")
        if self.provider == "http" and not self.http_url:
            raise ValueError("http_url is required when provider is http")
        return self
