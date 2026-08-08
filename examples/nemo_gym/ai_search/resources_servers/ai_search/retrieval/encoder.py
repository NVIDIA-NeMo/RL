# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Query and passage encoders used by dense retrieval."""

import hashlib
import re
from collections.abc import Iterable
from typing import Any

import numpy as np

from resources_servers.ai_search.retrieval.config import EncoderConfig
from resources_servers.ai_search.retrieval.types import FloatMatrix, QueryEncoder


_TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


class HashEncoder:
    """Deterministic dependency-light encoder for tests and smoke diagnostics.

    This encoder is deliberately simple and is not the production default. It
    lets CPU-only tests exercise the same document-store and index contracts.
    """

    def __init__(self, config: EncoderConfig) -> None:
        self.config = config

    @property
    def dimension(self) -> int:
        return self.config.dimension

    def _encode(self, texts: Iterable[str]) -> FloatMatrix:
        text_list = list(texts)
        vectors = np.zeros((len(text_list), self.dimension), dtype=np.float32)
        for row_index, text in enumerate(text_list):
            tokens = _TOKEN_PATTERN.findall(text.casefold())
            features = tokens + [
                f"{left}::{right}" for left, right in zip(tokens, tokens[1:])
            ]
            for feature in features:
                digest = hashlib.blake2b(
                    feature.encode("utf-8"), digest_size=8
                ).digest()
                value = int.from_bytes(digest, byteorder="little", signed=False)
                column = value % self.dimension
                sign = 1.0 if value & (1 << 63) else -1.0
                vectors[row_index, column] += sign

        if self.config.normalize and len(text_list) > 0:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            np.divide(vectors, np.maximum(norms, 1e-12), out=vectors)
        return vectors

    def encode_queries(self, texts: list[str]) -> FloatMatrix:
        return self._encode(texts)

    def encode_passages(self, texts: list[str]) -> FloatMatrix:
        return self._encode(texts)

    def close(self) -> None:
        """Release resources (the hash encoder owns none)."""


class E5Encoder:
    """Hugging Face E5 encoder with mean pooling and normalized outputs."""

    def __init__(self, config: EncoderConfig) -> None:
        self.config = config
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "The E5 encoder requires torch and transformers. Install the "
                "AI-search server project before starting the environment."
            ) from error

        if config.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"Encoder device is {config.device!r}, but CUDA is unavailable"
            )

        dtype_by_name = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=config.trust_remote_code,
        )
        self._model = AutoModel.from_pretrained(
            config.model_name,
            dtype=dtype_by_name[config.dtype],
            trust_remote_code=config.trust_remote_code,
        )
        self._model.to(config.device)
        self._model.eval()

        actual_dimension = int(self._model.config.hidden_size)
        if actual_dimension != config.dimension:
            raise ValueError(
                f"Encoder dimension mismatch: config={config.dimension}, "
                f"model={actual_dimension} ({config.model_name})"
            )

    @property
    def dimension(self) -> int:
        return self.config.dimension

    def _encode(self, texts: list[str], prefix: str) -> FloatMatrix:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        output_batches: list[FloatMatrix] = []
        for start in range(0, len(texts), self.config.batch_size):
            batch = [
                f"{prefix}{text}"
                for text in texts[start : start + self.config.batch_size]
            ]
            tokenized = self._tokenizer(
                batch,
                max_length=self.config.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            tokenized = {
                key: value.to(self.config.device) for key, value in tokenized.items()
            }
            with self._torch.inference_mode():
                hidden = self._model(**tokenized).last_hidden_state
                attention_mask = tokenized["attention_mask"].unsqueeze(-1)
                pooled = (hidden * attention_mask).sum(dim=1) / attention_mask.sum(
                    dim=1
                ).clamp(min=1)
                pooled = pooled.float()
                if self.config.normalize:
                    pooled = self._torch.nn.functional.normalize(pooled, dim=1)
            output_batches.append(
                np.asarray(pooled.cpu().numpy(), dtype=np.float32, order="C")
            )

        return np.concatenate(output_batches, axis=0)

    def encode_queries(self, texts: list[str]) -> FloatMatrix:
        return self._encode(texts, self.config.query_prefix)

    def encode_passages(self, texts: list[str]) -> FloatMatrix:
        return self._encode(texts, self.config.passage_prefix)

    def close(self) -> None:
        """Release model allocations while CUDA libraries are still loaded."""
        model = getattr(self, "_model", None)
        if model is None:
            return
        self._model = None
        del model
        if self.config.device.startswith("cuda"):
            self._torch.cuda.empty_cache()


def build_encoder(config: EncoderConfig) -> QueryEncoder:
    """Construct a supported encoder from an explicit registry."""
    encoders: dict[str, type[HashEncoder] | type[E5Encoder]] = {
        "hash": HashEncoder,
        "e5": E5Encoder,
    }
    try:
        encoder_type = encoders[config.kind]
    except KeyError as error:  # pragma: no cover - Pydantic rejects this first.
        raise ValueError(f"Unsupported encoder kind: {config.kind}") from error
    return encoder_type(config)


def encoder_manifest(config: EncoderConfig) -> dict[str, Any]:
    """Return stable encoder fields written alongside embedding artifacts."""
    return {
        "kind": config.kind,
        "model_name": config.model_name,
        "dimension": config.dimension,
        "max_length": config.max_length,
        "query_prefix": config.query_prefix,
        "passage_prefix": config.passage_prefix,
        "normalize": config.normalize,
    }
