# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Small deterministic BM25 server shared by all comparison implementations."""

import argparse
import json
import math
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


@dataclass(frozen=True)
class Document:
    """One indexed comparison passage."""

    id: str
    title: str
    text: str
    contents: str


class RetrieveRequest(BaseModel):
    """Union of the Search-R1 and ZeroSearch request shapes."""

    queries: list[str] | None = None
    query: str | None = None
    topk: int | None = None
    top_k: int | None = None
    return_scores: bool = False


class BM25Index:
    """In-memory BM25 index with deterministic score and tie ordering."""

    def __init__(self, corpus_path: Path) -> None:
        self.documents: list[Document] = []
        with corpus_path.open(encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                self.documents.append(
                    Document(
                        id=str(row["id"]),
                        title=str(row["title"]),
                        text=str(row["text"]),
                        contents=str(row["contents"]),
                    )
                )
        if not self.documents:
            raise ValueError(f"The retrieval corpus is empty: {corpus_path}")

        self.term_frequencies = [
            Counter(self._tokenize(document.contents)) for document in self.documents
        ]
        self.lengths = [sum(frequencies.values()) for frequencies in self.term_frequencies]
        self.average_length = sum(self.lengths) / len(self.lengths)
        document_frequencies: Counter[str] = Counter()
        for frequencies in self.term_frequencies:
            document_frequencies.update(frequencies.keys())
        document_count = len(self.documents)
        self.idf = {
            token: math.log(1.0 + (document_count - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequencies.items()
        }

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return TOKEN_PATTERN.findall(text.casefold())

    def search(self, query: str, top_k: int) -> list[tuple[Document, float]]:
        if top_k < 1 or top_k > len(self.documents):
            raise ValueError(f"top_k must be in [1, {len(self.documents)}]")
        query_terms = self._tokenize(query)
        k1 = 1.2
        b = 0.75
        scored: list[tuple[float, int]] = []
        for index, frequencies in enumerate(self.term_frequencies):
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if frequency == 0:
                    continue
                denominator = frequency + k1 * (
                    1.0 - b + b * self.lengths[index] / self.average_length
                )
                score += self.idf.get(term, 0.0) * frequency * (k1 + 1.0) / denominator
            scored.append((score, index))
        ranked = sorted(scored, key=lambda item: (-item[0], item[1]))[:top_k]
        return [(self.documents[index], score) for score, index in ranked]


def create_app(index: BM25Index, log_path: Path | None) -> FastAPI:
    """Create a dual-protocol retrieval application."""
    app = FastAPI()
    log_lock = threading.Lock()

    def record(payload: dict[str, object]) -> None:
        if log_path is None:
            return
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_lock, log_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(payload, sort_keys=True) + "\n")

    @app.get("/health")
    def health() -> dict[str, object]:
        return {"status": "ok", "documents": len(index.documents)}

    @app.post("/retrieve")
    def retrieve(request: RetrieveRequest):
        started = time.perf_counter()
        if request.queries is not None:
            queries = request.queries
            protocol = "search_r1"
            top_k = request.topk if request.topk is not None else 3
        elif request.query is not None:
            queries = [request.query]
            protocol = "zero_search"
            top_k = request.top_k if request.top_k is not None else 3
        else:
            raise ValueError("Either queries or query must be supplied")

        ranked_batches = [index.search(query, top_k) for query in queries]
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        record(
            {
                "timestamp_ns": time.time_ns(),
                "protocol": protocol,
                "queries": len(queries),
                "top_k": top_k,
                "elapsed_ms": elapsed_ms,
            }
        )

        if request.queries is not None:
            result = []
            for ranked in ranked_batches:
                result.append(
                    [
                        {
                            "document": {
                                "id": document.id,
                                "title": document.title,
                                "text": document.text,
                                "contents": document.contents,
                            },
                            "score": score,
                        }
                        for document, score in ranked
                    ]
                )
            return {"result": result, "timing_ms": elapsed_ms}

        return [
            {
                "id": document.id,
                "title": document.title,
                "text": document.text,
                "score": score,
            }
            for document, score in ranked_batches[0]
        ]

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--log-path", type=Path)
    args = parser.parse_args()
    index = BM25Index(args.corpus.resolve())
    uvicorn.run(
        create_app(index=index, log_path=args.log_path),
        host=args.host,
        port=args.port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
