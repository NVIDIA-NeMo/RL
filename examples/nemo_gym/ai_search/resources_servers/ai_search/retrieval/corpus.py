# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Document storage abstractions for the AI-search example."""

import hashlib
import json
from pathlib import Path
from typing import Protocol

from resources_servers.ai_search.retrieval.types import Document


class DocumentStore(Protocol):
    """Random-access document store contract."""

    def __len__(self) -> int:
        """Return the number of documents."""
        ...

    def get(self, row_index: int) -> Document:
        """Return the document at a vector-index row."""
        ...


class JsonlDocumentStore:
    """Validated in-memory document store backed by a JSONL corpus."""

    def __init__(self, corpus_path: Path) -> None:
        if not corpus_path.is_file():
            raise FileNotFoundError(f"Corpus file does not exist: {corpus_path}")

        documents: list[Document] = []
        seen_ids: set[str] = set()
        with corpus_path.open("r", encoding="utf-8") as corpus_file:
            for line_number, line in enumerate(corpus_file, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON in {corpus_path} at line {line_number}"
                    ) from error

                missing = {"id", "title", "text"} - row.keys()
                if missing:
                    raise ValueError(
                        f"Corpus line {line_number} is missing fields: {sorted(missing)}"
                    )
                document = Document(
                    id=str(row["id"]),
                    title=str(row["title"]),
                    text=str(row["text"]),
                )
                if not document.id or not document.text:
                    raise ValueError(
                        f"Corpus line {line_number} has an empty id or text"
                    )
                if document.id in seen_ids:
                    raise ValueError(f"Duplicate document id: {document.id}")
                seen_ids.add(document.id)
                documents.append(document)

        if not documents:
            raise ValueError(f"Corpus is empty: {corpus_path}")

        self._documents = tuple(documents)
        self.path = corpus_path

    def __len__(self) -> int:
        return len(self._documents)

    def get(self, row_index: int) -> Document:
        return self._documents[row_index]

    def passages(self) -> list[str]:
        """Return title-prefixed text used for passage embedding."""
        return [f"{doc.title}\n{doc.text}" for doc in self._documents]


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Calculate a streaming SHA-256 digest without loading the file at once."""
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        while chunk := file_obj.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
