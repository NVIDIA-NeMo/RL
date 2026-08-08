# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build passage embeddings and optional serialized cuVS index artifacts."""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from resources_servers.ai_search.retrieval.config import SearchRuntimeConfig
from resources_servers.ai_search.retrieval.corpus import (
    JsonlDocumentStore,
    sha256_file,
)
from resources_servers.ai_search.retrieval.encoder import (
    build_encoder,
    encoder_manifest,
)
from resources_servers.ai_search.retrieval.index import build_vector_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode a JSONL corpus and prepare cuVS artifacts"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ai_search.yaml"),
        help="NeMo Gym resources-server config",
    )
    parser.add_argument(
        "--force", action="store_true", help="Replace existing embedding artifacts"
    )
    parser.add_argument(
        "--build-serialized-index",
        action="store_true",
        help="Also build/save the configured cuVS index now",
    )
    return parser.parse_args()


def _load_runtime_config(config_path: Path) -> SearchRuntimeConfig:
    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    try:
        config_dict = raw["ai_search"]["resources_servers"]["ai_search"]["search"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"Could not find ai_search.resources_servers.ai_search.search in {config_path}"
        ) from error
    config = SearchRuntimeConfig.model_validate(config_dict)
    server_dir = Path(__file__).resolve().parent
    config.corpus_path = (
        config.corpus_path
        if config.corpus_path.is_absolute()
        else (server_dir / config.corpus_path).resolve()
    )
    config.embeddings_path = (
        config.embeddings_path
        if config.embeddings_path.is_absolute()
        else (server_dir / config.embeddings_path).resolve()
    )
    if config.index.serialized_index_path is not None:
        config.index.serialized_index_path = (
            config.index.serialized_index_path
            if config.index.serialized_index_path.is_absolute()
            else (server_dir / config.index.serialized_index_path).resolve()
        )
    return config


def _atomic_save_array(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as file_obj:
        np.save(file_obj, values)
    os.replace(temporary, path)


def _atomic_save_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = _load_runtime_config(config_path)
    manifest_path = config.embeddings_path.with_suffix(".manifest.json")
    if (config.embeddings_path.exists() or manifest_path.exists()) and not args.force:
        raise FileExistsError(
            "Embedding artifacts already exist. Pass --force to replace them: "
            f"{config.embeddings_path}"
        )

    store = JsonlDocumentStore(config.corpus_path)
    encoder = build_encoder(config.encoder)
    started = datetime.now(tz=timezone.utc)
    embeddings = encoder.encode_passages(store.passages())
    if embeddings.shape != (len(store), config.encoder.dimension):
        raise RuntimeError(
            "Encoder returned an unexpected shape: "
            f"{embeddings.shape} != {(len(store), config.encoder.dimension)}"
        )
    embeddings = np.asarray(embeddings, dtype=np.float32, order="C")
    _atomic_save_array(config.embeddings_path, embeddings)

    manifest = {
        "schema_version": 1,
        "created_at": started.isoformat(),
        "corpus_path": str(config.corpus_path),
        "corpus_sha256": sha256_file(config.corpus_path),
        "documents": len(store),
        "dimension": embeddings.shape[1],
        "dtype": str(embeddings.dtype),
        "encoder": encoder_manifest(config.encoder),
    }
    _atomic_save_json(manifest_path, manifest)
    print(
        f"Wrote {len(store)} x {embeddings.shape[1]} embeddings to "
        f"{config.embeddings_path}"
    )

    if args.build_serialized_index:
        if config.index.kind != "cuvs":
            raise ValueError("--build-serialized-index requires index.kind=cuvs")
        if config.index.serialized_index_path is None:
            raise ValueError(
                "--build-serialized-index requires index.serialized_index_path"
            )
        if config.index.serialized_index_path.exists() and not args.force:
            raise FileExistsError(
                f"Serialized index already exists: {config.index.serialized_index_path}"
            )
        config.index.save_built_index = True
        vector_index = build_vector_index(embeddings, config.index)
        print(
            f"Wrote {config.index.algorithm} cuVS index to "
            f"{config.index.serialized_index_path} "
            f"({vector_index.build_time_ms:.2f} ms)"
        )


if __name__ == "__main__":
    main()
