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

"""Prepare CLEVR-CoGenT as an indexed multimodal Energon dataset.

The output contains one JSON conversation and one lossless PNG per sample. The
conversation matches NeMo-RL's existing CLEVR formatter, but refers to the image
through an Energon media index instead of embedding a base64 data URL.

Example::

    uv run nemo_rl/data/energon/scripts/prepare_energon_dataset.py \
        --splits train valA

Pass ``--image-scale`` to upscale every image by that factor along both axes
before it is written (``--image-scale 3`` yields 9x the pixel count). The
recorded media metadata always matches the bytes that are written.

Example::

    uv run nemo_rl/data/energon/scripts/prepare_energon_dataset.py \
        --splits train valA \
        --image-scale 3 \
        --output-dir /data/nemorl-datasets/clevr-v2

The default output directory is ``/data/nemorl-datasets/clevr``. Pass
``--output-dir`` to use another location.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import re
from collections import deque
from collections.abc import Iterable, Iterator, Mapping
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import Any

import webdataset as wds  # pyrefly: ignore[import-error]  Optional mcore dependency.
from datasets import DownloadConfig, load_dataset
from datasets import Image as HFImage
from megatron.energon import BaseWebdatasetFactory
from PIL import Image as PILImage

logger = logging.getLogger(__name__)

_SUPPORTED_SPLITS = ("train", "valA", "valB")
_DEFAULT_OUTPUT_DIR = Path("/data/nemorl-datasets/clevr")
_HF_DATASETS = {
    "train": "MMInstruction/Clevr_CoGenT_TrainA_70K_Complex",
    "valA": "MMInstruction/Clevr_CoGenT_ValA",
    "valB": "MMInstruction/Clevr_CoGenT_ValB",
}
_IMAGE_EXTENSIONS = {
    ".bmp": "bmp",
    ".jpeg": "jpg",
    ".jpg": "jpg",
    ".png": "png",
    ".webp": "webp",
}
_IMAGE_FORMATS = {
    "BMP": "bmp",
    "JPEG": "jpg",
    "PNG": "png",
    "WEBP": "webp",
}
# Re-encode settings per extension, chosen to stay lossless wherever the format
# allows it so that upscaling does not also degrade the source pixels.
_IMAGE_SAVE_OPTIONS: dict[str, tuple[str, dict[str, Any]]] = {
    "bmp": ("BMP", {}),
    "jpg": ("JPEG", {"quality": 95, "subsampling": 0}),
    "png": ("PNG", {"compress_level": 6}),
    "webp": ("WEBP", {"lossless": True}),
}
_DATASET_YAML = """\
__module__: megatron.energon
__class__: CrudeWebdataset
"""


def _image_bytes_extension_and_size(image: Any) -> tuple[bytes, str, int, int]:
    if not isinstance(image, Mapping):
        raise TypeError("CLEVR images must use datasets.Image(decode=False) records.")

    path = image.get("path")
    raw_bytes = image.get("bytes")
    if raw_bytes is None:
        if not isinstance(path, str) or not path:
            raise ValueError("A raw CLEVR image needs bytes or a readable path.")
        raw_bytes = Path(path).read_bytes()
    if not isinstance(raw_bytes, (bytes, bytearray, memoryview)):
        raise TypeError("Raw CLEVR image bytes have an unsupported type.")
    image_bytes = bytes(raw_bytes)

    extension = (
        _IMAGE_EXTENSIONS.get(Path(path).suffix.lower())
        if isinstance(path, str)
        else None
    )
    with PILImage.open(io.BytesIO(image_bytes)) as image_header:
        width, height = image_header.size
        if extension is None:
            extension = _IMAGE_FORMATS.get(str(image_header.format).upper())
    if extension is None:
        raise ValueError("CLEVR image format is unsupported by the Energon dataset.")
    return image_bytes, extension, width, height


def _rescale_image(
    image_bytes: bytes, extension: str, *, scale: float
) -> tuple[bytes, int, int]:
    """Upscale one encoded image by ``scale`` along both axes and re-encode it."""
    save_format, save_options = _IMAGE_SAVE_OPTIONS[extension]
    with PILImage.open(io.BytesIO(image_bytes)) as source:
        source.load()
        width = max(1, round(source.width * scale))
        height = max(1, round(source.height * scale))
        resized = source.resize((width, height), PILImage.Resampling.LANCZOS)
    buffer = io.BytesIO()
    resized.save(buffer, format=save_format, **save_options)
    return buffer.getvalue(), width, height


def _energon_payload(
    example: dict[str, Any], *, image_member: str, width: int, height: int
) -> dict[str, Any]:
    # Keep this equivalent to format_clevr_cogent_dataset() without importing
    # the full NeMo-RL dataset registry and all of its optional dependencies.
    solution = str(example["solution"])
    tagged_answer = re.search(r"<answer>(.*?)</answer>", solution)
    answer = tagged_answer.group(1).strip() if tagged_answer else solution.strip()

    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "media_index": 0},
                    {"type": "text", "text": str(example["problem"])},
                ],
            },
            {"role": "assistant", "content": answer},
        ],
        "media": [
            {
                "type": "image",
                "member": image_member,
                "metadata": {"width": width, "height": height},
            }
        ],
    }


def _decoded_examples(
    examples: Iterable[dict[str, Any]], max_samples: int | None
) -> Iterator[tuple[int, dict[str, Any], bytes, str, int, int]]:
    """Yield ``(index, example, image_bytes, extension, width, height)`` in order."""
    for index, example in enumerate(examples):
        if max_samples is not None and index >= max_samples:
            return
        image_bytes, extension, width, height = _image_bytes_extension_and_size(
            example.get("image")
        )
        yield index, example, image_bytes, extension, width, height


def _sample_records(
    *,
    split: str,
    examples: Iterable[dict[str, Any]],
    max_samples: int | None,
    image_scale: float,
    image_workers: int,
) -> Iterator[dict[str, Any]]:
    """Yield webdataset records in input order, rescaling images if requested.

    Rescaling dominates the runtime of this script, so it is farmed out to a
    process pool. Submissions are bounded so that peak memory stays proportional
    to ``image_workers`` rather than to the size of the split.
    """

    def _record(
        index: int,
        example: dict[str, Any],
        image_bytes: bytes,
        extension: str,
        width: int,
        height: int,
    ) -> dict[str, Any]:
        return {
            "__key__": f"{split}-{index:08d}",
            "json": json.dumps(
                _energon_payload(
                    example, image_member=extension, width=width, height=height
                ),
                ensure_ascii=False,
            ).encode("utf-8"),
            extension: image_bytes,
        }

    decoded = _decoded_examples(examples, max_samples)
    if image_scale == 1.0:
        for index, example, image_bytes, extension, width, height in decoded:
            yield _record(index, example, image_bytes, extension, width, height)
        return
    if image_workers == 1:
        for index, example, image_bytes, extension, _, _ in decoded:
            image_bytes, width, height = _rescale_image(
                image_bytes, extension, scale=image_scale
            )
            yield _record(index, example, image_bytes, extension, width, height)
        return

    pending: deque[tuple[int, dict[str, Any], str, Future[tuple[bytes, int, int]]]] = (
        deque()
    )

    def _next_ready() -> dict[str, Any]:
        index, example, extension, future = pending.popleft()
        image_bytes, width, height = future.result()
        return _record(index, example, image_bytes, extension, width, height)

    max_pending = image_workers * 4
    with ProcessPoolExecutor(max_workers=image_workers) as pool:
        for index, example, image_bytes, extension, _, _ in decoded:
            pending.append(
                (
                    index,
                    example,
                    extension,
                    pool.submit(
                        _rescale_image, image_bytes, extension, scale=image_scale
                    ),
                )
            )
            if len(pending) >= max_pending:
                yield _next_ready()
        while pending:
            yield _next_ready()


def _write_split(
    *,
    split: str,
    examples: Iterable[dict[str, Any]],
    output_dir: Path,
    max_samples_per_shard: int,
    max_samples: int | None,
    image_scale: float,
    image_workers: int,
) -> tuple[int, list[str]]:
    pattern = output_dir / f"{split}-shard-%06d.tar"
    count = 0
    with wds.ShardWriter(str(pattern), maxcount=max_samples_per_shard) as writer:
        for record in _sample_records(
            split=split,
            examples=examples,
            max_samples=max_samples,
            image_scale=image_scale,
            image_workers=image_workers,
        ):
            writer.write(record)
            count += 1

    shard_paths = sorted(output_dir.glob(f"{split}-shard-*.tar"))
    if count == 0:
        for shard_path in shard_paths:
            shard_path.unlink()
        raise RuntimeError(f"CLEVR split {split!r} did not produce any samples.")
    return count, [str(path.relative_to(output_dir)) for path in shard_paths]


def prepare_clevr_energon(
    *,
    output_dir: Path,
    splits: Iterable[str],
    max_samples_per_shard: int,
    max_samples: int | None,
    num_workers: int,
    download_workers: int,
    image_scale: float = 1.0,
    image_workers: int = 1,
    datasets: Mapping[str, Iterable[dict[str, Any]]] | None = None,
) -> dict[str, int]:
    """Write CLEVR shards and build the metadata required by Energon."""
    split_names = list(splits)
    if not split_names or len(split_names) != len(set(split_names)):
        raise ValueError("At least one unique split is required.")
    unsupported = sorted(set(split_names) - set(_SUPPORTED_SPLITS))
    if unsupported:
        raise ValueError(f"Unsupported CLEVR splits: {unsupported}.")
    if max_samples_per_shard <= 0:
        raise ValueError("max_samples_per_shard must be positive.")
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive when set.")
    if num_workers <= 0:
        raise ValueError("num_workers must be positive.")
    if download_workers <= 0:
        raise ValueError("download_workers must be positive.")
    if image_scale <= 0:
        raise ValueError("image_scale must be positive.")
    if image_workers <= 0:
        raise ValueError("image_workers must be positive.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    all_shards: list[str] = []
    download_config = DownloadConfig(num_proc=download_workers)
    for split in split_names:
        logger.info(
            "Converting CLEVR-CoGenT split %s (image scale %.3gx)", split, image_scale
        )
        if datasets is not None:
            examples = datasets[split]
        else:
            examples = load_dataset(
                _HF_DATASETS[split],
                download_config=download_config,
                num_proc=download_workers,
            )["train"].cast_column("image", HFImage(decode=False))
        count, shard_paths = _write_split(
            split=split,
            examples=examples,
            output_dir=output_dir,
            max_samples_per_shard=max_samples_per_shard,
            max_samples=max_samples,
            image_scale=image_scale,
            image_workers=image_workers,
        )
        counts[split] = count
        all_shards.extend(shard_paths)

    BaseWebdatasetFactory.prepare_dataset(
        output_dir,
        all_shards,
        split_parts_patterns=[
            (split, rf"{re.escape(split)}-shard-.*\.tar") for split in split_names
        ],
        shuffle_seed=None,
        workers=num_workers,
    )
    metadata_dir = output_dir / ".nv-meta"
    (metadata_dir / "dataset.yaml").write_text(_DATASET_YAML, encoding="utf-8")
    logger.info("Prepared %s", output_dir)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {_DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=_SUPPORTED_SPLITS,
        default=["train", "valA"],
    )
    parser.add_argument("--max-samples-per-shard", type=int, default=1000)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional per-split limit for smoke tests.",
    )
    parser.add_argument(
        "--image-scale",
        type=float,
        default=1.0,
        help=(
            "Upscale factor applied to image width and height before writing "
            "(e.g. 3 stores images at 9x the pixel count). Default: 1.0."
        ),
    )
    parser.add_argument(
        "--image-workers",
        type=int,
        default=16,
        help=(
            "Processes used to rescale images. Only used when --image-scale is "
            "not 1.0. Default: 16."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--download-workers",
        type=int,
        default=8,
        help="Parallel workers for Hugging Face download and dataset preparation.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    counts = prepare_clevr_energon(
        output_dir=args.output_dir,
        splits=args.splits,
        max_samples_per_shard=args.max_samples_per_shard,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
        download_workers=args.download_workers,
        image_scale=args.image_scale,
        image_workers=args.image_workers,
    )
    logger.info("Wrote samples by split: %s", counts)


if __name__ == "__main__":
    main()
