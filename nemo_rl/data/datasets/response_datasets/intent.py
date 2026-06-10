# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""IntentDataset: HumanOmniV2 IntentTrain / IntentBench loader for GRPO.

Loads the PhilipC/IntentTrain (training) or PhilipC/IntentBench (validation)
datasets that ship as a JSON manifest plus a ``videos.zip`` archive on
HuggingFace, filters samples to the configured ``problem_type`` allow-list, and
emits OpenAI-style messages whose user content carries both a video reference
and the audio track extracted from that same video. The ``vlm_hf_data_processor``
consumes both modalities jointly with ``use_audio_in_video=True`` so
Qwen2.5-Omni aligns audio and video tokens during inference.
"""

import json
import logging
import os
import zipfile
from typing import Any

import numpy as np
from huggingface_hub import snapshot_download

from nemo_rl.data.datasets.raw_dataset import RawDataset
from nemo_rl.data.datasets.utils import get_huggingface_cache_path

logger = logging.getLogger(__name__)

# Per-problem-type instruction string appended to the question, mirroring
# HumanOmniV2's TYPE_TEMPLATE so the model knows the answer format.
_TYPE_TEMPLATE = {
    "multiple choice": (
        " Please provide only the single option letter (e.g., A, B, C, D, etc.) "
        "within the <answer> </answer> tags."
    ),
    "emer_ov_mc": (
        " Please provide only the single or multiple option letter "
        "(e.g., A for single option or A,E for multi option, etc.) "
        "within the <answer> </answer> tags."
    ),
    "numerical": (
        " Please provide the numerical value (e.g., 42 or 3.14) "
        "within the <answer> </answer> tags."
    ),
    "judge": (" Please answer Yes or No within the <answer> </answer> tags."),
    "free-form": (
        " Please provide your text answer within the <answer> </answer> tags."
    ),
}

# Per-split HF repo + manifest filenames for the HumanOmniV2 IntentTrain /
# IntentBench releases. Each split downloads a videos.zip and one or more JSON
# manifests; manifest entries point at relative paths inside the extracted
# archive.
_SPLIT_CONFIG = {
    "train": {
        "repo_id": "PhilipC/IntentTrain",
        "manifests": ["emer_rewrite.json", "social_iq_v2_rewrite.json"],
        "task_name": "intent-train",
    },
    "validation": {
        "repo_id": "PhilipC/IntentBench",
        "manifests": ["qa.json"],
        "task_name": "intent-bench",
    },
}

_EXTRACTION_SENTINEL = ".intent_videos_extracted"


def _extract_videos_zip_once(snapshot_dir: str) -> str:
    """Idempotently extract ``videos.zip`` inside ``snapshot_dir``.

    Returns the directory the archive was extracted into. A sentinel file is
    written after a successful extraction so subsequent constructions skip
    re-extraction.
    """
    archive = os.path.join(snapshot_dir, "videos.zip")
    if not os.path.isfile(archive):
        raise FileNotFoundError(
            f"videos.zip not found in HuggingFace snapshot at {snapshot_dir}. "
            "Was the dataset downloaded correctly?"
        )

    sentinel = os.path.join(snapshot_dir, _EXTRACTION_SENTINEL)
    if os.path.isfile(sentinel):
        return snapshot_dir

    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(snapshot_dir)

    with open(sentinel, "w", encoding="utf-8") as f:
        f.write("ok\n")
    return snapshot_dir


def _resolve_video_path(snapshot_dir: str, relpath: str) -> str | None:
    """Resolve a manifest's relative video path to an absolute file on disk.

    The IntentTrain/IntentBench archives extract their contents either directly
    under the snapshot directory or under a ``videos/`` subdirectory. Try both
    and return the first path that exists, or ``None`` if neither does.
    """
    candidate = os.path.join(snapshot_dir, relpath)
    if os.path.isfile(candidate):
        return candidate
    candidate = os.path.join(snapshot_dir, "videos", relpath)
    if os.path.isfile(candidate):
        return candidate
    return None


def _load_audio_from_video(video_path: str, sampling_rate: int = 16000) -> np.ndarray:
    """Decode the audio track of a video file as a 1-D float32 array.

    Uses decord's ``AudioReader`` because it's already a project dependency for
    video decoding. Raises ``RuntimeError`` if the video has no decodable audio
    track so callers can drop or skip the sample.
    """
    import decord

    try:
        reader = decord.AudioReader(video_path, sample_rate=sampling_rate, mono=True)
        # Shape: (channels, T). With mono=True channels=1; squeeze to (T,).
        audio = reader[:].asnumpy()
        if audio.ndim > 1:
            audio = audio[0]
        return audio.astype(np.float32)
    except Exception as e:  # decord raises a variety of errors for missing audio
        raise RuntimeError(f"Failed to decode audio from {video_path}: {e}") from e


def _read_manifest(snapshot_dir: str, manifest_filename: str) -> list[dict[str, Any]]:
    manifest_path = os.path.join(snapshot_dir, manifest_filename)
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            f"Manifest {manifest_filename} not found in HF snapshot at "
            f"{snapshot_dir}. Available files: {sorted(os.listdir(snapshot_dir))}"
        )
    with open(manifest_path, "r", encoding="utf-8") as f:
        if manifest_filename.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


class IntentDataset(RawDataset):
    """HumanOmniV2 IntentTrain / IntentBench loader for VLM GRPO.

    Each sample emits a video file path plus a text prompt; the audio track is
    folded in at processor time via ``use_audio_in_video=True`` so the
    Qwen2.5-Omni processor decodes both modalities jointly. Samples whose
    ``problem_type`` is not in ``allowed_problem_types`` are dropped before
    iteration.

    Args:
        split: ``"train"`` (PhilipC/IntentTrain) or ``"validation"``
            (PhilipC/IntentBench).
        allowed_problem_types: List of ``problem_type`` values to retain.
            Defaults to ``["multiple choice"]`` per DEC-2.
        max_samples: Optional cap on the number of samples after filtering.
            Useful for smoke runs.
    """

    def __init__(
        self,
        split: str = "train",
        allowed_problem_types: list[str] | None = None,
        max_samples: int | None = None,
        **kwargs: Any,
    ) -> None:
        if split not in _SPLIT_CONFIG:
            raise ValueError(
                f"Invalid split: {split!r}. Supported: {sorted(_SPLIT_CONFIG.keys())}."
            )
        self.split = split
        self._cfg = _SPLIT_CONFIG[split]
        self.task_name = self._cfg["task_name"]
        self.allowed_problem_types = list(
            allowed_problem_types
            if allowed_problem_types is not None
            else ["multiple choice"]
        )

        self.snapshot_dir = self._download_and_extract()

        records = self._load_records()
        records = self._filter_records(records)
        if max_samples is not None:
            records = records[:max_samples]
        if not records:
            raise ValueError(
                f"IntentDataset({split=}) yielded 0 samples after filtering by "
                f"allowed_problem_types={self.allowed_problem_types}. "
                "Check the manifest contents and filter list."
            )

        from datasets import Dataset

        self.dataset = Dataset.from_list(records)
        self.dataset = self.dataset.add_column(
            "task_name", [self.task_name] * len(self.dataset)
        )
        self.preprocessor = self.format_data
        self.val_dataset = None

    def _download_and_extract(self) -> str:
        """Download the HF dataset snapshot and extract ``videos.zip`` once."""
        repo_id = self._cfg["repo_id"]
        cache_dir = get_huggingface_cache_path(repo_id)
        if not cache_dir:
            cache_dir = snapshot_download(repo_id=repo_id, repo_type="dataset")
        if not cache_dir:
            raise ValueError(f"Cannot download {repo_id}.")
        return _extract_videos_zip_once(cache_dir)

    def _load_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for manifest in self._cfg["manifests"]:
            try:
                manifest_records = _read_manifest(self.snapshot_dir, manifest)
            except FileNotFoundError:
                if len(self._cfg["manifests"]) == 1:
                    raise
                logger.warning(
                    "Manifest %s missing in snapshot %s; skipping",
                    manifest,
                    self.snapshot_dir,
                )
                continue
            records.extend(manifest_records)
        if not records:
            raise ValueError(
                f"No manifest entries loaded for {self._cfg['repo_id']}. "
                f"Expected one of: {self._cfg['manifests']}."
            )
        return records

    def _filter_records(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        allowed = set(self.allowed_problem_types)
        filtered: list[dict[str, Any]] = []
        for record in records:
            problem_type = record.get("problem_type")
            if problem_type not in allowed:
                continue
            data_type = record.get("data_type", "video")
            if data_type != "video":
                # Mixed modalities (e.g. image-only entries from
                # Video-R1_rewrite.json) are out of scope; the recipe is
                # video-first per DEC-1 / DEC-2.
                continue
            relpath = record.get("video") or record.get("path")
            if not isinstance(relpath, str):
                continue
            local_path = _resolve_video_path(self.snapshot_dir, relpath)
            if local_path is None:
                logger.warning(
                    "Skipping manifest entry: video not found for relpath=%s",
                    relpath,
                )
                continue
            filtered.append(
                {
                    "problem": record.get("problem", ""),
                    "problem_type": problem_type,
                    "answer": record.get("answer", ""),
                    "video_path": local_path,
                }
            )
        return filtered

    def format_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """Format a manifest record into NeMo-RL OpenAI-style messages.

        Each yielded sample carries the video file path AND a numpy audio
        array decoded from the same file at 16 kHz mono. Downstream the VLM
        processor invokes Qwen2.5-Omni with ``use_audio_in_video=True`` so the
        two streams are aligned.
        """
        instruction = _TYPE_TEMPLATE.get(data["problem_type"], "")
        prompt_text = f"{data['problem']}{instruction}"
        audio_array = _load_audio_from_video(data["video_path"])
        user_content = [
            {"type": "video", "video": data["video_path"]},
            {"type": "audio", "audio": audio_array},
            {"type": "text", "text": prompt_text},
        ]
        return {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": str(data["answer"])},
            ],
            "task_name": self.task_name,
        }


class IntentTrainDataset(IntentDataset):
    """Convenience wrapper that pins ``split="train"`` for IntentTrain."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("split", "train")
        super().__init__(**kwargs)


class IntentBenchDataset(IntentDataset):
    """Convenience wrapper that pins ``split="validation"`` for IntentBench."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("split", "validation")
        super().__init__(**kwargs)
