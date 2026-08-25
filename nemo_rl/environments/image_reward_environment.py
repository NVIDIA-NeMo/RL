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
"""Image-native reward environment for flow-GRPO.

Does NOT inherit :class:`nemo_rl.environments.interfaces.EnvironmentInterface`:
the latter's ``step(message_log_batch, metadata)`` is token-centric and the
image-reward signal does not need the LLM message-log path. The diffusion
trainer calls :meth:`ImageRewardEnvironment.score_images` directly.

Plugin contract (:class:`BaseImageReward`):

- ``name``: short identifier used in logged metric keys.
- ``weight``: scalar multiplier applied when aggregating into the total reward.
- ``score(images, prompts, metadata) -> dict[str, Tensor]``: each tensor must
  be 1-D with length equal to ``images.shape[0]``. Plugins may emit multiple
  component scores in the dict; aggregation sums all components.
"""

import functools
import hashlib
import os
import re
from typing import Annotated, Any, Callable, Protocol

import ray
import torch
from pydantic import BaseModel, Field


class ImageRewardPluginSpec(BaseModel, extra="allow"):
    """One entry of the `env.image_reward.plugins` YAML list."""

    name: str
    weight: float = 1.0


class ImageRewardEnvConfig(BaseModel, extra="allow"):
    """Schema for the `env.image_reward` YAML block."""

    plugins: list[ImageRewardPluginSpec]
    num_cpus_per_worker: int = 2
    num_gpus_per_worker: float = 0.0
    # Reward actor replicas per plugin; raise for slow rewards (e.g. OCR).
    num_workers_per_plugin: Annotated[int, Field(ge=1)] = 1


class BaseImageReward(Protocol):
    name: str
    weight: float

    def score(
        self,
        images: torch.Tensor,
        prompts: list[str],
        metadata: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]: ...


class DummyImageReward:
    """Deterministic reward derived from prompt hash and the per-image mean.

    The deterministic property is important for the functional test: replaying the
    same prompts on the same generated tensors must give the same reward, so
    we can detect rollout-level non-determinism without entangling it with
    reward noise.
    """

    name: str = "dummy"
    weight: float = 1.0

    def score(
        self,
        images: torch.Tensor,
        prompts: list[str],
        metadata: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        if images.shape[0] != len(prompts):
            raise ValueError(
                f"images batch={images.shape[0]} but prompts len={len(prompts)}"
            )
        prompt_scores = torch.tensor(
            [
                # Stable hash → [0, 1) via the first 8 hex chars of sha256.
                int(hashlib.sha256(p.encode("utf-8")).hexdigest()[:8], 16) / 0x100000000
                for p in prompts
            ],
            dtype=torch.float32,
        )
        image_means = images.float().mean(dim=tuple(range(1, images.ndim)))
        score = (prompt_scores.to(image_means.device) + image_means.clamp(-1, 1)) / 2.0
        return {"dummy": score}


@ray.remote
class _RewardWorker:  # pragma: no cover
    """Lightweight Ray actor that owns one reward plugin instance."""

    def __init__(self, plugin_factory: Callable[[], BaseImageReward]) -> None:
        self._plugin = plugin_factory()

    def score(
        self,
        images: torch.Tensor,
        prompts: list[str],
        metadata: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        return self._plugin.score(images, prompts, metadata)


class JpegCompressibilityReward:
    """Port of verl-omni `jpeg_compressibility` reward.

    Reward = `- jpeg_size_kb / 500`. More compressible (smaller JPEG) → higher
    reward. Used for cross-stack parity comparison against verl-omni since it
    is verl-omni's only zero-config rule-based image reward.
    """

    name: str = "jpeg_compressibility"
    weight: float = 1.0
    quality: int = 95

    def score(
        self,
        images: torch.Tensor,
        prompts: list[str],
        metadata: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        import io as _io

        from PIL import Image

        arr = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        arr = arr.transpose(0, 2, 3, 1)  # NCHW → NHWC
        sizes_kb = []
        for img_np in arr:
            buf = _io.BytesIO()
            Image.fromarray(img_np).save(buf, format="JPEG", quality=self.quality)
            sizes_kb.append(buf.tell() / 1000.0)
        rewards = -torch.tensor(sizes_kb, dtype=torch.float32) / 500.0
        return {"jpeg_compressibility": rewards}


class PickScoreReward:
    """Prompt-image preference score from PickScore-v1 (CLIP-H fine-tune).

    Scores each image against its own prompt with
    `logit_scale * cosine(text_emb, image_emb)` (raw scale ~16-26; the
    GRPO group normalization makes the absolute scale irrelevant).

    The model is ~4 GB; configure the reward pool with
    `num_gpus_per_worker: 1` to run it on GPU. Images arrive as CPU
    NCHW float [0,1] tensors and scores are returned on CPU, matching
    the trainer's device convention.
    """

    name: str = "pickscore"
    weight: float = 1.0

    model_name: str = "yuvalkirstain/PickScore_v1"
    processor_name: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    batch_size: int = 16

    def __init__(self) -> None:
        # Deferred heavy imports: the factory is pickled into the Ray actor.
        import torch as _torch
        from transformers import AutoModel, AutoProcessor

        self._device = "cuda" if _torch.cuda.is_available() else "cpu"
        self._processor = AutoProcessor.from_pretrained(self.processor_name)
        self._model = AutoModel.from_pretrained(self.model_name).eval().to(self._device)

    def score(
        self,
        images: torch.Tensor,
        prompts: list[str],
        metadata: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        from PIL import Image

        if images.shape[0] != len(prompts):
            raise ValueError(
                f"images batch={images.shape[0]} but prompts len={len(prompts)}"
            )
        arr = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        pil_images = [Image.fromarray(a.transpose(1, 2, 0)) for a in arr]

        scores: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(pil_images), self.batch_size):
                chunk_imgs = pil_images[start : start + self.batch_size]
                chunk_prompts = prompts[start : start + self.batch_size]
                image_inputs = self._processor(
                    images=chunk_imgs, return_tensors="pt"
                ).to(self._device)
                text_inputs = self._processor(
                    text=chunk_prompts,
                    padding=True,
                    truncation=True,
                    max_length=77,
                    return_tensors="pt",
                ).to(self._device)
                # Full forward: `logits_per_image` applies logit_scale to the
                # normalized embeddings and is stable across transformers 4/5
                # (get_*_features returns a ModelOutput in transformers 5.x).
                outputs = self._model(
                    input_ids=text_inputs["input_ids"],
                    attention_mask=text_inputs.get("attention_mask"),
                    pixel_values=image_inputs["pixel_values"],
                )
                scores.append(outputs.logits_per_image.diagonal().float().cpu())
        return {"pickscore": torch.cat(scores)}


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1]


def ocr_edit_distance_score(recognized: str, target: str) -> float:
    """flow_grpo `ocr.py` scoring: substring hit → 1.0, else 1 - dist/len(gt).

    Both strings are lower-cased and space-stripped; the edit distance is
    capped at len(gt) so the score stays in [0, 1]. Empty ground truth → 0.
    """
    target_n = target.lower().replace(" ", "")
    if not target_n:
        return 0.0
    recognized_n = recognized.lower().replace(" ", "")
    if target_n in recognized_n:
        return 1.0
    dist = min(_levenshtein(recognized_n, target_n), len(target_n))
    return 1.0 - dist / len(target_n)


class OcrEditDistanceReward:
    """Rule-based OCR reward (PaddleOCR + edit distance vs metadata ground_truth).

    The primary Flow-GRPO text-rendering task reward. `ocr_fn` is injectable
    for tests; the default engine loads PaddleOCR (en, no angle cls) lazily so
    the module imports without the paddle dependency installed.
    """

    name: str = "ocr"
    weight: float = 1.0

    def __init__(self, ocr_fn: Callable[[Any], str] | None = None) -> None:
        if ocr_fn is not None:
            self._ocr_fn = ocr_fn
            return
        import fcntl
        import os

        # Lazy import: paddleocr is an optional heavy dependency.
        from paddleocr import PaddleOCR  # pyrefly: ignore  # import-error

        # 16+ replicas cold-starting concurrently race on the model download
        # into ~/.paddleocr (observed: half-extracted inference.pdiparams).
        # Serialize construction; after the first replica downloads, the rest
        # construct from the cache.
        lock_dir = os.path.expanduser("~/.paddleocr")
        os.makedirs(lock_dir, exist_ok=True)
        with open(os.path.join(lock_dir, ".nemo-rl-init.lock"), "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            engine = PaddleOCR(use_angle_cls=False, lang="en", show_log=False)

        def run(img_np: Any) -> str:  # HWC uint8
            result = engine.ocr(img_np, cls=False)
            lines = result[0] if result and result[0] else []
            return " ".join(item[1][0] for item in lines)

        self._ocr_fn = run

    def score(
        self,
        images: torch.Tensor,
        prompts: list[str],
        metadata: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        if images.shape[0] != len(metadata):
            raise ValueError(
                f"images batch={images.shape[0]} but metadata len={len(metadata)}"
            )
        for meta in metadata:
            if "ground_truth" not in meta:
                raise KeyError(
                    "ocr reward needs 'ground_truth' in each prompt's metadata "
                    "(produced by tools/export_ocr_prompts.py)"
                )
        arr = (
            (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        ).transpose(0, 2, 3, 1)
        scores = [
            ocr_edit_distance_score(self._ocr_fn(img_np), str(meta["ground_truth"]))
            for img_np, meta in zip(arr, metadata)
        ]
        return {"ocr": torch.tensor(scores, dtype=torch.float32)}


# verl-omni genrm_ocr.py DEFAULT_GRM_PROMPT, verbatim.
GENRM_OCR_PROMPT = "Please output only the text content from the image without any additional descriptions or formatting."


def genrm_ocr_score(text: str, ground_truth: str) -> float:
    """verl-omni `genrm_ocr._levenshtein_score`: GRM transcription vs ground truth.

    Both sides are stripped of all whitespace and lower-cased; a substring hit
    costs 0, otherwise the Levenshtein distance applies, capped at len(gt) so
    hallucinated extra characters cost at most a full miss. Empty ground
    truth: only an empty transcription is a perfect match.
    """
    gt = re.sub(r"\s+", "", ground_truth).lower()
    text = re.sub(r"\s+", "", text).lower()
    dist = 0 if gt in text else _levenshtein(text, gt)
    dist = min(dist, len(gt))
    if len(gt) > 0:
        return 1 - dist / len(gt)
    return 1.0 if len(text) == 0 else 0.0


class GenRmOcrOptions(BaseModel, extra="allow"):
    """Extra keys of a `genrm_ocr` plugins entry.

    Defaults mirror verl-omni genrm_ocr.py (`DEFAULT_GRM_MODEL_PATH` and
    `DEFAULT_SAMPLING_PARAMS`).
    """

    model: str = "~/models/tiny-random/qwen3-vl"
    temperature: float = 0.7
    top_p: float = 0.8
    max_tokens: int = 4096


class GenRmOcrReward:
    """OCR reward from a generative reward model (port of verl-omni `genrm_ocr`).

    Each image is PNG-base64 encoded into an OpenAI-compatible
    `/chat/completions` request against the endpoint from the `GENRM_BASE_URL`
    environment variable (e.g. `http://localhost:30000/v1`); the returned
    transcription is scored against `metadata["ground_truth"]` with
    :func:`genrm_ocr_score`. Pure HTTP client: run it with
    `num_gpus_per_worker: 0` and scale via `num_workers_per_plugin`.
    """

    name: str = "genrm_ocr"
    weight: float = 1.0

    def __init__(self, **options: Any) -> None:
        opts = GenRmOcrOptions.model_validate(options)
        base_url = os.environ.get("GENRM_BASE_URL")
        if not base_url:
            raise ValueError(
                "genrm_ocr requires the GENRM_BASE_URL environment variable, "
                "e.g. http://localhost:30000/v1"
            )
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._model = os.path.expanduser(opts.model)
        self._sampling_params: dict[str, Any] = {
            "temperature": opts.temperature,
            "top_p": opts.top_p,
            "max_tokens": opts.max_tokens,
        }

    def _transcribe(self, image_data_url: str) -> str:
        import requests

        request = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                        {"type": "text", "text": GENRM_OCR_PROMPT},
                    ],
                },
            ],
            "model": self._model,
            **self._sampling_params,
        }
        # Mirrors verl-omni genrm_ocr: unbounded timeout, no retry; a failed
        # request raises and fails the training step loudly.
        response = requests.post(self._url, json=request, timeout=None)
        return response.json()["choices"][0]["message"]["content"]

    def score(
        self,
        images: torch.Tensor,
        prompts: list[str],
        metadata: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        import base64
        import io as _io

        from PIL import Image

        if images.shape[0] != len(metadata):
            raise ValueError(
                f"images batch={images.shape[0]} but metadata len={len(metadata)}"
            )
        for meta in metadata:
            if "ground_truth" not in meta:
                raise KeyError(
                    "genrm_ocr reward needs 'ground_truth' in each prompt's "
                    "metadata (produced by tools/export_ocr_prompts.py)"
                )
        arr = (
            (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        ).transpose(0, 2, 3, 1)
        scores = []
        for img_np, meta in zip(arr, metadata):
            buf = _io.BytesIO()
            Image.fromarray(img_np).save(buf, format="PNG")
            data_url = "data:image/png;base64," + base64.b64encode(
                buf.getvalue()
            ).decode("utf-8")
            transcription = self._transcribe(data_url)
            scores.append(genrm_ocr_score(transcription, str(meta["ground_truth"])))
        return {"genrm_ocr": torch.tensor(scores, dtype=torch.float32)}


_PLUGIN_REGISTRY: dict[str, Callable[..., BaseImageReward]] = {
    "dummy": lambda: DummyImageReward(),
    "jpeg_compressibility": lambda: JpegCompressibilityReward(),
    "pickscore": lambda: PickScoreReward(),
    "ocr": lambda: OcrEditDistanceReward(),
    "genrm_ocr": lambda **options: GenRmOcrReward(**options),
}


def register_image_reward(name: str, factory: Callable[..., BaseImageReward]) -> None:
    """Register a new reward plugin factory."""
    if name in _PLUGIN_REGISTRY:
        raise ValueError(f"Image reward plugin {name!r} is already registered")
    _PLUGIN_REGISTRY[name] = factory


class ImageRewardEnvironment:
    """A Ray-managed pool of `num_workers_per_plugin` reward workers per plugin.

    Each plugin contributes a weighted component to the aggregated reward.
    Components are summed; per-component means are emitted as metrics. Batches
    are sharded contiguously across a plugin's replicas for throughput.
    """

    def __init__(self, config: ImageRewardEnvConfig | dict[str, Any]) -> None:
        # Normalize through the schema so plugin weights and worker resources
        # take their defaults from ImageRewardEnvConfig, not call sites.
        cfg = ImageRewardEnvConfig.model_validate(config)
        self._workers: list[list[ray.actor.ActorHandle]] = []
        self._weights: list[float] = []
        self._names: list[str] = []
        self._replicas_per_plugin = cfg.num_workers_per_plugin
        for spec in cfg.plugins:
            if spec.name not in _PLUGIN_REGISTRY:
                raise KeyError(
                    f"Unknown image reward plugin {spec.name!r}; "
                    f"registered={list(_PLUGIN_REGISTRY)}"
                )
            if spec.name in self._names:
                raise ValueError(
                    f"Duplicate image reward plugin {spec.name!r}; component keys "
                    "would collide and its weight would be double-counted"
                )
            factory = _PLUGIN_REGISTRY[spec.name]
            # Spec keys beyond name/weight (extra="allow") are plugin options,
            # bound as factory kwargs (e.g. genrm_ocr model/sampling params).
            if spec.model_extra:
                factory = functools.partial(factory, **spec.model_extra)
            group = [
                _RewardWorker.options(
                    num_cpus=cfg.num_cpus_per_worker, num_gpus=cfg.num_gpus_per_worker
                ).remote(factory)
                for _ in range(cfg.num_workers_per_plugin)
            ]
            self._workers.append(group)
            self._names.append(spec.name)
            self._weights.append(spec.weight)

    def score_images(
        self,
        images: torch.Tensor,
        prompts: list[str],
        metadata: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        n = images.shape[0]
        futures = []
        for group in self._workers:
            k = len(group)
            # Contiguous shards; empty shards (when n < k) are skipped.
            bounds = [(i * n // k, (i + 1) * n // k) for i in range(k)]
            futures.append(
                [
                    (
                        lo,
                        hi,
                        group[i].score.remote(
                            images[lo:hi], prompts[lo:hi], metadata[lo:hi]
                        ),
                    )
                    for i, (lo, hi) in enumerate(bounds)
                    if hi > lo
                ]
            )
        total = torch.zeros(n, dtype=torch.float32)
        components: dict[str, torch.Tensor] = {}
        # Reassemble each component tensor from its shards by slice position.
        for name, shard_futs in zip(self._names, futures):
            for lo, hi, fut in shard_futs:
                for comp_key, comp_value in ray.get(fut).items():
                    full_key = f"{name}/{comp_key}"
                    if full_key not in components:
                        components[full_key] = torch.zeros(n, dtype=torch.float32)
                    components[full_key][lo:hi] = comp_value
        # Each component belongs to exactly one plugin (distinct names), so
        # every weighted contribution is applied exactly once.
        for name, weight in zip(self._names, self._weights):
            for full_key, comp_value in components.items():
                if full_key.startswith(f"{name}/"):
                    total = total + weight * comp_value
        metrics: dict[str, Any] = {
            f"reward/{k}_mean": float(v.float().mean().item())
            for k, v in components.items()
        }
        metrics["reward/total_mean"] = float(total.mean().item())
        return total, metrics

    def shutdown(self) -> bool:
        for group in self._workers:
            for w in group:
                ray.kill(w)
        self._workers = []
        return True
