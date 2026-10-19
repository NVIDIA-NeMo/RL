# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SA-V "tracks"-schema verifier: point / box / text-referred tracking.

A sibling of sav_tracking for the flat {"tracks":[...]} answer schemas used by
the sav_{pt,box,ref}_rl jsonls (which pair with SFT data trained on the same
prompts — the wording and answer shapes here must stay byte-compatible with
those files). One server scores all three task types; each record declares its
type via `task`:

  task="pt"  — reference points given in frame 0; answer entries
               {"point":i,"frame":k,"xy":[x,y],"visible":true|false} (xy null/omitted
               when invisible). GT-visible frames score 1.0 iff the predicted point
               falls inside the object's segmentation mask (binary, no distance).
  task="box" — reference boxes given in frame 0; answer entries
               {"box":i,"frame":k,"bbox":[x1,y1,x2,y2],"visible":true|false}
               (bbox null when invisible). GT-visible frames score by IoU.
  task="ref" — objects referred by a text query; answer entries
               {"id":i,"frame":k,"xy":[x,y]} where OMITTING a (id, frame) entry is
               the invisibility claim and ids are model-chosen: predicted ids are
               assigned one-to-one to GT objects to maximize total score, then each
               GT-visible frame scores 1.0 iff the assigned point is inside the mask.

Ground truth lives in `objects`: one entry per tracked object with per-frame
targets; visible targets carry the object's segmentation mask as COCO
compressed RLE ({"size":[h,w],"counts":"..."}). Mask membership is tested with
a pure-Python RLE walk (no pycocotools dependency): coordinates come in on the
0-1000 grid and are mapped to the mask's own pixel space.

Reward: per-frame credit averaged per object, then across objects (objects
weighted equally). GT-invisible frames pay absence_score for a correct
invisibility claim and 0 for any location. Extra / duplicate / invalid entries
scale the reward down by targets/(targets+extras). Format failures score 0.
"""

import itertools
import json
import math
import re
from typing import Any, Literal, Optional

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.reward_profile import compute_pass_majority_metrics, highest_k_metrics


ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
TaskType = Literal["pt", "box", "ref"]
GRID = 1000.0


# ---------------------------------------------------------------------------
# COCO RLE (compressed string or raw counts list) — membership only.
# ---------------------------------------------------------------------------


def _rle_counts_to_ints(counts: Any) -> Optional[list[int]]:
    """Decode COCO RLE counts to a list of run lengths.

    Accepts the raw uncompressed form (list of ints) or the compressed LEB128-style
    string produced by pycocotools (each value is a delta from two runs back,
    except the first two).
    """
    if isinstance(counts, list):
        out = []
        for value in counts:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                return None
            out.append(value)
        return out
    if not isinstance(counts, str):
        return None
    runs: list[int] = []
    pos = 0
    n = len(counts)
    while pos < n:
        value = 0
        shift = 0
        more = True
        while more:
            if pos >= n:
                return None
            c = ord(counts[pos]) - 48
            pos += 1
            value |= (c & 0x1F) << shift
            more = bool(c & 0x20)
            shift += 5
            if not more and (c & 0x10):
                value |= -1 << shift
        if len(runs) > 2:
            value += runs[-2]
        if value < 0:
            return None
        runs.append(value)
    return runs


def _point_in_rle(mask: dict[str, Any], x_grid: float, y_grid: float) -> Optional[bool]:
    """Is grid1000 point (x, y) inside the mask? None if the RLE is malformed.

    COCO RLE is column-major: linear index = x_px * h + y_px. The runs alternate
    background/foreground starting with background, so the point is foreground
    iff its cumulative run index is odd.
    """
    size = mask.get("size")
    if not isinstance(size, list) or len(size) != 2:
        return None
    try:
        h, w = int(size[0]), int(size[1])
    except (TypeError, ValueError):
        return None
    if h <= 0 or w <= 0:
        return None
    runs = _rle_counts_to_ints(mask.get("counts"))
    if runs is None:
        return None
    x_px = min(max(int(x_grid / GRID * w), 0), w - 1)
    y_px = min(max(int(y_grid / GRID * h), 0), h - 1)
    index = x_px * h + y_px
    cumulative = 0
    for run_index, run in enumerate(runs):
        cumulative += run
        if index < cumulative:
            return run_index % 2 == 1
    return False


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class TrackTarget(BaseModel):
    """Ground truth for one (object, frame) cell."""

    frame: int
    visible: bool = True
    # box task: GT bbox on the 0-1000 grid (visible targets only).
    bbox: Optional[list[float]] = Field(default=None, min_length=4, max_length=4)
    # pt/ref tasks: GT point (metrics only — the reward is mask membership).
    xy: Optional[list[float]] = Field(default=None, min_length=2, max_length=2)
    # pt/ref tasks: COCO RLE mask ({"size":[h,w],"counts":...}) for visible targets.
    mask: Optional[dict[str, Any]] = None


class TrackObject(BaseModel):
    """Ground truth for one tracked object/point."""

    id: int
    reference_bbox: Optional[list[float]] = None
    reference_xy: Optional[list[float]] = None
    targets: list[TrackTarget]


class ParsedEntry(BaseModel):
    object_id: Optional[int] = None
    frame: Optional[int] = None
    bbox: Optional[list[float]] = None
    xy: Optional[list[float]] = None
    # False means the model explicitly (pt/box) or by omission (ref) claimed invisible.
    visible: bool = True
    valid: bool = False
    error: Optional[str] = None


class FrameScore(BaseModel):
    frame: int
    score: float
    iou: float = 0.0
    in_mask: Optional[bool] = None
    matched: bool = False
    gt_visible: bool = True
    predicted_visible: Optional[bool] = None


class ObjectScore(BaseModel):
    id: int
    matched: bool
    assigned_prediction_id: Optional[int] = None
    frame_scores: list[FrameScore]
    mean_score: float
    visibility_correct: int
    visibility_total: int
    missing_frames: list[int]


class SavTracksResourcesServerConfig(BaseResourcesServerConfig):
    # Credit for correctly declaring an object not visible in a frame. A false
    # invisibility claim on a visible object always scores 0, so claiming
    # invisibility is never a cheap escape from localization.
    absence_score: float = Field(default=1.0, ge=0.0, le=1.0)
    iou_threshold: float = 0.5


class SavTracksRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    task: TaskType = "box"
    reference_frame: int = 0
    objects: list[TrackObject] = []
    verifier: Optional[str] = None
    source_id: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None

    require_answer_tags: bool = True
    penalize_extra_predictions: bool = True
    absence_score: float = Field(default=1.0, ge=0.0, le=1.0)
    iou_threshold: float = 0.5


class SavTracksVerifyRequest(SavTracksRunRequest, BaseVerifyRequest):
    pass


class SavTracksVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    source_id: Optional[str] = None
    extracted_answer: Optional[str]
    format_ok: bool
    object_scores: list[ObjectScore] = []
    mean_iou: float = 0.0
    in_mask_rate: Optional[float] = None
    visibility_accuracy: Optional[float] = None
    missing_entry_count: int = 0
    extra_entry_count: int = 0
    invalid_entry_count: int = 0


# ---------------------------------------------------------------------------
# Answer extraction / parsing
# ---------------------------------------------------------------------------


def _extract_last_assistant_text(body: BaseVerifyRequest) -> str:
    texts: list[str] = []
    for output in body.response.output:
        if getattr(output, "type", None) != "message" or getattr(output, "role", None) != "assistant":
            continue
        content = getattr(output, "content", None)
        if isinstance(content, list):
            for item in content:
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    texts.append(text)
        elif isinstance(content, str):
            texts.append(content)
    return "\n".join(texts).strip()


def _extract_answer_json(text: str, require_answer_tags: bool) -> tuple[Optional[str], bool]:
    matches = ANSWER_PATTERN.findall(text)
    if matches:
        return matches[-1].strip(), True
    if require_answer_tags:
        return None, False
    stripped = text.strip()
    return stripped if stripped else None, True


def _as_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            parsed = float(value.strip())
        except ValueError:
            return None
        if parsed.is_integer():
            return int(parsed)
    return None


def _as_float_list(value: Any, length: int) -> Optional[list[float]]:
    if not isinstance(value, list) or len(value) != length:
        return None
    out: list[float] = []
    for item in value:
        if isinstance(item, bool):
            return None
        try:
            parsed = float(item)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(parsed):
            return None
        out.append(parsed)
    return out


def _bbox_is_valid_xyxy(box: list[float]) -> bool:
    x1, y1, x2, y2 = box
    return 0.0 <= x1 and 0.0 <= y1 and x2 <= GRID and y2 <= GRID and x1 < x2 and y1 < y2


def _xy_is_valid(xy: list[float]) -> bool:
    return 0.0 <= xy[0] <= GRID and 0.0 <= xy[1] <= GRID


def _parse_track_entry(raw: Any, task: TaskType) -> ParsedEntry:
    if not isinstance(raw, dict):
        return ParsedEntry(error="track entry is not an object")

    id_key = {"pt": "point", "box": "box", "ref": "id"}[task]
    object_id = _as_int(raw.get(id_key))
    frame = _as_int(raw.get("frame"))
    if object_id is None:
        return ParsedEntry(frame=frame, error=f'"{id_key}" is missing or not an integer')
    if frame is None:
        return ParsedEntry(object_id=object_id, error='"frame" is missing or not an integer')

    if raw.get("visible") is False:
        # pt/box always claim visibility explicitly; ref templates usually claim
        # invisibility by omission but one variant emits explicit entries too.
        # An invisible claim may carry a null (or omitted) location, but not a real one.
        loc = raw.get("bbox") if task == "box" else raw.get("xy")
        if loc is not None:
            return ParsedEntry(
                object_id=object_id,
                frame=frame,
                visible=False,
                error='a "visible":false entry must not include a location',
            )
        return ParsedEntry(object_id=object_id, frame=frame, visible=False, valid=True)

    if task == "box":
        bbox = _as_float_list(raw.get("bbox"), 4)
        if bbox is None or not _bbox_is_valid_xyxy(bbox):
            return ParsedEntry(object_id=object_id, frame=frame, error="bbox is missing, out of range, or not xyxy")
        return ParsedEntry(object_id=object_id, frame=frame, bbox=bbox, valid=True)

    xy = _as_float_list(raw.get("xy"), 2)
    if xy is None or not _xy_is_valid(xy):
        return ParsedEntry(object_id=object_id, frame=frame, error="xy is missing or out of range")
    return ParsedEntry(object_id=object_id, frame=frame, xy=xy, valid=True)


def _parse_tracks(answer_json: str, task: TaskType) -> tuple[Optional[str], list[ParsedEntry], bool]:
    try:
        parsed = json.loads(answer_json)
    except json.JSONDecodeError:
        return None, [], False

    extracted_answer = json.dumps(parsed, sort_keys=True, separators=(",", ":"))
    if not isinstance(parsed, dict) or not isinstance(parsed.get("tracks"), list):
        return extracted_answer, [], False

    entries = [_parse_track_entry(raw, task) for raw in parsed["tracks"]]
    format_ok = all(entry.valid for entry in entries)
    return extracted_answer, entries, format_ok


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _iou(pred: list[float], target: list[float]) -> float:
    ix1 = max(pred[0], target[0])
    iy1 = max(pred[1], target[1])
    ix2 = min(pred[2], target[2])
    iy2 = min(pred[3], target[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    pred_area = max(0.0, pred[2] - pred[0]) * max(0.0, pred[3] - pred[1])
    target_area = max(0.0, target[2] - target[0]) * max(0.0, target[3] - target[1])
    union = pred_area + target_area - intersection
    return intersection / union if union > 0 else 0.0


def _localization_score(
    task: TaskType, entry: ParsedEntry, target: TrackTarget
) -> tuple[float, float, Optional[bool]]:
    """Score a visible prediction against a visible target: (score, iou, in_mask)."""
    if task == "box":
        if entry.bbox is None or target.bbox is None:
            return 0.0, 0.0, None
        iou = _iou(entry.bbox, target.bbox)
        return iou, iou, None
    # pt / ref: binary mask membership.
    if entry.xy is None or target.mask is None:
        return 0.0, 0.0, None
    inside = _point_in_rle(target.mask, entry.xy[0], entry.xy[1])
    if inside is None:
        return 0.0, 0.0, None
    return (1.0 if inside else 0.0), 0.0, inside


def _score_object(
    task: TaskType,
    gt: TrackObject,
    entries_by_frame: dict[int, list[ParsedEntry]],
    absence_score: float,
    omission_is_invisible: bool,
    assigned_prediction_id: Optional[int] = None,
    matched: bool = True,
) -> tuple[ObjectScore, int]:
    """Score one object's entries against its targets.

    Per-frame credit: GT visible -> localization score (a false invisibility claim
    scores 0); GT invisible -> absence_score for a correct invisibility claim, 0 for
    any location. With omission_is_invisible (ref task), a missing entry counts as
    the invisibility claim instead of an unconditional 0.
    Returns the ObjectScore and the object's extra-entry count (duplicates +
    off-target frames).
    """
    frame_scores: list[FrameScore] = []
    missing_frames: list[int] = []
    visibility_correct = 0
    duplicate_count = 0
    expected_frames = {target.frame for target in gt.targets}

    for target in gt.targets:
        gt_visible = target.visible
        candidates = entries_by_frame.get(target.frame, []) if matched else []
        if not candidates:
            if omission_is_invisible and matched:
                score = absence_score if not gt_visible else 0.0
                if not gt_visible:
                    visibility_correct += 1
                frame_scores.append(
                    FrameScore(
                        frame=target.frame,
                        score=score,
                        matched=True,
                        gt_visible=gt_visible,
                        predicted_visible=False,
                    )
                )
            else:
                missing_frames.append(target.frame)
                frame_scores.append(FrameScore(frame=target.frame, score=0.0, gt_visible=gt_visible))
            continue

        duplicate_count += max(0, len(candidates) - 1)
        scored: list[tuple[float, float, Optional[bool], bool]] = []
        for candidate in candidates:
            if not gt_visible:
                scored.append((absence_score if not candidate.visible else 0.0, 0.0, None, candidate.visible))
            elif not candidate.visible:
                scored.append((0.0, 0.0, None, False))
            else:
                score, iou, in_mask = _localization_score(task, candidate, target)
                scored.append((score, iou, in_mask, True))
        score, iou, in_mask, predicted_visible = max(scored, key=lambda item: item[0])
        if predicted_visible == gt_visible:
            visibility_correct += 1
        frame_scores.append(
            FrameScore(
                frame=target.frame,
                score=score,
                iou=iou,
                in_mask=in_mask,
                matched=True,
                gt_visible=gt_visible,
                predicted_visible=predicted_visible,
            )
        )

    extra_frame_count = sum(1 for frame in entries_by_frame if frame not in expected_frames) if matched else 0

    object_score = ObjectScore(
        id=gt.id,
        matched=matched,
        assigned_prediction_id=assigned_prediction_id,
        frame_scores=frame_scores,
        mean_score=sum(fs.score for fs in frame_scores) / len(frame_scores) if frame_scores else 0.0,
        visibility_correct=visibility_correct,
        visibility_total=len(gt.targets),
        missing_frames=missing_frames,
    )
    return object_score, duplicate_count + extra_frame_count


def _group_entries(entries: list[ParsedEntry]) -> dict[int, dict[int, list[ParsedEntry]]]:
    """valid entries grouped as {object_id: {frame: [entries]}}."""
    grouped: dict[int, dict[int, list[ParsedEntry]]] = {}
    for entry in entries:
        if entry.valid and entry.object_id is not None and entry.frame is not None:
            grouped.setdefault(entry.object_id, {}).setdefault(int(entry.frame), []).append(entry)
    return grouped


def _assign_ref_ids(
    gt_objects: list[TrackObject],
    grouped: dict[int, dict[int, list[ParsedEntry]]],
    absence_score: float,
) -> dict[int, Optional[int]]:
    """One-to-one assignment of predicted ids to GT objects maximizing total score.

    Exhaustive over permutations when small (ref records have few objects); greedy
    fallback beyond that.
    """
    pred_ids = sorted(grouped)
    if not pred_ids:
        return {gt.id: None for gt in gt_objects}

    pair_scores: dict[tuple[int, int], float] = {}
    for gt in gt_objects:
        for pred_id in pred_ids:
            object_score, _ = _score_object(
                "ref", gt, grouped[pred_id], absence_score, omission_is_invisible=True, matched=True
            )
            pair_scores[(gt.id, pred_id)] = object_score.mean_score

    assignment: dict[int, Optional[int]] = {gt.id: None for gt in gt_objects}
    if len(gt_objects) <= 6 and len(pred_ids) <= 6:
        best_total = -1.0
        slots = min(len(gt_objects), len(pred_ids))
        for gt_subset in itertools.permutations([gt.id for gt in gt_objects], slots):
            for pred_subset in itertools.permutations(pred_ids, slots):
                total = sum(pair_scores[(g, p)] for g, p in zip(gt_subset, pred_subset))
                if total > best_total:
                    best_total = total
                    assignment = {gt.id: None for gt in gt_objects}
                    assignment.update(dict(zip(gt_subset, pred_subset)))
        return assignment

    remaining_preds = set(pred_ids)
    for gt_id, pred_id in sorted(pair_scores, key=lambda key: -pair_scores[key]):
        if assignment.get(gt_id) is None and pred_id in remaining_preds:
            assignment[gt_id] = pred_id
            remaining_preds.discard(pred_id)
    return assignment


def _score_tracks(
    body: SavTracksRunRequest,
    entries: list[ParsedEntry],
    absence_score: float,
) -> dict[str, Any]:
    task: TaskType = body.task
    gt_objects = body.objects or []
    grouped = _group_entries(entries)
    invalid_entry_count = sum(1 for entry in entries if not entry.valid)

    if task == "ref":
        assignment = _assign_ref_ids(gt_objects, grouped, absence_score)
        assigned_preds = {pred_id for pred_id in assignment.values() if pred_id is not None}
        extra_object_entries = sum(
            sum(len(frame_entries) for frame_entries in grouped[pred_id].values())
            for pred_id in grouped
            if pred_id not in assigned_preds
        )
    else:
        assignment = {gt.id: (gt.id if gt.id in grouped else None) for gt in gt_objects}
        expected_ids = {gt.id for gt in gt_objects}
        extra_object_entries = sum(
            sum(len(frame_entries) for frame_entries in grouped[pred_id].values())
            for pred_id in grouped
            if pred_id not in expected_ids
        )

    extra_count = extra_object_entries + invalid_entry_count
    object_scores: list[ObjectScore] = []
    for gt in gt_objects:
        pred_id = assignment.get(gt.id)
        object_score, extras = _score_object(
            task,
            gt,
            grouped.get(pred_id, {}) if pred_id is not None else {},
            absence_score,
            omission_is_invisible=(task == "ref"),
            assigned_prediction_id=pred_id if task == "ref" else None,
            matched=pred_id is not None,
        )
        object_scores.append(object_score)
        extra_count += extras

    total_targets = sum(score.visibility_total for score in object_scores)
    if total_targets == 0:
        return {
            "reward": 1.0 if extra_count == 0 else 0.0,
            "object_scores": object_scores,
            "mean_iou": 0.0,
            "in_mask_rate": None,
            "visibility_accuracy": None,
            "missing_entry_count": 0,
            "extra_entry_count": extra_count,
            "invalid_entry_count": invalid_entry_count,
        }

    # Objects are weighted equally regardless of how many frames each has.
    reward = sum(score.mean_score for score in object_scores) / len(object_scores)
    if body.penalize_extra_predictions and extra_count > 0:
        reward *= total_targets / (total_targets + extra_count)

    visible_frame_scores = [fs for score in object_scores for fs in score.frame_scores if fs.gt_visible]
    mean_iou = sum(fs.iou for fs in visible_frame_scores) / len(visible_frame_scores) if visible_frame_scores else 0.0
    mask_checked = [fs for fs in visible_frame_scores if fs.in_mask is not None]
    in_mask_rate = sum(1 for fs in mask_checked if fs.in_mask) / len(mask_checked) if mask_checked else None
    visibility_accuracy = sum(score.visibility_correct for score in object_scores) / total_targets
    missing_entry_count = sum(len(score.missing_frames) for score in object_scores)

    return {
        "reward": reward,
        "object_scores": object_scores,
        "mean_iou": mean_iou,
        "in_mask_rate": in_mask_rate,
        "visibility_accuracy": visibility_accuracy,
        "missing_entry_count": missing_entry_count,
        "extra_entry_count": extra_count,
        "invalid_entry_count": invalid_entry_count,
    }


class SavTracksResourcesServer(SimpleResourcesServer):
    config: SavTracksResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        return app

    def compute_metrics(self, tasks):
        def score_fn(result):
            scores = {
                "reward": result.get("reward", 0.0),
                "format_ok": result.get("format_ok", False),
            }
            if result.get("visibility_accuracy") is not None:
                scores["visibility_acc"] = result["visibility_accuracy"]
            if result.get("in_mask_rate") is not None:
                scores["in_mask_rate"] = result["in_mask_rate"]
            return scores

        return compute_pass_majority_metrics(
            tasks,
            score_fn=score_fn,
            answer_key="extracted_answer",
        )[0]

    def get_key_metrics(self, agent_metrics):
        key = {}
        for metric_name in ("mean/reward", "pass@1[avg-of-1]/format_ok"):
            if metric_name in agent_metrics:
                key[metric_name] = agent_metrics[metric_name]
        key.update(highest_k_metrics(agent_metrics, "pass@1[avg-of-{k}]", score_names=["reward"]))
        return key

    def _effective_absence_score(self, body: SavTracksVerifyRequest) -> float:
        return body.absence_score if "absence_score" in body.model_fields_set else self.config.absence_score

    async def verify(self, body: SavTracksVerifyRequest) -> SavTracksVerifyResponse:
        absence_score = self._effective_absence_score(body)
        text = _extract_last_assistant_text(body)
        answer_json, has_required_wrapper = _extract_answer_json(text, body.require_answer_tags)

        extracted_answer = None
        entries: list[ParsedEntry] = []
        format_ok = False
        if answer_json is not None:
            extracted_answer, entries, format_ok = _parse_tracks(answer_json, body.task)
            format_ok = format_ok and has_required_wrapper

        result = _score_tracks(body, entries, absence_score)
        reward = result.pop("reward")
        if not format_ok:
            reward = 0.0

        response_payload = body.model_dump()
        response_payload["absence_score"] = absence_score
        return SavTracksVerifyResponse(
            **response_payload,
            reward=reward,
            extracted_answer=extracted_answer,
            format_ok=format_ok,
            **result,
        )


if __name__ == "__main__":
    SavTracksResourcesServer.run_webserver()
