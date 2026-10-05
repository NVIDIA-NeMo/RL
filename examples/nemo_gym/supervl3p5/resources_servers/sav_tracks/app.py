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
"""Verifier for Segment Anything Video point, box, and referring tracks."""

import json
import math
import re
from functools import lru_cache
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)


ANSWER_PATTERN = re.compile(r"<answer>\s*(\{.*?\})\s*</answer>", re.DOTALL)


class SavTracksResourcesServerConfig(BaseResourcesServerConfig):
    pass


class CocoRLE(BaseModel):
    size: tuple[int, int]
    counts: str | list[int]


class SavTarget(BaseModel):
    frame: int
    visible: bool
    xy: Optional[tuple[float, float]] = None
    bbox: Optional[tuple[float, float, float, float]] = None
    mask: Optional[CocoRLE] = None


class SavObjectTrack(BaseModel):
    id: int
    targets: list[SavTarget]
    reference_frame: Optional[int] = None
    reference_xy: Optional[tuple[float, float]] = None
    reference_bbox: Optional[tuple[float, float, float, float]] = None


class SavTracksVerifyRequest(BaseVerifyRequest):
    task: Literal["pt", "box", "ref"]
    coordinate_space: Literal["grid1000"]
    objects: list[SavObjectTrack]


class SavTracksVerifyResponse(BaseVerifyResponse):
    format_score: float = 0.0
    accuracy: float = 0.0
    resolved: bool = False
    parsed_tracks: list[dict[str, Any]] = Field(default_factory=list)


def _extract_last_assistant_text(body: BaseVerifyRequest) -> str:
    texts: list[str] = []
    for output in body.response.output:
        if (
            getattr(output, "type", None) != "message"
            or getattr(output, "role", None) != "assistant"
        ):
            continue
        content = getattr(output, "content", None)
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            texts.extend(
                part.text
                for part in content
                if isinstance(getattr(part, "text", None), str)
            )
    return "\n".join(texts).strip()


def _parse_tracks(text: str) -> tuple[list[dict[str, Any]], bool]:
    matches = list(ANSWER_PATTERN.finditer(text))
    if len(matches) != 1:
        return [], False
    try:
        payload = json.loads(matches[0].group(1))
    except json.JSONDecodeError:
        return [], False
    if not isinstance(payload, dict) or set(payload) != {"tracks"}:
        return [], False
    tracks = payload["tracks"]
    if not isinstance(tracks, list) or not all(
        isinstance(track, dict) for track in tracks
    ):
        return [], False
    return tracks, not (text[: matches[0].start()] + text[matches[0].end() :]).strip()


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _grid_point(value: Any) -> Optional[tuple[float, float]]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    if not all(_is_number(item) and 0 <= float(item) <= 1000 for item in value):
        return None
    return float(value[0]), float(value[1])


def _grid_box(value: Any) -> Optional[tuple[float, float, float, float]]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    if not all(_is_number(item) and 0 <= float(item) <= 1000 for item in value):
        return None
    box = tuple(float(item) for item in value)
    if box[0] > box[2] or box[1] > box[3]:
        return None
    return box


@lru_cache(maxsize=1024)
def _decode_compressed_rle(counts: str) -> tuple[int, ...]:
    """Decode COCO's compressed run lengths without requiring pycocotools."""
    decoded: list[int] = []
    position = 0
    while position < len(counts):
        value = 0
        shift = 0
        more = True
        while more:
            if position >= len(counts):
                raise ValueError("truncated compressed COCO RLE")
            char = ord(counts[position]) - 48
            position += 1
            if char < 0 or char > 0x3F:
                raise ValueError("invalid compressed COCO RLE character")
            value |= (char & 0x1F) << shift
            more = bool(char & 0x20)
            shift += 5
            if not more and char & 0x10:
                value |= -1 << shift
        if len(decoded) > 2:
            value += decoded[-2]
        if value < 0:
            raise ValueError("negative COCO RLE run")
        decoded.append(value)
    return tuple(decoded)


def _point_in_mask(mask: CocoRLE, point: tuple[float, float]) -> bool:
    height, width = mask.size
    if height <= 0 or width <= 0:
        return False
    pixel_x = min(width - 1, int(point[0] * width / 1000.0))
    pixel_y = min(height - 1, int(point[1] * height / 1000.0))
    flat_index = pixel_y + pixel_x * height
    try:
        runs = (
            _decode_compressed_rle(mask.counts)
            if isinstance(mask.counts, str)
            else tuple(mask.counts)
        )
    except ValueError:
        return False
    foreground = False
    for run in runs:
        if not isinstance(run, int) or isinstance(run, bool) or run < 0:
            return False
        if flat_index < run:
            return foreground
        flat_index -= run
        foreground = not foreground
    return False


def _box_iou(
    left: tuple[float, float, float, float],
    *,
    right: tuple[float, float, float, float],
) -> float:
    intersection_width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _positive_int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _nonnegative_int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _maximum_weight_assignment(weights: list[list[float]]) -> list[tuple[int, int]]:
    """Return a maximum-weight row/column matching in polynomial time.

    Dummy zero-weight columns let any row remain unmatched. This rectangular
    Hungarian implementation is O(rows² * (real_columns + rows)), avoiding the
    previous exponential bitmask search over model-controlled prediction IDs.
    """
    row_count = len(weights)
    if row_count == 0:
        return []
    real_column_count = len(weights[0]) if weights[0] else 0
    column_count = real_column_count + row_count
    max_weight = max((max(row, default=0.0) for row in weights), default=0.0)

    # 1-indexed Hungarian minimization. Maximizing weight is equivalent to
    # minimizing max_weight - weight; dummy columns carry zero weight.
    row_potential = [0.0] * (row_count + 1)
    column_potential = [0.0] * (column_count + 1)
    matched_row = [0] * (column_count + 1)
    previous_column = [0] * (column_count + 1)
    for row_index in range(1, row_count + 1):
        matched_row[0] = row_index
        min_cost = [math.inf] * (column_count + 1)
        used = [False] * (column_count + 1)
        column = 0
        while True:
            used[column] = True
            current_row = matched_row[column]
            delta = math.inf
            next_column = 0
            for candidate_column in range(1, column_count + 1):
                if used[candidate_column]:
                    continue
                weight = (
                    weights[current_row - 1][candidate_column - 1]
                    if candidate_column <= real_column_count
                    else 0.0
                )
                reduced_cost = (
                    max_weight
                    - weight
                    - row_potential[current_row]
                    - column_potential[candidate_column]
                )
                if reduced_cost < min_cost[candidate_column]:
                    min_cost[candidate_column] = reduced_cost
                    previous_column[candidate_column] = column
                if min_cost[candidate_column] < delta:
                    delta = min_cost[candidate_column]
                    next_column = candidate_column
            for candidate_column in range(column_count + 1):
                if used[candidate_column]:
                    row_potential[matched_row[candidate_column]] += delta
                    column_potential[candidate_column] -= delta
                else:
                    min_cost[candidate_column] -= delta
            column = next_column
            if matched_row[column] == 0:
                break
        while True:
            prior = previous_column[column]
            matched_row[column] = matched_row[prior]
            column = prior
            if column == 0:
                break

    return [
        (row_index - 1, column_index - 1)
        for column_index, row_index in enumerate(matched_row[1:], start=1)
        if row_index != 0 and column_index <= real_column_count
    ]


def _score_keyed_tracks(
    task: Literal["pt", "box"],
    objects: list[SavObjectTrack],
    tracks: list[dict[str, Any]],
) -> tuple[float, bool]:
    id_key = "point" if task == "pt" else "box"
    predictions: dict[tuple[int, int], dict[str, Any]] = {}
    schema_valid = True
    for track in tracks:
        object_id = _positive_int(track.get(id_key))
        frame = _nonnegative_int(track.get("frame"))
        visible = track.get("visible")
        coordinate = (
            _grid_point(track.get("xy"))
            if task == "pt"
            else _grid_box(track.get("bbox"))
        )
        if object_id is None or frame is None or not isinstance(visible, bool):
            schema_valid = False
            continue
        if visible and coordinate is None:
            schema_valid = False
            continue
        key = (object_id, frame)
        if key in predictions:
            schema_valid = False
            continue
        predictions[key] = track

    total = 0
    score = 0.0
    expected_keys: set[tuple[int, int]] = set()
    for obj in objects:
        for target in obj.targets:
            total += 1
            key = (obj.id, target.frame)
            expected_keys.add(key)
            prediction = predictions.get(key)
            if prediction is None or prediction.get("visible") is not target.visible:
                continue
            if not target.visible:
                score += 1.0
            elif task == "pt":
                point = _grid_point(prediction.get("xy"))
                if (
                    point is not None
                    and target.mask is not None
                    and _point_in_mask(target.mask, point)
                ):
                    score += 1.0
            else:
                predicted_box = _grid_box(prediction.get("bbox"))
                if predicted_box is not None and target.bbox is not None:
                    score += _box_iou(predicted_box, right=target.bbox)

    extras = len(set(predictions) - expected_keys)
    denominator = total + extras
    return (score / denominator if denominator else 0.0), schema_valid


def _score_ref_tracks(
    objects: list[SavObjectTrack], tracks: list[dict[str, Any]]
) -> tuple[float, bool]:
    predictions: dict[int, dict[int, tuple[float, float]]] = {}
    schema_valid = True
    for track in tracks:
        predicted_id = _positive_int(track.get("id"))
        frame = _nonnegative_int(track.get("frame"))
        point = _grid_point(track.get("xy"))
        if predicted_id is None or frame is None or point is None:
            schema_valid = False
            continue
        by_frame = predictions.setdefault(predicted_id, {})
        if frame in by_frame:
            schema_valid = False
            continue
        by_frame[frame] = point

    predicted_ids = list(predictions)
    pair_scores: list[list[tuple[float, int]]] = []
    for obj in objects:
        row: list[tuple[float, int]] = []
        for predicted_id in predicted_ids:
            score = 0.0
            used = 0
            for target in obj.targets:
                point = predictions[predicted_id].get(target.frame)
                if point is not None:
                    used += 1
                if not target.visible:
                    score += float(point is None)
                elif (
                    point is not None
                    and target.mask is not None
                    and _point_in_mask(target.mask, point)
                ):
                    score += 1.0
            row.append((score, used))
        pair_scores.append(row)

    total_targets = sum(len(obj.targets) for obj in objects)
    total_predictions = sum(len(by_frame) for by_frame in predictions.values())
    # Preserve the former tuple comparison: maximize score first, then the
    # number of consumed predictions. The multiplier makes one score point
    # dominate every possible used-prediction tie-break.
    tie_break_multiplier = total_predictions + 1
    assignment = _maximum_weight_assignment(
        [
            [
                pair_score * tie_break_multiplier + pair_used
                for pair_score, pair_used in row
            ]
            for row in pair_scores
        ]
    )
    score = sum(pair_scores[row][column][0] for row, column in assignment)
    used_predictions = sum(pair_scores[row][column][1] for row, column in assignment)
    extras = max(0, total_predictions - used_predictions)
    denominator = total_targets + extras
    return (score / denominator if denominator else 0.0), schema_valid


class SavTracksResourcesServer(SimpleResourcesServer):
    config: SavTracksResourcesServerConfig

    async def verify(self, body: SavTracksVerifyRequest) -> SavTracksVerifyResponse:
        tracks, envelope_valid = _parse_tracks(_extract_last_assistant_text(body))
        if body.task == "ref":
            accuracy, schema_valid = _score_ref_tracks(body.objects, tracks)
        else:
            accuracy, schema_valid = _score_keyed_tracks(
                body.task, body.objects, tracks
            )
        format_score = float(envelope_valid and schema_valid)
        reward = format_score * accuracy
        return SavTracksVerifyResponse(
            **body.model_dump(),
            reward=reward,
            format_score=format_score,
            accuracy=accuracy,
            resolved=reward >= 0.999,
            parsed_tracks=tracks,
        )


if __name__ == "__main__":
    SavTracksResourcesServer.run_webserver()
