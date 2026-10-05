# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from app import (
    CocoRLE,
    SavObjectTrack,
    SavTarget,
    _box_iou,
    _parse_tracks,
    _point_in_mask,
    _score_keyed_tracks,
    _score_ref_tracks,
)


def _mask() -> CocoRLE:
    # 4x4 Fortran-order mask with foreground at x=1..2, y=1..2.
    return CocoRLE(size=(4, 4), counts=[5, 2, 2, 2, 5])


def test_parse_tracks_requires_one_clean_answer_envelope() -> None:
    tracks, valid = _parse_tracks(
        '<answer>{"tracks":[{"point":1,"frame":1,"visible":true,"xy":[500,500]}]}</answer>'
    )
    assert valid
    assert tracks[0]["point"] == 1
    assert _parse_tracks("reasoning " + '<answer>{"tracks":[]}</answer>')[1] is False


def test_point_in_coco_rle_mask() -> None:
    assert _point_in_mask(_mask(), (500, 500))
    assert not _point_in_mask(_mask(), (0, 0))


def test_box_iou() -> None:
    assert _box_iou((0, 0, 100, 100), right=(0, 0, 100, 100)) == 1.0
    assert _box_iou((0, 0, 100, 100), right=(50, 0, 150, 100)) == pytest.approx(1 / 3)


def test_point_tracks_score_visibility_and_mask_membership() -> None:
    objects = [
        SavObjectTrack(
            id=1,
            targets=[
                SavTarget(frame=1, visible=True, xy=(500, 500), mask=_mask()),
                SavTarget(frame=2, visible=False),
            ],
        )
    ]
    tracks = [
        {"point": 1, "frame": 1, "visible": True, "xy": [500, 500]},
        {"point": 1, "frame": 2, "visible": False},
    ]
    score, valid = _score_keyed_tracks("pt", objects, tracks)
    assert valid
    assert score == 1.0


def test_ref_tracks_match_consistent_ids_independent_of_label() -> None:
    objects = [
        SavObjectTrack(
            id=1,
            targets=[
                SavTarget(frame=1, visible=True, xy=(500, 500), mask=_mask()),
                SavTarget(frame=2, visible=False),
            ],
        ),
    ]
    score, valid = _score_ref_tracks(objects, [{"id": 9, "frame": 1, "xy": [500, 500]}])
    assert valid
    assert score == 1.0
