# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from resources_servers.sav_tracks.app import (
    SavTracksRunRequest,
    _extract_answer_json,
    _parse_tracks,
    _point_in_rle,
    _rle_counts_to_ints,
    _score_tracks,
)


def encode_rle(mask_rows: list[list[int]]) -> dict:
    """COCO compressed RLE of a small binary mask given as rows (h x w)."""
    h = len(mask_rows)
    w = len(mask_rows[0])
    # column-major flatten
    flat = [mask_rows[y][x] for x in range(w) for y in range(h)]
    runs = []
    current = 0
    length = 0
    for value in flat:
        if value == current:
            length += 1
        else:
            runs.append(length)
            current = value
            length = 1
    runs.append(length)
    # pycocotools string encoding: delta from two runs back (beyond the first two)
    chars = []
    for i, run in enumerate(runs):
        value = run if i <= 2 else run - runs[i - 2]
        more = True
        while more:
            c = value & 0x1F
            value >>= 5
            if c & 0x10:
                more = value != -1
            else:
                more = value != 0
            if more:
                c |= 0x20
            chars.append(chr(c + 48))
    return {"size": [h, w], "counts": "".join(chars)}


# A 10x10 mask whose foreground is the square x,y in [3,7) — grid1000 coords in
# [300,700) land inside.
SQUARE_MASK = encode_rle([[1 if 3 <= x < 7 and 3 <= y < 7 else 0 for x in range(10)] for y in range(10)])


def make_body(task, objects, **kwargs):
    return SavTracksRunRequest.model_validate(
        {"responses_create_params": {"input": "unused"}, "task": task, "objects": objects, **kwargs}
    )


def parse(task, answer):
    _, entries, format_ok = _parse_tracks(answer, task)
    return entries, format_ok


# ---------------------------------------------------------------- RLE helpers


def test_rle_roundtrip_membership():
    assert _point_in_rle(SQUARE_MASK, 450, 450) is True
    assert _point_in_rle(SQUARE_MASK, 100, 450) is False
    assert _point_in_rle(SQUARE_MASK, 450, 100) is False
    assert _point_in_rle(SQUARE_MASK, 999, 999) is False
    # boundary: grid 300 -> pixel 3 (inside), grid 700 -> pixel 7 (outside)
    assert _point_in_rle(SQUARE_MASK, 300, 300) is True
    assert _point_in_rle(SQUARE_MASK, 700, 300) is False


def test_rle_raw_counts_list():
    # 2x2 mask, column-major: [bg, fg, fg, bg] -> pixel (0,1) and (1,0) are fg
    mask = {"size": [2, 2], "counts": [1, 2, 1]}
    assert _point_in_rle(mask, 0, 999) is True
    assert _point_in_rle(mask, 999, 0) is True
    assert _point_in_rle(mask, 0, 0) is False
    assert _point_in_rle(mask, 999, 999) is False


def test_rle_malformed():
    assert _point_in_rle({"size": [10], "counts": "abc"}, 0, 0) is None
    assert _point_in_rle({"size": [10, 10], "counts": None}, 0, 0) is None
    assert _rle_counts_to_ints([3, -1]) is None


# ---------------------------------------------------------------- answer extraction


def test_answer_tag_extraction():
    answer, has_wrapper = _extract_answer_json("<think>x</think>\n<answer>{}</answer>", require_answer_tags=True)
    assert has_wrapper and answer == "{}"
    answer, has_wrapper = _extract_answer_json("no tags", require_answer_tags=True)
    assert not has_wrapper and answer is None


# ---------------------------------------------------------------- pt task


def test_pt_in_mask_scores_one():
    entries, format_ok = parse("pt", '{"tracks":[{"point":1,"frame":1,"xy":[450,450],"visible":true}]}')
    assert format_ok
    body = make_body(
        "pt",
        [{"id": 1, "reference_xy": [450, 450], "targets": [{"frame": 1, "visible": True, "mask": SQUARE_MASK}]}],
    )
    result = _score_tracks(body, entries, absence_score=1.0)
    assert result["reward"] == 1.0
    assert result["in_mask_rate"] == 1.0


def test_pt_out_of_mask_scores_zero():
    entries, format_ok = parse("pt", '{"tracks":[{"point":1,"frame":1,"xy":[100,100],"visible":true}]}')
    assert format_ok
    body = make_body("pt", [{"id": 1, "targets": [{"frame": 1, "visible": True, "mask": SQUARE_MASK}]}])
    result = _score_tracks(body, entries, absence_score=1.0)
    assert result["reward"] == 0.0
    assert result["in_mask_rate"] == 0.0


def test_pt_absence_credit_and_false_absence():
    body = make_body(
        "pt",
        [
            {
                "id": 1,
                "targets": [
                    {"frame": 1, "visible": False},
                    {"frame": 2, "visible": True, "mask": SQUARE_MASK},
                ],
            }
        ],
    )
    # correct absence + correct point -> 1.0; xy null on invisible claim is valid
    entries, format_ok = parse(
        "pt",
        '{"tracks":[{"point":1,"frame":1,"xy":null,"visible":false},'
        '{"point":1,"frame":2,"xy":[400,400],"visible":true}]}',
    )
    assert format_ok
    assert _score_tracks(body, entries, absence_score=1.0)["reward"] == 1.0
    # claiming invisible on the visible frame scores 0 there
    entries, _ = parse(
        "pt",
        '{"tracks":[{"point":1,"frame":1,"visible":false},{"point":1,"frame":2,"visible":false}]}',
    )
    assert _score_tracks(body, entries, absence_score=1.0)["reward"] == 0.5
    # pointing on the invisible frame scores 0 there
    entries, _ = parse(
        "pt",
        '{"tracks":[{"point":1,"frame":1,"xy":[400,400],"visible":true},'
        '{"point":1,"frame":2,"xy":[400,400],"visible":true}]}',
    )
    assert _score_tracks(body, entries, absence_score=1.0)["reward"] == 0.5


def test_pt_visible_entry_with_location_and_false_flag_is_invalid():
    entries, format_ok = parse("pt", '{"tracks":[{"point":1,"frame":1,"xy":[400,400],"visible":false}]}')
    assert not format_ok
    assert entries[0].error is not None


def test_pt_missing_entry_scores_zero_not_absence():
    body = make_body("pt", [{"id": 1, "targets": [{"frame": 1, "visible": False}]}])
    result = _score_tracks(body, [], absence_score=1.0)
    # pt omission is NOT an invisibility claim (unlike ref)
    assert result["reward"] == 0.0
    assert result["missing_entry_count"] == 1


def test_pt_extras_scale_reward():
    body = make_body("pt", [{"id": 1, "targets": [{"frame": 1, "visible": True, "mask": SQUARE_MASK}]}])
    entries, format_ok = parse(
        "pt",
        '{"tracks":[{"point":1,"frame":1,"xy":[450,450],"visible":true},'
        '{"point":2,"frame":1,"xy":[450,450],"visible":true}]}',
    )
    assert format_ok
    result = _score_tracks(body, entries, absence_score=1.0)
    assert result["reward"] == 1.0 * 1 / (1 + 1)


def test_pt_multi_object_mean():
    body = make_body(
        "pt",
        [
            {"id": 1, "targets": [{"frame": 1, "visible": True, "mask": SQUARE_MASK}]},
            {"id": 2, "targets": [{"frame": 1, "visible": True, "mask": SQUARE_MASK}]},
        ],
    )
    entries, _ = parse(
        "pt",
        '{"tracks":[{"point":1,"frame":1,"xy":[450,450],"visible":true},'
        '{"point":2,"frame":1,"xy":[100,100],"visible":true}]}',
    )
    result = _score_tracks(body, entries, absence_score=1.0)
    assert result["reward"] == 0.5
    assert result["visibility_accuracy"] == 1.0


# ---------------------------------------------------------------- box task


def test_box_exact_iou_scores_one():
    body = make_body(
        "box",
        [
            {
                "id": 1,
                "reference_bbox": [0, 0, 100, 100],
                "targets": [{"frame": 1, "visible": True, "bbox": [100, 100, 300, 300]}],
            }
        ],
    )
    entries, format_ok = parse("box", '{"tracks":[{"box":1,"frame":1,"bbox":[100,100,300,300],"visible":true}]}')
    assert format_ok
    result = _score_tracks(body, entries, absence_score=1.0)
    assert result["reward"] == 1.0
    assert result["mean_iou"] == 1.0


def test_box_partial_iou():
    body = make_body("box", [{"id": 1, "targets": [{"frame": 1, "visible": True, "bbox": [0, 0, 200, 200]}]}])
    entries, _ = parse("box", '{"tracks":[{"box":1,"frame":1,"bbox":[100,0,300,200],"visible":true}]}')
    result = _score_tracks(body, entries, absence_score=1.0)
    assert abs(result["reward"] - 1 / 3) < 1e-9


def test_box_null_bbox_invisible_claim_is_valid():
    # the GT jsonls themselves emit "bbox": null alongside "visible": false
    entries, format_ok = parse("box", '{"tracks":[{"box":1,"frame":1,"bbox":null,"visible":false}]}')
    assert format_ok
    body = make_body("box", [{"id": 1, "targets": [{"frame": 1, "visible": False}]}])
    assert _score_tracks(body, entries, absence_score=1.0)["reward"] == 1.0


def test_box_out_of_range_bbox_is_invalid():
    entries, format_ok = parse("box", '{"tracks":[{"box":1,"frame":1,"bbox":[0,0,1200,100],"visible":true}]}')
    assert not format_ok
    entries, format_ok = parse("box", '{"tracks":[{"box":1,"frame":1,"bbox":[300,100,100,300],"visible":true}]}')
    assert not format_ok


def test_box_duplicate_entries_penalized_best_kept():
    body = make_body("box", [{"id": 1, "targets": [{"frame": 1, "visible": True, "bbox": [0, 0, 200, 200]}]}])
    entries, _ = parse(
        "box",
        '{"tracks":[{"box":1,"frame":1,"bbox":[0,0,200,200],"visible":true},'
        '{"box":1,"frame":1,"bbox":[500,500,700,700],"visible":true}]}',
    )
    result = _score_tracks(body, entries, absence_score=1.0)
    # best candidate (IoU 1.0) kept, duplicate counts as an extra
    assert result["reward"] == 1.0 * 1 / (1 + 1)


# ---------------------------------------------------------------- ref task


def test_ref_omission_is_absence_claim():
    body = make_body(
        "ref",
        [
            {
                "id": 1,
                "targets": [
                    {"frame": 1, "visible": True, "mask": SQUARE_MASK},
                    {"frame": 2, "visible": False},
                ],
            }
        ],
    )
    entries, format_ok = parse("ref", '{"tracks":[{"id":1,"frame":1,"xy":[450,450]}]}')
    assert format_ok
    result = _score_tracks(body, entries, absence_score=1.0)
    assert result["reward"] == 1.0
    assert result["visibility_accuracy"] == 1.0


def test_ref_id_assignment_is_label_invariant():
    # second object lives in the top-left corner square x,y in [0,2)
    corner_mask = encode_rle([[1 if x < 2 and y < 2 else 0 for x in range(10)] for y in range(10)])
    body = make_body(
        "ref",
        [
            {"id": 1, "targets": [{"frame": 1, "visible": True, "mask": SQUARE_MASK}]},
            {"id": 2, "targets": [{"frame": 1, "visible": True, "mask": corner_mask}]},
        ],
    )
    # model picks arbitrary ids, in the "wrong" order relative to GT
    entries, _ = parse(
        "ref",
        '{"tracks":[{"id":9,"frame":1,"xy":[450,450]},{"id":5,"frame":1,"xy":[50,50]}]}',
    )
    result = _score_tracks(body, entries, absence_score=1.0)
    assert result["reward"] == 1.0
    scores = {s.id: s for s in result["object_scores"]}
    assert scores[1].assigned_prediction_id == 9
    assert scores[2].assigned_prediction_id == 5


def test_ref_unmatched_gt_object_scores_zero():
    body = make_body(
        "ref",
        [
            {"id": 1, "targets": [{"frame": 1, "visible": True, "mask": SQUARE_MASK}]},
            {"id": 2, "targets": [{"frame": 1, "visible": True, "mask": SQUARE_MASK}]},
        ],
    )
    entries, _ = parse("ref", '{"tracks":[{"id":1,"frame":1,"xy":[450,450]}]}')
    result = _score_tracks(body, entries, absence_score=1.0)
    # one object found (1.0), one unmatched (0.0) — omission of a whole GT object
    # is not an absence claim
    assert result["reward"] == 0.5


def test_ref_extra_predicted_id_penalized():
    body = make_body("ref", [{"id": 1, "targets": [{"frame": 1, "visible": True, "mask": SQUARE_MASK}]}])
    entries, _ = parse(
        "ref",
        '{"tracks":[{"id":1,"frame":1,"xy":[450,450]},{"id":2,"frame":1,"xy":[100,100]}]}',
    )
    result = _score_tracks(body, entries, absence_score=1.0)
    assert result["reward"] == 1.0 * 1 / (1 + 1)


def test_ref_missing_xy_is_invalid():
    entries, format_ok = parse("ref", '{"tracks":[{"id":1,"frame":1}]}')
    assert not format_ok


def test_ref_explicit_invisible_entry_is_valid():
    # one ref template answers with explicit visibility flags instead of omission
    body = make_body(
        "ref",
        [
            {
                "id": 1,
                "targets": [
                    {"frame": 1, "visible": True, "mask": SQUARE_MASK},
                    {"frame": 2, "visible": False},
                ],
            }
        ],
    )
    entries, format_ok = parse(
        "ref",
        '{"tracks":[{"id":1,"frame":1,"xy":[450,450],"visible":true},{"id":1,"frame":2,"visible":false}]}',
    )
    assert format_ok
    assert _score_tracks(body, entries, absence_score=1.0)["reward"] == 1.0


# ---------------------------------------------------------------- format handling


def test_non_json_answer_no_reward():
    entries, format_ok = parse("pt", "not json")
    assert not format_ok and entries == []


def test_wrong_top_level_shape():
    entries, format_ok = parse("pt", '{"points":[]}')
    assert not format_ok


def test_empty_targets_reward_requires_no_extras():
    body = make_body("pt", [])
    assert _score_tracks(body, [], absence_score=1.0)["reward"] == 1.0
    entries, _ = parse("pt", '{"tracks":[{"point":1,"frame":1,"xy":[450,450],"visible":true}]}')
    assert _score_tracks(body, entries, absence_score=1.0)["reward"] == 0.0
