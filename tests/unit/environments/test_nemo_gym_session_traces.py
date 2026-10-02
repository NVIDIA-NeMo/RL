"""Multi-trace (train_on_all_session_traces) conversion of NeMo-Gym results."""

import pytest

from nemo_rl.environments.nemo_gym import NemoGym

_NEMO_GYM = NemoGym.__ray_metadata__.modified_class


class _Tokenizer:
    def batch_decode(self, batch):
        return [" ".join(str(t) for t in ids) for ids in batch]


class _Self:
    """Bare NemoGym stand-in carrying only what the conversion reads."""

    _postprocess_nemo_gym_to_nemo_rl_result = (
        _NEMO_GYM._postprocess_nemo_gym_to_nemo_rl_result
    )
    _postprocess_session_traces = _NEMO_GYM._postprocess_session_traces

    def __init__(self, train_on_all_session_traces: bool) -> None:
        self.cfg = {"train_on_all_session_traces": train_on_all_session_traces}


def _turn(prompt, generation):
    return {
        "prompt_token_ids": list(prompt),
        "generation_token_ids": list(generation),
        "generation_log_probs": [-0.1] * len(generation),
    }


def _segment(session_id, parent, turns, segment_index="0"):
    return {
        "output": turns,
        "metadata": {
            "session_id": session_id,
            "parent_session_id": parent,
            "segment_index": segment_index,
            "segment_boundary_reason": "",
        },
    }


def _result():
    subagent = _segment("sub", "root", [_turn([7, 8], [9])])
    root = _segment("root", "", [_turn([1, 2], [3]), _turn([1, 2, 3, 4], [5, 6])])
    empty = _segment("sub2", "root", [_turn([10], [])])
    return {
        # Gym's singular response is the main session's last segment.
        "response": {"output": [_turn([1, 2], [3])]},
        # Subagent first: conversion must still put the main session first.
        "responses": [subagent, root, empty],
        "responses_create_params": {"input": []},
        "reward": 1.0,
    }


def test_all_session_traces_become_training_sequences():
    result = _NEMO_GYM._postprocess_nemo_gym_to_nemo_rl_result(
        _Self(True), {}, _result(), _Tokenizer()
    )

    traces = result["session_traces"]
    assert [t["trace_metadata"]["session_id"] for t in traces] == ["root", "sub"]
    assert [t["trace_metadata"]["trace_in_rollout_idx"] for t in traces] == [0, 1]
    assert traces[1]["trace_metadata"]["parent_session_id"] == "root"
    root_log = traces[0]["message_log"]
    assert [m["token_ids"].tolist() for m in root_log] == [[1, 2], [3], [4], [5, 6]]
    assert [m["token_ids"].tolist() for m in traces[1]["message_log"]] == [[7, 8], [9]]
    # Top-level fields describe the main session, which keys the prompt group.
    assert result["message_log"] is root_log
    assert result["input_message_log"][0]["token_ids"].tolist() == [1, 2]
    # The singular response no longer carries a second copy of the tokens.
    assert "prompt_token_ids" not in result["full_result"]["response"]["output"][0]


def test_flag_off_keeps_the_single_response_path():
    result = _NEMO_GYM._postprocess_nemo_gym_to_nemo_rl_result(
        _Self(False), {}, _result(), _Tokenizer()
    )

    assert "session_traces" not in result
    assert [m["token_ids"].tolist() for m in result["message_log"]] == [[1, 2], [3]]


def test_a_rollout_with_no_trainable_segment_still_fails():
    nemo_gym_result = _result()
    nemo_gym_result["responses"] = [_segment("root", "", [_turn([1], [])])]

    with pytest.raises(ValueError, match="no generation data"):
        _NEMO_GYM._postprocess_nemo_gym_to_nemo_rl_result(
            _Self(True), {}, nemo_gym_result, _Tokenizer()
        )
