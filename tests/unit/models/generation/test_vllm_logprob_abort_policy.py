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

"""Tests for the generation log-prob abort policy."""

import logging

import pytest

from nemo_rl.models.generation.vllm.utils import (
    GENERATION_LOGPROB_CONSECUTIVE_FAILURE_LIMIT,
    GENERATION_LOGPROB_FAILURE_WINDOW,
    GENERATION_LOGPROB_MIN_RATE_SAMPLES,
    GenerationLogprobValidator,
    resolve_generation_logprob_settings,
)

_UTILS_LOGGER = "nemo_rl.models.generation.vllm.utils"


class FakeLogprob:
    def __init__(self, logprob):
        self.logprob = logprob


def _bad_sample(validator, index=0):
    return validator.extract(
        [11], [{99: FakeLogprob(-1.0)}], sample_label=f"sample_idx={index}"
    )


def _good_sample(validator, index=0):
    return validator.extract(
        [11], [{11: FakeLogprob(-1.0)}], sample_label=f"sample_idx={index}"
    )


def _cfg(**overrides):
    base = {
        "strict_generation_logprobs": True,
        "max_generation_logprob_failure_rate": 0.01,
    }
    base.update(overrides)
    return base


# ------------------------------------------------------------- resolution


def test_defaults_from_the_exemplar_yaml():
    assert resolve_generation_logprob_settings(_cfg()) == (True, 0.01)


@pytest.mark.parametrize(
    ("vllm_cfg", "expected"),
    [
        pytest.param(
            _cfg(strict_generation_logprobs=None), (True, 0.01), id="strict_null"
        ),
        pytest.param(
            _cfg(max_generation_logprob_failure_rate=None), (True, 1.0), id="rate_null"
        ),
        pytest.param(
            _cfg(strict_generation_logprobs=False), (False, 0.01), id="strict_off"
        ),
        pytest.param(
            _cfg(max_generation_logprob_failure_rate="0.25"), (True, 0.25), id="string"
        ),
    ],
)
def test_settings_resolution(vllm_cfg, expected):
    assert resolve_generation_logprob_settings(vllm_cfg) == expected


@pytest.mark.parametrize("rate", [-0.1, 1.5, float("nan"), "abc"])
def test_invalid_failure_rate_is_rejected(rate):
    with pytest.raises(ValueError, match="max_generation_logprob_failure_rate"):
        resolve_generation_logprob_settings(
            _cfg(max_generation_logprob_failure_rate=rate)
        )


def test_missing_keys_fail_loudly():
    """Defaults live in the exemplar YAML, not at the call site."""
    with pytest.raises(KeyError):
        resolve_generation_logprob_settings({})


# ------------------------------------------------------------ abort logic


def test_a_single_failure_does_not_abort_the_worker():
    validator = GenerationLogprobValidator(strict=True, max_failure_rate=0.01)
    assert not _bad_sample(validator).valid
    assert validator.failures == 1


def test_worker_aborts_once_the_windowed_failure_rate_is_exceeded():
    validator = GenerationLogprobValidator(strict=True, max_failure_rate=0.01)
    for index in range(99):
        _good_sample(validator, index)
    _bad_sample(validator, 99)  # 1/100 == 1%, not above the threshold
    assert validator.failures == 1

    with pytest.raises(RuntimeError, match="above the tolerated"):
        _bad_sample(validator, 100)  # 2/101 > 1%


def test_a_permissive_rate_still_needs_a_meaningful_sample_count():
    """ceil(1/0.5) is 2; judging a rate on two samples is not a rate."""
    validator = GenerationLogprobValidator(strict=True, max_failure_rate=0.5)
    for index in range(GENERATION_LOGPROB_MIN_RATE_SAMPLES - 1):
        _bad_sample(validator, index) if index % 3 == 0 else _good_sample(
            validator, index
        )
    assert validator.failure_rate > 0.5 or validator.failures > 0
    # Nothing has aborted yet despite a >50% rate being arithmetically
    # reachable from sample 2 onwards.
    assert validator.samples == GENERATION_LOGPROB_MIN_RATE_SAMPLES - 1


def test_late_onset_corruption_is_not_diluted_by_history():
    validator = GenerationLogprobValidator(strict=True, max_failure_rate=0.1)
    for index in range(5 * GENERATION_LOGPROB_FAILURE_WINDOW):
        _good_sample(validator, index)
    assert validator.cumulative_failure_rate == 0.0

    # One in five now fails: 20% of the window, a few percent of the run, and
    # never enough in a row to hit the consecutive trigger.
    with pytest.raises(RuntimeError, match="of the last"):
        for index in range(2 * GENERATION_LOGPROB_FAILURE_WINDOW):
            _bad_sample(validator, index)
            for offset in range(4):
                _good_sample(validator, index * 10 + offset)
    assert validator.consecutive_failures < GENERATION_LOGPROB_CONSECUTIVE_FAILURE_LIMIT
    assert validator.cumulative_failure_rate < 0.1


def test_consecutive_failures_abort_before_the_window_fills():
    validator = GenerationLogprobValidator(strict=True, max_failure_rate=0.9)
    with pytest.raises(RuntimeError, match="consecutive samples failed"):
        for index in range(GENERATION_LOGPROB_CONSECUTIVE_FAILURE_LIMIT):
            _bad_sample(validator, index)


def test_a_clean_sample_resets_the_consecutive_counter():
    validator = GenerationLogprobValidator(strict=True, max_failure_rate=0.9)
    for index in range(GENERATION_LOGPROB_CONSECUTIVE_FAILURE_LIMIT - 1):
        _bad_sample(validator, index)
    _good_sample(validator)
    assert validator.consecutive_failures == 0


def test_zero_threshold_aborts_on_the_first_failure():
    validator = GenerationLogprobValidator(strict=True, max_failure_rate=0.0)
    with pytest.raises(RuntimeError, match="tolerated failure rate is 0"):
        _bad_sample(validator)


def test_rate_of_one_never_aborts_including_consecutively():
    """1.0 means "tolerate anything"; the consecutive trigger must respect it."""
    validator = GenerationLogprobValidator(strict=True, max_failure_rate=1.0)
    for index in range(GENERATION_LOGPROB_CONSECUTIVE_FAILURE_LIMIT * 4):
        assert not _bad_sample(validator, index).valid
    assert validator.failures == GENERATION_LOGPROB_CONSECUTIVE_FAILURE_LIMIT * 4


def test_non_strict_never_aborts():
    validator = GenerationLogprobValidator(strict=False, max_failure_rate=0.0)
    for index in range(GENERATION_LOGPROB_CONSECUTIVE_FAILURE_LIMIT * 2):
        _bad_sample(validator, index)
    assert validator.failures == GENERATION_LOGPROB_CONSECUTIVE_FAILURE_LIMIT * 2


def test_samples_are_still_reported_invalid_when_the_policy_is_off():
    """The per-sample masking in the base change does not depend on this."""
    validator = GenerationLogprobValidator(strict=False, max_failure_rate=1.0)
    assert not _bad_sample(validator).valid


def test_failures_are_warned_on_a_backoff_schedule(caplog):
    caplog.set_level(logging.WARNING, logger=_UTILS_LOGGER)
    validator = GenerationLogprobValidator(strict=True, max_failure_rate=1.0)
    for index in range(8):
        _bad_sample(validator, index)

    warnings = [
        record
        for record in caplog.records
        if "Generation log-prob validation failed" in record.getMessage()
    ]
    # 1st, 2nd, 4th and 8th failure, not all eight.
    assert len(warnings) == 4


def test_from_config_reads_the_vllm_cfg_keys():
    validator = GenerationLogprobValidator.from_config(
        _cfg(
            strict_generation_logprobs=False,
            max_generation_logprob_failure_rate=0.5,
        )
    )
    assert validator.strict is False
    assert validator.max_failure_rate == pytest.approx(0.5)
