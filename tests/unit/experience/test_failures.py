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

"""Tests for the rollout failure taxonomy.

The split these tests pin down is load-bearing: an infra failure re-dispatches the
prompt onto another shard, a data failure does not. Misclassifying data as infra
retries a doomed prompt; misclassifying infra as data fails a recoverable run.
"""

import asyncio

import aiohttp
import pytest
import ray.exceptions
from multidict import CIMultiDict, CIMultiDictProxy
from ray import cloudpickle as ray_cloudpickle
from yarl import URL

from nemo_rl.environments.nemo_gym import _typed_gym_failure
from nemo_rl.experience.failures import (
    FailureClass,
    GenerationUnavailable,
    GymTransportError,
    NoHealthyShards,
    RolloutDataFailure,
    RolloutFailure,
    RolloutInfraFailure,
    RolloutRedispatchExhausted,
    RolloutStall,
    RolloutTimeout,
    classify_rollout_failure,
    http_status_is_infra,
)


class ClientOSError(OSError):
    """A look-alike for aiohttp's ClientOSError, used to exercise the name fallback.

    Named exactly as aiohttp names it, because the fallback path matches on MRO class
    name for environments where aiohttp cannot be imported. The shape matters too:
    aiohttp's ClientOSError derives from OSError but *not* from ConnectionError, so an
    isinstance-only table would miss it.
    """


def _response_error(status: int) -> aiohttp.ClientResponseError:
    return aiohttp.ClientResponseError(
        None, (), status=status, message=f"synthetic {status}"
    )


INFRA_CASES = [
    pytest.param(RolloutInfraFailure("x"), id="infra-base"),
    pytest.param(RolloutTimeout("x"), id="timeout"),
    pytest.param(GenerationUnavailable("x"), id="generation-unavailable"),
    pytest.param(NoHealthyShards("x"), id="no-healthy-shards"),
    pytest.param(GymTransportError("x"), id="gym-transport"),
    pytest.param(TimeoutError("x"), id="builtin-timeout"),
    pytest.param(asyncio.TimeoutError("x"), id="asyncio-timeout"),
    pytest.param(ConnectionRefusedError("x"), id="connection-refused"),
    pytest.param(ConnectionResetError("x"), id="connection-reset"),
    pytest.param(ray.exceptions.RayActorError(), id="ray-actor-error"),
    pytest.param(ray.exceptions.ActorDiedError(), id="ray-actor-died"),
    pytest.param(ray.exceptions.WorkerCrashedError(), id="ray-worker-crashed"),
    pytest.param(ray.exceptions.LocalRayletDiedError(), id="ray-raylet-died"),
    pytest.param(ray.exceptions.GetTimeoutError(), id="ray-get-timeout"),
    pytest.param(ray.exceptions.RpcError("boom"), id="ray-rpc-error"),
    pytest.param(ray.exceptions.NodeDiedError("boom"), id="ray-node-died"),
    pytest.param(ClientOSError("x"), id="client-os-error-by-name-fallback"),
    # Real aiohttp transport errors -- the ones a dying vLLM endpoint produces.
    pytest.param(
        aiohttp.ClientConnectorError(None, OSError("refused")),
        id="aiohttp-connector-error",
    ),
    pytest.param(aiohttp.ServerDisconnectedError(), id="aiohttp-server-disconnected"),
    pytest.param(aiohttp.ServerTimeoutError(), id="aiohttp-server-timeout"),
    pytest.param(aiohttp.ClientPayloadError("truncated"), id="aiohttp-payload-error"),
]

DATA_CASES = [
    pytest.param(RolloutDataFailure("x"), id="data-base"),
    pytest.param(ValueError("prompt too long"), id="value-error"),
    pytest.param(AssertionError("non-contiguous tokens"), id="assertion-error"),
    pytest.param(KeyError("reward"), id="key-error"),
    pytest.param(RuntimeError("generation logprobs contain NaN"), id="runtime-error"),
]


@pytest.mark.parametrize("exc", INFRA_CASES)
def test_infra_exceptions_classify_as_infra(exc):
    assert classify_rollout_failure(exc) is FailureClass.INFRA


@pytest.mark.parametrize("exc", DATA_CASES)
def test_unrecognized_and_data_exceptions_classify_as_data(exc):
    assert classify_rollout_failure(exc) is FailureClass.DATA


@pytest.mark.parametrize("status", [500, 502, 503, 504, 520, 408, 429])
def test_server_side_and_retriable_http_statuses_are_infra(status):
    assert classify_rollout_failure(_response_error(status)) is FailureClass.INFRA


@pytest.mark.parametrize("status", [400, 401, 403, 404, 413, 422])
def test_client_side_http_statuses_are_data(status):
    """A 4xx describes the request, so another shard would reject it identically.

    The motivating case is real: vLLM answers an over-long prompt with
    ``400 {"message": "This model's maximum context length is ..."}``. Re-dispatching
    that burns the infra budget on a prompt no shard can serve.
    """
    assert classify_rollout_failure(_response_error(status)) is FailureClass.DATA


def test_every_status_nemo_gym_retries_is_classified_infra():
    """Stay consistent with NeMo-Gym's own retry set.

    Gym retries RETRY_ERROR_CODES = [429, 502, 503, 504, 520] + [500] internally
    (nemo_gym/openai_utils.py). Anything Gym considers worth retrying must not be
    treated here as a permanent property of the prompt.
    """
    for status in (429, 500, 502, 503, 504, 520):
        assert classify_rollout_failure(_response_error(status)) is FailureClass.INFRA


def test_infra_cause_promotes_an_otherwise_unrecognized_exception():
    """A wrapper around a dead actor is still an infrastructure failure."""
    exc = RuntimeError("rollout failed")
    exc.__cause__ = ray.exceptions.RayActorError()
    assert classify_rollout_failure(exc) is FailureClass.INFRA


def test_explicit_data_failure_wins_over_an_infra_cause():
    """Callers that know a failure is prompt-specific must not be second-guessed.

    A data failure can legitimately be raised while some infra error sits in the cause
    chain (e.g. a shard hiccup surfaced as an empty generation that the prompt would
    reproduce anyway). The explicit classification is authoritative.
    """
    exc = RolloutDataFailure("prompt exceeds max_model_len")
    exc.__cause__ = ConnectionResetError("x")
    assert classify_rollout_failure(exc) is FailureClass.DATA


def test_cause_chain_is_walked_more_than_one_level():
    outer = RuntimeError("outer")
    middle = RuntimeError("middle")
    middle.__cause__ = TimeoutError("inner")
    outer.__cause__ = middle
    assert classify_rollout_failure(outer) is FailureClass.INFRA


def test_cyclic_cause_chain_terminates():
    """A self-referential chain must not spin the classifier."""
    a = RuntimeError("a")
    b = RuntimeError("b")
    a.__cause__ = b
    b.__cause__ = a
    assert classify_rollout_failure(a) is FailureClass.DATA


def test_cause_chain_deeper_than_the_bound_terminates():
    head = RuntimeError("0")
    node = head
    for i in range(1, 50):
        nxt = RuntimeError(str(i))
        node.__cause__ = nxt
        node = nxt
    # The infra marker sits past the bound, so it is not reached -- but the call must
    # still return rather than walk 50 links.
    node.__cause__ = TimeoutError("deep")
    assert classify_rollout_failure(head) is FailureClass.DATA


def test_context_is_not_followed():
    """Only __cause__ (explicit `raise ... from`) counts, not incidental __context__.

    __context__ is set by any exception raised inside an except block, so following it
    would let an unrelated earlier error silently reclassify this one.
    """
    exc = ValueError("bad prompt")
    exc.__context__ = ray.exceptions.RayActorError()
    assert classify_rollout_failure(exc) is FailureClass.DATA


def test_redispatch_exhausted_is_not_catchable_as_a_rollout_failure():
    """The per-attempt retry loop catches RolloutFailure; this must escape it."""
    assert not issubclass(RolloutRedispatchExhausted, RolloutFailure)
    assert not issubclass(RolloutStall, RolloutFailure)


def test_infra_and_data_are_disjoint_branches_of_one_base():
    assert issubclass(RolloutInfraFailure, RolloutFailure)
    assert issubclass(RolloutDataFailure, RolloutFailure)
    assert not issubclass(RolloutInfraFailure, RolloutDataFailure)
    assert not issubclass(RolloutDataFailure, RolloutInfraFailure)


class TestTheRayActorBoundary:
    """The Ray actor boundary is where classification information goes to die.

    Every NeMo-Gym HTTP failure is raised inside the ``NemoGym`` actor and has to cross
    back to the driver to reach the retry policy. ``_response_error`` above -- and every
    other case in this file -- builds a *headerless* ``ClientResponseError``, which
    pickles cleanly. Production never produces that shape: aiohttp's
    ``raise_for_status`` passes ``headers=self.headers``, a ``CIMultiDictProxy``, which
    cloudpickle cannot serialize. Ray then drops the cause and the driver receives a bare
    error carrying neither type nor ``.status`` -- so the status split below classified
    every gym failure DATA, capping the gym path at 2 attempts and leaving the 5-attempt
    infra budget unreachable on the path it was built for.

    These pin the real shape so that cannot silently come back.
    """

    @staticmethod
    def _realistic_response_error(status: int) -> aiohttp.ClientResponseError:
        """Built the way aiohttp's ``raise_for_status`` builds it -- with real headers."""
        headers = CIMultiDictProxy(CIMultiDict({"Content-Type": "application/json"}))
        request_info = aiohttp.RequestInfo(
            URL("http://gym/run"), "POST", headers, URL("http://gym")
        )
        return aiohttp.ClientResponseError(
            request_info,
            (),
            status=status,
            message=f"synthetic {status}",
            headers=headers,
        )

    def test_the_realistic_error_cannot_survive_pickling(self):
        """The premise of the fix. If this ever passes, _typed_gym_failure is dead weight."""
        with pytest.raises(Exception):
            ray_cloudpickle.dumps(self._realistic_response_error(503))

    def test_the_headerless_shape_pickles_which_is_why_the_other_tests_missed_this(
        self,
    ):
        ray_cloudpickle.dumps(_response_error(503))

    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            (500, FailureClass.INFRA),  # what Gym's middleware turns inner errors into
            (503, FailureClass.INFRA),
            (408, FailureClass.INFRA),
            (429, FailureClass.INFRA),
            (400, FailureClass.DATA),  # context-length overflow and friends
            (404, FailureClass.DATA),
        ],
    )
    def test_what_nemo_gym_raises_survives_the_boundary_and_still_classifies(
        self, status, expected
    ):
        typed = _typed_gym_failure(self._realistic_response_error(status))
        assert typed is not None, f"HTTP {status} should have been classified at source"
        # The boundary itself.
        restored = ray_cloudpickle.loads(ray_cloudpickle.dumps(typed))
        assert classify_rollout_failure(restored) is expected
        assert str(status) in str(restored), "the status must survive for the operator"

    def test_an_exception_without_a_status_is_left_untouched(self):
        """No status means no HTTP verdict to make; the caller re-raises as-is."""
        assert _typed_gym_failure(RuntimeError("not an HTTP failure")) is None

    def test_both_sides_of_the_boundary_share_one_status_policy(self):
        """nemo_gym classifies at source, failures.py on the driver -- one rule, not two."""
        for status in (400, 404, 408, 429, 500, 503):
            at_source = isinstance(
                _typed_gym_failure(self._realistic_response_error(status)),
                GymTransportError,
            )
            assert at_source is http_status_is_infra(status)
