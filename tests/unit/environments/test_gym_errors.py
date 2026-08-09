"""Unit tests for making Gym HTTP errors survive the Ray actor boundary.

Uses `ray.cloudpickle` rather than a hand-rolled check because that is exactly
what Ray uses to ship a task's cause back to the caller.
"""

import pytest
import ray.cloudpickle as cloudpickle
from aiohttp import (
    ClientOSError,
    ClientPayloadError,
    ClientResponseError,
    RequestInfo,
    ServerDisconnectedError,
)
from aiohttp.http_exceptions import ContentLengthError
from multidict import CIMultiDict, CIMultiDictProxy
from yarl import URL

from nemo_rl.environments.gym_errors import (
    MAX_BODY_CHARS,
    GymHTTPStatusError,
    to_picklable_gym_error,
)

_URL = "http://10.109.18.187:5739/run"


def _client_response_error(
    *, status: int = 500, body: bytes | None = b'{"detail":"boom"}'
) -> ClientResponseError:
    """Build a ClientResponseError shaped like the ones aiohttp really raises."""
    url = URL(_URL)
    headers = CIMultiDictProxy(CIMultiDict({"Content-Type": "application/json"}))
    request_info = RequestInfo(url=url, method="POST", headers=headers, real_url=url)
    error = ClientResponseError(
        request_info,
        (),
        status=status,
        message="Internal Server Error",
        headers=headers,
    )
    if body is not None:
        # Gym's raise_for_status attaches the body it already read.
        error.response_content = body
    return error


class TestClientResponseErrorIsUnpicklable:
    """The defect itself: without conversion Ray loses the cause."""

    def test_raw_client_response_error_cannot_be_pickled(self) -> None:
        with pytest.raises(TypeError, match="CIMultiDictProxy"):
            cloudpickle.dumps(_client_response_error())

    @pytest.mark.parametrize(
        "error",
        [
            ClientOSError(111, "Connection refused"),
            ServerDisconnectedError("peer closed"),
            ClientPayloadError("short read"),
            ContentLengthError("bad length"),
        ],
        ids=[
            "ClientOSError",
            "ServerDisconnectedError",
            "ClientPayloadError",
            "ContentLengthError",
        ],
    )
    def test_other_aiohttp_errors_already_survive(self, error: Exception) -> None:
        # These need no conversion, which is why to_picklable_gym_error leaves
        # them alone -- wrapping them would only blur their type.
        revived = cloudpickle.loads(cloudpickle.dumps(error))
        assert type(error).__name__ == type(revived).__name__


class TestToPicklableGymError:
    def test_converts_client_response_error(self) -> None:
        converted = to_picklable_gym_error(_client_response_error())

        assert isinstance(converted, GymHTTPStatusError)
        assert 500 == converted.status
        assert _URL == converted.url
        assert "POST" == converted.method
        assert "ClientResponseError" == converted.original_type
        assert "Internal Server Error" == converted.message
        assert '{"detail":"boom"}' == converted.body

    @pytest.mark.parametrize(
        "error",
        [
            ClientOSError(111, "Connection refused"),
            ServerDisconnectedError("peer closed"),
            ClientPayloadError("short read"),
            ContentLengthError("bad length"),
            RuntimeError("not an aiohttp error at all"),
        ],
        ids=[
            "ClientOSError",
            "ServerDisconnectedError",
            "ClientPayloadError",
            "ContentLengthError",
            "RuntimeError",
        ],
    )
    def test_leaves_picklable_errors_untouched(self, error: Exception) -> None:
        assert to_picklable_gym_error(error) is error

    def test_missing_body_is_tolerated(self) -> None:
        converted = to_picklable_gym_error(_client_response_error(body=None))

        assert converted.body is None
        assert 500 == converted.status

    def test_long_body_is_truncated(self) -> None:
        converted = to_picklable_gym_error(
            _client_response_error(body=b"x" * (MAX_BODY_CHARS + 250))
        )

        assert converted.body is not None
        assert converted.body.startswith("x" * MAX_BODY_CHARS)
        assert "truncated from" in converted.body


class TestConvertedErrorRoundTrip:
    def test_round_trip_preserves_fields_and_type_name(self) -> None:
        converted = to_picklable_gym_error(_client_response_error(status=400))

        revived = cloudpickle.loads(cloudpickle.dumps(converted))

        # The type name is what _rollout_failure_counts buckets on, so this is
        # the assertion that the failure breakdown regains its resolution.
        assert "GymHTTPStatusError" == type(revived).__name__
        assert 400 == revived.status
        assert _URL == revived.url
        assert "POST" == revived.method
        assert "ClientResponseError" == revived.original_type
        assert '{"detail":"boom"}' == revived.body
        assert str(converted) == str(revived)

    def test_round_trip_survives_being_raised_from_an_except_block(self) -> None:
        # Raising inside `except` sets __context__ to the unpicklable original;
        # the round trip must not drag it along.
        try:
            raise _client_response_error()
        except ClientResponseError as original:
            converted = to_picklable_gym_error(original)
            try:
                raise converted from None
            except GymHTTPStatusError as raised:
                revived = cloudpickle.loads(cloudpickle.dumps(raised))

        assert 500 == revived.status
        assert "GymHTTPStatusError" == type(revived).__name__

    def test_ray_delivers_the_cause_to_the_caller(self) -> None:
        # Ray's own delivery path, minus the cluster: RayTaskError pickles the
        # cause, and as_instanceof_cause() builds the class the caller catches
        # and that _rollout_with_retries reads type(e).__name__ from.
        from ray.exceptions import RayTaskError

        converted = to_picklable_gym_error(_client_response_error())
        delivered = RayTaskError(
            "run_rollouts", "<traceback>", converted
        ).as_instanceof_cause()

        # Ray names the class after the cause, so distinct causes no longer
        # collapse into one bare "RayTaskError" bucket.
        assert "GymHTTPStatusError" in type(delivered).__name__
        assert isinstance(delivered, GymHTTPStatusError)
        assert 500 == delivered.status
        assert "ClientResponseError" == delivered.original_type

    def test_message_names_the_original_type_and_endpoint(self) -> None:
        converted = to_picklable_gym_error(_client_response_error())

        text = str(converted)
        assert "500" in text
        assert "ClientResponseError" in text
        assert _URL in text
