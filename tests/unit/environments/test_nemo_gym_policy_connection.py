"""Optional Gym endpoint overrides retain the managed endpoints by default."""

import pytest

from nemo_rl.environments.nemo_gym_request import _policy_connection


@pytest.fixture(autouse=True)
def clear_endpoint_overrides(monkeypatch):
    monkeypatch.delenv("NEMO_GYM_POLICY_MODEL_NAME", raising=False)
    monkeypatch.delenv("NEMO_GYM_POLICY_BASE_URL", raising=False)


@pytest.mark.parametrize("override", [None, "", "[]"])
def test_managed_endpoints_remain_the_default(monkeypatch, override):
    if override is not None:
        monkeypatch.setenv("NEMO_GYM_POLICY_BASE_URL", override)
    monkeypatch.setenv("NEMO_GYM_POLICY_MODEL_NAME", "")
    assert _policy_connection(model_name="managed", base_urls=["http://local/v1"]) == (
        "managed",
        ["http://local/v1"],
    )


@pytest.mark.parametrize("model_override", [None, "external-model"])
def test_external_urls_with_optional_model_override(monkeypatch, model_override):
    monkeypatch.setenv("NEMO_GYM_POLICY_BASE_URL", '["http://one/v1", "http://two/v1"]')
    if model_override is not None:
        monkeypatch.setenv("NEMO_GYM_POLICY_MODEL_NAME", model_override)
    managed = ["http://local/v1"]
    assert _policy_connection(model_name="managed", base_urls=managed) == (
        model_override or "managed",
        ["http://one/v1", "http://two/v1"],
    )
    assert managed == ["http://local/v1"]


def test_model_override_does_not_require_external_urls(monkeypatch):
    monkeypatch.setenv("NEMO_GYM_POLICY_MODEL_NAME", "served-alias")
    assert _policy_connection(model_name="managed", base_urls=["http://local/v1"]) == (
        "served-alias",
        ["http://local/v1"],
    )


@pytest.mark.parametrize(
    "override",
    [
        "http://one/v1",
        '"http://one/v1"',
        "{}",
        "null",
        "1",
        "[null]",
        "[1]",
        '[""]',
        '[" "]',
    ],
)
def test_invalid_endpoint_override_fails_with_variable_name(monkeypatch, override):
    monkeypatch.setenv("NEMO_GYM_POLICY_BASE_URL", override)
    with pytest.raises(
        ValueError, match="NEMO_GYM_POLICY_BASE_URL must be a JSON list"
    ):
        _policy_connection(model_name="managed", base_urls=["http://local/v1"])
