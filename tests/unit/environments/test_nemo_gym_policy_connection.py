"""Optional Gym endpoint overrides retain the managed endpoints by default."""

import pytest

from nemo_rl.environments.nemo_gym_request import (
    _policy_connection,
    _validate_prompt_token_prefix,
)


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


@pytest.mark.parametrize("override", [None, "", "[]"])
@pytest.mark.parametrize("prompt", [[1, 9, 3], [1], []])
def test_managed_policy_rejects_changed_or_truncated_history(
    monkeypatch: pytest.MonkeyPatch, override: str | None, prompt: list[int]
) -> None:
    if override is not None:
        monkeypatch.setenv("NEMO_GYM_POLICY_BASE_URL", override)
    with pytest.raises(AssertionError, match="Non-contiguous messages found!"):
        _validate_prompt_token_prefix(seen_token_ids=[1, 2], prompt_token_ids=prompt)


@pytest.mark.parametrize(
    "seen,prompt", [([], [1]), ([1, 2], [1, 2]), ([1, 2], [1, 2, 3])]
)
def test_managed_policy_accepts_append_only_history(
    seen: list[int], prompt: list[int]
) -> None:
    _validate_prompt_token_prefix(seen_token_ids=seen, prompt_token_ids=prompt)


@pytest.mark.parametrize("prompt", [[1, 9, 3], [1], []])
def test_external_policy_bypasses_prefix_check_without_rewriting_tokens(
    monkeypatch: pytest.MonkeyPatch, prompt: list[int]
) -> None:
    monkeypatch.setenv("NEMO_GYM_POLICY_BASE_URL", '["http://external/v1"]')
    seen = [1, 2]
    original_prompt = prompt.copy()
    _validate_prompt_token_prefix(seen_token_ids=seen, prompt_token_ids=prompt)
    assert seen == [1, 2]
    assert prompt == original_prompt


def test_model_name_override_alone_keeps_prefix_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NEMO_GYM_POLICY_MODEL_NAME", "external-alias")
    with pytest.raises(AssertionError, match="Non-contiguous messages found!"):
        _validate_prompt_token_prefix(seen_token_ids=[1, 2], prompt_token_ids=[9])


def test_invalid_endpoint_override_does_not_bypass_prefix_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NEMO_GYM_POLICY_BASE_URL", "not-json")
    with pytest.raises(ValueError, match="NEMO_GYM_POLICY_BASE_URL"):
        _validate_prompt_token_prefix(seen_token_ids=[1, 2], prompt_token_ids=[9])
