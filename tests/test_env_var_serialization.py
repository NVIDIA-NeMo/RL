import pytest

from nemo_rl.utils.env_var_serialization import (
    decode_env_int_list,
    decode_env_str_list,
    encode_env_list,
)


def test_encode_env_list_uses_json():
    assert encode_env_list(["10.0.0.1", "10.0.0.2"]) == '["10.0.0.1", "10.0.0.2"]'
    assert encode_env_list([25001, 25002]) == "[25001, 25002]"


def test_decode_env_lists_round_trip():
    assert decode_env_str_list('["10.0.0.1", "10.0.0.2"]', "AVAILABLE_ADDR_LIST") == [
        "10.0.0.1",
        "10.0.0.2",
    ]
    assert decode_env_int_list("[25001, 25002]", "AVAILABLE_PORT_LIST") == [
        25001,
        25002,
    ]


@pytest.mark.parametrize(
    ("raw_value", "env_var_name"),
    [
        ('__import__("os").system("echo pwned")', "AVAILABLE_ADDR_LIST"),
        ('{"addr": "10.0.0.1"}', "AVAILABLE_PORT_LIST"),
    ],
)
def test_decode_env_list_rejects_non_list_json(raw_value, env_var_name):
    with pytest.raises(ValueError, match=env_var_name):
        if env_var_name == "AVAILABLE_ADDR_LIST":
            decode_env_str_list(raw_value, env_var_name)
        else:
            decode_env_int_list(raw_value, env_var_name)


def test_decode_env_list_rejects_wrong_element_types():
    with pytest.raises(ValueError, match="AVAILABLE_ADDR_LIST"):
        decode_env_str_list('["10.0.0.1", 2]', "AVAILABLE_ADDR_LIST")

    with pytest.raises(ValueError, match="AVAILABLE_PORT_LIST"):
        decode_env_int_list('[25001, "25002"]', "AVAILABLE_PORT_LIST")
