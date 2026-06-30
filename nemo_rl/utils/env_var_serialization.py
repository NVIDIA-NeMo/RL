import json


def encode_env_list(values: list[str] | list[int]) -> str:
    """Serialize an environment-variable list payload as JSON."""
    return json.dumps(values)


def decode_env_str_list(raw_value: str, env_var_name: str) -> list[str]:
    """Parse a JSON-encoded environment variable as a list of strings."""
    parsed = _decode_env_list(raw_value, env_var_name)
    if not all(isinstance(item, str) for item in parsed):
        raise ValueError(f"{env_var_name} must be a JSON list of strings")
    return parsed


def decode_env_int_list(raw_value: str, env_var_name: str) -> list[int]:
    """Parse a JSON-encoded environment variable as a list of integers."""
    parsed = _decode_env_list(raw_value, env_var_name)
    if not all(type(item) is int for item in parsed):
        raise ValueError(f"{env_var_name} must be a JSON list of integers")
    return parsed


def _decode_env_list(raw_value: str, env_var_name: str) -> list[object]:
    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{env_var_name} must be valid JSON") from exc

    if not isinstance(parsed, list):
        raise ValueError(f"{env_var_name} must be a JSON list")

    return parsed
