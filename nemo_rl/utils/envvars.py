
import os

def set_envvars(envvars: dict[str, str]):
    """Set environment variables.

    Args:
        envvars: Dictionary of environment variables to set.
    """
    print("Setting environment variables:")
    for key, value in envvars.items():
        os.environ[key] = value
        print(f" - {key}: {value}")
