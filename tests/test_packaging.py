from pathlib import Path
import tomllib

from setuptools import find_packages


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_setuptools_package_discovery_includes_nemo_rl_subpackages():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    package_find = pyproject["tool"]["setuptools"]["packages"]["find"]

    assert package_find["include"] == ["nemo_rl*"]

    discovered = set(find_packages(where=str(REPO_ROOT), include=package_find["include"]))
    expected = {
        "nemo_rl",
        "nemo_rl.algorithms",
        "nemo_rl.data",
        "nemo_rl.environments",
        "nemo_rl.models",
    }

    assert expected.issubset(discovered)
