from pathlib import Path

from experiments.nemo_gym_phase2.verify_runtime import (
    classify_requirement_issues,
    intentional_dependency_policy,
)


def test_intentional_dependency_policy_reads_uv_overrides_and_excludes(
    tmp_path: Path,
) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """
[tool.uv]
override-dependencies = [
  "setuptools>=80.10.2",
  "llguidance>=1.3.0,<1.4.0",
  "opencv-python-headless; sys_platform == 'never'",
]
exclude-dependencies = ["nvidia-cutlass-dsl-libs-base"]
""",
        encoding="utf-8",
    )

    policy = intentional_dependency_policy(pyproject)

    assert set(policy) == {
        "llguidance",
        "nvidia-cutlass-dsl-libs-base",
        "opencv-python-headless",
        "setuptools",
    }
    assert policy["setuptools"] == ["setuptools>=80.10.2"]


def test_classify_requirement_issues_rejects_unclassified_conflicts() -> None:
    issues = [
        {"required_name": "setuptools", "requirement": "setuptools<81"},
        {"required_name": "surprise-package", "requirement": "surprise-package>=2"},
    ]
    policy = {"setuptools": ["setuptools>=80.10.2"]}

    intentional, unexpected = classify_requirement_issues(issues, policy)

    assert intentional == [
        {
            "required_name": "setuptools",
            "requirement": "setuptools<81",
            "project_policy": ["setuptools>=80.10.2"],
        }
    ]
    assert unexpected == [issues[1]]
