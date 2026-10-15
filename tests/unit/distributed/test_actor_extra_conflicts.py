"""Check actor extras against uv conflicts without importing GPU dependencies."""

import runpy
import tomllib
from pathlib import Path


def test_actor_extras_do_not_conflict():
    root = Path(__file__).resolve().parents[3]
    actors = runpy.run_path(
        str(root / "nemo_rl/distributed/actor_environments.py")
    )["ACTOR_ENVIRONMENTS"]
    config = tomllib.loads((root / "pyproject.toml").read_text())
    for actor, extras in actors.items():
        selected = {extra.replace("_", "-") for extra in extras or []}
        for conflict in config["tool"]["uv"]["conflicts"]:
            members = {item["extra"].replace("_", "-") for item in conflict}
            assert len(selected & members) < 2, (actor, selected & members)
