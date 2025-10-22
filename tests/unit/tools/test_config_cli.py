# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import importlib.util
import inspect
import os
from pathlib import Path
from textwrap import dedent
from typing import Any

import pytest
from omegaconf import OmegaConf


def _load_cli_module() -> Any:
    # Use a path relative to this test file to import tools/config_cli.py
    test_file = Path(__file__).resolve()
    repo_root = test_file.parents[3]
    cli_path = repo_root / "tools" / "config_cli.py"
    assert cli_path.exists(), f"Expected CLI at {cli_path}"
    spec = importlib.util.spec_from_file_location("config_cli", str(cli_path))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


@pytest.fixture(scope="module")
def cli() -> Any:
    return _load_cli_module()


def test__resolve_path_absolute_and_relative(cli: Any, tmp_path: Path) -> None:
    base = tmp_path
    # absolute input stays absolute
    abs_in = "/etc/hosts"
    assert str(cli.resolve_path(base, abs_in)) == abs_in
    # relative input resolves against base
    rel_in = "sub/dir/file.yaml"
    expected = (base / rel_in).resolve()
    assert cli.resolve_path(base, rel_in) == expected


def test__prune_equal_basic(cli: Any) -> None:
    # Dict pruning: remove keys equal to base, keep differences
    a = {"a": 1, "b": {"c": 2, "d": 3}}
    b = {"a": 1, "b": {"c": 9, "d": 3}}
    out = cli._prune_equal(a, b)
    assert out == {"b": {"c": 2}}

    # List pruning: equal lists of same length return REMOVE sentinel
    a_list = [1, 2, 3]
    b_list = [1, 2, 3]
    out_list = cli._prune_equal(a_list, b_list)
    assert out_list is cli.REMOVE

    # Base-type equality returns REMOVE
    assert cli._prune_equal(5, 5) is cli.REMOVE
    # Different base-types keep original
    assert cli._prune_equal(5, 6) == 5


def test__ensure_defaults_relative_variants(cli: Any, tmp_path: Path) -> None:
    base = tmp_path / "configs" / "base.yaml"
    child = tmp_path / "recipes" / "child.yaml"
    child.parent.mkdir(parents=True, exist_ok=True)
    base.parent.mkdir(parents=True, exist_ok=True)
    base.write_text("base: true\n")
    child.write_text("child: true\n")

    # Case 1: no defaults in child
    cfg: dict[str, Any] = {"child": True}
    cli._ensure_defaults_relative(child, base, cfg)
    rel = os.path.relpath(str(base), start=str(child.parent))
    assert cfg["defaults"] == rel

    # Case 2: defaults as string (ensure base inserted first if missing)
    cfg2: dict[str, Any] = {"defaults": "something.yaml"}
    cli._ensure_defaults_relative(child, base, cfg2)
    val = cfg2["defaults"]
    if isinstance(val, list):
        assert val[0] == rel
    else:
        # collapsed to a string only if single element
        assert val == rel or val == "something.yaml"

    # Case 3: defaults list, ensure base is present and order preserved otherwise
    cfg3: dict[str, Any] = {"defaults": ["x.yaml", "y.yaml"]}
    cli._ensure_defaults_relative(child, base, cfg3)
    assert isinstance(cfg3["defaults"], list)
    assert cfg3["defaults"][0] == rel


def test_minimize_in_place_and_check(
    cli: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    base.write_text(
        dedent(
            """
            common:
              a: 1
              list: [1, 2]
              nested:
                x: 0
            top_only: 7
            """
        ).strip()
    )
    child.write_text(
        dedent(
            """
            defaults: parent.yaml
            common:
              a: 1
              list: [1, 2]
              nested:
                x: 1
            new_top: 42
            """
        ).strip()
    )

    # Before minimizing, check should fail
    ns = type("NS", (), {"base": str(base), "config": str(child)})
    ret = cli.minimize_check(ns)
    assert ret == 1
    err = capsys.readouterr().err
    assert "Suggested fix" in err

    # Minimize in place
    ns2 = type("NS", (), {"base": str(base), "config": str(child), "in_place": True})
    ret2 = cli.minimize(ns2)
    assert ret2 == 0
    minimized = child.read_text().strip()
    rel = os.path.relpath(str(base), start=str(child.parent))
    assert minimized.splitlines()[0].startswith("defaults:")
    assert rel in minimized
    # Ensure pruned keys are gone and differences stay
    assert "top_only" not in minimized
    assert "new_top" in minimized
    assert "nested:\n  x: 1" in minimized.replace(
        "\r\n", "\n"
    ) or "nested:\n    x: 1" in minimized.replace("\r\n", "\n")

    # After minimizing, check should pass
    ret3 = cli.minimize_check(ns)
    assert ret3 == 0


def test_expand_and_compare(
    cli: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    parent = tmp_path / "parent.yaml"
    child = tmp_path / "child.yaml"
    parent.write_text(
        dedent(
            """
            base_value: 10
            block:
              a: 1
              b: 2
            """
        ).strip()
    )
    child.write_text(
        dedent(
            """
            defaults: parent.yaml
            base_value: 11
            block:
              b: 3
              c: 4
            """
        ).strip()
    )

    # expand should merge without resolving interpolations; capture stdout
    ns = type("NS", (), {"config": str(child), "in_place": False})
    ret = cli.expand(ns)
    assert ret == 0
    out = capsys.readouterr().out
    # Expect merged keys present
    assert "base_value: 11" in out
    assert "a: 1" in out and "b: 3" in out and "c: 4" in out

    # compare identical files prints identical message
    ns_cmp = type("NS", (), {"left": str(child), "right": str(child)})
    ret_cmp = cli.compare(ns_cmp)
    assert ret_cmp == 0
    out_cmp = capsys.readouterr().out
    assert "Configs are identical" in out_cmp

    # compare different files prints sections: changed
    alt = tmp_path / "alt.yaml"
    alt.write_text(
        dedent(
            """
            defaults: parent.yaml
            base_value: 12
            block:
              a: 9
              b: 3
              d: 5
            """
        ).strip()
    )
    ns_cmp2 = type("NS", (), {"left": str(child), "right": str(alt)})
    ret_cmp2 = cli.compare(ns_cmp2)
    assert ret_cmp2 == 0
    out_cmp2 = capsys.readouterr().out
    assert "Comparing configs" in out_cmp2
    assert "Added in Right" in out_cmp2
    assert "Changed (Left -> Right)" in out_cmp2


def test_vendored_loader_behavior_matches_upstream(tmp_path: Path) -> None:
    # Prepare simple parent/child config files
    parent = tmp_path / "parent.yaml"
    child = tmp_path / "child.yaml"
    parent.write_text(
        dedent(
            """
            base: 1
            block:
              a: 2
              b: 3
            """
        ).strip()
    )
    child.write_text(
        dedent(
            """
            defaults: parent.yaml
            base: 9
            block:
              b: 7
              c: 4
            """
        ).strip()
    )

    # Use text-level expansion comparison by importing both implementations
    # Vendored
    cli = _load_cli_module()
    vendored_cfg = cli.load_config_with_inheritance(str(child))
    vendored = OmegaConf.to_container(vendored_cfg)

    # Upstream via direct import; if it fails, the test should fail
    import nemo_rl.utils.config as upstream

    upstream_cfg = upstream.load_config_with_inheritance(str(child))
    upstream_out = OmegaConf.to_container(upstream_cfg)

    assert vendored == upstream_out


def test_vendored_loader_drift_against_upstream_source() -> None:
    # Enforce exact copy-paste: the vendored function's source must match upstream exactly
    cli = _load_cli_module()
    vendored_fn = cli.load_config_with_inheritance

    import nemo_rl.utils.config as upstream

    upstream_fn = upstream.load_config_with_inheritance

    up_src = inspect.getsource(upstream_fn).strip()
    ven_src = inspect.getsource(vendored_fn).strip()
    assert up_src == ven_src
