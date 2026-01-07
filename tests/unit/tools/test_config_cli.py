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

# All tests in this module should run first
pytestmark = pytest.mark.run_first


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


def test_minimize_in_place_and_check_with_explicit_base(
    cli: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test minimize with explicit --base option (rebase mode)."""
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

    # Before minimizing with explicit base, check should fail
    ns = type("NS", (), {"base": str(base), "config": str(child)})
    ret = cli.minimize_check(ns)
    assert ret == 1
    err = capsys.readouterr().err
    assert "Suggested fix" in err

    # Minimize in place with explicit base
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


def test_infer_base_from_defaults(cli: Any, tmp_path: Path) -> None:
    """Test that _infer_base_from_defaults correctly resolves the base path."""
    parent = tmp_path / "parent.yaml"
    child = tmp_path / "recipes" / "child.yaml"
    child.parent.mkdir(parents=True, exist_ok=True)
    parent.write_text("key: value\n")
    child.write_text("defaults: ../parent.yaml\noverride: 1\n")

    child_cfg = OmegaConf.load(child)
    base_path = cli._infer_base_from_defaults(child.resolve(), child_cfg)
    assert base_path == parent.resolve()


def test_infer_base_from_defaults_missing_defaults(cli: Any, tmp_path: Path) -> None:
    """Test that missing defaults raises an error."""
    child = tmp_path / "child.yaml"
    child.write_text("key: value\n")

    child_cfg = OmegaConf.load(child)
    with pytest.raises(ValueError, match="no 'defaults' key"):
        cli._infer_base_from_defaults(child.resolve(), child_cfg)


def test_infer_base_from_defaults_list_defaults(cli: Any, tmp_path: Path) -> None:
    """Test that list defaults raises an error (we enforce single inheritance)."""
    child = tmp_path / "child.yaml"
    child.write_text("defaults:\n  - parent1.yaml\n  - parent2.yaml\nkey: value\n")

    child_cfg = OmegaConf.load(child)
    with pytest.raises(ValueError, match="list"):
        cli._infer_base_from_defaults(child.resolve(), child_cfg)


def test_minimize_inferred_base_preserves_chain_overrides(
    cli: Any, tmp_path: Path
) -> None:
    """Test minimize with inferred base correctly handles grandchild → parent → grandparent.

    Scenario:
      - grandparent.yaml: sets teacher.tp = 1
      - parent.yaml (defaults: grandparent): sets teacher.tp = 4
      - child.yaml (defaults: parent): sets teacher.tp = 2 (override back)

    When minimizing child.yaml (with base inferred from defaults=parent.yaml),
    the teacher.tp = 2 must be kept because it differs from the expanded parent (which has 4).
    """
    grandparent = tmp_path / "grandparent.yaml"
    parent = tmp_path / "parent.yaml"
    child = tmp_path / "child.yaml"

    grandparent.write_text(
        dedent(
            """
            teacher:
              tp: 1
              other: base_value
            policy:
              lr: 0.001
            """
        ).strip()
    )
    parent.write_text(
        dedent(
            """
            defaults: grandparent.yaml
            teacher:
              tp: 4
            """
        ).strip()
    )
    child.write_text(
        dedent(
            """
            defaults: parent.yaml
            teacher:
              tp: 2
            custom: child_only
            """
        ).strip()
    )

    # Minimize child with inferred base (should use parent.yaml)
    ns = type("NS", (), {"config": str(child), "base": None, "in_place": False})
    ret = cli.minimize(ns)
    assert ret == 0

    # Re-read child to check what minimize would output
    # (since in_place=False, we need to capture stdout)
    import io
    import sys

    old_stdout = sys.stdout
    sys.stdout = captured = io.StringIO()
    cli.minimize(ns)
    sys.stdout = old_stdout
    minimized = captured.getvalue()

    # teacher.tp = 2 must be kept (differs from parent's expanded value of 4)
    assert "tp: 2" in minimized
    # custom key must be kept
    assert "custom: child_only" in minimized
    # defaults must be preserved as-is
    assert "defaults: parent.yaml" in minimized


def test_minimize_inferred_base_removes_redundant_keys(
    cli: Any, tmp_path: Path
) -> None:
    """Test that keys matching the expanded parent are correctly removed."""
    grandparent = tmp_path / "grandparent.yaml"
    parent = tmp_path / "parent.yaml"
    child = tmp_path / "child.yaml"

    grandparent.write_text(
        dedent(
            """
            common:
              a: 1
              b: 2
            """
        ).strip()
    )
    parent.write_text(
        dedent(
            """
            defaults: grandparent.yaml
            common:
              b: 3
            extra: from_parent
            """
        ).strip()
    )
    # Child redundantly sets common.b = 3 (same as parent) - should be removed
    # Child sets common.a = 1 (same as grandparent, but parent doesn't override) - should be removed
    # Child sets extra = from_parent (same as parent) - should be removed
    # Child sets unique = child_only - should be kept
    child.write_text(
        dedent(
            """
            defaults: parent.yaml
            common:
              a: 1
              b: 3
            extra: from_parent
            unique: child_only
            """
        ).strip()
    )

    import io
    import sys

    ns = type("NS", (), {"config": str(child), "base": None, "in_place": False})
    old_stdout = sys.stdout
    sys.stdout = captured = io.StringIO()
    cli.minimize(ns)
    sys.stdout = old_stdout
    minimized = captured.getvalue()

    # Redundant keys should be removed
    assert "a: 1" not in minimized
    assert "b: 3" not in minimized
    assert "extra: from_parent" not in minimized
    # Unique key should be kept
    assert "unique: child_only" in minimized
    # defaults preserved
    assert "defaults: parent.yaml" in minimized


def test_minimize_check_inferred_base(cli: Any, tmp_path: Path) -> None:
    """Test minimize-check with inferred base."""
    parent = tmp_path / "parent.yaml"
    child = tmp_path / "child.yaml"

    parent.write_text(
        dedent(
            """
            common:
              a: 1
              b: 2
            """
        ).strip()
    )
    # Child is already minimal
    child.write_text(
        dedent(
            """
            defaults: parent.yaml
            common:
              b: 3
            """
        ).strip()
    )

    ns = type("NS", (), {"config": str(child), "base": None})
    ret = cli.minimize_check(ns)
    assert ret == 0  # Already minimized

    # Now add a redundant key
    child.write_text(
        dedent(
            """
            defaults: parent.yaml
            common:
              a: 1
              b: 3
            """
        ).strip()
    )

    ret2 = cli.minimize_check(ns)
    assert ret2 == 1  # Needs minimizing


def test_minimize_with_explicit_base_rebases(cli: Any, tmp_path: Path) -> None:
    """Test that --base option rebases the config to a different parent."""
    grandparent = tmp_path / "grandparent.yaml"
    parent = tmp_path / "parent.yaml"
    child = tmp_path / "child.yaml"

    grandparent.write_text(
        dedent(
            """
            teacher:
              tp: 1
            policy:
              lr: 0.001
            """
        ).strip()
    )
    parent.write_text(
        dedent(
            """
            defaults: grandparent.yaml
            teacher:
              tp: 4
            """
        ).strip()
    )
    child.write_text(
        dedent(
            """
            defaults: parent.yaml
            teacher:
              tp: 2
            """
        ).strip()
    )

    import io
    import sys

    # Minimize with explicit base=grandparent (rebase mode)
    ns = type(
        "NS", (), {"config": str(child), "base": str(grandparent), "in_place": False}
    )
    old_stdout = sys.stdout
    sys.stdout = captured = io.StringIO()
    cli.minimize(ns)
    sys.stdout = old_stdout
    minimized = captured.getvalue()

    # defaults should now point to grandparent
    assert "grandparent.yaml" in minimized
    # teacher.tp = 2 differs from grandparent's 1, so kept
    assert "tp: 2" in minimized
