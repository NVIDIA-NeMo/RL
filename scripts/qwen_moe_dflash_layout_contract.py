"""Run the DFlash public-layout contract without importing Torch or pytest."""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def _extract_layout(source: str) -> Any:
    tree = ast.parse(source)
    layout = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_dflash_weight_layout"
    )
    module = ast.Module(body=[layout], type_ignores=[])
    namespace: dict[str, Any] = {"Any": Any}
    exec(compile(module, "_dflash_weight_layout.py", "exec"), namespace)
    return namespace["_dflash_weight_layout"]


def _assert_layouts(layout: Any) -> None:
    qwen3_30b_a3b = SimpleNamespace(
        hidden_size=2048,
        intermediate_size=6144,
        num_attention_heads=32,
        num_key_value_heads=4,
        head_dim=128,
        num_target_taps=5,
    )
    qwen3_8b = SimpleNamespace(
        hidden_size=4096,
        intermediate_size=12288,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        num_target_taps=5,
    )
    expected = {
        ("q30", "layers.0.self_attn.q_proj.weight"): ((4096, 2048), 0),
        ("q30", "layers.0.self_attn.o_proj.weight"): ((2048, 4096), 1),
        ("q30", "layers.0.self_attn.k_proj.weight"): ((512, 2048), 0),
        ("q30", "layers.0.self_attn.v_proj.weight"): ((512, 2048), 0),
        ("q30", "layers.0.mlp.gate_proj.weight"): ((6144, 2048), 0),
        ("q30", "layers.0.mlp.up_proj.weight"): ((6144, 2048), 0),
        ("q30", "layers.0.mlp.down_proj.weight"): ((2048, 6144), 1),
        ("q8", "layers.0.self_attn.q_proj.weight"): ((4096, 4096), 0),
        ("q8", "layers.0.self_attn.o_proj.weight"): ((4096, 4096), 1),
        ("q8", "layers.0.self_attn.k_proj.weight"): ((1024, 4096), 0),
        ("q8", "layers.0.self_attn.v_proj.weight"): ((1024, 4096), 0),
        ("q8", "layers.0.mlp.gate_proj.weight"): ((12288, 4096), 0),
        ("q8", "layers.0.mlp.up_proj.weight"): ((12288, 4096), 0),
        ("q8", "layers.0.mlp.down_proj.weight"): ((4096, 12288), 1),
    }
    configs = {"q30": qwen3_30b_a3b, "q8": qwen3_8b}
    for (variant, parameter_name), expected_layout in expected.items():
        actual = layout(parameter_name, config=configs[variant])
        assert actual == expected_layout, (
            f"{variant} {parameter_name}: expected {expected_layout}, got {actual}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-stdin", action="store_true")
    parser.add_argument(
        "--source-file",
        type=Path,
        default=Path("nemo_rl/models/megatron/draft/utils.py"),
    )
    args = parser.parse_args()
    source = sys.stdin.read() if args.source_stdin else args.source_file.read_text()
    _assert_layouts(_extract_layout(source))
    print("QWEN_MOE_DFLASH_LAYOUT_CONTRACT_PASS")


if __name__ == "__main__":
    main()
