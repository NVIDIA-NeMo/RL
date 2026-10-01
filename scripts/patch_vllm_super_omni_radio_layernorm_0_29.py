#!/usr/bin/env python3
"""Apply the Super Omni RADIO final-LayerNorm fix to installed vLLM 0.29.

The source change is from vLLM commit
5dc66ef26ff28f503034d9ed21564808a894fe44. NeMo-RL worker environments
contain their own vLLM installation, so the launcher runs this script after
creating each worker environment rather than replacing the compiled package.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import os
import shutil
import sys
import tempfile
from pathlib import Path


PATCH_MARKER = "# NeMo-RL backport of vLLM commit 5dc66ef26f."
EXPECTED_VLLM_VERSION = "0.29.0"


def _replace_once(source: str, old: str, new: str, description: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"Expected exactly one {description} insertion point, found {count}. "
            "The installed vLLM source does not match the qualified 0.29.0 layout."
        )
    return source.replace(old, new, 1)


def patch_source(source: str) -> tuple[str, bool]:
    """Return patched source and whether a modification was required."""
    required_fragments = (
        "self.vision_final_layernorm",
        "def _apply_vision_final_layernorm",
        '"vision_projector.vision_final_layernorm."',
        '"vision_final_layernorm"',
        "_loaded_vision_final_layernorm_params",
    )
    if PATCH_MARKER in source:
        missing = [
            fragment for fragment in required_fragments if fragment not in source
        ]
        if missing:
            raise RuntimeError(
                f"Existing RADIO final-LayerNorm patch is incomplete: {missing}"
            )
        return source, False

    source = _replace_once(
        source,
        """            self.mlp1 = mlp1.to(llm_dtype)
            self.sound_encoder: ProjectedParakeet | None = None
""",
        f"""            self.mlp1 = mlp1.to(llm_dtype)
            {PATCH_MARKER}
            self.vision_final_layernorm: nn.LayerNorm | None = None
            if (getattr(config.text_config, "num_nextn_predict_layers", 0) or 0) > 0:
                # Megatron adds this post-RADIO norm when the inherited vision
                # config has MTP enabled. Keep it available for dummy-load +
                # refit, but do not apply it until its checkpoint tensors load.
                self.vision_final_layernorm = nn.LayerNorm(
                    vit_hidden_size,
                    eps=getattr(vision_config, "layer_norm_eps", 1.0e-6),
                ).float()
            self._loaded_vision_final_layernorm_params: set[str] = set()
            self._vision_final_layernorm_enabled = False
            self.sound_encoder: ProjectedParakeet | None = None
""",
        "model initialization",
    )

    source = _replace_once(
        source,
        """    def extract_feature_dynamic(
""",
        """    def _apply_vision_final_layernorm(
        self, vit_embeds: torch.Tensor
    ) -> torch.Tensor:
        if not self._vision_final_layernorm_enabled:
            return vit_embeds
        assert self.vision_final_layernorm is not None
        output_dtype = vit_embeds.dtype
        return self.vision_final_layernorm(vit_embeds.float()).to(output_dtype)

    def extract_feature_dynamic(
""",
        "LayerNorm helper",
    )

    source = _replace_once(
        source,
        """        _, vit_embeds = self.vision_model(pixel_values, imgs_sizes=imgs_sizes)
        vit_embeds = vit_embeds.to(dtype=torch.bfloat16)
""",
        """        _, vit_embeds = self.vision_model(pixel_values, imgs_sizes=imgs_sizes)
        vit_embeds = self._apply_vision_final_layernorm(vit_embeds)
        vit_embeds = vit_embeds.to(dtype=torch.bfloat16)
""",
        "dynamic-image forward",
    )

    source = _replace_once(
        source,
        """            else:
                _, vit_embeds = self.vision_model(chunk)
            vit_embeds = vit_embeds.to(dtype=torch.bfloat16)
""",
        """            else:
                _, vit_embeds = self.vision_model(chunk)
            vit_embeds = self._apply_vision_final_layernorm(vit_embeds)
            vit_embeds = vit_embeds.to(dtype=torch.bfloat16)
""",
        "fixed/chunked image-video forward",
    )

    source = _replace_once(
        source,
        """            connector=["mlp1", "sound_encoder.projection"],
""",
        """            connector=[
                "mlp1",
                "vision_final_layernorm",
                "sound_encoder.projection",
            ],
""",
        "multimodal connector mapping",
    )

    source = _replace_once(
        source,
        """        adapter_dict = dict(self.mlp1.named_parameters())

        def is_llm(name: str) -> bool:
""",
        """        adapter_dict = dict(self.mlp1.named_parameters())
        final_layernorm = getattr(self, "vision_final_layernorm", None)
        final_layernorm_dict = (
            dict(final_layernorm.named_parameters())
            if load_multimodal_weights and final_layernorm is not None
            else {}
        )

        def is_llm(name: str) -> bool:
""",
        "LayerNorm parameter map",
    )

    source = _replace_once(
        source,
        """        def is_adapter_weights(weight: tuple[str, torch.Tensor]):
            return weight[0].startswith("mlp1")

        def is_vision_weights(name: str) -> bool:
""",
        """        def is_adapter_weights(weight: tuple[str, torch.Tensor]):
            return weight[0].startswith("mlp1")

        def get_final_layernorm_name(name: str) -> str | None:
            for source_prefix in (
                "vision_final_layernorm.",
                "vision_projector.vision_final_layernorm.",
            ):
                if name.startswith(source_prefix):
                    return name.removeprefix(source_prefix)
            return None

        def is_vision_weights(name: str) -> bool:
""",
        "LayerNorm checkpoint-name mapper",
    )

    source = _replace_once(
        source,
        """        adapter_weights: list[tuple[str, torch.Tensor]] = []
        vision_weights: list[tuple[str, torch.Tensor]] = []
""",
        """        adapter_weights: list[tuple[str, torch.Tensor]] = []
        final_layernorm_weights: list[tuple[str, torch.Tensor]] = []
        vision_weights: list[tuple[str, torch.Tensor]] = []
""",
        "LayerNorm weight buffer",
    )

    source = _replace_once(
        source,
        """                    adapter_weights.append((trimmed_name, w.detach().clone()))
                elif is_vision_weights(name):
""",
        """                    adapter_weights.append((trimmed_name, w.detach().clone()))
                elif (
                    final_layernorm_name := get_final_layernorm_name(name)
                ) is not None:
                    if not final_layernorm_dict:
                        continue
                    final_layernorm_weights.append(
                        (final_layernorm_name, w.detach().clone())
                    )
                elif is_vision_weights(name):
""",
        "LayerNorm weight routing",
    )

    source = _replace_once(
        source,
        """                    default_weight_loader(param, w)
            self.vision_model.load_weights(vision_weights)
""",
        """                    default_weight_loader(param, w)
            for trimmed_name, w in final_layernorm_weights:
                param = final_layernorm_dict[trimmed_name]
                with torch.no_grad():
                    default_weight_loader(param, w)
                self._loaded_vision_final_layernorm_params.add(trimmed_name)
            if final_layernorm_weights and (
                self._loaded_vision_final_layernorm_params
                >= final_layernorm_dict.keys()
            ):
                if not self._vision_final_layernorm_enabled:
                    logger.info_once(
                        "Loaded and enabled checkpoint-backed RADIO final LayerNorm",
                        scope="global",
                    )
                self._vision_final_layernorm_enabled = True
            self.vision_model.load_weights(vision_weights)
""",
        "LayerNorm parameter loading",
    )

    compile(source, "nano_nemotron_vl.py", "exec")
    return source, True


def installed_model_path() -> Path:
    version = importlib.metadata.version("vllm")
    if version.split("+", 1)[0] != EXPECTED_VLLM_VERSION:
        raise RuntimeError(
            f"RADIO final-LayerNorm patch requires vLLM {EXPECTED_VLLM_VERSION}, "
            f"but this worker has {version}"
        )
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("Could not locate the installed vLLM package")
    return (
        Path(next(iter(spec.submodule_search_locations)))
        / "model_executor"
        / "models"
        / "nano_nemotron_vl.py"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        type=Path,
        help="Patch this source file instead of installed vLLM (validation only).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    target = args.target or installed_model_path()
    original = target.read_text()
    patched, changed = patch_source(original)
    if args.dry_run:
        print(
            f"[vllm-radio-ln] {'would patch' if changed else 'already patched'}: "
            f"{target}"
        )
        return 0
    if not changed:
        print(f"[vllm-radio-ln] already patched: {target}")
        return 0

    backup = target.with_suffix(target.suffix + ".nrl-pre-radio-layernorm")
    if not backup.exists():
        shutil.copy2(target, backup)
    mode = target.stat().st_mode
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=target.parent,
        prefix=f".{target.name}.",
        delete=False,
    ) as tmp:
        tmp.write(patched)
        tmp_path = Path(tmp.name)
    os.chmod(tmp_path, mode)
    os.replace(tmp_path, target)
    print(f"[vllm-radio-ln] patched vLLM {EXPECTED_VLLM_VERSION}: {target}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[vllm-radio-ln] ERROR: {exc}", file=sys.stderr)
        raise
