# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Inline Hugging Face export for final Megatron checkpoints.

Implements the Megatron half of ``save_consolidated`` so that both training
backends can hand back an HF-format copy of the final checkpoint (the
Automodel/DTensor half lives in ``nemo_rl.models.automodel.checkpoint``).

Design constraints (from the #1494 discussion):
- strictly opt-in and off by default; only ``"final"`` is supported inline
  because converting inside the training job costs wall-clock time that
  scales with model size;
- the native Megatron checkpoint is always written and finalized first, so a
  preemption during the export only loses the export, never resumability;
- the conversion runs as a subprocess on rank 0 (``export_model_from_megatron``
  builds its own CPU/gloo distributed context, which must not share a process
  with the live NCCL training groups).

This module imports cleanly without Megatron so the validation and
orchestration logic is unit-testable on CPU-only machines; Megatron is only
imported inside the ``__main__`` entry point.
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_VALID_SAVE_CONSOLIDATED_VALUES = ("false", "final")


def validate_hf_export_config(megatron_cfg: dict[str, Any]) -> None:
    """Fail loudly on unsupported ``megatron_cfg.checkpoint.save_consolidated``.

    Called at worker startup so a misconfiguration surfaces before training
    instead of as a surprise after the final save.
    """
    checkpoint_cfg = megatron_cfg.get("checkpoint", None) or {}
    value = checkpoint_cfg.get("save_consolidated", None)
    if value is None:
        return
    if value not in _VALID_SAVE_CONSOLIDATED_VALUES:
        raise ValueError(
            "megatron_cfg.checkpoint.save_consolidated must be one of "
            f"{list(_VALID_SAVE_CONSOLIDATED_VALUES)}; got {value!r}. Note that "
            "YAML booleans (true/false) are not accepted -- quote the value. "
            "Inline export of intermediate checkpoints is not supported; use "
            "examples/converters/convert_megatron_to_hf.py offline instead."
        )
    use_peft = megatron_cfg.get("peft", {}).get("enabled", False)
    if value == "final" and use_peft:
        raise ValueError(
            "megatron_cfg.checkpoint.save_consolidated='final' is not supported "
            "for LoRA/PEFT runs because the Megatron checkpoint contains only "
            "adapter weights; export them offline with "
            "examples/converters/convert_lora_to_hf.py (merged or adapter-only)."
        )


def save_tokenizer_sidecar(tokenizer: Any, tokenizer_path: str) -> None:
    """Write the tokenizer next to the checkpoint.

    The algorithms already pass ``tokenizer_path`` for every backend; the
    Megatron worker used to drop it. A failing tokenizer write is logged and
    skipped: it must never take down a run at checkpoint time.
    """
    try:
        os.makedirs(tokenizer_path, exist_ok=True)
        tokenizer.save_pretrained(tokenizer_path)
    except Exception:
        logger.warning(
            "Failed to save tokenizer sidecar to %s; continuing without it.",
            tokenizer_path,
            exc_info=True,
        )


def latest_iteration_dir(weights_path: str) -> Optional[str]:
    """Return the newest ``iter_*`` directory under ``weights_path``, if any."""
    try:
        entries = [e for e in os.listdir(weights_path) if e.startswith("iter_")]
    except OSError:
        return None
    if not entries:
        return None
    # Iteration directories are zero-padded, so lexicographic order matches
    # numeric order and max() picks the most recent save.
    return os.path.join(weights_path, max(entries))


def run_hf_export_subprocess(
    hf_model_name: str,
    ckpt_path: str,
    output_path: str,
    tokenizer_dir: Optional[str] = None,
    *,
    runner: Callable[..., Any] = subprocess.run,
) -> None:
    """Convert one Megatron checkpoint directory to HF format in a subprocess.

    The subprocess isolates ``export_model_from_megatron``'s temporary gloo
    context from the live NCCL training process group.
    """
    argv = [
        sys.executable,
        "-m",
        "nemo_rl.models.megatron.hf_export",
        "--hf-model-name",
        hf_model_name,
        "--megatron-ckpt-path",
        str(ckpt_path),
        "--hf-ckpt-path",
        str(output_path),
    ]
    if tokenizer_dir is not None:
        argv += ["--tokenizer-dir", str(tokenizer_dir)]
    result = runner(argv, capture_output=True, text=True)
    if result.returncode != 0:
        stderr_tail = (result.stderr or "")[-2000:]
        raise RuntimeError(
            f"HF export subprocess failed with exit code {result.returncode} "
            f"for checkpoint {ckpt_path}:\n{stderr_tail}"
        )


def export_final_hf_checkpoint(
    hf_model_name: str,
    weights_path: str,
    tokenizer_dir: Optional[str] = None,
    *,
    is_rank0: bool,
    barrier: Optional[Callable[[], None]] = None,
    run_subprocess: Optional[Callable[..., None]] = None,
) -> None:
    """Export the final checkpoint to HF format; warn-only on failure.

    Every rank must call this so they resynchronize on ``barrier`` after rank
    0 finishes (or fails) -- the export must never desynchronize the training
    job that is about to tear down. The native checkpoint is already on disk
    and final when this runs, so a failed export only costs the HF copy.
    """
    run_subprocess = run_subprocess or run_hf_export_subprocess
    if is_rank0:
        iter_dir = latest_iteration_dir(weights_path)
        if iter_dir is None:
            logger.warning(
                "No iter_* checkpoint directory found under %s; skipping "
                "inline HF export.",
                weights_path,
            )
        else:
            output_path = os.path.join(
                os.path.dirname(os.path.abspath(weights_path)), "hf_export"
            )
            logger.warning(
                "save_consolidated='final': starting inline HF export of %s. "
                "This runs a CPU conversion whose duration scales with model "
                "size and blocks checkpoint finalization until it completes; "
                "examples/converters/convert_megatron_to_hf.py is the offline "
                "alternative.",
                iter_dir,
            )
            try:
                run_subprocess(
                    hf_model_name=hf_model_name,
                    ckpt_path=str(iter_dir),
                    output_path=output_path,
                    tokenizer_dir=tokenizer_dir,
                )
                logger.info("Inline HF export written to %s", output_path)
            except Exception:
                logger.warning(
                    "Inline HF export of %s failed; the native Megatron "
                    "checkpoint is unaffected and can be converted offline.",
                    iter_dir,
                    exc_info=True,
                )
    if barrier is not None:
        barrier()


def main(argv: Optional[list[str]] = None) -> None:
    """Subprocess entry point: convert one Megatron checkpoint to HF format.

    Writes into a temporary sibling directory and renames into place so a
    crashed conversion never leaves a partial ``hf_export`` directory behind.
    """
    parser = argparse.ArgumentParser(
        description="Convert a Megatron checkpoint directory to HF format."
    )
    parser.add_argument("--hf-model-name", required=True)
    parser.add_argument("--megatron-ckpt-path", required=True)
    parser.add_argument("--hf-ckpt-path", required=True)
    parser.add_argument(
        "--tokenizer-dir",
        default=None,
        help="Optional tokenizer directory to copy into the HF checkpoint.",
    )
    args = parser.parse_args(argv)

    output_path = os.path.abspath(args.hf_ckpt_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tmp_output = tempfile.mkdtemp(
        dir=os.path.dirname(output_path),
        prefix=os.path.basename(output_path) + ".tmp.",
    )
    try:
        # Imported here so this module (and its unit tests) stay importable
        # without the mcore extra.
        from nemo_rl.models.megatron.community_import import (
            export_model_from_megatron,
        )

        export_model_from_megatron(
            hf_model_name=args.hf_model_name,
            input_path=args.megatron_ckpt_path,
            output_path=tmp_output,
            hf_tokenizer_path=args.tokenizer_dir or args.hf_model_name,
            overwrite=True,
        )
        if args.tokenizer_dir and os.path.isdir(args.tokenizer_dir):
            shutil.copytree(
                args.tokenizer_dir,
                os.path.join(tmp_output, "tokenizer"),
                dirs_exist_ok=True,
            )
        # Raises if the target already exists: a promoted export is never
        # silently overwritten.
        os.rename(tmp_output, output_path)
    except BaseException:
        shutil.rmtree(tmp_output, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
