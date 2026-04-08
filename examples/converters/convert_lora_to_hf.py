"""Export a Megatron LoRA adapter checkpoint to HuggingFace format.

This script supports two workflows:

1. Merge the base model and LoRA adapter, then export a standard HuggingFace model.
2. Export only the LoRA adapter to a HuggingFace PEFT-compatible directory without merging.

Usage (requires mcore extra):

    # Export adapter only (recommended when you want PEFT format)
    uv run --extra mcore python examples/converters/convert_lora_to_hf.py \
        --adapter-only \
        --adapter-ckpt results/dpo_glm5/step_5/policy/weights/iter_0000000 \
        --hf-model-name zai-org/GLM-5 \
        --hf-ckpt-path ./hf_lora_adapter

    # Merge base model + adapter and export a full HF checkpoint
    uv run --extra mcore python examples/converters/convert_lora_to_hf.py \
        --base-ckpt ~/.cache/huggingface/nemo_rl/zai-org/GLM-5/iter_0000000 \
        --adapter-ckpt results/dpo_glm5/step_5/policy/weights/iter_0000000 \
        --hf-model-name zai-org/GLM-5 \
        --hf-ckpt-path ./merged_hf_model

"""

import argparse
import gc
import logging
import os
import sys

import yaml

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export Megatron LoRA checkpoint to HuggingFace format"
    )
    parser.add_argument(
        "--base-ckpt",
        type=str,
        default=None,
        help="Path to base model Megatron checkpoint (iter_XXXXXXX directory). Required unless --adapter-only is set.",
    )
    parser.add_argument(
        "--adapter-ckpt",
        type=str,
        required=True,
        help="Path to LoRA adapter Megatron checkpoint (iter_XXXXXXX directory)",
    )
    parser.add_argument(
        "--hf-model-name",
        type=str,
        required=True,
        help="HuggingFace model name (e.g. zai-org/GLM-5)",
    )
    parser.add_argument(
        "--hf-ckpt-path",
        type=str,
        required=True,
        help="Output path for the exported HF checkpoint or adapter directory",
    )
    parser.add_argument(
        "--adapter-only",
        action="store_true",
        help="Export only the LoRA adapter in HuggingFace PEFT format without merging into the base model.",
    )
    args = parser.parse_args()
    if not args.adapter_only and not args.base_ckpt:
        parser.error("--base-ckpt is required unless --adapter-only is set")
    return args


def export_lora_adapter_to_hf(
    adapter_ckpt: str,
    hf_model_name: str,
    hf_ckpt_path: str,
) -> str:
    """Export a Megatron LoRA checkpoint to HuggingFace PEFT adapter format.

    Args:
        adapter_ckpt: Path to the LoRA adapter Megatron checkpoint (iter_XXXXXXX directory).
        hf_model_name: HuggingFace model identifier for the base model.
        hf_ckpt_path: Output directory for the HuggingFace PEFT adapter files.

    Returns:
        The *hf_ckpt_path* that was written to.

    Raises:
        FileExistsError: If *hf_ckpt_path* already exists.
    """
    if os.path.exists(hf_ckpt_path):
        raise FileExistsError(f"Output path already exists: {hf_ckpt_path}")

    from megatron.bridge import AutoBridge

    bridge = AutoBridge.from_hf_pretrained(hf_model_name, trust_remote_code=True)
    logger.info("Exporting LoRA adapter in HuggingFace PEFT format...")
    bridge.export_adapter_ckpt(adapter_ckpt, hf_ckpt_path)
    logger.info(f"Done! HF adapter saved to: {hf_ckpt_path}")
    return hf_ckpt_path


def merge_lora_to_hf(
    base_ckpt: str,
    adapter_ckpt: str,
    hf_model_name: str,
    hf_ckpt_path: str,
) -> str:
    """Merge a Megatron LoRA adapter with its base model and export to HuggingFace format.

    Args:
        base_ckpt: Path to the base model Megatron checkpoint (iter_XXXXXXX directory).
        adapter_ckpt: Path to the LoRA adapter Megatron checkpoint (iter_XXXXXXX directory).
                      Must contain a ``run_config.yaml`` with a ``peft`` section.
        hf_model_name: HuggingFace model identifier (e.g. ``zai-org/GLM-5``).
        hf_ckpt_path: Output directory for the merged HuggingFace checkpoint.

    Returns:
        The *hf_ckpt_path* that was written to.

    Raises:
        FileExistsError: If *hf_ckpt_path* already exists.
        ValueError: If the adapter's ``run_config.yaml`` has no ``peft`` section.
    """
    if os.path.exists(hf_ckpt_path):
        raise FileExistsError(f"Output path already exists: {hf_ckpt_path}")

    from megatron.bridge import AutoBridge
    from megatron.bridge.peft.lora import LoRA
    from megatron.bridge.training.checkpointing import (
        _generate_model_state_dict,
        _load_model_weights_from_checkpoint,
        apply_peft_adapter_filter_to_state_dict,
    )
    from megatron.bridge.training.model_load_save import (
        load_model_config,
        megatron_cpu_init_context,
        temporary_distributed_context,
    )
    from megatron.core import dist_checkpointing

    bridge = AutoBridge.from_hf_pretrained(hf_model_name, trust_remote_code=True)

    adapter_run_config_path = os.path.join(adapter_ckpt, "run_config.yaml")
    with open(adapter_run_config_path) as f:
        adapter_run_config = yaml.safe_load(f)

    peft_section = adapter_run_config.get("peft")
    if peft_section is None:
        raise ValueError(f"No 'peft' section found in {adapter_run_config_path}")

    logger.info(
        f"LoRA config: dim={peft_section.get('dim')}, alpha={peft_section.get('alpha')}"
    )

    with temporary_distributed_context(backend="gloo"):
        from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed

        model_parallel_cuda_manual_seed(0)

        model_cfg, _ = load_model_config(adapter_ckpt)
        model_cfg.tensor_model_parallel_size = 1
        model_cfg.pipeline_model_parallel_size = 1
        model_cfg.num_layers_in_first_pipeline_stage = None
        model_cfg.num_layers_in_last_pipeline_stage = None
        model_cfg.context_parallel_size = 1
        model_cfg.expert_model_parallel_size = 1
        model_cfg.expert_tensor_parallel_size = 1
        model_cfg.moe_extended_tp = False
        model_cfg.sequence_parallel = False
        model_cfg.perform_initialization = False
        model_cfg.virtual_pipeline_model_parallel_size = None
        model_cfg.hierarchical_context_parallel_sizes = None
        model_cfg.fp8 = None
        model_cfg.fp8_param = False

        peft = LoRA(
            target_modules=peft_section.get("target_modules", []),
            exclude_modules=peft_section.get("exclude_modules", []),
            dim=peft_section["dim"],
            alpha=peft_section["alpha"],
            dropout=peft_section.get("dropout", 0.0),
            dropout_position=peft_section.get("dropout_position", "pre"),
            lora_A_init_method=peft_section.get("lora_A_init_method", "xavier"),
            lora_B_init_method=peft_section.get("lora_B_init_method", "zero"),
            a2a_experimental=peft_section.get("a2a_experimental", False),
        )

        logger.info(
            "Building base model on CPU (LoRA wrappers applied after base weights are loaded)..."
        )
        if hasattr(model_cfg, "finalize"):
            model_cfg.finalize()
        with megatron_cpu_init_context(model_cfg):
            megatron_model = model_cfg.provide_distributed_model(
                wrap_with_ddp=False,
                use_cpu_initialization=True,
            )
        gc.collect()

        for m in megatron_model:
            m.requires_grad_(False)

        logger.info(f"Loading base model from {base_ckpt}...")
        _load_model_weights_from_checkpoint(base_ckpt, megatron_model, strict=False)
        gc.collect()

        logger.info("Applying LoRA wrappers to model...")
        megatron_model = peft(megatron_model, training=False)
        gc.collect()

        logger.info(f"Loading LoRA adapter from {adapter_ckpt}...")
        adapter_sharded_state_dict = _generate_model_state_dict(megatron_model, {})
        adapter_sharded_state_dict = apply_peft_adapter_filter_to_state_dict(
            adapter_sharded_state_dict, peft
        )
        loaded_adapter_state_dict = dist_checkpointing.load(
            adapter_sharded_state_dict, adapter_ckpt
        )
        model_key = (
            "model"
            if "model" in loaded_adapter_state_dict
            else next(k for k in loaded_adapter_state_dict if k.startswith("model"))
        )
        for m in megatron_model:
            m.load_state_dict(loaded_adapter_state_dict[model_key], strict=False)
        gc.collect()

        logger.info("Saving merged model in HuggingFace format...")
        bridge.save_hf_pretrained(
            megatron_model,
            hf_ckpt_path,
            strict=False,
            merge_adapter_weights=True,
        )

        del megatron_model
        gc.collect()
        logger.info("Freed model memory.")
        sys.stderr.flush()
        sys.stdout.flush()

    logger.info(f"Done! Merged HF model saved to: {hf_ckpt_path}")
    return hf_ckpt_path


def main():
    args = parse_args()
    if args.adapter_only:
        export_lora_adapter_to_hf(
            adapter_ckpt=args.adapter_ckpt,
            hf_model_name=args.hf_model_name,
            hf_ckpt_path=args.hf_ckpt_path,
        )
    else:
        merge_lora_to_hf(
            base_ckpt=args.base_ckpt,
            adapter_ckpt=args.adapter_ckpt,
            hf_model_name=args.hf_model_name,
            hf_ckpt_path=args.hf_ckpt_path,
        )


if __name__ == "__main__":
    main()
