def is_lora_weight_name(name: str) -> bool:
    """Return True if a parameter name corresponds to a LoRA weight."""
    return (
        name.endswith(".lora_A.weight")
        or name.endswith(".lora_B.weight")
        or name.endswith(".lora_scaling.weight")
    )


def is_base_model_weight_name(name: str) -> bool:
    """Return True if a parameter name corresponds to a base (non-LoRA) weight."""
    return not is_lora_weight_name(name)
