"""Token-prefix helpers shared by the external vLLM endpoint plugin."""

from typing import Any


def replace_prefix_tokens(
    tokenizer: Any,
    model_prefix_token_ids: list[int],
    template_prefix_token_ids: list[int],
    template_token_ids: list[int],
) -> list[int]:
    """Preserve exact prior model tokens while retaining the rendered suffix.

    Keep this implementation behaviorally identical to
    ``nemo_rl.models.generation.openai_server_utils.replace_prefix_tokens``.
    The small duplication makes the plugin independently installable in the
    official vLLM image, where the NeMo RL package is not present.
    """
    if not model_prefix_token_ids:
        return template_token_ids

    eos_token_id = tokenizer.eos_token_id
    assert eos_token_id is not None, "Tokenizer must have an EOS token ID"

    model_cut_end = len(model_prefix_token_ids)
    if model_prefix_token_ids[-1] == eos_token_id:
        model_cut_end -= 1

    count_needed = template_prefix_token_ids.count(eos_token_id)
    count_seen = 0
    template_cut_start = -1
    for pos, token_id in enumerate(template_token_ids):
        if token_id == eos_token_id:
            count_seen += 1
            if count_seen == count_needed:
                template_cut_start = pos
                break

    assert template_cut_start >= 0, (
        f"EOS token #{count_needed} not found in template_token_ids "
        f"(only found {count_seen} EOS tokens total)!\n"
        "Template prefix token IDs (everything before the final assistant "
        f"message): {template_prefix_token_ids}\n\n"
        "Template token IDs (everything that was sent to the model endpoint): "
        f"{template_token_ids}\n\n"
        "Template prefix repr (detokenized): "
        f"{tokenizer.decode(template_prefix_token_ids)!r}\n\n"
        f"Template repr (detokenized): {tokenizer.decode(template_token_ids)!r}"
    )

    return (
        model_prefix_token_ids[:model_cut_end] + template_token_ids[template_cut_start:]
    )
