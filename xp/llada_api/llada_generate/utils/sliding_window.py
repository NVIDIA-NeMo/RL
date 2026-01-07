# update block_slice_index (sliding window without reallocations)
if (
    (int(first_col_global.item()) > (sliding_window_threshold - 1))
    and (current_block_start + sliding_window_size + block_length) <= (prompt_len + gen_length)
):
    # update KV cache with truncated block
    update_slice = slice(current_block_start, current_block_start + sliding_window_size)
    output = model(
        x_accum[:, update_slice],
        past_key_values=past_key_values,
        use_cache=True
    )
    past_key_values = output.past_key_values
    # refresh context-next logit for the next block
    next_logits_context = output.logits[:, -1:, :]

    # advance current_block_index in-place (no cat, x_accum is already preallocated)
    current_block_start = current_block_start + sliding_window_size
    block_slice = slice(current_block_start, current_block_start + block_length)
    if dbg_print:
        print(f'Updated block_slice: {block_slice}')