# Training on text that contains `<image>`

## Summary

A media placeholder is an ordinary vocabulary entry, so text can legitimately
contain it — a competitive-programming statement that spells `<image>` in its
prose, for instance. The model reads every occurrence as an anchor for a
projected image feature and fails when no feature exists, so those rows cannot
be trained on.

This design adds a **media-token validity mask**: the caller, which knows how
many media items each row carries, marks which placeholder positions are real
anchors. Positions it does not mark are skipped by the media merge and keep
whatever language embedding the forward gave them.

## The problem

`NemotronOmniModel._merge_projected_media` enforces a strict 1:1 contract — one
projected feature per valid placeholder:

```python
media_mask = input_ids == media_token_id
if attention_mask is not None:
    media_mask = media_mask & attention_mask.bool()

expected_features = int(media_mask.sum().item())
actual_features = media_embeddings.shape[0]
if expected_features != actual_features:
    raise ValueError("Expanded-sequence media alignment failed: ...")
```

The contract is right. The question is where `media_mask` comes from. Before
this change the model derived it from whichever mask it had:

```python
media_token_validity_mask = None
if padding_mask is not None:
    media_token_validity_mask = ~padding_mask
elif attention_mask is not None and attention_mask.dim() == input_ids.dim():
    media_token_validity_mask = attention_mask
```

Both of those answer **"is this a real token?"**. The merge needs **"is this a
media anchor?"**. Those coincide only while every media token in a non-padding
position anchors an image. A text row that spells the placeholder breaks the
equivalence: the position is a real token, so the derived mask marks it valid,
so the merge demands a feature that was never meant to exist.

This is not hypothetical. In a 100k-row sample of a Nemotron post-training
blend, 2,539 rows (2.5%) contain a literal `<image>` with **zero** attached
images — competitive-programming statements where `<image>` replaced inline
math, e.g. `"for any character <image> there is exactly one character"`. They
carry between 1 and 12 placeholders each.

There is a second, subtler source. Chat templates that treat a literal
`<image>` in the prose as the placement anchor suppress their own generated
image block. A row with two literal tokens and one attached image therefore
renders two placeholders for one feature.

## Why not sanitize

The previous approach rewrote the data at rollout time: drop the literal token
when images were attached, or replace it with the word `image` when not.

It worked, but it had three problems:

1. **It edits the user's prose.** `"for any character <image> there is"` became
   `"for any character image there is"`. The model trains on text the author
   did not write.
2. **It hides malformed data.** A row with two placeholders and one image is a
   defect; sanitizing silently patched it. Filtering the blend surfaced 392 such
   rows, which turned out to be unanswerable questions.
3. **It cannot express the legitimate case.** There is no rewrite that both
   preserves the prose and tells the model "this one is not an anchor".

## Design

Give the caller a way to state the answer directly, since the caller is the only
party that knows it. The model keeps its strict contract; it just stops guessing
the input to that contract.

### Model change

`NemotronOmniModel.forward` takes a keyword-only argument that takes precedence
over the derived masks:

```python
if media_token_validity_mask is None:
    if padding_mask is not None:
        media_token_validity_mask = ~padding_mask
    elif attention_mask is not None and attention_mask.dim() == input_ids.dim():
        media_token_validity_mask = attention_mask
```

Behavior is unchanged when the argument is omitted. This is the only change to
the model, and it is additive.

### Building the mask

`build_media_token_validity_mask(input_ids, media_token_id, media_counts_by_row)`
marks media-token positions in rows whose media count is zero:

- Every row carries media → returns `None`; the model derives its own mask.
- Text rows exist but none spell the token → returns `None` for the same reason.
- Otherwise → a `[B, S]` bool mask, `False` at media-token positions of
  media-less rows.

Rows that **do** carry media keep every position valid. A genuine
placeholder/feature disagreement there is still reported rather than masked
away, so the mask cannot be used to silence real misalignment.

### Carrying it through sequence packing

This is the part that determines whether the mask means anything.

The mask is built in **sample space**, where row `i` of `input_ids` pairs with
`media_counts_by_row[i]`. Sequence packing then concatenates many samples into
one THD sequence and context-parallel-shards it. After that, one "row" holds
many samples and each rank holds a slice — the per-row question the mask answers
can no longer be asked.

So the mask must travel through the *same* transform as `input_ids` rather than
be derived downstream. It is packed alongside them, exactly as `mtp_loss_mask`
already is:

```python
if "media_token_validity_mask" in data_dict:
    packed_media_mask, local_media_mask, _, _, _ = _pack_sequences_for_megatron(
        data_dict["media_token_validity_mask"].to(data_dict["input_ids"].dtype),
        seq_lengths,
        pad_individual_seqs_to_multiple_of,
        pad_packed_seq_to_multiple_of,
        pad_full_seq_to,
        cp_rank=get_context_parallel_rank(),
        cp_size=get_context_parallel_world_size(),
    )
    media_token_validity_mask = (
        packed_media_mask
        if model_slices_context_parallel_inputs
        else local_media_mask
    ).bool()
```

Two details matter:

- **Packed in token dtype.** Packing pads with `value=0`, which is a valid token
  id but not a valid bool. It is converted back to bool after packing. Padding
  positions become `False`, which is harmless because padding holds no media
  token.
- **Packed vs CP-local.** A model that slices context parallelism itself
  receives the full THD row so it can insert media before selecting its
  CP-owned embeddings. Its mask must stay unsharded to line up. Every other
  model consumes the CP-local shard. This mirrors the `mtp_loss_mask` choice
  for the same reason. `NemotronOmniModel` sets
  `model_slices_context_parallel_inputs = True`, so it gets the full row.

### Capability detection

The mask is only sent to models whose `forward` actually declares it:

```python
def _model_accepts_media_token_validity_mask(model) -> bool:
    ...
    if "media_token_validity_mask" in inspect.signature(chunk.forward).parameters:
        return True
```

The check is on the signature rather than a class flag because a model that does
not know about the mask would absorb it into `**kwargs` and ignore it — which
looks identical to the mask having been applied. Failing to send it is visible;
sending it into a void is not.

### Where it is attached

At all three forward sites: training and **both** logprob paths. Logprobs run
the same forward as training, so a batch that needs the mask needs it there too
— otherwise logprobs would be computed against a different media alignment than
the one trained on.

### Self-packing models

Models that pack internally (`delegate_pack_to_model`) raise
`NotImplementedError` rather than silently dropping the mask. A mask built
against caller-side rows would reach the merge in a layout that no longer
matches its tokens, and a misaligned media mask attaches features to the wrong
positions without erroring.

## Consequences

**Removed.** `sanitize_nemo_gym_example_image_placeholders`,
`_normalize_image_placeholders`, `_count_image_payloads`, the
`sanitize_image_placeholders` config flag, and their tests — 212 lines.

**Data prep is now required, not optional.** Without the sanitizer there is no
safety net: a blend whose placeholder and image counts disagree fails loudly at
the media merge. That is the intended behavior — the failure is a real defect —
but it means malformed rows must be filtered before training rather than being
absorbed at rollout time.

**Text rows containing `<image>` can be trained on** instead of dropped, which
is what the mask exists for.

## Validation

| Test | Result |
|---|---|
| 50-step Super Omni run, sanitization removed | 50/50 steps, zero media-alignment / device / IMA errors; reward 0.458 (steps 1–8) → 0.520 (steps 26–49), peak 0.5879 vs. a 0.4932 baseline best |
| Mixed batch: 128 rows with real images + 128 text-only rows carrying real `<image>` statements | Reached Step 1/1, rollouts 100%, zero alignment failures |
| Unit: mask construction | 6 tests |
| Unit: mask survives packing, lands on the same tokens | 2 tests (require mcore) |

The packing tests are the load-bearing ones: they assert the mask still marks
the intended tokens after `_pack_sequences_for_megatron`, because a mask that is
merely misaligned does not raise — it attaches features to the wrong positions.
