# SpeechLM GSPO

The registered `asr` processor reads a consumer Parquet row with an
`audio_filepath` (or `audio_path`), an `instruction` (or `prompt`), and a separate
`reference` (or `expected_answer`). Audio stays at its consumer path and must be
nonempty 16-kHz audio. The processor renders one user message containing the
instruction, a newline, and `<|audio|>`, with no system message. The reference is
reward metadata and never enters the prompt. Declared `input_order` must be
`text_before_audio`. Existing Parquet consumers should retain their `messages`
column so ResponseDataset preserves the metadata columns.

The recipe consumes a prepared dataset; it does not mine or rebalance audio.
For a faithful reproduction, keep the original seed-conditioned consumer and
record its source seed/checkpoint, mining selection criteria, component/task/
language balances and exact prompt frequencies. Record the resulting consumer
hash and row/audio counts. Audio-level benchmark exclusion needs a separate
versioned overlap audit against the evaluation manifests; distinct row IDs or
paths alone do not establish exclusion. Keep any unverified exclusion claim
explicitly pending rather than changing the consumer during training.

Use `env_name: asr` even with the multimodal launcher: it keeps audio transport
and routes rewards through the ASR environment. Policy training receives
`audio_samples` and `audio_lengths`; the existing `vllm_audios` batch field
passes the same waveform and sample rate to rollout generation.

WER lowercases with Unicode casefold and separates punctuation, retaining
alphanumeric characters, apostrophes, underscores, and combining marks. It
does not apply NFKC. Reward is `1 - error`; an empty reference counts hypothesis
words, so negative rewards are valid. Mandarin ordinary ASR requires explicit
`reward_metric: cer` and `cer_whitespace_mode: han_adjacent`, using public
NeMo-Skills `evaluate_cer` with standard normalization. Other tasks use WER.

The ASR environment requires `rapidfuzz`; Mandarin CER additionally requires
public NeMo-Skills with the Han-adjacent whitespace option. Install these in the
driver/generation environment where rewards run. Policy training and canonical
export require Speech and Automodel but do not require Skills. The implementation
uses normal package installation and does not replace framework modules. A
prepared image can retain its existing actor interpreters and compiled
dependencies while installing public Speech and, where needed, Skills core with
constraints for each environment.

Optional `responses_create_params.max_output_tokens` is a positive integer; a
null value retains the rollout configuration cap. A request cap is clamped to
the global cap and remaining context by the standard vLLM worker.

The CER400 recipe offers 40 utterances with 64 responses each, retaining the
first 20 groups with nonconstant rewards in input order. It uses the standard
refill loop with `grpo.dynamic_sampling_max_gen_batches: 2`; groups are never
ranked by their reward spread. Advantages use the leave-one-out mean baseline
and the full group's sample standard deviation through
`grpo.adv_estimator.full_group_std: true`, with zero advantages for constant
groups. `policy.native_model.trainable_names` explicitly selects the original
269 FP32 trainable tensors (1,074,318,400 scalars) before optimizer construction;
all other seed tensors remain frozen, including native MTP weights and metadata.
MTP auxiliary loss and matched-evaluation speculation remain disabled.

The CER400 recipe is in `examples/configs/recipes/vlm/`. Its 20-update plus
two-update resume driver is in `tests/test_suites/vlm/`. Set
`SPEECHLM_GSPO_MODEL`, `SPEECHLM_GSPO_DATA`, `SPEECHLM_GSPO_OUTPUT_DIR` and
`SPEECHLM_GSPO_PYTHON` to the seed, consumer Parquet, output root and prepared
generation actor interpreter. The driver uses the existing multimodal entrypoint,
saves native weights/optimizer/dataloader state and verifies the restored cursor,
optimizer/scheduler state and exactly two resumed updates.
