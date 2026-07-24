# Native NeMo-RL 100-step reproduction V3 — final report

**Verdict: evidence integrity and exact previous-campaign preprocessing both pass, but the predeclared 0.1 pairwise loss-gap gate fails narrowly in each mode: no-clip adapter D at step 8, and clip=1 adapter C at step 7.**

## Scope

This battery compares each genuine standalone single-LoRA run with the corresponding slot in one four-adapter native NeMo-RL Multi-LoRA run:

- true single A ↔ multi slot A
- true single B ↔ multi slot B
- true single C ↔ multi slot C
- true single D ↔ multi slot D

It uses the exact previous nousnet campaign preprocessing recovered from the effective runtime pipeline: tokenizer model-default formatting plus `data.default.prompt_file: examples/prompts/cot.txt`. It does not compare an aggregate multi-adapter loss with individual singles.

## Provenance

Branch: `phuc/multilora-native-nemorl`

Config/submit commit: `b9c87f145` (`native100 v3: restore campaign COT preprocessing explicitly`)

No clipping:

- multi: SLURM 251641
- single A/B/C/D: 251642 / 251643 / 251644 / 251645

Independent per-adapter clipping at max norm 1.0:

- multi: SLURM 251646
- single A/B/C/D: 251647 / 251648 / 251649 / 251650

Previous campaign comparator:

- no clip: multi 245966; singles A/B/C/D 245174 / 245967 / 245158 / 245159
- clip=1: multi 245160; singles A/B/C/D 245161 / 245162 / 245163 / 245164

## Integrity and freshness

All ten V3 jobs independently pass the strict fresh-run validator:

- terminal optimizer step 100 was observed;
- eight data-parallel trace files exist per job;
- each single contains exactly 800 rows;
- each multi contains exactly 3,200 rows;
- no terminal step-100 checkpoint was loaded;
- every log contains an anchored `SFT_DONE_OK` marker.

Every native single also matches its corresponding previous-campaign input on 800/800 `(step, rank)` rows by exact SHA-256 and valid-token count. Thus `PREPROCESSING_PARITY=PASS` in both modes.

SLURM records exit code 15 after completion because of the known Ray teardown race. The payload evidence above, rather than scheduler accounting, is the completion authority.

Logs print `CLEAN: nousnet not importable` for all jobs. Multi logs additionally print `multi_lora: enabled with 4 adapters`, proving that the shared runs used the vendored NeMo-RL-native implementation without a hidden nousnet dependency.

## Pairwise results

### No gradient clipping

- A: PASS — mean |Δloss| 0.006060; maximum 0.029767 at step 29; final 0.001741.
- B: PASS — mean 0.006463; maximum 0.045701 at step 33; final 0.008788.
- C: PASS — mean 0.009650; maximum 0.032360 at step 7; final 0.007575.
- D: FAIL — mean 0.011286; one isolated violation, 0.108475 at step 8; final 0.029218.

Overall predeclared verdict: `PAIRWISE FAIL` (3/4 adapters pass).

### Independent per-adapter clipping, max norm 1.0

- A: PASS — mean |Δloss| 0.009694; maximum 0.091448 at step 45; final 0.016059.
- B: PASS — mean 0.009386; maximum 0.056355 at step 12; final 0.000277.
- C: FAIL — mean 0.011472; one isolated violation, 0.178175 at step 7; final 0.008427.
- D: PASS — mean 0.008943; maximum 0.032748 at step 39; final 0.014738.

Overall predeclared verdict: `PAIRWISE FAIL` (3/4 adapters pass).

## Interpretation

These are magnitude failures, not integrity failures. Each failing adapter has exactly one isolated threshold excursion, and both recover afterward:

- no-clip D: only step 8 is at or above 0.1; 92/100 steps are below 0.02 and 97/100 below 0.05;
- clip=1 C: only step 7 is at or above 0.1; 90/100 steps are below 0.02 and 98/100 below 0.05.

The residual scale is consistent with the previous campaign's measured short-horizon reproducibility floor rather than sustained drift. In the previous campaign, no-clip D had two isolated violations (steps 8 and 46, maximum 0.167969), while clipped A/B/C/D each had one isolated violation. V3 means are 0.0061–0.0115 versus campaign means 0.0074–0.0150, and all pairs are exact at step 1.

This does not relabel the declared gate as a pass. The bounded conclusion is: the native NeMo-RL port reproduces the previous campaign's exact inputs and the same spike-not-drift numerical behavior, while two pairwise cells formally fail the strict `<0.1 at every step` yardstick.

## Artifacts

- `results/native100v3_noclip_pairwise.{txt,json,csv,png}`
- `results/native100v3_clip1_pairwise.{txt,json,csv,png}`
- `results/native100v3_noclip_vs_campaign.txt`
- `results/native100v3_clip1_vs_campaign.txt`
- `results/native100v3_noclip_loss_with_delta.png`
- `results/native100v3_clip1_loss_with_delta.png`

The dedicated V3 operator and completion watcher are no longer armed. No user action is pending.
