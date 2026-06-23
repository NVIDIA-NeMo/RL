# DiffuGRPO / CoupledGRPO Instability — Identified Gaps vs Reference Implementations

Investigation summary: why our NeMo-RL **DiffuGRPO** and **CoupledGRPO** runs collapse during
RL, and how they differ from the published reference implementations
(**d1**, **DiffuCoder Coupled-GRPO**, **MMaDA UniGRPO**) and from our own stable
**BlockJustGRPO**.

Confidence tags: **[data]** = measured from our run dumps; **[code]** = confirmed by
reading reference source; **[hypothesis]** = inferred, not yet proven.

---

## 0. TL;DR

- Both DiffuGRPO and CoupledGRPO **collapse**: validation accuracy and train reward decay,
  then fall to ~0 with generations degenerating to gibberish. BlockJustGRPO (leftmost
  block-reveal) trains cleanly on the *same recipe*. **[data]**
- The collapse is an **entropy blow-up** (distribution flattening), driven by a
  **negative-advantage-dominated gradient** (sub-50% pass rate) with **no restoring force**
  (KL=0, full fine-tune). **[data]**
- We have **two independent classes of gaps** vs the references:
  1. **Recipe / stabilizer gaps** — affect both DiffuGRPO and CoupledGRPO.
  2. **Logprob-estimator gaps** — specific to CoupledGRPO (its estimator is *biased*).

---

## 1. The collapse, characterized [data]

- Coupled (token-level) collapses ~step 50; seqloss (sequence-level loss) delays it to ~step 120;
  +KL=0.01 alone (with length bias) still collapsed. None *prevent* it.
- Two phases:
  - **Phase 1 (slow bleed):** coherent-but-wrong answers, normal length, reward drifts down.
  - **Phase 2 (runaway):** length explodes then crashes; output becomes rare-token gibberish
    (e.g. `telefon impre Ba Sadd ... boxed{6}`), reward ~0.
- `prev_logprob` (the policy's confidence in its *own* sampled tokens) slides from ~-0.5 to ~-8
  across the collapse -> the distribution is flattening toward uniform (entropy rising), not
  peaking. (`approx_entropy` in W&B is the direct readout: H(pi_curr); it climbs.)
  
## 3. Recipe / stabilizer gaps (both DiffuGRPO and CoupledGRPO)

vs **DiffuCoder Coupled-GRPO** `recipes/config_coupled_code.yaml` **[code]** and **d1** **[code]**:

| knob | DiffuCoder | d1 | Ours | gap |
| --- | --- | --- | --- | --- |
| KL penalty `beta` | 0.01 | > 0 | **0.0** | no KL anchor |
| reference model | **EMA-synced** to reference policy every 64 steps (`sync_ref_model`) | frozen ref + LoRA | **none** | no restoring force |
| LoRA / adapter | full FT | **LoRA** (rank-constrained) | full FT | d1 can't drift far; we can |
| advantage std-norm | `scale_rewards: false` | (n/a) | `normalize_rewards: true` (/std) | we over-weight low-variance prompts (Dr-GRPO bias) |
| `max_grad_norm` | 0.2 | - | 1.0 | 5x looser clip |
| loss length-norm | - | - | `token_level_loss: true` (length-biased) | fixed by our seqloss variant; also a Dr-GRPO bias |
| learning rate | 1e-6 | - | 3e-7 | - |
| PPO clip `epsilon` | 0.5 | set (value n/a) | 0.2 | ours clips ~2.5x tighter |
| inner updates `num_iterations` (mu) | 2 | >1 | 1 | ours is single-step / fully on-policy |

Key points:
- **No KL + no synced/EMA reference**: DiffuCoder leashes the policy to a slow-moving copy of
  itself (moving trust region); d1 leashes via LoRA + frozen-ref KL. We have **neither** -> the
  entropy blow-up is unconstrained. A plain fixed-KL=0.01 run is NOT the same as KL-vs-EMA-ref.
- **std normalization (Dr-GRPO bias #2)**: dividing advantage by group std amplifies gradients
  on near-homogeneous (very easy/hard) groups. Fix = `grpo.normalize_rewards=false`. **[code:
  supported in `advantage_estimator.py`]**
- **length normalization (Dr-GRPO bias #1)**: token-level loss weights samples by length;
  wrong answers are ~50% longer -> net-negative gradient pressure. **[data]** Mitigated by our
  `token_level_loss=false` (seqloss) variant.
- **inner updates (mu): two-step vs single-step**: DiffuCoder (and d1) take **mu>1** gradient
  updates per rollout batch, reusing the generations with the PPO ratio against cached
  `old_logprobs` (mildly off-policy). We use **mu=1** (`num_iterations=1`): one update per
  rollout, fully on-policy, so `curr_logprobs == prev_logprobs` at the update and the
  importance ratio is ~1.
- **PPO clip `epsilon`**: DiffuCoder uses a **wider** clip (0.5) vs our **0.2**. Note this is
  coupled to mu: with our mu=1 the ratio is ~1 so the clip **never engages** -> epsilon is
  effectively inert for us today. It only starts to matter if we adopt mu>1, at which point
  DiffuCoder's wider 0.5 allows larger per-step moves (paired with their other anchors).

## 4. Logprob-estimator gaps — **CoupledGRPO specific** [code]

vs **DiffuCoder `coupled_grpo.py`**:

| aspect | DiffuCoder (reference) | Ours | gap |
| --- | --- | --- | --- |
| forward passes | 3: fully-masked + mask-`t` + mask-`(1-t)` | 2: mask-`t` + complement | missing fully-masked anchor |
| per-token combine | `(logp_fullymasked + weighted_coupled)/2` | sum of 2 levels (token from its level) | no fully-masked term |
| **inverse-prob weighting** | **`1/t` and `1/(1-t)`** | **none** | **biased estimator** |
| masking ratio `t` | `U(0.2, 0.8)` bounded, shared per batch | `U(0,1)` full range, per-sample | extreme masks -> high variance |

Per token:
- DiffuCoder: `final_logp = [ logp_fullymasked + logp_coupled * (1/mask_rate) ] / 2`
- Ours:       `final_logp = logp_coupled`  (no `1/t`, no fully-masked term)

**Why the `1/t` term matters:** it is the diffusion-ELBO weight that makes the masked-token
logprob an *unbiased* estimate of the sequence likelihood. Without it we optimize the *raw*
masked logprob — a different objective. Our `gen_kl` looked fine only because that check uses
the **final_step verify forward**, NOT the 2-level coupled estimate that actually feeds the
gradient.

## 5. The constant-vs-variable masking insight [code + hypothesis]

| method | masking rate | masking-ratio correction | status |
| --- | --- | --- | --- |
| **d1 / our DiffuGRPO** | **constant** (completion fully masked) | not needed (`1/t` is uniform) | estimator OK |
| **DiffuCoder coupled** | variable (`t`, `1-t`, bounded) | explicit `1/t` weighting | reference |
| **MMaDA UniGRPO** | variable (ratio sampled from a range) | "modified log-likelihood approximation" | reference |
| **Ours (coupled)** | variable (`t in [0,1]`) | **none** | **biased** |

- Every reference using **variable** masking includes a masking-ratio-aware log-likelihood
  (DiffuCoder = `1/t`; MMaDA UniGRPO = modified LL). The only one that can skip it is **d1**,
  because its masking rate is **constant** (fully masked). 
- Therefore: **our DiffuGRPO estimator is correct** (constant mask, matches d1) -> its collapse
  is a **recipe** problem. **Our CoupledGRPO estimator is biased** (variable mask, no
  correction) **plus** the same recipe gaps -> it is the worse offender.


## 6. Plan — order of further improvements

Agreed order (recipe first, then logprob gaps):

1. **Phase 1 — recipe stabilizers** (DONE / under test in coupled_grpo_seqloss_recipe):
   KL=0.01 (+reference forward on), `normalize_rewards=false`, `max_grad_norm=0.2`,
   masking `t in [0.2, 0.8]`, on the seqloss base (`token_level_loss=false`).
2. **Phase 2 — fix the CoupledGRPO logprob estimator** (highest-value coupled-specific fix
   if Phase 1 does not improve accuracy). In `coupled_grpo_logprobs.py`:
   - add the `1/t` and `1/(1-t)` inverse-probability (ELBO) weighting -> unbiased estimate;
   - add the fully-masked averaging term: `final = (logp_fullymasked + weighted_coupled)/2`;
   - (t-bound `[0.2,0.8]` already applied in Phase 1).
3. **Phase 3 — remaining recipe anchors** (if still unstable):
   - EMA-synced reference (DiffuCoder syncs every 64 steps; we use a fixed reference) --
     requires a NeMo-RL support check;
   - `num_iterations(mu)=2` inner updates + wider `epsilon=0.5`.
4. **Fallback** — swap the coupled/fully-masked logprob for **BlockJustGRPO's leftmost
   left-to-right reveal**, which already trains stably in our stack (reward 0.36 -> 0.66),
   sidestepping the biased estimator entirely.

Open items:
- Confirm UniGRPO's exact normalization (`1/t` vs per-token mean) from Gen-Verse/dLLM-RL or
  martian422/MaskGRPO -- decides the precise Phase-2 form.
- Confirm whether NeMo-RL GRPO supports an EMA / periodically-synced reference model.

## Reference sources

- d1 (diffu-GRPO): github.com/dllm-reasoning/d1 — `diffu-grpo/diffu_grpo_trainer.py`
- DiffuCoder Coupled-GRPO: github.com/apple/ml-diffucoder — `src/open_r1/coupled_grpo.py`,
  `recipes/config_coupled_code.yaml`
- MMaDA UniGRPO: github.com/Gen-Verse/MMaDA, github.com/Gen-Verse/dLLM-RL
- TRL `SyncRefModelCallback`: github.com/huggingface/trl — `trl/trainer/callbacks.py`
- Dr. GRPO: Liu et al., "Understanding R1-Zero-Like Training"
