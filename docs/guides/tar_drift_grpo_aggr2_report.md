# TAR Drift Report: GRPO Aggressive Sweep (Seeds 123/124/125)

## Objective
Evaluate whether the draft model experiences speculative-decoding drift (via token acceptance rate, TAR) when running aggressive GRPO updates on the verifier/policy (`train`) compared with a frozen-policy control (`frozen`).

## Experimental Setup
- Sweep root: `/home/scratch.shaunakj_other/logs/tar-drift-ab-aggr2-seed-sweep-s2-steps30-2026-02-25-161717`
- Seeds: `123, 124, 125`
- Steps per arm: `30`
- Speculative draft model: `Qwen3-0.6B`
- Target/verifier model: `Qwen3-32B`
- Speculative tokens (`num_speculative_tokens`): `2`

### Arms
- `train`: GRPO updates enabled with `++policy.optimizer.kwargs.lr=1e-4`
- `frozen`: control arm with `++policy.optimizer.kwargs.lr=0.0`

### What Was Cranked Up (vs base defaults)
| Knob | Base default | This sweep | Direction |
|---|---:|---:|---|
| `policy.optimizer.kwargs.lr` | `5e-6` | `1e-4` (`train`) | 20x higher |
| `grpo.max_num_epochs` | `1` | `4` | more updates per batch |
| `loss_fn.ratio_clip_min/max` | `0.2 / 0.2` | `0.5 / 0.5` | wider policy-ratio movement |
| `loss_fn.reference_policy_kl_penalty` | `0.01` | `0.0` | removed KL anchor |
| `policy.max_grad_norm` | `1.0` | `5.0` | looser clipping |
| `grpo.max_num_steps` | `10` (recipe) | `30` | longer horizon |
| `policy.generation.temperature` | `0.6` (recipe) | `1.0` | higher entropy sampling |
| `num_speculative_tokens` | `3` (recipe) | `2` | narrower speculative window |

## What “Frozen” Means in Parameter Terms
`frozen` in this sweep means **optimizer LR is set to `0.0`** for the policy/verifier (`++policy.optimizer.kwargs.lr=0.0`).

Practical implication:
- forward/backward/logprob computations still run,
- but optimizer updates are no-op (no effective parameter movement),
- this is a **zero-learning-rate freeze**, not a strict `requires_grad=False` hard freeze.

The draft model itself is not being trained in either arm; drift is observed as changing acceptance behavior relative to policy/verifier dynamics.

## What Is the Speculative Decoding Window?
In this setup, the speculative decoding window is controlled by:
- `++policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens=2`

Meaning: each speculation cycle can propose up to **2 draft tokens** before verifier confirmation/rejection.

Note: this is different from context window. Context length is controlled by `max_model_len/max_total_sequence_length` (set to `5120`).

## Results Summary
### Per-arm summary
```tsv
seed	arm	n	tar_first	tar_last	tar_mean	tar_slope	rew_first	rew_last	rew_mean	rew_slope
123	frozen	30	0.6537	0.6333	0.7009	-0.000041	0.7500	0.6250	0.7167	0.004672
123	train	30	0.6537	0.8708	0.6670	0.007167	0.7500	0.0000	0.4583	-0.012236
124	frozen	30	0.6313	0.6978	0.6898	0.000495	0.8750	0.5000	0.6667	0.004672
124	train	30	0.6313	0.4996	0.5338	-0.014400	0.8750	0.0000	0.4625	-0.022386
125	frozen	30	0.6807	0.7653	0.7189	0.000003	0.5000	1.0000	0.7125	-0.002308
125	train	30	0.6807	0.5164	0.6616	-0.007423	0.5000	0.5000	0.7250	-0.003838
```

### Per-seed train-minus-frozen deltas
```tsv
seed	tar_last_delta	tar_mean_delta	rew_last_delta	rew_mean_delta	tar_delta_slope	reward_delta_slope
123	+0.2375	-0.0339	-0.6250	-0.2583	+0.007208	-0.016908
124	-0.1982	-0.1560	-0.5000	-0.2042	-0.014895	-0.027058
125	-0.2489	-0.0573	-0.5000	+0.0125	-0.007426	-0.001529
```

### Aggregate drift trend (mean over seeds)
```tsv
step	mean_tar_delta	mean_reward_delta
1	+0.0000	+0.0000
2	+0.0094	+0.0000
3	-0.0106	+0.0000
4	-0.0173	-0.0417
5	-0.0334	+0.0000
6	-0.0060	+0.0000
7	-0.0153	+0.0000
8	-0.0495	+0.0000
9	-0.0805	-0.0417
10	-0.0824	+0.0000
11	-0.0736	-0.2500
12	-0.0920	-0.1250
13	-0.1083	-0.0833
14	-0.0596	+0.0000
15	-0.0950	+0.0000
16	-0.0962	-0.0833
17	-0.0799	-0.3333
18	-0.0911	-0.0417
19	-0.0842	+0.0417
20	-0.1029	-0.1667
21	-0.0567	-0.0833
22	+0.0215	-0.1667
23	-0.2027	-0.5417
24	-0.2455	-0.5417
25	-0.1454	-0.5000
26	-0.1331	+0.0000
27	-0.1997	-0.4167
28	-0.2159	-0.1250
29	-0.0557	-0.4583
30	-0.0699	-0.5417
```

Aggregate delta slopes:
- `tar_delta_slope = -0.005037`
- `reward_delta_slope = -0.015165`
- `last_mean_tar_delta = -0.0699`
- `last_mean_reward_delta = -0.5417`

## Interpretation
- Drift is **seed-dependent** but overall trends negative for TAR under aggressive GRPO.
- Seed outcomes:
  - `123`: late TAR recovery in `train` (final TAR above frozen), but reward worsens.
  - `124`: strong negative TAR drift with late collapse in `train`.
  - `125`: consistent late negative TAR drift in `train`.
- Net across seeds: average `tar_last_delta(train-frozen) = -0.0699`, average `tar_mean_delta = -0.0824`.

## Plot-Ready Full Tables
### Step-level values
```tsv
seed	arm	step	tar	reward
123	frozen	1	0.6537	0.7500
123	frozen	2	0.7041	0.6250
123	frozen	3	0.6844	0.6250
123	frozen	4	0.6496	0.5000
123	frozen	5	0.7326	0.8750
123	frozen	6	0.7091	0.5000
123	frozen	7	0.6499	0.5000
123	frozen	8	0.7680	1.0000
123	frozen	9	0.7408	0.8750
123	frozen	10	0.6951	0.0000
123	frozen	11	0.6624	0.8750
123	frozen	12	0.6347	0.6250
123	frozen	13	0.7231	0.6250
123	frozen	14	0.7627	0.5000
123	frozen	15	0.7352	1.0000
123	frozen	16	0.7234	0.7500
123	frozen	17	0.7071	1.0000
123	frozen	18	0.6809	0.8750
123	frozen	19	0.6550	0.0000
123	frozen	20	0.7223	1.0000
123	frozen	21	0.7887	1.0000
123	frozen	22	0.7567	1.0000
123	frozen	23	0.6857	1.0000
123	frozen	24	0.7364	1.0000
123	frozen	25	0.7487	1.0000
123	frozen	26	0.6505	0.5000
123	frozen	27	0.7088	1.0000
123	frozen	28	0.6797	0.0000
123	frozen	29	0.6438	0.8750
123	frozen	30	0.6333	0.6250
123	train	1	0.6537	0.7500
123	train	2	0.7119	0.5000
123	train	3	0.6615	0.7500
123	train	4	0.6297	0.5000
123	train	5	0.6690	0.8750
123	train	6	0.6887	0.5000
123	train	7	0.6009	0.5000
123	train	8	0.6383	1.0000
123	train	9	0.6146	0.6250
123	train	10	0.5629	0.0000
123	train	11	0.5095	0.1250
123	train	12	0.4780	0.5000
123	train	13	0.5716	0.1250
123	train	14	0.6157	0.5000
123	train	15	0.5745	1.0000
123	train	16	0.5622	0.3750
123	train	17	0.5537	0.1250
123	train	18	0.5601	0.8750
123	train	19	0.5765	0.2500
123	train	20	0.5949	0.0000
123	train	21	0.7300	0.3750
123	train	22	0.7578	1.0000
123	train	23	0.8292	0.3750
123	train	24	0.7844	0.0000
123	train	25	0.8274	0.5000
123	train	26	0.7627	0.5000
123	train	27	0.7856	0.5000
123	train	28	0.7612	0.1250
123	train	29	0.8732	0.5000
123	train	30	0.8708	0.0000
124	frozen	1	0.6313	0.8750
124	frozen	2	0.6672	0.1250
124	frozen	3	0.6940	0.1250
124	frozen	4	0.6884	0.2500
124	frozen	5	0.6568	0.6250
124	frozen	6	0.6975	1.0000
124	frozen	7	0.7295	1.0000
124	frozen	8	0.6112	1.0000
124	frozen	9	0.6744	0.7500
124	frozen	10	0.7007	0.5000
124	frozen	11	0.7792	1.0000
124	frozen	12	0.6986	0.5000
124	frozen	13	0.6655	0.0000
124	frozen	14	0.7775	1.0000
124	frozen	15	0.6763	0.5000
124	frozen	16	0.6737	1.0000
124	frozen	17	0.6350	0.6250
124	frozen	18	0.6886	0.6250
124	frozen	19	0.7791	1.0000
124	frozen	20	0.6465	0.8750
124	frozen	21	0.7633	0.5000
124	frozen	22	0.6218	0.6250
124	frozen	23	0.7740	1.0000
124	frozen	24	0.6901	0.5000
124	frozen	25	0.7207	1.0000
124	frozen	26	0.5560	0.0000
124	frozen	27	0.7278	1.0000
124	frozen	28	0.7102	0.5000
124	frozen	29	0.6609	1.0000
124	frozen	30	0.6978	0.5000
124	train	1	0.6313	0.8750
124	train	2	0.6881	0.2500
124	train	3	0.6848	0.0000
124	train	4	0.6548	0.1250
124	train	5	0.6188	0.6250
124	train	6	0.7000	1.0000
124	train	7	0.7319	1.0000
124	train	8	0.5932	1.0000
124	train	9	0.5599	0.8750
124	train	10	0.5855	0.5000
124	train	11	0.7202	1.0000
124	train	12	0.5592	0.2500
124	train	13	0.4993	0.1250
124	train	14	0.7553	1.0000
124	train	15	0.5560	0.5000
124	train	16	0.5746	1.0000
124	train	17	0.5803	0.5000
124	train	18	0.5658	0.5000
124	train	19	0.6379	0.8750
124	train	20	0.5434	0.8750
124	train	21	0.7231	0.8750
124	train	22	0.7855	0.1250
124	train	23	0.1644	0.0000
124	train	24	0.1033	0.0000
124	train	25	0.3283	0.0000
124	train	26	0.2062	0.0000
124	train	27	0.2103	0.0000
124	train	28	0.1542	0.0000
124	train	29	0.3990	0.0000
124	train	30	0.4996	0.0000
125	frozen	1	0.6807	0.5000
125	frozen	2	0.7818	1.0000
125	frozen	3	0.7300	0.5000
125	frozen	4	0.6925	0.5000
125	frozen	5	0.7693	1.0000
125	frozen	6	0.7529	1.0000
125	frozen	7	0.7243	1.0000
125	frozen	8	0.7055	0.5000
125	frozen	9	0.7281	1.0000
125	frozen	10	0.6999	0.7500
125	frozen	11	0.7076	0.5000
125	frozen	12	0.6513	0.5000
125	frozen	13	0.6871	0.3750
125	frozen	14	0.6990	1.0000
125	frozen	15	0.7262	0.5000
125	frozen	16	0.7170	0.8750
125	frozen	17	0.6404	1.0000
125	frozen	18	0.7899	1.0000
125	frozen	19	0.7447	1.0000
125	frozen	20	0.7114	0.0000
125	frozen	21	0.7439	0.5000
125	frozen	22	0.7077	1.0000
125	frozen	23	0.6674	0.5000
125	frozen	24	0.7733	0.6250
125	frozen	25	0.7392	1.0000
125	frozen	26	0.7157	0.5000
125	frozen	27	0.7173	0.7500
125	frozen	28	0.7424	0.5000
125	frozen	29	0.6547	0.5000
125	frozen	30	0.7653	1.0000
125	train	1	0.6807	0.5000
125	train	2	0.7813	1.0000
125	train	3	0.7302	0.5000
125	train	4	0.6942	0.5000
125	train	5	0.7706	1.0000
125	train	6	0.7528	1.0000
125	train	7	0.7251	1.0000
125	train	8	0.7046	0.5000
125	train	9	0.7274	1.0000
125	train	10	0.7002	0.7500
125	train	11	0.6988	0.5000
125	train	12	0.6714	0.5000
125	train	13	0.6799	0.5000
125	train	14	0.6894	1.0000
125	train	15	0.7223	0.5000
125	train	16	0.6888	1.0000
125	train	17	0.6089	1.0000
125	train	18	0.7603	1.0000
125	train	19	0.7117	1.0000
125	train	20	0.6332	0.5000
125	train	21	0.6726	0.5000
125	train	22	0.6073	1.0000
125	train	23	0.5254	0.5000
125	train	24	0.5756	0.5000
125	train	25	0.6168	1.0000
125	train	26	0.5540	0.5000
125	train	27	0.5589	1.0000
125	train	28	0.5692	0.5000
125	train	29	0.5200	0.5000
125	train	30	0.5164	0.5000
```

### Step-level train-minus-frozen deltas
```tsv
seed	step	tar_delta_train_minus_frozen	reward_delta_train_minus_frozen
123	1	+0.0000	+0.0000
123	2	+0.0078	-0.1250
123	3	-0.0229	+0.1250
123	4	-0.0199	+0.0000
123	5	-0.0636	+0.0000
123	6	-0.0204	+0.0000
123	7	-0.0490	+0.0000
123	8	-0.1297	+0.0000
123	9	-0.1262	-0.2500
123	10	-0.1322	+0.0000
123	11	-0.1529	-0.7500
123	12	-0.1567	-0.1250
123	13	-0.1515	-0.5000
123	14	-0.1470	+0.0000
123	15	-0.1607	+0.0000
123	16	-0.1612	-0.3750
123	17	-0.1534	-0.8750
123	18	-0.1208	+0.0000
123	19	-0.0785	+0.2500
123	20	-0.1274	-1.0000
123	21	-0.0587	-0.6250
123	22	+0.0011	+0.0000
123	23	+0.1435	-0.6250
123	24	+0.0480	-1.0000
123	25	+0.0787	-0.5000
123	26	+0.1122	+0.0000
123	27	+0.0768	-0.5000
123	28	+0.0815	+0.1250
123	29	+0.2294	-0.3750
123	30	+0.2375	-0.6250
124	1	+0.0000	+0.0000
124	2	+0.0209	+0.1250
124	3	-0.0092	-0.1250
124	4	-0.0336	-0.1250
124	5	-0.0380	+0.0000
124	6	+0.0025	+0.0000
124	7	+0.0024	+0.0000
124	8	-0.0180	+0.0000
124	9	-0.1145	+0.1250
124	10	-0.1152	+0.0000
124	11	-0.0590	+0.0000
124	12	-0.1394	-0.2500
124	13	-0.1662	+0.1250
124	14	-0.0222	+0.0000
124	15	-0.1203	+0.0000
124	16	-0.0991	+0.0000
124	17	-0.0547	-0.1250
124	18	-0.1228	-0.1250
124	19	-0.1412	-0.1250
124	20	-0.1031	+0.0000
124	21	-0.0402	+0.3750
124	22	+0.1637	-0.5000
124	23	-0.6096	-1.0000
124	24	-0.5868	-0.5000
124	25	-0.3924	-1.0000
124	26	-0.3498	+0.0000
124	27	-0.5175	-1.0000
124	28	-0.5560	-0.5000
124	29	-0.2619	-1.0000
124	30	-0.1982	-0.5000
125	1	+0.0000	+0.0000
125	2	-0.0005	+0.0000
125	3	+0.0002	+0.0000
125	4	+0.0017	+0.0000
125	5	+0.0013	+0.0000
125	6	-0.0001	+0.0000
125	7	+0.0008	+0.0000
125	8	-0.0009	+0.0000
125	9	-0.0007	+0.0000
125	10	+0.0003	+0.0000
125	11	-0.0088	+0.0000
125	12	+0.0201	+0.0000
125	13	-0.0072	+0.1250
125	14	-0.0096	+0.0000
125	15	-0.0039	+0.0000
125	16	-0.0282	+0.1250
125	17	-0.0315	+0.0000
125	18	-0.0296	+0.0000
125	19	-0.0330	+0.0000
125	20	-0.0782	+0.5000
125	21	-0.0713	+0.0000
125	22	-0.1004	+0.0000
125	23	-0.1420	+0.0000
125	24	-0.1977	-0.1250
125	25	-0.1224	+0.0000
125	26	-0.1617	+0.0000
125	27	-0.1584	+0.2500
125	28	-0.1732	+0.0000
125	29	-0.1347	+0.0000
125	30	-0.2489	-0.5000
```

## Primary Artifacts
- Sweep status: `/home/scratch.shaunakj_other/logs/tar-drift-ab-aggr2-seed-sweep-s2-steps30-2026-02-25-161717/status.log`
- Resume script (exact overrides): `/home/scratch.shaunakj_other/logs/tar-drift-ab-aggr2-seed-sweep-s2-steps30-2026-02-25-161717/autosave/resume_remaining_arms.sh`
- Base recipe: `/home/scratch.shaunakj_other/Development/RL/examples/configs/recipes/llm/grpo-qwen3-32b-spec-decode-lowbatch-1n8g.yaml`
- Base GRPO defaults: `/home/scratch.shaunakj_other/Development/RL/examples/configs/grpo_math_1B.yaml`
