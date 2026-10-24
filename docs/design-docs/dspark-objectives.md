# DSpark Objectives

`dspark_tiled_objective` computes three losses for valid draft slots:

| Loss | Purpose |
| --- | --- |
| Hard cross-entropy (CE) | Train on sampled draft labels. |
| Total variation (TV) | Compare the draft and target token distributions. |
| Confidence | Predict token acceptance when a confidence head is present. |

The function processes draft tokens in bounded chunks. It keeps target
logits and the target output head stop-gradient. The Markov head uses
previous-token IDs from the target vocabulary. Loss labels use the selected
draft-vocabulary order. The two vocabularies can have different sizes.

Each loss returns raw numerator and valid-slot count bins. The combined loss
uses the configured CE, TV, and confidence weights. Position weights scale
both numerators and denominators. Reduce counts across data-parallel ranks
before calling `DSparkLossBins.normalized`.

`build_dspark_provider` attaches the Markov head and optional confidence head
to a checkpointable draft body. PR #3711 defines the head shapes and
checkpoint keys.
Training-loop configuration and rollout execution are outside this change.
