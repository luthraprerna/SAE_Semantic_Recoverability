# Self-Described Concepts in SAE Feature Space

Probing whether model self-descriptions of behavioral concepts (e.g. *incorrect answers*, *flattery*) correspond to coherent, discriminative features in Gemma-2-2B's residual stream, using Gemma Scope SAEs at layers 6, 13, and 20.

## Stages

| # | Stage | What it does |
|---|---|---|
| 1 | Setup + smoke test | Loads Gemma-2-2B and Gemma Scope SAE (16k width); verifies both work end-to-end. |
| 2 | Prompt set | Length-matched, lexically de-confounded concept and control prompts: `incorrect_answers` / `incorrect_control`, `flattery` / `flattery_control`, `cooking_unrelated`. |
| 3 + 4 | Activation capture + SAE encoding | Hooks layers 6/13/20, encodes per-token activations through the SAE, aggregates to mean / max / last-token feature vectors per prompt. |
| 5 | Discriminative features | Per (concept, layer): Cohen's *d* and Mann–Whitney U with FDR correction. Top-20 features per concept. |
| 6 | Null baselines | 1000 shuffled-label permutations + random-feature baselines to verify that top-feature effect sizes aren't noise. |
| 7 | Auto-interp labels | Pulls human-readable feature descriptions from Neuronpedia for every top feature. |
| 8 | Lexical vs semantic decomposition | Embeds feature labels with `all-mpnet-base-v2` and measures alignment against the model's self-verbalization for each concept. |
| 8.5 | Max-pool robustness | Re-runs stages 5–8 on max-pooled (rather than mean-pooled) features as a robustness check. |
| 9 | Figures + tables | Recomputes per-feature details, generates the 6-panel main figure, layer-trajectory plot, lexical-vs-semantic figure, and LaTeX top-features table. |
| 9.5 | Cluster analysis | K-means over top-30 feature label embeddings per concept (k=2–6, silhouette-selected). Tests whether discriminative features form one coherent cluster aligned to the concept (*monolithic*) or fragment into sub-themes (*decomposed*). |

## Result summary

Discriminative features exist and are statistically robust — top features clear Cohen's *d* > 1.5 at layer 13 for both concepts and survive the permutation null. But the features don't sit where the model's self-description says they should:

| Concept | Best-cluster *k* | Silhouette | Top-cluster sim. to self-verbalization |
|---|---|---|---|
| `cooking_unrelated` (sanity check) | 2 | 0.162 | **0.863** |
| `incorrect_answers` | 2 | 0.137 | 0.381 |
| `flattery` | 6 | 0.078 | 0.466 |

`cooking_unrelated` behaves as expected: the dominant cluster is recognizably about cooking. `incorrect_answers` and `flattery` don't — their top discriminative features fragment across sub-themes (punctuation, pronouns, formatting, judgment vocabulary) and none align tightly with the model's own description of what those behaviors are.

So the concepts are detectable, but at the feature level they appear *decomposed* rather than monolithic — the model's self-verbalization is not a faithful pointer to a single feature direction.
