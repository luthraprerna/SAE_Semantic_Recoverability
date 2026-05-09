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
