# Feedback Systems Based on Generative Models

## 1. Model Selection & Training

Choose a model capable of:

- **Skill-level separation** — correctly distinguishing players of varying skill levels (by MMR, win/loss, or any other metric deemed important for guidance).
- **Feature reconstruction** — reconstructing input features, which are a direct result of both players' actions.

The model must be trained with both objectives in mind: discriminative skill separation and faithful reconstruction.

## 2. Latent Space Embedding

After training, the latent space becomes the core tool for analysis. The model can encode new, previously unseen samples, mapping each to a **point in the latent space**.

## 3. Identifying Skill Gaps

Assume we embed a player whose skill is insufficient. They should land closer to the density corresponding to **low-skill players**.

> **Key property:** Players ranging from bad to worse exhibit a higher distance from their embedding to the center of the distribution of good (winning) players.

## 4. Latent-Space Path Finding & Feature Guidance

We seek **paths** from the low-skill embedding point to the closest density of the opposite "winning" class:

- **Shortest path** — minimal traversal through the latent space.
- **Smoothest path** — minimizing abrupt changes in reconstructed features.

The **reconstruction of interpolated points** along these paths corresponds to the change in features the player should focus on.

> To keep feedback actionable, select the **single feature showing the largest difference** along the path — guiding the player to focus on one improvement at a time.

## 5. Tracking Improvement

Player improvement is visible as a **trajectory in the latent space** — successive embeddings of a player's games should trace a path moving from the low-skill region toward the high-skill density over time.

---

## Implementation

See [`feedback_path.py`](../feedback_path.py) for the working implementation. It supports:

- `--method centroid` — linear interpolation toward the global winning centroid
- `--method nearest` — interpolation toward the k-nearest winning neighbours
- `--player 1|2` — per-player analysis
- Produces 4 visualisations and a top-k feature delta report
