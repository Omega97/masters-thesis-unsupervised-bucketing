
## 🏆 Most Relevant Works (Gradient Clustering → MoE)

These papers form the core methodological lineage for your approach.

---

### 1. GradientSpace (Sridharan et al., 2025) ⭐⭐⭐⭐⭐
**Similar idea to mine, but applied on LLMs**

- **Summary:** Clusters samples directly in full-dimensional gradient space (using LoRA gradients), trains a specialized LoRA expert per cluster, and deploys a lightweight encoder-based router for fast single-expert inference—all while keeping the base model frozen.
- **Similarity to your method:** Almost identical pipeline: gradient-space clustering → expert fine-tuning → lightweight router.
- **Key difference from your method:** Targets LLM instruction tuning (not chess/NNUE); uses LoRA gradients (not head-parameter gradients); router uses a small encoder (not L1-activation linear dispatcher). Your method is a domain-specific adaptation to chess NNUE value estimation.

- **arXiv:** [https://arxiv.org/abs/2512.06678](https://arxiv.org/abs/2512.06678)
- **PDF:** [https://arxiv.org/pdf/2512.06678.pdf](https://arxiv.org/pdf/2512.06678.pdf)

---

### 2. ELREA (Li et al., ICLR 2025) ⭐⭐⭐⭐☆
**The direct precursor to GradientSpace; a strong methodological match.**
Amazon, 🇨🇳

- **Summary:** Clusters training instructions by their gradient directions, representing different areas of expertise, to reduce conflicts during optimization.
- **Similarity to your method:** Core idea of gradient-space partitioning as the basis for a mixture of experts.
- **Key difference from your method:** (a) randomly projects gradients to lower dimensions (losing information); (b) routes to a *weighted ensemble* of experts at inference, requiring multiple forward passes and on-the-fly gradient computation. Your method improves on both points: full-dimensional clustering and a fast deterministic router.

- **arXiv:** [https://arxiv.org/abs/2502.00089](https://arxiv.org/abs/2502.00089)
- **PDF:** [https://arxiv.org/pdf/2502.00089.pdf](https://arxiv.org/pdf/2502.00089.pdf)
- **ICLR 2025 Poster:** [https://www.iclr.cc/](https://www.iclr.cc/) (cercare il poster ELREA)

---

### 3. TAGCOS (Zhang et al., 2024) ⭐⭐⭐⭐½
**Methodologically very close, but for a different purpose.**

- **Summary:** Leverages sample gradients as data representations, performs clustering to group similar data, and applies a greedy algorithm for coreset selection (selecting representative samples).
- **Similarity to your method:** Per-sample gradients → clustering → groups of similar examples. The conceptual connection is very strong: the gradient is treated as a representation of what a particular sample wants the model to learn.
- **Key difference from your method:** The purpose is different—instead of training one expert per cluster, they select a representative *coreset* for efficient fine-tuning. Your method trains specialized heads on each cluster.

- **arXiv:** [https://arxiv.org/abs/2407.15235](https://arxiv.org/abs/2407.15235)
- **PDF:** [https://arxiv.org/pdf/2407.15235.pdf](https://arxiv.org/pdf/2407.15235.pdf) (disponibile dalla pagina arXiv)

---

## 🧠 Theoretical & Adjacent Works

### 4. Kawata et al. (ICML 2025) ⭐⭐⭐⭐
**Provides theoretical backing for your core hypothesis.**

- **Summary:** Proves that MoE trained with SGD can detect and exploit latent cluster structure that a single vanilla network cannot.
- **Similarity to your method:** Provides formal justification for why gradient-informed partitioning could beat both handcrafted and activation-based bucketing.
- **Key difference from your method:** Studies theoretical nonlinear regression, not explicit clustering of per-sample gradients or expert fine-tuning on a chess dataset.

- **arXiv:** [https://arxiv.org/abs/2506.01656](https://arxiv.org/abs/2506.01656)
- **PDF:** [https://arxiv.org/pdf/2506.01656.pdf](https://arxiv.org/pdf/2506.01656.pdf) (disponibile dalla pagina arXiv)
- **ICML 2025:** Accettato a ICML 2025

---

### 5. Gradient Atoms (2026) ⭐⭐⭐½
**Related work on unsupervised gradient decomposition.**

- **Summary:** Decomposes per-document training gradients into sparse components ("atoms") via dictionary learning in a preconditioned eigenspace.
- **Similarity to your method:** Uses gradients for unsupervised discovery of structure.
- **Key difference from your method:** Focuses on behaviour discovery and attribution, not on training specialized experts for routing. Can be cited as adjacent work on gradient-defined modularity.

- **arXiv:** [https://arxiv.org/abs/2603.14665](https://arxiv.org/abs/2603.14665)
- **PDF:** [https://arxiv.org/pdf/2603.14665.pdf](https://arxiv.org/pdf/2603.14665.pdf) (disponibile dalla pagina arXiv)
- **GitHub:** [https://github.com/jrosseruk/gradient_atoms](https://github.com/jrosseruk/gradient_atoms)

---

## ♟️ Chess-Specific Context (Your Baseline)

### 6. M2CTS (Helfenstein et al., 2024) ⭐⭐☆☆☆
**The most relevant chess-domain paper, but methodologically different.**

- **Summary:** Integrates MoE into an AlphaZero-style MCTS engine for chess, using **manual phase-based bucketing** (opening, middlegame, endgame).
- **Similarity to your method:** Demonstrates that even coarse-grained state-space partitioning helps in chess; achieves +122 Elo improvements.
- **Key difference from your method:** The partition is **handcrafted** from chess phase definitions, not discovered from sample gradients. This paper represents your **baseline or motivating prior work**—you can cite it as evidence that chess benefits from expert specialization, while leaving open the question of whether the partitions can be learned automatically.

- **arXiv:** [https://arxiv.org/abs/2401.16852](https://arxiv.org/abs/2401.16852)
- **PDF:** [https://arxiv.org/pdf/2401.16852.pdf](https://arxiv.org/pdf/2401.16852.pdf) (disponibile dalla pagina arXiv)
- **HTML:** [https://arxiv.org/html/2401.16852](https://arxiv.org/html/2401.16852)

---

## 📊 Summary Table

| Paper              | Per-Sample Gradient | Clustering | Bucketing | MoE | Clustering Signal             | Domain                 | Router                       | Expert Type   |
| ------------------ | :------------------ | :--------- | :-------- | :-- | ----------------------------- | ---------------------- | ---------------------------- | ------------- |
| **GradientSpace**  | ✅                   | ✅          | ✅         | ✅   | Sample-gradients (LoRA)       | LLM instruction tuning | Lightweight encoder          | LoRA adapters |
| **ELREA**          | ✅                   | ✅          | ✅         | ✅   | Sample-gradients (projected)  | LLM instruction tuning | Gradient-similarity ensemble | LoRA adapters |
| **TAGCOS**         | ✅                   | ✅          | ✅         | ❌   | Sample-gradients              | LLM instruction tuning | N/A (coreset selection)      | N/A           |
| **Gradient Atoms** | ✅                   | ✅          | ❌         | ❌   | Gradient decomposition        | LLM                    | N/A                          | N/A           |
| **M2CTS**          | ❌                   | ❌          | ✅         | ✅   | **Manual (game phase)**       | Chess / MCTS           | Phase detector               | Full networks |
| **Kawata et al.**  | ❌                   | ❌          | ❌         | ✅   | Theoretical (latent clusters) | Regression             | N/A                          | N/A           |


---

## 🚨 Key Novelty Boundary

Based on the literature found, the broad idea of clustering data by gradient direction and training specialized experts is already present in ELREA and GradientSpace.

**Your more defensible novelty lies in the combination of:**

1. **Sample-gradients with respect to the NNUE head** (not LoRA or full LLM parameters).
2. **Chess-position/state-space partitioning** without handcrafted game-phase labels.
3. **Frozen shared representation plus specialized lightweight heads**.
4. **A learned L1-activation-based dispatcher** (avoiding gradients during inference).
5. **Deterministic single-expert routing with negligible engine overhead**.
6. Evaluation on **WDL targets from a strong chess value function**, rather than language-model benchmarks.

**GradientSpace appears extremely close to your full algorithmic structure**—it should be treated as a **central prior** rather than merely background literature. Your paper will need to make the domain-specific and systems-level differences explicit, especially the choice of head gradients, the NNUE architecture, dispatcher features, and the constraints of chess-engine inference.

---

#ai-generated

