
# Abstract

The following text is meant to be interpreted as a single block. The sections are for clarity.

---

#### Topic / Context:
In the context of board games, a common approach to increasing model performance is to partition the state space and allocate distinct experts to each region. While this is the central idea behind Mixture of Experts (MoE) architectures, the bucketing is almost always defined manually, requiring domain expertise and yielding partitions that may be suboptimal for expert specialisation.

#### Gap / Problem / Research Question
There is no principled, unsupervised method for discovering a meaningful partition of the state space that *maximises expert specialisation*, particularly when the model is already trained and the routing must be both fast and interpretable.

#### Contribution
We propose a novel unsupervised bucketing method based on sample gradients — the gradients of the loss with respect to the head parameters of a base model. These vectors encode the direction in weight space that would improve the model's prediction for each individual data point. We apply the method to chess, using a standard NNUE (Efficiently Updatable Neural Network) as the base model trained on 5 million positions labelled with outcome probabilities (WDL) by a strong value function (Lc0).

#### Methods
For each position, we compute the normalised sample gradient with respect to the head parameters, and apply a clustering algorithm to define the buckets. We then train a lightweight linear dispatcher to predict the bucket from the L1 activations, enabling fast, deterministic routing at inference time. Finally, we fine‑tune the expert heads on each bucket, starting from the base model, while keeping the L1 weights frozen. This yields a set of fast, specialised experts, each adapted to a distinct region of the state space, without requiring handcrafted bucketing.

#### Key Findings / Expected Results (if everything goes well... 🍀)
Preliminary results indicate that the resulting MoE model achieves lower test cross‑entropy and mean absolute error than the single‑head baseline, with negligible runtime overhead. The sample‑gradient clusters reveal interpretable structure in the state space (e.g., tactical vs. positional, opening vs. endgame), and the dispatcher achieves high accuracy (>70%), enabling fast, deterministic routing at inference time.
