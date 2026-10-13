
# Unsupervised State Bucketing for Mixture of Experts in Resource-Constrained Chess Engines

---

*"Divide and conquer."*

#note I chose this quote for the thesis because it evokes a tactical situation, not unlike a chess position, and closely resembles the goal of MoE. 

---

# Abstract

<span style="color: #808080;">[Topic / Context]</span> In the context of board games, a common approach for improving model performance is to partition the state space and allocate distinct experts to each region. While this is the central idea behind *Mixture of Experts* (MoE) architectures, the bucketing is often defined manually, requiring domain-specific expertise, and yielding potentially suboptimal partitions.

#todo resource-constrained

<span style="color: #808080;">[Contribution]</span> We investigate and compare several approaches to extending the *Efficiently Updatable Neural Networks* (NNUE) paradigm beyond its conventional formulation, with a particular focus on techniques that have been successfully explored in other areas of machine learning but remain relatively uncommon in NNUE-based chess engines. Mixture-of-Experts architectures are a natural extension for chess evaluation that allows multiple specialized heads to share a common NNUE representation. We also explore sample-gradient-based partitioning, using the gradients of individual training samples to identify groups of positions with similar learning requirements, an approach more commonly studied in large-scale language models and other deep learning settings. Finally, in an effort to make the NNUE architecture more applicable to other fields, we experiment with making its conventional accumulator a learned world model of the game.

<span style="color: #808080;">[Key Findings🍀)]</span> The results highlight different trade-offs between the investigated approaches. L1-based clustering combined with a Mixture-of-Experts architecture significantly improves evaluation performance, while introducing only a minor increase in memory usage and inference cost; the lightweight dispatcher accurately reproduces the learned partition. In contrast, sample-gradient-based partitioning is difficult to exploit at inference time, since the gradients depend on the unknown true value of the position, and a lightweight dispatcher struggles to approximate the resulting complex partition. Finally, the world-model approach shows that an NNUE-like architecture can be trained without feedback from a teacher engine or game outcomes, although the unsupervised training objective can reduce evaluation strength. At the same time, learning the world model makes the training process more efficient by shifting a substantial part of the learning effort to the shared representation.

---

> Note for the AI: *In this abstract, the §Scraps section and gray text are for clarity only, and must not be transferred to the Latex. Placeholders in square brackets (e.g. `[dataset_size]`) must be replaced with the current values from `_ai-info_.md` when converting this document to Latex.*

[[1 - Introduction]]
