
# Introduction


---

## 1.1 Motivation

<span style="color: #808080;">[Need / Constraint]</span> A chess engine spends most of its time asking the same question: *how good is this position?* On a desktop, that question can be answered by a large network; on a microcontroller, it cannot. In the later case, the evaluation function must be small, integer-friendly, and cheap to run at every node of an alpha-beta search. *Efficiently Updatable Neural Networks* (**NNUE**) are the current standard: an *accumulator layer* (sparse first layer that tracks the board incrementally), followed by a small fully connected head.

<span style="color: #808080;">[Heuristic Gap]</span> With that architecture, a single head will struggle to capture the nuances of openings, quiet middlegames, tactical melees, and simplified endgames. Different positions may require understanding of very diverse aspects of the game. Mixture-of-Experts (**MoE**) evaluation is the natural solution: partition the state space into buckets, and assign an *expert head* to each region, while keeping a shared representation. In chess engines, this partition is almost always written by hand — piece count, king location, game phase. Those rules are cheap and interpretable, but they are game-specific heuristics and are not necessarily optimal. They encode what a programmer thinks is a distinct regime, not what the model actually needs in order to specialise.

#todo better phrasing?

<span style="color: #808080;">[Core Idea]</span> Unsupervised alternatives exist. Clustering hidden-layer activations, for example, groups positions that look similar in the eyes of the network. The weakness of this approach lies in the fact that similarity of representation is not the same as similarity of *learning signal*: activations describe what the model already encodes, not the direction in which its head should change. If the goal is expert specialisation, the partition should be defined by that direction. This thesis is motivated by the idea that a lightweight, data-driven bucketing — cheap enough for an on-device NNUE, and not tied to chess-specific features — is both a practical need for tiny engines and a more general way to mix experts over a state space.

## 1.2 Problem Statement

<span style="color: #808080;">[Problem]</span> The central problem addressed in this thesis is the design of an efficient, *data-driven* method for partitioning the state space of a chess engine into regions of specialist expertise, within the tight *inference-time* resource constraints of embedded devices.

<span style="color: #808080;">[Formal Setup]</span> More specifically, consider a standard NNUE evaluation function composed of a frozen shared representation $W_{L1}$ and a trainable head $(W_{L2}, W_{out})$. Given a dataset of positions $\mathcal{D} = \{(s_i, v_i)\}$, we can train a base model $w_{\text{base}}$. To improve upon this base model via expert specialization, we seek a partition of the state space into $B$ buckets $\{\mathcal{D}_1, \dots, \mathcal{D}_B\}$ such that training a separate head on each bucket yields a set of *task vectors* $\delta_i = \theta_i - \theta_{\text{base}}$ that are maximally diverse.

#todo maybe not really task vectors? but ok

<span style="color: #808080;">[Key Difficulty]</span> This objective is complicated by the fact that the partition must be discovered from the training data, yet the criterion for "good" specialization—diversity of task vectors—lives in the space of model weights, not in the input space. At the same time, the resulting routing mechanism must be *lightweight* enough to run on a microcontroller at every node of an alpha-beta search, ruling out expensive computations such as evaluating multiple full networks. Current heuristic bucketing strategies (e.g., based on piece count or king position) are computationally cheap but encode arbitrary *game-specific* assumptions about which positions are similar, failing to capture the actual learning signal that drives head specialization.

<span style="color: #808080;">[Research Question]</span> This work, therefore, investigates whether it is possible to learn an unsupervised partition directly from instance-specific gradients—using them as a proxy for the direction in which each head should specialize—and whether such a partition can be deployed via a lightweight dispatcher that respects the memory and computational constraints of an embedded device, without sacrificing the quality of the resulting mixture-of-experts evaluation.


## 1.3 Contributions

<span style="color: #808080;">[Prior Work]</span> Recent work has explored the idea of clustering samples by their gradient directions to create specialized experts. **ELREA** (Li et al., ICLR 2025) partitions training instructions by gradient directions to reduce optimization conflicts, while **GradientSpace** (Sridharan et al., 2025) clusters LoRA gradients and deploys a lightweight encoder-based router for fast single-expert inference. Both demonstrate that gradient-space partitioning is a powerful principle for mixture-of-experts architectures.

<span style="color: #808080;">[Domain Gap]</span> However, these methods are designed for **large language models** (LLMs), where the parameter space is enormous, experts are implemented as LoRA adapters, and inference latency is measured in milliseconds rather than microseconds. The problem of state-space partitioning for a chess evaluation function on an embedded device presents a fundamentally different set of constraints: the model must fit in a very small amount of memory, routing must occur at every node of an alpha-beta search with negligible overhead, and the *state space* partition should be discoverable from the data by looking at the model only.

#todo rephrase? something like that...

<span style="color: #808080;">[This Work]</span> Building on these prior works, this thesis introduces a perspective on unsupervised gradient-based bucketing adapted to the domain of chess engines. We propose a Mixture-of-Experts NNUE architecture where the state space is partitioned by clustering per-sample gradients of the loss with respect to the head parameters $(W_{L2},W_{out})$, capturing the learning signal that drives expert specialization without relying on handcrafted heuristics such as piece count or game phase. At inference time, a lightweight dispatcher—trained to predict bucket assignments from frozen L1 activations—routes each position to a single specialized head with negligible overhead, avoiding the cost of *ensembling* or *on-the-fly gradient computation*. We validate our approach empirically on chess, comparing against single-head baselines and heuristic bucketing strategies, and demonstrate that *gradient-informed partitioning* yields more distinct experts and improved evaluation quality on a dataset of positions labeled with WDL values from a strong engine.

## 1.4 Thesis Outline

<span style="color: #808080;">[Roadmap]</span>
#todo maybe not necessary? blank for now 

---

> **Note for AI**: *The parts marked with a #todo are yet to be completed. The tagged comments are NOT to be exported to the Latex document, and are not meant to be implemented while exporting this document to Latex. The gray labels are for clarity only and must not be transferred to the Latex.*
