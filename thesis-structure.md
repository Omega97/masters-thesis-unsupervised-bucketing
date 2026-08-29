

# Thesis Structure: Unsupervised Bucketing via Sample Gradients

---

## Thesis Title (Draft)

> *"Unsupervised State Bucketing for Mixture of Experts in Resource‑Constrained Chess Engines"*

---

## Keywords 

```
#chess #blucketing #NNUE #RL #clustering #value_function #NN #MoE
```

---

## Abstract

*Brief summary of the problem, method, results, and contributions.*

- **Problem**: Chess engines on tiny devices need efficient evaluation functions, but handcrafted bucketing (e.g., by piece count) is suboptimal.
- **Method**: Train a base NNUE, compute sample gradients (gradients w.r.t. head parameters), cluster them to define buckets, train a dispatcher, and fine‑tune expert heads.
- **Results**: The MoE model outperforms the single‑head baseline with negligible runtime overhead. #todo that's the goal!
- **Contributions**: 
	- A general‑purpose unsupervised bucketing method applicable beyond chess; 
	- a lightweight MoE architecture for NNUE; 
	- empirical validation on a real chess engine.

---

## 1. Introduction

### 1.1 Motivation

- Chess engines on embedded devices (Wio Terminal: 192 KB RAM, 500 KB flash).
- Need for efficient evaluation functions (NNUE).
- Handcrafted bucketing (piece count, queen presence) is heuristic and may not capture the true structure of the state space.

### 1.2 Problem Statement

- How can we automatically learn a partition of the state space that maximises expert specialisation?
- How can we do this without human supervision?

### 1.3 Contributions

- A novel **unsupervised bucketing method** based on sample gradients (gradients w.r.t. head parameters). 
- A lightweight **dispatcher** that routes positions to the right expert at inference time.
- A **Mixture of Experts (MoE)** NNUE architecture for resource‑constrained devices.
- Empirical validation on chess, showing improved evaluation quality over a single‑head baseline.

### 1.4 Thesis Outline

*Brief roadmap of the chapters.*

---

## 2. Background & Related Work

### 2.1 Chess Engines & Evaluation Functions

- Classical evaluation (material, piece‑square tables, mobility, king safety).
- NNUE (Efficiently Updatable Neural Networks) – sparse, accumulator‑based.
- Alpha‑Beta search, iterative deepening, move ordering.

### 2.2 Mixture of Experts (MoE)

- Idea: multiple expert networks, each specialising in a sub‑domain.
- Routing mechanisms: learned vs. fixed.
- Applications in vision, NLP, and reinforcement learning

### 2.3 Sample Gradients

- ...

### 2.4 Bucketing in NNUE

- default techniques: piece count, king location, piece presence (queen, double bishop, etc.) ...
- Limitations: handcrafted, may not align with model needs.

### 2.5 Related Work

- EfficientNet, pruning, quantization.
- Knowledge distillation, teacher‑student frameworks.
- Reinforcement learning from self‑play (AlphaZero, Lc0).
- #todo look better for work related to sample gradients

---

## 3. Method: Unsupervised Bucketing via sample gradients

### 3.1 Overview

- Five‑step pipeline:
  1. Train base model.
  2. Compute sample gradients.
  3. Cluster sample gradients.
  4. Train dispatcher.
  5. Fine‑tune expert heads.

### 3.2 Notation & Definitions

- State space $\mathcal{S}$, model $f_w$, loss $\mathcal{L}$, target $\hat v$.
- sample gradient: $\delta_i = \nabla_{w^{(head)}} \mathcal L(f_w(s_i), \hat v_i)$.
- Dispatcher: $g_\phi(h) = \text{softmax}(W_\phi h)$.

### 3.3 Step 1: Train Base Model

- Architecture: L1 (dual‑POV, shared), L2, output head.
- Training: soft cross‑entropy on WDL labels from Lc0.

### 3.4 Step 2: Compute Sample Gradients

- Per‑sample gradients w.r.t. head parameters.
- Normalisation (L2 or standardisation).
- Storage and computational considerations. #todo maybe?

### 3.5 Step 3: Cluster Sample Gradients

- K‑Means (or alternative: DBSCAN, Density Peak).
- Choosing \(B\) (number of clusters). #todo some clustering algo choose B for you
- Validation: inertia, silhouette score, cluster interpretability.

### 3.6 Step 4: Train Dispatcher

- Input: L1 activations \(h\).
- Output: bucket ID \(b\).
- Architecture: linear layer (or small MLP).
- Loss: cross‑entropy.

### 3.7 Step 5: Fine‑Tune Expert Heads

- Freeze L1, fine‑tune head on each bucket.
- Result: \(B\) specialised experts.

### 3.8 Generalisation Beyond Chess

- The method only requires: state space, model, loss, target (e.g. expected reward).
- Applicable to other games (Go, Shogi), robotics, etc. , maybe to whenever there is a world model

---

## 4. Implementation: NNUE + MoE for Chess

### 4.1 Dataset & Teacher

- Lichess PGNs → FEN positions → Lc0 WDL labels (depth 1).
- Dataset size: ~5M positions.
- Preprocessing: 844‑dim feature encoder (716 base + 128 tactical), dual‑POV.

### 4.2 Base NNUE Architecture

- L1: 844 to 64 (shared, dual‑POV).
- L2: 128 to 128.
- Output: 128 to 3 (WDL logits).
- Activations: CReLU (L1), ReLU (L2), softmax (output).
- Training: soft cross‑entropy, Adam, 100 epochs.

### 4.3 Sample Gradient Computation

- Implementation details: `torch.autograd.grad`, per‑sample gradients.
- Storage: `.npy` arrays (or `.parquet` for metadata).

### 4.4 Clustering

- Mini‑Batch K‑Means (scalability).
- Density Peak Clustering
- other? #todo
- \(B = 8\) (matching Stockfish’s bucketing) #todo check
- Cluster visualisation (e.g., t‑SNE, PCA).

### 4.5 Dispatcher Training

- Input: L1 activations (concatenated `[own ‖ opp]`). #todo maybe `[own]` is enough
- Architecture: linear layer $2W \to B$. (or W -> B)
- Training: cross‑entropy, Adam, 20 epochs.

### 4.6 Expert Fine‑Tuning

- #todo One sweep per bucket, or full training?
- Validation: per‑bucket test CE.

### 4.7 Final MoE Model

- Shared L1 + Dispatcher + Expert Heads.
- Inference: L1 → dispatcher → selected expert → output.

### 4.8 Integration with Cfish

- Replace `evaluate()` hook with MoE NNUE.
- Quantization: int8 weights, int16 accumulators, int32 MAC.
- Memory: L1 sparse weights (70‑80% pruning), dispatcher + heads in flash.

---

## 5. Experiments & Results

### 5.1 Experimental Setup

- Dataset split: train (2.4M), test (31k).
- Hardware: CPU (training), Wio Terminal (inference).
- Baselines:
	 - Single‑head NNUE.
	 - Dense FFNN (256×256).
	 - Handcrafted bucketing (piece count + queen presence).

### 5.2 Evaluation Metrics

- Test cross‑entropy (CE).
- Mean Absolute Error (MAE) on expected reward.
- Dispatcher accuracy.
- Per‑bucket test CE (improvement over base head).
- Playing strength (ACPL / Elo) via Stockfish analysis.

### 5.3 Results: Base Model

- Test CE: **0.71** (after 50 epochs).
- MAE: **0.25**.
- Comparison with FFNN (0.74 CE, 499k params).

### 5.4 Results: Clustering & Dispatcher

- Cluster quality (silhouette score).
- Dispatcher accuracy: **> 70%**.
- Interpretation of clusters (e.g., endgame vs. middlegame, tactical vs. positional).

### 5.5 Results: MoE Model

- Test CE: **TBD** (expected: < 0.71).
- Per‑bucket improvement over base head.
- Ablation: B=4 vs B=8 vs B=16.

### 5.6 Results: Playing Strength

#todo to compute
- ACPL / Elo vs baseline.
- Cfish integration: nps, depth, move time on Wio.

### 5.7 Ablation Studies

#todo maybe
- Effect of cluster number \(B\).
- Effect of dispatcher architecture (linear vs. MLP).
- Effect of fine‑tuning duration (1 vs. 5 epochs).

### 5.8 Visualisation

- t‑SNE / PCA of Sample Gradients.
- Dispatcher decision boundaries.
- Expert activation distribution.

---

## 6. Discussion

### 6.1 Interpretation of Results

- #todo 

### 6.2 Limitations

- #todo 

### 6.3 Generalisation

#todo 
- Can this method be applied to other domains?
- What changes would be needed?

### 6.4 Comparison with Related Work

#todo 
- How does this compare to handcrafted bucketing?
- How does it compare to learned routing in other MoE systems?

---

## 7. Conclusion & Future Work

### 7.1 Summary of Contributions

- #todo

### 7.2 Future Work

- #todo

### 7.3 Closing Remarks

- #todo

---

## Appendices

#todo

---

## Bibliography

- Standard chess engine literature (Stockfish, Cfish).
- NNUE papers (Nasu, Stockfish team).
- MoE literature (Shazeer et al., Fedus et al.).
- AlphaZero / Lc0 (Silver et al., 2017; Leela Chess Zero).

---

## Notes for Writing

- **Length**: About 40-50 pages (Master’s thesis, paper-style).
- **Focus**: The method (Chapter 3) and experiments (Chapter 5) are the core.
- **Novelty**: The unsupervised bucketing via Sample Gradients is the key contribution. The chess application is the validation.
- **Tone**: Technical, clear, with motivation for each design choice.
