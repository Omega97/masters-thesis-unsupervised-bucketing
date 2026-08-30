
## Introduction

> The parts marked with a #todo are yet to be completed.

---

### 1.1 Motivation

#todo
- Chess engines on embedded devices (Wio Terminal: 192 KB RAM, 500 KB flash).
- Need for efficient evaluation functions (NNUE).
- Handcrafted bucketing (piece count, queen presence) is heuristic and may not capture the true structure of the state space.

### 1.2 Problem Statement

#todo
- How can we automatically learn a partition of the state space that maximises expert specialisation?
- How can we do this without human supervision?

### 1.3 Contributions

#todo
- A novel **unsupervised bucketing method** based on sample gradients (gradients w.r.t. head parameters). 
- A lightweight **dispatcher** that routes positions to the right expert at inference time.
- A **Mixture of Experts (MoE)** NNUE architecture for resource‑constrained devices.
- Empirical validation on chess, showing improved evaluation quality over a single‑head baseline.

### 1.4 Thesis Outline

#todo
*Brief roadmap of the chapters.*