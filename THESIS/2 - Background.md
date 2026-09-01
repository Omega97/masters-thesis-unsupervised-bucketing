## Background and Related Work

> The parts marked with a #todo are yet to be completed.

---

### 2.1 Chess Engines and Evaluation Functions

#todo Classical evaluation (material, piece-square tables, mobility, king safety). NNUE as a sparse, accumulator-based evaluator. Alpha-beta search, iterative deepening, and move ordering, and why the eval must be cheap.

### 2.2 Mixture of Experts

#todo Multiple expert networks over sub-domains. Learned vs.\ fixed routing. Uses in vision, NLP, and RL, and what carries over to a tiny chess eval.

### 2.3 Sample Gradients

#todo Per-example gradients w.r.t.\ head parameters as a representation of the learning signal. Why they differ from activations. Pointers to gradient-clustering literature.

### 2.4 Bucketing in NNUE

#todo Handcrafted buckets: piece count, king location, piece presence (queen, bishop pair, \ldots). Why they are cheap, and why they may not match what the model needs.

### 2.5 Related Work

#todo Efficiency (pruning, quantization). Distillation / teacher–student. Self-play RL (AlphaZero, Lc0). Work closest to sample-gradient clustering and MoE routing (see `related-work.md`).
