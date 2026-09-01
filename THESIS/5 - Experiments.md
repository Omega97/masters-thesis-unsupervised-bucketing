## Experiments and Results

> The parts marked with a #todo are yet to be completed.

---

### 5.1 Experimental Setup

#todo Splits (train / test sizes). Training on CPU, inference on Wio Terminal. Baselines: single-head NNUE; dense FFNN; handcrafted bucketing (piece count + queen presence).

### 5.2 Evaluation Metrics

#todo Test CE; MAE on expected reward; dispatcher accuracy; per-bucket CE vs.\ the base head; playing strength (ACPL / Elo).

### 5.3 Results: Base Model

#todo Snapshot numbers for the single-head NNUE vs.\ a dense FFNN (params, train/test CE, MAE).

### 5.4 Results: Clustering and Dispatcher

#todo Silhouette / cluster quality. Dispatcher accuracy. What the clusters correspond to (phase, tactics, \ldots).

### 5.5 Results: MoE Model

#todo Test CE vs.\ the single-head baseline. Per-bucket gains. Ablation on \(B \in \{4,8,16\}\).

### 5.6 Results: Playing Strength

#todo ACPL / Elo vs.\ baselines. Cfish on Wio: nps, depth, move time.

### 5.7 Ablation Studies

#todo Number of clusters \(B\); dispatcher (linear vs.\ MLP); fine-tuning length.

### 5.8 Visualisation

#todo t-SNE / PCA of sample-gradients. Dispatcher decision boundaries. Expert activation / routing histograms.
