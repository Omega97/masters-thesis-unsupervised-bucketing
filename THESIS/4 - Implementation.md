## Implementation: NNUE and MoE for Chess

> The parts marked with a #todo are yet to be completed.

---

### 4.1 Dataset and Teacher

#todo Lichess PGNs \(\to\) FEN \(\to\) Lc0 WDL labels (depth 1). About 5M positions. 844-dim encoder (716 base + 128 tactical), dual-POV.

### 4.2 Base NNUE Architecture

#todo L1 \(844 \to 64\) shared dual-POV; L2 \(128 \to 128\); output \(128 \to 3\) WDL logits. CReLU / ReLU / softmax. Soft CE, Adam, epoch budget.

### 4.3 Sample Gradient Computation

#todo `torch.autograd.grad`, per-sample grads w.r.t.\ the head. Storage (`.npy` / `.parquet`).

### 4.4 Clustering

#todo Mini-Batch K-Means; Density Peak; other algorithms. Working \(B\) (e.g.\ 8, Stockfish-style — to check). t-SNE / PCA plots.

### 4.5 Dispatcher Training

#todo Input \([\mathrm{own} \Vert \mathrm{opp}]\) vs.\ own side only. Linear \(2W \to B\) (or \(W \to B\)). CE, Adam, short training.

### 4.6 Expert Fine-Tuning

#todo One sweep per bucket vs.\ longer training. Per-bucket test CE as the stopping signal.

### 4.7 Final MoE Model

#todo Shared L1 + dispatcher + expert heads. Inference: L1 \(\to\) dispatcher \(\to\) selected expert \(\to\) output.

### 4.8 Integration with Cfish

#todo Hook `evaluate()`. Quantization (int8 weights, int16 accumulators, int32 MAC). Memory: sparse L1, dispatcher and heads in flash.
