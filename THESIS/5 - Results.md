
#todo testare la performance di un modello (architettura fissa) in base alla dimensione del dataset!

---

### 5.1 Experimental Setup

#todo Splits (train / test sizes). Training on CPU, inference on Wio Terminal. Baselines: single-head NNUE; dense FFNN; handcrafted bucketing (piece count + queen presence).

---

### 5.2 Evaluation Metrics

#todo centipawn-based probabilistic Elo estimator
#todo Test CE on WDL; MAE on expected value (even if the model is trained with CE, not MSE on EV); dispatcher accuracy; per-bucket CE vs.\ the base head.

#todo Elo protocol (not just the name): matches, time control, opponent, BayesElo or SPRT, ACPL. Playing-strength *numbers* go in 5.6.

---

### 5.3 Results 

#todo We will list the CE of the models, and the Elo estimates...

#### Linear Model

#todo 

#### Single-layer FFNN

#todo 

#### Double-layer FFNN

#todo 

#### Base NNUE


#### MoE NNUE - bucketing by L1 - fixed $B$

#todo 

#### MoE NNUE - bucketing by L1 - fixed $B$

#todo 

#### MoE NNUE - bucketing by L1 - variable $B$

#todo 

#### MoE NNUE - bucketing with sample-gradients - fixed $B$

#todo 

#### MoE NNUE - bucketing with sample-gradients - variable $B$

#todo 

---

### 5.4 Results: Comparison of the Models

#todo ACPL / Elo vs.\ baselines, using the protocol of 5.2. Cfish on Wio: nps, depth, move time.

---

### 5.5 Visualization: Clustering 

#todo t-SNE / PCA of sample-gradients. Dispatcher decision boundaries. Expert activation / routing histograms.
#todo exploratory analysis of the clusters - what type of position do they group together?

---

> **Note for AI**: *The parts marked with a #todo are yet to be completed. The tagged comments are NOT to be exported to the Latex document, and are not meant to be implemented while exporting this document to Latex.*

