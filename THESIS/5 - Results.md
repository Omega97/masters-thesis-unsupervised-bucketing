

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

#todo ...

We also tested the soft cross-entropy of the base NNUE model as a function of the size of the dataset, from 50k samples, all the way to 5M. The model has a 256-neuron accumulator layer (2x128), and a 256 neuron L2 layer, which lays on the larger side of the models we trained.

```
py -3.12 -u scripts/train_nnue.py --epochs 20 --lr 0.01 --hidden-dim 128 --hidden2-dim 256 --batches-per-epoch 50 --batch-size 1024 --run-name dual_h128_H256_fast_ft97 --plot plots/dual_nnue_ce_128_256_ft97.png --test-subset-size 5000 --train-val-subset-size 5000 --test-fraction 0.97 --fast
```
#todo maybe convert command to table of parameters

<div align="center">
    <img src="THESIS/thesis-plots/nnue_ce_vs_train_size.png" width="600">
</div>

We found that, below 2M positions, the model clearly overfits the training set. On the other hand, above 3M positions no overfitting is visible. These figures are to be kept in mind when partitioning a dataset for the MoE, as the per-model data should never exceed this threshold. 


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

