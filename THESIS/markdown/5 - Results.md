

---

### 5.1 Experimental Setup

<span style="color: #808080;">[Setup]</span>
#todo Splits (train / test sizes). Training on CPU, inference on Wio Terminal. Baselines: single-head NNUE; dense FFNN; handcrafted bucketing (piece count + queen presence).

---

### 5.2 Evaluation Metrics

<span style="color: #808080;">[Metrics]</span>
#todo centipawn-based probabilistic Elo estimator
#todo Test CE on WDL; MAE on expected value (even if the model is trained with CE, not MSE on EV); dispatcher accuracy; per-bucket CE vs.\ the base head.

#todo Elo protocol (not just the name): matches, time control, opponent, BayesElo or SPRT, ACPL. Playing-strength *numbers* go in 5.6.

---

### 5.3 Results 

<span style="color: #808080;">[Results]</span>
#todo We will list the CE of the models, and the Elo estimates...

#### Linear Model

<span style="color: #808080;">[Linear Baseline]</span>
#todo 

#### Single-layer FFNN

<span style="color: #808080;">[Shallow FFNN]</span>
#todo 

#### Double-layer FFNN

<span style="color: #808080;">[Deeper FFNN]</span>
#todo 

#### Base NNUE

<span style="color: #808080;">[Base NNUE]</span>
#todo ...

<span style="color: #808080;">[Dataset Size]</span> We also tested the soft cross-entropy of the base NNUE model as a function of the size of the dataset, from 50k samples, all the way to [dataset_size]. The model has a $2\times[W]$-neuron accumulator layer, and a [H]-neuron L2 layer, which lays on the larger side of the models we trained.

```
python3.12 -u scripts/train_nnue-gpu.py --epochs 100 --lr 0.01 --lr-end 0.001 --hidden-dim 128 --hidden2-dim 256 --test-fraction 0.01 --batch-size 10000 --batches-per-epoch 512 --train-val-subset-size 10000 --run-name dual_h128_H256_e100_bpe512_bs10000 --plot plots/dual_h128_H256_e100_bpe512_bs10000.png
```
#note do NOT modify

<div align="center">
    <img src="THESIS/thesis-plots/variable_dataset_size_ce.png" width="600">
</div>

<span style="color: #808080;">[Overfitting Threshold]</span> We found that, below 7M positions, the model clearly overfits the training set. On the other hand, above 7M positions no overfitting is visible. These figures are to be kept in mind when partitioning a dataset for the MoE, as the per-model data should never exceed this threshold. 


#### MoE NNUE - bucketing by L1 - fixed $B$

<span style="color: #808080;">[L1, Fixed B]</span>
#todo 

#### MoE NNUE - bucketing by L1 - fixed $B$

<span style="color: #808080;">[L1, Fixed B]</span>
#todo 

#### MoE NNUE - bucketing by L1 - variable $B$

<span style="color: #808080;">[L1, Variable B]</span>
#todo 

#### MoE NNUE - bucketing with sample-gradients - fixed $B$

<span style="color: #808080;">[Gradients, Fixed B]</span>
#todo 

#### MoE NNUE - bucketing with sample-gradients - variable $B$

<span style="color: #808080;">[Gradients, Variable B]</span>
#todo 

---

### 5.4 Results: Comparison of the Models

<span style="color: #808080;">[Comparison]</span>
#todo ACPL / Elo vs.\ baselines, using the protocol of 5.2. Cfish on Wio: nps, depth, move time.

---

### 5.5 Visualization: Clustering 

<span style="color: #808080;">[Visualisation]</span>
#todo t-SNE / PCA of sample-gradients. Dispatcher decision boundaries. Expert activation / routing histograms.
#todo exploratory analysis of the clusters - what type of position do they group together?

---

> **Note for AI**: *The parts marked with a #todo are yet to be completed. The tagged comments are NOT to be exported to the Latex document, and are not meant to be implemented while exporting this document to Latex. The gray labels are for clarity only and must not be transferred to the Latex. Placeholders in square brackets (e.g. `[dataset_size]`, `[W]`) must be replaced with the current values from `_ai-info_.md` when converting this document to Latex.*
