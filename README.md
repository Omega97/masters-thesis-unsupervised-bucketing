
# Unsupervised State Bucketing for Mixture of Experts in Resource‑Constrained Chess Engines

#todo title is work-in-progress

| Disclaimer<br>The text in this document is *handwritten*. AI was only used to help with phrasing and formatting. |
| ---------------------------------------------------------------------------------------------------------------- |

---

## 🎯 The Goal of this Thesis

> _We propose a novel approach for bucketing states in chess. Instead of relying on handcrafted features (piece count, game phase, etc.), we learn the bucketing directly from the data._
> *The key idea is to train a **base model**, then compute for each data point a **sample gradient** – the direction in weight space that the neural network would like to take in order to better internalise that example. By clustering these sample gradients, we partition the state space into **meaningful buckets** without any human supervision.*
> *This technique is meth to be lightweight, and to transcend the game of chess.*

---

## 🔬 The Training Method

The pipeline consists of five main steps:

### 1. Train a Base Model

We first train a **single** NNUE model $f_w(s) = v$ on a large dataset of chess positions labelled with the **WDL probabilities** (probability to win, draw, or lose, given by strong teacher, e.g., *Lc0*). 

The result is a base evaluator with parameters: 

$$
w = (w^{(\text{L1})}, w^{(\text{L2})}, w^{(\text{out})})
$$

This model serves as the starting point for all subsequent steps.

### 2. Compute Sample Gradient

For each position $s_i$ in the dataset, we compute the **sample gradient**:

$$
\delta_i = \nabla_{w^{(\text{head)}}} \, \mathcal L(f_w(s_i), \hat v_i)
$$

where $w^{(\text{head})} = (w^{(\text{L2})}, w^{(\text{out})})$ are the parameters of the **expert head** (L2 + output). This gradient indicates the direction in weight space that would improve the model's prediction for that specific example. The interpretation is that positions that “want” similar updates are grouped together – they require similar specialisation.

### 3. Cluster Sample Gradients

We apply the clustering algorithm of choice (like **K‑Means**, **DBSCAN**, or **Density Peak Clustering**) to the normalised *sample gradients* $\hat \delta_i$ to produce $B$ clusters. Each cluster defines a **bucket** of positions that share a similar learning signal.

### 4. Train a Dispatcher

Since **sample gradients cannot be computed at inference time** (they require gradients and correct labels for $\hat v$), we train a **dispatcher** $g_\phi(h)$ to predict the bucket ID from the **L1 activations** $h$.

$$
b = \arg\max g_\phi(h)
$$

We end up with the dispatcher, a lightweight classifier (e.g., a linear layer) trained on $(h_i, b_i)$ pairs.

### 5. Fine‑Tune Expert Heads

We run the dispatcher once on the entire dataset to re-compute the labels. #todo optional?
For each bucket $b$, we fine‑tune the **expert head** $w^{(\text{head})}_b$ on the bucket’s data, starting from the base model. The L1 weights remain frozen, ensuring that all experts share the same representation. This gives us a set of $B$ expert models, where each one is specialized on its own aspect of the game, that the algorithm decided organically.

---

## 🚙 Inference

The forward pass for the model is quite straightforward:

1. Compute the L1 layer (in the specific case of the NNUE it's a [dual accumulator layer](https://en.wikipedia.org/wiki/Efficiently_updatable_neural_network))
    
2. Run the dispatcher $g_\phi(h)$ to get the bucket index $b = \arg\max g_\phi(h)$
     
3. Pass the L1 layer to the corresponding *expert*. 

---

## 📊 Results

### Base Models

| Arch                   | Run                   | Width                                      | Params  | train_ce | **test_ce** |
| ---------------------- | --------------------- | ------------------------------------------ | ------- | -------- | ----------- |
| **Linear** (no hidden) | `linear_wdl_smoke`    | In 844×2 <br>Out 3                         | 5 067   | 0.7760   | **0.7645**  |
| **1 hidden** (medium)  | `medium_h20`          | In 844×2 <br>L1 20 CReLU <br>Out 3         | 33 843  | 0.7149   | **0.6872**  |
| **2 hidden** (NNUE)    | `dual_h128_H256_fast` | In 844×2 <br>L1 128×2, <br>L2 256<br>Out 3 | 174 723 | 0.6797   | **0.6966**  |
#todo more training runs need to be performed with the NNUE


### MoE with Handcrafted Bucketing

#todo elo


### MoE with L1 Clustering

#todo elo


### MoE with Sample Gradient Bucketing (our approach)

#todo elo



---

## 📝 Key References

- [NNUE: Efficiently Updatable Neural Networks](https://en.wikipedia.org/wiki/Efficiently_updatable_neural_network)
    
- [SARDINE Engine Blueprint]() – My related internship project 
    
- [Goal.md](https://github.com/Omega97/TinyML_Internship/blob/main/Goal.md) – High‑level project goals

---

#core