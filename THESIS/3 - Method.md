# Method: Unsupervised Bucketing via Sample Gradients

#idea The dispatcher needs only to be called on half of the activations of the dual *accumulator layer*, making it 128x3 parameters in size.

#todo quantization and pruning 

---

## 3.1 Overview

#todo Five-step pipeline: train a base model; compute sample-gradients; cluster them; train a dispatcher; fine-tune expert heads.

## 3.2 Notation and Definitions

We introduce here the formal notation used throughout this chapter and the remainder of the thesis. The notation is organized into sets, scalars, vectors and matrices, functions, and key operators.

### 3.2.1 Sets

| Symbol                                                    | Description                                                                                                                |
| :-------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------- |
| $\mathcal{S}$                                             | The space of all possible chess positions (states).                                                                        |
| $\mathcal{D}$                                           | The training dataset of positions with labels: $\mathcal{D} = \{(s_i, v_i)\}_{i=1}^{N}$.                                 |
| $\mathcal{P} = \{\mathcal{D}_1, \dots, \mathcal{D}_B\}$ | A partition of the state space into $B$ buckets, where each $\mathcal{D}_i \subset \mathcal{S}$ is a non-empty subset. |
| $\Theta$                                                | The space of all model parameters (weights).                                                                               |
| $\mathbb{R}$                                            | The set of real numbers.                                                                                                   |

### 3.2.2 Scalars

| Symbol | Description |
| :--- | :--- |
| $B$ | The number of experts (buckets). |
| $N$ | The total number of positions in the training dataset. |
| $N_i$ | The number of positions in bucket $\mathcal{D}_i$. |
| $h$ | The hidden dimension of the NNUE accumulator. |
| $d_{\text{in}}$ | The input dimension of the NNUE accumulator (844 in this work). |
| $v_i \in \mathbb{R}$ | The scalar label (expected reward) for position $s_i$. |
| $\eta$ | The learning rate used for gradient updates. |

### 3.2.3 Vectors and Matrices

| Symbol | Type | Description |
| :--- | :--- | :--- |
| $W_{L1} \in \mathbb{R}^{d_{\text{in}} \times h}$ | Matrix | Weights of the first layer (accumulator), **frozen** after base training. |
| $W_{L2} \in \mathbb{R}^{h \times h}$ | Matrix | Weights of the second layer. |
| $W_{out} \in \mathbb{R}^{h \times 3}$ | Matrix | Weights of the output layer (3 logits for WDL). |
| $w_{\text{base}} \in \mathbb{R}^{P}$ | Vector | All parameters of the base model: $w_{\text{base}} = \text{vec}(W_{L1}, W_{L2}^{base}, W_{out}^{base})$. |
| $\theta_i \in \mathbb{R}^{P}$ | Vector | All parameters of the expert model $i$: $\theta_i = \text{vec}(W_{L1}, W_{L2}^{(i)}, W_{out}^{(i)})$. |
| $\delta_i \in \mathbb{R}^{P_{\text{head}}}$ | Vector | The **task vector** for expert $i$: $\delta_i = \theta_i^{\text{head}} - \theta_{\text{base}}^{\text{head}}$, where $\theta^{\text{head}} = \text{vec}(W_{L2}, W_{out})$. |
| $\Delta_i \in \mathbb{R}^{P_{\text{head}}}$ | Vector | The **per-sample gradient** for position $s_i$: $\Delta_i = \nabla_{w_{\text{head}}} \mathcal{L}(\hat{f}_{w_{\text{base}}}(s_i), v_i)$. |
| $x_i \in \{0,1\}^{d_{\text{in}}}$ | Vector | The sparse binary feature vector for position $s_i$. |
| $h_i = W_{L1} \cdot x_i \in \mathbb{R}^{h}$ | Vector | The accumulator output (L1 activation) for position $s_i$. |

### 3.2.4 Functions and Models

| Symbol | Description |
| :--- | :--- |
| $f_w: \mathcal{S} \rightarrow \mathbb{R}^3$ | The NNUE evaluation function parameterized by weights $w$, producing WDL logits. |
| $\hat{f}_w: \mathcal{S} \rightarrow \mathbb{R}$ | The NNUE scalar value function: $\hat{f}_w(s) = \text{softmax}(f_w(s))_W - \text{softmax}(f_w(s))_L$. |
| $g_\phi: \mathcal{S} \rightarrow \{1, \dots, B\}$ | The **dispatcher** function, parameterized by $\phi$, mapping a position to a bucket index. |
| $\mathcal{L}(y, v)$ | The loss function, typically soft cross-entropy on WDL probabilities. |
| $\mathcal{L}_{\text{acc}}(y, v)$ | The accumulated loss over a batch or epoch. |

### 3.2.5 Key Operators

| Symbol                                                     | Description                                                           |
| :--------------------------------------------------------- | :-------------------------------------------------------------------- |
| $\text{vec}(\cdot)$                                        | The vectorization operator, flattening a matrix into a column vector. |
| $\nabla_w\,  \mathcal{L}$                                  | The gradient of the loss with respect to the parameters $w$.          |
| $\| \cdot \|$                                              | The Euclidean (L2) norm.                                              |
| $\text{softmax}(z)_k = \frac{e^{z_k}}{\sum_j e^{z_j}}$     | The softmax function over logits $z$.                                 |
| $d_{\text{cos}}(u, v) = 1 - \frac{u \cdot v}{\|u\| \|v\|}$ | The cosine distance between two vectors $u$ and $v$.                  |

---

### 3.2.6 Relationship Between $\Delta_i$ and $\delta_i$

A crucial distinction in this work is between **per-sample gradients** $\Delta_i$ and **task vectors** $\delta_i$:

- $\Delta_i = \nabla_{w_{\text{head}}} \mathcal{L}(\hat{f}_{w_{\text{base}}}(s_i), v_i)$ is the gradient of the loss with respect to the head parameters, evaluated at a **single position** $s_i$ using the base model. It represents the direction in which the head would need to move to reduce the loss on that specific position.

- $\delta_i = \theta_i - \theta_{\text{base}}$ is the actual change in the head parameters after **fine-tuning** on all positions in bucket $\mathcal{D}_i$. It is the accumulated effect of the per-sample gradients over the bucket.

The central hypothesis of this work is that clustering positions by their $\Delta_i$ yields a partition $\mathcal{P}$ for which the resulting $\delta_i$ are maximally diverse, meaning each expert specializes in a distinct region of the state space.

#note this distinction is worth emphasizing 

---

## 3.3 Step 1: Train the Base Model

The first step of our method is to train a **base evaluation model** that will serve as the foundation for all subsequent steps. This model provides two essential functions: it supplies the reference point from which we compute sample gradients, and it contributes the frozen L1 representation used by the dispatcher at inference time.

### 3.3.1 Model Architecture

The base model follows the NNUE architecture described in Section 2.1.3: a sparse accumulator layer $W_{L1}$ that maps a binary feature representation to a hidden state $h \in \mathbb{R}^h$, followed by a small fully-connected head $(W_{L2}, W_{out})$ that produces WDL logits. The architecture is kept deliberately small to reflect the resource constraints of the target hardware—specifically, a hidden dimension of $h = 64$ for the accumulator and $H = 128$ for the L2 layer, resulting in approximately 71,000 trainable parameters.

### 3.3.2 Training Objective

The model is trained to minimize the **soft cross-entropy loss** between its predicted WDL distribution and the teacher labels provided by Lc0 (see Section 4.1.4). For a batch of $N$ positions with teacher probabilities $\hat{p}_i = (\hat{P}_i(W), \hat{P}_i(D), \hat{P}_i(L))$ and model outputs $p_i = (P_i(W), P_i(D), P_i(L))$, the loss is:

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^N \left[ \hat{P}_i(W) \log P_i(W) + \hat{P}_i(D) \log P_i(D) + \hat{P}_i(L) \log P_i(L) \right]$$

#note for me: Soft cross-entropy (or soft-target cross-entropy) is a generalized loss function in machine learning that uses full probability distributions instead of strict "hard" one-hot vectors for target labels

This choice of loss is deliberate. Unlike mean squared error on a scalar value (centipawns or expected reward $v = P(W) - P(L)$), the cross-entropy loss encourages the model to match the **full outcome distribution** rather than just its mean. This provides a richer training signal and naturally handles the non-linear relationship between WDL probabilities and the scalar evaluation used during search.

### 3.3.3 Training Protocol

The base model is trained on the full dataset of approximately 5 million positions using the Adam optimiser with a learning rate of $10^{-2}$, linearly decayed to $10^{-3}$ over the course of training. We use a batch size of 1024 and train until convergence on the held-out test set (5% of the total dataset). All training is performed in PyTorch on a DGX Nvidia Spark GPU.

#todo update hyperparameters like batch size

### 3.3.4 The Role of the Base Model

Once trained, the base model $w_{\text{base}} = (W_{L1}, W_{L2}^{\text{base}}, W_{out}^{\text{base}})$ serves three critical functions in our pipeline:

1. **Reference point for gradients**: For each position $s_i$, we compute the gradient $\Delta_i = \nabla_{w_{\text{head}}} \mathcal{L}(\hat{f}_{w^{\text{base}}}(s_i), v_i)$ of the loss with respect to the head parameters, evaluated at the base model. This gradient represents the direction in which the head would need to move to better fit that specific position—the *learning signal* that drives our bucketing.

2. **Frozen representation for routing**: The L1 weights $W_{L1}$ are **frozen** after base training and are never updated during expert fine-tuning. This ensures that all experts operate on the same shared representation, and that the dispatcher can rely on stable L1 activations $h = W_{L1} \cdot x$ as input features.

3. **Starting point for expert fine-tuning**: Each expert head $(W_{L2}^{(i)}, W_{out}^{(i)})$ is initialised from the base head $(W_{L2}^{\text{base}}, W_{out}^{\text{base}})$ before being fine-tuned on its assigned bucket. This ensures that all experts start from the same well-trained foundation, and that the only difference between them is the data they see during fine-tuning.

### 3.3.5 Why Freeze L1?

Freezing the L1 layer is a deliberate design choice. The accumulator is the most expensive component of the NNUE architecture in terms of parameter count (over 54,000 weights), and updating it during fine-tuning would be computationally prohibitive. More importantly, freezing L1 ensures that the representation space remains stable across all experts: the dispatcher, trained on L1 activations, can reliably route positions without needing to account for different representations. This stability is essential for the lightweight inference pipeline, where the dispatcher must operate with negligible overhead.

#todo make sure that we get across that the accumulator HAS to be the same for all expert, otherwise it doesn't work 

---

## 3.4 Step 2: Compute Sample Gradients

#todo Per-sample gradients w.r.t.\ head parameters. Normalisation (L2 or standardisation). Storage and compute cost.

---

## 3.5 Step 3: Cluster Sample Gradients

#todo K-Means and alternatives (DBSCAN, Density Peak; some choose $B$). How to pick $B$. Validation: inertia, silhouette, interpretability.

---

## 3.6 Step 4: Train the Dispatcher

#todo Input: L1 activations $h$. Output: bucket id $b$. Linear layer or small MLP. Cross-entropy. Needed because gradients are not available at inference.

---

## 3.7 Step 5: Fine-Tune Expert Heads

#todo Freeze L1; fine-tune one head per bucket from the base model. Result: $B$ specialised experts.

---

## 3.8 Generalisation Beyond Chess

#todo The method needs only a state space, a model, a loss, and a target. Other games, robotics, world-model settings.

---

> **Note for AI**: *The parts marked with a #todo are yet to be completed. The tagged comments are NOT to be exported to the Latex document, and are not meant to be implemented while exporting this document to Latex.*

