# Method: Unsupervised Bucketing via Sample Gradients

#idea The dispatcher needs only to be called on half of the activations of the dual *accumulator layer*, making it 128x3 parameters in size.

#idea it's not necessary to perform clustering on the whole training set

#todo quantization and pruning 

#todo remember that when we partition the dataset, each expert has less data, but it still has to be "enough" (performance vs number of data)

#todo emphasize the fact that the clustering with sample gradients is not as easy to transpose to inference time as L1-based clustering could be, but with the dispatcher it becomes similarly fast.

---

## 3.1 Overview

#todo Five-step pipeline: train a base model; compute sample gradients; cluster them; train a dispatcher; fine-tune expert heads.

## 3.2 Notation and Definitions

We introduce here the formal notation used throughout this chapter and the remainder of the thesis. The notation is organized into sets, scalars, vectors and matrices, functions, and key operators.

### 3.2.1 Sets

| Symbol                                                  | Description                                                                                                            |
| :------------------------------------------------------ | :--------------------------------------------------------------------------------------------------------------------- |
| $\mathcal{S}$                                           | The space of all possible chess positions (states).                                                                    |
| $\mathcal{D}$                                           | The training dataset of positions with labels: $\mathcal{D} = \{(s_i, v_i)\}_{i=1}^{N}$.                               |
| $\mathcal{P} = \{\mathcal{D}_1, \dots, \mathcal{D}_B\}$ | A partition of the state space into $B$ buckets, where each $\mathcal{D}_i \subset \mathcal{S}$ is a non-empty subset. |
| $\Theta$                                                | The space of all model parameters (weights).                                                                           |


### 3.2.2 Scalars

| Symbol               | Description                                                     |
| :------------------- | :-------------------------------------------------------------- |
| $B$                  | The number of experts (buckets).                                |
| $N$                  | The total number of positions in the training dataset.          |
| $N_i$                | The number of positions in bucket $\mathcal{D}_i$.              |
| $h$                  | The hidden dimension of the NNUE accumulator.                   |
| $d_{\text{in}}$      | The input dimension of the NNUE accumulator (844 in this work). |
| $v_i \in \mathbb{R}$ | The scalar label (expected reward) for position $s_i$.          |
| $\eta$               | The learning rate used for gradient updates.                    |

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

### 3.2.6 Relationship Between $\Delta_i$ and $\delta_i$

A crucial distinction in this work is between **per-sample gradients** $\Delta_i$ and **task vectors** $\delta_i$. These two concepts are similar to each other, but it is worth emphasizing that the task vector $\delta_i = \theta_i - \theta_{\text{base}}$ is the difference in model parameters between before and after the fine-tuning, while the sample gradient $\Delta_i = \nabla_{w_{\text{head}}} \mathcal{L}(\hat{f}_{w_{\text{base}}}(s_i), v_i)$ is the gradient of the loss with respect to the head parameters, evaluated at a **single position** $s_i$ using the base model. 

The central hypothesis of this work is that clustering positions by their $\Delta_i$ yields a partitioning for which the resulting $\delta_i$ are maximally diverse - each expert specializes in a distinct region of the state space - and that the dispatcher is able to approximate this partitioning with enough accuracy to preserve its general structure.

#todo rephrase

---

## 3.3 Step 1: Train the Base Model

The first step of our method is to train a **base evaluation model** that will serve as the foundation for all subsequent steps. This model provides two essential functions: it supplies the reference point from which the sample gradients are computed, and it provides the frozen L1 representation used by the dispatcher at inference time.

### 3.3.1 Model Architecture

The base model follows the NNUE architecture described in Section 2.1.3: a sparse accumulator layer $W_{L1}$ that maps a binary feature representation to a hidden state $h \in \mathbb{R}^h$, followed by a small fully-connected head $(W_{L2}, W_{out})$ that produces WDL logits. The architecture is kept deliberately small to reflect the resource constraints of the target hardware—specifically, a hidden dimension of $h = 64$ for the accumulator and $H = 128$ for the L2 layer, resulting in approximately 71,000 trainable parameters.

#todo remove hard numbers? 

### 3.3.2 Training Objective

The model is trained to minimize the **soft cross-entropy loss** between its predicted WDL distribution and the teacher labels provided by Lc0 (see Section 4.1.4). For a batch of $N$ positions with teacher probabilities $\hat{p}_i = (\hat{P}_i(W), \hat{P}_i(D), \hat{P}_i(L))$ and model outputs $p_i = (P_i(W), P_i(D), P_i(L))$, the loss is:

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^N \left[ \hat{P}_i(W) \log P_i(W) + \hat{P}_i(D) \log P_i(D) + \hat{P}_i(L) \log P_i(L) \right]$$

#note for me: Soft cross-entropy (or soft-target cross-entropy) is a generalized loss function in machine learning that uses full probability distributions instead of strict "hard" one-hot vectors for target labels

This choice of loss is deliberate. Unlike mean squared error on a scalar value (centipawns or expected reward $v = P(W) - P(L)$), the cross-entropy loss encourages the model to match the **full outcome distribution** rather than just its mean. This provides a richer training signal and naturally handles the non-linear relationship between WDL probabilities and the scalar evaluation used during search.

### 3.3.3 Training Protocol

The base model is trained on the full dataset of approximately 5 million positions using the Adam optimiser with a learning rate of $10^{-2}$, linearly decayed to $10^{-3}$ over the course of training. We use a batch size of 1024 and train until convergence on the held-out test set (5% of the total dataset). All training is performed in PyTorch on a DGX Nvidia Spark GPU.

#todo update hyperparameters like batch size

### 3.3.4 The Role of the Base Model

Once trained, the base model $w_{\text{base}} = (W_{L1}, W_{L2}^{\text{base}}, W_{out}^{\text{base}})$ serves several critical functions in our pipeline, one of which is as reference point for gradients: For each position $s_i$, we compute the sample gradient $\Delta_i$ evaluated at the base model. This gradient represents the direction in which the head would need to move to better fit that specific position—the *learning signal* that drives our bucketing.

An other important role of the base model is as frozen representation for routing. The L1 weights $W_{L1}$ are **frozen** after base training and are never updated during expert fine-tuning. This ensures that all experts operate on the same shared representation, and that the dispatcher can rely on stable L1 activations $h = W_{L1} \cdot x$ as input features.

Finally, the base model is of course also the starting point for expert fine-tuning. Each expert head $(W_{L2}^{(i)}, W_{out}^{(i)})$ is initialised from the base head $(W_{L2}^{\text{base}}, W_{out}^{\text{base}})$ before being fine-tuned on its assigned bucket. 

#note here is where the main talk about the "frozen weights" should happen

### 3.3.5 Why Freeze L1?

Freezing the L1 layer is a deliberate design choice. The accumulator is the most expensive component of the NNUE architecture in terms of parameter count, and updating it during fine-tuning would be computationally prohibitive. More importantly, freezing L1 ensures that the representation space remains stable across all experts: the dispatcher, trained on L1 activations, can reliably route positions without needing to account for different representations. This stability is essential for the lightweight inference pipeline, where the dispatcher must operate with negligible overhead.

#todo make sure that we get across that the accumulator HAS to be the same for all expert, otherwise it doesn't work 

---

## 3.4 Step 2: Compute Sample Gradients

With the base model trained, the second step is to compute, for each position in the dataset, the **sample gradient** of the loss with respect to the head parameters. These gradients encode the direction in which the head parameters would need to move to improve the prediction for each individual position. They represent the *learning signal* that we will later use for bucketing.

### 3.4.1 Definition of Sample Gradient

For each position $s_i$ in the dataset $\mathcal{D} = \{(s_i, v_i)\}_{i=1}^N$, the sample gradient is defined as:

$$\Delta_i = \nabla_{w_{\text{head}}} \mathcal{L}(\hat{f}_{w^{\text{base}}}(s_i), v_i)$$

where:

- $w_{\text{head}} = \text{vec}(W_{L2}, W_{out})$ is the vector of all head parameters (the L2 weights and biases, and the output weights and biases)
- $\hat{f}_{w_{\text{base}}}$ is the base model evaluated at the current weights
- $\mathcal{L}$ is the soft cross-entropy loss on WDL probabilities
- $v_i = (p_W, p_D, p_L)$ is the teacher label for position $s_i$

The gradient is computed **at the base model** $w_{\text{base}}$, before any fine-tuning occurs. It represents the instantaneous direction in weight space that would reduce the loss on that specific position, independent of all other positions.

### 3.4.2 Implementation

In practice, PyTorch's `torch.autograd.grad` function simultaneously and efficiently computes the sample gradients for a batch of inputs. As previously discussed, the gradients in question are relative only to the head parameters $(W_{L2}, W_{out})$, not the L1 accumulator weights.

For a batch of positions, we first compute the forward-pass of the model to obtain WDL predictions, and then the cross-entropy loss between predictions and teacher labels. The method `torch.autograd.grad(loss, head_parameters, retain_graph=False)` is called to obtain the gradients. Finally, the resulting gradient tensors are detach and flattened into a single vector per position. The computation is parallelised across the GPU and is performed in a single pass over the dataset. The gradients are stored on disk for later use in the clustering step.

#todo maybe it still sounds a bit like a list converted to prose...


### 3.4.3 Normalisation

The raw gradients can have highly variable magnitudes depending on the position, the current state of the model, and on weather the multiplicity of the state is considered or not. The *L2 normalization* of each gradient vector has been shown to work well in gradient-clustering literature (ELREA, GradientSpace) and preserves the relative angular structure of the gradients.

$$\Delta_i^{\text{norm}} = \frac{\Delta_i}{\|\Delta_i\| + \epsilon}$$

This projects each gradient onto the unit hypersphere, preserving direction while removing magnitude information. This is appropriate because the *direction* of the gradient encodes the type of specialisation needed, while the magnitude is more sensitive to the current loss value and position difficulty.

#todo distinction between L2 norm and layer? silly


### 3.4.4 Storage and Compute Considerations

Computing and storing sample gradients for 5 million positions presents practical challenges. Each gradient vector has dimension $P_{\text{head}} \approx 17,000$ (flattened L2 and output weights). Storing this as 32-bit floats would require approximately:

$$5 \times 10^6 \times 17,000 \times 4 \text{ bytes} \approx 340 \text{ GB}$$
#todo update numbers

#todo considerations on whether to introduce rounding, and whether to use just part of the dataset

---

## 3.5 Step 3: Cluster Sample Gradients

With the normalised sample gradients $\{\Delta_i^{\text{norm}}\}_{i=1}^N$ computed for every position in the dataset, the third step is to partition the data into $B$ clusters (buckets) such that positions with similar learning signals are grouped together. This partition will define the assignment of positions to expert heads during fine-tuning.

### 3.5.1 The Objective of Clustering

Recall the central hypothesis of this work: clustering positions by their sample gradients $\Delta_i$ yields a partition $\mathcal{P} = \{\mathcal{D}_1, \dots, \mathcal{D}_B\}$ for which the resulting task vectors $\delta_i = \theta_i - \theta_{\text{base}}$ are maximally diverse. The role of the clustering algorithm is to discover a partition that approximates this objective. From a practical standpoint, we seek an efficient clustering algorithm that produces clusters that are cohesive, balanced, stable, and, hopefully, as separated as possible. Different clustering algorithms make different trade-offs with respect to these criteria. We consider two broad families: fixed-$B$ algorithms and density-based algorithms.

### 3.5.2 Fixed-$B$ Algorithms: K-Means and Variants

The most widely used clustering algorithm is **K-Means**, which partitions the data into $B$ clusters by minimising the within-cluster sum of squares. For a set of clusters $\{\mathcal{C}_1, \dots, \mathcal{C}_B\}$, K-Means minimises:

$$\sum_{k=1}^B \sum_{i \in \mathcal{C}_k} \|\Delta_i - \mu_k\|^2$$

where $\mu_k = \frac{1}{|\mathcal{C}_k|} \sum_{i \in \mathcal{C}_k} \Delta_i$ is the centroid of cluster $k$. This algorithm is fast and well-understood. Specifically, *Mini-Batch K-Means* is particularly attractive for our scale, as it efficiently handles large datasets by processing data in mini-batches, making it suitable for our dataset, that comprises of millions of chess positions. It reduces memory requirements and converges faster than standard K-Means, while producing nearly identical results.

#todo Disadvantages:
- **Sensitivity to initialisation**: While k-means++ mitigates this, K-Means can still converge to local optima.
- **Assumption of spherical clusters**: K-Means performs best when clusters are roughly spherical and of similar size. Our gradient space may not satisfy this assumption.
- **Fixed $B$ is a hyperparameter**: The choice of $B$ is critical and must be determined separately (see Section 3.5.4).
- **Outlier sensitivity**: K-Means can be influenced by outliers, which may distort centroids.

### 3.5.3 Density-Based Algorithms

An alternative family of algorithms, **density-based clustering**, does not require a fixed number of clusters. Instead, these methods identify clusters as regions of high density separated by regions of low density.

**Density Peak Clustering** (Rodriguez & Laio, 2014) is a particularly relevant algorithm for our setting. It works by identifying cluster centres as points that have high local density (many neighbours within a cutoff distance) and are far from points with higher density (suggesting they are local maxima).

The number of clusters emerges naturally from the data: each point with density higher than all its neighbours and with large distance to the nearest higher-density point is a cluster centre. The remaining points are assigned to the same cluster as their nearest higher-density neighbour.

We chose to use this algorithm because it automatically determines $B$, so it will be interesting to find out what the clusters of board positions actually represent. This algorithm is also notoriously robustness to outliers; points with low density and large distance to higher-density points are naturally identified as outliers.

#todo Disadvantages:
- **Parameter sensitivity**: The algorithm requires choosing a distance cutoff (for density estimation) and a threshold for the distance to higher-density points. These parameters can significantly affect the number of clusters.
- **Computational cost**: Computing pairwise distances for 5 million points is prohibitive (O(N²)). We would need to use approximations (e.g., approximate nearest neighbours) or subsample the data.
- **Cluster size imbalance**: Density-based methods may produce clusters of very different sizes, which can be problematic for expert fine-tuning (some experts would have too little data).
- **Unstable number of clusters**: The number of clusters can vary with the parameters or with slight perturbations in the data, complicating the design of a fixed-architecture MoE.

#todo **DBSCAN** is another density-based algorithm worth considering. It groups points that are closely packed together (high density) and marks points in low-density regions as noise. However, it shares similar challenges with Density Peak: parameter sensitivity (eps, min_samples) and computational cost for large datasets.

### 3.5.4 Choosing $B$ and the Algorithm

The choice between fixed-$B$ and density-based clustering depends on the relative importance of architectural efficiency and data-driven discovery. A density-based approach that automatically determines $B$ could reveal the natural structure in the gradient space, potentially identifying a number of clusters that better reflects the underlying distribution of learning signals. On the other hand, hardware constraints might impose a hard limit on how many expert heads our device is allowed to store based on our architecture of choice.

#todo which approach we end on preferring based on some results

### 3.5.5 Validation and Diagnostics

#todo Once clustering is complete, we validate the quality of the partition using standard metrics: Cluster size distribution, Cosine distance between centroids, Inertia, Silhouette score.

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

