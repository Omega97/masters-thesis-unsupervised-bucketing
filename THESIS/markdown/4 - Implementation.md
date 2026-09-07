# Implementation: NNUE and MoE for Chess


#idea can/should we focus the dataset on positions hard for the NNUE (eval flipped by 180°)? Is there a trick to let the NNUE figure out the steps ahead, like the solution to a tactic? **More layers but sparse connections**?

#idea **certi algoritmi di clustering decidono $B$** 

#idea U Map in 3D, DBScan, Density Peak Clustering
  
#idea PCA per aiutare il clustering?

#todo we could emphasize the hardest to read positions, but with quiescent search we hardly ever have to evaluate those correctly
  
#todo visualizzare i cluster; plot per la tesi

#todo to specify why we use cross-entropy loss

#todo estimate number of value function calls

---

## 4.1 Dataset and Teacher

<span style="color: #808080;">[Data Overview]</span> The quality of an NNUE evaluation function depends critically on the dataset used for training. For this work, we constructed a dataset of approximately **[dataset_size] chess positions** extracted from human games and labeled with high-quality value estimates from a strong teacher network, **Leela Chess Zero** (Lc0), the *spiritual* successor of *AlphaZero*.

#todo update hyperparameters

### 4.1.1 Dataset Construction

<span style="color: #808080;">[Sources]</span> The raw positions are sourced from **Lichess** monthly PGN archives, containing standard-rated games played by humans across all time controls. We filter games to include only those with at least 16 moves, excluding very short games that often end in early blunders or resignations and would introduce noisy or uninformative positions into the training set ( #todo omit?). From each remaining game, we sample positions randomly with probability [keep_prob], ensuring a diverse and representative collection of states across all phases of play, and mostly avoiding highly correlated positions.

<span style="color: #808080;">[Phase Mix]</span> The dataset retains the natural distribution of game phases found in human play: a majority of middlegame positions, with fewer openings and endgames. We intentionally avoid resampling to balance phases, as the natural distribution better reflects the positions the engine will encounter during actual play. Each position is stored as a FEN string along with the multiplicity of the position—a visit count that may be used for optional weighting during training.

<span style="color: #808080;">[Tactical Extra]</span> A small fraction of the dataset is also a collection of interesting tactical positions, and high level bot games. They capture a portion of the state space that may be outside of regular human play.

### 4.1.2 Feature Encoding

<span style="color: #808080;">[Encoding]</span> For the NNUE model, each FEN string is encoded as a **sparse binary feature vector** of length $d_{\text{in}}=[d_in]$. This encoding is designed to capture both the positional and tactical structure of the board in a form suitable for the accumulator layer.

<span style="color: #808080;">[Feature Split]</span> The $d_{\text{in}}$ features are divided into two categories:

- **716 base features**: These encode piece-square pairs, representing the presence of each piece type on each square. The feature set is pruned to remove impossible pawn ranks and compressed to reduce redundancy (e.g., the king plane is stored in a compact form). #todo explain better? In the **[d_in]‑dim SARDINE encoder**, the king plane is **compressed** from 64 squares to **32**, saving features.

- **128 tactical features**: These encode dynamic aspects of the position, specifically which pieces are under attack and which pieces are attacking the king. These features provide the network with explicit information about immediate tactical threats. #todo is it worth trying without these? I don't think so...

#todo list to prose

<span style="color: #808080;">[Dual POV]</span> A key design choice is the **dual‑POV** encoding: for each position, the encoder produces two sets of sparse indices—one from the perspective of the **side‑to‑move** (STM) and one from the **opponent's** perspective (obtained by flipping the board rank-wise and swapping colors). This dual representation allows the network to learn symmetric evaluations and is consistent with the NNUE architecture's ability to evaluate positions from either player's viewpoint.

<span style="color: #808080;">[Storage]</span> The encoded features are pre‑computed and stored in `.npz` slices for efficient loading during training.

### 4.1.3 Data Splits

<span style="color: #808080;">[Splits]</span> The dataset is partitioned into training and test splits. The **training set** consists of approximately [dataset_size] positions, distributed across 165 slices for balanced I/O and stochastic sampling. The **test set** comprises a random portion of [test_fraction] of the position that are held out of the training set.
#todo remember to update numbers when they change...

### 4.1.4 Teacher Model: Lc0

<span style="color: #808080;">[Teacher]</span> To provide accurate target labels, we use **Lc0** (Leela Chess Zero) as the teacher model. Lc0 is a *convolutional neural network* trained via self‑play reinforcement learning, following the AlphaZero paradigm. Its value head outputs a probability distribution over the three possible game outcomes—Win, Draw, Loss—from the perspective of the side‑to‑move:

$$p_{\text{WDL}}(s) = \text{softmax}(\text{logits}(s)) = (p_W, p_D, p_L)$$

From this distribution, we compute the scalar expected reward:

$$v(s) = p_W - p_L \in [-1, +1]$$

which represents the expected outcome of the game from the current position. This scalar is the training target for the NNUE value head.

<span style="color: #808080;">[Why Lc0]</span> Lc0 is chosen as the teacher for several reasons. First, it natively outputs WDL probabilities, which align directly with the NNUE's output head. Second, its strength—rated well above 3500 Elo—makes it a highly reliable source of positional evaluations.  Finally, Lc0 is open‑source and provides pre‑trained networks, making the labelling pipeline reproducible.

<span style="color: #808080;">[Labelling Setup]</span> For this work, we label positions using Lc0's **latest best network** (e.g., `791556.pb.gz` from the Lc0 training server). We run Lc0 in UCI mode with `--show-wdl` enabled and evaluate each position with a single MCTS search. While depth‑1 evaluations may occasionally miss short‑term tactics, the resulting label noise is acceptable given the target Elo range of the engine (approximately 1700). For a cleaner but more expensive relabelling, one could increase the search depth.
#todo part of the dataset is already at depth 2...

### 4.1.5 Labelling Pipeline

<span style="color: #808080;">[Pipeline]</span> The complete labelling pipeline is straightforward. We parse Lichess PGNs and sample positions uniformly at random from each game, saving FEN strings and visit counts. This reduces the correlation between the positions in the final dataset. For each unique FEN, we invoke Lc0 in *UCI mode* at **depth 1** to obtain WDL probabilities from the STM perspective. We then procede to save the WDL probabilities alongside the FEN and visit counts in JSON format. In the encoding step we pre‑compute the [d_in]‑dimensional sparse feature vectors (both STM and opponent POVs) and store them in `.npz` slices for efficient training. The final result is a dataset of pairs of sparse input board positions and their relative WDL probabilities.

#todo UCI mode? STM perspective?
#todo specify WDL and depth 1?
#todo ensure the prose around it is not just a list of bullet points in disguise.

---

## 4.2 Base NNUE Architecture

<span style="color: #808080;">[Architecture]</span> The base NNUE (Efficiently Updatable Neural Network) model serves as the foundation upon which the Mixture of Experts extension is built. Its architecture is designed to balance representational capacity with the stringent memory and computational constraints of the target hardware. The model follows the dual‑perspective paradigm introduced by the original NNUE design, but incorporates modifications tailored to the specific feature set and bucketing objectives of this work.

### 4.2.1 Model Overview

<span style="color: #808080;">[Overview]</span> The base model $f_\theta$ is a feed‑forward neural network with three parameterised layers: a shared sparse first layer (L1), a dense second layer (L2), and a linear output head, with *softmax* activations. The input to the model is the sparse binary feature representation described in Section 4.1.2, consisting of [d_in] active features per perspective. The model processes both the side‑to‑move (STM) and opponent perspectives through the same L1 layer, producing two accumulator vectors that are later concatenated and passed through the remaining layers.

<span style="color: #808080;">[Forward Pass]</span> Formally, the model computes:

$$
h_{\text{own}} = W_{\text{L1}} \, x_{\text{own}}, \qquad
h_{\text{opp}} = W_{\text{L1}} \, x_{\text{opp}},
$$

where $x_{\text{own}}, x_{\text{opp}} \in \{0,1\}^{[d_in]}$ are the sparse feature vectors for the two perspectives, and $W_{\text{L1}} \in \mathbb{R}^{d_{\text{in}} \times W}$ is the shared weight matrix of the accumulator layer. The output of the L1 layer is a pair of vectors $h_{\text{own}}, h_{\text{opp}} \in \mathbb{R}^W$, where $W$ is the hidden dimension of the accumulator, set to $W = [W]$ in this work.

<div align="center">
    <img src="THESIS/thesis-plots/sardine_nnue_architecture.png" width="600">
</div>

### 4.2.2 Shared L1 Accumulator

<span style="color: #808080;">[Accumulator]</span> The L1 layer, often referred to as the accumulator, is the defining component of the NNUE architecture. Its weight matrix $W_{\text{L1}}$ is shared between the two perspectives, enabling the network to learn a common representation of board structure while retaining perspective‑specific information through the different input features. The sparsity of the input features allows the accumulator to be updated efficiently: rather than recomputing the entire matrix‑vector product for each new position, the network maintains the accumulator vector incrementally, adding or subtracting the contributions of features that change as pieces move.

<span style="color: #808080;">[CReLU]</span> The L1 activations are passed through a **CReLU** (Clipped Rectified Linear Unit) activation function, which maps each activation to the range $[0, 127]$:

$$
a_{\text{own}} = \text{clamp}(h_{\text{own}}, 0, 127), \qquad
a_{\text{opp}} = \text{clamp}(h_{\text{opp}}, 0, 127).
$$

This clipping is essential for integer quantization, as it bounds the dynamic range of the accumulator values and allows the use of low‑precision integer arithmetic during inference.

### 4.2.3 Side‑to‑Move Reorder

<span style="color: #808080;">[STM Reorder]</span> Before concatenating the two accumulator vectors, we apply a **side‑to‑move (STM) reorder** to ensure that the expert head always receives the perspective of the current player first. The reordering is governed by the binary flag $\text{stm\_white} \in \{0,1\}$, which indicates whether White is to move:

$$
h_{\text{first}} =
\begin{cases}
a_{\text{own}}, & \text{if } \text{stm\_white} = 1, \\
a_{\text{opp}}, & \text{otherwise},
\end{cases}
\qquad
h_{\text{second}} =
\begin{cases}
a_{\text{opp}}, & \text{if } \text{stm\_white} = 1, \\
a_{\text{own}}, & \text{otherwise}.
\end{cases}
$$

<span style="color: #808080;">[Concatenation]</span> The two vectors are then concatenated to form the input to the L2 layer:

$$
h = [h_{\text{first}} \, \| \, h_{\text{second}}] \in \mathbb{R}^{2W}.
$$

<span style="color: #808080;">[Perspective Invariance]</span> This reordering step is critical for making the evaluation perspective‑invariant: the network always receives the board from the point of view of the side to move, and the output $v \in [-1, +1]$ is always interpreted as the expected reward for the current player, regardless of colour.

### 4.2.4 L2 Layer and Output Head

<span style="color: #808080;">[L2 Layer]</span> The concatenated vector $h$ is passed through a dense L2 layer with hidden dimension $H = [H]$:

$$
z = \text{ReLU}(W_{\text{L2}} \, h + b_{\text{L2}}),
$$

where $W_{\text{L2}} \in \mathbb{R}^{2W \times H}$ and $b_{\text{L2}} \in \mathbb{R}^H$ are the weight matrix and bias of the L2 layer. The ReLU activation introduces non‑linearity and has been shown to work well with the sparse accumulator features.

<span style="color: #808080;">[Output Head]</span> Finally, the L2 activations are projected to a **three‑dimensional output** representing the logits for Win, Draw, and Loss probabilities:

$$
\text{logits} = W_{\text{out}} \, z + b_{\text{out}},
$$

where $W_{\text{out}} \in \mathbb{R}^{H \times 3}$ and $b_{\text{out}} \in \mathbb{R}^3$. During training, these logits are converted to probabilities via the softmax function:

$$
p_{\text{WDL}}(s) = \text{softmax}(\text{logits}(s)) = (p_W, p_D, p_L).
$$

<span style="color: #808080;">[Scalar Eval]</span> The scalar evaluation used for search is obtained as $v = p_W - p_L$, the expected reward from the side‑to‑move perspective.

### 4.2.5 Training Objective

<span style="color: #808080;">[Soft-CE]</span> The model is trained to minimise the **soft cross‑entropy** loss between the predicted WDL probabilities and the teacher labels provided by Lc0. For a batch of positions with targets $y_i = (\hat P_i(W), \hat P_i(D), \hat P_i(L))$ and model outputs $p_i = (P_i(W), P_i(D), P_i(L))$, the loss is:

$$
\mathcal{L} = -\frac{1}{N} \sum_{i=1}^N \left[\hat P_i(W) \log P_i(W) + \hat P_i(D) \log P_i(D) + \hat P_i(L) \log P_i(L) \right].
$$
#todo soft-CE? better notation?

<span style="color: #808080;">[Why This Loss]</span> This loss is well‑suited for the task because the teacher labels are probability distributions, not point estimates. Using soft cross‑entropy encourages the model to *match the full outcome distribution* rather than merely the scalar expected reward, providing a richer training signal. Additionally, the loss naturally handles the non‑linearity of the scalar evaluation via the softmax, and its gradient does not saturate at extreme values, unlike the squared error on $v$.
#todo I have to think about it 

### 4.2.6 Parameter Count and Model Size

<span style="color: #808080;">[Model Size]</span> With $W = [W]$ and $H = [H]$, the total number of trainable parameters is approximately [n_params]. This compact size is deliberately chosen to fit within the memory constraints of the target hardware: the L1 weights ($[d_in] \times [W]$ int8 values) dominate the parameter count, while the L2 and output layers contribute only a small fraction. The model is therefore both computationally efficient and storage‑friendly, with a footprint that can be further reduced through pruning and quantisation.
#todo naming convention for the number of neurons per layer...

### 4.2.7 Training Protocol

<span style="color: #808080;">[Training Protocol]</span> The base model is trained on the full training set (approximately [dataset_size] positions) using the Adam optimiser with a learning rate of [lr_start], linearly decayed to [lr_end] over the course of training. We use a batch size of [batch_size] and train for up to 1000 epochs. The model's performance is evaluated on a held‑out test set of random positions, [test_fraction] of the total dataset, ensuring that generalisation is measured on unseen data. The training is conducted on a *DGX Nvidia Spark GPU*.

#todo update numbers: number of positions, batch size

---

## 4.3 Sample Gradient Computation


### 4.3.1 Implementation with PyTorch

The sample gradients are computed using PyTorch's automatic differentiation engine. For each batch of positions, a forward pass through the base model is performed to obtain WDL logits. Then, the soft cross-entropy loss is computed against the teacher labels. Finally, `torch.autograd.grad` is called to obtain the gradients of the loss with respect to the head parameters.

The computation proceeds as follows. For a batch of $M$ positions, the base model $f_{w_{\text{base}}}$ produces logits $\ell_i \in \mathbb{R}^3$ for each position. The loss is computed as the average soft cross-entropy over the batch:

$$\mathcal{L}_{\text{batch}} = \frac{1}{M} \sum_{i=1}^M \mathcal{L}_{\text{CE}}(\text{softmax}(\ell_i), \hat{p}_i)$$

where $\hat{p}_i$ are the teacher's WDL probabilities. We then call:

```python
grads = torch.autograd.grad(
    loss,
    head_parameters,
    retain_graph=False,
    create_graph=False
)
```

The `head_parameters` list contains the trainable parameters of the L2 layer and the output head: $(W_{L2}, b_{L2}, W_{out}, b_{out})$. The L1 weights are excluded, as they remain frozen throughout the entire pipeline.

#todo imply the bias?

The resulting gradient tensors are detached from the computation graph to free memory, then flattened and concatenated into a single vector per position:

$$\Delta_i = \text{concat}\left[\text{vec}(\nabla_{W_{L2}} \mathcal{L}_i),\; \nabla_{b_{L2}} \mathcal{L}_i,\; \text{vec}(\nabla_{W_{out}} \mathcal{L}_i),\; \nabla_{b_{out}} \mathcal{L}_i\right]$$

The computation is parallelised across the GPU and performed in a single pass over the dataset. Gradients are computed in batches of 1024 positions, and the resulting vectors are accumulated on disk rather than held in memory, preventing memory exhaustion. The `retain_graph=False` option ensures that the computational graph is freed after each batch, further reducing memory usage.

#todo check batch size

### 4.3.2 Parameter Selection

#todo skip?
- Which parameters are included: `(W_{L2}, b_{L2}, W_{out}, b_{out})`
- Why the L1 accumulator is excluded (frozen, shared, computationally expensive)
- Vectorisation and flattening of gradients for clustering

### 4.3.3 Normalisation

#todo skip?
- L2 normalisation of each gradient vector
- Implementation details: `grad / (norm + eps)`
- Whether the same normalisation is applied to all gradients

### 4.3.4 Storage and Memory Management

Computing and storing sample gradients for 5 million positions presents a significant practical challenge. Each gradient vector has dimension $P_{\text{head}} \approx 17,000$, corresponding to the flattened parameters of the L2 layer and output head. Storing the full set in 32-bit floating-point would require approximately 340 GB. We therefore adopt half-precision storage, reducing this to roughly 170 GB while preserving sufficient numerical precision for clustering, as the gradients are normalised and clustered based on their directions rather than their exact magnitudes.

The gradients are stored in memory-mapped `.npy` files, which provide efficient random access without loading the entire dataset into memory. Computation proceeds in batches of 1024 positions, with each batch written to disk immediately after computation and the memory-mapped array pre-allocated to avoid accumulating gradients in GPU or CPU memory. For exploratory analysis, gradient computation can be performed on a subset of the dataset, but for the final MoE architecture we use the full dataset. 

#todo considerations on dimensionality reduction?
#todo replace hard numbers with variables 


### 4.3.5 Computational Cost and Timing

#todo
- Time required to compute gradients for 5 million positions
- GPU vs CPU considerations
- Batching strategy and throughput



---

## 4.4 Clustering

<span style="color: #808080;">[Clustering]</span>
#todo Mini-Batch K-Means; Density Peak; other algorithms. Working $B$ (e.g.\ 8, Stockfish-style — to check). t-SNE / PCA plots.

---

## 4.5 Dispatcher Training

<span style="color: #808080;">[Dispatcher]</span>
#todo Input $[\mathrm{own} \Vert \mathrm{opp}]$ vs.\ own side only. Linear $2W \to B$ (or $W \to B$). CE, Adam, short training.

---

## 4.6 Expert Fine-Tuning

<span style="color: #808080;">[Fine-Tuning]</span>
#todo One sweep per bucket vs.\ longer training. Per-bucket test CE as the stopping signal.

---

## 4.7 Final MoE Model

<span style="color: #808080;">[Final MoE]</span>
#todo Shared L1 + dispatcher + expert heads. Inference: L1 $\to$ dispatcher $\to$ selected expert $\to$ output.

---

## 4.8 Cfish as the host engine

<span style="color: #808080;">[Host Engine]</span>
#todo Instantiate 2.1.6: what Cfish is (C port of Stockfish), `evaluate()`, search tree (α-β, iterative deepening, transposition table), Wio constraints (RAM, flash, nps).

---

## 4.9 Integration with Cfish

<span style="color: #808080;">[Integration]</span>
#todo Hook `evaluate()` with the MoE NNUE. Quantization (int8 weights, int16 accumulators, int32 MAC). Memory: sparse L1, dispatcher and heads in flash. 

---

> **Note for AI**: *The parts marked with a #todo are yet to be completed. The tagged comments are NOT to be exported to the Latex document, and are not meant to be implemented while exporting this document to Latex. The gray labels are for clarity only and must not be transferred to the Latex. Placeholders in square brackets (e.g. `[dataset_size]`, `[W]`) must be replaced with the current values from `_ai-info_.md` when converting this document to Latex.*
