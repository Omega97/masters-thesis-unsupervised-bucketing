# 5 Experimental Results

## 5.1 Experimental Setup

### 5.1.1 Experimental Configuration

\[Describe the dataset, train/validation/test splits, hardware, training environment, target Wio Terminal, and relevant implementation details.\]

### 5.1.2 Experimental Models and Baselines

\[Describe the models and baselines evaluated throughout the experiments, including the linear model, shallow and deeper FFNNs, base NNUE, handcrafted bucketing, activation-based bucketing, and gradient-based MoE variants.\]

### 5.1.3 Evaluation Protocol

\[Describe the general evaluation protocol, including the test set, teacher targets, chess-engine configuration, and playing-strength evaluation procedure.\]

---

## 5.2 Clustering

### 5.2.1 Clustering Algorithm

<span style="color: #808080;">[Sampling]</span> The clustering is run on subsamples of the training split rather than on the full corpus of roughly 150 million positions. Exploratory experiments use one million positions drawn from the seed-0 training split (the subsample cached for the base model, with the held-out test positions excluded), and the final partitions are fitted on two million positions. In all cases the sample gradients are computed with respect to the head parameters of the trained base model ($W=128$, $H=256$), L2-normalised so that each vector lies on the unit hypersphere, and the clustering therefore operates on directions rather than magnitudes.

<span style="color: #808080;">[Representations]</span> Three candidate representations are clustered in parallel, each evaluated for a range of cluster counts $B$. The first is the raw board encoding, the side-to-move-ordered concatenation of the two 844-dimensional binary views, giving a 1688-dimensional vector. The second is the L1 activation space, the concatenation of the two 128-dimensional accumulator vectors after the side-to-move reorder, giving a 256-dimensional vector. The third is the proposed signal, the sample gradients. Because the flattened head gradient is very high-dimensional, it is projected to a 48-dimensional space through a fixed random (Gaussian) projection before clustering. The board and L1 representations serve as the activation-based references, while the gradients constitute the signal under test.

<span style="color: #808080;">[Algorithms]</span> The primary algorithm is Mini-Batch K-Means with $k$-means++ initialisation, a mini-batch size of $10\,000$, and a fixed random seed, run for $B \in \{2, 4, 8, 16\}$. As a density-based alternative that does not require $B$, DBSCAN is applied targeting eight clusters; on the board and L1 representations it is run in a 48-dimensional PCA of the data because the nearest-neighbour density estimates in the native width are not usable. A handcrafted reference partition of eight bins is obtained by bucketing on the scalar piece count, mirroring the conventional NNUE bucketing. The final gradient-based partitions used for the mixture-of-experts are fitted on two million positions for $B \in \{2, 4, 8\}$.

### 5.2.2 Clustering Metrics

<span style="color: #808080;">[Metrics]</span> The quality of each partition is assessed with the diagnostics introduced in Section 3.5.5: the silhouette score, which measures how well each point fits its own cluster relative to the nearest other cluster; the within-cluster inertia, which measures compactness; and the pairwise cosine distance between cluster centroids, reported both as a mean and as a minimum, which serves as a proxy for the diversity of the learning signals captured by the buckets. Cluster balance is monitored through the size distribution of the buckets, since a bucket that is too small cannot support stable expert fine-tuning.

<span style="color: #808080;">[Agreement]</span> In addition, the agreement between partitions is quantified with the adjusted Rand index (ARI) and the normalised mutual information (NMI), both of which measure how well two assignments coincide while correcting for chance. These are used to compare the activation-based and gradient-based partitions with one another and with the piece-count reference. Finally, the stability of the gradient partition is assessed by refitting the clustering on a larger sample and checking that the silhouette and the centroid geometry are preserved as the number of samples increases.

### 5.2.3 Clustering Results

<span style="color: #808080;">[Comparison of representations]</span> Table 5.1 reports the mini-batch K-Means diagnostics across the three representations and the piece-count reference. Two findings stand out. First, the gradient-based partitions are markedly better separated than the activation-based ones: for $B=2$ the two gradient centroids are nearly antipodal, with a cosine distance of $1.94$ (a cosine similarity of $-0.94$), and even at $B=16$ the mean cosine distance remains above $1.0$. In contrast, the board and L1 centroids remain close to one another at every value of $B$, with cosine distances well below $0.3$. Second, the gradient partitions achieve the highest silhouette scores of any learned representation at every value of $B$, whereas the L1 silhouette collapses to near zero, and eventually negative, as $B$ grows.

| Representation | $B$ | Silhouette | Cosine dist. (mean) | Cosine dist. (min) | Min. share | Max. share |
| :--- | :-: | :-: | :-: | :-: | :-: | :-: |
| board | 2 | 0.112 | 0.235 | 0.235 | 0.313 | 0.687 |
| board | 4 | 0.075 | 0.221 | 0.120 | 0.074 | 0.431 |
| board | 8 | 0.041 | 0.257 | 0.043 | 0.062 | 0.285 |
| board | 16 | 0.019 | 0.288 | 0.061 | 0.017 | 0.159 |
| L1 | 2 | 0.154 | 0.174 | 0.174 | 0.275 | 0.725 |
| L1 | 4 | 0.106 | 0.208 | 0.115 | 0.104 | 0.541 |
| L1 | 8 | 0.005 | 0.203 | 0.069 | 0.044 | 0.249 |
| L1 | 16 | −0.005 | 0.244 | 0.057 | 0.015 | 0.173 |
| gradients | 2 | 0.250 | 1.938 | 1.938 | 0.441 | 0.559 |
| gradients | 4 | 0.196 | 1.229 | 0.607 | 0.151 | 0.361 |
| gradients | 8 | 0.191 | 1.126 | 0.367 | 0.086 | 0.182 |
| gradients | 16 | 0.134 | 1.049 | 0.123 | 0.034 | 0.131 |
| piece count | 8 | 0.606 | 0.000 | 0.000 | 0.069 | 0.187 |

*Table 5.1 — Mini-Batch K-Means clustering diagnostics for the three representations and the piece-count reference, on one million training positions. Silhouette is estimated on a $10\,000$-point subsample for the gradient runs and a $2\,000$-point subsample for the board and L1 runs; the share columns give the smallest and largest bucket as a fraction of the sample.*

<span style="color: #808080;">[Piece-count reference]</span> The piece-count reference attains the highest silhouette score ($0.61$), but this is an artefact of its one-dimensional nature: contiguous, well-separated bins along a single scalar are trivial to recover. Its centroid cosine distance is identically zero, since all bucket means are positive one-dimensional numbers that point in the same direction. The piece-count partition therefore looks compact by silhouette but conveys nothing about the learning signal, and it should not be read as a well-formed candidate for expert specialisation.

<span style="color: #808080;">[DBSCAN]</span> The density-based alternative is summarised in Table 5.2. DBSCAN recovers very few clusters from each representation — three from the board encoding, two from the L1 activations, and five from the gradients — despite targeting eight, and it flags a very large fraction of the fit sample as noise: $39.6\%$ for the board encoding, $57.2\%$ for the L1 activations, and $75.2\%$ for the gradients. The recovered clusters are heavily imbalanced, each dominated by a single large core with one or more small satellites, and their silhouette scores, while superficially higher than those of K-Means on the board and L1 data, reflect that dominant core rather than a meaningful multimodal structure. This is consistent with the expectation set out in Section 3.5.3 that density-based clustering is unreliable in a high-dimensional gradient space, and it motivates the use of a fixed $B$ for the mixture-of-experts.

| Representation | Clusters | Bucket sizes | Noise (fit subset) | Silhouette |
| :--- | :-: | :--- | :-: | :-: |
| board | 3 | 78.2% / 10.5% / 11.3% | 39.6% | 0.172 |
| L1 | 2 | 85.6% / 14.4% | 57.2% | 0.217 |
| gradients | 5 | 21.1% / 23.1% / 7.8% / 40.2% / 7.7% | 75.2% | 0.152 |

*Table 5.2 — DBSCAN results targeting eight clusters, fitted on a $100\,000$-point subsample with `min_samples`$=80$. Bucket sizes are the cluster shares of the full one-million-position assignment; the noise column gives the fraction of the $100\,000$-point fit flagged as noise.*

<span style="color: #808080;">[Partition agreement]</span> Table 5.3 reports the agreement between the partitions at $B=8$. The board and L1 partitions agree only moderately with each other (NMI $0.24$) and agree weakly with the piece-count reference, suggesting that even activation-based partitions capture structure beyond the game phase. The gradient partition, by contrast, is nearly orthogonal to every other partition: its agreement with the board and L1 assignments is at the level of chance (ARI $0.06$–$0.08$), and its agreement with the piece-count bucketing is essentially zero (ARI $0.04$, NMI $0.09$). This is precisely the behaviour the method is designed to induce: clustering the learning signal recovers a partition that does not simply reproduce the board geometry, the internal representation, or the conventional game-phase bucketing.

| Pair | ARI | NMI |
| :--- | :-: | :-: |
| board vs. L1 | 0.124 | 0.236 |
| board vs. gradients | 0.057 | 0.063 |
| L1 vs. gradients | 0.078 | 0.150 |
| board vs. piece count | 0.189 | 0.303 |
| L1 vs. piece count | 0.147 | 0.290 |
| gradients vs. piece count | 0.042 | 0.088 |

*Table 5.3 — Agreement (adjusted Rand index and normalised mutual information) between partitions at $B=8$, computed on one million positions. Higher values indicate closer agreement; the chance baseline is $0$.*

<span style="color: #808080;">[Stability]</span> The final gradient-based clustering, fitted on two million positions for $B \in \{2, 4, 8\}$, reproduces the structure observed on one million positions. The silhouette scores are $0.250$, $0.215$, and $0.179$ for $B=2$, $4$, and $8$ respectively, within $0.02$ of the one-million-position estimates ($0.250$, $0.196$, and $0.191$), and the mean centroid separation is preserved: the mean off-diagonal centroid cosine is $-0.938$, $-0.307$, and $-0.128$, in line with the $-0.938$, $-0.229$, and $-0.126$ observed on the smaller sample. The buckets remain well balanced at each value of $B$, ranging from $55.8\%/44.2\%$ at $B=2$ to between $10.8\%$ and $16.0\%$ at $B=8$. The partition is therefore stable with respect to the sample size. As $B$ grows the centroids spread out and their mean pairwise similarity approaches zero: at $B=2$ they are nearly antipodal, while at $B=8$ the pairwise centroid cosines range from strongly negative (near $-0.95$) to moderately aligned (near $+0.72$), consistent with a partition that tiles a single dense gradient manifold rather than isolating well-separated modes.

<div align="center">
    <img src="RESULTS/clustering/plots/previous_gradient_2m/silhouette.png" width="600">
</div>

<span style="color: #808080;">[Visualisation]</span> The qualitative evidence supports the quantitative picture. Projections of the gradient space onto the first principal components, and the corresponding t-SNE and UMAP embeddings, show a single connected cloud along which the clusters form contiguous regions, consistent with the low silhouette values and the gradual loss of centroid separation as $B$ increases. In contrast, the L1 and board projections show clusters that are poorly separated and strongly overlapping, mirroring their near-zero cosine distances. The PCA projection of the gradient-based $B=2$ partition in particular shows the two clusters lying on opposite sides of the origin, in line with the antipodal centroids reported above.

<div align="center">
    <img src="RESULTS/clustering/plots/pca_2d_gradients.png" width="600">
</div>

<div align="center">
    <img src="RESULTS/clustering/plots/tsne_2d_gradients.png" width="600">
</div>

<span style="color: #808080;">[Interpretation]</span> Taken together, the results indicate that the gradient-based partition is the most promising candidate for expert specialisation. The antipodal centroids at $B=2$ confirm that the two buckets correspond to opposite directions in head-parameter space — positions whose gradients pull the head in contrary ways — which is exactly the kind of diversity the method seeks. The near-zero agreement with the piece-count and activation-based partitions further suggests that these buckets do not correspond to coarse game-phase or geometric categories, but to a genuinely different structure grounded in the learning dynamics. The implication for the mixture-of-experts is that the gradient partition should induce expert task vectors that are more distinct from one another than those produced by any activation-based bucketing, a hypothesis examined in Section 5.5.

---

## 5.3 Dispatcher

### 5.3.1 Dispatcher Model

\[Describe the lightweight dispatcher, its input representation, architecture, number of parameters, training procedure, and inference cost.\]

\[Explain how the dispatcher is trained to approximate the offline clustering partition and how it differs from directly performing clustering at inference time.\]

### 5.3.2 Dispatcher Metrics

\[Measure the agreement between the dispatcher predictions and the reference clustering. Report classification accuracy, Macro-F1, ARI/NMI, and the confusion matrix as appropriate.\]

\[Also evaluate the severity of routing errors by measuring the similarity between a sample's gradient and the centroid of the predicted cluster, compared with its reference cluster.\]

\[Include Random and Dummy dispatchers as baselines.\]

### 5.3.3 Dispatcher Results

\[Report the dispatcher performance and compare it with the Random, Dummy, and other relevant routing baselines.\]

\[Analyze whether incorrect assignments tend to occur between nearby clusters in gradient space and whether the dispatcher provides a sufficiently accurate approximation of the offline partition.\]

\[Report the computational and memory overhead of the dispatcher.\]

---

## 5.4 Base Model

### 5.4.1 Base Model Architecture and Training

\[Describe the base NNUE architecture, training procedure, WDL targets, loss function, and relevant hyperparameters.\]

\[Describe the simpler neural baselines used to establish the capacity and computational requirements of the evaluation function.\]

### 5.4.2 Base Model Metrics

\[Report test soft cross-entropy on WDL predictions and MAE on expected value.\]

\[Report model size, parameter count, inference cost, and, where applicable, playing-strength metrics such as Elo, ACPL, depth, or nodes per second.\]

\[For the base NNUE, evaluate performance as a function of training-set size to identify the point at which additional data no longer provides substantial benefit or where overfitting becomes relevant.\]

### 5.4.3 Base Model Results

\[Present the results for the linear model, shallow FFNN, deeper FFNN, and base NNUE.\]

\[Present the dataset-size experiment and identify the training-data regime used for the subsequent MoE experiments.\]

\[Establish the base NNUE as the reference model against which the specialized models are evaluated.\]

---

## 5.5 Mixture of Experts

### 5.5.1 MoE Architecture and Training

\[Describe the shared L1 representation, expert heads, bucket assignment, expert-training procedure, and inference-time routing.\]

\[Describe the activation-based and gradient-based bucketing approaches.\]

\[Separate experiments with fixed $B$ from experiments in which $B$ is varied.\]

\[Explain how the available training data is partitioned among experts and how the dataset-size constraints identified for the base model affect expert training.\]

### 5.5.2 MoE Metrics

\[Evaluate WDL cross-entropy and expected-value MAE for the complete MoE.\]

\[Report per-bucket cross-entropy to determine whether individual experts specialize relative to the base head.\]

\[Compare Oracle routing with learned dispatcher routing where appropriate.\]

\[Report model size, memory footprint, inference latency, nodes per second, and other relevant computational costs.\]

\[For the complete engine, report the playing-strength metrics defined by the evaluation protocol.\]

### 5.5.3 MoE Results

\[Present the results for activation-based bucketing with fixed $B$.\]

\[Present the results for activation-based bucketing with variable $B$.\]

\[Present the results for gradient-based bucketing with fixed $B$.\]

\[Present the results for gradient-based bucketing with variable $B$.\]

\[Compare the resulting expert specialization, predictive quality, and computational cost across the different bucketing strategies.\]

\[Distinguish the effect of the partition itself from the effect of the learned dispatcher by comparing Oracle and learned routing where applicable.\]

---

## 5.6 Summary of Experimental Findings

\[Summarize the main empirical findings without introducing new analysis.\]

\[State which experimental observations will be examined in greater depth in the Discussion, including clustering stability, dispatcher approximation, expert specialization, predictive performance, and computational trade-offs.\]