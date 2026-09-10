# Background and Related Work

#todo **AlphaZero vs. Cfish**: Clearly separate MCTS (AlphaZero/Lc0) from alpha-beta search (Stockfish/Cfish) to avoid confusion.

#todo **WDL vs. EV vs. centipawns**: Clarify early that you use WDL + soft-CE, and why.

---

## 2.1 Chess Engines and Evaluation Functions

<span style="color: #808080;">[Role of Eval]</span> At the core of every chess engine lies an **evaluation function** $f: \mathcal{S} \rightarrow \mathbb{R}$ that assigns a scalar value to a board position, indicating the expected outcome from that position (typically from the perspective of the player to move). This value guides the search algorithm—usually a variant of **alpha-beta search** with iterative deepening—by pruning unpromising branches and selecting the most promising moves.

### 2.1.1 Classical Evaluation

<span style="color: #808080;">[Classical Form]</span> For decades, chess evaluation functions were handcrafted. A classical evaluator is a weighted linear combination of chess-specific features:

$$f(s) = \sum_{k} w_k \cdot \phi_k(s)$$
where $\phi_k(s)$ are feature functions and $w_k$ are scalar weights. Typical *chess-specific* features include material balance, piece-square tables (PSTs), mobility, king safety, pawn structure. These features were carefully tuned by chess experts over decades, using a combination of intuition, empirical testing, and later, automated optimization techniques.

<span style="color: #808080;">[PeSTO]</span> A notable example is **PeSTO** (Piece-Square Tables Only), an evaluation function by Ronald Friederich that relies exclusively on piece-square tables. Its tables are optimized via *Texel*'s tuning method and use a *tapered evaluation* that interpolates between separate opening and endgame tables based on the game stage. Conceptually, PeSTO is equivalent to a linear feed‑forward network: both compute a weighted sum of piece‑square features, with no interaction terms between pieces. While still a handcrafted linear form, PeSTO demonstrates how far such classical approaches can be pushed through data-driven tuning.

<span style="color: #808080;">[Limitations of HCE]</span> While classical evaluators are extremely fast, they suffer from fundamental limitations. The human-designed features encode the human intuition about chess, which may not align with the optimal understanding of the game. Moreover, these models do not take into account the positions as a whole; the same parameters give a positional bonus for a central knight whether the position is a quiet middlegame or a tactical melee, even though the knight's practical value may differ dramatically across phases.

<span style="color: #808080;">[Limitations II - Horizon Effect]</span> Classical evaluators are also vulnerable to the _horizon effect_. A search algorithm constrained to a fixed depth evaluates positions at the search frontier as if the game were stable at that point. If a tactical sequence begins at the horizon but extends one ply further, the engine cannot see the consequences and may assign an inaccurate evaluation. Classical evaluators, which rely on static features such as material balance and piece-square tables, are particularly susceptible: they treat a position at the horizon as if it were quiet, ignoring the tactical possibilities that would unfold if the search continued. In contrast, models trained on complete game outcomes can often encode patterns that extend beyond immediate material and positional features, making them less vulnerable to horizon-induced misevaluations.

#nota Manteniamo una discussione generale sulle limitazioni dei feature artigianali (sono statici, non contestuali)

### 2.1.2 The Shift to Neural Evaluation

<span style="color: #808080;">[Neural Eval]</span> The limitations of handcrafted features led to the adoption of neural networks as evaluation functions. **AlphaZero** (Silver et al., 2018) demonstrated that a deep convolutional network, trained via self-play reinforcement learning, could surpass the best classical engines. However, these networks are computationally expensive—requiring millions of operations per evaluation—making them unsuitable for resource-constrained devices or for engines that must evaluate millions of positions per second.

<span style="color: #808080;">[AlphaZero value function]</span> AlphaZero's value head outputs a scalar $v\in[−1,+1]$, interpreted as the expected game outcome from the current player's perspective. This scalar is the training target for the value network. However, AlphaZero's training and inference rely on **Monte Carlo Tree Search (MCTS)**, which builds a search tree by repeatedly simulating trajectories and using the neural network to evaluate leaf nodes. This process requires many forward passes of the network per position, making it computationally expensive and poorly suited for engines that evaluate millions of positions per second with alpha-beta search. The AlphaZero network itself is a deep residual architecture with millions of parameters, requiring floating-point operations and substantial memory, which far exceeds the capacity of microcontrollers. In contrast, the NNUE architecture adopted in this work reduces the per-evaluation cost to a handful of integer operations while retaining the representational power of a neural network. We revisit AlphaZero's value formulation in Section 2.1.4, where we contrast its scalar expected value with the WDL distribution used in this work.

### 2.1.3 NNUE: A Hybrid Approach

<span style="color: #808080;">[Hybrid Design]</span> The **Efficiently Updatable Neural Network (NNUE)** architecture, first introduced in the *shogi* engine *YaneuraOu* and later adopted by *Stockfish*, strikes a pragmatic balance. NNUE combines the representational power of a neural network with the incremental efficiency of classical evaluators.

<span style="color: #808080;">[Two Ingredients]</span> The architecture consists of two ingredients: a sparse first layer called the *accumulator*, and a small fully-connected head. 

<span style="color: #808080;">[Accumulator]</span> The accumulator layer maps a binary representation of the board to a hidden representation $h \in \mathbb{R}^{d}$ via $h = W_{L1} \cdot x$, where $x$ is a sparse binary vector (typically 768 or 1024 dimensions) indicating the presence of pieces on squares. The key insight is that a chess move changes only a few bits of $x$, so $h$ can be **updated incrementally** by adding and subtracting the corresponding columns of $W_{L1}$, rather than by recomputing the entire matrix-vector product from scratch. This makes the computation of L1 essentially *free*.

<span style="color: #808080;">[Head / WDL]</span> The second ingredient, the fully connected head, is typically one or two hidden layers followed by a scalar output, mapping $h$ to the position value. In our case, we found it easier to train a probability distribution across all the three possible game result for the player; win, draw, and loss (WDL), by minimizing the cross-entropy between the output of the teacher and the student.

<span style="color: #808080;">[Search / Horizon]</span> A complete NNUE engine also leverages alpha-beta search with iterative deepening: starting from the root position, the engine searches deeper and deeper, using the NNUE evaluation at leaf nodes to guide the pruning and ordering of moves. At every node of this search tree, the evaluation function is called hundreds or thousands of times, making its speed critical. Also, being trained on enormous amounts of data, a NNUE doesn't suffer from the *horizon problem* as much as a PST does. 

#todo Keep this subsection conceptual (accumulator, head, incremental update). Own dims / CReLU / 3-way WDL belong in 4.2, not here.

### 2.1.4 Value Targets: Centipawns, Expected Value, and WDL

<span style="color: #808080;">[Which Value Targets]</span> A fundamental design choice in any chess evaluation function is the representation of the target value. Different engines adopt different representations, each with implications for training, calibration, and search integration. Three main approaches have emerged in practice: scalar centipawns, scalar expected value, and full WDL distributions.

#### 2.1.4.1 Scalar centipawns
<span style="color: #808080;">[Centipawns]</span> Classical evaluators and Stockfish-style NNUEs output a scalar value in centipawns, representing the expected advantage from the current player's perspective. A positive value indicates an advantage for the side to move; a negative value indicates a disadvantage. This representation is simple, interpretable, and compatible with existing search algorithms. However, it compresses the uncertainty of the game outcome into a single number: a position with a 100 centipawn advantage might be a forced win, a quiet positional advantage, or a tactical trap, all of which require different handling during search. Moreover, training a network to predict centipawns typically uses mean squared error, which treats all deviations equally regardless of whether they lie near the decision boundary.

#todo mention problems: centipawns don't have a cap, don't represent probability of victory, are NOT 1% of the value of a pawn (the value of a pawn changes based on the position)

#### 2.1.4.2 Scalar expected value (EV)
<span style="color: #808080;">[EV]</span> AlphaZero (Silver et al., 2018) adopted a different scalar representation: $v \in [-1, +1]$, interpreted as the expected game outcome from the current player's perspective. This value is the difference between the probability of a win and the probability of a loss: $v = p_W - p_L$. A value of $+1$ indicates a certain win, $-1$ a certain loss, and $0$ a draw or perfectly balanced position. This representation is more naturally calibrated to game outcomes than centipawns and avoids the arbitrary scaling of classical evaluation. However, like centipawns, it compresses the full distribution into a single scalar, losing information about the probability of a draw. A position with $v = 0$ could be a balanced middlegame, a drawish endgame, or a position where the engine is equally uncertain about win and loss—all of which have different implications for search.

#### 2.1.4.3 Full WDL distribution
<span style="color: #808080;">[Full WDL distribution]</span> The approach adopted in this work is to train the NNUE to output a full probability distribution over the three possible game outcomes: Win, Draw, and Loss. The output head produces three logits, converted to probabilities via softmax:

$$p_{\text{WDL}}(s) = (p_W, p_D, p_L), \qquad p_W + p_D + p_L = 1$$

The scalar expected value used during search is then derived as $v = p_W - p_L$, but the network is trained to match the full distribution rather than just the scalar. This representation has several advantages. First, it preserves information about uncertainty: a position with $p_W = 0, p_D = 1, p_L = 0$ has the same scalar value as one with $p_W = 0.5, p_D = 0, p_L = 0.5$, but the two positions are fundamentally different. Second, it enables the use of **soft cross-entropy** as the loss function, which provides a richer training signal than mean squared error: the network is encouraged to match the full shape of the distribution, not just its mean. Third, the loss does not saturate at extreme values, as squared error on $v$ would. Finally, the softmax output is naturally calibrated as a probability distribution, which can be useful for downstream tasks such as move selection or search heuristics.

#### 2.1.4.4 Choice of loss and its implications 
<span style="color: #808080;">[Loss]</span> The choice of target representation determines the loss function. For scalar targets (centipawns or EV), mean squared error is the natural choice. For WDL distributions, soft cross-entropy is more appropriate. In this work we adopt the WDL representation with soft cross-entropy loss, as it provides the richest training signal while still yielding a scalar value compatible with alpha-beta search. This choice is reflected in the architecture (three output neurons instead of one) and in the teacher model (Lc0 natively outputs WDL probabilities). We revisit the practical implications of this choice in Chapter 4, where we describe the training procedure in detail.

### 2.1.5 Why Evaluation Must Be Cheap

<span style="color: #808080;">[Search Cost]</span> The search tree explored by an **alpha-beta engine** grows exponentially with depth. Even with effective move ordering and pruning techniques, the number of evaluated positions increases dramatically as the search goes deeper. A chess engine that searches deeper consistently outperforms one that searches shallower, as each additional ply reveals tactical patterns and strategic nuances that would otherwise remain hidden.

<span style="color: #808080;">[Depth Trade-off]</span> This means that any overhead added to the evaluation function is directly subtracted from the engine's effective search depth. If the evaluator becomes slower, the engine must reduce its search depth to maintain the same response time—sacrificing playing strength in the process.

<span style="color: #808080;">[Embedded Constraint]</span> On an embedded device such as the Wio Terminal (192 KB RAM, 500 KB flash), this constraint is even more severe. Memory is limited, floating-point operations are expensive, and every instruction counts. Any routing mechanism for a mixture-of-experts NNUE must add only a trivial cost—ideally, a handful of integer operations or a simple table look-up—to avoid degrading the engine's search performance.

<span style="color: #808080;">[Core Trade-off]</span> In short, the evaluation function must be expressive enough to assess positions accurately, yet cheap enough to be called millions of times during a game. This trade-off is the central engineering challenge addressed by this thesis.

#todo One sentence: playing strength is reported in Elo / ACPL; protocol in Chapter 5.

### 2.1.6 Current State of the Art for Tiny Hardware: Cfish

<span style="color: #808080;">[What is Cfish]</span> For resource-constrained devices, the most relevant reference implementation is **Cfish**, a port of the Stockfish chess engine written in plain C by Ronald de Man. While Stockfish is written in C++ and targets desktop-class hardware, Cfish strips away the C++ abstractions and compiles to a much smaller binary, making it viable for microcontrollers and other embedded platforms. The engine can be compiled with various evaluation backends, including a pure NNUE mode that excludes the classical handcrafted evaluation entirely, resulting in an even smaller executable.

<span style="color: #808080;">[Architecture]</span> From an architectural standpoint, Cfish shares the same core search and evaluation logic as Stockfish. The input to the evaluation function is a board position represented internally as a compact bitboard structure. The NNUE evaluation itself follows the _halfkp_ feature set: for each piece on the board, the network considers the piece's square together with the square of the king of the same color, producing a set of activated features that are fed into the accumulator. The accumulator is updated incrementally as moves are made, avoiding recomputation from scratch. The output of the NNUE is a scalar value, typically in centipawns, representing the expected advantage from the perspective of the side to move.

<span style="color: #808080;">[Why Relevant]</span> Cfish is particularly relevant to this thesis for two reasons. First, it demonstrates that a full-featured NNUE engine _can_ be made to run on constrained hardware, provided the implementation is careful about memory layout and avoids C++ overhead. Second, it serves as a practical baseline: any Mixture-of-Experts NNUE architecture proposed for tiny devices should be at least as fast and compact as Cfish in its pure NNUE mode, while offering improved evaluation accuracy through expert specialization. In this sense, Cfish represents both the _state of the art_ and the _performance target_ for the embedded chess engine developed in this work.

#todo Pointer: Cfish as host engine (evaluate(), α-β, ID, TT, Wio, MoE hook) is instantiated in 4.8–4.9. Keep 2.1 conceptual.

---

## 2.2 Mixture of Experts

<span style="color: #808080;">[What is MoE]</span> The **Mixture of Experts (MoE)** architecture is a neural network design pattern in which multiple specialized sub-networks, or *experts*, are combined through a routing mechanism that selects or weights their contributions based on the input. The central idea is that different regions of the input space may require different processing, and dedicating separate capacity to each region can improve overall performance without substantially increasing the cost of a single forward pass.

### 2.2.1 Fixed vs. Learned Routing

<span style="color: #808080;">[Two categories of MoE]</span> MoE architectures can be broadly divided into two categories based on how the routing is determined: *fixed routing* and *learned routing*.

#### 2.2.1.1 Fixed routing
<span style="color: #808080;">[Predefined rule]</span> In *fixed routing* schemes, the assignment of inputs to experts is determined by a predefined rule, often based on domain knowledge. In chess engines, this corresponds to handcrafted bucketing: positions are assigned to buckets based on material count, piece presence, or game phase. The routing is deterministic, interpretable, and computationally cheap, but it relies on human intuition about which regions of the state space are meaningfully distinct. The rule is fixed after design and cannot adapt to the data.

#### 2.2.1.2 Learned routing 
<span style="color: #808080;">[Learned rule]</span> In *learned routing* schemes, a trainable *gating network* (or router) learns to assign inputs to experts based on the input features themselves. The gating network typically produces a probability distribution over experts, and the final output is a weighted combination of expert outputs, or a single expert selected by argmax. This approach is more flexible: the router can learn to assign inputs to experts in ways that may not align with human intuition, potentially discovering structure in the data that handcrafted rules would miss. However, learned routing introduces additional parameters and computational cost, and it may require careful design to avoid load imbalance or mode collapse.

### 2.2.2 Applications in Vision, NLP, and Reinforcement Learning

<span style="color: #808080;">[MoE Applicaitons]</span> Mixture of Experts has been successfully applied across a wide range of domains. In natural language processing, large-scale MoE models such as the Switch Transformer (Fedus et al., 2021) and GLaM (Du et al., 2022) achieve state-of-the-art performance by scaling the number of parameters while keeping per-token computation constant: only a subset of experts is activated for each token. In computer vision, MoE architectures have been used to efficiently scale convolutional networks and vision transformers, where different experts specialise in different visual patterns or object classes. In reinforcement learning, MoE has been applied to multi-task and multi-domain settings, where different experts specialise in different tasks or environments, and a gating network selects the appropriate expert for the current context.

#note we are going really wide with the references

<span style="color: #808080;">[MoE in Microcontrollers]</span> Despite their success in these domains, MoE architectures are rarely deployed in resource-constrained settings such as microcontrollers. The gating network and the multiple expert heads introduce memory and computational overhead that is acceptable on servers but prohibitive on embedded devices. Moreover, many MoE implementations rely on sparse activation (only a subset of experts is used per forward pass) and require specialised hardware support for efficient execution. These constraints are less severe in the chess domain, where the evaluation function must be simple and fast, but they still shape the design choices of this work.

### 2.2.3 LoRA as a Lightweight Expert Implementation

<span style="color: #808080;">[How does LoRA work?]</span> In the context of large language models, a common approach to implementing experts is **Low-Rank Adaptation (LoRA)** (Hu et al., 2021). LoRA freezes the base model's weights and injects trainable low-rank matrices into each layer, enabling efficient fine-tuning with a small number of additional parameters. Each expert can be represented by a set of LoRA adapters that modify the base model's behaviour in a task-specific or domain-specific way.

<span style="color: #808080;">[ELREA = LoRA + MoE]</span> **ELREA** (Li et al., ICLR 2025) and **GradientSpace** (Sridharan et al., 2025) both adopt this paradigm. In ELREA, training instructions are partitioned by their gradient directions, and a LoRA expert is fine-tuned on each partition. In GradientSpace, LoRA gradients are clustered, and a lightweight encoder-based router selects the appropriate LoRA expert for each input. These approaches demonstrate that gradient-informed partitioning can be effective, and they provide the closest methodological precedent for the work presented in this thesis.

<span style="color: #808080;">[Why LoRA is not a good fit?]</span> However, LoRA is designed for large transformer models where the base model has hundreds of millions or billions of parameters. In our setting, the base model is a tiny NNUE with approximately [model_size] parameters, and the head that we specialise is already small. LoRA is therefore neither necessary nor appropriate: the expert heads are implemented as separate instances of the L2 and output layers, initialised from the base head and fine-tuned on their assigned buckets. This is simpler, more memory-efficient, and better suited to the integer quantisation required for deployment on microcontrollers.

#idea should we store the experts as corrections to base models? probably no, we would have to re-calculate too much...

### 2.2.4 What Carries Over to a Tiny Chess Evaluation

<span style="color: #808080;">[Gradient-based partitioning]</span> From the broader MoE literature, the key ideas that inform this work are: the principle of partitioning the input space to enable specialisation; the distinction between fixed and learned routing; and the observation that gradient-based clustering can be used to discover meaningful partitions. However, the specific constraints of embedded chess engines impose a different set of trade-offs. The routing mechanism must be nearly free, ruling out expensive gating networks; the expert heads must be small and integer-friendly, ruling out LoRA adapters; and the partition must be learned from value estimation targets rather than instruction-following data. This thesis adapts the **gradient-based partitioning** paradigm to these constraints, proposing a lightweight linear dispatcher that routes positions to specialised heads with negligible overhead, and validating the approach on a chess NNUE for resource-constrained devices.

---

## 2.3 Sample Gradients


### 2.3.1 Per-Sample Gradients 

#todo Per-Sample Gradients 

### 2.3.2 Gradient Similarity and Specialization 

#todo Gradient Similarity and Specialization 

### 2.3.3 Gradient-Based Clustering

#todo Gradient-Based Clustering

<span style="color: #808080;">[Learning Signal]</span>
#todo Per-example gradients w.r.t.\ head parameters as a representation of the learning signal. Why they differ from activations. Pointers to gradient-clustering literature.

---

## 2.4 Bucketing in NNUE


### 2.4.1 Handcrafted Bucketing 

<span style="color: #808080;">[Phase-Based Bucketing]</span> In NNUE-based chess engines, the state space is often partitioned into discrete buckets using manually designed rules. Each bucket corresponds to a region of the state space where positions are assumed to share similar characteristics, and a separate set of output weights is trained for each bucket. This approach, sometimes referred to as _phase-based bucketing_, is a pragmatic compromise: it allows the evaluation function to adapt to different types of positions while keeping the per‑head network small and fast.

<span style="color: #808080;">[Features]</span> Common features used for bucketing include the total material count (the number of pieces remaining on the board), the presence or absence of specific pieces such as queens or bishops, the location of the kings, and the overall game phase derived from material. For example, Stockfish historically used a phase classification that interpolates between opening and endgame evaluations based on material remaining: positions with many pieces are treated as openings or middlegames, while positions with few pieces are treated as endgames. The Kaggle FIDE & Google Efficient Chess AI Challenge popularised a variant where the game is divided into three phases—opening, middlegame, endgame—based on material thresholds, with separate evaluation heads for each phase.

<span style="color: #808080;">[Cheap and Interpretable]</span> These handcrafted bucketing schemes are computationally cheap: the features required for routing are simple integer counts and bitwise operations that add negligible overhead to the evaluation function. They are also interpretable: a chess programmer can inspect the buckets and understand why a position is assigned to a particular region.

<span style="color: #808080;">[Limitations of HCB]</span> However, handcrafted bucketing suffers from fundamental limitations. The features are designed based on human intuition about chess, which may not align with the actual structure of the learning signal. A position in the middlegame with a queen on the board may require a very different adjustment to the evaluation head depending on whether it is a quiet positional struggle or a tactical melee—yet both are assigned to the same bucket based on material count. Conversely, two positions that appear superficially different (e.g., an endgame with a rook vs. a middlegame with heavy pieces) may require similar adjustments to the head, but are placed in different buckets. In other words, handcrafted features encode what a programmer _thinks_ are distinct regimes, not what the model _needs_ in order to specialise. This limitation motivates the exploration of data-driven alternatives, where the partition is learned directly from the training signal rather than prescribed by chess expertise.

#nota discussione _specifica_ sulle limitazioni del bucketing (regole di partizionamento euristiche, discrepanza tra fase di gioco e segnale di apprendimento, etc.).

### 2.4.2 Learned Bucketing

<span style="color: #808080;">[Motivation]</span> The limitations of handcrafted bucketing have motivated research into data-driven alternatives, where the partition of the state space is learned directly from the training data rather than prescribed by human intuition. These approaches can be broadly divided into two categories: those that learn the partition from the model's internal representations, and those that learn a routing policy that selects among experts based on the input state.

#### 2.4.2.1 Clustering hidden-layer activations
<span style="color: #808080;">[Limitations of clustering by activations]</span> One natural approach is to cluster the activations of the network's hidden layers, grouping positions that the model already represents similarly. For example, unsupervised concept discovery methods have been applied to decompose the activation space of chess networks such as AlphaZero. The intuition is that positions that produce similar hidden representations are likely to require similar processing from the subsequent layers, making them natural candidates for the same expert head. This approach has the advantage of being fully data-driven and requiring no chess-specific feature engineering. However, it suffers from a fundamental limitation: activations capture what the model *knows*, not what it *needs* to adjust. Two positions may produce similar hidden representations because the model already evaluates them correctly, yet require very different updates to the head; conversely, positions with dissimilar activations may require similar adjustments. The partition is defined by the model's current state, not by the learning signal that drives specialisation. This limitation is precisely what motivates the gradient-based approach proposed in this thesis.

#### 2.4.2.2 Learned routing for mixture-of-experts
<span style="color: #808080;">[Limitations of rounting big MoE models]</span> An alternative to clustering is to learn a routing policy that directly selects among experts based on the input state. In the broader machine learning literature, mixture-of-experts architectures typically employ a learnable gating network that produces a weighted combination of expert outputs. In the chess domain, recent work has explored learned routing for expert selection. **Hexaïssa** (AAAI 2026) formulates expert selection as a MoE problem, learning a gating policy that dynamically selects among heterogeneous state-of-the-art engines such as Stockfish. **M2CTS** (Helfenstein et al., 2024) combines MoE with *Monte Carlo Tree Search* (MCTS), using a modular framework that adapts strategy dynamically based on game phase and achieves a significant increase in engine strength. These approaches demonstrate that learned routing can be effective in chess, but they typically operate at the level of entire engines or large networks, not at the level of lightweight NNUE heads for embedded devices. Moreover, they often rely on **expensive routing mechanisms** that would be prohibitive at every node of an alpha-beta search.

#### 2.4.2.3 The gap addressed by this thesis
<span style="color: #808080;">[Learning from the Signal]</span> Existing learned bucketing methods for chess either rely on activations (which capture representation, not learning signal) or require computationally expensive routing that is unsuitable for embedded inference. This thesis addresses this gap by proposing a method that learns the partition from the sample gradients—the learning signal itself—and then distills this partition into a lightweight linear dispatcher that adds negligible overhead at inference time. This combines the data-driven advantages of learned bucketing with the efficiency requirements of on-device evaluation, providing a practical alternative to both handcrafted rules and computationally expensive routing policies.

---

## 2.5 Related Work

#todo When covering ELREA / GradientSpace: LoRA experts vs NNUE heads (see 2.2).

### 2.5.1 Efficient Chess Engines 

#todo Efficient Chess Engines 

### 2.5.2 Teacher–Student Evaluation 

#todo Teacher–Student Evaluation 

### 2.5.3 Gradient-Based Expert Specialization

#todo Gradient-Based Expert Specialization


---

> **Note for AI**: *The parts marked with a #todo are yet to be completed. The tagged comments are NOT to be exported to the Latex document, and are not meant to be implemented while exporting this document to Latex. The gray labels are for clarity only and must not be transferred to the Latex. Placeholders in square brackets (e.g. `[dataset_size]`, `[W]`) must be replaced with the current values from `_ai-info_.md` when converting this document to Latex.*
