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

<span style="color: #808080;">[Limitations]</span> While classical evaluators are extremely fast, they suffer from fundamental limitations. The human-designed features encode the human intuition about chess, which may not align with the optimal understanding of the game. Moreover, these models do not take into account the positions as a whole; the same parameters give a positional bonus for a central knight whether the position is a quiet middlegame or a tactical melee, even though the knight's practical value may differ dramatically across phases.

<span style="color: #808080;">[Limitations II - Horizon Effect]</span> Classical evaluators are also vulnerable to the _horizon effect_. A search algorithm constrained to a fixed depth evaluates positions at the search frontier as if the game were stable at that point. If a tactical sequence begins at the horizon but extends one ply further, the engine cannot see the consequences and may assign an inaccurate evaluation. Classical evaluators, which rely on static features such as material balance and piece-square tables, are particularly susceptible: they treat a position at the horizon as if it were quiet, ignoring the tactical possibilities that would unfold if the search continued. In contrast, models trained on complete game outcomes can often encode patterns that extend beyond immediate material and positional features, making them less vulnerable to horizon-induced misevaluations.

### 2.1.2 The Shift to Neural Evaluation

<span style="color: #808080;">[Neural Eval]</span> The limitations of handcrafted features led to the adoption of neural networks as evaluation functions. **AlphaZero** (Silver et al., 2018) demonstrated that a deep convolutional network, trained via self-play reinforcement learning, could surpass the best classical engines. However, these networks are computationally expensive—requiring millions of operations per evaluation—making them unsuitable for resource-constrained devices or for engines that must evaluate millions of positions per second.

<span style="color: #808080;">[AlphaZero value function]</span> AlphaZero's value head outputs a scalar $v\in[−1,+1]$, interpreted as the expected game outcome from the current player's perspective. This scalar is the training target for the value network. However, AlphaZero's training and inference rely on **Monte Carlo Tree Search (MCTS)**, which builds a search tree by repeatedly simulating trajectories and using the neural network to evaluate leaf nodes. This process requires many forward passes of the network per position, making it computationally expensive and poorly suited for engines that evaluate millions of positions per second with alpha-beta search. The AlphaZero network itself is a deep residual architecture with millions of parameters, requiring floating-point operations and substantial memory, which far exceeds the capacity of microcontrollers. In contrast, the NNUE architecture adopted in this work reduces the per-evaluation cost to a handful of integer operations while retaining the representational power of a neural network. We revisit AlphaZero's value formulation in Section 2.1.4, where we contrast its scalar expected value with the WDL distribution used in this work.

### 2.1.3 NNUE: A Hybrid Approach

<span style="color: #808080;">[Hybrid Design]</span> The **Efficiently Updatable Neural Network (NNUE)** architecture, first introduced in the *shogi* engine *YaneuraOu* and later adopted by *Stockfish*, strikes a pragmatic balance. NNUE combines the representational power of a neural network with the incremental efficiency of classical evaluators.

<span style="color: #808080;">[Two Ingredients]</span> The architecture consists of two ingredients: a sparse first layer called the *accumulator*, and a small fully-connected head. 

<span style="color: #808080;">[Accumulator]</span> The accumulator layer maps a binary representation of the board to a hidden representation $h \in \mathbb{R}^{d}$ via $h = W_{L1} \cdot x$, where $x$ is a sparse binary vector (typically 768 or 1024 dimensions) indicating the presence of pieces on squares. The key insight is that a chess move changes only a few bits of $x$, so $h$ can be **updated incrementally** by adding and subtracting the corresponding columns of $W_{L1}$, rather than by recomputing the entire matrix-vector product from scratch. This makes the computation of L1 essentially *free*.

<span style="color: #808080;">[Head / WDL]</span> The second ingredient, the fully connected head, is typically one or two hidden layers followed by a scalar output, mapping $h$ to the position value. In our case, we found it easier to train a probability distribution across all the three possible game result for the player; win, draw, and loss (WDL), by minimizing the cross-entropy between the output of the teacher and the student.

<span style="color: #808080;">[Search / Horizon]</span> A complete NNUE engine also leverages **alpha-beta search** with iterative deepening: starting from the root position, the engine searches deeper and deeper, using the NNUE evaluation at leaf nodes to guide the pruning and ordering of moves. At every node of this search tree, the evaluation function is called hundreds or thousands of times, making its speed critical. Also, being trained on enormous amounts of data, a NNUE doesn't suffer from the *horizon problem* as much as a PST does. 

#todo Keep this subsection conceptual (accumulator, head, incremental update). Own dims / CReLU / 3-way WDL belong in 4.2, not here.

## 2.1.4 Value targets: centipawns, EV, WDL

<span style="color: #808080;">[Value Targets]</span>
#todo Conceptual comparison after NNUE / AlphaZero: scalar centipawns (classical / Stockfish-style NNUE), expected value $v \in [-1,1]$ (AlphaZero), full WDL distribution. This is a representation choice, not only a metric. Loss choice (soft CE on WDL vs MSE on cp/EV) is fixed in 3.2–3.3 and 4.2.

### 2.1.5 Why Evaluation Must Be Cheap

<span style="color: #808080;">[Search Cost]</span> The search tree explored by an **alpha-beta engine** grows exponentially with depth. Even with effective move ordering and pruning techniques, the number of evaluated positions increases dramatically as the search goes deeper. A chess engine that searches deeper consistently outperforms one that searches shallower, as each additional ply reveals tactical patterns and strategic nuances that would otherwise remain hidden.

<span style="color: #808080;">[Depth Trade-off]</span> This means that any overhead added to the evaluation function is directly subtracted from the engine's effective search depth. If the evaluator becomes slower, the engine must reduce its search depth to maintain the same response time—sacrificing playing strength in the process.

<span style="color: #808080;">[Embedded Constraint]</span> On an embedded device such as the Wio Terminal (192 KB RAM, 500 KB flash), this constraint is even more severe. Memory is limited, floating-point operations are expensive, and every instruction counts. Any routing mechanism for a mixture-of-experts NNUE must add only a trivial cost—ideally, a handful of integer operations or a simple table look-up—to avoid degrading the engine's search performance.

<span style="color: #808080;">[Core Trade-off]</span> In short, the evaluation function must be expressive enough to assess positions accurately, yet cheap enough to be called millions of times during a game. This trade-off is the central engineering challenge addressed by this thesis.

#todo too long?

#todo One sentence: playing strength is reported in Elo / ACPL; protocol in Chapter 5.

### 2.1.6 Current State of the Art for Tiny Hardware: Cfish

<span style="color: #808080;">[What is Cfish]</span> For resource-constrained devices, the most relevant reference implementation is **Cfish**, a port of the Stockfish chess engine written in plain C by Ronald de Man. While Stockfish is written in C++ and targets desktop-class hardware, Cfish strips away the C++ abstractions and compiles to a much smaller binary, making it viable for microcontrollers and other embedded platforms. The engine can be compiled with various evaluation backends, including a pure NNUE mode that excludes the classical handcrafted evaluation entirely, resulting in an even smaller executable.

<span style="color: #808080;">[Architecture]</span> From an architectural standpoint, Cfish shares the same core search and evaluation logic as Stockfish. The input to the evaluation function is a board position represented internally as a compact bitboard structure. The NNUE evaluation itself follows the _halfkp_ feature set: for each piece on the board, the network considers the piece's square together with the square of the king of the same color, producing a set of activated features that are fed into the accumulator. The accumulator is updated incrementally as moves are made, avoiding recomputation from scratch. The output of the NNUE is a scalar value, typically in centipawns, representing the expected advantage from the perspective of the side to move.

<span style="color: #808080;">[Why Relevant]</span> Cfish is particularly relevant to this thesis for two reasons. First, it demonstrates that a full-featured NNUE engine _can_ be made to run on constrained hardware, provided the implementation is careful about memory layout and avoids C++ overhead. Second, it serves as a practical baseline: any Mixture-of-Experts NNUE architecture proposed for tiny devices should be at least as fast and compact as Cfish in its pure NNUE mode, while offering improved evaluation accuracy through expert specialization. In this sense, Cfish represents both the _state of the art_ and the _performance target_ for the embedded chess engine developed in this work.

#todo Pointer: Cfish as host engine (evaluate(), α-β, ID, TT, Wio, MoE hook) is instantiated in 4.8–4.9. Keep 2.1 conceptual.

#todo also too long?

---

## 2.2 Mixture of Experts

<span style="color: #808080;">[MoE Primer]</span>
#todo Multiple expert networks over sub-domains. Learned vs.\ fixed routing. Uses in vision, NLP, and RL, and what carries over to a tiny chess eval.

#todo LoRA: how ELREA / GradientSpace implement experts on LLMs. Contrast LoRA adapters vs a tiny NNUE head. Not used in this method.

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

<span style="color: #808080;">[Handcrafted Buckets]</span>
#todo Handcrafted buckets: piece count, king location, piece presence (queen, bishop pair, \ldots). Why they are cheap, and why they may not match what the model needs.

### 2.4.2 Learned Bucketing


#todo reference the Kaggle challenge: [FIDE & Google Efficient Chess AI Challenge](https://www.kaggle.com/competitions/fide-google-efficiency-chess-ai-challenge)

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
