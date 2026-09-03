## Background and Related Work

> The parts marked with a #todo are yet to be completed.

---

### 2.1 Chess Engines and Evaluation Functions

At the core of every chess engine lies the **evaluation function**: a function \(f: \mathcal{S} \rightarrow \mathbb{R}\) that assigns a scalar value to a board position, indicating the expected outcome from that position (typically from the perspective of the player to move). This value guides the search algorithm—usually a variant of **alpha-beta search** with iterative deepening—by pruning unpromising branches and selecting the most promising moves.

#### 2.1.1 Classical Evaluation

For decades, chess evaluation functions were handcrafted. A classical evaluator is a weighted linear combination of chess-specific features:

$$f(s) = \sum_{k} w_k \cdot \phi_k(s)$$
where $\phi_k(s)$ are feature functions and $w_k$ are scalar weights. Typical *chess-specific* features include material balance, piece-square tables (PSTs), mobility, king safety, pawn structure. These features were carefully tuned by chess experts over decades, using a combination of intuition, empirical testing, and later, automated optimization techniques.
#todo mention PeSTo?

While classical evaluators are extremely fast, they suffer from fundamental limitations. The human-designed features encode the human intuition about chess, which may not align with the optimal understanding of the game. Moreover, this model does not take into account the positions as a whole; the same parameters give a positional bonus for a central knight whether the position is a quiet middlegame or a tactical melee, even though the knight's practical value may differ dramatically across phases.

#### 2.1.2 The Shift to Neural Evaluation

The limitations of handcrafted features led to the adoption of neural networks as evaluation functions. **AlphaZero** (Silver et al., 2018) demonstrated that a deep convolutional network, trained via self-play reinforcement learning, could surpass the best classical engines. However, these networks are computationally expensive—requiring millions of operations per evaluation—making them unsuitable for resource-constrained devices or for engines that must evaluate millions of positions per second.

#todo AlphaZero value function: scalar \(v \in [-1,1]\) (expected outcome), trained with MCTS (not α-β). Cost per node vs NNUE. Do not mix AZ's MCTS with Cfish search. Contrast with Lc0 as teacher in 4.1.

#### 2.1.3 NNUE: A Hybrid Approach

The **Efficiently Updatable Neural Network (NNUE)** architecture, first introduced in the shogi engine *YaneuraOu* and later adopted by *Stockfish*, strikes a pragmatic balance. NNUE combines the representational power of a neural network with the incremental efficiency of classical evaluators.

The architecture consists of two ingredients: a sparse first layer called the *accumulator*, and a small fully-connected head. 

The accumulator layer maps a binary representation of the board to a hidden representation $h \in \mathbb{R}^{d}$ via $h = W_{L1} \cdot x$, where $x$ is a sparse binary vector (typically 768 or 1024 dimensions) indicating the presence of pieces on squares. The key insight is that a chess move changes only a few bits of $x$, so $h$ can be **updated incrementally** by adding and subtracting the corresponding columns of $W_{L1}$, rather than recomputing the entire matrix-vector product from scratch. This makes the computation of L1 essentially *free*.

The second ingredient, the fully connected head, is typically one or two hidden layers followed by a scalar output, mapping $h$ to the position value. In our case, we found easier to train a probability distribution across all the three possible game result for the player; win, draw, and loss (WDL).

A complete NNUE engine also leverages **alpha-beta search** with iterative deepening: starting from the root position, the engine searches deeper and deeper, using the NNUE evaluation at leaf nodes to guide the pruning and ordering of moves. At every node of this search tree, the evaluation function is called hundreds or thousands of times, making its speed critical. Also, being trained un enormous amounts of data, a NNUE doesn't suffer from the *horizon effect* as much as a PST does. #todo

#todo Keep this subsection conceptual (accumulator, head, incremental update). Own dims / CReLU / 3-way WDL belong in 4.2, not here.

#### 2.1.4 Value targets: centipawns, EV, WDL

#todo Conceptual comparison after NNUE / AlphaZero: scalar centipawns (classical / Stockfish-style NNUE), expected value \(v \in [-1,1]\) (AlphaZero), full WDL distribution. This is a representation choice, not only a metric. Loss choice (soft CE on WDL vs MSE on cp/EV) is fixed in 3.2–3.3 and 4.2.

#### 2.1.5 Why Evaluation Must Be Cheap

The search tree explored by an **alpha-beta engine** grows exponentially with depth. Even with effective move ordering and pruning techniques, the number of evaluated positions increases dramatically as the search goes deeper. A chess engine that searches deeper consistently outperforms one that searches shallower, as each additional ply reveals tactical patterns and strategic nuances that would otherwise remain hidden.

This means that any overhead added to the evaluation function is directly subtracted from the engine's effective search depth. If the evaluator becomes slower, the engine must reduce its search depth to maintain the same response time—sacrificing playing strength in the process.

On an embedded device such as the Wio Terminal, this constraint is even more severe. Memory is limited, floating-point operations are expensive, and every instruction counts. Any routing mechanism for a mixture-of-experts NNUE must add only a trivial cost—ideally, a handful of integer operations or a simple table look-up—to avoid degrading the engine's search performance.

In short, the evaluation function must be expressive enough to assess positions accurately, yet cheap enough to be called millions of times during a game. This trade-off is the central engineering challenge addressed by this thesis.

#todo One sentence: playing strength is reported in Elo / ACPL; protocol in Chapter 5.

#### 2.1.6 Current State of the Art for Tiny Hardware: Cfish

For resource-constrained devices, the most relevant reference implementation is **Cfish**, a port of the Stockfish chess engine written in plain C by Ronald de Man. While Stockfish is written in C++ and targets desktop-class hardware, Cfish strips away the C++ abstractions and compiles to a much smaller binary, making it viable for microcontrollers and other embedded platforms. The engine can be compiled with various evaluation backends, including a pure NNUE mode that excludes the classical handcrafted evaluation entirely, resulting in an even smaller executable.

From an architectural standpoint, Cfish shares the same core search and evaluation logic as Stockfish. The input to the evaluation function is a board position represented internally as a compact bitboard structure. The NNUE evaluation itself follows the _halfkp_ feature set: for each piece on the board, the network considers the piece's square together with the square of the king of the same color, producing a set of activated features that are fed into the accumulator. The accumulator is updated incrementally as moves are made, avoiding recomputation from scratch. The output of the NNUE is a scalar value, typically in centipawns, representing the expected advantage from the perspective of the side to move.

Cfish is particularly relevant to this thesis for two reasons. First, it demonstrates that a full-featured NNUE engine _can_ be made to run on constrained hardware, provided the implementation is careful about memory layout and avoids C++ overhead. Second, it serves as a practical baseline: any Mixture-of-Experts NNUE architecture proposed for tiny devices should be at least as fast and compact as Cfish in its pure NNUE mode, while offering improved evaluation accuracy through expert specialization. In this sense, Cfish represents both the _state of the art_ and the _performance target_ for the embedded chess engine developed in this work.

#todo Pointer: Cfish as host engine (evaluate(), α-β, ID, TT, Wio, MoE hook) is instantiated in 4.8–4.9. Keep 2.1 conceptual.

### 2.2 Mixture of Experts

#todo Multiple expert networks over sub-domains. Learned vs.\ fixed routing. Uses in vision, NLP, and RL, and what carries over to a tiny chess eval.

#todo LoRA: how ELREA / GradientSpace implement experts on LLMs. Contrast LoRA adapters vs a tiny NNUE head. Not used in this method.

### 2.3 Sample Gradients

#todo Per-example gradients w.r.t.\ head parameters as a representation of the learning signal. Why they differ from activations. Pointers to gradient-clustering literature.

### 2.4 Bucketing in NNUE

#todo Handcrafted buckets: piece count, king location, piece presence (queen, bishop pair, \ldots). Why they are cheap, and why they may not match what the model needs.

### 2.5 Related Work

#todo Efficiency (pruning, quantization). Distillation / teacher–student. Self-play RL (AlphaZero, Lc0). Work closest to sample-gradient clustering and MoE routing (see `related-work.md`).

#todo When covering ELREA / GradientSpace: LoRA experts vs NNUE heads (see 2.2).
