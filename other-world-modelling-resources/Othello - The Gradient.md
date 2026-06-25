
[Othello - The Gradient](https://thegradient.pub/othello/)

---

**Summary: "Large Language Models: World Models or Surface Statistics?" (The Gradient, Jan 2023)**

This insightful article by Kenneth Li explores a fundamental question in AI: Do large language models (LLMs) like GPT merely memorize surface-level statistical patterns from training data, or do they build genuine **internal world models**—structured, interpretable representations of the underlying processes generating sequences?

### The Core Mystery & Parable
The piece opens with a compelling **crow parable**: A crow hears Othello game moves called out but never sees the board. It begins predicting legal moves accurately. Researchers later discover it has arranged birdseed into a grid mirroring the board state, and manipulating the seeds changes its predictions. This analogy beautifully illustrates the debate—does the model (crow/LLM) rely on shallow correlations, or has it formed a controllable, human-understandable model of reality?

### Othello-GPT: A Controlled Experiment
To investigate, the authors train a GPT-style model (**Othello-GPT**) exclusively on sequences of legal Othello moves (e.g., "E3, D3, C4...")—**without any prior knowledge of board structure, rules, or geometry**. Othello is chosen as a sweet spot: complex enough to prevent pure memorization but simpler than chess.

**Key findings:**
- The trained model predicts **legal moves with ~99.99% accuracy** (vs. ~7% for random initialization).
- This mirrors real LLMs generating coherent text/code without explicit programming.

### Evidence for World Models
1. **Probing Internal Representations**:
   - ==Linear probes (small classifiers) trained on the model's activations== can decode the full 8x8 board state (black/white/empty for each of 64 squares) with high accuracy (~1.7% error vs. 26% baseline).
   - Remarkably, ==PCA on these "concept vectors" reveals an **emergent geometry** resembling a draped Othello board==—neighboring squares cluster together in the high-dimensional space.

2. **Causal Intervention (The "Prank")**:
   - Researchers ==manipulate the model's internal activations mid-forward-pass to represent altered board states== (e.g., flipping a disc's color).
   - This reliably changes downstream predictions to match the new (sometimes impossible) board, proving the world model is **causally used** for generation, not just correlational. Even "unreachable" states work well.

3. **Interpretability Application**:
   - They introduce **latent saliency maps** via intervention-based attribution, revealing how the model "thinks" about legality (e.g., highlighting sandwiching pieces for flips). This offers a window into the model's reasoning beyond input tokens.

### Implications and Open Questions
The work provides strong evidence that **next-token prediction can induce emergent world models**, creating a bidirectional mapping between the black-box neural net and human-understandable representations. This has exciting ramifications for **mechanistic interpretability**, alignment, and steering models.

However, challenges remain:
- In real LLMs, we don't know the "shape" of the world model in advance.
- How do we reverse-engineer complex, multifaceted world models?
- Can we intervene minimally without disrupting unrelated knowledge?


This experiment is a landmark in interpretability research—elegant, rigorous, and full of potential for future work on understanding (and controlling) the "minds" of LLMs. It invites deeper curiosity: What other hidden world models might current frontier models be building about physics, social dynamics, or code semantics? I'd love to discuss extensions, like applying similar probes to modern transformers! 

Even simple autoregressive training on sequences can lead to rich internal simulations of reality. This bridges statistical learning with more structured, causal understanding.

---

#WorldModels #Interpretability #thesis
