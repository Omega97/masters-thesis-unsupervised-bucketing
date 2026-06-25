
## Thoughts

https://arxiv.org/pdf/2210.13382

This paper was a pleasant read. The concepts are not too complex, but the approach to probe for an internal world model is sound and interesting. Othello ties well to my chess AI.

---

### 1 INTRODUCTION

Recent evidence suggests language models can develop internal representations for very simple concepts, such as color, direction, or tracking boolean states during synthetic tasks

Steps:
1. they train a language model (a GPT variant we call ==Othello-GPT==) to extend partial game transcripts (a list of moves made by players) with legal moves
2. look for world representations that might be used by the network

Pipeline:
- the dataset consists of a list of game transcripts
- train an LLM to predict the next move
- train a non-linear probe to predict the board state
- test on the validation set

 Knowledge of the internal world model can be used as an interpretability tool.
 
---

### 2 “LANGUAGE MODELING” OF OTHELLO GAME TRANSCRIPTS


Othello:
- $10^{58}$ game records
- $10^{28}$ board positions

DATASETS: “CHAMPIONSHIP” AND “SYNTHETIC”

Othello-GPT does far better than chance in predicting legal moves

Weirdly, it does much worse on human data than synthetic ones. Why?? I think it has to do with the fact that the human games sample good moves rather than random ones, so the model is less accurate, because it returns random moves.

They exploited the symmetries of the game to test for memorization.

---

### 3 EXPLORING INTERNAL REPRESENTATIONS WITH PROBES

 They train probes that predict the board state from the network’s internal activations after a given sequence of moves.

The probe is a list of functions of the activations, that correctly reconstructs the state of each square of the board

They show that any world model of their network is not a simple linear function of the activations

They also show that the world model of their network is approximately a **non-linear function of the activations**.

---

### 4 VALIDATING PROBES WITH INTERVENTIONAL EXPERIMENTS

They investigate the causal relation between the **emergent representation** and the **model's prediction**.

> Given a set of activations from the Othello-GPT, a probe predicts a baseline board state B. 
> We record the move predictions associated with B, then modify these activations such that our probe reports an updated board state B′. 
> Through our protocol, only a single tile s distinguishes B′ from B’s board state. This small modification results in a different set of possible legal moves for B′. If the new predictions match our expectations for B′—and not the predictions we recorded for B—we conclude the representation had a causal effect on the model.

> we select an initial layer Ls then modify it and subsequent layers’ activations

Interventions minimize cross-entropy between the probe-predicted probability distribution $p_\theta$ and the desired board state $B'$, wrt the intervention parameters $x$.

$$\Delta x = - \alpha \nabla \mathcal{L}_{CE}(p_\theta(x),B')$$

- Researchers edit the model’s internal activations at a chosen layer using gradient descent.
- Goal: Flip the probe’s predicted state of a specific tile (e.g., change E6 from white to black) while leaving other tiles unchanged.
- After intervention, they continue the forward pass and observe how the model’s legal move predictions change.

---

### 5 LATENT SALIENCY MAPS: ATTRIBUTION VIA INTERVENTION

Instead of attributing predictions directly to input tokens (standard saliency), the authors create latent saliency maps by intervening in the model’s internal world representation (the emergent board state)

Researchers compute a latent saliency map by attempting to flip the color of each tile, to see which pieces influence the decision most.

- In the synthetic dataset, the model learned the rules of the game
- In the human-made dataset, the model learned strategic moves.

---

**MOC**: [[World-Modelling-papers-read]]
**Related**: [[Othello - The Gradient]] (accessible explainer)
**Tree**: [[Thesis Knowledge Tree]]

#WorldModels #Interpretability #EmergentRepresentations #Othello #PaperNote #thesis

