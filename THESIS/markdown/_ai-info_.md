
> *Note: This folder contains the text of the Master's thesis in chess AI. This and all the subsequent documents are meant to be transposed to Latex in pdf form.*

---

## Hyperparameters

> *Note for the AI: refer to this file for every hyperparameter and variable *

- Dataset size (FEN positions, with duplicates): 50M
- Dataset size (unique FEN positions): 49.5M
- Percentage test set: 1%
- Batch size: 10k
- Learning rate: decaying from $10^{-2}$ to $10^{-3}$
- Target model architecture: $d_{\text{in}}=844$, dual NNUE, L1 accumulator layer 128x2 neurons, L2 256 neurons, output WDL for current player
- Probability of keeping any position in the dataset: 5%

---
