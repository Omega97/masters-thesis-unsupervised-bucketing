
> *Note: This folder contains the text of the Master's thesis in chess AI. This and all the subsequent documents are meant to be transposed to Latex in pdf form.*

---

## Hyperparameters

> *Note for the AI: refer to this file for every hyperparameter and variable. When converting markdown to Latex, replace every square-bracket placeholder (e.g. `[dataset_size]`, `[W]`) with the **current value** from the table below. Do not leave placeholders in the Latex.*

| Placeholder | Meaning | Current value |
| :--- | :--- | :--- |
| `[dataset_size]` | Dataset size (FEN positions, with duplicates) | 50M |
| `[dataset_size_unique]` | Dataset size (unique FEN positions) | 49.5M |
| `[test_fraction]` | Percentage of the dataset held out as test set | 1% |
| `[batch_size]` | Training batch size | 10k |
| `[lr_start]` | Initial learning rate | $10^{-2}$ |
| `[lr_end]` | Final (decayed) learning rate | $10^{-3}$ |
| `[d_in]` | Input / sparse-feature dimension | 844 |
| `[W]` | L1 accumulator width **per perspective** (dual NNUE: `[W]`×2) | 128 |
| `[H]` | L2 hidden width | 256 |
| `[keep_prob]` | Probability of keeping any sampled position in the dataset | 5% |
| `[n_params]` | Total trainable parameters of the target architecture | *derive from `[d_in]`, `[W]`, `[H]` at export time* |
| `[P_head]` | Flattened head-parameter dimension (L2 + output) | *derive from `[W]`, `[H]` at export time* |

Target model architecture: $d_{\text{in}}=[d_in]$, dual NNUE, L1 accumulator layer $[W]\times 2$ neurons, L2 $[H]$ neurons, output WDL for the current player.

---

## Scraps


#todo  important: dig deeper in literature to find a similar method

### Abstract

#### Key Findings / Expected Results (if everything goes well... 🍀)
Preliminary results indicate that the resulting MoE model achieves lower test cross‑entropy and mean absolute error than the single‑head baseline, with negligible runtime overhead. The sample‑gradient clusters reveal interpretable structure in the state space, and the dispatcher achieves high accuracy, enabling fast, deterministic routing at inference time.

#todo Lazy loading with memory mapping?