# Unsupervised Bucketing Technique for MoE Chess Value Functions

| *Disclaimer: none of the text in this repo is entirely AI-generated. AI was only used to help with phrasing and formatting.* |
| ---------------------------------------------------------------------------------------------------------------------------- |

---

## The Goal of this Thesis

> _We propose a novel approach for bucketing states in chess. Instead of relying on handcrafted features (piece count, game phase, etc.), we learn the bucketing directly from the data._
> *The key idea is to train a **base model**, then compute for each data point a **skill vector** – the direction in weight space that the neural network would like to take in order to better internalise that example. By clustering these skill vectors, we partition the state space into **meaningful buckets** without any human supervision.*
> *This technique transcends the game of chess.*

#core