
## Thoughts

https://arxiv.org/abs/2309.17080

The researchers partition the ==self-driving== model into a *world model* (reasons about the scene’s high-level components and dynamics) and the *video diffusion decoder* (translating latent representations back into high-quality videos with realistic detail)

---

### 1 Introduction

Rollout of future states for self-driving vehicles
Gaia-1 combines world models and generative video generation

GAIA-1 works as follows. 
- First, we partition the model into two components: the world model and the video diffusion decoder. 
	- The world model (vector-quantized representations of video frames) reasons about the scene’s high-level components and dynamics, while 
	- the diffusion model takes on the responsibility of translating latent representations back into high-quality videos with realistic detail.
- Subsequently, we reframe the challenge of predicting the future into predicting the next token in the sequence. 

**images**, **text** and **actions** are encoded as a sequence of tokens

GAIA-1 is designed to be multimodal, allowing video, text and action to be used as prompts to generate diverse and realistic driving scenarios

---

### 2 Model

The three different input modalities (video, text, action), which are encoded into a shared d-dimensional embedding space

- Image tokens: Each image frame of a video is represented as discrete tokens
- Text tokens: LLM
- Action tokens: 

#### 2.3 World Model

The world model is an autoregressive transformer that predicts the next image token in a sequence, conditioned on past image tokens (from a VQ-based image tokenizer), text tokens (from T5-large), and action tokens (speed + curvature).

- **Input format**: Tokens are interleaved as _text → image → action_ per timestep, with factorized spatio-temporal positional embeddings.
- **Objective**: Next-token prediction with causal masking (standard cross-entropy loss). Random dropout of conditioning tokens enables unconditional, action-conditioned, or text-conditioned generation.
- **Temporal subsampling**: Videos are downsampled to 6.25 Hz during training to manage sequence length while allowing reasoning over longer horizons (temporal upsampling happens in the decoder).
- **Key insight**: Framing world modeling as discrete sequence prediction (inspired by LLMs) enables scaling and emergent understanding of driving dynamics without explicit supervision.

This component learns high-level scene representations and dynamics in latent token space before pixel-level rendering.

...

---

### 5 Inference

#### 5.1 World Model

Inference focuses on reliable autoregressive rollout:

- **Sampling strategy**: Pure argmax leads to repetitive loops; naive sampling risks out-of-distribution tokens. **Top-k sampling** (k=50) balances realism and diversity, matching the perplexity distribution of real tokens.
- Long rollouts
- Text conditioning
- Action control

---

### 8 Related Work

#### World models

Section 8 situates GAIA-1 among prior world models (e.g., RSSM-based like Dreamer, video prediction models) and generative approaches. It highlights GAIA-1's novelty in scaling discrete token-based world modeling to real-world driving with multimodal control and high-fidelity decoding.

---
#WorldModels #Generative #GAIA #PaperNote 