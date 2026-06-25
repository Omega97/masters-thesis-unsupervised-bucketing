
## Thoughts

https://arxiv.org/abs/2301.04104

I like this generalist and robust world model-based approach to ==RL across diverse domains==. It tackles vastly different environments (3D, 2D, continuous, discrete, ...) with the same hyperparameters, and with great results.

---

## 1. Introduction

DreamerV3 is a general algorithm that outperforms specialized methods across over 150 diverse tasks, with a single configuration.

Dreamer is the first algorithm to collect diamonds in Minecraft from scratch without human data or curricula. Dreamer learns to obtain diamonds from scratch given sparse rewards, a long-standing challenge in artificial intelligence for which previous approaches required human data or domain-specific heuristics.

Using **fixed hyperparameters across all domains**, Dreamer **outperforms tuned expert algorithms** across a wide range of benchmarks and data budgets

- The world model predicts the outcomes of potential actions, a critic neural network judges the value of each outcome, and an actor neural network chooses actions to reach the best outcomes

Dreamer robustly learns and uses world models to achieve strong task performance through a range of robustness techniques based on: 
- normalization, 
- balancing, and 
- transformations.

---

## 2. Learning algorithm

The algorithm consists of three neural networks: 
- the world model predicts the outcomes of potential actions, 
- the critic judges the value of each outcome, and 
- the actor chooses actions to reach the most valuable outcomes

The components are trained concurrently from replayed experience while the agent interacts with the environment

---

### World model learning

 The model: Recurrent State-Space Model (RSSM)

- sensory input $x_t$
    
- embedding $z_t$   
	- stochastic representations
	  
- recurrent state $h_t$
    
- actions $a_t$
    
- rewards $r_t$
    
- episode continuation flags $c_t ∈ \{0, 1\}$
	- $c_t​=1$: The episode **continues** after timestep t t t (the agent can take more actions).
	- $c_t​=0$: The episode **terminates** (end of episode, e.g., due to death, timeout, success, or failure).
	- Trained with logistic regression (binary cross-entropy)

RSSM:

- Sequence model: $h_t=f_\phi(h_{t-1},z_{t-1},a_{t-1})$
    
- Encoder: $z_t \sim q_\phi(z_t \mid h_t,x_t)$ 
	- CNN / MLP
	  
- Dynamics predictor: $\hat z_t \sim p_\phi(\hat z_t \mid h_t)$

Predictors (given recurrent state and embedding)

- Reward predictor: $\hat r_t ∼ p_\phi(\hat r_t \mid h_t, z_t)$
    
- Continue predictor: $\hat c_t \sim p_\phi(\hat c_t \mid h_t, z_t)$
    
- Decoder: $\hat x_t \sim p_\phi(\hat x_t \mid h_t, z_t)$


- $v_\psi$: the critic's value function
    
- $\phi$: world model parameters, optimized end-to-end to minimize the prediction loss $\mathcal L_{pred}$, the dynamics loss $\mathcal L_{dyn}$, and the representation loss $\mathcal L_{rep}$ 
    
- the complete loss is the EV of a weighted average of the three, estimated for all $t \le T$

...

combining free bits with a small representation loss resolves this dilemma, allowing for fixed hyperparameters across domains

---

### Critic learning

By learning a full return distribution in a normalized, binned way and combining imagination with replay, Dreamer achieves remarkable stability across domains with vastly different reward scales and sparsities — a major reason for its "out-of-the-box" generality.

...

---

### Actor learning

Standard approaches (advantage normalization, fixed entropy targets, or reward scaling) often fail across domains with vastly different reward densities and scales. DreamerV3’s return normalization with a lower bound elegantly balances exploration and exploitation without per-domain tuning — a major reason for its generality (Atari, Minecraft, robotics, etc.).
...

---

### Robust predictions

...

---

## 3. Results

..

---

#WorldModels #DiverseDomains #Dreamer #PaperNote
