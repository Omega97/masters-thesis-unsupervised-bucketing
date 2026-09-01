## Method: Unsupervised Bucketing via Sample Gradients

> The parts marked with a #todo are yet to be completed.

---

### 3.1 Overview

#todo Five-step pipeline: train a base model; compute sample-gradients; cluster them; train a dispatcher; fine-tune expert heads.

### 3.2 Notation and Definitions

#todo State space \(\mathcal{S}\), model \(f_w\), loss \(\mathcal{L}\), target \(\hat{v}\). Sample-gradient \(\delta_i = \nabla_{w^{(\mathrm{head})}} \mathcal{L}(f_w(s_i), \hat{v}_i)\). Dispatcher \(g_\phi(h)\). Convention: weight matrices as \(W_{L1}\), not \(w^{(L1)}\).

### 3.3 Step 1: Train the Base Model

#todo Shared dual-POV L1, L2, output head. Soft cross-entropy on WDL labels from a strong teacher (Lc0).

### 3.4 Step 2: Compute Sample Gradients

#todo Per-sample gradients w.r.t.\ head parameters. Normalisation (L2 or standardisation). Storage and compute cost.

### 3.5 Step 3: Cluster Sample Gradients

#todo K-Means and alternatives (DBSCAN, Density Peak; some choose \(B\)). How to pick \(B\). Validation: inertia, silhouette, interpretability.

### 3.6 Step 4: Train the Dispatcher

#todo Input: L1 activations \(h\). Output: bucket id \(b\). Linear layer or small MLP. Cross-entropy. Needed because gradients are not available at inference.

### 3.7 Step 5: Fine-Tune Expert Heads

#todo Freeze L1; fine-tune one head per bucket from the base model. Result: \(B\) specialised experts.

### 3.8 Generalisation Beyond Chess

#todo The method needs only a state space, a model, a loss, and a target. Other games, robotics, world-model settings.
