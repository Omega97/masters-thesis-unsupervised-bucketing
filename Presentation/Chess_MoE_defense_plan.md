# Defense deck plan

Source: `Chess_MoE_pre-defense.pptx` (39 slides). Target: **10 spoken slides**, about a quarter of that file, for a 15–20 minute talk. Professors are still in the room. The rest of the audience should be able to follow the story without having read the thesis.

The pre-defense worked because the pictures carried the method. This version keeps those pictures, drops the catalogues, and puts a plain-language answer on every slide that drew a question.

## Purpose, said twice

Open on it. Close on it. One sentence, same words both times:

> A tiny chess engine has to pick a specialist evaluator without slowing down search.

That is the project. Everything else is how we tried, and what we showed.

## Goals we can claim

Say these as results, in this order. They are the achieved goals.

1. **The routing gap is real and measured.** L1 buckets are recoverable at **98%**. Gradient buckets, with the same L1 input and the same dispatcher, are recoverable at **52.5%**.
2. **A hard Mixture-of-Experts can sit on a normal NNUE.** One shared accumulator stays incremental. One expert head runs. The other heads stay in memory and do not run.
3. **The right grouping signal is the per-sample head gradient, and clustering it is not trivial.** In its own space the gradients separate (silhouette **0.191** at K = 8). A linear view from L1 does not recover them.
4. **L1 clustering is the easy half, and that is useful.** Activations are almost unseparated (silhouette **0.003**), and a linear dispatcher still gets the bucket right. Easy to assign, weak as a specialization signal.
5. **On a net that still has room, specialization shows up.** Weaker base W64/H128, test CE **0.6537**. Four L1 experts, linear dispatcher, longer fine-tuning: **0.6499**.
6. **KITSUNE routes with no gradient at play time and matches that result.** Weaker base, K = 16, two micro-buckets per expert: routed mixture CE **0.65043**. It matches the best single-stage L1 run. It does not beat it.

Say the limit in the same breath as the result. On the strong reference (W128/H256, CE **0.6303**) every deployed router stays at about **0.630**. A minimum-loss oracle on the same gradient experts reaches **0.6005** at K = 16. The experts have headroom. The online assignment does not spend it. Elo, nodes per second, and a measured fit in 192 KB RAM are future work.

## The six questions, built into the talk

| Question they asked | Slide | The line to leave on screen |
|---|---|---|
| What are sample gradients? Show the formula. | 5 | One position, one vector: the direction the head would move to fit that position. |
| How does KITSUNE work, exactly? | 8 | Cut L1 into small convex pieces. Merge pieces with similar gradients offline. At play time, a linear map plus a table. |
| Is the architecture dense? | 4 | No. Shared L1 is always on. Exactly one head runs. |
| Why does this work? | 7, then 9 | Positions that want the same update can share a head. L1 tells you where you are. Gradients tell you how the head should change. KITSUNE uses each one for the job it is good at. |
| Do we expect L1 clustering to be easy? | 6 | Yes. The regions are convex. Recovery is 98%. |
| Is clustering the sample gradients trivial? | 5–6 | No. The vectors are ~66,000-dimensional, the manifold is not convex in L1, and recovery from L1 stops at 52.5%. |

“Easy” for L1 means easy to recover with a linear dispatcher. It does not mean the clusters are tight. Silhouette near zero and 98% accuracy are both true. Say both, or a professor will set them against each other.

## Spoken deck: 10 slides

Timing assumes a short pause on 5, 6, and 8. Those three are the ones that failed to land last time.

### 1. Title — the question

Keep the title, your name, University of Trieste, DSAI, Alessio Ansuini (supervisor, Area Science Park), Marco Zennaro (internship supervisor, ICTP).

Replace the subtitle with the purpose sentence. Fix the broken spacing in “Supervisor” on the current title slide.

Visual: one board, one small device, one arrow between them. No paragraph.

### 2. Why a small net is the whole problem

Merge current slides 2 and 33 into one flashy contrast.

Left: a huge net (AlphaZero style) — value, policy, search in the net. Right: NNUE on a microcontroller — 120 MHz, 192 KB RAM, 512 KB flash. Alpha–beta calls the evaluator on every node. A slower eval can cost a ply.

One line under it: we keep the small net, and we try to make it a mixture of specialists.

Drop the edge-computing keyword cloud.

### 3. NNUE in four beats

Keep slide 4. It is the clearest slide in the pre-defense.

1. A move touches few of the 844 features.
2. Subtract the old features, add the new ones.
3. The L1 accumulator stays.
4. The rest of the net is cheap because of that.

Use one architecture drawing (current slide 3 or 7), not both, and not the duplicate accumulator slides 5 and 6.

Spoken aside, not a slide: the net is trained by distillation from Leela Chess Zero win/draw/loss labels. ~150M positions. The big reference has ~175k parameters. Overfitting eases around 7M positions.

### 4. Dense? No. One accumulator, one head.

This slide exists to answer the architecture question before anyone asks it.

Picture, left to right:

`844 features → shared L1 (frozen, incremental) → dispatcher → exactly one expert head → WDL`

Under the row of heads: the others are stored and idle. Extra experts cost memory. They do not cost a second forward pass.

Say “hard routing” out loud. Soft-gated and Switch-style routing were tried. They do not beat the base. That fact can live in one clause. The soft-gate diagram does not get a slide.

Skip the parameter-count graphic (slide 9) unless a question asks. If it comes up: the mixture is only a little larger than the base, because L1 is shared and only the head is copied. Current labels on that graphic: base around 216k, mixture around 215–263k depending on what is counted, dispatcher for 256 classes, L1 = 128, L2 = 256. Reconcile the three mixture numbers before putting them back on a slide. They read as contradictory in a quick glance.

### 5. Sample gradients — formula, then a picture

Full slide. Big formula. Then one sentence. Then the picture you already have (slides 10–11).

\[
\Delta_i = \nabla_{w_{\text{head}}} \mathcal{L}\big(\hat f_{w_{\text{base}}}(s_i), v_i\big)
\]

Read it as: for position \(s_i\) with teacher label \(v_i\), take the gradient of the loss at the base model, and take it only with respect to the head. L1 is frozen, so it is not in the vector.

Then the normalization, smaller, because clustering uses direction:

\[
\Delta_i^{\text{norm}} = \Delta_i / \|\Delta_i\|_2
\]

Plain language under the formula:

> The gradient is what this position wants the head to learn. Similar arrows can share an expert. Opposite arrows fight inside one head.

Head dimension on the reference net: \(P_{\text{head}} = 66{,}563\) (flattened L2 and output). That is why the clustering is not a toy. Say the number.

Keep one of the gradient scatter pictures. Label three real positions on it if the figure already has them. Drop the second, nearly identical, clustering slide.

### 6. Two clusterings, one punchline

This replaces slides 14 and 34. Two columns, huge percentages.

| | L1 activations | Sample gradients |
|---|---|---|
| What it is | Where the position sits | How the head should move |
| Separated? | Almost no. Silhouette 0.003 | Yes, in gradient space. Silhouette 0.191 |
| Can a cheap dispatcher recover the bucket from L1? | **98%** | **52.5%** |
| Do we expect this? | **Yes. Easy.** Convex regions. | **No. Not trivial.** Non-convex once you only look at L1. |

Footer, one line: same L1 input, same dispatcher, eight buckets. The 45.5-point gap is the routing gap.

Say the expectation before the number. “We expected L1 to be easy, and it is. We did not expect gradient buckets to fall out of L1, and they do not.”

### 7. Why the idea still holds when the strong net barely moves

This replaces the “but why??” table (slide 15) and the oracle slide (slide 36).

Three numbers only:

- Strong reference CE **0.6303**. Piece-count, L1, gradient, and soft-gated mixtures all sit on **0.630**.
- Same gradient experts, perfect hindsight assignment: oracle **0.6005** (K = 16).
- Online dispatcher on those experts: **0.6301**.

The sentence:

> The specialists know different things. A router that only sees L1 cannot always find the right one. On a net that is already strong, that miss cancels the gain.

Then the weaker base, as the place the gain is visible: **0.6537 → 0.6499** with four L1 experts. Purpose check: we were trying to specialize inside a small net, and on a small net that still has error left, the bucketed heads do reduce cross-entropy. This is not an Elo claim.

Drop the full model-by-model table from the spoken deck. It is backup.

### 8. KITSUNE, one mechanism, three beats

Merge slides 17, 18, 19, and 20. This is the slide to rehearse.

Title expands once, then you use the name:

**KITSUNE** — Knowledge-guided Initialization and Two-stage Segregation for Unsupervised Neural Experts.

Three panels. Reuse the stitching drawing. Fix “mearged”.

1. **Partition.** Cut L1 into many small convex micro-buckets. This is the easy clustering.
2. **Stitch, offline.** Each micro-bucket gets its mean sample gradient. Cluster those means into K experts. Disconnected regions of L1 may share one expert. This never runs during play.
3. **Play.** Shared L1 is already updated. A linear map names micro-bucket \(m\). A static table \(T[m]\) names the expert. One head runs. No teacher label, no gradient, no second network.

Why this is the answer to “why does this work?”: we stopped asking one dispatcher to draw a non-convex gradient manifold. The manifold is handled once, offline. Online, we only ask L1 to name a convex cell.

### 9. What we achieved

One scoreboard. Purpose sentence repeated in the corner, smaller.

Achieved, on the weaker base (checkpoint CE 0.6537):

- Best single-stage L1 hard MoE: **0.6499**.
- KITSUNE routed mixture, K = 16, factor 2: **0.65043**. About 0.0033 below the base, and 0.0005 above that L1 run. It matches. It does not win.
- Best expert on its own bucket, factor 4: **0.60234**. That is a 7.85% reduction against the full-base CE, on that expert’s positions. It is evidence the heads specialize. It is not the mixture’s score.

Not achieved, same slide, short:

- Strong-base KITSUNE is open.
- Factor 8 at K = 16 is worse (mixture 0.65138). On the reported sweep, useful micro-bucket ratios are 2 and 4.
- No Elo. No nodes per second. Hardware numbers are the budget, not a measurement.

Do not conclude “the refinement factor must be way bigger.” Slide 22 says that, and it is stronger than Chapter 6. If someone asks, the plot can come from backup: more micro-clusters were tried; the measured mixture does not keep improving. The thesis claim is the two-stage split, plus the weak-base numbers above.

### 10. Close

One sentence, then stop.

> We can bucket a small chess net by what each position wants the head to learn, and we can route that choice with a lookup the search can afford. The gain shows up when the net still has room. The routing gap is the part that remains.

Thank you. Questions.

No outlook diagram in the spoken close. The next-step picture (slide 23) is backup: load-balanced stitching, strong-base rerun, Cfish integration, Elo.

## What happens to each pre-defense slide

| Pre-defense | Verdict |
|---|---|
| 1 Title | Keep. Swap the subtitle for the purpose sentence. Fix “Supervisor”. |
| 2 Big picture, two families | Fold the contrast into slide 2. Cut the keyword pile. |
| 3 Base NNUE drawing | Keep one drawing, on slide 3. |
| 4 Incremental update, 4 steps | Keep as the body of slide 3. Best slide in the deck. |
| 5 and 6 Accumulator, duplicated | Drop the duplicate. One sentence survives inside slide 3: L1 is recycled across moves. |
| 7 Base architecture | Drop if slide 3 already shows the net. |
| 8 MoE architecture | Becomes the picture on slide 4. |
| 9 Parameter counts | Backup only, after the three totals are reconciled. |
| 10–11 Gradient clustering pictures | Keep one, on the formula slide. |
| 12 Training protocol, 150M / 7M / 175k | Spoken aside on slide 3. Not its own slide. |
| 13 All architectures we tried | Drop. One spoken clause on slide 4: soft-gated and Switch did not beat the base. |
| 14 98% vs 52.5% | Keep. This is half of slide 6. |
| 15 Strong-base CE table | Compress to the three numbers on slide 7. Full table in backup. |
| 16 Fine-tuning trajectories | Backup. Mention Rprop only if asked: it is what made expert fine-tuning stable. |
| 17 KITSUNE 1–2–3 | Keep as the skeleton of slide 8. |
| 18–19 Stitching drawings | Keep one, inside slide 8. Fix the typo. |
| 20 Online path, no gradients | Keep as panel 3 of slide 8. |
| 21 KITSUNE table | Mine three numbers for slide 9. Full table in backup. |
| 22 “Needs to be way bigger” | Backup only. Do not defend it as a thesis result. |
| 23 Outlook process | Backup. |
| 24 Take-home | Rewrite as slide 10. Remove the “only when mini-clusters are very high” claim from the spoken close. |
| 25 Thank you | Keep, attached to slide 10. |
| 26–27, 31, 39 blanks and stray images | Drop. |
| 28 “Some more slides…” | Replace with a one-word backup divider, not spoken. |
| 29–30, 32 Related work and spare figures | Backup. Cite only if a professor names a paper: Nasu (NNUE), de Man (Cfish), Shazeer / Switch, Kawata on latent cluster structure, ELREA, GradientSpace, TAGCOS. |
| 33 Hardware budget | Fold the three specs into slide 2. |
| 34 Silhouette 0.191 vs 0.003 | Fold into slide 6. |
| 35 Menu of bucket signals | Drop as a table. Slide 6 already says which two signals matter. Piece-count is the hand-made baseline and stays a row in the backup table. |
| 36 Oracle 0.6005 | Fold into slide 7. |
| 37 Weaker base 0.6537 → 0.6499 | Fold into slides 7 and 9. |
| 38 Hard MoE diagram | This is slide 4. Prefer it over the busier MoE drawing if you must choose one. |

## How the slides should feel

The pre-defense explains. The defense should show one object and one number.

- One idea per slide. The formula slide is allowed two lines of math and one sentence.
- Type large enough that the back row can read the percentage without the spoken sentence.
- Dark boards and the existing diagrams stay. They are the flash. New decoration is unnecessary.
- No literature slide in the main line. No “all the things we implemented” graph.
- Animations: build slide 8 as three beats, and slide 6 as “expectation, then number”. Everything else appears at once.
- Spoken language stays concrete. “Bucket”, “head”, “lookup”. Define MoE, NNUE, and WDL once, on slides 2–4, then keep using them.
- Cross-entropy is “lower is better”, said once on slide 7. Do not convert it to Elo in your head while talking.

## Backup, not spoken

Five slides, after the close, in this order. Open one only when the question names it.

1. **Formula, unpacked.** Loss is soft cross-entropy against the teacher WDL. Gradient via autograd, batch-wise, head parameters only. L2-normalize. Mini-batch k-means on a 2M-position subsample. Full storage of fp32 gradients is hundreds of gigabytes; the run stores fp16 because direction is what clustering uses.
2. **Strong-base table.** Reference 0.6303, piece-count K = 8 at 0.6303, L1 K = 16 at 0.6300, gradient K = 8 at 0.6300, soft-gated top-2 at 0.6304. Oracle column from the current slide 36.
3. **KITSUNE table.** The seven rows from slide 21 (Table 5.16). Footer: expert gain uses full-base CE, not a same-bucket baseline. Factor 8 is the worse row.
4. **Hardware and deployment.** 120 MHz, 192 KB, 512 KB. Cfish is the search. Integration is specified. Speed and Elo are not measured.
5. **Related work, one line each.** Only the papers already on slide 30.

## Rehearsal checks

- You can write \(\Delta_i\) from memory and point at \(w_{\text{head}}\) versus L1.
- You can walk KITSUNE in three beats without the word “stitch” doing all the work: cut, group by mean gradient, look up.
- Asked “is it dense?”, you answer from slide 4 in one sentence.
- Asked “why did the strong net not move?”, you give 0.6303, 0.6005, and 0.6301, then the weak-base pair.
- You never say the mixture beat the best L1 router. You say it matched it, and that the routing gap is the result.
