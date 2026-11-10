# Whole-Page Optimization with Global Constraints: a build-it-yourself guide

This guide walks you through one paper, section by section, and has you rebuild
its method and experiments in this repo:

> Weicong Ding, Dinesh Govindaraj, and S V N Vishwanathan. 2019. **Whole Page
> Optimization with Global Constraints.** KDD '19 (Applied Data Science Track),
> pp. 3153–3161. https://doi.org/10.1145/3292500.3330675

and the Amazon patent filed on the same system,
**US 10,915,336 B1**, "Optimizing content engagement with imposed content
constraints" (filed Nov. 5, 2018; granted Feb. 9, 2021).

It assumes linear algebra and an intro AI course (search, gradient descent,
linear and logistic regression, basic probability). Everything beyond that
(submodularity, Lagrangian duality, Bayesian linear regression) is taught here,
just in time, with pointers to the best place to read more.

Citation convention: **[n]** is reference *n* in the KDD paper's bibliography
(full list in [`REFERENCES.md`](REFERENCES.md)). **[X1]–[X11]** are outside
references this guide adds. **Eq. (k)**, **Alg. k**, **Fig. k** and **Table k**
always mean the KDD paper's numbering.

---

## Contents

0. [How to use this guide](#0-how-to-use-this-guide)
1. [The problem (paper Abstract and Section 1)](#1-the-problem-paper-abstract-and-section-1)
2. [Problem formulation (Section 2)](#2-problem-formulation-paper-section-2)
3. [Model-based submodular diversity (Section 3)](#3-model-based-submodular-diversity-paper-section-3)
4. [Impression constraints (Section 4.1)](#4-impression-constraints-paper-section-41)
5. [The primal-dual algorithm (Section 4.2)](#5-the-primal-dual-algorithm-paper-section-42)
6. [Greedy page composition (Section 4.3)](#6-greedy-page-composition-paper-section-43)
7. [Offline experiments (Section 5)](#7-offline-experiments-paper-section-5)
8. [Online experiments (Section 6)](#8-online-experiments-paper-section-6)
9. [The patent, mapped to the code](#9-the-patent-mapped-to-the-code)
10. [Moving to real data](#10-moving-to-real-data)
11. [Extensions](#11-extensions-pick-one)
12. [Definition of done](#12-definition-of-done)
- [Appendix A: glossary](#appendix-a-glossary)
- [Appendix B: answers to the self-checks](#appendix-b-answers-to-the-self-checks)
- [Appendix C: bugs you will probably hit](#appendix-c-bugs-you-will-probably-hit)

---

## 0. How to use this guide

### 0.1 The loop

For every section:

1. **Read the paper section** named in the heading. Short sections, dense math.
2. **Read the matching guide section.** It re-derives the math slowly, gives a
   worked example, and points to background reading.
3. **Do the exercise.** Each one is a function in `src/wpo/` that currently
   raises `NotImplementedError("TODO: ...")`. The message names the guide section.
4. **Run its tests.** `pytest tests/test_features.py` etc. Green means done.
5. **Answer the self-check questions** without looking. Answers are in Appendix B.

When all exercises pass, `python experiments/run_all.py` regenerates the paper's
tables and figures from **your** code. Compare with `results/reference/`.

If you are stuck for more than ~30 minutes on an exercise, read the matching
block in `solutions/wpo/`. The point is to understand it, not to suffer.

### 0.2 Map: paper ↔ guide ↔ code ↔ tests

| Paper | Guide | Your code (`src/wpo/`) | Tests |
|---|---|---|---|
| §2, Eq. (1) | §2 | `world.py` (given) | `test_infrastructure.py` |
| §3, Eq. (2)–(3) | §3.3–3.4 | `features.py`: `diversity_gain`, `diversity_features` | `test_features.py` |
| §3, Eq. (4) | §3.5 | `model.py`: `BayesianLinearModel.fit` | `test_model.py` |
| §4.1, Eq. (5)–(7) | §4 | `baselines.py` (given) | `test_greedy.py` (slotting) |
| §4.2, Eq. (8), Alg. 1 | §5 | `primal_dual.py`: `dual_update`, `solve_duals` | `test_primal_dual.py` |
| §4.3, Eq. (9), Alg. 2 | §6 | `greedy.py`: `compose_pages` | `test_greedy.py` |
| §5, Eq. (10) | §7.2 | `metrics.py`: `div_pair`, `miss_at_n` | `test_metrics.py` |
| §5, Tables 1–3, Figs. 2–3 | §7.3–7.8 | `experiments/run_all.py` (given) | compare with `results/reference/` |

Eight functions, roughly 60 lines of code in total. The rest of the repo
(simulator, data loaders, experiment harness, demo app) is provided so your
effort goes into the paper's ideas.

### 0.3 Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
WPO_IMPL=solutions pytest -q          # reference passes: your environment works
pytest -q                             # your src/: 22 failures until you do the exercises
python experiments/run_all.py --impl solutions   # ~20 s, writes results/
```

---

## 1. The problem (paper Abstract and Section 1)

### 1.1 What is being optimized

A streaming homepage (Fig. 1 of the paper shows Prime Video) is a vertical
stack of **widgets**, also called carousels or rows. Each widget is a themed
list of titles: "Popular movies", "New to rent or buy", "Starz channel
movies". The system has several hundred candidate widgets per user and must
choose and order the top *n*. Choosing what goes *inside* a widget is a
separate problem the paper explicitly sets aside [23].

The paper names three requirements that pull against each other:

1. **Relevance**: put rows this user will engage with near the top.
2. **Diversity**: avoid a page of near-duplicates. The paper cites evidence
   that diverse pages raise engagement [7, 14, 18, 26–28].
3. **Constraints**: meet business requirements, such as giving enough exposure
   to product lines the business wants people to discover.

Its contribution is the first framework to handle all three at once.

### 1.2 Why the product lines create a constraint

Amazon Video has three product lines (Section 1):

- **SVOD**: subscription video on demand, included with Prime.
- **TVOD**: transactional video on demand, rent or buy per title.
- **3P**: third-party channels, extra monthly subscriptions.

They compete for the same user budget (money or watch time), and their
"success" signals differ. A user starts a new TV season less often than a
movie, and subscribes to a new channel rarely. A model that only maximizes
short-term engagement will underexpose TVOD and 3P, users never discover them,
and long-term growth suffers [1, 9, 29]. That is the business reason for
impression targets.

### 1.3 Slotting, and the paper's key insight

The naive fix is **slotting**: reserve some positions on *every* page for each
product line. It's simple and meets the target exactly, but it is
one-size-fits-all. It wastes space for a user who never rents and holds back
a user who rents constantly.

The key insight: **enforce the business constraint on average across all
pages, not on each page.** The paper's toy example, worked out:

| | 10 SVOD-loving users | 10 TVOD-loving users | Total SVOD impressions | Total TVOD impressions |
|---|---|---|---|---|
| Target | | | 20 | 20 |
| Slotting (1 + 1 per page) | 1 SVOD, 1 TVOD each | 1 SVOD, 1 TVOD each | 20 ✓ | 20 ✓ |
| Global | 2 SVOD each | 2 TVOD each | 20 ✓ | 20 ✓ |

Both meet the targets. Only the global version gives everyone what they like.

### 1.4 Where prior work sits

- **Relevance ranking** typically scores each widget independently with CTR
  or CVR models [2, 11, 12, 15, 17]. Independence ignores that similar rows
  next to each other bore users [10, 14, 20, 26].
- **Whole-page models** score every possible layout [6, 18, 27]. Optimal in
  principle, but the number of layouts is combinatorial, too slow for a
  high-traffic service.
- **Diversity models** approximate whole-page effects through the *count* of
  similar widgets, which is cheap to optimize [7, 10, 14, 20, 26].
- **Business constraints** have been handled when only one item is shown, or
  with no interaction between items [1, 8, 16, 19, 29].

This paper combines a diversity model with global constraints.

### 1.5 Self-check

1. Why does pointwise ranking (sort by predicted CTR) tend to produce
   undiverse pages?
2. In the 20-user example, what would happen if 15 users loved SVOD and 5
   loved TVOD, with the same targets? Could "global" still give everyone two
   rows of their favorite?
3. Name one cost of the global approach that slotting doesn't have.

---

## 2. Problem formulation (paper Section 2)

### 2.1 Notation, with the code names

| Paper | Meaning | In this repo |
|---|---|---|
| $\mathbf{u}$ | page context: customer, device, time of day | `Users.context[i]` (vector $[1, \text{hist\_prop}, \text{weekend}]$) |
| $\mathcal{U}^t$ | all page contexts in time period $t$ | `Users` for one day, `sample_day(world, t, N)` |
| $\mathbf{c}$ | widget context: id, its items, category metadata | a widget index `0..W-1` |
| $\mathcal{C}^{\mathbf{u}}$ | candidate widgets for page $\mathbf{u}$ | all `W` widgets |
| $\mathcal{C}^{\mathbf{u}}_n$ | the ordered top-$n$ widgets on the page | `pages[i, :n]` |
| $\mathbf{c}^{\mathbf{u}}_i$ | widget in slot $i$ | `pages[i, s]` |
| $\phi(\mathbf{c}, \mathbf{u})$ | relevance feature vector (given) | `relevance_features(...)` |
| $m$ | number of product categories | `world.n_categories` (4) |
| $I(\mathbf{c}) \in \{0,1\}^m$ | one-hot category of a widget | `world.widget_cat[c]` (the index, not the one-hot) |
| $I(\mathcal{C}^{\mathbf{u}}_n) \in \mathbb{R}^m$ | category counts on a page, $\sum_i I(\mathbf{c}^{\mathbf{u}}_i)$ | `category_counts(pages, widget_cat, m)` |
| $f(\mathcal{C}^{\mathbf{u}}_n \mid \mathbf{u}, \mathbf{w})$ | model's value of a page | `model_page_values(R, D, ...)` |

A **category** is a (content type, purchase option) pair, for example
(movie, Prime subscription). In the simulator: `prime_movie`, `prime_tv`,
`rent_buy`, `channel`.

### 2.2 The optimization problem, Eq. (1)

Over all pages in period $t$:

$$
\max_{\{\mathcal{C}^{\mathbf{u}}_n : \mathbf{u}\in\mathcal{U}^t\}}
\;\sum_{\mathbf{u}\in\mathcal{U}^t} f(\mathcal{C}^{\mathbf{u}}_n \mid \mathbf{u}, \mathbf{w})
\qquad\text{s.t.}\qquad
\sum_{\mathbf{u}\in\mathcal{U}^t} g(\mathcal{C}^{\mathbf{u}}_n) \le \mathbf{0}
\tag{1a, 1b}
$$

Read it as: "choose a page for every visit today to maximize total predicted
value, subject to constraints on the *sum* over all visits." The inequality is
element-wise, one per constraint.

Two things to notice:

- **The constraint couples the pages.** Without it, each page could be
  optimized alone. With it, what you show user A affects what you must show
  user B. Section 4.2's job is to undo this coupling.
- **$g$ is separable**: a sum over pages, and later over slots. That structure
  is what makes the decoupling trick work.

$f$ carries relevance and diversity. $g$ carries the business requirements.

### 2.3 Prior work as special cases of Eq. (1)

This is a useful habit when reading papers: find the knob settings that turn
the new formulation into old ones.

- **$n = 1$ or a modular (additive) $f$** gives the setting of constrained ad
  allocation and widget selection [1, 5, 8, 19, 21, 24] and bandits with
  budgets [3, 22, 25, 29]. Those methods model relevance linearly with local
  or global constraints but ignore whole-page interactions.
- **No $g$** gives whole-page optimization with diversity [4, 7, 20, 26, 28]
  or pairwise interactions [18, 27]. Then Eq. (1) splits into independent
  per-page problems.

### 2.4 Why not put the constraint in the loss?

An alternative [9] adds the constraint as a penalty while *training* $f$. The
paper argues against it: then $\mathbf{w}$ is trained to do two jobs at once,
and every change of business target means retraining. In Eq. (1), $\mathbf{w}$
only predicts page value, so targets can change per period **without
retraining**. You will test this property in the definition of done (§12).

### 2.5 Self-check

1. Write $I(\mathcal{C}^{\mathbf{u}}_n)$ for a page with rows of categories
   (prime_movie, rent_buy, prime_movie, channel), with $m = 4$.
2. What does "separable" buy us? Give an example of a non-separable
   constraint.
3. If $|\mathcal{U}^t| = 10^6$ pages and each page has 100 candidates and
   $n = 10$, roughly how many joint configurations does Eq. (1) range over?

---

## 3. Model-based submodular diversity (paper Section 3)

This section defines $f$, the page value model. Within one page, drop the
superscript $\mathbf{u}$ and write the page as $C_n$.

### 3.1 From independent rows to rows that depend on what's above, Eq. (2)

The independent model scores each row alone: $f(C_n) = \sum_i f(\mathbf{c}_i \mid \mathbf{u}, \mathbf{w})$.
The paper instead lets each row's value depend on the rows above it:

$$
f(C_n \mid \mathbf{u}, \mathbf{w}) = \sum_{i=1}^{n} f(\mathbf{c}_i \mid C_{i-1}, \mathbf{u}, \mathbf{w})
\tag{2}
$$

The implicit assumption is that **users scan the page top to bottom**, so row
$i$ is experienced in the context of rows $1..i-1$. This is the same idea as
cascade models of clicks in web search [X5]. It's an assumption, not a fact:
carousel research suggests users also scan horizontally and skip around
[X8]. Keep it in mind when you interpret results.

The value being predicted is the probability the user streams something from
the row: $f(\mathbf{c}_i \mid C_{i-1}, \mathbf{u}, \mathbf{w}) \approx \Pr(y_i = 1)$,
where $y_i = 1$ if they stream from row $i$. Several rows on a page can be 1.

### 3.2 A short primer on submodularity

A set function $F$ (input: a set; output: a number) is **submodular** if adding
an item helps less when you already have more:

$$
A \subseteq B,\; x \notin B \;\Longrightarrow\;
F(A \cup \{x\}) - F(A) \;\ge\; F(B \cup \{x\}) - F(B).
$$

That is the formal version of diminishing returns. It is **monotone** if
adding never hurts: $F(A \cup \{x\}) \ge F(A)$.

Example. Let $F(S) = \log(1 + \#\text{movie rows in } S)$. The first movie row
adds $\log 2 \approx 0.69$, the second adds $\log 1.5 \approx 0.41$, the
third $\log(4/3) \approx 0.29$. Each extra movie row is worth less, which is
exactly what we want diversity to mean.

Why care? Monotone submodular functions are the class where the **greedy
algorithm is provably good** (§6.1). Read [X2] Sections 1–3 for a friendly,
thorough introduction. Useful closure facts: a non-negative weighted sum of
submodular functions is submodular, and adding a modular (purely additive)
function keeps submodularity.

### 3.3 The diversity gain $\Delta\rho$ — Exercise 1

Let $\rho(x) = \log(1 + x)$, applied to each category count. The page's
diversity is $\mathbf{1}^\top \boldsymbol\rho(I(C_n))$. The **incremental
diversity gain** of adding row $\mathbf{c}_i$ is

$$
\Delta\boldsymbol\rho(\mathbf{c}_i, C_{i-1}) = \boldsymbol\rho(I(C_i)) - \boldsymbol\rho(I(C_{i-1})).
$$

Since only the count of $\mathbf{c}_i$'s own category changes, this vector is
zero except in that category, where it equals $\rho(k+1) - \rho(k)$ for $k$ =
rows of that category already above.

| Rows of this category above | Gain $\log\frac{k+2}{k+1}$ |
|---|---|
| 0 | 0.693 |
| 1 | 0.405 |
| 2 | 0.288 |
| 3 | 0.223 |

The gains **telescope**: summing over the page gives
$\sum_i \Delta\boldsymbol\rho = \boldsymbol\rho(I(C_n))$. That identity is why
the per-row model of Eq. (3) adds up to a page-level relevance + diversity
objective (§3.6).

> **Exercise 1.** Implement `diversity_gain(count_before)` in
> `src/wpo/features.py`. One line. Tests: `test_gain_values`,
> `test_gain_diminishes`, `test_gains_telescope_to_rho`.

### 3.4 Personalized diversity: $\mathbf{u} \times \Delta\boldsymbol\rho$ — Exercise 2

The full per-row model, Eq. (3), is linear in features:

$$
f(\mathbf{c}_i \mid C_{i-1}, \mathbf{u}, \mathbf{w}) =
\big[\, \phi(\mathbf{c}_i, \mathbf{u})^\top,\; (\mathbf{u} \times \Delta\boldsymbol\rho(\mathbf{c}_i, C_{i-1}))^\top \,\big]\, \mathbf{w}
\tag{3}
$$

where $[\cdot,\cdot]$ is concatenation and $\mathbf{u} \times \Delta\boldsymbol\rho$
is the **outer product** of the context vector (length $d_u$) with the gain
vector (length $m$), flattened to length $d_u m$.

Why the outer product? If diversity features were just $\Delta\boldsymbol\rho$,
every user would get the same diversity weight per category. With the outer
product, the effective weight on category $k$ becomes a linear function of the
context:

$$
D_{\mathbf{u},k} = \mathbf{u}^\top W^{\text{div}}_{:,k},
\qquad W^{\text{div}} \in \mathbb{R}^{d_u \times m}.
$$

So a user who has shown no interest in a category can get *less* push toward
diversifying into it (the paper makes this point at the end of Section 3).

**Linear algebra view, used throughout the code.** Stack all users' contexts
as rows of $U$ ($N \times d_u$). Then $D = U W^{\text{div}}$ is an $N \times m$
matrix, one matrix product. Likewise the relevance part of every (user,
widget) pair is an $N \times W$ matrix $R$. The score of placing widget $c$
next on user $u$'s page is

$$
R_{u,c} + D_{u,\text{cat}(c)} \cdot \Delta\rho\big(\text{count of cat}(c)\text{ above}\big).
$$

That's Eq. (3) rearranged so all users can be processed at once.
`PageScorer` in `model.py` builds $R$ and $D$; `test_features_agree_with_vectorized_scorer`
checks the two forms agree.

> **Exercise 2.** Implement `diversity_features(context, widget_cats, count_before, m)`.
> Layout: index `j * m + k` is "context feature $j$ times gain in category
> $k$". Test: `test_diversity_features_layout`, then
> `test_features_agree_with_vectorized_scorer`.

### 3.5 The probability model, Eq. (4) — Exercise 3

The paper puts a Gaussian likelihood on the outcome and a Gaussian prior on
each weight:

$$
p(y_i \mid \mathbf{w}, \mathbf{x}_i) = \mathcal{N}(y_i;\, \mathbf{x}_i^\top \mathbf{w},\, \beta_0^2),
\qquad
p(\mathbf{w}) = \prod_d \mathcal{N}(w_d;\, \mu_0,\, \sigma_0^2)
\tag{4}
$$

with $\beta_0 = 1$, $\mu_0 = 0.001$, $\sigma_0^2 = 1$ (Section 5). Yes, a
Gaussian on a 0/1 outcome is a modeling shortcut; it makes this a linear
regression on $y$.

The paper learns the posterior with online message passing [15, 17], which
scales to production traffic and updates as data arrives. For a laptop-sized
problem, the posterior has a **closed form** (Bayesian linear regression,
[X4] Section 3.3). Stack features into $X$ and outcomes into $\mathbf{y}$:

$$
A = \frac{1}{\sigma_0^2} I + \frac{1}{\beta_0^2} X^\top X,
\qquad
\mathbf{m}_N = A^{-1}\!\left(\frac{\mu_0}{\sigma_0^2}\mathbf{1} + \frac{1}{\beta_0^2} X^\top \mathbf{y}\right),
\qquad
S_N = A^{-1}.
$$

Two things worth seeing for yourself:

- **The posterior mean is ridge regression.** With $\mu_0 = 0$, $\mathbf{m}_N$
  minimizes $\|\mathbf{y} - X\mathbf{w}\|^2 + \lambda\|\mathbf{w}\|^2$ with
  $\lambda = \beta_0^2/\sigma_0^2$. The prior *is* the regularizer. The test
  `test_posterior_mean_is_ridge` checks this against scikit-learn.
- **The covariance $S_N$ measures uncertainty.** It shrinks as data grows.
  Alg. 2 line 1 samples $\mathbf{w}$ from this posterior (Thompson sampling,
  §6.3), so uncertain rows get explored.

Use `np.linalg.solve(A, rhs)` for the mean, not `inv(A) @ rhs`; it's faster and
more stable.

> **Exercise 3.** Implement `BayesianLinearModel.fit(X, y)` in `model.py`.
> Set `self.mean` and `self.cov`. Tests: all of `test_model.py`.

**The non-negativity caveat.** Eq. (3) is submodular (so greedy is safe) only
when the diversity weights are non-negative. A Gaussian prior doesn't enforce
that. The paper reports that the learned weights came out non-negative on its
data anyway. **Check yours**: `summary.json` reports
`frac_nonneg_div_weights` (1.0 in the reference run). If it isn't 1.0, you
have a genuine finding to discuss, and fixes to try (a truncated prior, or
clipping $D$ at 0).

### 3.6 Page-level form, and what it means

Summing Eq. (3) down the page and using the telescoping identity:

$$
f(C_n \mid \mathbf{u}, \mathbf{w}) =
\Big[\textstyle\sum_i \phi(\mathbf{c}_i, \mathbf{u})^\top,\; (\mathbf{u} \times \boldsymbol\rho(I(C_n)))^\top\Big]\mathbf{w}.
$$

This is the standard "relevance + diversity regularizer" objective for picking
top-$n$ items [4, 7, 14, 26], with one difference: the diversity weights are
**learned from data and context-aware**, not hand-tuned constants. That is why
the paper calls it *model-based* diversity.

### 3.7 Our relevance features $\phi$

The paper treats $\phi$ as given. Ours (`relevance_features`) are deliberately
simple: a bias, the dot product of observed user taste with the widget's genre
mix, the user's historical propensity for the widget's category, and a one-hot
widget id. Models never see the simulator's true tastes, only noisy
observations.

### 3.8 Self-check

1. Show that $F(S) = \max_{s \in S} v_s$ (value of the best item) is
   submodular.
2. Why would a single global diversity weight per category be a bad idea on
   a service where 45% of users never rent?
3. With $\beta_0 = 1$ and $\sigma_0 = 0.5$, what ridge $\lambda$ is the
   posterior mean equivalent to?
4. If the learned weight on (weekend × gain in `channel`) is large and
   positive, what does that say about weekend pages?

---

## 4. Impression constraints (paper Section 4.1)

### 4.1 The constraint, Eq. (5)

Restrict $g$ to be separable across slots, $g(C^{\mathbf{u}}_n) = \sum_i g(\mathbf{c}^{\mathbf{u}}_i)$.
The constraint the paper studies is a **minimum average number of rows per
category, per page**:

$$
\mathbf{b}^t - \frac{1}{|\mathcal{U}^t|}\sum_{\mathbf{u}\in\mathcal{U}^t} I(C^{\mathbf{u}}_n) \;\le\; \mathbf{0}
\tag{5}
$$

$\mathbf{b}^t \in \mathbb{R}^m$ is the target balance for period $t$, given by
the business. In our runs: $b = (0, 0, 1.8, 1.8)$ for (prime_movie, prime_tv,
rent_buy, channel), i.e., at least 1.8 rent/buy rows and 1.8 channel rows per
8-row page **on average**. Any single page may have 0 or 4.

The paper's assumption, worth stating in your write-up: if impressions are
balanced across all customers, the downstream business metrics will be too.

### 4.2 The full problem, Eqs. (6)–(7)

Plug Eq. (2) and Eq. (5) into Eq. (1), written as a minimization of the
negative:

$$
\min_{\{\mathbf{c}^{\mathbf{u}}_i\}} \; -\sum_{\mathbf{u}}\sum_{i=1}^{n} f(\mathbf{c}^{\mathbf{u}}_i \mid C^{\mathbf{u}}_{i-1}, \mathbf{u}, \mathbf{w})
\quad\text{s.t.}\quad
\mathbf{b}^t - \frac{1}{|\mathcal{U}^t|}\sum_{\mathbf{u}}\sum_{i} I(\mathbf{c}^{\mathbf{u}}_i) \le \mathbf{0}
\tag{6, 7}
$$

### 4.3 The alternatives the paper rejects

- **Slotting**: pin positions to categories on every page, rank within each
  category by relevance. Meets the balance exactly; one size fits all.
  `baselines.slotting_patterns` implements a fair version: each page gets
  $\lfloor b_k \rfloor$ pinned rows of category $k$ plus one more with
  probability $b_k - \lfloor b_k \rfloor$, so the targets hold in expectation.
  Free slots are still filled by the full relevance + diversity model, so the
  comparison isolates the constraint mechanism.
- **Calibration**: convert each category's conversions into a common
  "currency" and rank by expected value. Simple, but the exchange rates come
  from slow long-term models and can't react to daily changes in content or
  traffic.

### 4.4 Other constraints

The same machinery handles an **expected conversion** constraint,
$\mathbf{b} - \sum_{\mathbf{u}}\sum_i p(\mathbf{c}^{\mathbf{u}}_i) I(\mathbf{c}^{\mathbf{u}}_i) \le 0$,
where $p$ is the conversion probability. The patent's claim 4 also names
impression and conversion constraints. This is one of the extensions (§11).

### 4.5 Two obstacles

Eqs. (6)–(7) can't be used in production as written:

1. **Coupling.** A homepage must render the moment it is requested, but the
   constraint ties every page to every other page.
2. **Unknown future.** The sum is over $\mathcal{U}^t$, today's traffic,
   which you don't have yet when the first request arrives.

Section 4.2 of the paper solves the first with duality and the second by
reusing yesterday's solution.

### 4.6 Self-check

1. With $b_{\text{rent}} = 1.8$, could every page have exactly 1.8 rent rows?
   What does slotting do about that?
2. Why is it reasonable to set $\mathbf{b}^t$ to "the balance the current
   production system already achieves," as the paper does in Section 5?

---

## 5. The primal-dual algorithm (paper Section 4.2)

This is the heart of the paper. If you understand this section, you understand
the method.

### 5.1 Intuition first: a bonus per category

Give each category $k$ a **bonus** $\xi_k \ge 0$ and compose every page
independently by maximizing *model score + bonus*. If a category is
under-shown overall, raise its bonus; if it's over-shown, lower it. Pages are
decoupled (each just sees a fixed bonus vector), and the bonuses carry all the
information about the global constraint.

A two-user example shows why this gets the cheapest trade-offs automatically.
One slot per page; target: at least 0.5 rental rows per page on average.

| | Score of drama (SVOD) | Score of rental (TVOD) |
|---|---|---|
| User 1 | 0.6 | 0.5 |
| User 2 | 0.7 | 0.2 |

- $\xi = 0$: both see the drama. Rentals per page: 0. Constraint missed.
- User 1 switches to the rental once $0.5 + \xi > 0.6$, i.e. $\xi > 0.1$.
  User 2 needs $\xi > 0.5$.
- So any $\xi \in (0.1, 0.5]$ shows the rental to user 1 only: average 0.5
  rentals ✓, at a cost of $0.6 - 0.5 = 0.1$ in total score.

The bonus finds the user for whom showing the rental costs least. Slotting
would show it to both, costing $0.1 + 0.5 = 0.6$. That is the paper's whole
story in miniature.

### 5.2 Formally: the Lagrangian and the dual, Eq. (8)

Attach a multiplier $\boldsymbol\xi \in \mathbb{R}^m$, $\boldsymbol\xi \ge 0$,
to the constraint and form the Lagrangian of Eqs. (6)–(7). The paper's dual
form is

$$
\max_{\boldsymbol\xi \ge 0}\; \min_{\{\mathbf{c}^{\mathbf{u}}_i\}}\;
-\sum_{\mathbf{u}}\Big(\sum_i f(\mathbf{c}^{\mathbf{u}}_i \mid C_{i-1}, \mathbf{u}, \mathbf{w})
+ \boldsymbol\xi^\top\big(\mathbf{b} - \textstyle\sum_i I(\mathbf{c}^{\mathbf{u}}_i)\big)\Big)
\tag{8}
$$

How to read it, step by step:

1. For a **fixed** $\boldsymbol\xi \ge 0$, the inner minimization is a
   relaxed problem: violations are allowed but penalized at price $\xi_k$ per
   missing row (and rewarded per extra row).
2. For any $\boldsymbol\xi \ge 0$, the inner value bounds the true constrained
   optimum (weak duality). The outer maximization finds the tightest such
   bound, i.e., the best prices.
3. Why $\boldsymbol\xi \ge 0$? The constraint is an inequality ("at least").
   A negative price would reward missing the target.

The paper calls Eq. (6)–(7) "equivalent" to its dual. For a discrete problem
like ours, strong duality isn't guaranteed in general; treat the dual as an
excellent, practical relaxation. Your LP check in §7.8 shows how tight it is
in the special case where the problem is a linear program. For the theory, read
[X3] Chapter 5 (Sections 5.1–5.5 are enough). Sign conventions vary between
sources; always re-derive the direction of your update rather than copying a
formula.

### 5.3 Decoupling, Eq. (9)

The term $\boldsymbol\xi^\top\mathbf{b}$ doesn't depend on the pages, and the
rest is a sum over pages. So for fixed $\boldsymbol\xi$, **each page is
solved on its own**:

$$
\arg\max_{\{\mathbf{c}^{\mathbf{u}}_i\}} \sum_{i=1}^{n} f(\mathbf{c}^{\mathbf{u}}_i \mid C_{i-1}, \mathbf{u}, \mathbf{w}) + \boldsymbol\xi^\top I(\mathbf{c}^{\mathbf{u}}_i)
\tag{9}
$$

That's the unconstrained page problem (Eq. 2) plus a per-category bonus,
exactly the intuition of §5.1. The dual variables are the knob that trades
page value against the global balance.

### 5.4 The dual update — Exercise 4

With pages fixed, the outer objective is linear in $\boldsymbol\xi$, and its
gradient with respect to $\boldsymbol\xi$ (in the "maximize" direction, after
the paper's normalization by $|\mathcal{U}^t|$) is

$$
\mathbf{b} - \frac{1}{|\mathcal{U}^t|}\sum_{\mathbf{u}}\sum_i I(\mathbf{c}^{\mathbf{u}}_i)
\;=\; \mathbf{b} - \overline{I},
$$

the **shortfall**: target minus the average count per page. Gradient ascent,
plus two practical additions:

$$
\boldsymbol\xi \leftarrow \Big[\boldsymbol\xi + \eta_j(\mathbf{b} - \overline{I}) - \eta_j\gamma_j\boldsymbol\xi\Big]_+
\qquad\text{(Alg. 1, line 6)}
$$

- $[x]_+ = \max(x, 0)$ element-wise is the **projection** back onto
  $\boldsymbol\xi \ge 0$ (projected gradient ascent).
- $-\eta_j\gamma_j\boldsymbol\xi$ comes from an $\ell_2$ penalty
  $\gamma\|\boldsymbol\xi\|^2$ that keeps the prices from growing without bound,
  a standard trick in this literature [8, 16, 19].

Read the update in words: *a category that is short gets a bigger bonus; a
category over its target sees its bonus decay toward zero.* Unconstrained
categories ($b_k = 0$) always have shortfall $\le 0$, so their bonus stays 0.

> **Exercise 4.** Implement `dual_update(xi, b, avg_counts, eta, gamma)` in
> `primal_dual.py`. One line. Tests: `test_update_direction_and_projection`,
> `test_regularization_shrinks`.

### 5.5 Algorithm 1 — Exercise 5

```
xi = 0
for j = 1..J:
    compose every page with Eq. (9) using the current xi      # Alg. 2, §6
    avg = average category counts over all pages
    record (j, miss@n, xi, avg)
    if miss@n <= eps: stop                                     # line 8, Eq. (10)
    xi = dual_update(xi, b, avg, eta0 / j, gamma0 / sqrt(j))   # line 6
```

The paper uses diminishing step sizes $\eta_j = 0.01/j$, $\gamma_j = 0.1/\sqrt{j}$,
$\epsilon = 0.05$, $J = 50$, starting from $\boldsymbol\xi = 0$. Its gradient
is normalized differently from ours (per-page average vs. sum), so **the
numerical value of $\eta_0$ does not transfer; tune it.** What you will see:

- $\eta_0$ too small: with $\eta_j = \eta_0/j$, the total distance the bonus
  can travel after $J$ steps is about $\eta_0 \ln J$ times the shortfall
  (the harmonic series). The bonus stalls short of where it needs to be.
- $\eta_0$ too large: the first step overshoots, every page gets flooded with
  the constrained categories, and page value drops.
- In the reference run, $\eta_0 = 0.06$ reaches $\epsilon = 0.02$ in 34
  iterations; $\eta_0 = 0.1$ meets the target in one step but overshoots and
  costs more page value.

Record history **before** updating so `history[0]` is the unconstrained
starting point.

> **Exercise 5.** Implement `solve_duals(...)`. Return `(xi, pages, history)`.
> Test: `test_solver_reduces_miss_and_records_history`.

### 5.6 The unknown-future problem: use yesterday's prices

Alg. 1 needs all of $\mathcal{U}^t$. The paper's fix: run Alg. 1 **offline on
yesterday's traffic** $\mathcal{U}^{t-1}$, and serve today's requests with
$\boldsymbol\xi^t \approx \boldsymbol\xi^{t-1}$. This assumes the traffic mix
and targets change slowly from one period to the next. Similar strategies
appear in [1, 13, 16]. Fig. 3 tests the assumption, and so will you (§7.4).

The resulting production system has three modules:

1. **Training**: learn $f(\cdot \mid \cdot, \mathbf{w})$ from logs (§3.5).
2. **Dual optimization**: run Alg. 1 on $\mathcal{U}^{t-1}$, $\mathbf{b}^{t-1}$ (offline, daily).
3. **Online composition**: render each request with Alg. 2 and $\boldsymbol\xi^{t-1}$ (§6).

### 5.7 Convergence, and how this relates to the literature

When $n = 1$ and $f$ is linear, Alg. 1 reduces to established
online/primal-dual algorithms with proven convergence and regret bounds [1, 8,
13, 16, 19]. For the general case the paper shows convergence empirically
(Fig. 2) and leaves the analysis to future work. Related threads:

- online convex optimization with long-term constraints [21, 22, 29], where the
  constraint only needs to hold on average over time, a close cousin of
  "on average over pages";
- primal-dual online ranking with constraints, applied to web traffic shaping [24];
- email volume optimization at LinkedIn [16] and delivery-based ad allocation
  [19], both production systems that use per-constraint dual prices.

### 5.8 Self-check

1. In the two-user example, what happens if $\xi = 0.6$? Is the constraint
   met? Is it the cheapest solution?
2. Why does projecting onto $\boldsymbol\xi \ge 0$ matter for a "minimum"
   constraint? What would change for a "maximum" constraint?
3. The update for an unconstrained category is always 0. Why?
4. List two ways the "yesterday's $\boldsymbol\xi$" assumption can fail.

---

## 6. Greedy page composition (paper Section 4.3)

### 6.1 Why greedy is good enough

Eq. (9) is a monotone submodular objective (when the diversity weights and
$\boldsymbol\xi$ are non-negative) plus a modular term. For maximizing a
monotone submodular set function under a cardinality limit, greedy
("repeatedly add the item with the largest marginal gain") achieves at least
$1 - 1/e \approx 63.2\%$ of the optimum [X1]. The paper invokes this bound for
Alg. 2.

Two honest caveats to discuss in your write-up:

- The guarantee is about **sets**. A page is an **ordered** list. Because each
  row's diversity term depends only on the *count* of its category above, the
  page's value depends on order only through which row lands in which
  position, and greedy places the best marginal row first. Your brute-force
  check (§7.8) measures how close greedy actually gets.
- It needs **non-negative diversity weights** (§3.5).

In practice greedy is often optimal or nearly so; 63% is a worst-case floor.

### 6.2 Algorithm 2, line by line — Exercise 6

```
sample w from the posterior                                  # line 1 (§6.3)
page = []
for i = 1..n:                                                # line 3
    best = -inf  (the paper writes 0; see the warning)       # line 4
    for each remaining candidate a:                          # line 5
        s_a = f(a | rows above, u, w) + xi^T I(a)            # line 6, Eq. (9)
        keep the a with the largest s_a                      # line 7
    append a*, remove it from the candidates                 # lines 9-10
```

**Vectorize it.** Looping over users in Python is slow. Process all $N$ users
at once using the $R$ and $D$ matrices from §3.4:

1. Keep `counts` ($N \times m$, category counts so far) and `used`
   ($N \times W$ booleans).
2. For slot $i$: `gain = diversity_gain(counts[:, widget_cat])`, an
   $N \times W$ matrix of each candidate's gain for each user.
3. `scores = R + D[:, widget_cat] * gain + xi[widget_cat]`.
4. Mask used widgets with $-\infty$; take `argmax` per row; update `used` and
   `counts`.

Slotting reuses the same function: a pinned slot masks out every widget not in
the pinned category. If the pinned category has no unused widget left on that
page, fall back to a free choice (the test covers it).

**Warning about line 4.** Initializing the best score to 0 means a page where
every candidate scores negative gets no widget at all. Initialize to
$-\infty$. `test_negative_scores_still_fill_the_page` checks this. Spotting
small issues like this when you implement a paper is part of the skill.

> **Exercise 6.** Implement `compose_pages(R, D, widget_cat, n_slots, xi, slot_category)`
> in `greedy.py`. Tests: all of `test_greedy.py`, including
> `test_greedy_is_near_optimal`, which checks the $1 - 1/e$ bound against
> brute force on small instances.

### 6.3 Line 1: Thompson sampling

Alg. 2 starts by **sampling** $\mathbf{w}$ from the posterior rather than using
the mean. That is Thompson sampling: rows the model is unsure about sometimes
get optimistic scores and get shown, which generates the data to learn about
them. This balances exploration and exploitation, the same tension the paper
raises between short-term engagement and discovery. See [X9] for an accessible
empirical study and [15, 18] for production uses. Our experiments use the
posterior mean for clean comparisons; adding sampling is an extension (§11).

### 6.4 Self-check

1. Give a 3-widget example where greedy is not optimal for a non-submodular
   objective.
2. Why must `xi` be indexed by `widget_cat` before adding it to the scores?
3. What is the time complexity of your vectorized `compose_pages` in $N$, $W$,
   and $n$?

---

## 7. Offline experiments (paper Section 5)

### 7.1 The paper's setup vs. ours

| | Paper | This repo |
|---|---|---|
| Data | Two weeks of Prime Video homepage requests, Aug. 2018 | Simulator (`world.py`): 4 categories, 60 widgets, 3,000 users/day, 14 days |
| Time period $t$ | one day | one day |
| Training | page model updated daily with the day's impressions, tested on the next day; metrics from week 2 | model trained on days 0–6 of uniformly random logged pages; evaluated on days 7–13 |
| Learning | distributed online message passing [15] | closed-form Bayesian linear regression (§3.5) |
| Targets $\mathbf{b}^t$ | set to the production system's current balance | rent_buy ≥ 1.8 and channel ≥ 1.8 rows per 8-row page |
| Ground truth | not observable (only logged clicks) | **known**: true stream probability of any row on any page |

The last row is the big advantage of a simulator: you can measure the *true*
expected value of pages no one ever logged. The cost is that results describe
the mechanism in a world you designed, not Amazon's customers.

One deliberate simplification: our training logs come from a **uniformly
random** logging policy, so they're unbiased. Real logs come from the
production policy, which shows some rows far more than others, and a model
trained on them inherits that bias. The paper itself flags this when
explaining why the pairwise-interaction baseline (MVT) looked weak: it
composes unusual pages the logs rarely contain. Try a biased logging policy as
an extension.

### 7.2 Metrics — Exercises 7 and 8

- **AUC** on held-out logged rows: how well the model ranks rows that got
  streamed above rows that didn't.
- **Precision@K**: the paper counts positive rows in the top-K of each page
  and divides by K. With ground truth we compute its expectation: the mean
  *true* stream probability of the top-K rows (`expected_precision_at_k`).
- **Div-pair@n**: pairwise dissimilarity of the page. The formula as printed
  in the paper is hard to parse; the standard reading, and what we implement,
  is "the fraction of pairs of rows on a page that are in different
  categories":
  $$
  \text{Div-pair@}n = 1 - \frac{\sum_k I_k (I_k - 1)/2}{n(n-1)/2},
  $$
  averaged over pages. All rows the same category gives 0; all different gives 1.
- **miss@n**, Eq. (10): the average relative shortfall from the targets,
  $$
  \text{miss@}n = \frac{1}{m}\sum_{s=1}^{m}\Big[1 - \frac{\sum_{\mathbf{u}} I_s(C^{\mathbf{u}}_n)}{b^t_s\,|\mathcal{U}^t|}\Big]_+ .
  $$
  For $b_s = 0$ the ratio is undefined, so we average over constrained
  categories only. Alg. 1 uses miss@n as its stopping test.
- **Page reward**: the average of Eq. (6)'s objective, i.e., the model's
  estimate of page value. We also report the **true** page reward (expected
  streamed rows per page under the simulator), which the paper couldn't.

> **Exercise 7.** Implement `div_pair(pages, widget_cat, m)`.
> **Exercise 8.** Implement `miss_at_n(avg_counts, b)`.
> Tests: `test_metrics.py`.

When all eight exercises pass:

```bash
pytest -q                          # all green
python experiments/run_all.py      # your code -> results/
diff -r results/reference results  # same seed, same numbers (figures aside)
```

### 7.3 Replication 1: Algorithm 1 converges (Fig. 2)

**Paper.** On one day of 1M sampled homepages, miss@n falls to ~0 within a
few iterations, a single constraint's miss falls with it, and its dual
variable levels off. The algorithm meets the stopping rule after about 30
iterations.

**Ours** (`results/reference/fig2_convergence.png`). Starting from 29.8%
miss (the unconstrained page), miss drops below 13% after one step and below
$\epsilon = 2\%$ after 34 iterations, while the rent_buy dual rises and
flattens near 0.054. Same shape as the paper.

**What to try.** Vary `--eta0` (0.02, 0.06, 0.2) and watch §5.5's two failure
modes.

### 7.4 Replication 2: yesterday's duals work today (Fig. 3)

**Paper.** Using $\boldsymbol\xi^{t-1}$ on day $t$ gives a small average test
miss. It rises on transitions into and out of the weekend, because the
weekend audience differs. The worst-performing constraint has a high relative
miss because its absolute target is tiny.

**Ours** (`fig3_daily_miss.png`), test miss by evaluation day, with the
oracle (duals tuned on the same day) for comparison:

| Eval day | 1 | 2 | 3 | 4 (wkend) | 5 (wkend) | 6 | 7 |
|---|---|---|---|---|---|---|---|
| Avg miss, $\xi^{t-1}$ | 1.2% | 2.5% | 2.0% | 7.6% | 4.4% | **12.4%** | 1.1% |
| Worst category | 2.0% | 3.7% | 4.0% | 15.3% | 5.1% | 24.9% | 1.7% |
| Oracle | 2.0% | 2.0% | 2.0% | 2.0% | 2.0% | 2.0% | 2.0% |

The spikes come exactly at the transitions, as in the paper. Day 6 is the
worst: it uses prices learned on the second weekend day, when channel lovers
were over-represented, so the channel price was low (it fell from about 0.05
to 0.02). Applied to a weekday audience, channel rows fall short. This is the
assumption of §5.6 failing in a predictable way. A fix to explore:
use "same weekday last week" prices, or update $\boldsymbol\xi$ within the day.

### 7.5 Replication 3: whole-page features help (Table 1)

Unconstrained models, lifts relative to the CVR baseline.

| Metric | Paper: WPO-Free | Paper: DIV | Ours: WPO-Free | Ours: DIV |
|---|---|---|---|---|
| AUC | +4.03% | same as CVR | +2.88% (0.728 → 0.749) | same as CVR |
| Precision@top | +2.22% (P@10) | −1.06% | +6.90% (P@4) | +6.22% |
| Precision@all | +1.56% (P@20) | +1.26% | +10.89% (P@8) | +10.39% |
| Div-pair | +5.91% | +34.4% | +60.2% | +63.9% |
| miss (absolute; CVR first) | CVR 10%, WPO 8.9% | 0% (tuned) | CVR 35.1%, WPO 26.7% | 19.6% |

**What matched.** Adding the context-aware diversity features improves
prediction (AUC) and page quality, and raises diversity. Unconstrained models
still miss the targets, so constraints are needed.

**What differs and why.** Our lifts are larger because the simulator's repeat
penalty is strong (rows of the same category really do lose value), so CVR's
undiverse pages are bad. Our targets were set tighter than "current
production balance," so the unconstrained miss is larger. Our DIV baseline is
unconstrained here; the paper's DIV was hand-tuned to hit the targets.

**Skipped:** MVT [18, 27], the pairwise-interaction whole-page model searched
by hill climbing. Implementing it is an extension.

### 7.6 Replication 4: global constraints are cheap, slotting is not (Table 2)

Lifts relative to WPO-Free (unconstrained, model-based diversity).

| Metric | Paper: Global | Paper: Slot | Paper: DIV | Ours: Global | Ours: Slot | Ours: DIV-Global |
|---|---|---|---|---|---|---|
| Precision@top | −0.07% | −9.08% | −2.36% | −1.44% | −15.05% | −2.64% |
| Precision@all | −1.47% | +0.77% | +0.30% | −1.84% | −7.20% | −1.89% |
| Div-pair | +17.6%* | +22.1% | +19.3% | +3.00% | +15.07% | +4.64% |
| miss (absolute) | 1.5% | 0% | 0.4% | 4.5% | 0% | 3.7% |
| Page reward (model) | −1.31% | **−25.6%** | −4.9% | −1.15% | **−6.90%** | −1.59% |
| Page reward (true) | n/a | n/a | n/a | −1.84% | −7.20% | −1.89% |

\*The paper's Table 2 lists WPO-Free as the baseline for every column.

**What matched.** The headline ordering: global constraints nearly meet the
targets while giving up very little page value; slotting meets them exactly
but gives up much more; slotting produces the most diverse pages (it forces
categories onto every page). The paper reads this as evidence that local
(per-page) constraints are suboptimal, and our true page reward confirms it.

**What differs and why.**
- Slotting costs us 7% rather than 26%. Our slotting fills its free slots with
  the full model and pins only ~3.6 of 8 rows; the paper's production slotting
  specified the purchase type of every one of the top-10 slots.
- Our DIV-Global is nearly as good as WPO-Global (−1.89% vs. −1.84% true).
  In the paper the gap was larger (−4.9% vs. −1.31%). The paper attributes
  WPO's advantage to *personalized* diversity weights. In our simulator, users
  differ only mildly in how much they tolerate repeated categories, so
  personalization has little to exploit. **This is a finding, not a bug:**
  increase the tolerance heterogeneity in `true_click_prob` and see whether
  the gap opens.
- Our test miss (4.5%) is above the paper's 1.5%, driven by the weekend
  transitions of §7.4.

### 7.7 Replication 5: impressions go to the people who want them (Table 3)

How the constrained category's rows are distributed across users bucketed by
propensity, as a % difference from the target $b_X$. The paper buckets by
streams in category X over the past 28 days; we bucket by the simulator's
true affinity for rent_buy (last evaluation day).

| Propensity for X | Paper: DIV | Paper: WPO-Global | Ours: DIV-Global | Ours: WPO-Global | Ours: Slot |
|---|---|---|---|---|---|
| 0 / never | −3.0% | −15.2% | −53.5% | −62.7% | +0.4% |
| low | +1.0% | +7.9% | −6.3% | +6.1% | +8.4% |
| moderate | +15.1% | +32.7% | +19.0% | +31.1% | +17.6% |
| high | +18.1% | +59.2% | +38.9% | +50.0% | +25.5% |
| extremely high | +27.5% | +67.7% | +109.2% | +103.7% | +56.5% |

**What matched.** Exposure rises monotonically with propensity, and the global
approach sends far fewer rentals to people who never rent than the
non-personalized baseline does. Slotting shows rentals to non-renters at the
full target rate, which is exactly the waste the paper set out to remove.

### 7.8 Algorithm checks

From `results/reference/checks.md`:

- **Greedy vs. brute force** (30 users, 9 candidates, $n = 4$, Eq. (9)
  objective): greedy was optimal in every case (min ratio 1.0000, floor
  0.632). The worst-case bound is loose in practice.
- **$n = 1$, linear $f$, vs. an exact LP** (`scipy.optimize.linprog`): the
  LP relaxes pages to probabilities and gives an upper bound. Primal-dual
  matched it to 5 decimal places (0.17533) with 0% miss. The LP's constraint
  duals were (0.0212, 0.0038); ours were (0.0212, 0.0028). The second differs
  because with discrete choices a whole *interval* of prices produces the same
  pages (recall the interval $(0.1, 0.5]$ in §5.1); any price in it is
  optimal.
- **Iterations to $\epsilon$**: 34 at $\eta_0 = 0.06$.

---

## 8. Online experiments (paper Section 6)

**Paper.** Both the hand-tuned DIV model and WPO-Global ran in production A/B
tests, each set up to match the existing category coverage. DIV improved
streaming minutes by 0.44% and distinct streaming days (DSDs) by 0.12% over
the production model. WPO-Global then improved minutes by 0.77% and DSDs by
0.32% *on top of* DIV, while meeting the business targets. Minutes measure
short-term activity; DSDs measure stickiness (how many different days people
come back).

**You can't replicate this**: it needs real users. You can run a **simulated
A/B test**: randomize simulated users into arms, serve each arm's policy for
several days, measure true expected streamed rows (a stand-in for minutes) and
a return-day proxy, and report lifts with bootstrap confidence intervals over
users. Label it clearly as simulated. For how real experiments are run and
analyzed, [X11] is the standard reference.

---

## 9. The patent, mapped to the code

US 10,915,336 B1 describes the same system in general legal language (it
names effectiveness scores, constraint values, and content components instead
of page values, duals, and widgets).

| Patent term | Paper term | Code |
|---|---|---|
| Content component | Widget / carousel | widget index |
| Effectiveness score; effectiveness model 122 | $f(\mathbf{c}_i \mid C_{i-1}, \mathbf{u}, \mathbf{w})$, Eq. (3) | `model.py`, `PageScorer` |
| Effectiveness model trainer 124 | posterior learning, Eq. (4) | `BayesianLinearModel.fit` |
| Global constraint (target average inclusions per page) | impression constraint, Eq. (5) | `b` |
| Constraint value; constraint model 126 | dual variables $\boldsymbol\xi$ | `xi` |
| Fig. 3 (adjust scores, pick highest, repeat) | Alg. 2 | `compose_pages` |
| Fig. 4 (obtain constraint, initial value, assemble, adjust) | Alg. 1 + §5.6 | `solve_duals`, `run_all.py` daily loop |
| Initial value from an offline simulation of the prior period | $\boldsymbol\xi^{t} \approx \boldsymbol\xi^{t-1}$ | `xi_prev` in `run_all.py` |

The independent claims are 1 (system), 5 (method, with an ML model doing the
ranking) and 14 (computer-readable medium). All three require scoring
components for effectiveness, adjusting those scores with a value derived
from a global constraint defined as a target average number of inclusions per
page over a time period, and composing the page from the adjusted ranking.
Your code implements one concrete embodiment of that.

---

## 10. Moving to real data

The simulator is the core testbed because only it gives ground truth. Real
data makes the inputs realistic. See [`DATASETS.md`](DATASETS.md) for
downloads, licenses, and schemas. In short:

1. **MovieLens + TMDB** builds a *semi-synthetic* world: real catalog, real
   user tastes from ratings, real weekday/weekend activity, and (with a free
   TMDB key) real stream/rent/buy availability; clicks still simulated.
   ```bash
   python scripts/download_movielens.py
   export TMDB_TOKEN=...            # optional but recommended
   python scripts/fetch_tmdb_providers.py --limit 3000
   python experiments/run_all.py --world movielens --providers data/providers.json
   ```
   Without `--providers`, categories are pseudo-random and the run prints a
   warning. Don't report results from that mode as findings.
2. **ContentWise Impressions** is real carousel impression data from a
   streaming service, with row positions. Use it to calibrate: the category
   mix by row position (the real analogue of "set $\mathbf{b}$ to the
   production balance") and which categories get engaged with.
   ```bash
   python scripts/contentwise_calibrate.py --data-dir data/contentwise
   ```
   Then set your simulator's targets and category shares from what you find.

---

## 11. Extensions (pick one)

Each is a real research-engineering question, sized for a weekend or two.

1. **Make personalization matter.** Increase how differently users tolerate
   repeated categories in `true_click_prob`. Does the WPO-Global vs.
   DIV-Global gap of §7.6 open up toward the paper's?
2. **Stress the "yesterday's prices" assumption.** Add faster drift, or a
   one-day traffic shock. Compare $\boldsymbol\xi^{t-1}$, same-weekday-last-week
   prices, and intra-day dual updates (a step of Alg. 1 every hour, as in the
   online methods of [22, 24, 29]).
3. **Conversion constraints** (§4.4, patent claim 4). Constrain expected
   streams, not impressions, of a category. What changes in the update?
4. **Thompson sampling** (Alg. 2, line 1). Sample $\mathbf{w}$ per page with
   `sample_weights` and run a multi-day loop where each day's served pages
   become the next day's training logs. Compare against the posterior mean.
5. **Biased logs.** Train on logs from the CVR policy instead of uniform
   pages. How much do AUC and page reward degrade?
6. **MVT baseline** [18, 27]. Pairwise features between adjacent rows plus
   hill-climbing search. Can you add global constraints to it?

---

## 12. Definition of done

**Understanding** (explain without notes):
- derive Eq. (8) from Eqs. (6)–(7) and explain why fixed $\boldsymbol\xi$
  decouples the pages (Eq. 9);
- explain the dual update as projected gradient ascent and what $\eta$,
  $\gamma$, and the projection each do;
- state when greedy's $1 - 1/e$ guarantee applies and why it needs
  non-negative diversity weights;
- contrast global constraints with slotting, calibration, hand-tuned
  diversity [26], and penalty-in-the-loss [9].

**Implementation:** all eight exercises pass their tests; three modules are
separable (training, dual optimization, online composition). Prove the
"retarget without retraining" property: change `--target` and only the dual
step reruns.

**Replication:** each row below has a figure or table, marked pass or fail
with an explanation.

| Claim | Pass criterion |
|---|---|
| Whole-page features beat CVR (Table 1) | higher held-out AUC and true page reward |
| Unconstrained models miss targets (Table 1) | clearly non-zero miss@n |
| Alg. 1 converges (Fig. 2) | miss@n below $\epsilon$ within 50 iterations, $\xi$ flattens |
| Global ≫ slotting on page value (Table 2) | ordering Free > Global ≥ DIV-Global > Slot; Global miss small |
| Yesterday's duals work (Fig. 3) | test miss near the oracle on stable days; transitions explained |
| Exposure follows propensity (Table 3) | monotone across buckets, steeper than DIV for the "never" bucket |
| Greedy near optimal (§4.3) | brute-force ratio ≥ 0.632, actual ratio reported |
| $n = 1$ special case (§4.2) | primal-dual within ~1% of the LP optimum |

**Rigor:** fixed seeds; held out by time; results across ≥ 3 seeds
(`--seed 0 1 2`, report mean ± spread); data provenance stated; every
difference from the paper explained.

**One extension** from §11, written up.

**Write-up** (1–2 pages): what matched, what didn't, why, and the limits of
simulated data. A failed check with a good explanation counts.

---

## Appendix A: glossary

- **Widget / carousel / row / content component**: one horizontal list of titles.
- **Category**: (content type, purchase option), e.g., (movie, rent or buy).
- **SVOD / TVOD / 3P**: subscription / transactional (rent or buy) / third-party channel.
- **Impression**: a row was shown on a rendered page.
- **Modular function**: a set function that is a plain sum over items.
- **Submodular**: diminishing returns (§3.2).
- **Lagrange multiplier / dual variable / price / constraint value**: $\xi_k$, the bonus per row of category $k$.
- **Projection**: clipping back into the feasible set, here $\xi \ge 0$.
- **Slotting / pinning**: reserving positions for categories on every page.
- **miss@n**: average relative shortfall from impression targets, Eq. (10).
- **DSD**: distinct streaming days.

## Appendix B: answers to the self-checks

**§1.** (1) The single highest-CTR category dominates every slot because each
row is scored without regard to the rows above. (2) No: with 15 SVOD users at
2 SVOD rows each you'd have 30 SVOD and 10 TVOD impressions; some SVOD users
must take TVOD rows, and the dual prices pick the ones for whom it costs
least. (3) It must know (or predict) everyone's traffic; individual pages can
deviate, and the targets hold only on average.

**§2.** (1) $(2, 0, 1, 1)$. (2) The constraint becomes a sum of per-page terms,
so it can be priced per page; a non-separable example is "the *median* user
sees at least one rental." (3) About $(100 \cdot 99 \cdots 91)^{10^6}$, i.e.,
astronomically many; decoupling is essential.

**§3.** (1) Adding $x$ raises the max by $\max(0, v_x - \max_A)$, and
$\max_A \le \max_B$ when $A \subseteq B$, so the gain can only shrink.
(2) It would push rentals onto the 45% who never rent as hard as onto renters.
(3) $\lambda = \beta_0^2/\sigma_0^2 = 4$. (4) On weekends, users value a
*first* channel row more (diversifying into channels pays off more).

**§4.** (1) No, page counts are integers. Fractional slotting pins one row on
every page and a second row on 80% of pages. (2) It's feasible by construction
(production already achieves it), and it changes the question to "can we keep
the same balance and serve each user better?"

**§5.** (1) At 0.6 both users switch: 1.0 rentals per page, constraint met
but over-satisfied, cost 0.6 instead of 0.1. Not cheapest. (2) Without it, an
over-satisfied category would get a negative bonus and be pushed below where
pure relevance would put it; for a maximum constraint the sign of the
shortfall flips. (3) Its shortfall $b_k - \bar I_k = -\bar I_k \le 0$ and the
projection clips at 0. (4) Traffic mix shifts (weekends, holidays, a big
release); targets $\mathbf{b}$ change; the model is retrained and its score
scale moves.

**§6.** (1) Values $v(\{a\}) = 3$, $v(\{b\}) = v(\{c\}) = 2$, $v(\{b, c\}) = 10$,
pairs with $a$ = 4, choose 2: greedy takes $a$ first and gets 4, optimal is
10. (2) Bonuses are per category; widgets inherit their category's bonus.
(3) $O(n \cdot N \cdot W)$ time, $O(N \cdot W)$ memory.

## Appendix C: bugs you will probably hit

- **Greedy places nothing / repeats a widget.** Initialize to $-\infty$ and
  mask used widgets *before* `argmax`.
- **Duals explode or stall.** Check the update's sign (shortfall is
  $b - \bar I$, not the reverse), the projection, and the step size (§5.5).
- **Features don't match the scorer.** Check the outer-product flattening order:
  index `j * m + k`.
- **Shapes.** `page_feature_matrix` stacks rows user-major; `y.reshape(-1)` must
  match.
- **miss@n is NaN.** You divided by a zero target; skip unconstrained categories.
- **Everything passes but results differ from `results/reference/`.**
  Check that you used `np.linalg.solve` with the same prior, and the same seed.
