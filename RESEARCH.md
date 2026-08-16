# Related work and positioning

Notes gathered while building `metafit`, organised around the question the project
actually faces: *why choose a short additive equation over an accurate opaque regressor,
and what should be expected of it?*

## 1. The problem: algorithm selection as a learning problem

Predicting how well an algorithm will perform on a dataset it has not seen is the
**algorithm selection problem**, framed by Rice (1976) as a mapping from a feature space
of problem instances to a performance space of algorithms. The meta-learning literature
operationalises it by describing each dataset with *meta-features* — statistical,
information-theoretic and complexity measures — and learning a regressor from those to a
performance metric. The features in `data/meta_dataset.csv` (`class_ent`, `gravity`,
`ns_ratio`, `nr_cor_attr`, `eq_num_attr`, …) are standard meta-features of this kind.

- Brazdil, Giraud-Carrier, Soares, Vilalta, *Metalearning: Applications to Automated
  Machine Learning and Data Mining* (2nd ed., 2022) — the reference text for the
  meta-feature → performance framing.
- Rice, "The Algorithm Selection Problem", *Advances in Computers* 15 (1976).
- Vanschoren, "Meta-Learning: A Survey", arXiv:1810.03548 (2018).

**Relevance.** This project sits squarely in that tradition but inverts the usual
priority: the deliverable is the *mapping itself*, in readable form, rather than its
accuracy.

## 2. The finding this project has to confront

> Billa, Orlandi, Guidetti, Mandreoli, **"Interpretable ML Under the Microscope:
> Performance, Meta-Features, and the Regression-Classification Predictability Gap"**,
> arXiv:2601.00428 (2026). Sixteen interpretable methods across 216 tabular datasets.

Their central result is directly relevant and worth quoting in full:

> "in regression tasks, models exhibit a predictable performance hierarchy dominated by
> EBMs and SR that can be inferred from dataset characteristics. In contrast,
> **classification performance remains highly dataset-dependent with no stable
> hierarchy**, showing that standard complexity measures fail to provide actionable
> guidance."

**Relevance.** This is independent corroboration of the central measurement here. Our
leave-one-dataset-out R² of ~0.46 for a classification metric (MCC), against an in-sample
0.603, is not a failure of the method — it reflects a documented property of classifier
performance prediction. The paper also names the **"interpretability tax"**: methods
optimising for structural sparsity pay significantly in training time. `metafit` pays a
different tax — accuracy — and quantifies it explicitly through the term-count curve.

This paper is the strongest single citation for framing our modest cross-validated
numbers as a finding rather than a shortfall.

## 3. Sparse regression over a term library (the method)

`metafit` is, structurally, **sparse regression over a fixed library of candidate
functions** — the same machinery as SINDy, applied to meta-learning rather than dynamics.

- Brunton, Proctor, Kutz, "Discovering governing equations from data by sparse
  identification of nonlinear dynamical systems", *PNAS* 113(15) (2016). The original
  library-plus-sparse-regression formulation.
- Schmelzer, Dwight, Cinnella, **"Discovery of Algebraic Reynolds-Stress Models Using
  Sparse Symbolic Regression"** (SpaRTA), arXiv:1905.07510 (2019). Deterministic sparse
  regression over a candidate library, chosen explicitly over genetic programming for
  the interpretability of the result. The closest methodological ancestor of this work.
- Kaptanoglu et al., "Scalable Sparse Regression for Model Discovery", arXiv:2405.09579
  (2024).

**Relevance.** These justify the deterministic-library approach against the GP
alternative. SpaRTA's motivation — that GP produces expressions too unwieldy to interpret
even when accurate — is precisely the complaint that motivated this project.

## 4. Why not genetic programming (the previous approach)

The prior attempt here used genetic evolution and reached R² > 0.7 with expressions too
long to interpret. That is a known and named failure mode.

- **Bloat.** Uncontrolled growth of expression size without fitness gain is the
  best-studied pathology in GP. Parsimony pressure and multi-objective (accuracy vs size)
  formulations are the standard mitigations.
- de França, "Alleviating Overfitting in Transformation-Interaction-Rational Symbolic
  Regression with Multi-Objective Optimization", arXiv:2501.01905 (2025). The
  **Transformation-Interaction-Rational (TIR)** representation constrains SR to a ratio
  of two nonlinear functions, each a linear regression over transformed variables —
  a deliberate restriction of the search space to bias toward simpler expressions.
- Cranmer, "Interpretable Machine Learning for Science with PySR", arXiv:2305.01582
  (2023).
- de França et al., "Call for Action: towards the next generation of symbolic regression
  benchmark" (SRBench update), arXiv:2505.03977 (2025).
- Virgolin et al., "Coefficient Mutation in GP-GOMEA for Symbolic Regression",
  arXiv:2204.12159 (2022) — GP struggles to optimise real-valued coefficients, which
  linear-in-the-weights methods get exactly and for free.

**Relevance.** TIR is the strongest argument for our design: restricting the *form* up
front (additive, linear in the weights, over a curated term vocabulary) is an established
route to interpretable SR, not a naive simplification. And because `metafit` is linear in
its weights, the coefficients are solved exactly by ridge regression — sidestepping the
coefficient-optimisation weakness that GP has to work around.

**A caution on the R² > 0.7 figure.** Our measurements suggest it is unlikely to survive a
grouped split. The identical equation scores far higher under random 10-fold than under
leave-one-dataset-out here. Since dataset meta-features are constant across a dataset's
rows, a random split lets any model memorise dataset identity. Additionally, the
**additive oracle sits at 0.6605**: with exact per-dataset and per-model effects, a
*two-way additive* model cannot exceed that. Our equation does pass it, because half its
terms are mixed dataset×model products which express interaction the two-way form cannot —
but only at lengths where cross-validated accuracy has already collapsed. A reported 0.7
therefore implies either genuine interaction structure or a leaky split, and the two are
distinguishable by the protocol.

## 5. Performance-influence models — the closest methodological sibling

Searching the systems-performance literature (the MASCOTS / SIGMETRICS / ESEC-FSE
neighbourhood) turns up a line of work building models of **exactly this form**, for a
structurally identical problem.

- Siegmund, Grebhahn, Apel, Kästner, "Performance-influence models for highly
  configurable software systems", ESEC/FSE 2015 — the canonical reference. A
  performance-influence model is
  $\Pi = \beta_0 + \sum_i \beta_i o_i + \sum_{i,j} \beta_{ij} o_i o_j$:
  linear in its weights, over terms that are options and pairwise option *interactions*,
  fitted by stepwise forward/backward selection. That is `metafit`'s model class and
  `metafit`'s search strategy, arrived at independently for a different domain.
- Velez et al., "White-Box Analysis over Machine Learning: Modeling Performance of
  Configurable Systems", arXiv:2101.05362 (2021).
- Velez et al., "ConfigCrusher: Towards White-Box Performance Analysis for Configurable
  Systems", arXiv:1905.02066 (2019).
- Jamshidi et al., "Transfer Learning for Performance Modeling of Configurable Systems",
  arXiv:1709.02280 (2017).
- Lesoil et al., "The Interaction between Inputs and Configurations fed to Software
  Systems", arXiv:2112.07279 (2021).
- Gong & Chen, "Predicting Software Performance with Divide-and-Learn", arXiv:2306.06651
  (2023).

**Why the analogy is tight.** Their configuration options map to our features, their
software system to our classifier, and their *workload* to our dataset. Lesoil et al. study
input×configuration interaction, which is our dataset×model interaction under another name.
Jamshidi et al. transfer a performance model across environments, which is our
leave-one-dataset-out question. Gong & Chen's "divide-and-learn" responds to sparsity by
partitioning samples into groups and fitting per group — our grouped structure exactly.

**What this community already knows that is worth borrowing.** Interaction terms are the
standard remedy and are included by default rather than discovered, which is consistent
with our finding that mixed dataset×model terms carry E2's entire lift. Their sampling
literature exists because *which* configurations you measure dominates model quality — the
analogue here is which (dataset, model) pairs get run, and our 476-of-500 grid is nearly
complete, so we do not have their hardest problem.

**Where we differ.** They report accuracy on held-out configurations of the *same* system;
our leave-one-dataset-out protocol holds out a whole system. That is the harder question
and explains part of the gap between their reported accuracies and ours.

## 6. The target metric

- Chicco & Jurman, "The advantages of the Matthews correlation coefficient (MCC) over F1
  score and accuracy in binary classification evaluation", *BMC Genomics* 21 (2020).
- Itaya et al., "Statistical Inference of the Matthews Correlation Coefficient for
  Multiclass Classification", arXiv:2503.06450 (2025).

**Relevance.** Justifies MCC as the target for imbalanced security/IoT datasets. It also
raises a modelling caveat we handle explicitly: MCC is bounded in [-1, 1] and this
meta-dataset piles 17% of its mass at exactly 1.0. A linear form is unaware of the bound,
so predictions are clipped. A logit/`atanh` transform of the target was tried and
**hurt badly** (cross-validated R² went negative) — the saturation point maps to infinity.

## 7. Validation protocol

The grouped-split requirement is standard practice wherever records share a group
identity, and is the same concern as subject-wise splitting in clinical ML.

- Walsh et al., "DOME: Recommendations for supervised machine learning validation in
  biology", arXiv:2006.16189 (2020) — community standards on how validation should be
  reported, including the leakage traps.

**Relevance.** Supports reporting leave-one-dataset-out as the headline and treating
random k-fold as a diagnostic for leakage rather than a result.

## 8. Where this work is positioned

| | prior work | `metafit` |
|---|---|---|
| Model class | opaque regressors (RF, GBM, NN); or GP-evolved long expressions | fixed additive form, linear in the weights |
| Reported R² | ~0.9 (opaque), >0.7 (GP) | 0.603 in-sample, 0.458 LOO-dataset |
| Validation | often random k-fold | leave-one-dataset-out and leave-one-model-out |
| Extractable guidance | little | each weight reads directly in feature units |
| Ceiling stated | rarely | additive oracle at 0.6605, rank-1 at 0.783, E1 capped at 0.354 |

The contribution is not a higher number. It is (a) an equation that can be read, (b) an
explicit accuracy-versus-length curve instead of a single operating point, (c) the
E1/E2 contrast isolating how much of MCC is attributable to model capability rather than
dataset difficulty, and (d) honest grouped-split validation with the leakage gap
quantified.

## 9. Open threads

- **Interaction-aware but interpretable terms.** The oracle ladder puts a rank-1
  interaction at +0.122 R², and the performance-influence literature (§5) includes
  interaction terms by default. Our symmetric `sum_ratio` fix was a step in that
  direction and was worth most of the 0.558 → 0.603 improvement; a principled
  interaction basis is the obvious continuation.
- **Ranking.** For selecting a model on a new dataset, the trivial per-model-mean
  baseline out-ranks E2 (Spearman 0.70 vs 0.61). A learning-to-rank objective rather
  than squared error would be the natural next step.
- **More datasets.** Twenty is the binding constraint on every cross-validated number
  here; OpenML-scale meta-data would settle whether the 0.6605 additive ceiling is a
  property of this sample or of the approach.
- **Interaction structure.** The gap between E2 (0.603) and the rank-1 oracle (0.783)
  is entirely dataset×model interaction the current term vocabulary does not reach.
