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
leave-one-dataset-out R² of ~0.47 for a classification metric (MCC), against an in-sample
0.614, is not a failure of the method — it reflects a documented property of classifier
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

**Why the analogy is tight.** Their configuration options map to our features, their
software system to our classifier, and their *workload* to our dataset. Lesoil et al. study
input×configuration interaction, which is our dataset×model interaction under another name.
Jamshidi et al. transfer a performance model across environments, which is our
leave-one-dataset-out question.

**What this community already knows that is worth borrowing.** Interaction terms are the
standard remedy and are included by default rather than discovered, which is consistent
with our finding that mixed dataset×model terms carry E3's entire lift. Their sampling
literature exists because *which* configurations you measure dominates model quality — the
analogue here is which (dataset, model) pairs get run, and our 476-of-500 grid is nearly
complete, so we do not have their hardest problem.

**Where we differ.** They report accuracy on held-out configurations of the *same* system;
our leave-one-dataset-out protocol holds out a whole system. That is the harder question
and explains part of the gap between their reported accuracies and ours.

## 5b. Two-way tables with covariates on one side — the model behind the ceiling

Chapter 9 measures a ceiling by adding, to the fitted equation, a table of one level and
one slope per classifier. That construction is not new; it is the standard model for a
two-way table where one margin can be described by covariates and the other cannot. It is
**not published as a model here** — it is how the study puts a number on what better model
descriptors would be worth.

- Denis, "Two-way analysis using covariates", *Statistics* 19(2) (1988) — **factorial
  regression**: a two-way table modelled with covariates on the rows, the columns, or both,
  with free coefficients wherever covariates are unavailable.
- van Eeuwijk, Denis, Kang, "Incorporating additional information on genotypes and
  environments in models for two-way genotype by environment tables", in *Genotype-by-
  Environment Interaction* (1996). The direct ancestor: AMMI's free interaction latents
  replaced, on one side only, by a regression on measured covariates.
- Finlay & Wilkinson, "The analysis of adaptation in a plant-breeding programme",
  *Australian Journal of Agricultural Research* 14 (1963) — the earliest form of the same
  idea: each subject gets its own *slope* on an index of the condition.
- Efron & Morris, "Data analysis using Stein's estimator and its generalizations", *JASA*
  70 (1975) — the shrinkage applied to both the levels and the slopes.
- Hastie & Tibshirani, *Generalized Additive Models* (1990) — backfitting, the alternative
  fitting scheme, which was measured here and is worse.

**Relevance.** The agronomy literature already cited for AMMI ([chapter 5](assets/docs/05-oracles.md))
answers the question AMMI raises. AMMI's latents are free on both margins, so it explains a
grid and predicts nothing outside it; factorial regression with covariates on one margin is
the predictive version, and it is exactly what leave-one-dataset-out permits — the datasets
are new, the classifiers are not. Using it as a *ceiling* rather than as a result is the
honest reading: it says what the meta-features fail to capture, in the units the study
reports.

## 5c. Algorithm selection as collaborative filtering

The per-model table is an effect learned from the observed (dataset x model) grid, which is
the collaborative half of a hybrid recommender. That framing has an established literature
in AutoML.

- Mısır & Sebag, "Alors: An algorithm recommender system", *Artificial Intelligence* 244
  (2017). Collaborative filtering over an algorithm-by-instance performance matrix, with
  meta-features used to place a *new* instance — the cold-start case, which is our
  leave-one-dataset-out protocol.
- Fusi, Sheth, Elibol, "Probabilistic Matrix Factorization for Automated Machine Learning",
  arXiv:1705.05355 (NeurIPS 2018).
- Yang, Akimoto, Kim, Udell, "OBOE: Collaborative Filtering for AutoML Model Selection",
  arXiv:1808.03233 (KDD 2019).

**Relevance.** These establish that latent-factor models over a pipeline-by-dataset matrix
are standard practice for algorithm recommendation, and they are why the +0.106 measured in
[chapter 9](assets/docs/09-model-effects.md) is unsurprising in size. They are also what
this study deliberately does *not* deliver: a latent factor per model is an uninterpreted
coordinate, and a table of them supports no term analysis and no transferable practice.
Reporting the number as a ceiling states the trade honestly — this is what an interpretable
additive equation gives up against a factorised recommender on this corpus, and it is
0.106 of R², not the order of magnitude a reader might assume.

## 5d. Ranking objectives

- Herbrich, Graepel, Obermayer, "Large margin rank boundaries for ordinal regression",
  *Advances in Large Margin Classifiers* (2000); Joachims, "Optimizing search engines using
  clickthrough data", KDD 2002 — pairwise learning-to-rank.
- Brazdil & Soares, "A comparison of ranking methods for classification algorithm
  selection", ECML 2000 — ranking as *the* output of algorithm selection, and the
  average-ranking baseline that our per-model mean reproduces.
- Mundlak, "On the pooling of time series and cross section data", *Econometrica* 46 (1978)
  — the within (fixed-effects) transform, which is what makes a pairwise-ranking least
  squares objective closed-form.

**Relevance.** The obvious response to "the per-model mean out-ranks E3" is to optimise the
ranking directly, and the within transform makes that a one-line change rather than a new
optimiser. It was implemented and it **ranks worse** (Spearman 0.532 against 0.625). The
result is worth reporting precisely because the literature makes it look like free money:
the binding constraint here is the thinness of the model descriptors, not the loss.

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
| Reported R² | ~0.9 (opaque), >0.7 (GP) | 0.614 in-sample, 0.478 LOO-dataset |
| Validation | often random k-fold | leave-one-dataset-out and leave-one-model-out |
| Extractable guidance | little | each weight reads directly in feature units |
| Ceiling stated | rarely | additive oracle at 0.6605, rank-1 at 0.783, E1 capped at 0.354 |
| Reachable ceiling | not distinguished | +0.018 of the rank-1 rung from covariates; +0.106 is what perfect model descriptors would still buy |

The contribution is not a higher number. It is (a) an equation that can be read, (b) an
explicit accuracy-versus-length curve instead of a single operating point, (c) the
E1/E3 contrast isolating how much of MCC is attributable to model capability rather than
dataset difficulty, and (d) honest grouped-split validation with the leakage gap
quantified.

## 9. Open threads

- **Interaction-aware but interpretable terms.** The oracle ladder puts a rank-1
  interaction at +0.122 R², and the performance-influence literature (§5) includes
  interaction terms by default. Our symmetric `sum_ratio` fix was a step in that
  direction and was worth most of the 0.558 → 0.603 improvement. **Mostly closed as a
  question** (§5b): only +0.018 of that rung is reachable with covariates on both margins,
  so what remains open is not the interaction basis but the model descriptors.
- **Ranking.** *Closed, but not by the obvious route* (§5d). A pairwise-ranking objective
  ranked worse than squared error (Spearman 0.532 against 0.625), leaving the per-model
  mean ahead at 0.703. What closed the gap was a better model descriptor, not a better
  loss: with `Model Capability` in the model side E3 reaches 0.706 with top-1 regret 0.008
  against the baseline's 0.011. The margin on rank correlation is a tie; the regret figure
  is the real gain. This is the diagnosis confirming itself — the baseline out-ranked the
  equation because it knew which models are generally good, and the remedy was to tell the
  equation.
- **More datasets.** Twenty is the binding constraint on every cross-validated number
  here; OpenML-scale meta-data would settle whether the 0.6605 additive ceiling is a
  property of this sample or of the approach.
- **Interaction structure.** The gap between E3 (0.614) and the rank-1 oracle (0.783)
  is entirely dataset×model interaction the current term vocabulary does not reach.
