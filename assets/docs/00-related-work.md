# 0. Related work and positioning

The method abbreviations used in this chapter are machine learning (**ML**), symbolic
regression (**SR**), explainable boosting machine (**EBM**), sparse identification of
nonlinear dynamics (**SINDy**), Transformation-Interaction-Rational (**TIR**), genetic
programming (**GP**), additive main effects and multiplicative interaction (**AMMI**),
Sparse Regression of Turbulent Stress Anisotropy (**SpaRTA**), and data, optimization,
model, and evaluation (**DOME**). The metric, protocol, and domain abbreviations are Matthews
Correlation Coefficient (**MCC**), coefficient of determination (**R²**), average precision
(**AP**), mean reciprocal rank (**MRR**), hit at rank one (**Hit@1**), F1 score (**F1**),
in-sample (**IS**), leave-one-dataset-out (**LODO**), leave-one-model-out (**LOMO**),
doubly held out (**DHO**), and Internet of Things (**IoT**).
Model-family abbreviations are random forest (**RF**), gradient-boosting machine (**GBM**),
and neural network (**NN**).

The equation labels are **E1** (dataset features only), **E3-Valid** (the plateau-selected
equation using dataset and model features), and **E3-MAX** (the maximum-capability equation
using those feature groups). The updated corpus contains 484 observed pairs across
20 network-oriented datasets and 25 classifiers. E3-Valid has 15 terms; E3-MAX has
24 terms, with both using the retained arity-three grammar.

Notes gathered while building `ml-meta-perf`, organised around the question the project
actually faces: *why choose a short additive equation over an accurate opaque regressor,
and what should be expected of it?*

## 1. The problem: algorithm selection as a learning problem

Predicting how well an algorithm will perform on a dataset it has not seen is the
**algorithm selection problem**, framed by Rice (1976) as a mapping from a feature space
of problem instances to a performance space of algorithms. The meta-learning literature
operationalises it by describing each dataset with *meta-features* — statistical,
information-theoretic and complexity measures — and learning a regressor from those to a
performance metric. The features in the shipped corpus (`class_ent`, `gravity`,
`ns_ratio`, `nr_cor_attr`, `eq_num_attr`, …) are standard meta-features of this kind.

- Brazdil, Giraud-Carrier, Soares, Vilalta, *Metalearning: Applications to Automated
  Machine Learning and Data Mining* (2nd ed., 2022) — the reference text for the
  meta-feature → performance framing.
- Rice, "The Algorithm Selection Problem", *Advances in Computers* 15 (1976).
- Vanschoren, "Meta-Learning", in Hutter, Kotthoff & Vanschoren (eds), *Automated Machine
  Learning: Methods, Systems, Challenges*, Springer (2019), ch. 2.

**Relevance.** This project sits squarely in that tradition but inverts the usual
priority: the deliverable is the *mapping itself*, in readable form, rather than its
accuracy.

## 2. The finding this project has to confront

> Billa, Orlandi, Guidetti, Mandreoli, **"Interpretable ML Under the Microscope:
> Performance, Meta-Features, and the Regression-Classification Predictability Gap"**,
> arXiv:2601.00428 (2026). Preprint. Sixteen interpretable methods across 216 tabular
> datasets.

Their central result is directly relevant and worth quoting in full:

> "in regression tasks, models exhibit a predictable performance hierarchy dominated by
> EBMs and SR that can be inferred from dataset characteristics. In contrast,
> **classification performance remains highly dataset-dependent with no stable
> hierarchy**, showing that standard complexity measures fail to provide actionable
> guidance."

**Relevance.** These findings motivate grouped evaluation of classification-performance
prediction rather than an assumed universal learner hierarchy. Current fixed-form scores
and the accuracy-versus-length comparison are reported in [chapter 4](04-equation.md)
and [chapter 5](05-evaluation.md); they remain conditional on same-corpus form and
configuration selection. The paper also discusses the computational cost of structural
sparsity, a separate concern from prediction error.

## 3. Sparse regression over a term library (the method)

`ml-meta-perf` is, structurally, **sparse regression over a fixed library of candidate
functions** — the same machinery as SINDy, applied to meta-learning rather than dynamics.

- Brunton, Proctor, Kutz, "Discovering governing equations from data by sparse
  identification of nonlinear dynamical systems", *Proceedings of the National Academy of
  Sciences (PNAS)* 113(15) (2016). The original
  library-plus-sparse-regression formulation.
- Schmelzer, Dwight, Cinnella, **"Discovery of Algebraic Reynolds-Stress Models Using
  Sparse Symbolic Regression"** (SpaRTA), *Flow, Turbulence and Combustion* 104, 579-603
  (2020). Deterministic sparse regression over a candidate library, chosen explicitly over
  genetic programming for the interpretability of the result. The closest methodological
  ancestor of this work.
- Golden, "Scalable Sparse Regression for Model Discovery: The Fast Lane to Insight",
  arXiv:2405.09579 (2024). Preprint.

**Relevance.** These justify the deterministic-library approach against the GP
alternative. SpaRTA's motivation — that GP produces expressions too unwieldy to interpret
even when accurate — is precisely the complaint that motivated this project.

## 4. Why not genetic programming (the previous approach)

The prior attempt here used genetic evolution and produced expressions too long to
interpret. Expression growth is a known concern in that approach.

- **Bloat.** Uncontrolled growth of expression size without fitness gain is the
  best-studied pathology in GP. Parsimony pressure and multi-objective (accuracy vs size)
  formulations are the standard mitigations.
- de França, "Alleviating Overfitting in Transformation-Interaction-Rational Symbolic
  Regression with Multi-Objective Optimization", *Genetic Programming and Evolvable
  Machines* 24(2) (2023). The **Transformation-Interaction-Rational (TIR)** representation
  constrains SR to a ratio of two nonlinear functions, each a linear regression over
  transformed variables — a deliberate restriction of the search space to bias toward
  simpler expressions.
- Cranmer, "Interpretable Machine Learning for Science with PySR and SymbolicRegression.jl",
  arXiv:2305.01582 (2023). Preprint.
- de França et al., "Call for Action: towards the next generation of symbolic regression
  benchmark" (SRBench update), *Genetic and Evolutionary Computation Conference (GECCO)
  2025 Companion*.
- Virgolin & Bosman, "Coefficient Mutation in the Gene-pool Optimal Mixing Evolutionary
  Algorithm for Symbolic Regression", *GECCO 2022 Companion* — GP struggles to optimise
  real-valued coefficients, which linear-in-the-weights methods get exactly and for free.

**Relevance.** TIR is the strongest argument for our design: restricting the *form* up
front (additive, linear in the weights, over a curated term vocabulary) is an established
route to interpretable SR, not a naive simplification. And because `ml-meta-perf` is linear in
its weights, the coefficients are solved exactly by ridge regression — sidestepping the
coefficient-optimisation weakness that GP has to work around.

**A caution on apparent fit.** Dataset meta-features are constant across a dataset's
rows, so random row splits can share dataset identity between fitting and evaluation.
Grouped protocols distinguish dataset and classifier information availability; they do
not independently validate a form selected using the complete corpus. The current
identity references and mixed-term diagnostics are reported in [chapter 4](04-equation.md).

## 5. Performance-influence models — the closest methodological sibling

Searching the systems-performance literature around the International Symposium on Modeling,
Analysis, and Simulation of Computer and Telecommunication Systems (**MASCOTS**), the
Association for Computing Machinery (**ACM**) SIGMETRICS International Conference on
Measurement and Modeling of Computer Systems (**SIGMETRICS**), and the European Software
Engineering Conference and Symposium on the Foundations of Software Engineering
(**ESEC/FSE**) turns up a line of work building models of **exactly this form**, for a
structurally identical problem.

- Siegmund, Grebhahn, Apel, Kästner, "Performance-influence models for highly
  configurable software systems", ESEC/FSE 2015 — the canonical reference. A
  performance-influence model is
  $\Pi = \beta_0 + \sum_i \beta_i o_i + \sum_{i,j} \beta_{ij} o_i o_j$:
  linear in its weights, over terms that are options and pairwise option *interactions*,
  fitted by stepwise forward/backward selection. That is `ml-meta-perf`'s model class and
  `ml-meta-perf`'s search strategy, arrived at independently for a different domain.
- Velez et al., "White-Box Analysis over Machine Learning: Modeling Performance of
  Configurable Systems", *International Conference on Software Engineering (ICSE) 2021*,
  1072-1084.
- Velez et al., "ConfigCrusher: Towards White-Box Performance Analysis for Configurable
  Systems", *Automated Software Engineering* 27 (2020).
- Jamshidi et al., "Transfer Learning for Performance Modeling of Configurable Systems:
  An Exploratory Analysis", *International Conference on Automated Software Engineering
  (ASE) 2017*, 497-508.
- Lesoil et al., "The Interaction between Inputs and Configurations fed to Software
  Systems: an Empirical Study", arXiv:2112.07279 (2021). Preprint.

**Why the analogy is tight.** Their configuration options map to our features, their
software system to our classifier, and their *workload* to our dataset. Lesoil et al. study
input×configuration interaction, which is our dataset×model interaction under another name.
Jamshidi et al. transfer a performance model across environments, which is our
LODO question.

**What this community already knows that is worth borrowing.** Interaction terms are the
standard remedy and are included by default rather than discovered, which is consistent
with this study’s inclusion of mixed dataset×model terms. Their sampling
literature exists because *which* configurations you measure dominates model quality — the
analogue here is which (dataset, model) pairs get run. The updated corpus observes
484 of 500 possible pairs; the 16 missing cells follow the declared eligibility rule.

**Where we differ.** They report accuracy on held-out configurations of the *same* system;
our LODO protocol holds out a dataset, corresponding to a new workload in this analogy,
while LOMO excludes a classifier. These information settings require separate comparisons.

## 5b. Two-way tables with covariates on one side — the model behind the ceiling

[Chapter 4](04-equation.md#the-ceiling-on-model-descriptors) measures a ceiling by adding, to the fitted equation, a table of one level and
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
- Efron & Morris, "Data analysis using Stein's estimator and its generalizations", *Journal
  of the American Statistical Association (JASA)* 70 (1975) — the shrinkage applied to both
  the levels and the slopes.
- Hastie & Tibshirani, *Generalized Additive Models* (1990) — backfitting as an
  alternative fitting scheme.

**Relevance.** The agronomy literature already cited for AMMI ([chapter 4](04-equation.md))
answers the question AMMI raises. AMMI's latents are free on both margins, so it explains a
grid and predicts nothing outside it; factorial regression with covariates on one margin is
the predictive version, and it is exactly what LODO permits — the datasets
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
  LODO protocol.
- Fusi, Sheth, Elibol, "Probabilistic Matrix Factorization for Automated Machine Learning",
  *NeurIPS 2018*, 3352-3361.
- Yang, Akimoto, Kim, Udell, "Oboe: Collaborative Filtering for AutoML Model Selection",
  *Conference on Knowledge Discovery and Data Mining (KDD) 2019*.

**Relevance.** These establish latent-factor models over a pipeline-by-dataset matrix
as an algorithm-recommendation approach. Their per-model effects use information that
is available under LODO but unavailable for a completely excluded classifier. The
current descriptor and identity-reference comparisons are reported in
[chapter 4](04-equation.md#the-ceiling-on-model-descriptors).

## 5d. Ranking objectives

- Herbrich, Graepel, Obermayer, "Large margin rank boundaries for ordinal regression",
  *Advances in Large Margin Classifiers* (2000); Joachims, "Optimizing search engines using
  clickthrough data", KDD 2002 — pairwise learning-to-rank.
- Brazdil & Soares, "A comparison of ranking methods for classification algorithm
  selection", European Conference on Machine Learning (**ECML**) 2000 — ranking as *the*
  output of algorithm selection, and the
  average-ranking baseline that our per-model mean reproduces.
- Mundlak, "On the pooling of time series and cross section data", *Econometrica* 46 (1978)
  — the within (fixed-effects) transform, which is what makes a pairwise-ranking least
  squares objective closed-form.

**Relevance.** This study evaluates the MCC equation as a model selector with AP, MRR,
Hit@1, regret and Spearman correlation in [chapter 5](05-evaluation.md). The equation is
still fitted by squared error; a separate pairwise ranking objective is not part of the
current implementation.

## 6. The target metric

- Chicco & Jurman, "The advantages of the Matthews correlation coefficient (MCC) over F1
  score and accuracy in binary classification evaluation", *BMC Genomics* 21 (2020).
- Itaya et al., "Statistical Inference of the Matthews Correlation Coefficient for
  Multiclass Classification", arXiv:2503.06450 (2025). Preprint.

**Relevance.** MCC summarizes classification performance while accounting for all
confusion-matrix cells. Its bounded range [-1, 1] motivates the documented prediction
bounds. Endpoint observations also require care with target transformations such as
`atanh`, which is undefined at exactly -1 or 1.

## 7. Validation protocol

The grouped-split requirement is standard practice wherever records share a group
identity, and is the same concern as subject-wise splitting in clinical ML.

- Walsh et al., "DOME: recommendations for supervised machine learning validation in
  biology", *Nature Methods* 18, 1122-1127 (2021) — community standards on how validation should be
  reported, including the leakage traps.

**Relevance.** Supports grouped protocols matched to the prediction setting: LODO,
LOMO, and joint dataset/classifier exclusion under DHO. Random k-fold remains a
diagnostic for shared identities.

## 8. Where this work is positioned

| | prior work | `ml-meta-perf` |
|---|---|---|
| Model class | opaque regressors (RF, GBM, NN); or GP-evolved long expressions | fixed additive form, linear in the weights |
| Empirical comparison | study-specific targets, corpora, and splits | current fixed-form scores and matched comparisons in chapters 4–5 |
| Validation | depends on each study’s prediction setting | IS, LODO, LOMO, and DHO with explicit selection dependence |
| Extractable guidance | little | each weight reads directly in feature units |
| Identity references | study-dependent | dataset, classifier, additive, and interaction references in chapter 4 |
| Information availability | study-dependent | classifier-history effects distinguished from descriptor-only transfer |

The contribution comprises an inspectable equation, an explicit accuracy-versus-length
curve, feature-block comparisons of dataset and learner descriptors, and grouped
coefficient-refitting protocols whose selection dependence is stated explicitly.

## 9. Open threads

- **Interaction-aware but interpretable terms.** The performance-influence literature
  (§5) motivates mixed terms. The retained arity-three vocabulary includes symmetric
  `sum_ratio` expressions; richer interaction representations and measured learner
  descriptors remain possible extensions.
- **Ranking.** Historical classifier means and medians are available under LODO;
  descriptor-based prediction remains usable when the classifier itself is excluded.
  Ranking comparisons require matched information availability, as in chapter 5.
- **More datasets.** Twenty dataset groups limit the independent dataset-level evidence.
  Broader coverage would assess whether the observed relationships persist beyond the
  current network-oriented corpus.
- **Interaction structure.** Compare the fitted equation with the current additive and
  interaction references in chapter 4, while distinguishing explanatory references
  from predictors that can transfer to unseen groups.
