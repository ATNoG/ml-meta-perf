# 7. Other targets: the method beyond MCC

*Implemented in `ml_meta_perf.config`, `ml_meta_perf.model`, `ml_meta_perf.validate` and `ml_meta_perf.data`.*

The method of chapters 2 to 5 does not depend on the target being MCC. This chapter states
what changes when the target is something else (training time, bytes sent, energy, a
regression error) and what does not. It is written to grow with the tool: each section gives
the mathematics of one rule the code applies, so that a reader can check the rule rather than
trust it.

Throughout, $y$ is the target, $\mathbf{f}$ the features of one row, $t_1,\dots,t_k$ the terms
of the published equation and $w_0,\dots,w_k$ its weights (chapter 2).

## 7.1 The prediction and its bounds

The equation is linear in its weights,

$$\hat{y}_{\text{raw}}(\mathbf{f}) = w_0 + \sum_{i=1}^{k} w_i \, t_i(\mathbf{f}),$$

and the published prediction applies the *bounds of the target*, $B$:

$$\hat{y}(\mathbf{f}) = \operatorname{clip}_B\!\big(\hat{y}_{\text{raw}}(\mathbf{f})\big), \qquad
\operatorname{clip}_{[a,b]}(z) = \min\!\big(\max(z, a), b\big).$$

$B = [a, b]$ is a property of the target and is stored with the equation. Either end may be open
(written `null` in the study file): $a = -\infty$ or $b = +\infty$, so that

$$\operatorname{clip}_{[a,\infty)}(z) = \max(z, a), \qquad \operatorname{clip}_{(-\infty,b]}(z) = \min(z, b),$$

and a target open on both sides has $B = \varnothing$ and is never clipped. The target dictates its
bounds:

| target | $B$ |
|---|---|
| MCC | $[-1, 1]$ |
| FlexFL performance, classification (MCC) | $[-1, 1]$ |
| FlexFL performance, regression (SMAPE) | $[0, 2]$ |
| a time, a byte count, an epoch count: cannot be negative, has no ceiling | $[0, \infty)$ |
| a quantity with neither limit | $\varnothing$ |

A prediction for a cost that goes below zero is a fault of the fit and is floored at zero rather
than reported; one that goes above anything seen is the extrapolation the equation exists to make
and is never capped.

## 7.2 Cross-validation and the bounds

Under a leave-one-group-out protocol (chapter 5) each fold $g$ refits the weights on the rows
outside the group, $\mathcal{T}_g$, and predicts the rows inside it, $\mathcal{H}_g$. For a
**bounded** target the held-out prediction is further limited to what that fold was trained on:

$$\hat{y}_i^{\text{cv}} = \operatorname{clip}_{[\,m_g,\; M_g\,]}\!\big(\hat{y}_{\text{raw},i}\big),
\qquad m_g = \min_{j \in \mathcal{T}_g} y_j, \quad M_g = \max_{j \in \mathcal{T}_g} y_j, \quad i \in \mathcal{H}_g .$$

This is tighter than $B$ and uses no held-out information: the interval $[m_g, M_g]$ is what the
fold's own equation was shown. It exists because a single fold predicted at the theoretical
floor once accounted for 89% of the total squared error of the MCC corpus.

The fold range is applied only to a target bounded on **both** sides. For any other target the
prediction is clipped at the bounds it has and nowhere else,

$$\hat{y}_i^{\text{cv}} = \operatorname{clip}_B\!\big(\hat{y}_{\text{raw},i}\big)
\quad (B \text{ open on at least one side}), \qquad
\hat{y}_i^{\text{cv}} = \hat{y}_{\text{raw},i} \quad (B = \varnothing).$$

Clipping to $[m_g, M_g]$ would forbid extrapolation. If the costliest dataset is held out,
$y_i > M_g$ for every one of its rows, so a clipped prediction has error at least
$y_i - M_g > 0$ whatever the equation says: the score would measure the clip and not the
equation. The same rule applies to the doubly held out protocol, where the training set
$\mathcal{T}_{g,h}$ excludes both the dataset and the model of the held-out cell.

The number of predictions the clip moved is reported with every protocol
(`clipped_predictions`); for an unbounded target it is zero by construction.

## 7.3 Discrete and binary features are not independent features here

A feature that takes only the values 0 and 1 (an indicator, such as "the run was
asynchronous") behaves differently from a continuous one, for three reasons that follow from
the form of the equation.

1. **Inside a product it switches the term off.** For an indicator $d \in \{0,1\}$ and any
   features $a, b$, the term $d\,a\,b$ equals $a\,b$ when $d = 1$ and $0$ when $d = 0$. The weight
   is then a slope that exists only on the rows where the indicator is on: it describes one
   group, not a relationship that holds across the corpus.
2. **In a sum it is only an offset.** The term $w\,d$ shifts the prediction by $w$ on the rows
   where $d = 1$. Every transform the grammar offers collapses: $d^2 = d$ and $\sqrt{d} = d$, so
   $d$, $d^2$ and $\sqrt{d}$ are the same column, and $\log d$ and $1/d$ are undefined at 0.
3. **A complete set of indicators is collinear with the intercept.** If $d_1,\dots,d_m$ encode one
   categorical factor, $\sum_{j=1}^{m} d_j = 1$ on every row, which is the intercept column. Only
   $m-1$ of them carry information, and which one is dropped changes the weights without changing
   the predictions.

Consequently such a feature may be included, but it is read as *"this group sits higher or lower
by $w$"* and never as a quantity whose effect scales with its value. When a corpus encodes a
categorical factor (for example the training mode of a federated run) the equation's terms over
those columns are offsets between modes, and the factor is a candidate for a protocol that
holds a whole mode out, not for a leave-one-model-out reading of it (see chapter 5).

## 7.4 What a study is made of

Everything the method needs to know about a problem is a *study description*, and the aim of
this tool is that nothing of it lives in the code. Formally, a corpus is a set of rows

$$\mathcal{D} = \{(d_i,\; m_i,\; \mathbf{u}_i,\; \mathbf{v}_i,\; y_i)\}_{i=1}^{n},$$

where $d_i$ names the dataset (the first grouping factor), $m_i$ names the model or
configuration (the second), $\mathbf{u}_i \in \mathbb{R}^{p}$ are the features of the dataset,
$\mathbf{v}_i \in \mathbb{R}^{q}$ are the features of the model, and $y_i$ is the target. A study
description is the tuple

$$\mathcal{S} = \big(c_d,\; c_m,\; F_d,\; F_m,\; c_y,\; B\big),$$

with $c_d, c_m$ the columns that hold the two group labels, $F_d, F_m$ the feature columns of
each kind, $c_y$ the target column and $B$ the bounds of section 7.1. The three equations of
the MCC study differ only in which features they may use:

$$\text{E1: } F_d, \qquad \text{E2: } F_m, \qquad \text{E3: } F_d \cup F_m .$$

A study with a single grouping factor (one NN family trained in several ways, as in FlexFL) has
no $c_m$; it fits E3 over its features and its protocols are the ones defined by the groups it
has (section 5).

$\mathcal{S}$ is read from three sections of `config/study.json`, and the facts that are not columns of the corpus come from a fourth:

| symbol | section and key |
|---|---|
| $c_d,\ c_m$ | `corpus.dataset_column`, `corpus.model_column` |
| $F_d,\ F_m$ | `features.dataset`, `features.model` |
| $c_y$ | `response.column` |
| $B$ | `response.bounds` (`null` for $B = \varnothing$) |

Static facts about the corpus that are not columns of it sit in the `side_files` section, each a
JSON file beside the CSV: a taxonomy $\varphi : M \to \Phi$ giving the learner family of every
model $m \in M$ (and the families grouped as tree-based or neural), the assigned ordinal
$o_k : M \text{ or } \Phi \to \mathbb{N}$ behind each model-descriptor column $k$ with its
provenance, and a glossary mapping each feature to its meaning. A study without one says `null`,
and anything that needs it fails with a message naming the missing file instead of guessing.
Totality is checked: $\varphi$ is defined on every model that occurs in the corpus, the glossary
on every feature, and the tree and neural groups are disjoint (`tests/test_data.py`).

`StudyConfig.schema()` turns the first three into the `Schema` object the loader and the fit consume, and
`default_schema()` returns the one `config/study.json` describes. The file is checked when it is
read, before any data is: the two group columns must differ, the bounds must satisfy $a < b$,
each feature list must be non-empty, no column may be listed twice or play two roles
($F_d \cap F_m = \varnothing$, and neither contains $c_d$, $c_m$ or $c_y$). No part of
$\mathcal{S}$ is a constant in the code: the MCC columns and feature lists that used to be
module constants of `ml_meta_perf.data` are gone, and `tests/test_data_surface.py` pins what the
shipped file must say for the MCC study to be the study the chapters describe.

### Where the corpus is

`corpus.path` is resolved once, by the rule

$$\operatorname{resolve}(p) = \begin{cases}
p & p \text{ absolute},\\
r/p & p \text{ relative and } r/p \text{ is a file},\\
w/p & p \text{ relative, } r/p \text{ is not a file and } w/p \text{ is},\\
r/p & \text{otherwise (so the error names where the file was expected)},
\end{cases}$$

where $r$ is the repository root of the running source and $w$ the working directory. The root
comes first so that an editable install and `PYTHONPATH=src` find the shipped corpus; the working
directory covers a regular install run from a clone. A path given on the command line with
`--data` is not resolved: it is used as written.

## 7.5 Fitting a cost on a log scale

A cost (a time, bytes sent, energy) spans orders of magnitude and grows multiplicatively: doubling
the data roughly doubles the time. An additive equation fitted by least squares on the raw cost
spends nearly all of its error budget on the few largest runs. `response.transform = "log1p"` fits
the transformed target instead,

$$z = \log(1 + y), \qquad \hat{z} = w_0 + \sum_{i=1}^{k} w_i\, t_i(\mathbf{f}), \qquad
\hat{y} = e^{\hat{z}} - 1 = e^{w_0} \prod_{i=1}^{k} e^{\,w_i t_i(\mathbf{f})} - 1 .$$

The equation is still a sum and is still read term by term, but on the log scale: every term
*multiplies* the prediction by $e^{w_i t_i}$. With the grammar's $\log$ transform a term
$w \log x$ contributes the factor $x^{w}$, so for a cost that follows a power law
$y = c\, x_1^{a} x_2^{b}$ the fit recovers the exponents $a$ and $b$ as weights, because for
$y \gg 1$

$$\log(1 + y) = \log y + \log\!\Big(1 + \tfrac{1}{y}\Big) \approx \log c + a \log x_1 + b \log x_2 .$$

A transform needs a target it is defined on. `log1p` requires $y \ge 0$: the study file must say
so, with $B = \varnothing$ or a low bound $a \ge 0$, and the data is checked as it is read. A
target that can be negative, such as MCC, has no log scale. The bounds are carried to the scale of
the fit: the equation predicts $\hat{z}$, so it is clipped at $\log(1 + a)$ and $\log(1 + b)$ for
$B = [a, b]$, and a time with $B = [0, \infty)$ is floored at $z = 0$, which is $y = 0$. $R^2$ and MAE of a transformed
target are on the scale of $z$, which is also the scale the equation was fitted on; a mean absolute
error on $z$ is approximately a relative error on $y$, since $\log(1+y) - \log(1+\hat{y}) \approx
\log(y/\hat{y})$ for large values.
