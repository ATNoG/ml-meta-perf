# 7. Other targets: the method beyond MCC

*Implemented in `ml_meta_perf.model`, `ml_meta_perf.validate` and `ml_meta_perf.data`.*

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

$B$ is a property of the target and is stored with the equation:

| target | $B$ |
|---|---|
| MCC | $[-1, 1]$ |
| FlexFL performance, classification (MCC) | $[-1, 1]$ |
| FlexFL performance, regression (SMAPE) | $[0, 2]$ |
| a cost: time, bytes, energy, counts | none, written $B = \varnothing$ |

For $B = \varnothing$ the prediction is returned as written: $\hat{y} = \hat{y}_{\text{raw}}$.
A cost has no upper limit and a clip at zero would hide a fit that goes negative, which is a
fault worth seeing.

## 7.2 Cross-validation and the bounds

Under a leave-one-group-out protocol (chapter 5) each fold $g$ refits the weights on the rows
outside the group, $\mathcal{T}_g$, and predicts the rows inside it, $\mathcal{H}_g$. For a
**bounded** target the held-out prediction is further limited to what that fold was trained on:

$$\hat{y}_i^{\text{cv}} = \operatorname{clip}_{[\,m_g,\; M_g\,]}\!\big(\hat{y}_{\text{raw},i}\big),
\qquad m_g = \min_{j \in \mathcal{T}_g} y_j, \quad M_g = \max_{j \in \mathcal{T}_g} y_j, \quad i \in \mathcal{H}_g .$$

This is tighter than $B$ and uses no held-out information: the interval $[m_g, M_g]$ is what the
fold's own equation was shown. It exists because a single fold predicted at the theoretical
floor once accounted for 89% of the total squared error of the MCC corpus.

For an **unbounded** target ($B = \varnothing$) the fold range is not applied:

$$\hat{y}_i^{\text{cv}} = \hat{y}_{\text{raw},i}.$$

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
