# AI-proposed model descriptors

This document brings together the model descriptors proposed by **ChatGPT**,
**Claude1**, and **Perplexity**. The formulas have been condensed from the
documentation and frozen artifacts stored in the repository.

General convention: unless stated otherwise, a value of `0` means that the
descriptor is not applicable or is not encoded for that model.

## 1. Comparative list of descriptors

Each column corresponds to one AI. An em dash means that the AI has no additional
descriptor in that position; items on the same row are not necessarily equivalent.

| ChatGPT (26) | Claude1 (7) | Perplexity (22) |
|---|---|---|
| `component_count` | `no_tunable_hp` | `reg_l2_log` |
| `sequential_depth` | `capacity` | `reg_l1_log` |
| `unbounded_depth` | `regularization` | `capacity_depth` |
| `representation_width` | `ensemble_size` | `capacity_width` |
| `structural_size_proxy` | `learning_rate` | `lr_log` |
| `train_work_proxy` | `stochasticity` | `iters_log` |
| `inference_work_proxy` | `class_balance` | `randomness` |
| `learning_rate` | — | `class_weight_signal` |
| `update_aggressiveness` | — | `knn_k` |
| `l1_coefficient` | — | `knn_weights_signal` |
| `l2_coefficient` | — | `knn_metric_signal` |
| `elasticnet_coefficient` | — | `var_smoothing_log` |
| `dropout_rate` | — | `attn_depth` |
| `sparsity_coefficient` | — | `attn_width` |
| `smoothing_coefficient` | — | `attn_heads` |
| `covariance_shrinkage` | — | `attn_dropout_shifted` |
| `covariance_regularization` | — | `n_layers` |
| `local_support_size` | — | `total_units` |
| `distance_norm_order` | — | `max_units` |
| `distance_weighting` | — | `ensemble_n` |
| `feature_subset_power` | — | `ensemble_lr_log` |
| `class_balance` | — | `no_tunable_hp` |
| `attention_heads` | — | — |
| `relaxation_factor` | — | — |
| `loss_smoothness` | — | — |
| `fixed_configuration` | — | — |

## 2. Semantic exclusion groups

The rule is: **a descriptor subset may contain at most one distinct descriptor from
each group**. A descriptor may belong to more than one group, in which case it occupies
all of them. The same variable may still be reused in several terms of a symbolic
equation.

The table is based on the repository's `model-descriptor-semantic-groups-v1` policy.
It also shows direct equivalents that may be removed by deduplication before semantic
exclusion is applied.

| Group | Shared semantic concept | Mutually exclusive descriptors |
|---|---|---|
| E01 | Fixed configuration or no tunable hyperparameters | ChatGPT: `fixed_configuration` · Claude1: `no_tunable_hp` · Perplexity: `no_tunable_hp` |
| E02 | Learning rate | ChatGPT: `learning_rate` · Claude1: `learning_rate` · Perplexity: `lr_log`, `ensemble_lr_log` |
| E03 | Class balancing | ChatGPT: `class_balance` · Claude1: `class_balance` · Perplexity: `class_weight_signal` |
| E04 | KNN neighbourhood size | ChatGPT: `local_support_size` · Perplexity: `knn_k` |
| E05 | KNN distance metric | ChatGPT: `distance_norm_order` · Perplexity: `knn_metric_signal` |
| E06 | KNN distance weighting | ChatGPT: `distance_weighting` · Perplexity: `knn_weights_signal` |
| E07 | Attention heads | ChatGPT: `attention_heads` · Perplexity: `attn_heads` |
| E08 | Ensemble size or number of iterations | ChatGPT: `component_count` · Claude1: `ensemble_size` · Perplexity: `iters_log`, `ensemble_n` |
| E09 | Model depth | ChatGPT: `sequential_depth` · Claude1: `capacity` · Perplexity: `capacity_depth`, `attn_depth`, `n_layers` |
| E10 | Model width | ChatGPT: `representation_width` · Claude1: `capacity` · Perplexity: `capacity_width`, `attn_width`, `max_units` |
| E11 | Aggregate model capacity | ChatGPT: `structural_size_proxy` · Claude1: `capacity` · Perplexity: `total_units` |
| E16 | L2 regularization | ChatGPT: `l2_coefficient` · Perplexity: `reg_l2_log` |
| E17 | L1, elastic-net, or sparsity regularization | ChatGPT: `l1_coefficient`, `elasticnet_coefficient`, `sparsity_coefficient` · Perplexity: `reg_l1_log` |
| E18 | Covariance stabilization | ChatGPT: `covariance_shrinkage`, `covariance_regularization` · Perplexity: `reg_l2_log` |
| E19 | Smoothing | ChatGPT: `smoothing_coefficient` · Perplexity: `var_smoothing_log` |
| E20 | Dropout or stochastic noise | ChatGPT: `dropout_rate` · Claude1: `stochasticity` · Perplexity: `randomness`, `attn_dropout_shifted` |
| E21 | Feature-selection randomness | ChatGPT: `feature_subset_power` · Perplexity: `randomness` |
| E22 | TabNet relaxation | ChatGPT: `relaxation_factor` · Perplexity: `randomness` |

The following descriptors have no direct competitor in this adaptation and therefore
do not form useful single-member exclusion groups: `unbounded_depth`,
`train_work_proxy`, `inference_work_proxy`, `update_aggressiveness`,
`loss_smoothness`, and Claude1's aggregate `regularization` descriptor.

## 3. ChatGPT descriptor formulas

### 3.1 Notation

- \(T\): number of estimators or components.
- \(D\): maximum depth.
- \(L\): number of leaves used by the proxy. For ordinary trees,
  \(L=2^D\); for LightGBM, \(L=\min(2^D,\text{num\_leaves})\).
- \(h_1,\ldots,h_k\): widths of the dense layers.
- \(U=\sum_i h_i\) and \(C_{int}=\sum_{i=1}^{k-1}h_i h_{i+1}\).
- For an explicit configuration without a special rule, the six structural
  descriptors start at `1`. For a `unique` configuration, they all start at `0`.

### 3.2 Definitions

| Descriptor | Exact formula or rule |
|---|---|
| `component_count` | \(T=\text{n\_estimators}\) for XGBoost, LightGBM, AdaBoost, and Bagging; `1` for other explicit configurations; `0` for `unique`. |
| `sequential_depth` | \(D\) for trees; \(k\) for MLP/DNN and TabTransformer; `n_steps` for TabNet; `n_blocks` for FT-Transformer; `1` for the generic explicit case. If `max_depth=None`, the value is `0`. |
| `unbounded_depth` | \(\mathbf{1}[\text{max\_depth is None}]\) for tree models; `0` otherwise. |
| `representation_width` | Finite tree: \(L\); unbounded tree with a leaf limit: that limit; dense network: \(\max_i h_i\); TabNet: \(n_d+n_a\); FT-Transformer: `d_block`; KNN: `n_neighbors`; generic case: `1`. |
| `structural_size_proxy` | Tree: \(T\max(1,2L-1)\); AdaBoost/Bagging: \(T\); MLP/DNN/TabTransformer: \(U+C_{int}\); TabNet: \(n_{steps}(n_d+n_a)^2\); FT-Transformer: \(n_{blocks}d_{block}^2\); KNN: `n_neighbors`; generic case: `1`. |
| `train_work_proxy` | Finite tree: \(T\max(1,D)L\); unbounded tree with a leaf limit: \(TL\); AdaBoost/Bagging: \(T\); dense networks: \(U+C_{int}\); TabNet: \(n_{steps}(n_d+n_a)^2\); FT-Transformer: \(n_{blocks}d_{block}^2\); KNN and generic case: `1`. |
| `inference_work_proxy` | Finite tree: \(T\max(1,D)\); unbounded tree: `0`; AdaBoost/Bagging: \(T\); dense networks: \(U+C_{int}\); TabNet: \(n_{steps}(n_d+n_a)^2\); FT-Transformer: \(n_{blocks}d_{block}^2\); KNN: `n_neighbors`; generic case: `1`. |
| `learning_rate` | Raw value: `learning_rate` for XGBoost/AdaBoost and `lr` for TabTransformer/FT-Transformer. |
| `update_aggressiveness` | PassiveAggressive: \(C\). |
| `l1_coefficient` | LR with L1 penalty: \(1/C\); SGD or Perceptron with L1: `alpha`. |
| `l2_coefficient` | LR or LinearSVC: \(1/C\); Ridge: `alpha`; SGD/Perceptron with L2: `alpha`; XGBoost: `reg_lambda`; MLP/DNN: `alpha`; TabTransformer/FT-Transformer: `weight_decay`. |
| `elasticnet_coefficient` | SGD with `penalty='elasticnet'`: `alpha`. |
| `dropout_rate` | FT-Transformer: \((\text{attention\_dropout}+\text{ffn\_dropout})/2\). |
| `sparsity_coefficient` | TabNet: `lambda_sparse`. |
| `smoothing_coefficient` | GaussianNB: `var_smoothing`; BernoulliNB: `alpha`. |
| `covariance_shrinkage` | LDA/QDA: `0` if `shrinkage=None`; otherwise, the numeric `shrinkage` value. |
| `covariance_regularization` | QDA: `reg_param`. |
| `local_support_size` | KNN: `n_neighbors`. |
| `distance_norm_order` | KNN: `1` for the Manhattan metric; `2` for the Euclidean metric. |
| `distance_weighting` | KNN: `0` for uniform weights; `1` for distance weights. |
| `feature_subset_power` | DT/ExtraTree: `1.0` if `max_features=None`; `0.5` if `max_features='sqrt'`. With \(F\) features, this represents \(F^{\text{feature\_subset\_power}}\). |
| `class_balance` | \(\mathbf{1}[\text{class\_weight='balanced'}]\) for LR, Ridge, SGD, Perceptron, DT, ExtraTree, LinearSVC, and both LightGBM variants. |
| `attention_heads` | FT-Transformer: `attention_n_heads`. |
| `relaxation_factor` | TabNet: `gamma`. |
| `loss_smoothness` | `1` for LR, Ridge, and SGD with `loss='log_loss'`; `0` for SGD with hinge loss, PassiveAggressive, Perceptron, and all other models. |
| `fixed_configuration` | \(\mathbf{1}[\text{Model Parameters}='unique']\). |

## 4. Claude1 descriptor formulas

Let

\[
\operatorname{val}(M,p,v)=
\begin{cases}
\log_{10}(v), & \text{if the grid of }p\text{ in }M\text{ spans at least two orders of magnitude},\\
v, & \text{otherwise.}
\end{cases}
\]

Also, \(\operatorname{units}(h)=\sum_i h_i\). Non-applicable values are filled
with `0.0` in the `filled` artifact.

| Descriptor | Formula by model family |
|---|---|
| `no_tunable_hp` | \(\mathbf{1}[\text{Model Parameters}='unique']\); active for TabICL and TabPFN. |
| `capacity` | DT/ExtraTree: \(\operatorname{val}(\text{max\_depth})\), except for `None`; XGBoost: \(\operatorname{val}(\text{max\_depth})+\operatorname{val}(\text{n\_estimators})\); LightGBM: \(\operatorname{val}(\text{num\_leaves})+\operatorname{val}(\text{max\_depth})+\operatorname{val}(\text{n\_estimators})\); AdaBoost/Bagging: \(\operatorname{val}(\text{n\_estimators})\); MLP/DNN: \(\operatorname{units}(\text{hidden\_layer\_sizes})\); TabNet: \(n_d+n_a+n_{steps}\); TabTransformer: \(\operatorname{units}(\text{mlp\_hidden\_mults})\); FT-Transformer: \(n_{blocks}d_{block}\); KNN: \(-n_{neighbors}\). |
| `regularization` | LR: `0` without a penalty; otherwise \(-\operatorname{val}(C)\). Ridge: \(\operatorname{val}(\alpha)\). SGD/Perceptron: `0` without a penalty; otherwise \(\operatorname{val}(\alpha)\). LinearSVC/PassiveAggressive: \(-\operatorname{val}(C)\). GaussianNB: \(\operatorname{val}(\text{var\_smoothing})\). BernoulliNB: \(\operatorname{val}(\alpha)\). XGBoost: raw `reg_lambda`. MLP/DNN: \(\operatorname{val}(\alpha)\). TabNet: \(\operatorname{val}(\lambda_{sparse})\). TabTransformer/FT-Transformer: \(\operatorname{val}(\text{weight\_decay})\). LDA: `0` without shrinkage, otherwise `shrinkage`. QDA: `(shrinkage or 0) + reg_param`. |
| `ensemble_size` | XGBoost, LightGBM, AdaBoost, and Bagging: \(\operatorname{val}(\text{n\_estimators})\); TabNet: `n_steps`. |
| `learning_rate` | XGBoost/AdaBoost: \(\operatorname{val}(\text{learning\_rate})\); TabTransformer/FT-Transformer: \(\operatorname{val}(lr)\). |
| `stochasticity` | FT-Transformer: `attention_dropout + ffn_dropout`. |
| `class_balance` | \(\mathbf{1}[\text{class\_weight='balanced'}]\) for model families that provide `class_weight`. |

## 5. Perplexity descriptor formulas

Let \(\varepsilon\) be a small positive constant used to avoid taking the logarithm
of zero. Perplexity shifts some logarithms so that `0` remains reserved for the
“not applicable” state.

| Descriptor | Proposed formula or rule |
|---|---|
| `reg_l2_log` | For `C`: \(-\log_{10}(C)+5\). For `alpha`, `reg_lambda`, `weight_decay`, or `lambda_sparse`: \(\log_{10}(x+\varepsilon)+5\). For `shrinkage` or `reg_param`: \(x+1\). If several contributions exist, they are averaged. |
| `reg_l1_log` | With an L1/elastic-net penalty: \(\log_{10}(\alpha)+5\); with `lambda_sparse`: \(\log_{10}(\lambda_{sparse}+\varepsilon)+5\). |
| `capacity_depth` | Maximum available value among `max_depth`, `n_blocks`, and `n_steps`. |
| `capacity_width` | `num_leaves`; or \(\sum_i h_i\) for hidden layers; or \(n_d+n_a\) for TabNet; or `d_block`; or \(\sum_i\text{mlp_hidden_mults}_i\). If several measures exist, the maximum is used. |
| `lr_log` | \(\log_{10}(lr)+5\), using `learning_rate` or `lr`; if several rates exist, their shifted logarithms are averaged. |
| `iters_log` | \(\log_{10}(\text{n\_estimators})+1\). |
| `randomness` | `max_features=None` → `1.0`; `max_features='sqrt'` or another value → `0.5`; dropout → mean dropout `+ 0.01`; TabNet → `gamma + 0.01`. If several contributions exist, they are averaged. |
| `class_weight_signal` | `1` if `class_weight='balanced'`; `-1` if `class_weight=None`; `0` if the parameter does not exist. |
| `knn_k` | KNN: `n_neighbors`. |
| `knn_weights_signal` | KNN: `1` for `weights='distance'`; `-1` for `weights='uniform'`. |
| `knn_metric_signal` | KNN: `1` for Manhattan; `-1` for Euclidean. |
| `var_smoothing_log` | GaussianNB: \(\log_{10}(\text{var\_smoothing})+6\). |
| `attn_depth` | FT-Transformer: `n_blocks`. |
| `attn_width` | FT-Transformer: `d_block`. |
| `attn_heads` | FT-Transformer: `attention_n_heads`. |
| `attn_dropout_shifted` | FT-Transformer: \((\text{attention\_dropout}+\text{ffn\_dropout})/2+0.01\). |
| `n_layers` | MLP/DNN: \(\operatorname{len}(\text{hidden\_layer\_sizes})\). |
| `total_units` | MLP/DNN: \(\sum_i h_i\). |
| `max_units` | MLP/DNN: \(\max_i h_i\). |
| `ensemble_n` | XGBoost, LightGBM, AdaBoost, and Bagging: `n_estimators`. |
| `ensemble_lr_log` | Ensemble models with a learning rate: \(\log_{10}(\text{learning\_rate})+5\). |
| `no_tunable_hp` | \(\mathbf{1}[\text{Model Parameters}='unique']\). |

## 6. Interpretation notes

- Descriptors with similar names do not always use the same scale. For example,
  ChatGPT keeps `learning_rate` on its raw scale, Claude1 applies `val` according
  to the grid range, and Perplexity uses a shifted logarithm.
- ChatGPT's size and work proxies are not execution times, FLOPs, or exact counts
  of trainable parameters.
- Claude1's `filled` variant uses `0.0` both for “not applicable” and for some
  neutral states, so the encoding loses the distinction between them.
- In the Perplexity representation, the `+5`, `+1`, `+6`, and `+0.01` shifts are
  intended to prevent an applicable value from being exactly zero.

## 7. Repository sources

- `src/ml_meta_perf/chatgpt/hyperparameter_descriptors.md`
- `src/ml_meta_perf/chatgpt/hyperparameters_encoded.csv`
- `src/ml_meta_perf/claude1/descriptors_filled.md`
- `src/ml_meta_perf/claude1/hyperparameters_encoded_filled.csv`
- `src/ml_meta_perf/perplexity/descriptores_hyperparametros.md`
- `src/ml_meta_perf/perplexity/hyperparameters_encoded.csv`
- `src/ml_meta_perf/descriptor_subset_search.py`

