# WIP: searching the AI-proposed descriptors for a better `MODEL_FEATURES`

**Status: paused, waiting on input.** The user is fetching the exact feature list a
student found (same codebase, same corpus) that reaches LOO-model R² ≈ 0.7 and
LOO-dataset R² ≈ 0.6 with 10 model features (8 novel, 2 from the current
`MODEL_FEATURES`) in a 25-term equation. Current best found here is nowhere near that.
**When that list arrives, testing it directly (see "Fastest next step" below) is the
highest-value thing to do before anything else in this file.**

Nothing in this branch has touched `data.py`, the shipped `meta_dataset.csv`, or
`experiment.py`'s tuned defaults (`DEFAULT_E2`/`DEFAULT_E3`). All of it is additive
tooling plus one candidate CSV, gated so `pip install .` (no extras) and CI are
unaffected. Phase B (actually wiring a winning set into production) has not started.

## Goal

`assets/ai_model_descriptors.md` documents ~61 candidate model-hyperparameter
descriptors proposed by three AIs (ChatGPT, Claude1, Perplexity), computed against this
project's corpus and shipped as `assets/meta_dataset_all_descriptors.csv` (476 rows: the
12 dataset features, the 6 current `MODEL_FEATURES`, ~55 candidate descriptors, `MCC`).
Find a small, defensible replacement/extension for `data.MODEL_FEATURES` — not just
"whatever ends up in one fitted equation", but a set that would survive a search
methodology described in an academic write-up.

## What exists in this branch

- `src/ml_meta_perf/descriptors.py` — the candidate registry.
  - `CANDIDATE_FEATURES` (61 = current 6 + 26 ChatGPT + 7 Claude1 + 22 Perplexity),
    `EXCLUSION_GROUPS` (transcribed from `ai_model_descriptors.md` §2 — descriptors
    naming the same concept), `INADMISSIBLE` (6 descriptors dropped on a continuous/
    ordered-value rule this session's review established — each with its reason).
  - `Registry` — `build_registry(frame)` gives the admissible pool + a conflict graph
    from `EXCLUSION_GROUPS`. `resolve_conflicts(selected)` collapses any selected set to
    one member per group (a candidate can span several groups at once, e.g. Claude1's
    `capacity` spans depth/width/aggregate-capacity — handled as a proper conflict graph
    + greedy independent set, not "one slot per group", see the module docstring).
  - `compute_strengths(frame, candidates)` — max(|pearson|, |spearman|) vs MCC per
    candidate.
  - `Registry.reduce_to_representatives(strengths)` — collapses each exclusion group (and
    connected clusters of them) to its single strongest-correlated member. 55 → 27
    admissible.
  - `Registry.with_correlation_conflicts(frame, threshold=0.9)` — **added on user
    request**: extends the conflict graph with *empirical* pairwise correlation among ALL
    admissible candidates, not just named groups (two descriptors from different concepts
    can be near-duplicates on this one corpus). Feed its output into
    `reduce_to_representatives` the same way. Combined with group reduction: 55 → 25.

- `src/ml_meta_perf/feature_search.py` — the search engine. Three backends, all
  documented in the module docstring with their own reasoning:
  - `evaluate_subset(features, frame, dataset_features=DATASET_FEATURES, ...)` — scores
    one model-feature subset **in the E3 frame** (dataset features always in the
    library — this was originally model-features-alone/E2-style and that was a real bug,
    see "Mistakes already made" below). Returns leave-one-dataset-out R² at a reduced
    term budget. Pass `dataset_features=()` for the cheaper model-alone diagnostic.
  - `optuna_search` / `pyblindopt_search` — multi-objective (R², feature count) search
    over a `Registry`, via Optuna's NSGA-II and an OBLESA-seeded pyBlindOpt DE
    respectively, both calling `evaluate_subset` at `SEARCH_MAX_ARITY=2` for speed.
  - `stability_selection` — Meinshausen-Bühlmann-style repeats over dataset subsamples,
    tallying how often each candidate survives.
  - `greedy_forward_search(registry, frame, config=GREEDY_SEARCH_CONFIG, ...)` — **added
    on user request**, and the most trustworthy backend so far: scores every step with
    the **real** `experiment.run_equation` (arity-3), not a cheap proxy, climbing the
    **mean of LOO-dataset and LOO-model R²** (not LOO-dataset alone — see "Why LOO-model
    matters" below). `GREEDY_SEARCH_CONFIG` is a reduced-but-still-arity-3 budget
    (`pool_size=250, max_terms=16, headline_terms=16`) for speed, and **even this has a
    measured fidelity gap against full `DEFAULT_E3`** — see "The open problem" below.

- `src/ml_meta_perf/feature_search_cli.py` — `ml-meta-perf-select-features` console
  script wiring the above together with CLI flags. Not used for the later, more targeted
  investigation (direct Python snippets were faster to iterate with); should still work,
  untested against the latest `greedy_forward_search`/correlation-conflicts additions.

- `tests/test_descriptors.py`, `tests/test_feature_search.py` — 35 tests, all passing,
  fast (~11s). Verified clean (ruff, basedpyright, vulture, full 407-test suite) in both
  the dev venv (`requirements.txt`, has the `search` extra) and a simulated CI install
  (`pip install .` only) — basedpyright needs `pyright: ignore[reportMissingImports]` on
  the optuna/pyBlindOpt imports specifically because CI does not install the `search`
  extra; this was a real failure caught and fixed, don't remove those comments.

- `assets/ai_model_descriptors.md`, `assets/meta_dataset_all_descriptors.csv` — the
  source document and candidate corpus, copied in from `~/` (not shipped package data,
  research inputs only).

- `pyproject.toml` — new `[project.optional-dependencies] search = ["optuna>=4.0.0",
  "pyBlindOpt>=0.5.0"]` extra, plus the `ml-meta-perf-select-features` script entry.
- `requirements.txt` (repo root, new) — dev venv setup: `-e .[search]` plus ruff/
  basedpyright/vulture/pre-commit pinned to CI's exact versions.

## Environment

```
python3 -m venv venv          # NOT .venv -- pre-commit hooks and .gitignore assume venv/
venv/bin/pip install -r requirements.txt
venv/bin/pre-commit install
```

## Mistakes already made, so they aren't repeated

1. **Scored model features alone (E2-style) at first.** No dataset features in the
   library means no dataset×model cross term (ratio/product/sum_ratio) is ever built, so
   a model feature that's weak alone but valuable as `dataset_feature / model_feature`
   scored as worthless and got dropped. `02-equation-form.md` documents that interaction
   as carrying most of E3's lift over E1 — this is why `evaluate_subset` now defaults to
   `dataset_features=DATASET_FEATURES`. **If you're extending this search, never go back
   to model-alone scoring as the primary objective.**

2. **`Library.__init__` raises rather than returning empty** (`terms.py:511`) when every
   candidate term fails the stability filter — a real, reachable case here (several
   descriptors are near-zero for every model but one, e.g. `chatgpt__l1_coefficient`).
   `evaluate_subset` catches this specific `ValueError`. If you add another call site
   that builds a `Library` from a search-chosen subset, it needs the same guard.

3. **pyBlindOpt's `compute_objective` tries a whole-population vectorized call first**,
   and only falls back to one-call-per-individual if that raises (see
   `pyBlindOpt/utils.py`'s `compute_objective` docstring — it's actually well documented
   there). An objective that only handles the batch case will crash on fallback with a
   confusing `'numpy.float64' object is not iterable`. Write pyBlindOpt objectives as
   **one vector in, one scalar out** — don't try to be clever/vectorized, the underlying
   fits aren't vectorizable across candidate subsets anyway.

4. **A cheap search-time proxy that ranks candidates differently from the real pipeline
   is worse than no proxy.** This happened twice, at two different levels of
   "cheapness":
   - `SEARCH_MAX_ARITY=2` (no three-feature `sum_ratio`) in `optuna_search`/
     `pyblindopt_search`: a candidate set that looked strong under arity-2 search
     (R² ≈ 0.43-0.45) scored *worse than the current baseline* (0.478) under full
     `DEFAULT_E3` (arity-3). This is why `greedy_forward_search` uses arity-3.
   - Even `GREEDY_SEARCH_CONFIG` (arity-3, but `pool_size=250, max_terms=16,
     headline_terms=16` instead of `DEFAULT_E3`'s 600/32/20) is not fully faithful: the
     6-feature set it found scored loo_dataset=0.4827 at search time but **0.4285** under
     full `DEFAULT_E3` — a 0.054 gap, in the direction that would have looked like an
     improvement over baseline (0.478) at search time and is actually a regression.
     loo_model transferred much better (0.5223 search-time vs 0.5224 full — near-exact).
   - **This is the open problem**, not fully solved. See below.

## Why LOO-model (LOMO) matters here

`06-results.md` already documents it as the weaker of the two protocols for the current
6-feature set (0.428 vs LOO-dataset's 0.478), and the user's diagnosis (verified,
directly informed this branch's `greedy_forward_search` design) is that **the current
model features don't capture enough of what varies between models to generalise to an
unseen one** — even though `06-results.md` also shows the current 6 (with
`Model Capability`) already capture 99% of what model *identity* can explain
*in-sample*. The two are not the same claim: capturing in-sample model-identity
variance and generalising that description to a model never seen during fitting are
different things, and the gap between them is exactly what LOO-model measures.

## Current best confirmed result (full `DEFAULT_E3`, i.e. trustworthy)

Baseline, current 6 features, on `assets/meta_dataset_all_descriptors.csv` (same rows as
the shipped corpus, extra columns):

```
loo_dataset=0.4779  loo_model=0.4276  in_sample=0.6141  n_terms=20
```

`greedy_forward_search` (at `GREEDY_SEARCH_CONFIG`, then confirmed at full `DEFAULT_E3`)
found, after 6 steps (stopped early — see below):

```python
('Model Capability', 'Processing Units Number', 'Training Operations',
 'chatgpt__feature_subset_power', 'chatgpt__l1_coefficient', 'chatgpt__unbounded_depth')
```
```
loo_dataset=0.4285  loo_model=0.5224  in_sample=0.6038  n_terms=20
```

Net: **+0.095 on LOO-model, -0.050 on LOO-dataset** against baseline. A real, confirmed
result — not nothing — but far short of the student's reported 0.6/0.7, and the search
stopped at 6 of a possible 15 features because the *search-time* (not full-fidelity)
score plateaued, which given the fidelity gap above doesn't mean the real score would
have too.

Numbers were read directly off stdout during the session, not saved to a file inside the
repo (`results/` is gitignored). Reproduce with the snippet under "How to test a specific
feature set" below.

## Fastest next step: test the student's actual feature list

The single fastest way to close the gap and diagnose whether this is a search problem or
something else (the user's working hypothesis, based on `06-results.md`'s own numbers,
is that it's a search/feature problem, not a data problem — "the only change is features
for the ML models, that you can check on the CSV"). Once you have the list:

```python
import polars as pl
from ml_meta_perf.data import DATASET_FEATURES, MODEL_FEATURES
from ml_meta_perf.experiment import run_equation, DEFAULT_E3
import dataclasses

frame = pl.read_csv("assets/meta_dataset_all_descriptors.csv")

student_features = (...)  # fill in -- must be columns present in the CSV above
# If the student's equation has 25 terms, headline_terms=20 will truncate it:
config = dataclasses.replace(DEFAULT_E3, headline_terms=25, max_terms=32)
r = run_equation(frame, DATASET_FEATURES, student_features, config, "student")
print(r.cross_validated["loo_dataset"]["r2"], r.cross_validated["loo_model"]["r2"], r.in_sample["r2"])
```

If the student's named features aren't literal column names in
`meta_dataset_all_descriptors.csv` (e.g. they used slightly different naming or computed
their own descriptor values), that itself answers "CSV issue vs. search issue" — check
column names first (`frame.columns`) before assuming a mismatch means the data is wrong.

If this reproduces ~0.6/~0.7: **done** — that feature list is the answer; write it up and
move to Phase B (see bottom).

If it does NOT reproduce those numbers even with the right features: the gap is in
`DEFAULT_E3`'s configuration (penalty/arity/term-count), not feature choice — re-check
against a sweep like `02-equation-form.md`'s (chapter 2 already documents that penalty
must be re-tuned per configuration, not carried over).

## If the student's list isn't available or doesn't close the gap

1. **Re-run `greedy_forward_search` at full `DEFAULT_E3` fidelity**, not
   `GREEDY_SEARCH_CONFIG`. Warm-start from the 6-feature set above (skip re-deriving it)
   and extend forward through the remaining ~19 of the 25 correlation-and-group-pruned
   candidates, up to 10 features, using `config=DEFAULT_E3` (or a `dataclasses.replace`
   bumping `headline_terms`/`max_terms` toward 25, matching the student's equation
   length). Cost: full `DEFAULT_E3` is ~10.85s/eval measured this session (vs ~3.1s for
   `GREEDY_SEARCH_CONFIG`); warm-starting from 6 and adding up to 4 more against ~19-16
   remaining candidates is roughly 70 evals ≈ 13 minutes, run in background.
2. Consider that `headline_terms=20` may itself be the wrong length for a *different*
   feature set — `02-equation-form.md` is explicit that length and penalty were
   "re-swept rather than carried over" every time the feature set changed historically
   (`experiment.py:81-87`'s comment about `Model Capability` joining is the precedent).
   A fair comparison for any new candidate set may need its own penalty/length sweep,
   not `DEFAULT_E3`'s as-is.
3. The correlation threshold (`0.9`) in `with_correlation_conflicts` and the exclusion
   groups were not swept — a lower threshold (more aggressive pruning) or a hand check
   of the 25-candidate reduced pool against `EXCLUSION_GROUPS` might surface a
   differently-reduced pool worth searching.

## Phase B (not started, don't do this until a winning set is actually confirmed)

Wire the winning set into `data.py` (`MODEL_FEATURES`, `FEATURE_GLOSSARY`), merge the
winning new columns into the shipped `src/ml_meta_perf/meta_dataset.csv`, re-sweep
`DEFAULT_E2`/`DEFAULT_E3` (required whenever the feature set changes, per
`experiment.py:81-87`'s own precedent), update `assets/docs/`. This changes the
project's published headline numbers — confirm with the user before starting it, per
this session's own working agreement (see the two AskUserQuestion exchanges in this
session's transcript, if available, or just ask again).
