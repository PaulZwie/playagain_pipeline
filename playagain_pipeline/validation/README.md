# `playagain_pipeline.validation`

A reproducible, config-driven validation harness for the EMG pipeline.

## Why this exists

Until now, comparing feature sets or model architectures meant clicking
through the GUI, copy-pasting numbers from the log, and hoping the same
data was used the next time. That is fine for exploration but not for a
thesis or paper. This package fixes three problems:

1. **Two recorders, one corpus.** Sessions are written by two different
   stacks: the Python pipeline (`data/sessions/<subject>/...`) and the
   Unity C# game (`data/sessions/unity_sessions/...`,
   `DataManager.cs` + `DeviceManager.cs`). Both already write the same
   on-disk layout (`data.npy`, `metadata.json`, `gesture_set.json`), so
   `SessionCorpus` discovers them uniformly and tags each with a
   `source_domain ∈ {"pipeline", "unity"}`. Cross-domain experiments
   become a one-line config change.
2. **Honest cross-validation.** Splitting random *windows* from one
   session into train / test is the single biggest source of inflated
   accuracy in EMG papers. All splitters in `cv_strategies.py` operate
   at session granularity: no two windows from the same physical
   recording can ever be in both train and test.
3. **Reproducibility.** Every run dumps the exact config, the git SHA,
   the package versions, and the list of session paths it consumed
   into a timestamped folder. Re-running the same YAML on the same git
   commit produces the same numbers.

## File layout

```
validation/
├── __init__.py            re-exports the public API
├── __main__.py            `python -m playagain_pipeline.validation ...`
├── corpus.py              SessionRecord + SessionCorpus
├── cv_strategies.py       within-session, LOSO-session, LOSO-subject,
│                          k-fold-subjects, cross-domain (pipeline ↔ unity)
├── config.py              ExperimentConfig dataclass + YAML/JSON loader
├── runner.py              ValidationRunner — the orchestrator
├── experiments_example.yaml
└── README.md              (this file)
```

## The five-minute tour

```bash
# 1. See what's on disk
python -m playagain_pipeline.validation summary
#   SessionCorpus @ /…/playagain_pipeline/data
#     total sessions : 137
#     pipeline       :  92
#     unity          :  45
#     subjects       :  11

# 2. List individual sessions
python -m playagain_pipeline.validation list

# 3. Run the LOSO baseline experiment
python -m playagain_pipeline.validation run \
    playagain_pipeline/validation/experiments_example.yaml

#   …
#   Wrote: data/validation_runs/2026-04-14_113022__loso_baseline/
#   RandomForest        acc=0.812±0.041  f1=0.804±0.046  (n=11)
#   CatBoost            acc=0.838±0.038  f1=0.829±0.043  (n=11)
#   LDA                 acc=0.751±0.052  f1=0.738±0.057  (n=11)
```

## Anatomy of a result directory

```
data/validation_runs/2026-04-14_113022__loso_baseline/
├── experiment.json       ← the config, frozen
├── environment.json      ← git SHA, python, numpy/sklearn/torch versions
├── session_index.json    ← every session path used (subject, domain, sr)
├── results.json          ← per-fold + aggregate metrics, machine-readable
└── results.csv           ← per-fold flat table, pandas/Excel-friendly
```

The combination of `experiment.json` + `environment.json` +
`session_index.json` is sufficient to fully reconstruct the run. A
collaborator with the same git commit and the same `data/` directory
will get bit-identical numbers.

## CV strategies

| Strategy           | Train / test split unit   | When to use                                                                  |
|--------------------|---------------------------|------------------------------------------------------------------------------|
| `within_session`   | temporal tail of session  | Optimistic upper bound. Use only as a sanity check, never as a headline.     |
| `loso_session`     | one session held out      | Generalisation across recording sessions of the same (and other) subjects.  |
| `loso_subject`     | all sessions of one subj  | The honest single number. Default headline metric for a paper.              |
| `k_fold_subjects`  | k subject groups          | Use when LOSO is too expensive (>20 subjects).                              |
| `cross_domain`     | by `source_domain`        | "Does a model trained on pipeline data still work in the Unity game?"       |
| `holdout_split`    | configurable ratios       | Train / Val / Test holdout (deep-model tuning, early stopping). Stratified by subject by default. |

The Unity ↔ pipeline cross-domain experiment is the one the C# recorder
was built for. It directly answers whether the GUI training results
transfer to the real in-game stream.

`holdout_split` is the only strategy that produces a separate
**validation set**. When present, the runner passes it to the model's
`train()` method as `X_val` / `y_val`, so deep-learning models
(`mlp`, `cnn`, `attention_net`, `mstnet`) can use it for early stopping
and live training curves. Classical models ignore it gracefully and
report the test-set numbers only.

## How an experiment is glued to the existing pipeline

`runner.py` deliberately does **not** import any heavyweight ML deps at
the top level. The three integration points all live inside
`ValidationRunner` and are kept short:

| Method                | Reuses…                                            |
|-----------------------|----------------------------------------------------|
| `_materialise_fold`   | `core.data_manager.DataManager` for windowing, `models.feature_pipeline.FeaturePipeline` for features. Same code paths the GUI Train tab uses, so behaviour matches byte-for-byte. |
| `_fit_model`          | `models.classifier.ModelManager.create_model` + `train_model`. |
| `_evaluate`           | `sklearn.metrics.accuracy_score` + `f1_score`.     |

If you change a feature implementation in the main pipeline, the
validation harness picks it up automatically — that's the point.

## Adding a new experiment

1. Copy `experiments_example.yaml` to e.g. `experiments/my_run.yaml`.
2. Edit the `data`, `features`, `models`, and `cv` sections.
3. Commit the YAML alongside any code changes that produced its
   numbers — the `environment.json` will record the SHA.
4. Run it. The output directory name encodes the experiment name and
   the timestamp, so you can have hundreds of runs side-by-side without
   collisions.

## Adding a new CV strategy

Drop a function in `cv_strategies.py` that takes `List[SessionRecord]`
and yields dicts of the form `{"id", "train", "test", "split_kind",
...}`, then register it in the `STRATEGIES` dict at the bottom of the
file. The runner picks it up by name from any YAML — no other changes
needed.

## Adding a new metric

Currently `_evaluate` reports accuracy + macro-F1 + per-class F1 +
per-window inference latency. To add e.g. confusion matrices or
top-3 accuracy, extend `_evaluate` and the matching `FoldResult`
fields. Both are tiny.

## Caveats

* `_materialise_fold` calls `DataManager.extract_windows_from_signal`,
  which is the natural place for that helper to live in the existing
  codebase. If your `DataManager` does not yet expose a function with
  that name, factor the windowing logic out of
  `DataManager.create_dataset` into a reusable method — it's a 30-line
  refactor and pays back immediately.
* The Unity recorder writes per-sample labels in slightly different
  ways depending on the game version. `SessionRecord.load_labels`
  tries both `labels.npy` and a `label`/`gesture_id`/`ground_truth`
  column in `data.csv`. If you find a third variant, add it there.
* `cross_domain` requires both domains to be non-empty after the
  `data` filter. If you ask for `domains: [unity]` *and*
  `cv.strategy: cross_domain`, you'll get zero folds.

## Testing it on the existing data

A good first run, given the tree you posted:

```yaml
name: vp01_within_subject_sanity
seed: 42
data:
  subjects: [VP_01]
  domains: [pipeline]
windowing:  { window_ms: 200, stride_ms: 50, drop_rest: false }
features:
  - { name: mav }
  - { name: rms }
  - { name: wl }
  - { name: zc, params: { threshold: 0.01 } }
models:
  - { type: RandomForest, params: { n_estimators: 200 } }
cv:
  strategy: loso_session
```

VP_01 has ~25 sessions, so this produces 25 folds and finishes in a
couple of minutes on a laptop. The output is the most defensible
single number you can put next to "VP_01 Random Forest" in a thesis.
