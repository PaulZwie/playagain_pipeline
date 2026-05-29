# Thesis evaluation pipeline — chapter 6, 7, 8 outputs

This bundle extends the existing `playagain_pipeline/validation/` package so
every numerical placeholder in `06_Results.tex` and every figure
`\includegraphics{...}` reference is produced automatically from saved
runner outputs. It does **not** replace `runner.py` — it sits on top.

## What it produces

Every table and figure referenced from Chapters 6–8 of the thesis:

| Thesis reference | Artefact |
|---|---|
| Table 6.1 (per-participant corpus) | `table_6_1_participants.csv` |
| Table 6.2 (class distribution) | `table_6_2_class_distribution.csv` |
| §6.1.3 (recording origins) | `section_6_1_3_recording_origins.json` |
| Fig 6.1 (calibration confidence) | `fig_6_1_calibration_confidence.{pdf,png}` |
| §6.2 (median, IQR, flagged counts) | `calibration_report.json` |
| Table 6.3 (LOSO-session, mean ± SD) | `table_6_3_loso_session.csv` |
| Fig 6.3 (per-class F1 with error bars) | `fig_6_3_per_class_f1.{pdf,png}` |
| Fig 6.4 (normalised confusion matrices) | `fig_6_4_confusion_matrices.{pdf,png}` |
| Fig 6.5 (per-session variability) | `fig_6_5_per_session_variability.{pdf,png}` |
| Table 6.4 (LOSO-subject) | `table_6_4_loso_subject.csv` |
| Table 6.5 (feature ablation) | `table_6_5_feature_ablation.csv` |
| Fig 6.6 (feature ablation bar chart) | `fig_6_6_feature_ablation.{pdf,png}` |
| Table 6.6 (cross-domain) | `table_6_6_cross_domain.csv` |
| Table 6.7 (latency + 150 ms gate) | `table_6_7_latency.csv` |
| §7.4 (calibration ↔ F1 correlation) | `correlation_calibration_f1.json`, `fig_7_4_calibration_vs_f1.{pdf,png}` |

Chapter 8 reuses numbers from the chapter-6 tables — no separate artefact.

## File layout

The reporting modules now live alongside `runner.py`:

```
playagain_pipeline/validation/
├── corpus.py                        SessionRecord + SessionCorpus
├── runner.py                        ValidationRunner
├── cv_strategies.py                 fold generators
│
├── corpus_report.py                 §6.1 corpus + participant tables
├── calibration_report.py            §6.2 calibration confidence + flags
├── thesis_reports.py                LOSO aggregation, latency, x-domain tables
├── plots_thesis.py                  matplotlib helpers for figures 6.1–7.4
├── threshold_report.py / threshold_plots.py
│                                    threshold-sweep summaries (table 6.7 etc.)
├── game_report.py                   game-recording evaluation
├── recompute_calibration_metrics.py CLI: rewrite calibration stats in place
└── generate_thesis_outputs.py       one-shot orchestrator
```

## How to use it

### One-shot

After your runs are done, generate everything at once:

```bash
python -m playagain_pipeline.validation.generate_thesis_outputs \
    --data-dir         data/ \
    --primary          data/validation_runs/2026-04-12__loso_sess \
    --loso-subj        data/validation_runs/2026-04-13__loso_subj \
    --ablation         "rms=runs/abl_rms,mav=runs/abl_mav,var=runs/abl_var,wl=runs/abl_wl,zc=runs/abl_zc,ssc=runs/abl_ssc,iemg=runs/abl_iemg,ssi=runs/abl_ssi,combined=runs/abl_all" \
    --xdomain          "within_pipe=runs/wp,within_unity=runs/wu,p2u=runs/p2u,u2p=runs/u2p" \
    --primary-model    catboost \
    --out              thesis_outputs/
```

Anything left off is skipped — you can run partial generations as soon as the
matching runs finish.

### Module-by-module

Each module is also usable as a library:

```python
from playagain_pipeline.validation.corpus import SessionCorpus
from playagain_pipeline.validation.corpus_report import participant_summary
from playagain_pipeline.validation.thesis_reports import (
    load_run_result, summarise_run, latency_table,
)

corpus = SessionCorpus("data/"); corpus.discover()
print(participant_summary(corpus))

run = load_run_result("data/validation_runs/2026-04-12__loso_sess")
for model, s in summarise_run(run).items():
    print(model, s.fmt_mean_sd("macro_f1"))

for row in latency_table(run, gate_ms=150.0):
    print(row.model_type, row.inference_ms_mean, "gate:", row.passes_gate)
```

## Fold-id parsing

The reporting modules infer the held-out subject(s) from `fold_id` for
LOSO strategies, which already encode the info. For strategies that don't
(e.g. `k_fold_subjects`, `holdout_split`), the parsers fall back to the
fold-id heuristic — older saved runs continue to load. If you need an
exact join, have `ValidationRunner` write `test_subjects` /
`test_sessions` into each fold record alongside the metrics.

## Session metadata expectations

`corpus_report.py` and `calibration_report.py` rely on fields the
existing recorder already writes into `metadata.json`:

| Field | Source | Used for |
|---|---|---|
| `rotation_confidence` | calibrator | Fig 6.1, §6.2 |
| `rotation_offset` | calibrator | §6.2 offset range |
| `custom_metadata.source = "training_game"` | training-game recorder | Session-type tagging (T) |
| `custom_metadata.outreach = true` | outreach session flag | Session-type tagging (O) |
| `custom_metadata.participant_info.group` | recording dialog | Healthy / clinical (H/C) in Table 6.1 |
| `custom_metadata.exclude_from_eval = true` | post-hoc exclusion | Calibration "excluded" bucket |
| `custom_metadata.calibration_manual = true` | manual offset override | Calibration "manual" bucket |

If your existing sessions don't carry one of these, the relevant column
falls back to `"?"`, `"S"` (default session type), or simply omits the
session from the affected denominator. Nothing crashes.

## Why this lives outside `core.py` / `metrics.py`

`core.py` and `metrics.py` describe a *single* evaluation:
one model, one set of recordings, one set of metrics. The thesis tables
are aggregates *across folds* (mean ± SD), *across models* (rows),
*across runs* (feature ablation), and join external metadata
(calibration, session-type). That's a different abstraction layer, so
it gets its own modules rather than bloating `EvaluationResult`.

The new modules read what's already there:

- `RunResult` / `FoldResult` from `runner.py` — fold-level numbers
- `SessionCorpus` from `corpus.py` — session discovery
- `metadata.json` per session — calibration and participant info

…and produce thesis-shaped outputs without changing the recording or
training paths.

## Dependencies

- `numpy` (already required)
- `matplotlib` (already required — used by plot generators only)
- `scipy` *optional* — used for Pearson/Spearman p-values in
  `calibration_f1_correlation`. Falls back to numpy correlation
  (no p-value) when scipy is absent.

No PySide6, no torch, no sklearn dependency at module import time. The
reporting modules can run in a headless CI job without any ML stack.
