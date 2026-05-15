"""
validation/generate_thesis_outputs.py
─────────────────────────────────────
End-to-end driver that turns one or more saved validation runs into
the full set of thesis deliverables — tables (CSV) and figures
(PDF + PNG) — for Chapters 6, 7 and 8.

Typical invocation
──────────────────
    python -m playagain_pipeline.validation.generate_thesis_outputs \\
        --data-dir   data/ \\
        --primary    runs/2026-04-12__loso_session__catboost_lda_rf_svm \\
        --loso-subj  runs/2026-04-13__loso_subject__catboost_lda_rf_svm \\
        --ablation   "rms=runs/abl_rms,mav=runs/abl_mav,...,combined=runs/abl_all" \\
        --xdomain    "within_pipe=runs/wp,within_unity=runs/wu,p2u=runs/p2u,u2p=runs/u2p" \\
        --groups     data/participant_groups.json \\
        --primary-model catboost \\
        --out        thesis_outputs/

Outputs (under ``--out``)
─────────────────────────
    corpus_report.json
    table_6_1_participants.csv          (now includes a G column + group)
    table_6_2_class_distribution.csv
    section_6_1_3_recording_origins.json (now includes n_game)
    calibration_report.json
    calibration_per_session.csv
    fig_6_1_calibration_confidence.{pdf,png}
    table_6_3_loso_session.csv
    table_6_3b_loso_session_by_group.csv      (healthy vs impaired split)
    fig_6_3_per_class_f1.{pdf,png}
    fig_6_4_confusion_matrices.{pdf,png}
    fig_6_5_per_session_variability.{pdf,png}
    fig_6_5_per_session_variability_by_group.csv
    table_6_4_loso_subject.csv
    table_6_4b_loso_subject_by_group.csv      (healthy vs impaired split)
    table_6_5_feature_ablation.csv
    fig_6_6_feature_ablation.{pdf,png}
    table_6_6_cross_domain.csv
    table_6_7_latency.csv
    table_6_8_game_performance.csv            (game recordings, per cohort)
    table_6_9_game_per_subject.csv            (game recordings, per subject)
    fig_6_7_game_per_class_f1_data.csv
    fig_6_8_game_confusion_data.json
    game_report.json
    fig_7_4_calibration_vs_f1.{pdf,png}   (when calibration data available)
    correlation_calibration_f1.json

Healthy / impaired split
────────────────────────
``--groups`` points at a participant-group registry (see
:mod:`participant_groups`); it defaults to
``<data-dir>/participant_groups.json`` when that file exists. With it,
every LOSO table also gets a ``*_by_group`` companion and the game
report is split by cohort. Without it, cohorts fall back to
session-metadata inference and the split tables still build (subjects
with no resolvable cohort land in a ``"?"`` row).

Game recordings
───────────────
``data/game_recordings/`` is discovered automatically and scored from
its logged predictions (pass ``--skip-game`` to opt out). These
recordings carry true multi-class predictions vs ground truth, unlike
the RMS-threshold Unity recordings, so they are the better evidence for
deployed multi-class performance.

Anything that wasn't supplied (e.g. no ablation runs) is silently
skipped — the orchestrator logs what it produced.

Why this lives in the validation package
────────────────────────────────────────
The runner already lives here, and writing a separate top-level script
would duplicate the import path for ``corpus.py``. Running this as a
``python -m`` module gives correct relative imports for free.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .corpus import SessionCorpus
from .corpus_report import write_corpus_report
from .calibration_report import (
    DEFAULT_FLAG_THRESHOLD, calibration_stats, write_calibration_report,
)
from .game_corpus import GameCorpus
from .game_report import build_game_report, write_game_report
from .participant_groups import (
    ParticipantGroups, default_groups_path, metadata_group_resolver,
)
from .thesis_reports import (
    annotate_per_session_groups, calibration_f1_correlation,
    cross_domain_comparison, feature_ablation, group_model_rows,
    load_run_result, per_session_variability, summarise_run,
    write_cross_domain, write_feature_ablation, write_group_summary,
    write_run_report,
)
from .plots_thesis import (
        plot_calibration_confidence, plot_calibration_honest,
        plot_calibration_vs_f1, plot_calibration_vs_f1_per_model,
        plot_confusion_matrices, plot_feature_ablation,
        plot_per_class_f1, plot_per_session_variability,
    )



log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_kv_runs(spec: Optional[str]) -> List[Tuple[str, Path]]:
    """Parse a ``key=path,key=path`` spec into a list of pairs."""
    if not spec:
        return []
    out: List[Tuple[str, Path]] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"Expected 'name=path', got {chunk!r}")
        name, path = chunk.split("=", 1)
        out.append((name.strip(), Path(path.strip())))
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="generate_thesis_outputs",
        description=("Build the Chapter-6/7/8 tables and figures from "
                     "saved validation runs."),
    )
    p.add_argument("--data-dir", type=Path, required=True,
                   help="Pipeline data root (contains sessions/).")
    p.add_argument("--out", type=Path, required=True,
                   help="Output directory for tables and figures.")

    p.add_argument("--primary", type=Path,
                   help="LOSO-session run directory (Table 6.3, Fig 6.3–6.5, Table 6.7).")
    p.add_argument("--loso-subj", type=Path,
                   help="LOSO-subject run directory (Table 6.4).")
    p.add_argument("--ablation", type=str,
                   help=("Feature-ablation runs as 'name=path,name=path,...'. "
                         "One run per feature condition plus 'combined'."))
    p.add_argument("--xdomain", type=str,
                   help=("Cross-domain runs as 'within_pipe=path,within_unity=path,"
                         "p2u=path,u2p=path' (any subset)."))

    p.add_argument("--primary-model", default="catboost",
                   help=("Model name to use for per-session, ablation and "
                         "cross-domain plots (default: catboost)."))
    p.add_argument("--window-ms", type=int, default=200)
    p.add_argument("--stride-ms", type=int, default=50)
    p.add_argument("--drop-rest", action="store_true",
                   help="Exclude rest windows from corpus class-distribution counts.")

    # Healthy / impaired participant grouping.
    p.add_argument("--groups", type=Path,
                   help=("Participant-group registry file (healthy vs "
                         "impaired). Defaults to "
                         "<data-dir>/participant_groups.json when present. "
                         "When available, every LOSO table also gets a "
                         "per-cohort split and the corpus table's group "
                         "column becomes authoritative."))

    # Game recordings.
    p.add_argument("--skip-game", action="store_true",
                   help=("Do not build the game-recording performance "
                         "report. By default game recordings under "
                         "<data-dir>/game_recordings are discovered and "
                         "scored from their logged predictions."))

    p.add_argument("--flag-threshold", type=float,
                   default=DEFAULT_FLAG_THRESHOLD,
                   help="Calibration confidence flag threshold (default: 0.5).")
    p.add_argument("--gate-ms", type=float, default=150.0,
                   help="Stability-gate latency threshold for Table 6.7 (default: 150 ms).")

    p.add_argument("--verbose", action="store_true")
    return p


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> Dict[str, List[Path]]:
    """Build every artifact requested by the parsed args."""
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    )

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    produced: Dict[str, List[Path]] = {}

    # 0) Participant-group registry (healthy vs impaired) + game corpus.
    #    Both are optional and degrade gracefully: a missing registry
    #    just means cohorts fall back to metadata inference, and a
    #    missing game_recordings/ directory simply skips the game report.
    groups_path = args.groups or default_groups_path(args.data_dir)
    groups = ParticipantGroups.from_data_dir(
        args.data_dir,
        filename=groups_path.name,
    ) if args.groups is None else ParticipantGroups.from_file(args.groups)
    if groups.is_empty:
        log.info("No explicit participant-group registry — healthy/impaired "
                 "splits will use session-metadata inference only.")
    else:
        log.info("Participant groups: %s", groups.counts())

    game_corpus = None
    if not args.skip_game:
        game_corpus = GameCorpus(args.data_dir)
        game_corpus.discover(verbose=args.verbose)
        log.info("Discovered %d game recording(s).", len(game_corpus.all()))

    # 1) Corpus overview — §6.1
    log.info("Building corpus overview…")
    corpus = SessionCorpus(args.data_dir)
    corpus.discover(verbose=args.verbose)

    # A cohort fallback that infers group from session/game metadata for
    # any subject the explicit registry doesn't cover.
    group_fallback = metadata_group_resolver(corpus)

    corpus_paths = write_corpus_report(
        corpus, out_dir,
        window_ms=args.window_ms,
        stride_ms=args.stride_ms,
        drop_rest=args.drop_rest,
        groups=groups,
        game_corpus=game_corpus,
    )
    produced["corpus"] = list(corpus_paths.values())

    # 2) Calibration — §6.2
    log.info("Building calibration report…")
    cal_paths = write_calibration_report(
        corpus, out_dir, flag_threshold=args.flag_threshold,
    )
    produced["calibration"] = list(cal_paths.values())

    fig_path = out_dir / "fig_6_1_calibration_confidence"
    cal_fig = plot_calibration_confidence(
        cal_paths["calibration_json"], fig_path,
        threshold=args.flag_threshold,
    )
    produced["fig_6_1"] = cal_fig

    fig_path = out_dir / "fig_6_1b_calibration_honest"
    produced["fig_6_1b"] = plot_calibration_honest(
        cal_paths["calibration_json"], fig_path,
    )

    # 3) Primary LOSO-session run — §6.3 + Table 6.7 (latency)
    if args.primary:
        log.info("Aggregating LOSO-session run %s …", args.primary)
        run_stub = load_run_result(args.primary)
        primary_paths = write_run_report(
            run_stub, out_dir,
            primary_model=args.primary_model,
            gate_ms=args.gate_ms,
        )
        # Rename to the chapter-specific filenames the README documents.
        _renames = [
            ("model_summary_csv",  "table_6_3_loso_session.csv"),
            ("per_class_csv",      "fig_6_3_per_class_f1_data.csv"),
            ("confusion_json",     "fig_6_4_confusion_matrices_data.json"),
            ("per_session_csv",    "fig_6_5_per_session_variability_data.csv"),
            ("latency_csv",        "table_6_7_latency.csv"),
        ]
        for key, target in _renames:
            src = primary_paths[key]
            dst = out_dir / target
            if src.exists() and src != dst:
                src.replace(dst)
                primary_paths[key] = dst
        produced["primary_csvs"] = list(primary_paths.values())

        # §6.3 cohort split — healthy vs impaired LOSO-session means.
        # LOSO-session folds each hold out one subject, so they attribute
        # cleanly to a cohort. Always emitted (the table self-describes
        # an empty cohort) so the thesis can report the gap.
        group_csv = write_group_summary(
            run_stub, groups, out_dir,
            filename="table_6_3b_loso_session_by_group.csv",
            fallback=group_fallback,
        )
        produced["table_6_3b"] = [group_csv]

        # Fig 6.5 box-plot data, annotated with each subject's cohort so
        # the per-session variability plot can be faceted healthy/impaired.
        ps_rows = per_session_variability(run_stub, model=args.primary_model)
        ps_grouped = annotate_per_session_groups(
            ps_rows, groups, fallback=group_fallback,
        )
        ps_grp_csv = out_dir / "fig_6_5_per_session_variability_by_group.csv"
        with ps_grp_csv.open("w", newline="", encoding="utf-8") as f:
            import csv as _csv
            w = _csv.writer(f)
            w.writerow(["subject_id", "group", "model", "fold_id", "macro_f1"])
            for row in ps_grouped:
                for fid, v in zip(row["fold_ids"], row["f1_values"]):
                    w.writerow([row["subject_id"], row["group_label"],
                                row["model_type"], fid, f"{v:.4f}"])
        produced["fig_6_5_by_group"] = [ps_grp_csv]

        produced["fig_6_3"] = plot_per_class_f1(
            primary_paths["per_class_csv"],
            out_dir / "fig_6_3_per_class_f1",
            summary_csv=primary_paths["model_summary_csv"],
            per_fold_csv=primary_paths["per_class_per_fold_csv"],
        )

        produced["fig_6_4"] = plot_confusion_matrices(
            primary_paths["confusion_json"],
            out_dir / "fig_6_4_confusion_matrices",
            summary_csv=primary_paths["model_summary_csv"],
        )

        produced["fig_6_5"] = plot_per_session_variability(
            primary_paths["per_session_csv"],
            out_dir / "fig_6_5_per_session_variability",
            model=args.primary_model,
        )

        # §7.4 — calibration vs F1
        log.info("Joining calibration ↔ F1…")
        stats = calibration_stats(corpus, flag_threshold=args.flag_threshold)
        conf_map = {
            (s.subject_id, s.session_id): s.confidence
            for s in stats.per_session if s.confidence is not None
        }
        corrs = calibration_f1_correlation(run_stub, conf_map)
        corr_json = out_dir / "correlation_calibration_f1.json"
        with corr_json.open("w", encoding="utf-8") as f:
            json.dump({m: {
                "model_type": c.model_type,
                "n_pairs":   c.n_pairs,
                "pearson_r": c.pearson_r,  "pearson_p":  c.pearson_p,
                "spearman_r": c.spearman_r, "spearman_p": c.spearman_p,
                "joined":    [s.__dict__ for s in c.joined],
            } for m, c in corrs.items()}, f, indent=2)
        produced["correlation_json"] = [corr_json]
        if any(c.n_pairs >= 3 for c in corrs.values()):
            produced["fig_7_4"] = plot_calibration_vs_f1(
                corr_json, out_dir / "fig_7_4_calibration_vs_f1",
                model=args.primary_model,
            )
            produced["fig_7_4b"] = plot_calibration_vs_f1_per_model(
                corr_json, out_dir / "fig_7_4b_calibration_vs_f1_per_model",
            )

    # 4) LOSO-subject — Table 6.4
    if args.loso_subj:
        log.info("Aggregating LOSO-subject run %s …", args.loso_subj)
        ls = load_run_result(args.loso_subj)
        summaries = summarise_run(ls)
        import csv
        t4 = out_dir / "table_6_4_loso_subject.csv"
        with t4.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["model", "n_folds", "macro_f1_mean", "macro_f1_std",
                        "accuracy_mean", "accuracy_std"])
            for m, s in sorted(summaries.items(),
                               key=lambda kv: -kv[1].macro_f1_mean):
                w.writerow([m, s.n_folds,
                            f"{s.macro_f1_mean:.4f}", f"{s.macro_f1_std:.4f}",
                            f"{s.accuracy_mean:.4f}", f"{s.accuracy_std:.4f}"])
        produced["table_6_4"] = [t4]

        # §6.4 cohort split — healthy vs impaired LOSO-subject means.
        group_csv = write_group_summary(
            ls, groups, out_dir,
            filename="table_6_4b_loso_subject_by_group.csv",
            fallback=group_fallback,
        )
        produced["table_6_4b"] = [group_csv]

    # 5) Feature ablation — Table 6.5 + Fig 6.6
    ablation_specs = _parse_kv_runs(args.ablation)
    if ablation_specs:
        log.info("Aggregating %d ablation runs…", len(ablation_specs))
        loaded = [(name, load_run_result(path)) for name, path in ablation_specs]
        rows = feature_ablation(loaded, model=args.primary_model)
        ablation_csv = write_feature_ablation(rows, out_dir)
        # Match chapter filename
        target = out_dir / "table_6_5_feature_ablation.csv"
        ablation_csv.replace(target)
        produced["table_6_5"] = [target]
        produced["fig_6_6"] = plot_feature_ablation(
            target, out_dir / "fig_6_6_feature_ablation",
        )

    # 6) Cross-domain — Table 6.6
    xdomain_specs = dict(_parse_kv_runs(args.xdomain))
    if xdomain_specs:
        log.info("Aggregating cross-domain runs…")
        rows = cross_domain_comparison(
            within_pipeline   = load_run_result(xdomain_specs["within_pipe"])
                                if "within_pipe" in xdomain_specs else None,
            within_unity      = load_run_result(xdomain_specs["within_unity"])
                                if "within_unity" in xdomain_specs else None,
            pipeline_to_unity = load_run_result(xdomain_specs["p2u"])
                                if "p2u" in xdomain_specs else None,
            unity_to_pipeline = load_run_result(xdomain_specs["u2p"])
                                if "u2p" in xdomain_specs else None,
            model=args.primary_model,
        )
        xd_csv = write_cross_domain(rows, out_dir)
        target = out_dir / "table_6_6_cross_domain.csv"
        xd_csv.replace(target)
        produced["table_6_6"] = [target]

    # 7) Game-recording performance — healthy vs impaired, multi-class.
    #    Game recordings log the deployed model's predictions against
    #    RawGroundTruth frame by frame, so unlike the RMS-threshold Unity
    #    recordings they carry real multi-class evidence. Scored from the
    #    logged predictions — no retraining — and split by cohort.
    if game_corpus is not None and game_corpus.all():
        log.info("Building game-recording performance report…")
        game_report = build_game_report(
            args.data_dir, game_corpus, groups,
            fallback_resolver=group_fallback,
        )
        game_paths = write_game_report(game_report, out_dir)
        # Rename to chapter-style filenames the README documents.
        _game_renames = [
            ("game_performance_csv", "table_6_8_game_performance.csv"),
            ("game_per_subject_csv", "table_6_9_game_per_subject.csv"),
            ("game_per_class_csv",   "fig_6_7_game_per_class_f1_data.csv"),
            ("game_confusion_json",  "fig_6_8_game_confusion_data.json"),
        ]
        for key, target_name in _game_renames:
            src = game_paths.get(key)
            if src is not None and src.exists():
                dst = out_dir / target_name
                if src != dst:
                    src.replace(dst)
                    game_paths[key] = dst
        produced["game"] = list(game_paths.values())
    elif args.skip_game:
        log.info("Game-recording report skipped (--skip-game).")
    else:
        log.info("No game recordings found — game report skipped.")

    log.info("Wrote %d artefact group(s) to %s", len(produced), out_dir)
    for name, paths in produced.items():
        for p in paths:
            log.info("  • %s : %s", name, p.relative_to(out_dir))
    return produced

def run_in_subprocess(args: argparse.Namespace) -> int:
    """
    Re-launch this module as a completely separate Python process so that
    matplotlib never initialises inside the Qt GUI process.
    """
    import subprocess, sys
    cmd = [sys.executable, "-m",
           "playagain_pipeline.validation.generate_thesis_outputs"]
    # Reconstruct argv from the namespace
    cmd += ["--data-dir", str(args.data_dir)]
    cmd += ["--out",      str(args.out)]
    if args.primary:      cmd += ["--primary",       str(args.primary)]
    if args.loso_subj:    cmd += ["--loso-subj",     str(args.loso_subj)]
    if args.ablation:     cmd += ["--ablation",      args.ablation]
    if args.xdomain:      cmd += ["--xdomain",       args.xdomain]
    cmd += ["--primary-model", args.primary_model]
    cmd += ["--window-ms",     str(args.window_ms)]
    cmd += ["--stride-ms",     str(args.stride_ms)]
    cmd += ["--flag-threshold", str(args.flag_threshold)]
    cmd += ["--gate-ms",       str(args.gate_ms)]
    if args.groups:      cmd += ["--groups", str(args.groups)]
    if args.skip_game:   cmd.append("--skip-game")
    if args.drop_rest:    cmd.append("--drop-rest")
    if args.verbose:      cmd.append("--verbose")

    proc = subprocess.run(cmd, capture_output=False)   # streams to terminal
    return proc.returncode


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run(args)
        return 0
    except FileNotFoundError as exc:
        log.error("Missing input: %s", exc)
        return 2
    except Exception as exc:  # noqa: BLE001
        log.exception("Failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())