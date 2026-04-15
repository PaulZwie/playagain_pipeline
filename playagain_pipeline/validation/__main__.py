"""
__main__.py
───────────
Run with:

    python -m playagain_pipeline.validation run experiments/baseline.yaml
    python -m playagain_pipeline.validation list
    python -m playagain_pipeline.validation summary

`run`     executes a single experiment YAML against the corpus.
`list`    prints every discovered session and its source domain.
`summary` prints a short corpus summary (counts per domain, subjects).

The CLI is intentionally minimal — anything more complex belongs in a
script that imports the package directly.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import load_experiment
from .corpus import SessionCorpus
from .runner import ValidationRunner


def _default_data_dir() -> Path:
    """Walk up from this file to find playagain_pipeline/data."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "data"
        if cand.exists() and (cand / "sessions").exists():
            return cand
    # Fallback: cwd/data
    return Path.cwd() / "data"


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s :: %(message)s",
    )

    parser = argparse.ArgumentParser(prog="playagain-validation")
    parser.add_argument(
        "--data-dir", type=Path, default=_default_data_dir(),
        help="Pipeline data directory (default: auto-detected)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Run an experiment from a YAML/JSON file")
    run_p.add_argument("config", type=Path, help="Path to experiment YAML/JSON")
    run_p.add_argument("--output-root", type=Path, default=None)

    sub.add_parser("list",    help="List every discovered session")
    sub.add_parser("summary", help="Print corpus summary")

    args = parser.parse_args(argv)

    if args.cmd == "summary":
        corpus = SessionCorpus(args.data_dir)
        print(corpus.summary())
        return 0

    if args.cmd == "list":
        corpus = SessionCorpus(args.data_dir)
        for r in corpus.all():
            print(f"{r.source_domain:9s}  {r.subject_id:12s}  {r.session_id}")
        return 0

    if args.cmd == "run":
        cfg = load_experiment(args.config)
        runner = ValidationRunner(args.data_dir, args.output_root)
        result = runner.run(cfg)
        print(f"\n  Wrote: {result.output_dir}")
        agg = result.aggregate()
        for model, metrics in agg.items():
            print(f"  {model:18s}  acc={metrics['accuracy_mean']:.3f}±{metrics['accuracy_std']:.3f}  "
                  f"f1={metrics['macro_f1_mean']:.3f}±{metrics['macro_f1_std']:.3f}  "
                  f"(n={metrics['n_folds']})")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
