"""
evaluation/cross_domain_eval.py
───────────────────────────────
Per-participant, multi-model cross-domain transfer:

    train on a subject's PIPELINE sessions (data/sessions/<subj>/…),
    validate on that subject's GAME recordings (data/game_recordings/<subj>/…).

This replaces the single pooled "pipeline → game_recordings" number the runner
emits (table_6_6: one fold, CatBoost only) with one fold per participant who
contributed BOTH streams, every model in ModelManager.AVAILABLE_MODELS, per-class
F1, and a pooled confusion matrix per model.

Every model (classical AND deep) is built and trained through your own classifier
classes — ``ModelManager.AVAILABLE_MODELS[name]`` → ``.train(raw_windows, y)`` →
``.predict(raw_windows)`` — so the feature extraction, scaling and architectures
match the LOSO pipeline exactly. Raw windows (N, samples, channels) are passed to
both train and predict; each classifier extracts its own features internally.

Outputs (in ``--out``):
  table_6_6_cross_domain.csv              model, n_folds, macro_f1_mean/std, accuracy_mean/std
  table_6_6_cross_domain_per_subject.csv  subject, model, n_test, macro_f1, accuracy
  cross_domain_confusion.json             per-model pooled 4×4 confusion

Run:
  python -m playagain_pipeline.evaluation.cross_domain_eval \
      --data-dir data --out runs/xdomain \
      --models lda svm random_forest catboost mlp cnn attention_net mstnet
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("cross_domain_eval")

# Feature set used by the LOSO runs (experiment.json): 7 custom time-domain
# features. NOTE: the methodology text says 8 features / 256-dim (incl. SSI) but
# experiment.json lists 7 (no SSI) — reconcile which is canonical for the thesis.
DEFAULT_FEATURES = ["mav", "rms", "wl", "zc", "ssc", "var", "iemg"]
DEFAULT_FEATURE_CONFIG = {"mode": "custom", "features": DEFAULT_FEATURES}

WINDOW_MS, STRIDE_MS, FS = 200, 50, 2000
CLASS_LABELS = [0, 1, 2, 3]
CLASS_NAMES = ["Rest", "Fist", "Pinch", "Tripod"]
DEEP_MODELS = ("mlp", "cnn", "attention_net", "mstnet")


# ---------------------------------------------------------------------------
# Model factory — build through your own classifier registry so cross-domain
# matches the LOSO pipeline (same features, scaling, architectures).
# ---------------------------------------------------------------------------
def _build_classifier(name: str, feature_config, hp: dict):
    from playagain_pipeline.models.classifier import ModelManager
    reg = ModelManager.AVAILABLE_MODELS
    if name not in reg:
        raise ValueError(f"Unknown model {name!r}; available: {list(reg)}")
    return reg[name](name=f"{name}_xdomain", feature_config=feature_config, **hp)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class XDomainResult:
    model: str
    per_subject: List[Dict] = field(default_factory=list)
    pooled_confusion: Optional[np.ndarray] = None

    @property
    def macro_f1(self) -> np.ndarray:
        return np.array([r["macro_f1"] for r in self.per_subject], dtype=float)

    @property
    def accuracy(self) -> np.ndarray:
        return np.array([r["accuracy"] for r in self.per_subject], dtype=float)


# ---------------------------------------------------------------------------
# Window builders (raw — each classifier extracts its own features)
# ---------------------------------------------------------------------------
def _session_windows_raw(recs) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Pool RAW (windows, samples, channels) from a subject's pipeline sessions."""
    from playagain_pipeline.evaluation.session_eval import _windows_for_session
    from playagain_pipeline.models.classifier import apply_bad_channel_strategy

    Xs, ys = [], []
    for rec in recs:
        try:
            X, y, _ = _windows_for_session(
                rec, WINDOW_MS, STRIDE_MS, FS,
                bad_channel_mode="interpolate", apply_rotation=True,
                include_invalid=False, feature_extractor=None,   # raw windows
                bad_channel_apply_fn=apply_bad_channel_strategy,
            )
        except Exception as exc:                                  # noqa: BLE001
            log.warning("session %s skipped: %s", getattr(rec, "label", rec), exc)
            continue
        if X.shape[0]:
            Xs.append(X)
            ys.append(y)
    if not Xs:
        return None
    return np.concatenate(Xs), np.concatenate(ys)


def _apply_rotation(emg: np.ndarray, mapping, rot) -> np.ndarray:
    n_ch = emg.shape[1]
    if mapping and len(mapping) == n_ch:
        return emg[:, list(mapping)]
    if rot:
        return emg[:, [(i + int(rot)) % n_ch for i in range(n_ch)]]
    return emg


# gesture name -> training class id (Rest/Fist/Pinch/Tripod = 0/1/2/3)
_GAME_NAME_TO_ID = {"rest": 0, "fist": 1, "pinch": 2, "tripod": 3}


def _game_windows_raw(recs, *, apply_rotation: bool = True, drop_inactive: bool = False
                      ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Pool RAW windows + 4-class labels from a subject's game recordings.

    LABELS: use ``RequestedGesture`` (rest/fist/pinch/tripod), the authoritative
    4-class ground truth. Do NOT use ``RawGroundTruth`` — it is a BINARY
    active/rest flag ({0,1}), and using it as a class id collapses the test set
    to a single class (this was the cause of the ~0.267 "garbage" cross-domain
    number). ``drop_inactive`` therefore defaults to False so Rest windows are
    kept as a real class.

    ROTATION: ``apply_rotation`` rotates the game EMG into the canonical frame
    using each game session's stored channel_mapping, so it matches the rotated
    pipeline training data. The game recorder stores RAW (un-rotated) EMG, so
    apply_rotation=True is correct (verified: it beats False on held-out F1).
    """
    from playagain_pipeline.evaluation.loaders import load_game_csv

    Xs, ys = [], []
    for rec in recs:
        try:
            game = load_game_csv(rec)
        except Exception as exc:                                  # noqa: BLE001
            log.warning("game %s skipped: %s", getattr(rec, "label", rec), exc)
            continue
        emg = game.emg_matrix()
        df = game.df
        if emg.size == 0 or "RequestedGesture" not in df.columns:
            continue

        if apply_rotation:
            cfg = {}
            cp = rec.meta.get("config_path")
            if cp and Path(cp).exists():
                try:
                    cfg = json.loads(Path(cp).read_text()).get("calibration", {}) or {}
                except Exception:                                 # noqa: BLE001
                    cfg = {}
            emg = _apply_rotation(emg, cfg.get("channel_mapping"), cfg.get("rotation_offset", 0))

        fs = int(game.sampling_rate or FS)
        win = max(1, round(WINDOW_MS * fs / 1000))
        stride = max(1, round(STRIDE_MS * fs / 1000))
        if emg.shape[0] < win:
            continue
        starts = np.arange(0, emg.shape[0] - win + 1, stride, dtype=np.int64)
        centres = np.clip(starts + win // 2, 0, len(df) - 1)
        req = (df["RequestedGesture"].astype(str).str.strip().str.lower()
               .to_numpy())[centres]
        y = np.array([_GAME_NAME_TO_ID.get(n, -1) for n in req], dtype=np.int64)
        keep = y >= 0
        if drop_inactive and "GroundTruthActive" in df.columns:
            keep &= df["GroundTruthActive"].to_numpy(dtype=np.int64)[centres] > 0
        if not keep.any():
            continue
        X = np.stack([emg[s:s + win] for s in starts]).astype(np.float32)[keep]
        Xs.append(X)
        ys.append(y[keep])
    if not Xs:
        return None
    return np.concatenate(Xs), np.concatenate(ys)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_cross_domain(
    data_dir: Path,
    out_dir: Path,
    model_names: List[str],
    feature_config: Optional[dict] = DEFAULT_FEATURE_CONFIG,
    hyperparameters: Optional[Dict[str, dict]] = None,
    game_emg_is_raw: bool = True,
) -> Dict[str, XDomainResult]:
    from sklearn.metrics import confusion_matrix, f1_score
    from sklearn.model_selection import train_test_split
    from playagain_pipeline.evaluation.loaders import (
        discover_game_recordings, discover_sessions,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hyperparameters = hyperparameters or {}

    by_sess: Dict[str, list] = {}
    by_game: Dict[str, list] = {}
    for r in discover_sessions(data_dir):
        by_sess.setdefault(r.subject_id, []).append(r)
    for r in discover_game_recordings(data_dir):
        by_game.setdefault(r.subject_id, []).append(r)
    subjects = sorted(set(by_sess) & set(by_game))
    log.info("Cross-domain folds (subjects with both streams): %s", subjects)

    results = {m: XDomainResult(m) for m in model_names}
    pooled_true = {m: [] for m in model_names}
    pooled_pred = {m: [] for m in model_names}

    for subj in subjects:
        tr = _session_windows_raw(by_sess[subj])
        te = _game_windows_raw(by_game[subj], apply_rotation=game_emg_is_raw)
        if tr is None or te is None:
            log.warning("skip %s — no usable train/test windows", subj)
            continue
        Xtr_raw, ytr = tr
        Xte_raw, yte = te
        if len(np.unique(ytr)) < 2 or Xte_raw.shape[0] == 0:
            log.warning("skip %s — degenerate fold", subj)
            continue

        for m in model_names:
            try:
                clf = _build_classifier(m, feature_config, hyperparameters.get(m, {}))
                # Deep nets get a stratified val split for early stopping (mirrors
                # ModelManager.train_model); classical models train on all windows.
                if m.lower() in DEEP_MODELS:
                    try:
                        Xt, Xv, yt, yv = train_test_split(
                            Xtr_raw, ytr, test_size=0.2, stratify=ytr, random_state=42)
                    except ValueError:
                        Xt, Xv, yt, yv = Xtr_raw, None, ytr, None
                    clf.train(Xt, yt, Xv, yv, window_size_ms=WINDOW_MS,
                              sampling_rate=FS, num_channels=Xtr_raw.shape[2])
                else:
                    clf.train(Xtr_raw, ytr, window_size_ms=WINDOW_MS,
                              sampling_rate=FS, num_channels=Xtr_raw.shape[2])
                y_pred = np.asarray(clf.predict(Xte_raw)).astype(int)
            except Exception as exc:                              # noqa: BLE001
                log.exception("%s failed on %s: %s", m, subj, exc)
                continue

            mf1 = float(f1_score(yte, y_pred, labels=CLASS_LABELS,
                                 average="macro", zero_division=0))
            pcf1 = f1_score(yte, y_pred, labels=CLASS_LABELS,
                            average=None, zero_division=0)
            results[m].per_subject.append({
                "subject": subj,
                "n_test": int(yte.size),
                "macro_f1": mf1,
                "accuracy": float(np.mean(y_pred == yte)),
                "per_class_f1": {CLASS_NAMES[i]: float(pcf1[i]) for i in range(4)},
            })
            pooled_true[m].append(yte)
            pooled_pred[m].append(y_pred)
            log.info("  %-14s %-7s  macro_f1=%.3f  (n_test=%d)", m, subj, mf1, yte.size)

    for m in model_names:
        if pooled_true[m]:
            yt = np.concatenate(pooled_true[m])
            yp = np.concatenate(pooled_pred[m])
            results[m].pooled_confusion = confusion_matrix(yt, yp, labels=CLASS_LABELS)

    _write_outputs(out_dir, results)
    return results


def _write_outputs(out_dir: Path, results: Dict[str, XDomainResult]) -> None:
    with (out_dir / "table_6_6_cross_domain.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["train_domain", "test_domain", "model", "n_folds",
                    "macro_f1_mean", "macro_f1_std", "accuracy_mean", "accuracy_std"])
        for m, r in results.items():
            v, a = r.macro_f1, r.accuracy
            if v.size == 0:
                continue
            sd = lambda x: f"{x.std(ddof=1):.4f}" if x.size > 1 else "0.0000"
            w.writerow(["pipeline", "game_recordings", m, v.size,
                        f"{v.mean():.4f}", sd(v), f"{a.mean():.4f}", sd(a)])

    with (out_dir / "table_6_6_cross_domain_per_subject.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "model", "n_test", "macro_f1", "accuracy"])
        for m, r in results.items():
            for row in r.per_subject:
                w.writerow([row["subject"], m, row["n_test"],
                            f"{row['macro_f1']:.4f}", f"{row['accuracy']:.4f}"])

    conf = {m: (r.pooled_confusion.tolist() if r.pooled_confusion is not None else None)
            for m, r in results.items()}
    (out_dir / "cross_domain_confusion.json").write_text(
        json.dumps({"labels": CLASS_NAMES, "confusion": conf}, indent=2))
    log.info("wrote cross-domain tables to %s", out_dir)


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Per-participant cross-domain transfer.")
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("runs/xdomain"))
    ap.add_argument("--models", nargs="+",
                    default=["lda", "svm", "random_forest", "catboost",
                             "mlp", "cnn", "attention_net", "mstnet"])
    ap.add_argument("--game-emg-rotated", action="store_true",
                    help="set if game CSVs already store calibrated/rotated channels "
                         "(disables the rotation step to avoid double-rotating).")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    run_cross_domain(args.data_dir, args.out, args.models,
                     game_emg_is_raw=not args.game_emg_rotated)


if __name__ == "__main__":
    _cli()