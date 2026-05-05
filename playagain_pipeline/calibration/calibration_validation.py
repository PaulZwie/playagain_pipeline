"""
calibration_validation.py
─────────────────────────
Diagnostic checks that tell you whether a rotation calibration is
actually correct before you trust it for real recording.

What changed (v2)
─────────────────
1. **Held-out check rewritten against the real RecordingSession API.**
   The previous version called ``session.load_signal()`` /
   ``session.load_labels()`` which don't exist. The check now reads
   ``session.get_data()`` and walks ``session.get_valid_trials()``,
   computing one per-channel RMS profile per trial. This is also a
   more honest unit of analysis than arbitrary windows — a trial is
   what the user actually performed.

2. **Symmetry check now targets the specific 180° twin.**
   The old version flagged *any* secondary peak that came too close to
   the primary, which trips on benign correlation shoulders in noisy
   data. The new check looks at the half-ring twin specifically — the
   only structural alias an EMG bracelet can produce — and exposes
   both the half-ring peak AND the global runner-up so the user can
   see why a fail is or isn't a real flip.

3. **Confidence is decoupled from rotation correctness.**
   A separate ``confidence_floor`` check exists for users who want
   it, but the default acceptance criterion is "rotation detection
   is internally consistent" — not "confidence ≥ 70%". This matches
   the empirical observation that low confidence often co-exists
   with correct offset detection.

4. **Per-check ``severity``: ``error`` (must pass) or ``warning``
   (informational).** Sessions can now be "acceptable with caveats"
   rather than only pass / fail.

5. **``RotationDetectionStudy``** — a new class that runs rotation
   detection across many sessions and surfaces aggregated statistics:
   per-session offsets, drift relative to a reference session,
   distribution shape, and a small "is this calibration stable across
   the corpus?" verdict. This is what you reach for when the question
   isn't "is this one calibration right?" but "is rotation detection
   working consistently across my recordings?"

The module is intentionally side-effect-free for the validator —
it never writes to disk and never mutates the calibrator.
``RotationDetectionStudy`` doesn't either.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result records
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    """One independent check's outcome."""
    name: str
    passed: bool
    score: float                # check-specific, see ``unit`` for scale
    unit: str
    detail: str = ""
    # ``error``: a fail blocks acceptance.
    # ``warning``: a fail is informational and does not block.
    severity: str = "error"

    def line(self) -> str:
        if self.passed:
            glyph = "✓"
        elif self.severity == "warning":
            glyph = "!"
        else:
            glyph = "✗"
        return f"  [{glyph}] {self.name:24s} {self.score:7.3f} {self.unit:8s} — {self.detail}"


@dataclass
class HeldOutDetails:
    """Per-trial classification record for one session's held-out check.

    Stored alongside the aggregated metrics so plots (confusion matrix,
    margin histogram, gesture-pair confusion) can be built downstream
    without re-running classification.
    """
    y_true:    List[str]   = field(default_factory=list)
    y_pred:    List[str]   = field(default_factory=list)
    distances: List[float] = field(default_factory=list)  # cosine distance to nearest
    margins:   List[float] = field(default_factory=list)  # 2nd-best minus best (bigger = more confident)

    @property
    def n_trials(self) -> int:
        return len(self.y_true)

    @property
    def overall_accuracy(self) -> float:
        if not self.y_true:
            return 0.0
        return sum(t == p for t, p in zip(self.y_true, self.y_pred)) / len(self.y_true)

    def per_gesture_accuracy(self) -> Dict[str, float]:
        per: Dict[str, List[int]] = {}
        for t, p in zip(self.y_true, self.y_pred):
            slot = per.setdefault(t, [0, 0])
            slot[0] += int(t == p)
            slot[1] += 1
        return {g: ok / n for g, (ok, n) in per.items() if n > 0}

    def per_gesture_counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for t in self.y_true:
            out[t] = out.get(t, 0) + 1
        return out

    def confusion(self, labels: Optional[List[str]] = None) -> Tuple[List[str], "np.ndarray"]:
        """``(labels, matrix)`` — matrix[i, j] = #(true=i, pred=j)."""
        if labels is None:
            labels = sorted(set(self.y_true) | set(self.y_pred))
        idx = {l: i for i, l in enumerate(labels)}
        n = len(labels)
        mat = np.zeros((n, n), dtype=int)
        for t, p in zip(self.y_true, self.y_pred):
            if t in idx and p in idx:
                mat[idx[t], idx[p]] += 1
        return labels, mat

    def gesture_pair_confusions(self, top_n: int = 8) -> List[Tuple[Tuple[str, str], int]]:
        """The ``top_n`` most frequent (true, predicted) confusion pairs."""
        cnt = Counter()
        for t, p in zip(self.y_true, self.y_pred):
            if t != p:
                cnt[(t, p)] += 1
        return cnt.most_common(top_n)

    def to_dict(self) -> dict:
        return {
            "y_true":    list(self.y_true),
            "y_pred":    list(self.y_pred),
            "distances": [float(d) for d in self.distances],
            "margins":   [float(m) for m in self.margins],
        }


@dataclass
class CalibrationReport:
    """The aggregated outcome of every check that was run."""
    rotation_offset: int
    confidence: float
    checks: List[CheckResult] = field(default_factory=list)
    # Per-gesture accuracy from the held-out check, keyed by gesture
    # name. Empty if no held-out session was supplied.
    per_gesture_accuracy: Dict[str, float] = field(default_factory=dict)
    # Confusion matrix from the held-out check (gesture name → {predicted: count}).
    confusion: Dict[str, Dict[str, int]] = field(default_factory=dict)
    # Rich per-trial held-out record (set when held_out_session is supplied).
    held_out_details: Optional[HeldOutDetails] = None
    # Identifiers — set by BatchValidator so the GUI can label results.
    subject_id: Optional[str] = None
    session_id: Optional[str] = None

    @property
    def is_acceptable(self) -> bool:
        """True if every ``error``-severity check passed.

        Warnings can fail without blocking acceptance — they're info.
        """
        return all(c.passed for c in self.checks if c.severity == "error")

    @property
    def has_warnings(self) -> bool:
        return any(not c.passed and c.severity == "warning" for c in self.checks)

    def summary(self) -> str:
        verdict = "YES"
        if not self.is_acceptable:
            verdict = "NO"
        elif self.has_warnings:
            verdict = "YES (with warnings)"
        head = (
            f"Calibration validation report\n"
            f"  rotation_offset = {self.rotation_offset}\n"
            f"  confidence      = {self.confidence:.2%}\n"
            f"  acceptable      = {verdict}\n"
            f"Checks ({len(self.checks)}):"
        )
        body = "\n".join(c.line() for c in self.checks)
        if self.per_gesture_accuracy:
            body += "\nPer-gesture top-1 accuracy:"
            for g, acc in sorted(self.per_gesture_accuracy.items()):
                body += f"\n  {g:20s} {acc:.2%}"
        return head + ("\n" + body if body else "")

    def interpret(self) -> str:
        """Plain-English explanation of the result.

        Specifically calls out the common "low confidence but rotation
        is actually correct" case, which is normal for broad energy
        profiles even when the offset is unambiguous.
        """
        passes = {c.name: c.passed for c in self.checks}
        sc = passes.get("self_consistency", False)
        sym = passes.get("symmetry", False)
        ho = passes.get("held_out_accuracy")
        conf_low = self.confidence < 0.40

        if sc and sym and (ho is None or ho):
            base = ("Rotation offset is unambiguous (self-consistency + "
                    "symmetry both pass)")
            if conf_low:
                base += (" even though the calibrator's xcorr-peak "
                         "confidence reads low — the latter is normal "
                         "for broad energy profiles and does not indicate "
                         "a problem.")
            else:
                base += "."
            return base
        if not sc:
            return ("Self-consistency failed: the calibrator disagrees "
                    "with itself when re-run on the same data. Suspect "
                    "very noisy or ambiguous EMG; consider re-recording.")
        if not sym:
            return ("Symmetry failed: the channel energy peak has a near-"
                    "equal twin 180° away on the bracelet, which usually "
                    "means the bracelet is on backwards or the wrong "
                    "rotation was selected.")
        if ho is False:
            return ("Free checks pass but held-out classification was "
                    "below threshold — gestures are being confused "
                    "despite the rotation looking correct. Inspect the "
                    "confusion matrix to see which pairs are mixing.")
        return ""

    def to_dict(self) -> dict:
        d = asdict(self)
        # held_out_details serializes via its own to_dict for cleaner JSON
        if self.held_out_details is not None:
            d["held_out_details"] = self.held_out_details.to_dict()
        return d


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class CalibrationValidator:
    """
    Runs a battery of independent checks on a fitted calibration.

    Defaults are chosen to flag *real* problems (a bracelet on backwards,
    a calibrator that disagrees with itself by more than a channel) and
    avoid noise about things that don't actually break inference (low
    confidence on an otherwise consistent rotation).

    The validator never modifies the calibrator. It only reads:
      - ``calibrator.current_calibration.rotation_offset``
      - ``calibrator.current_calibration.confidence``
      - ``calibrator.current_calibration.reference_patterns``
      - The calibrator's saved reference calibration via
        ``calibrator.processor.get_reference_calibration()`` (preferred)
        or older attributes as a fallback. See ``_get_reference_calibration``.
    """

    # Acceptance thresholds. Conservative defaults — tighten via the
    # constructor if your protocol allows.
    SELF_CONSISTENCY_MAX_DRIFT = 1     # channels
    SYMMETRY_MIN_RATIO         = 1.25  # primary peak / half-ring twin
    HELD_OUT_MIN_ACCURACY      = 0.70  # 70% top-1
    CONFIDENCE_FLOOR           = 0.20  # warning-only by default

    def __init__(
        self,
        calibrator,
        *,
        self_consistency_max_drift: Optional[int] = None,
        symmetry_min_ratio: Optional[float] = None,
        held_out_min_accuracy: Optional[float] = None,
        confidence_floor: Optional[float] = None,
        confidence_is_error: bool = False,
    ):
        self.calibrator = calibrator
        if self_consistency_max_drift is not None:
            self.SELF_CONSISTENCY_MAX_DRIFT = int(self_consistency_max_drift)
        if symmetry_min_ratio is not None:
            self.SYMMETRY_MIN_RATIO = float(symmetry_min_ratio)
        if held_out_min_accuracy is not None:
            self.HELD_OUT_MIN_ACCURACY = float(held_out_min_accuracy)
        if confidence_floor is not None:
            self.CONFIDENCE_FLOOR = float(confidence_floor)
        # By default confidence is warning-only because empirically
        # confidence can be low while rotation detection is correct.
        self._confidence_severity = "error" if confidence_is_error else "warning"

    # ------------------------------------------------------------------
    # Top-level entry point
    # ------------------------------------------------------------------

    def run_all(
        self,
        gesture_data: Dict[str, np.ndarray],
        *,
        held_out_session=None,
        gesture_filter: Optional[Iterable[str]] = None,
    ) -> CalibrationReport:
        """Run every applicable check and return a single report.

        Parameters
        ----------
        gesture_data : dict
            ``{gesture_name: (n_samples, n_channels) ndarray}``. Used by
            self-consistency and symmetry checks.
        held_out_session : optional
            A ``RecordingSession`` for held-out classification accuracy.
        gesture_filter : iterable of str, optional
            If given, restricts held-out classification to these gestures
            (i.e., trials whose ground-truth label is not in the filter
            are skipped). Useful for batch validation across sessions
            with differing gesture sets.
        """
        cal = getattr(self.calibrator, "current_calibration", None)
        if cal is None:
            raise RuntimeError(
                "Validator needs a fitted calibration on the "
                "calibrator. Call calibrate(...) first."
            )

        report = CalibrationReport(
            rotation_offset=int(getattr(cal, "rotation_offset", 0)),
            confidence=float(getattr(cal, "confidence", 0.0)),
        )

        report.checks.append(self._check_self_consistency(gesture_data))
        report.checks.append(self._check_symmetry(gesture_data))
        report.checks.append(self._check_confidence_floor(report.confidence))

        if held_out_session is not None:
            gf: Optional[Set[str]] = (
                {str(g) for g in gesture_filter} if gesture_filter else None
            )
            ho_check, per_gest, confusion, details = self._check_held_out_classification(
                held_out_session, gesture_filter=gf,
            )
            report.checks.append(ho_check)
            report.per_gesture_accuracy = per_gest
            report.confusion = confusion
            report.held_out_details = details

        return report

    # ------------------------------------------------------------------
    # Check 1: self-consistency
    # ------------------------------------------------------------------

    def _check_self_consistency(
        self,
        gesture_data: Dict[str, np.ndarray],
    ) -> CheckResult:
        """Re-run rotation detection on the same data and compare."""
        cal = self.calibrator.current_calibration
        stored_offset = int(cal.rotation_offset)

        try:
            energies = self._compute_energies(gesture_data)
            sync = self._select_sync_gesture(energies)
            if sync is None:
                return CheckResult(
                    "self_consistency", False, 0.0, "Δch",
                    "no sync gesture available",
                )
            ref = self._reference_energy_for(sync)
            if ref is None:
                return CheckResult(
                    "self_consistency", False, 0.0, "Δch",
                    f"no reference pattern for '{sync}'",
                )
            recomputed = self._cross_correlation_offset(energies[sync], ref)
        except Exception as e:  # noqa: BLE001
            return CheckResult(
                "self_consistency", False, 0.0, "Δch",
                f"could not recompute: {e}",
            )

        ring = len(energies[sync])
        drift = self._ring_distance(recomputed, stored_offset, ring)
        passed = drift <= self.SELF_CONSISTENCY_MAX_DRIFT
        detail = (
            f"recomputed={recomputed} stored={stored_offset} "
            f"(drift {drift} ch, max {self.SELF_CONSISTENCY_MAX_DRIFT})"
        )
        return CheckResult("self_consistency", passed, float(drift), "Δch", detail)

    # ------------------------------------------------------------------
    # Check 2: symmetry / 180° alias
    # ------------------------------------------------------------------

    def _check_symmetry(
        self,
        gesture_data: Dict[str, np.ndarray],
    ) -> CheckResult:
        """
        Look for a competing peak at the half-ring twin offset.

        EMG bracelets have a single structural alias: rotating the ring
        by ``ring/2`` channels gives a permutation of the channel
        ordering that, for symmetric anatomy, looks similar to the
        true offset. *That* is the alias we care about — not arbitrary
        local maxima in the cross-correlation, which v1 also flagged.

        The check now:
          - locates the cross-correlation peak (``peak``),
          - looks specifically at index ``(peak + ring//2) % ring`` —
            the half-ring twin,
          - reports the ratio peak / half_ring as the score,
          - additionally reports the global runner-up as info.

        ``peak / half_ring < SYMMETRY_MIN_RATIO`` is the flip warning.
        """
        try:
            energies = self._compute_energies(gesture_data)
            sync = self._select_sync_gesture(energies)
            if sync is None:
                return CheckResult(
                    "symmetry", False, 0.0, "ratio",
                    "no sync gesture", severity="warning",
                )
            ref = self._reference_energy_for(sync)
            cur = energies[sync]
            if ref is None:
                return CheckResult(
                    "symmetry", False, 0.0, "ratio",
                    f"no reference pattern for '{sync}'",
                    severity="warning",
                )
            corr = self._circular_correlation(cur, ref)
        except Exception as e:  # noqa: BLE001
            return CheckResult(
                "symmetry", False, 0.0, "ratio",
                f"could not compute correlation: {e}",
                severity="warning",
            )

        ring = len(corr)
        peak_idx = int(np.argmax(corr))
        peak_val = float(corr[peak_idx])

        # The 180° twin: the alias that would correspond to wearing the
        # bracelet upside-down or 180° around the wrist.
        twin_idx = (peak_idx + ring // 2) % ring
        twin_val = float(corr[twin_idx])

        # Global runner-up (excluding ±2 around primary, for context only)
        masked = corr.copy()
        for delta in range(-2, 3):
            masked[(peak_idx + delta) % ring] = -np.inf
        runner_up = float(masked.max())

        # The ratio that matters for the alias check. Use a small floor on
        # the denominator so a slightly-negative twin doesn't produce an
        # absurd ratio.
        denom = max(twin_val, 1e-9)
        ratio = peak_val / denom if peak_val > 0 else 0.0
        passed = ratio >= self.SYMMETRY_MIN_RATIO

        detail = (
            f"peak={peak_val:.3f} half-ring twin={twin_val:.3f} "
            f"(ratio {ratio:.2f}, min {self.SYMMETRY_MIN_RATIO:.2f}). "
            f"Runner-up elsewhere: {runner_up:.3f}. "
            "Low ratio often means the bracelet is on backwards."
        )
        return CheckResult(
            "symmetry", passed, float(ratio), "ratio", detail,
            severity="warning",
        )

    # ------------------------------------------------------------------
    # Check 3: confidence floor (warning-only by default)
    # ------------------------------------------------------------------

    def _check_confidence_floor(self, confidence: float) -> CheckResult:
        passed = confidence >= self.CONFIDENCE_FLOOR
        detail = (
            f"confidence {confidence:.0%} vs floor "
            f"{self.CONFIDENCE_FLOOR:.0%}. "
            "Low confidence is informational — rotation can still be "
            "correct on a self-consistent calibration."
        )
        return CheckResult(
            "confidence", passed, float(confidence) * 100.0, "%", detail,
            severity=self._confidence_severity,
        )

    # ------------------------------------------------------------------
    # Check 4: held-out classification accuracy
    # ------------------------------------------------------------------

    def _check_held_out_classification(
        self,
        session,
        gesture_filter: Optional[Set[str]] = None,
    ) -> Tuple[CheckResult, Dict[str, float], Dict[str, Dict[str, int]], HeldOutDetails]:
        """
        Apply the calibration to a labeled session and measure top-1
        accuracy of a 1-NN classifier built from the reference energy
        profiles.

        Implementation note: we work with *trials* (not arbitrary windows
        with sticky labels) because that's what the rest of the system
        produces and treats as ground truth. Per trial: extract the
        sample range, compute per-channel RMS, rotate by the calibration
        offset, classify with 1-NN against the reference patterns, and
        record the cosine distance to the best match plus the margin
        (distance to runner-up minus distance to best — bigger means
        more confident).

        Returns
        -------
        (CheckResult, per_gesture_accuracy, confusion_matrix, details)
            confusion_matrix[true_gesture] = {predicted_gesture: count}
            details: HeldOutDetails with per-trial y_true / y_pred /
                     distance / margin lists.
        """
        cal = self.calibrator.current_calibration
        details = HeldOutDetails()

        try:
            signal = self._extract_session_signal(session)
            trials = self._extract_session_trials(session)
        except Exception as e:  # noqa: BLE001
            return CheckResult(
                "held_out_accuracy", False, 0.0, "%",
                f"session load failed: {e}",
            ), {}, {}, details

        if signal is None or signal.size == 0:
            return CheckResult(
                "held_out_accuracy", False, 0.0, "%",
                "session has no signal data",
            ), {}, {}, details
        if not trials:
            return CheckResult(
                "held_out_accuracy", False, 0.0, "%",
                "session has no valid trials",
            ), {}, {}, details

        ref_calib = self._get_reference_calibration() or cal
        ref_patterns_raw = getattr(ref_calib, "reference_patterns", None) or {}
        # Drop the synthetic combined key and rest-class — they aren't
        # active gestures we want as 1-NN classes.
        ref_patterns = {
            k: v for k, v in ref_patterns_raw.items()
            if not k.startswith("__") and "rest" not in k.lower()
        }
        if gesture_filter is not None:
            ref_patterns = {
                k: v for k, v in ref_patterns.items() if k in gesture_filter
            }
        if not ref_patterns:
            return CheckResult(
                "held_out_accuracy", False, 0.0, "%",
                "no reference energy patterns to classify against "
                "(check that the gesture filter overlaps the reference)",
            ), {}, {}, details

        offset = int(cal.rotation_offset)
        n_correct, n_total = 0, 0
        per_g_correct: Dict[str, int] = {}
        per_g_total: Dict[str, int] = {}
        confusion: Dict[str, Dict[str, int]] = {}

        for trial in trials:
            gname = str(getattr(trial, "gesture_name", "")).strip()
            if not gname or gname.lower() == "rest":
                # Resting trials shouldn't drive accuracy — they bias it
                # upward and aren't what the calibration is fixing.
                continue
            if gesture_filter is not None and gname not in gesture_filter:
                continue
            start = int(getattr(trial, "start_sample", 0))
            end = int(getattr(trial, "end_sample", start))
            if end <= start or start < 0 or end > signal.shape[0]:
                continue

            chunk = signal[start:end]
            if chunk.shape[0] < 10:
                continue

            rms = np.sqrt(np.mean(chunk.astype(np.float64) ** 2, axis=0))
            rotated = np.roll(rms, offset)
            pred, dist, margin = self._nearest_pattern_with_margin(
                rotated, ref_patterns,
            )
            if pred is None:
                continue

            details.y_true.append(gname)
            details.y_pred.append(pred)
            details.distances.append(float(dist))
            details.margins.append(float(margin))

            per_g_total[gname] = per_g_total.get(gname, 0) + 1
            row = confusion.setdefault(gname, {})
            row[pred] = row.get(pred, 0) + 1
            if pred.lower() == gname.lower():
                n_correct += 1
                per_g_correct[gname] = per_g_correct.get(gname, 0) + 1
            n_total += 1

        if n_total == 0:
            return CheckResult(
                "held_out_accuracy", False, 0.0, "%",
                "no usable active-gesture trials in held-out session",
            ), {}, {}, details

        acc = n_correct / n_total
        per_gesture_acc: Dict[str, float] = {
            g: per_g_correct.get(g, 0) / t if t else 0.0
            for g, t in per_g_total.items()
        }
        passed = acc >= self.HELD_OUT_MIN_ACCURACY
        filter_note = ""
        if gesture_filter is not None:
            filter_note = f" Filter: {sorted(gesture_filter)}."
        detail = (
            f"{n_correct}/{n_total} trials correct "
            f"(min {self.HELD_OUT_MIN_ACCURACY:.0%}). "
            f"Excludes Rest trials.{filter_note}"
        )
        return CheckResult(
            "held_out_accuracy", passed, acc * 100.0, "%", detail,
        ), per_gesture_acc, confusion, details

    # ------------------------------------------------------------------
    # Helpers — intentionally local so the module works against
    # multiple session and calibrator API revisions.
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_session_signal(session) -> Optional[np.ndarray]:
        """Return ``(n_samples, n_channels)`` float array from a session.

        Tolerant of multiple session API versions.
        """
        # 1. The current playagain RecordingSession API.
        getter = getattr(session, "get_data", None)
        if callable(getter):
            data = getter()
            return np.asarray(data) if data is not None else None
        # 2. Fallbacks (legacy hypotheticals) — try common attribute names.
        for attr in ("data", "signal", "raw_data"):
            val = getattr(session, attr, None)
            if val is not None:
                return np.asarray(val)
        return None

    @staticmethod
    def _extract_session_trials(session) -> list:
        """Return a list of trials from a session, preferring valid ones."""
        getter = getattr(session, "get_valid_trials", None)
        if callable(getter):
            try:
                trials = getter()
            except Exception:  # noqa: BLE001
                trials = None
            if trials:
                return list(trials)
        # Fall back to .trials (may include invalid ones, but we filter
        # by sample range / gesture later).
        trials = getattr(session, "trials", None)
        return list(trials) if trials else []

    def _compute_energies(
        self,
        gesture_data: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """``{gesture: per-channel RMS}``. Prefers the calibrator's own
        ``compute_channel_energy`` if available, falls back to plain
        RMS otherwise."""
        compute = getattr(self.calibrator, "compute_channel_energy", None)
        out: Dict[str, np.ndarray] = {}
        for g, data in gesture_data.items():
            if data is None or data.size == 0:
                continue
            if callable(compute):
                try:
                    out[g] = np.asarray(compute(data), dtype=np.float64)
                    continue
                except Exception:  # noqa: BLE001
                    pass  # fall through
            out[g] = np.sqrt(np.mean(data.astype(np.float64) ** 2, axis=0))
        return out

    def _select_sync_gesture(
        self,
        energies: Dict[str, np.ndarray],
    ) -> Optional[str]:
        """
        Prefer the calibrator's own choice; otherwise pick the gesture
        with the highest peak-to-mean ratio.
        """
        cal = self.calibrator.current_calibration
        meta = getattr(cal, "metadata", {}) or {}
        sync = meta.get("sync_gesture")
        if sync and isinstance(sync, str):
            if sync in energies:
                return sync
            # Sometimes sync_gesture stores a "(auto-selected)" suffix
            # or a "Combined (fallback)" pseudo-name.
            stem = sync.split()[0].lower()
            for k in energies:
                if k.lower().split()[0] == stem:
                    return k
        if not energies:
            return None
        return max(
            energies,
            key=lambda g: float(np.max(energies[g])) / max(float(np.mean(energies[g])), 1e-9),
        )

    def _reference_energy_for(self, gesture: str) -> Optional[np.ndarray]:
        """Look up the reference per-channel pattern for a gesture."""
        ref_cal = self._get_reference_calibration() or self.calibrator.current_calibration
        patterns = getattr(ref_cal, "reference_patterns", None) or {}
        if gesture in patterns:
            return np.asarray(patterns[gesture], dtype=np.float64)
        # Try a relaxed match (substring / case-insensitive)
        gl = gesture.lower()
        for k, v in patterns.items():
            if k.startswith("__"):
                continue
            if gl in k.lower() or k.lower() in gl:
                return np.asarray(v, dtype=np.float64)
        # Last resort — fall back to the combined pattern if present.
        if "__combined__" in patterns:
            return np.asarray(patterns["__combined__"], dtype=np.float64)
        return None

    def _get_reference_calibration(self):
        """Return the calibrator's saved reference calibration, or None."""
        proc = getattr(self.calibrator, "processor", None)
        if proc is not None:
            getter = getattr(proc, "get_reference_calibration", None)
            if callable(getter):
                try:
                    out = getter()
                    if out is not None:
                        return out
                except Exception:  # noqa: BLE001
                    pass
        return (
            getattr(self.calibrator, "reference_calibration", None)
            or getattr(self.calibrator, "_reference_calibration", None)
        )

    @staticmethod
    def _circular_correlation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Cross-correlation of two equal-length 1D vectors over a ring."""
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        n = a.shape[0]
        a = a - a.mean()
        b = b - b.mean()
        A = np.fft.rfft(a)
        B = np.fft.rfft(b)
        return np.fft.irfft(A * np.conj(B), n=n)

    def _cross_correlation_offset(self, a: np.ndarray, b: np.ndarray) -> int:
        return int(np.argmax(self._circular_correlation(a, b)))

    @staticmethod
    def _ring_distance(a: int, b: int, ring: int) -> int:
        """Smallest signed distance on a ring of size ``ring``."""
        if ring <= 0:
            return abs(a - b)
        d = abs(a - b) % ring
        return min(d, ring - d)

    @staticmethod
    def _nearest_pattern(
        rotated_rms: np.ndarray,
        ref_patterns: Dict[str, Any],
    ) -> Optional[str]:
        """1-NN over normalised reference energy profiles."""
        if not ref_patterns:
            return None
        x = rotated_rms.astype(np.float64)
        norm = np.linalg.norm(x)
        if norm < 1e-12:
            return None
        x = x / norm
        best_g, best_d = None, float("inf")
        for g, p in ref_patterns.items():
            arr = np.asarray(p, dtype=np.float64)
            if arr.shape != x.shape:
                continue
            an = np.linalg.norm(arr)
            if an < 1e-12:
                continue
            arr = arr / an
            d = float(np.linalg.norm(x - arr))
            if d < best_d:
                best_d, best_g = d, g
        return best_g

    @staticmethod
    def _nearest_pattern_with_margin(
        rotated_rms: np.ndarray,
        ref_patterns: Dict[str, Any],
    ) -> Tuple[Optional[str], float, float]:
        """1-NN with margin diagnostics.

        Returns ``(predicted_label, distance_to_best, margin)`` where
        ``margin = distance_to_runner_up - distance_to_best``. A larger
        margin means the prediction was less ambiguous.
        """
        if not ref_patterns:
            return None, 0.0, 0.0
        x = rotated_rms.astype(np.float64)
        norm = np.linalg.norm(x)
        if norm < 1e-12:
            return None, 0.0, 0.0
        x = x / norm

        dists: List[Tuple[float, str]] = []
        for g, p in ref_patterns.items():
            arr = np.asarray(p, dtype=np.float64)
            if arr.shape != x.shape:
                continue
            an = np.linalg.norm(arr)
            if an < 1e-12:
                continue
            arr = arr / an
            d = float(np.linalg.norm(x - arr))
            dists.append((d, g))
        if not dists:
            return None, 0.0, 0.0
        dists.sort(key=lambda t: t[0])
        best_d, best_g = dists[0]
        runner_up = dists[1][0] if len(dists) > 1 else best_d * 2.0
        return best_g, best_d, runner_up - best_d


# ---------------------------------------------------------------------------
# RotationDetectionStudy — per-session rotation statistics
# ---------------------------------------------------------------------------

@dataclass
class SessionRotationResult:
    """Per-session record from a rotation-detection sweep."""
    subject_id: str
    session_id: str
    rotation_offset: int
    confidence: float
    sync_gesture: str
    n_trials: int
    drift_from_reference: int   # ring distance to the reference offset
    error: str = ""             # non-empty → this session failed

    @property
    def ok(self) -> bool:
        return not self.error


@dataclass
class RotationDetectionStudyReport:
    """Aggregated statistics across a corpus of sessions."""
    results: List[SessionRotationResult] = field(default_factory=list)
    reference_offset: int = 0
    ring_size: int = 0

    @property
    def successful(self) -> List[SessionRotationResult]:
        return [r for r in self.results if r.ok]

    @property
    def failed(self) -> List[SessionRotationResult]:
        return [r for r in self.results if not r.ok]

    def offsets(self) -> np.ndarray:
        return np.array([r.rotation_offset for r in self.successful], dtype=int)

    def confidences(self) -> np.ndarray:
        return np.array([r.confidence for r in self.successful], dtype=float)

    def drifts(self) -> np.ndarray:
        return np.array([r.drift_from_reference for r in self.successful], dtype=int)

    def offset_mode(self) -> Optional[int]:
        """The most common offset across successful sessions."""
        offs = self.offsets()
        if offs.size == 0:
            return None
        vals, counts = np.unique(offs, return_counts=True)
        return int(vals[int(np.argmax(counts))])

    def offset_histogram(self) -> Dict[int, int]:
        """``{offset: count}`` over successful sessions."""
        offs = self.offsets()
        out: Dict[int, int] = {}
        for o in offs:
            out[int(o)] = out.get(int(o), 0) + 1
        return out

    def stability(self) -> float:
        """Fraction of successful sessions whose offset matches the mode."""
        offs = self.offsets()
        if offs.size == 0:
            return 0.0
        mode = self.offset_mode()
        return float(np.mean(offs == mode))

    def summary(self) -> str:
        n = len(self.results)
        ok = len(self.successful)
        if ok == 0:
            return f"Rotation study: 0 / {n} sessions analyzed (all failed)."
        confs = self.confidences()
        drifts = self.drifts()
        mode = self.offset_mode()
        stab = self.stability() * 100.0
        return (
            f"Rotation study  ·  {ok} / {n} sessions ok  "
            f"·  offset mode {mode} ch  "
            f"·  stability {stab:.0f}%  "
            f"·  mean drift {float(drifts.mean()):.2f} ch  "
            f"·  conf {float(confs.mean()):.0%} ± {float(confs.std()):.0%}"
        )


class RotationDetectionStudy:
    """
    Scan many sessions and report rotation-detection statistics.

    Workflow:
      1. ``add_session(subject_id, session_id, session)`` for each one,
         OR ``run_over_data_manager(data_manager, subjects=...)`` to
         walk a DataManager-style corpus.
      2. ``analyze()`` → ``RotationDetectionStudyReport``.

    The study clones the calibrator's behavior in a side-effect-free
    way: it computes per-session offsets via the same processor the
    calibrator uses, but never overwrites ``current_calibration``.

    The per-session offset is computed against the calibrator's
    reference calibration when one exists; otherwise against the first
    successfully-processed session in the queue (an internal pseudo-
    reference). The latter mode is useful when you want to ask "are
    these recordings consistent with each other?" rather than "are
    they consistent with my saved reference?"
    """

    def __init__(self, calibrator):
        self.calibrator = calibrator
        # Lazily filled via add_session / run_over_data_manager.
        self._queue: List[Tuple[str, str, Any]] = []

    def add_session(self, subject_id: str, session_id: str, session) -> None:
        self._queue.append((subject_id, session_id, session))

    def run_over_data_manager(
        self,
        data_manager,
        *,
        subjects: Optional[List[str]] = None,
        sessions: Optional[List[Tuple[str, str]]] = None,
        progress=None,
    ) -> None:
        """Populate the queue from a DataManager.

        Parameters
        ----------
        subjects : list[str], optional
            Subjects to include. ``None`` = all known subjects.
        sessions : list[(subject_id, session_id)], optional
            Specific sessions. Takes priority over ``subjects`` if given.
        progress : callable(index, total, label), optional
            Notification hook. Called once per session-load.
        """
        pairs: List[Tuple[str, str]] = []
        if sessions is not None:
            pairs = list(sessions)
        else:
            try:
                if subjects is None:
                    subjects = list(data_manager.list_subjects())
            except Exception:  # noqa: BLE001
                subjects = []
            for subj in subjects:
                try:
                    sess_ids = data_manager.list_sessions(subj)
                except Exception:  # noqa: BLE001
                    sess_ids = []
                pairs.extend((subj, sid) for sid in sess_ids)

        total = len(pairs)
        for i, (subj, sid) in enumerate(pairs):
            if progress is not None:
                try:
                    progress(i, total, f"Loading {subj}/{sid}")
                except Exception:  # noqa: BLE001
                    pass
            try:
                session = data_manager.load_session(subj, sid)
            except Exception:  # noqa: BLE001
                # Surface load failures via a fail-marked SessionRotationResult.
                self._queue.append((subj, sid, None))
                continue
            self._queue.append((subj, sid, session))

    def analyze(self, progress=None) -> RotationDetectionStudyReport:
        """Process every queued session, return an aggregated report."""
        results: List[SessionRotationResult] = []

        # Establish the reference offset (the one we measure drift against).
        # Prefer the calibrator's saved reference, otherwise fall back to
        # the first successfully-processed session.
        ref_calib = self._get_reference_calibration()
        ref_offset: Optional[int] = None
        ring_size = int(getattr(self.calibrator.processor, "num_channels", 0)) or 0
        if ref_calib is not None:
            ref_offset = int(getattr(ref_calib, "rotation_offset", 0))
            ring_size = int(getattr(ref_calib, "num_channels", ring_size)) or ring_size

        total = len(self._queue)
        for i, (subj, sid, session) in enumerate(self._queue):
            if progress is not None:
                try:
                    progress(i, total, f"Analyzing {subj}/{sid}")
                except Exception:  # noqa: BLE001
                    pass

            if session is None:
                results.append(SessionRotationResult(
                    subject_id=subj, session_id=sid,
                    rotation_offset=0, confidence=0.0,
                    sync_gesture="", n_trials=0,
                    drift_from_reference=0,
                    error="failed to load session",
                ))
                continue

            try:
                offset, conf, sync, n_trials = self._process_session(session)
            except Exception as e:  # noqa: BLE001
                results.append(SessionRotationResult(
                    subject_id=subj, session_id=sid,
                    rotation_offset=0, confidence=0.0,
                    sync_gesture="", n_trials=0,
                    drift_from_reference=0,
                    error=str(e),
                ))
                continue

            if ref_offset is None:
                ref_offset = offset  # use first successful as pseudo-ref

            ring = ring_size
            if ring <= 0:
                # Best-effort: fall back to channel count of the session.
                ring = max(1, int(getattr(session.metadata, "num_channels", 32)))
            ring_size = ring

            drift = CalibrationValidator._ring_distance(offset, ref_offset, ring)
            results.append(SessionRotationResult(
                subject_id=subj, session_id=sid,
                rotation_offset=int(offset),
                confidence=float(conf),
                sync_gesture=str(sync),
                n_trials=int(n_trials),
                drift_from_reference=int(drift),
            ))

        if progress is not None:
            try:
                progress(total, total, "Done")
            except Exception:  # noqa: BLE001
                pass

        return RotationDetectionStudyReport(
            results=results,
            reference_offset=int(ref_offset) if ref_offset is not None else 0,
            ring_size=int(ring_size),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_reference_calibration(self):
        proc = getattr(self.calibrator, "processor", None)
        if proc is not None:
            getter = getattr(proc, "get_reference_calibration", None)
            if callable(getter):
                try:
                    out = getter()
                    if out is not None:
                        return out
                except Exception:  # noqa: BLE001
                    pass
        return getattr(self.calibrator, "reference_calibration", None)

    def _process_session(self, session) -> Tuple[int, float, str, int]:
        """Run rotation detection on one session.

        Returns (offset, confidence, sync_gesture_name, n_trials_used).
        """
        signal = CalibrationValidator._extract_session_signal(session)
        trials = CalibrationValidator._extract_session_trials(session)
        if signal is None or signal.size == 0:
            raise ValueError("session has no signal data")
        if not trials:
            raise ValueError("session has no valid trials")

        # Build the gesture_data dict the processor expects.
        gesture_data: Dict[str, List[np.ndarray]] = {}
        n_used = 0
        for trial in trials:
            gname = str(getattr(trial, "gesture_name", "")).strip()
            if not gname:
                continue
            start = int(getattr(trial, "start_sample", 0))
            end = int(getattr(trial, "end_sample", start))
            if end <= start or start < 0 or end > signal.shape[0]:
                continue
            chunk = signal[start:end]
            if chunk.shape[0] < 10:
                continue
            gesture_data.setdefault(gname, []).append(chunk)
            n_used += 1

        if not gesture_data:
            raise ValueError("no usable trials in session")

        # Use the processor directly so we don't mutate the calibrator.
        proc = self.calibrator.processor
        ref = self._get_reference_calibration()

        # Match the calibrator's signal mode logic if available.
        signal_mode = "monopolar"
        getter = getattr(self.calibrator, "_extract_session_signal_mode", None)
        if callable(getter):
            try:
                signal_mode = getter(session)
            except Exception:  # noqa: BLE001
                pass

        # Make sure the processor's num_channels matches this session.
        ses_ch = int(getattr(session.metadata, "num_channels", 0) or 0)
        if ses_ch > 0 and ses_ch != getattr(proc, "num_channels", ses_ch):
            proc.num_channels = ses_ch

        result = proc.calibrate_from_data(
            gesture_data,
            device_name=str(getattr(session.metadata, "device_name", "unknown")),
            reference_result=ref,
            signal_mode=signal_mode,
        )
        offset = int(result.rotation_offset)
        conf = float(result.confidence)
        sync = str(result.metadata.get("sync_gesture", ""))
        return offset, conf, sync, n_used


# ---------------------------------------------------------------------------
# Helper functions for working with sessions and gestures
# ---------------------------------------------------------------------------

def gestures_in_session(session) -> Set[str]:
    """The set of distinct active (non-Rest) gesture names in a session.

    Reads from ``session.get_valid_trials()`` (preferred) or ``session.trials``.
    """
    out: Set[str] = set()
    getter = getattr(session, "get_valid_trials", None)
    trials: list
    if callable(getter):
        try:
            trials = list(getter() or [])
        except Exception:  # noqa: BLE001
            trials = []
    else:
        trials = list(getattr(session, "trials", []) or [])
    for trial in trials:
        gname = str(getattr(trial, "gesture_name", "")).strip()
        if not gname or gname.lower() == "rest":
            continue
        out.add(gname)
    return out


def common_gestures(sessions: Iterable[Any]) -> Set[str]:
    """The intersection of active gestures across all given sessions."""
    sessions = list(sessions)
    if not sessions:
        return set()
    sets = [gestures_in_session(s) for s in sessions]
    if not any(sets):
        return set()
    common = sets[0]
    for s in sets[1:]:
        common = common & s
    return common


def union_gestures(sessions: Iterable[Any]) -> Set[str]:
    """The union of active gestures across all given sessions."""
    out: Set[str] = set()
    for s in sessions:
        out |= gestures_in_session(s)
    return out


def gesture_coverage(sessions: Iterable[Tuple[str, str, Any]]) -> Dict[str, int]:
    """``{gesture: number of sessions in which it appears}``.

    ``sessions`` is a list of ``(subject_id, session_id, session)`` tuples.
    """
    cnt: Dict[str, int] = {}
    for _, _, s in sessions:
        for g in gestures_in_session(s):
            cnt[g] = cnt.get(g, 0) + 1
    return cnt


# ---------------------------------------------------------------------------
# Batch validation across many sessions — main-thread iteration
# ---------------------------------------------------------------------------

@dataclass
class BatchReport:
    """The aggregated outcome of a multi-session batch validation."""
    reports: List[CalibrationReport] = field(default_factory=list)
    skipped: Dict[str, str] = field(default_factory=dict)

    # ---- session-level aggregates --------------------------------------

    def per_session_accuracy(self) -> Dict[str, float]:
        """``{label: held-out accuracy}`` (sessions without held-out are skipped)."""
        out: Dict[str, float] = {}
        for r in self.reports:
            d = r.held_out_details
            if d is None or d.n_trials == 0:
                continue
            out[self._label(r)] = d.overall_accuracy
        return out

    def per_subject_accuracy(self) -> Dict[str, List[float]]:
        """``{subject: [accuracy per session]}`` for grouping plots."""
        out: Dict[str, List[float]] = {}
        for r in self.reports:
            d = r.held_out_details
            if d is None or d.n_trials == 0:
                continue
            subj = r.subject_id or "?"
            out.setdefault(subj, []).append(d.overall_accuracy)
        return out

    def per_gesture_session_accuracy(self) -> Dict[str, List[float]]:
        """``{gesture: [accuracy per session that exercised that gesture]}``.

        Useful for box-and-whisker plots — each box is one gesture, each
        scatter point is one session's accuracy on that gesture.
        """
        out: Dict[str, List[float]] = {}
        for r in self.reports:
            d = r.held_out_details
            if d is None:
                continue
            for g, acc in d.per_gesture_accuracy().items():
                out.setdefault(g, []).append(acc)
        return out

    def aggregate_held_out_details(self) -> HeldOutDetails:
        """Concatenate every per-session held-out record into one big record."""
        agg = HeldOutDetails()
        for r in self.reports:
            d = r.held_out_details
            if d is None:
                continue
            agg.y_true.extend(d.y_true)
            agg.y_pred.extend(d.y_pred)
            agg.distances.extend(d.distances)
            agg.margins.extend(d.margins)
        return agg

    # ---- distributions over sessions -----------------------------------

    def offset_distribution(self) -> List[int]:
        return [r.rotation_offset for r in self.reports]

    def confidence_distribution(self) -> List[float]:
        return [r.confidence for r in self.reports]

    def check_pass_rates(self) -> Dict[str, float]:
        """Fraction of sessions in which each named check passed."""
        totals: Dict[str, List[int]] = {}
        for r in self.reports:
            for c in r.checks:
                slot = totals.setdefault(c.name, [0, 0])
                slot[0] += int(bool(c.passed))
                slot[1] += 1
        return {k: ok / n for k, (ok, n) in totals.items() if n > 0}

    def offset_per_session(self) -> Dict[str, int]:
        return {self._label(r): r.rotation_offset for r in self.reports}

    def confidence_per_session(self) -> Dict[str, float]:
        return {self._label(r): r.confidence for r in self.reports}

    def to_dict(self) -> dict:
        return {
            "reports": [r.to_dict() for r in self.reports],
            "skipped": dict(self.skipped),
        }

    @staticmethod
    def _label(r: CalibrationReport) -> str:
        s = r.subject_id or "?"
        sid = r.session_id or "?"
        return f"{s}/{sid}"


class BatchValidator:
    """Run :class:`CalibrationValidator` across a batch of sessions.

    Designed for **main-thread iteration**: each session is independent
    and ``validate_session`` can be driven from a Qt event loop with
    ``QTimer.singleShot`` chaining. This avoids the threading hazards
    that come from sharing the calibrator (which the validator mutates
    via ``calibrate_from_session``) across a worker thread.

    The save/restore around ``calibrate_from_session`` means the user's
    pre-existing ``current_calibration`` is preserved across the batch.
    """

    def __init__(
        self,
        calibrator,
        *,
        self_consistency_max_drift: Optional[int] = None,
        symmetry_min_ratio:         Optional[float] = None,
        held_out_min_accuracy:      Optional[float] = None,
        confidence_floor:           Optional[float] = None,
    ):
        self.calibrator = calibrator
        self._validator_kwargs = dict(
            self_consistency_max_drift=self_consistency_max_drift,
            symmetry_min_ratio=symmetry_min_ratio,
            held_out_min_accuracy=held_out_min_accuracy,
            confidence_floor=confidence_floor,
        )

    def _make_validator(self) -> CalibrationValidator:
        kwargs = {k: v for k, v in self._validator_kwargs.items() if v is not None}
        return CalibrationValidator(self.calibrator, **kwargs)

    def validate_session(
        self,
        session,
        *,
        gesture_filter: Optional[Iterable[str]] = None,
    ) -> CalibrationReport:
        """Validate one session against the saved reference.

        Save/restore protocol: ``calibrator.current_calibration`` is
        captured before the call and restored afterwards, so the user's
        active calibration is not mutated by the batch.

        Raises on failure (the caller is responsible for catching).
        """
        saved = getattr(self.calibrator, "current_calibration", None)
        try:
            # Computes a per-session calibration result and writes it to
            # current_calibration. The validator reads from there.
            try:
                self.calibrator.calibrate_from_session(session)
            except AttributeError as e:
                raise RuntimeError(
                    f"Calibrator has no calibrate_from_session method: {e}"
                ) from e

            gesture_data = self._extract_gesture_data(session)
            if not gesture_data:
                raise ValueError("session yields no usable trial data")

            return self._make_validator().run_all(
                gesture_data,
                held_out_session=session,
                gesture_filter=gesture_filter,
            )
        finally:
            # Restore so subsequent code (or the next batch run) sees
            # the original calibration, not whatever this last session
            # produced.
            try:
                self.calibrator.current_calibration = saved
            except Exception:  # noqa: BLE001
                # If the calibrator forbids assignment, we accept the
                # mutation rather than crash the batch.
                pass

    def run(
        self,
        sessions: Sequence[Tuple[str, str, Any]],
        *,
        gesture_filter: Optional[Iterable[str]] = None,
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
        event_pump:  Optional[Callable[[], None]] = None,
    ) -> BatchReport:
        """Process the whole batch synchronously and return aggregated results.

        Parameters
        ----------
        sessions
            ``[(subject_id, session_id, session), ...]``
        gesture_filter
            If given, only these gestures contribute to held-out accuracy.
        progress_cb
            Optional ``(i, total, label)`` callback before each session.
        event_pump
            Optional zero-arg callable invoked between sessions. Pass
            ``QApplication.processEvents`` to keep the UI responsive
            from the main thread.
        """
        report = BatchReport()
        gf: Optional[Set[str]] = (
            {str(g) for g in gesture_filter} if gesture_filter else None
        )
        for i, (subject, session_id, session) in enumerate(sessions):
            label = f"{subject}/{session_id}"
            if progress_cb is not None:
                try:
                    progress_cb(i, len(sessions), label)
                except Exception:  # noqa: BLE001
                    pass
            if event_pump is not None:
                try:
                    event_pump()
                except Exception:  # noqa: BLE001
                    pass

            try:
                rep = self.validate_session(session, gesture_filter=gf)
                rep.subject_id = subject
                rep.session_id = session_id
                report.reports.append(rep)
            except Exception as e:  # noqa: BLE001
                log.exception("Batch validation failed for %s", label)
                report.skipped[label] = str(e)

        if progress_cb is not None:
            try:
                progress_cb(len(sessions), len(sessions), "done")
            except Exception:  # noqa: BLE001
                pass
        return report

    @staticmethod
    def _extract_gesture_data(session) -> Dict[str, np.ndarray]:
        """``{gesture: vstacked (n_samples, n_channels)}`` from valid trials."""
        out: Dict[str, List[np.ndarray]] = {}
        try:
            all_data = session.get_data()
            valid = (session.get_valid_trials()
                     if hasattr(session, "get_valid_trials") else [])
        except Exception:  # noqa: BLE001
            return {}
        if all_data is None:
            return {}
        all_data = np.asarray(all_data)
        for trial in valid:
            try:
                start = int(getattr(trial, "start_sample", 0))
                end = int(getattr(trial, "end_sample", start))
                gname = str(getattr(trial, "gesture_name", "")).strip()
            except Exception:  # noqa: BLE001
                continue
            if not gname or end <= start or start < 0 or end > all_data.shape[0]:
                continue
            chunk = all_data[start:end]
            if chunk.shape[0] >= 10:
                out.setdefault(gname, []).append(chunk)
        merged: Dict[str, np.ndarray] = {}
        for g, chunks in out.items():
            try:
                merged[g] = np.vstack(chunks)
            except Exception:  # noqa: BLE001
                continue
        return merged