"""
training_game_coordinator.py
────────────────────────────
Bridges the dataset-recording protocol with the Unity PlayAgain game so
children can collect their **first** EMG dataset by playing instead of
by sitting through a bare "perform gesture now" prompt.

Why this exists
───────────────
Collecting the initial dataset is the single biggest friction point in
the pipeline: the child has no trained model yet, so the game can't
recognise what they're doing. Without gamification they see a dry cue,
hold a gesture, rest, repeat — and quickly disengage.

This coordinator fixes that by running the game in **"easy mode"**:

  1. The child opens PlayAgain. An animal walks in on cue from the
     coordinator (``target_gesture`` message over the existing
     Unity↔Python TCP channel).

  2. The coordinator watches the EMG stream and computes RMS per
     chunk. The moment RMS crosses a per-subject threshold while the
     target trial is active, the coordinator broadcasts a synthetic
     prediction — ``{"gesture": <current_target>, "confidence": 1.0}``
     — so Unity's existing ``PipelineGestureClient`` fires
     ``OnGestureActiveChanged(true)`` and the animal gets fed.

  3. Unity signals back via its normal ``game_state`` callback when
     the animal walks away. The coordinator closes the current trial
     in the ``RecordingSession``, queues the next gesture, and repeats.

  4. When every trial is done the coordinator emits ``all_complete``
     and the GUI tears everything down.

Because the "fake prediction" path uses the same JSON schema Unity
already parses, no Unity-side change is required to make easy mode
work. A small optional patch to ``PipelineGestureClient.cs`` lets
Python drive *which* animal spawns next; without that patch the game
still plays — animals just come in their Unity-configured order.

Threading model
───────────────
  • ``on_emg_data(samples)`` is called from the device thread. It is
    lock-free and thread-safe by design: a single atomic write to a
    shared float and a cheap RMS computation.
  • Trial state (which gesture is active, when to broadcast) lives on
    a single QTimer tick driven by the GUI thread.
  • Signals cross threads via Qt's queued connections.

The coordinator never touches the UI directly — it emits Qt signals
and lets the GUI route them to labels / buttons / the session object.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal, Slot

from playagain_pipeline.core.session import RecordingSession
from playagain_pipeline.prediction_server import PredictionServer

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trial schedule
# ---------------------------------------------------------------------------

@dataclass
class TrialSpec:
    """
    One trial in the easy-mode training schedule.

    Attributes
    ----------
    gesture_name : str
        Name used across the pipeline — must match a gesture in the
        session's ``GestureSet`` (e.g. ``"fist"``, ``"pinch"``).
    hold_seconds : float
        How long the child is expected to hold the gesture. The
        coordinator doesn't enforce this — Unity's feeding duration
        does — it's carried through only so the protocol can tell
        the user what to expect.
    rest_seconds : float
        Rest between this trial and the next. Primarily informational.
    """
    gesture_name: str
    hold_seconds: float = 3.0
    rest_seconds: float = 2.0


@dataclass
class _TrialRuntime:
    """Runtime bookkeeping for an in-flight trial — not user-facing."""
    spec: TrialSpec
    index: int
    broadcast_sent: bool = False
    # Wall-clock millis of the last easy-mode broadcast sent for this
    # trial. Used by the watchdog tick to re-arm the detector if Unity
    # never raised the ``ground_truth=true`` rising edge — symptom of a
    # silently-dropped broadcast (gesture-name mismatch on Unity's
    # side, network blip, animal still walking in, etc.). Without this
    # one trial getting stuck froze the entire run.
    broadcast_sent_ms: float = 0.0
    unity_gt_active: bool = False       # Last ground_truth state from Unity
    # Last gesture requested by Unity for this trial (if provided via game_state).
    requested_gesture: str = ""
    # Last time we sent a target_gesture cue for this trial.
    last_cue_sent_ms: float = 0.0
    started_at_ms: float = field(default_factory=lambda: time.time() * 1000)
    # Whether we've emitted trial_started to the GUI / popup yet.
    started_emitted: bool = False


# ---------------------------------------------------------------------------
# RMS detector — deliberately tiny
# ---------------------------------------------------------------------------

class _RmsDetector:
    """
    Per-chunk RMS detector with a **frozen** rest baseline.

    Lifecycle
    ─────────
    The detector goes through three phases per session:

      ``IDLE``         — no calibration data yet, ``observe()`` is a no-op.
      ``CALIBRATING``  — Each ``observe()`` call appends the chunk to a
                          calibration buffer and returns False unconditionally.
                          The coordinator stays here for ``_BASELINE_SETTLE_MS``
                          while showing the child a "REST" cue.
      ``READY``        — ``finalize_calibration()`` has computed the rest RMS
                          and frozen it. From this point ``observe()`` triggers
                          when ``rms > frozen_baseline * trigger_ratio`` and
                          the caller has set ``armed=True``.

    Why frozen
    ──────────
    The previous EMA design constantly retrained the baseline during rest
    intervals, so when a synthetic recording started mid-gesture the baseline
    settled on gesture-level RMS and the next gesture could not exceed
    ``baseline × ratio``. With a frozen baseline learned from a dedicated
    rest cue, the threshold is stable and predictable for the entire
    session — much closer to how clinical EMG triggers work.

    The threshold ratio defaults to a forgiving 1.3 because the goal here
    is *motivation*, not precision: we just need to confirm the child
    *attempted* the gesture so the animal can be fed. Real classification
    happens later from the recorded data, where the protocol's ground-truth
    label tells the model what gesture was being performed.
    """

    PHASE_IDLE        = "idle"
    PHASE_CALIBRATING = "calibrating"
    PHASE_READY       = "ready"

    def __init__(self, trigger_ratio: float = 1.3):
        self._phase: str = self.PHASE_IDLE
        self._baseline: float = 0.0
        self._trigger_ratio: float = trigger_ratio
        self._last_rms: float = 0.0
        # Per-chunk RMS values collected during CALIBRATING. Median of
        # these becomes the baseline — robust against the occasional
        # twitchy chunk a child can't help producing during "rest".
        self._calibration_rms: list[float] = []

    # ------------------------------------------------------------------
    # Read-only state
    # ------------------------------------------------------------------

    @property
    def last_rms(self) -> float:
        return self._last_rms

    @property
    def baseline(self) -> float:
        return self._baseline

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def is_ready(self) -> bool:
        return self._phase == self.PHASE_READY

    @property
    def threshold(self) -> float:
        """The fixed RMS threshold a chunk must exceed to fire."""
        return self._baseline * self._trigger_ratio

    # ------------------------------------------------------------------
    # Phase transitions
    # ------------------------------------------------------------------

    def begin_calibration(self) -> None:
        """Switch to CALIBRATING and clear any previous calibration data."""
        self._phase = self.PHASE_CALIBRATING
        self._calibration_rms = []

    def finalize_calibration(self) -> bool:
        """
        Compute the frozen baseline from the collected rest RMS values
        and switch to READY.

        Returns
        -------
        bool
            True on success. False if no calibration data was ever
            observed (e.g. device was disconnected the whole time);
            in that case the detector stays in CALIBRATING and the
            coordinator should re-enter calibration or fall back.
        """
        if not self._calibration_rms:
            return False
        # Median is robust against the occasional gesture twitch a
        # child sneaks in during a "rest" cue. Mean would let one
        # bad chunk inflate the threshold.
        sorted_rms = sorted(self._calibration_rms)
        n = len(sorted_rms)
        if n % 2 == 1:
            self._baseline = sorted_rms[n // 2]
        else:
            self._baseline = 0.5 * (sorted_rms[n // 2 - 1] + sorted_rms[n // 2])
        self._phase = self.PHASE_READY
        return True

    def reset(self) -> None:
        """Return to IDLE, dropping the baseline and calibration buffer."""
        self._phase = self.PHASE_IDLE
        self._baseline = 0.0
        self._calibration_rms = []

    # ------------------------------------------------------------------
    # Sample ingestion
    # ------------------------------------------------------------------

    def observe(self, samples: np.ndarray, armed: bool) -> bool:
        """
        Ingest a chunk and return True iff an armed trigger fired.

        Parameters
        ----------
        samples : np.ndarray
            Shape ``(n_samples, n_channels)``.
        armed : bool
            Whether the coordinator currently considers a trigger
            useful (i.e. there is an active trial that has not yet
            broadcast). Triggers only fire when the detector is READY
            *and* armed.
        """
        if samples.ndim == 1:
            samples = samples.reshape(-1, 1)

        rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
        self._last_rms = rms

        if self._phase == self.PHASE_IDLE:
            return False

        if self._phase == self.PHASE_CALIBRATING:
            self._calibration_rms.append(rms)
            return False

        # READY
        if not armed:
            return False
        return rms > self.threshold and rms > 1e-8


# ---------------------------------------------------------------------------
# Public coordinator
# ---------------------------------------------------------------------------

class TrainingGameCoordinator(QObject):
    """
    Runs an easy-mode training session against the Unity game.

    Lifecycle
    ─────────
        coord = TrainingGameCoordinator(prediction_server=server, parent=main_win)
        coord.trial_started.connect(lambda spec, idx: ...)
        coord.trial_completed.connect(lambda spec, idx: ...)
        coord.all_complete.connect(lambda: ...)

        coord.set_schedule([TrialSpec("fist"), TrialSpec("pinch"), ...])
        coord.bind_session(session)                # optional — marks trials
        coord.start()
        # feed device samples via coord.on_emg_data(samples) from wherever
        # they already arrive (main window's _on_data_received does this).
        coord.stop()    # or wait for all_complete

    Integration points
    ──────────────────
      • ``on_emg_data(samples)`` — call from the existing device
        data handler. Does the RMS work + broadcast decision.
      • ``on_game_state_from_unity(ground_truth, requested, camera_blocking)``
        — registered as a callback on the PredictionServer. Detects
        Unity-side "animal fed" events to advance trials.
    """

    # ── Signals ────────────────────────────────────────────────────
    # Arguments use plain Python types / objects to sidestep any
    # QMetaType registration drama in older PySide builds.
    trial_started    = Signal(object, int)        # (TrialSpec, index)
    trial_completed  = Signal(object, int)        # (TrialSpec, index)
    trigger_fired    = Signal(str, float)         # (gesture, rms) — for UI feedback
    all_complete     = Signal()
    rms_updated      = Signal(float, float)       # (rms, baseline) — for a debug meter
    state_changed    = Signal(str)                # "idle" | "waiting_for_unity" | "running" | "stopped"
    game_level_started = Signal()                 # Unity reached Level 1 — session begins now
    # Per-gesture completion counts in balanced mode. Payload is a dict
    # ``{gesture_name: (completed, target)}`` so the GUI can render
    # "fist 2/3, pinch 1/3, tripod 0/3" in a status label. Emitted on
    # every trial completion AND once at start so the popup can
    # initialise the display to all-zeroes.
    balance_progress = Signal(dict)

    # Cooldown between triggers for the *same* trial so a single long
    # contraction doesn't fire twice and Unity has time to transition
    # the animal to feeding mode.
    _TRIGGER_COOLDOWN_MS = 1200

    # If we've broadcast an easy-mode prediction but Unity hasn't
    # raised ``ground_truth=true`` within this window, assume the
    # broadcast was lost (gesture-name mismatch, animal not yet
    # blocking, dropped packet) and re-arm the detector for a retry.
    # 3 s is comfortable: Unity's animation pipeline normally raises
    # ground_truth within 200–600 ms of receiving the broadcast.
    _BROADCAST_RETRY_MS = 3000

    # After the child releases (Unity sets ground_truth=False) we give
    # a short quiet period before arming the next trial. Keeps the
    # baseline clean of tail-end contraction energy.
    _INTER_TRIAL_REST_MS = 800

    # While waiting for Unity to actually start feeding (ground_truth
    # rising edge), re-send the cue periodically to survive client
    # reconnects during scene transitions.
    _TARGET_CUE_RESEND_MS = 1000

    # Safety fallback: if Unity never reports game_level_started (older
    # build without the notifier, or a crash before Level 1), start
    # anyway after this long so the GUI doesn't wedge. 60 s comfortably
    # covers a slow Unity boot plus the main-menu click.
    _GAME_START_TIMEOUT_MS = 60_000

    # Length of the rest-calibration window. The detector spends this
    # long collecting RMS values from the resting hand; the popup
    # shows a big "REST — calibrating baseline" cue throughout.
    # 2 s is long enough for a stable median across ~20 device chunks
    # at typical chunk rates and short enough not to bore a child.
    _BASELINE_SETTLE_MS = 2000

    # How long to wait after camera_blocking becomes True before arming the detector.
    # This gives Unity's camera / animation a short settle time so the recording
    # starts when the animal is visually stable in the blocking position.
    _CAMERA_SETTLE_MS = 1000

    # ── Private signals for cross-thread marshaling ────────────────
    # The prediction server's reader loop runs on its own network
    # thread. When it dispatches a callback (game_state /
    # game_level_started), our handlers cannot directly touch QTimers
    # or Qt parented to the GUI thread — Qt prints
    # "QObject::startTimer: Timers cannot be started from another
    # thread" and silently drops the call.
    #
    # The fix: thread-thunk via signals. Emitting a signal from any
    # thread is always safe; with the default AutoConnection, the
    # connected slot is invoked on the slot's owning thread (here, the
    # main GUI thread) via a queued connection. So both handlers
    # below just emit and return — the real work runs on the right
    # thread one event-loop hop later.
    _level_started_main = Signal()
    _game_state_main    = Signal(bool, str, bool)

    def __init__(
        self,
        prediction_server: PredictionServer,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self._server = prediction_server
        self._detector = _RmsDetector()

        self._schedule: List[TrialSpec] = []
        self._current: Optional[_TrialRuntime] = None
        self._next_index: int = 0

        # Balanced-completion state. When ``_balanced_mode`` is True the
        # coordinator stops following the static schedule list and
        # instead keeps cuing the most under-represented gesture until
        # each gesture in ``_target_per_gesture`` has been completed
        # ``target`` times. Why this exists: with an unpatched Unity
        # build the game sometimes spawns animals in its own order,
        # and the recording follows what Unity actually showed the
        # child — not our cue. The result was an imbalanced dataset
        # even though the schedule was balanced. Counting completions
        # per gesture and reissuing cues for missing ones guarantees
        # equal sample counts regardless of who wins the spawn fight.
        self._balanced_mode: bool = False
        self._target_per_gesture: dict[str, int] = {}
        self._completed_per_gesture: dict[str, int] = {}
        # Unity-override-aware delivery counter. Increments on EVERY
        # falling edge regardless of whether Unity delivered what we
        # cued. The new picker reads from this so it stops cuing a
        # gesture Unity has already over-delivered. Without this, the
        # picker kept cuing 'fist' (deficit 1) while Unity kept
        # spawning 'pinch' (already at target), which produced the
        # 2/5/2 imbalance the user reported.
        self._delivered_per_gesture: dict[str, int] = {}
        # Hard cap on total trials. Without this, a Unity build that
        # spawns the same animal forever would loop indefinitely.
        # Default 2× the expected total — generous but bounded.
        self._max_total_trials: int = 0

        self._running = False
        # True between start() and the Unity "level started" message
        # (or the timeout). During this window we register callbacks
        # but refuse to ingest EMG or issue gesture cues — the
        # recording must line up with Level 1, not with the Unity boot
        # screen or the "Fuchs Abenteuer" main menu.
        self._waiting_for_unity = False
        self._session: Optional[RecordingSession] = None
        self._last_trigger_ms: float = 0.0
        self._arm_time_ms: float = 0.0  # when the current trial becomes eligible

        # Spawn-sync lock: True between the moment we send a
        # target_gesture cue and the moment Unity confirms the previous
        # animal's ground_truth falling edge.  Prevents _advance() from
        # queuing a second animal before the first one has walked away,
        # which was the root cause of "number of animals inconsistent
        # with trials".
        self._spawn_pending: bool = False

        # Whether session.start_recording() should be invoked by the
        # coordinator (True) or the GUI already started it (False).
        # Set via bind_session(..., take_ownership=True) when the GUI
        # hands us an unstarted session that should begin with Level 1.
        self._owns_session_lifecycle = False

        # Lightweight tick — RMS meter + inter-trial gating. Runs at a
        # modest rate because the real triggering is sample-driven in
        # ``on_emg_data``, not timer-driven.
        self._tick = QTimer(self)
        self._tick.setInterval(100)
        self._tick.timeout.connect(self._on_tick)

        # Cross-thread fan-in. The network-thread callbacks emit these
        # signals; AutoConnection delivers them on the GUI thread, so
        # the slots below are free to touch QTimers and Qt-parented
        # state.
        self._level_started_main.connect(self._begin_running)
        self._game_state_main.connect(self._handle_game_state_on_main_thread)

        # Registered on the prediction server so Unity's messages
        # route back here. Both added on start() and removed on stop()
        # so the coordinator leaves no trace when inactive.
        self._game_state_cb = self.on_game_state_from_unity
        self._level_started_cb = self.on_game_level_started_from_unity
        self._session_config_cb = self.on_session_config_from_unity

        # Guards against Unity never reporting level_started. Only
        # active during the waiting window.
        self._game_start_timeout = QTimer(self)
        self._game_start_timeout.setSingleShot(True)
        self._game_start_timeout.timeout.connect(self._on_game_start_timeout)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_schedule(self, trials: List[TrialSpec]) -> None:
        """Install the trial order. Must be called before ``start()``."""
        if self._running:
            raise RuntimeError(
                "Cannot change schedule while the coordinator is running. "
                "Call stop() first."
            )
        self._schedule = list(trials)
        self._next_index = 0
        # Strict-schedule mode — turn off balanced advancement.
        self._balanced_mode = False
        self._target_per_gesture = {}
        self._completed_per_gesture = {}
        self._delivered_per_gesture = {}
        self._max_total_trials = 0

    def set_balanced_mode(
        self,
        gestures: List[str],
        reps_per_gesture: int,
        hold_seconds: float = 3.0,
        rest_seconds: float = 2.0,
        max_oversample: float = 2.0,
    ) -> None:
        """
        Run trials until **each** gesture in ``gestures`` has been
        completed ``reps_per_gesture`` times.

        Why a separate mode (vs. just ``set_schedule``)
        ───────────────────────────────────────────────
        With an unpatched Unity build the game spawns animals in its
        own order and may ignore our ``target_gesture`` cues. The
        recording follows what Unity actually showed the child, so
        even a perfectly balanced static schedule can produce an
        unbalanced dataset. Balanced mode fixes this by:

          1. Counting completed trials per gesture as Unity reports
             them (using the ``gesture_requested`` field from the
             game_state message — what the child actually saw).
          2. On each ``_advance``, re-cuing the most under-represented
             gesture rather than the next slot in a static list.
          3. Stopping when every target is met OR the safety cap of
             ``max_oversample × total_expected`` trials is hit.

        With a patched Unity that obeys our cues, balanced mode is
        equivalent to a strict schedule — the cued gesture matches the
        spawned one, every cue lands one completion, and we stop after
        exactly ``len(gestures) × reps_per_gesture`` trials.

        Parameters
        ----------
        gestures : list of str
            Gesture names to target. ``"rest"`` is filtered out
            automatically — rest is handled implicitly between trials.
        reps_per_gesture : int
            How many successful trials per gesture.
        hold_seconds, rest_seconds : float
            Carried into each TrialSpec.
        max_oversample : float
            Safety multiplier for the total trial cap. With 3 gestures
            × 3 reps and ``max_oversample=2.0`` we'll attempt at most
            18 trials before giving up on stragglers.
        """
        if self._running:
            raise RuntimeError(
                "Cannot change schedule while the coordinator is running. "
                "Call stop() first."
            )
        clean = [g for g in gestures if g and g.lower() != "rest"]
        if not clean:
            self._balanced_mode = False
            self._schedule = []
            self._next_index = 0
            return

        self._balanced_mode = True
        self._target_per_gesture = {g: int(reps_per_gesture) for g in clean}
        self._completed_per_gesture = {g: 0 for g in clean}
        self._delivered_per_gesture = {g: 0 for g in clean}
        total_expected = sum(self._target_per_gesture.values())
        self._max_total_trials = max(
            total_expected,
            int(round(total_expected * max_oversample)),
        )

        # Build a starting schedule of length ``total_expected`` so the
        # popup's "trial X / N" display has a meaningful denominator.
        # This list is only consulted as a length reference once we
        # enter ``_advance_balanced`` — the actual gesture chosen at
        # each step is whichever one is most under-represented.
        self._schedule = [
            TrialSpec(
                gesture_name=g,
                hold_seconds=hold_seconds,
                rest_seconds=rest_seconds,
            )
            for _ in range(int(reps_per_gesture))
            for g in clean
        ]
        self._next_index = 0

    def bind_session(
        self,
        session: Optional[RecordingSession],
        take_ownership: bool = False,
    ) -> None:
        """
        Attach a RecordingSession so the coordinator can call
        ``start_trial`` / ``end_trial`` at each Unity-driven boundary.

        Parameters
        ----------
        session : RecordingSession or None
            Session to attach. Passing None detaches — useful for
            dry-run / demo mode.
        take_ownership : bool
            When True, the coordinator will call
            ``session.start_recording()`` itself at the moment Unity
            reports Level 1 has started, rather than expecting the
            caller to have started it already. This is the recommended
            setting when driving the training game — it ensures the
            first recorded sample lines up with the first frame of
            gameplay, not with the Unity boot screen.
        """
        self._session = session
        self._owns_session_lifecycle = bool(take_ownership and session is not None)

    def set_trigger_ratio(self, ratio: float) -> None:
        """
        How many multiples of the **frozen** rest baseline count as an
        attempt. Default 1.3 is forgiving on purpose — this is the
        easy-mode "did the child try?" gate, and the protocol's
        ground-truth label drives the actual dataset, so a generous
        threshold mostly costs us a few false positives in exchange
        for a much-better-engaged child.

        Clinics may push this up to 1.5 or 1.8 once the child is
        comfortable.
        """
        self._detector._trigger_ratio = max(1.05, float(ratio))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def current_target(self) -> Optional[str]:
        return self._current.spec.gesture_name if self._current else None

    @property
    def progress(self) -> tuple[int, int]:
        """(completed, total) — completed counts trials already finished."""
        total = len(self._schedule)
        done = self._next_index - (1 if self._current is not None else 0)
        return (max(0, done), total)

    def start(self, wait_for_unity: bool = True) -> None:
        """
        Arm the coordinator.

        When ``wait_for_unity`` is True (default), the coordinator
        registers for Unity events but *does not* start the recording
        session or issue gesture cues yet. It stays in the
        ``waiting_for_unity`` state until either:

          • Unity sends a ``game_level_started`` message (clean path —
            the child clicked "Start" in the menu and Level 1 is
            loaded), or
          • ``_GAME_START_TIMEOUT_MS`` elapses (fallback for older
            Unity builds without the notifier).

        When ``wait_for_unity`` is False the coordinator begins
        immediately. Useful for headless tests without Unity running
        or for debugging the schedule logic standalone.
        """
        if self._running or self._waiting_for_unity:
            return
        if not self._schedule:
            raise RuntimeError(
                "Cannot start — no trials in schedule. Call set_schedule()."
            )

        self._detector.reset()
        self._next_index = 0
        self._current = None
        self._spawn_pending = False

        # Register for Unity → Python messages. Safe to attach even
        # while waiting; we just refuse to act on game_state until
        # the level-started flag flips.
        try:
            self._server.add_game_state_callback(self._game_state_cb)
        except Exception:
            log.exception("Failed to register game_state callback")

        # Older prediction-server builds may not expose the
        # level-started registry. Feature-detect so one-sided upgrades
        # don't crash — the timeout fallback covers the gap.
        if hasattr(self._server, "add_game_level_started_callback"):
            try:
                self._server.add_game_level_started_callback(self._level_started_cb)
            except Exception:
                log.exception("Failed to register game_level_started callback")

        # Register for session_config messages so Unity's settings-panel
        # checkboxes (balanced mode, repetitions, gesture list) are
        # automatically mirrored into this coordinator without any
        # manual Python-side configuration.
        if hasattr(self._server, "add_session_config_callback"):
            try:
                self._server.add_session_config_callback(self._session_config_cb)
            except Exception:
                log.exception("Failed to register session_config callback")

        if wait_for_unity:
            self._waiting_for_unity = True
            self._game_start_timeout.start(self._GAME_START_TIMEOUT_MS)
            self.state_changed.emit("waiting_for_unity")
            log.info(
                "Coordinator armed — waiting for Unity to reach Level 1 "
                "(timeout %.0fs).",
                self._GAME_START_TIMEOUT_MS / 1000.0,
            )
            return

        # Legacy / no-wait path: go immediately.
        self._begin_running()

    @Slot()
    def _begin_running(self) -> None:
        """
        Transition from waiting / idle into the ``calibrating`` state.

        Called by ``on_game_level_started_from_unity`` on the clean
        path and by the timeout fallback. Three things happen here:

          1. The recording session starts — so the first CSV sample
             lines up with Level 1, not with the Unity boot screen.
          2. The detector enters its CALIBRATING phase. The popup
             observes ``state_changed("calibrating")`` and shows a
             "REST — calibrating baseline" cue; the detector silently
             collects rest RMS values without ever firing.
          3. After ``_BASELINE_SETTLE_MS`` the baseline is frozen and
             the trial schedule begins via ``_finish_calibration``.
        """
        if self._running:
            return

        self._game_start_timeout.stop()
        self._waiting_for_unity = False

        # Start recording now — if we own the session lifecycle — so
        # the first CSV sample aligns with the first frame of Level 1,
        # not with the Unity boot screen.
        if (
            self._owns_session_lifecycle
            and self._session is not None
            and not self._session.is_recording
        ):
            try:
                self._session.start_recording()
                log.info("Recording session started (Unity reached Level 1).")
            except Exception:
                log.exception("Failed to start recording session on Unity cue")

        self._running = True
        self._tick.start()

        self._detector.begin_calibration()
        self.state_changed.emit("calibrating")
        self.game_level_started.emit()
        # Push the initial all-zeros progress to the GUI so the popup
        # can show "fist 0/3, pinch 0/3, tripod 0/3" before any trial
        # has run. No-op outside balanced mode.
        self._emit_balance_progress()
        log.info(
            "Calibrating rest baseline for %.1fs…",
            self._BASELINE_SETTLE_MS / 1000.0,
        )
        # Tell Unity to spawn a "rest" cue (no animal yet) so the
        # child keeps still during calibration. If the Unity build
        # ignores unknown gestures this is a harmless no-op.
        self._send_target_gesture_cue("rest", self._BASELINE_SETTLE_MS / 1000.0)

        QTimer.singleShot(self._BASELINE_SETTLE_MS, self._finish_calibration)

    @Slot()
    def _finish_calibration(self) -> None:
        """
        Freeze the baseline learned during calibration and start the
        actual trial schedule.

        If the device produced no data during calibration (e.g. it was
        disconnected), fall through to the schedule anyway with the
        detector still in CALIBRATING — manual buttons still work, the
        auto-trigger just won't fire. That's preferable to refusing to
        start the session.
        """
        if not self._running:
            return

        ok = self._detector.finalize_calibration()
        if ok:
            log.info(
                "Baseline frozen at RMS %.4f → threshold %.4f (ratio %.2fx)",
                self._detector.baseline,
                self._detector.threshold,
                self._detector._trigger_ratio,
            )
        else:
            log.warning(
                "No data observed during calibration — manual triggers will "
                "still work but the auto-RMS trigger is disabled this session."
            )

        self.state_changed.emit("running")
        self._spawn_pending = False
        self._advance()

    @Slot()
    def _on_game_start_timeout(self) -> None:
        """
        Fallback — Unity never reported game_level_started. Start the
        session anyway and log loud enough that anyone investigating
        will notice.
        """
        if not self._waiting_for_unity:
            return
        log.warning(
            "Unity did not report game_level_started within %.0fs — "
            "starting the session anyway. If you expected the Unity "
            "build to send this message, check that GameLevelStartedNotifier "
            "is present in the Level 1 scene.",
            self._GAME_START_TIMEOUT_MS / 1000.0,
        )
        self._begin_running()

    def on_game_level_started_from_unity(self) -> None:
        """
        Registered as a prediction_server ``game_level_started`` callback.

        IMPORTANT: this method runs on the prediction server's network
        reader thread, *not* the GUI thread. Doing real work here would
        trip Qt's "Timers cannot be started from another thread" guards
        and silently break the session. So all we do is emit a signal —
        the connected slot ``_begin_running`` is invoked on the GUI
        thread via Qt's auto-queued connection.

        Idempotent: a duplicate emit is a harmless no-op because
        ``_begin_running`` early-returns when already running.
        """
        if not self._waiting_for_unity:
            return
        log.info("Unity reported Level 1 loaded — marshaling to GUI thread.")
        self._level_started_main.emit()

    def on_session_config_from_unity(
        self,
        balanced: bool,
        sequential: bool,
        repetitions: int,
        gestures,           # list[str] | None
        hold_seconds: float,
        pause_seconds: float,
    ) -> None:
        """
        Registered as a prediction_server ``session_config`` callback.

        Called (from the network reader thread) when Unity sends a
        ``session_config`` message immediately after ``game_level_started``.
        Unity emits this message with the current settings-panel state —
        balanced/sequential checkbox, Repetitions slider, and the gesture
        names derived from the active LevelDefinition — so the coordinator
        can configure itself automatically without any manual Python GUI work.

        The coordinator must still be in the ``waiting_for_unity`` state
        (i.e. ``start()`` was called but ``game_level_started`` has not
        yet arrived).  If it has already transitioned to ``running``, we
        log a warning and ignore the message so we don't corrupt an
        in-flight session.

        Threading
        ---------
        ``set_balanced_mode`` and ``set_schedule`` both raise when
        ``_running`` is True. Because ``session_config`` is sent *before*
        ``game_level_started``, the coordinator is in ``waiting_for_unity``
        here and not yet running — the call is safe. The defensive guard
        below handles any edge-case race.

        Example
        -------
        Unity settings: BalancedGestureMode=True, Repetitions=3,
        LevelDefinition with Horse→Fist, Cow→Tripod, Pig→Pinch.

        Unity sends::

            {
                "type": "session_config",
                "balanced": true, "sequential": false,
                "repetitions": 3,
                "gestures": ["fist", "tripod", "pinch"],
                "hold_seconds": 3.0, "pause_seconds": 5.0
            }

        Result: ``set_balanced_mode(["fist","tripod","pinch"], reps_per_gesture=3)``
        → 9 trials, exactly 3 of each gesture, shuffled rounds.
        """
        if self._running:
            log.warning(
                "[coord] session_config received while already running — "
                "ignoring. Config must arrive before game_level_started."
            )
            return

        # If Unity sends null/empty gestures, keep whatever the GUI
        # configured; only override when a real list arrives.
        clean: list = (
            [g for g in gestures if g and g.lower() != "rest"]
            if gestures else []
        )
        if not clean:
            log.info(
                "[coord] session_config: no gesture list — keeping current "
                "schedule (balanced=%s, sequential=%s, reps=%d).",
                balanced, sequential, repetitions,
            )
            return

        if balanced:
            # ── Balanced mode ─────────────────────────────────────────────
            # Every gesture appears exactly `repetitions` times in shuffled
            # rounds, so the dataset stays balanced even if the session ends
            # early on a round boundary.
            # E.g. gestures=["fist","tripod","pinch"], reps=3 → 9 trials.
            log.info(
                "[coord] session_config → set_balanced_mode: "
                "gestures=%s, reps=%d, hold=%.1fs, pause=%.1fs",
                clean, repetitions, hold_seconds, pause_seconds,
            )
            try:
                self.set_balanced_mode(
                    gestures=clean,
                    reps_per_gesture=repetitions,
                    hold_seconds=hold_seconds,
                    rest_seconds=pause_seconds,
                )
            except RuntimeError as exc:
                log.error("[coord] set_balanced_mode failed: %s", exc)

        elif sequential:
            # ── Sequential (block) mode ───────────────────────────────────
            # All `repetitions` of gesture 1, then all of gesture 2, etc.
            # Matches the block structure of SequentialGestureMode in Unity.
            flat = [
                TrialSpec(
                    gesture_name=g,
                    hold_seconds=hold_seconds,
                    rest_seconds=pause_seconds,
                )
                for g in clean
                for _ in range(repetitions)
            ]
            log.info(
                "[coord] session_config → set_schedule (sequential): "
                "%d trials, order=%s",
                len(flat), [t.gesture_name for t in flat],
            )
            try:
                self.set_schedule(flat)
            except RuntimeError as exc:
                log.error("[coord] set_schedule (sequential) failed: %s", exc)

        else:
            # ── No special mode — use balanced as default ─────────────────
            # Unity knows which gestures are active even without a checkbox;
            # use balanced mode so the dataset stays representative.
            log.info(
                "[coord] session_config → set_balanced_mode (default, "
                "no Unity flag): gestures=%s, reps=%d",
                clean, repetitions,
            )
            try:
                self.set_balanced_mode(
                    gestures=clean,
                    reps_per_gesture=repetitions,
                    hold_seconds=hold_seconds,
                    rest_seconds=pause_seconds,
                )
            except RuntimeError as exc:
                log.error("[coord] set_balanced_mode (default) failed: %s", exc)

    def stop(self) -> None:
        """End the schedule, whether partway through or at completion."""
        if not (self._running or self._waiting_for_unity):
            return

        was_waiting = self._waiting_for_unity
        self._running = False
        self._waiting_for_unity = False
        self._spawn_pending = False
        self._tick.stop()
        self._game_start_timeout.stop()

        # Close any open trial in the session so we don't leak state.
        if self._session is not None and self._current is not None:
            try:
                self._session.end_trial(is_valid=False, notes="coordinator stopped early")
            except Exception:
                log.exception("Failed to close in-flight trial on stop")

        # If we were running the session ourselves, stop it too.
        if (
            self._owns_session_lifecycle
            and self._session is not None
            and self._session.is_recording
        ):
            try:
                self._session.stop_recording()
            except Exception:
                log.exception("Failed to stop recording session on teardown")

        self._current = None

        try:
            self._server.remove_game_state_callback(self._game_state_cb)
        except Exception:
            pass
        if hasattr(self._server, "remove_game_level_started_callback"):
            try:
                self._server.remove_game_level_started_callback(self._level_started_cb)
            except Exception:
                pass

        if hasattr(self._server, "remove_session_config_callback"):
            try:
                self._server.remove_session_config_callback(self._session_config_cb)
            except Exception:
                pass

        self.state_changed.emit("stopped" if not was_waiting else "idle")

    # ------------------------------------------------------------------
    # Data plane — called from the device thread
    # ------------------------------------------------------------------

    def on_emg_data(self, samples: np.ndarray) -> None:
        """
        Feed raw EMG samples in.

        Safe to call from any thread. All state mutated here is either
        atomic (single floats) or only read from the GUI thread via
        timers, so no lock is needed. The RMS signal emitted to the UI
        goes through Qt's queued connection across the thread boundary.
        """
        if samples is None or samples.size == 0:
            return

        # Arm only if we're in a trial *and* past the post-arm settling
        # window. While disarmed the detector tracks the rest baseline.
        now_ms = time.time() * 1000
        armed = bool(
            self._current is not None
            and not self._current.broadcast_sent
            and now_ms >= self._arm_time_ms
        )

        fired = self._detector.observe(samples, armed=armed)

        # Push the meter update to the UI — coalesced naturally by Qt.
        self.rms_updated.emit(self._detector.last_rms, self._detector.baseline)

        if not fired:
            return
        if now_ms - self._last_trigger_ms < self._TRIGGER_COOLDOWN_MS:
            return

        self._last_trigger_ms = now_ms
        cur = self._current
        if cur is None:
            return

        # cur.spec was rewritten to Unity's reported gesture in the
        # ``camera_blocking and not started_emitted`` branch (see
        # _handle_game_state_on_main_thread). Reading from spec keeps
        # the broadcast, the popup, the recording label, and the
        # balance counter all pinned to the same gesture.
        gesture_to_feed = cur.spec.gesture_name

        # Easy-mode: broadcast a confidence-1.0 "prediction" matching
        # the current target. Unity's PipelineGestureClient will call
        # OnGestureActiveChanged(true) and Level 1's GameManager will
        # feed the animal, exactly as if a real model had fired.
        cur.broadcast_sent = True
        cur.broadcast_sent_ms = now_ms
        self._broadcast_easy_mode_prediction(gesture_to_feed)
        self.trigger_fired.emit(gesture_to_feed, self._detector.last_rms)

    # ------------------------------------------------------------------
    # Unity → Python event plane
    # ------------------------------------------------------------------

    def on_game_state_from_unity(
        self,
        ground_truth_active: bool,
        requested_gesture: str,
        camera_blocking: bool,
    ) -> None:
        """
        Registered as a prediction_server ``game_state`` callback.

        Runs on the network reader thread — see the docstring on
        ``on_game_level_started_from_unity`` for the threading
        rationale. We just emit a queued signal; the actual edge
        detection happens in ``_handle_game_state_on_main_thread``.
        """
        self._game_state_main.emit(
            bool(ground_truth_active),
            str(requested_gesture or ""),
            bool(camera_blocking),
        )

    @Slot(bool, str, bool)
    def _handle_game_state_on_main_thread(
        self,
        ground_truth_active: bool,
        requested_gesture: str,
        camera_blocking: bool,
    ) -> None:
        """
        Main-thread body of the game-state callback.

        Unity's game_state flips ``ground_truth=true`` when the fox
        reaches an animal (feeding starts) and ``false`` when the
        animal has been fed and walks away. We treat the rising edge
        as "trial actually began in the game" and the falling edge as
        "trial done — advance".

        This coordinator changes the timing so that the GUI popup and
        the session trial start are emitted only when the animal has
        reached its blocking position (camera_blocking == True). The
        coordinator still sends the spawn cue earlier so Unity can walk
        the animal in, but the visible "request gesture" cue and the
        session trial start wait for camera_blocking to ensure the
        dataset aligns with the in-game camera.
        """
        if not self._running or self._current is None:
            return

        cur = self._current

        requested_clean = str(requested_gesture or "").strip().lower()
        if requested_clean and requested_clean != "none":
            cur.requested_gesture = requested_clean

        now_ms = time.time() * 1000

        # NEW: When the animal reaches blocking position, emit trial_started
        # and start the session trial (so recording aligns with the in-game camera).
        if camera_blocking and not cur.started_emitted:
            cur.started_emitted = True

            # Lock the trial's identity to whatever Unity actually
            # spawned. Without this, the popup labels the trial with
            # our planned cue while the recording labels it with
            # Unity's reported gesture — exactly the mismatch the user
            # asked us to fix. After this rewrite, ``cur.spec`` is the
            # single source of truth that everyone downstream (popup,
            # recording label, easy-mode broadcast, balance counter)
            # reads from. If Unity is silent (older build that doesn't
            # send ``gesture_requested``), ``cur.requested_gesture``
            # is empty and we keep our cued gesture as the fallback.
            if (cur.requested_gesture
                    and cur.requested_gesture != cur.spec.gesture_name):
                log.info(
                    "[coord] Unity reports '%s' but we cued '%s' — "
                    "re-aligning trial label so popup and recording match.",
                    cur.requested_gesture, cur.spec.gesture_name,
                )
                cur.spec = TrialSpec(
                    gesture_name=cur.requested_gesture,
                    hold_seconds=cur.spec.hold_seconds,
                    rest_seconds=cur.spec.rest_seconds,
                )

            # Emit trial_started so the popup updates now that the animal is in place.
            self.trial_started.emit(cur.spec, cur.index)
            # Arm the detector after a short camera-settle window to avoid tail energy.
            self._arm_time_ms = now_ms + self._CAMERA_SETTLE_MS
            # Start the session trial here (idempotent if session.start_trial is called again later).
            if self._session is not None:
                # cur.spec.gesture_name is now Unity-aligned (above) —
                # the recording label will match what trial_started
                # just told the popup to display.
                try:
                    self._session.start_trial(cur.spec.gesture_name)
                except Exception:
                    log.exception("start_trial failed for %s", cur.spec.gesture_name)

        # Rising edge — the game confirms the child has engaged with
        # the animal (feeding begins). We record the unity_gt_active
        # state but we no longer start the session trial here because
        # that now happens when camera_blocking becomes True.
        if ground_truth_active and not cur.unity_gt_active:
            cur.unity_gt_active = True
            # Nothing else to do here for session start — it was handled
            # on camera_blocking above.

        # Falling edge — animal was fed and has walked away. Close the
        # trial and schedule the next one after a short rest.
        if not ground_truth_active and cur.unity_gt_active:
            cur.unity_gt_active = False
            # Release the spawn lock — Unity has confirmed this animal
            # has departed, so _advance() may now send the next cue.
            self._spawn_pending = False
            if self._session is not None:
                try:
                    self._session.end_trial(is_valid=True)
                except Exception:
                    log.exception("end_trial failed for %s", cur.spec.gesture_name)

            # Track per-gesture completion AND delivery.
            #
            # ``completed_per_gesture`` counts trials that hit our cued
            # target — useful diagnostically.
            # ``delivered_per_gesture`` counts EVERY trial keyed by the
            # gesture Unity actually showed, regardless of cue. The
            # picker reads delivered, not completed, so it stops cuing
            # gestures Unity has already over-delivered.
            #
            # ``cur.spec.gesture_name`` was rewritten to Unity's
            # reported gesture in the camera_blocking branch above, so
            # both counters here use the gesture the EMG was actually
            # performed for.
            if self._balanced_mode:
                actual = cur.spec.gesture_name.lower()
                self._delivered_per_gesture[actual] = (
                    self._delivered_per_gesture.get(actual, 0) + 1
                )
                if actual in self._target_per_gesture:
                    self._completed_per_gesture[actual] = (
                        self._completed_per_gesture.get(actual, 0) + 1
                    )
                # Emit on EVERY delivery so the popup updates even when
                # Unity over-delivered something we already had enough of.
                self._emit_balance_progress()

            self.trial_completed.emit(cur.spec, cur.index)
            self._current = None
            # Give the baseline a moment to settle before arming the
            # next trial — otherwise the tail of this contraction
            # inflates the resting floor and makes the next trigger
            # harder.
            self._arm_time_ms = (time.time() * 1000) + self._INTER_TRIAL_REST_MS
            QTimer.singleShot(self._INTER_TRIAL_REST_MS, self._advance)

    # ------------------------------------------------------------------
    # Internal — trial advancement & broadcast
    # ------------------------------------------------------------------

    def _advance(self) -> None:
        if not self._running:
            return
        if self._current is not None:
            # A trial is already in flight; don't stack.
            return
        if self._spawn_pending:
            # We already sent a target_gesture cue and are waiting for
            # Unity to confirm the animal arrived (camera_blocking) and
            # then departed (ground_truth falling edge). Sending another
            # cue now would result in two animals for one trial slot.
            log.debug("_advance: spawn already pending — waiting for Unity confirmation")
            return

        # Branch on advancement strategy. Balanced mode keeps going
        # until every gesture has been completed N times; strict mode
        # walks the static schedule once.
        if self._balanced_mode:
            spec = self._pick_next_balanced()
            if spec is None:
                # All targets met (or safety cap hit) — finish.
                self._finish_run()
                return
        else:
            if self._next_index >= len(self._schedule):
                # Done!
                self._finish_run()
                return
            spec = self._schedule[self._next_index]

        self._current = _TrialRuntime(spec=spec, index=self._next_index)
        self._next_index += 1

        # Lock spawning until Unity confirms ground_truth falling edge
        # for this trial (see _handle_game_state_on_main_thread).
        self._spawn_pending = True

        # Tell Unity which animal/gesture to spawn next. We still send
        # the cue so Unity can walk the animal in, but we intentionally
        # do NOT emit trial_started here. The visible request and the
        # session trial start wait until Unity reports camera_blocking=True.
        self._send_target_gesture_cue(spec.gesture_name, spec.hold_seconds)
        self._current.last_cue_sent_ms = time.time() * 1000

        # Disarm the detector while the animal is walking in so we don't
        # accidentally trigger on movement. The detector will be armed
        # only after camera_blocking=True and a short settle window.
        self._arm_time_ms = float("inf")

        # Note: do not emit trial_started here. The popup will be updated
        # when camera_blocking becomes True (see _handle_game_state_on_main_thread).

    def _pick_next_balanced(self) -> Optional[TrialSpec]:
        """
        Choose the gesture that needs the most additional deliveries
        to hit its target.

        Why deliveries, not on-target completions
        ──────────────────────────────────────────
        The previous version of this method counted only trials where
        Unity delivered what we cued. With an unpatched Unity that
        ignores cues, the picker would keep cuing 'fist' (deficit 1)
        while Unity kept delivering 'pinch' (already over-delivered),
        producing the 2/5/2 imbalance the user saw. By switching to
        delivery counts we:

          1. Suppress further cues for over-delivered gestures (their
             deficit is now zero or negative — they're filtered out).
          2. Stop the run as soon as every gesture is *in the dataset*
             at least N times — which is the user-visible guarantee.

        With a patched Unity that obeys cues, deliveries == completions
        and behaviour is unchanged.
        """
        # Deficit based on what Unity has actually put into the dataset.
        deficits = {
            g: self._target_per_gesture[g] - self._delivered_per_gesture.get(g, 0)
            for g in self._target_per_gesture
        }
        remaining = {g: d for g, d in deficits.items() if d > 0}
        if not remaining:
            # Every gesture has been delivered at least its target count.
            return None

        # Safety cap — Unity may stubbornly spawn one gesture forever.
        if self._next_index >= self._max_total_trials:
            log.warning(
                "Balanced mode safety cap reached at %d trials; "
                "missing: %s. Delivered so far: %s",
                self._next_index, remaining, self._delivered_per_gesture,
            )
            return None

        # Pick the gesture with the largest delivery deficit, with
        # alphabetical tie-break for deterministic round-robin.
        max_deficit = max(remaining.values())
        candidates = sorted(
            g for g, d in remaining.items() if d == max_deficit
        )
        chosen = candidates[0]

        # Re-use any matching template from the original schedule so
        # hold_seconds/rest_seconds picked at set_balanced_mode time
        # carry through. Falls back to defaults if the template list
        # was empty (shouldn't happen but defensive).
        for tpl in self._schedule:
            if tpl.gesture_name == chosen:
                return TrialSpec(
                    gesture_name=chosen,
                    hold_seconds=tpl.hold_seconds,
                    rest_seconds=tpl.rest_seconds,
                )
        return TrialSpec(gesture_name=chosen)

    def _emit_balance_progress(self) -> None:
        """
        Emit the ``balance_progress`` signal with per-gesture **delivery**
        counts. No-op outside balanced mode.

        The dict reports ``(delivered, target)``. Delivered = "what's
        actually in your dataset right now", which is what the user
        cares about. With patched Unity, delivered == completed_on_target,
        so this remains correct for that case too.
        """
        if not self._balanced_mode:
            return
        payload = {
            g: (self._delivered_per_gesture.get(g, 0), self._target_per_gesture[g])
            for g in self._target_per_gesture
        }
        try:
            self.balance_progress.emit(payload)
        except RuntimeError:
            # Coordinator parent may already be torn down — emitting
            # on a deleted Qt object raises. Safe to drop.
            pass

    def _finish_run(self) -> None:
        """Common cleanup path for both modes when the run ends."""
        self._running = False
        self._tick.stop()
        try:
            self._server.remove_game_state_callback(self._game_state_cb)
        except Exception:
            pass
        self.state_changed.emit("idle")
        self.all_complete.emit()

    # ------------------------------------------------------------------
    # Manual control — used by the GameProtocolPopup buttons
    # ------------------------------------------------------------------

    # Delay between sending a target_gesture cue (animal walks in) and
    # firing the easy-mode prediction (animal eats). This needs to be
    # at least the time Unity's spawner needs to instantiate the animal
    # and walk it into feeding position. 1500 ms is comfortable on the
    # default Level 1 layout.
    _MANUAL_GESTURE_FEED_DELAY_MS = 1500

    def force_trigger(self) -> None:
        """
        Fire the current trial's easy-mode broadcast immediately,
        ignoring the RMS check.

        The "Force Feed Animal" button calls this. Use case: the
        baseline is mis-calibrated, or the synthetic replay isn't
        producing a clean threshold crossing, but the clinician can
        see the child trying and wants to advance the game.
        """
        cur = self._current
        if cur is None:
            return
        now_ms = time.time() * 1000
        if now_ms - self._last_trigger_ms < self._TRIGGER_COOLDOWN_MS / 2:
            return
        self._last_trigger_ms = now_ms
        cur.broadcast_sent = True
        # Stamp the time so the watchdog can re-arm if Unity ignored
        # the manual broadcast (same failure mode as the auto path).
        cur.broadcast_sent_ms = now_ms
        self._broadcast_easy_mode_prediction(cur.spec.gesture_name)
        log.info("Force-feed: broadcasting easy-mode for %s", cur.spec.gesture_name)
        self.trigger_fired.emit(cur.spec.gesture_name, self._detector.last_rms)

    def fire_manual_gesture(self, gesture_name: str) -> None:
        """
        Demo path: spawn the matching animal AND feed it — independent
        of the trial schedule.

        Wired to the per-gesture buttons in the popup. Two messages go
        out, in order:

          1. ``target_gesture`` — Unity's TargetGestureAnimalMapper
             writes ``SettingsManager.SelectedAnimalType``; the spawner
             then brings the matching animal in.
          2. After ``_MANUAL_GESTURE_FEED_DELAY_MS`` — an easy-mode
             prediction with the same gesture name, so the animal that
             just arrived gets fed immediately.

        The trial schedule is **not** advanced — these buttons are
        explicitly "play this gesture for demo / debug", not "this
        trial is complete". They also don't write any session
        annotations, so the recorded data isn't polluted by demo fires.

        The delay is safe to spam-click — each click queues another
        animal+feed pair and Unity processes them in order.
        """
        if not gesture_name:
            return
        gesture_name = gesture_name.strip().lower()
        log.info("Manual gesture demo: %s", gesture_name)

        # 1) Spawn-the-animal cue
        self._send_target_gesture_cue(gesture_name, hold_seconds=2.0)

        # 2) Fire the prediction once the animal has had time to arrive
        QTimer.singleShot(
            self._MANUAL_GESTURE_FEED_DELAY_MS,
            lambda: self._broadcast_easy_mode_prediction(gesture_name),
        )

        # -1.0 RMS sentinel marks this as a manual fire in any UI meter
        self.trigger_fired.emit(gesture_name, -1.0)

    def skip_current_trial(self, reason: str = "skipped via UI") -> None:
        """
        End the current trial without firing a broadcast and advance
        to the next one. The skipped trial is marked invalid in the
        session so it's excluded from training.

        Wired to the popup's "Skip" button — handy when the child gives
        up on a gesture or the synthetic replay produced no recognisable
        activity.
        """
        cur = self._current
        if cur is None:
            return
        if self._session is not None:
            try:
                self._session.end_trial(is_valid=False, notes=reason)
            except Exception:
                log.exception("Failed to end skipped trial")
        self.trial_completed.emit(cur.spec, cur.index)
        self._current = None
        # Release the spawn lock so the next advance() can proceed.
        self._spawn_pending = False
        self._arm_time_ms = (time.time() * 1000) + self._INTER_TRIAL_REST_MS
        QTimer.singleShot(self._INTER_TRIAL_REST_MS, self._advance)

    # ------------------------------------------------------------------
    # Internal — broadcast helpers
    # ------------------------------------------------------------------

    def _broadcast_easy_mode_prediction(self, gesture_name: str) -> None:
        """
        Enqueue a synthetic prediction payload onto the prediction
        server's existing send queue. Using the server's own queue
        means the same sender thread that normally dispatches model
        predictions will handle ours — no second send path, no races.
        """
        try:
            message = {
                "gesture": gesture_name,
                "gesture_id": -1,            # Sentinel — not a real class ID
                "confidence": 1.0,
                "probabilities": {gesture_name: 1.0},
                "timestamp": time.time(),
                "source": "easy_mode",        # Harmless extra field; Unity ignores.
            }
            if hasattr(self._server, "enqueue_json_message"):
                self._server.enqueue_json_message(message)
            else:
                data = (json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8")
                self._server._send_queue.put_nowait(data)  # noqa: SLF001 — compatibility fallback
        except Exception:
            log.exception("Failed to broadcast easy-mode prediction")

    def _send_target_gesture_cue(
        self,
        gesture_name: str,
        hold_seconds: float,
    ) -> None:
        """
        Push a ``target_gesture`` message on the same channel as the
        predictions. Unity treats unknown message types as no-ops, so
        old builds remain compatible. Patched builds (see the
        PipelineGestureClient diff in the feature delivery) can read
        this and spawn the matching animal.
        """
        try:
            message = {
                "type": "target_gesture",
                "gesture": gesture_name,
                "hold_seconds": float(hold_seconds),
                "timestamp": time.time(),
            }
            if hasattr(self._server, "enqueue_json_message"):
                self._server.enqueue_json_message(message)
            else:
                data = (json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8")
                self._server._send_queue.put_nowait(data)  # noqa: SLF001
        except Exception:
            log.exception("Failed to send target_gesture cue")

    # ------------------------------------------------------------------
    # Periodic tick — very light work
    # ------------------------------------------------------------------

    @Slot()
    def _on_tick(self) -> None:
        """Periodic watchdog. Two responsibilities:

          1. **Cue resend** — if Unity reconnected after we sent a
             ``target_gesture`` cue, the original cue may have gone to
             nobody. Re-send until Unity reaches blocking position.
          2. **Broadcast-retry** — if we already broadcast an
             easy-mode prediction but Unity never raised
             ``ground_truth=true``, the broadcast was almost certainly
             lost. Re-arm the detector so the next contraction can
             fire again. Without this defence one stuck trial freezes
             the entire run, which matches the "third trial didn't
             feed" symptom you reported.
        """
        cur = self._current
        if not self._running or cur is None:
            return

        now_ms = time.time() * 1000

        # ── 2. Broadcast-retry watchdog ────────────────────────────
        # Runs first because, if it fires, it puts us back into a
        # state where the cue-resend logic below shouldn't re-trigger.
        if (cur.broadcast_sent
                and not cur.unity_gt_active
                and cur.broadcast_sent_ms > 0
                and now_ms - cur.broadcast_sent_ms > self._BROADCAST_RETRY_MS):
            log.warning(
                "[coord] No ground_truth rising edge for '%s' after "
                "%.0f ms — re-arming detector. (Possible causes: "
                "gesture-name mismatch on Unity side, animal still "
                "walking in, or a dropped network message.)",
                cur.spec.gesture_name, now_ms - cur.broadcast_sent_ms,
            )
            cur.broadcast_sent = False
            cur.broadcast_sent_ms = 0.0
            # Bypass the per-trial trigger cooldown so the next
            # observed RMS spike can fire immediately.
            self._last_trigger_ms = 0.0
            return

        # ── 1. Cue resend ─────────────────────────────────────────
        if cur.unity_gt_active:
            return
        # Don't re-send once the animal has reached blocking position —
        # a second cue at this point would queue a phantom second animal.
        if cur.started_emitted:
            return
        if now_ms - cur.last_cue_sent_ms < self._TARGET_CUE_RESEND_MS:
            return

        self._send_target_gesture_cue(cur.spec.gesture_name, cur.spec.hold_seconds)
        cur.last_cue_sent_ms = now_ms


# ---------------------------------------------------------------------------
# Convenience — build a schedule from a RecordingSession's gesture set
# ---------------------------------------------------------------------------

def build_default_schedule(
    session: RecordingSession,
    repetitions: int = 5,
    hold_seconds: float = 3.0,
    rest_seconds: float = 2.0,
    include_rest: bool = False,
    randomize_within_round: bool = True,
    seed: Optional[int] = None,
) -> List[TrialSpec]:
    """
    Build a **balanced** trial schedule from a session's gesture set.

    Each non-rest gesture appears exactly ``repetitions`` times. Gestures
    are divided into rounds of one repetition per gesture.  Within each
    round the order is shuffled (default) or deterministically interleaved
    — both strategies guarantee equal representation across the full
    session, which matters for unbiased EMG datasets.

    Shuffled rounds also break up long monotonous runs (e.g. 5 × fist in
    a row) that a child might find frustrating, without sacrificing the
    per-gesture count guarantee that pure random sampling can violate.

    Parameters
    ----------
    include_rest : bool
        Default False. Rest is handled implicitly as the gap between
        animals in the game; forcing the child to "perform rest"
        feels odd. Set True only if you really want explicit rest
        trials in the dataset.
    randomize_within_round : bool
        When True (default), shuffle the gesture order within every
        round using ``random.shuffle``, seeded by ``seed`` if given.
        When False, use a fixed interleaved order (g1, g2, g3, …)
        — identical to the old behaviour.
    seed : int or None
        RNG seed for reproducible shuffling. Ignored when
        ``randomize_within_round`` is False.
    """
    import random as _random

    gestures = [
        g.name for g in session.gesture_set.gestures
        if include_rest or g.name.lower() != "rest"
    ]
    if not gestures:
        return []

    rng = _random.Random(seed)
    schedule: List[TrialSpec] = []

    for _ in range(repetitions):
        round_gestures = list(gestures)          # copy for this round
        if randomize_within_round:
            rng.shuffle(round_gestures)
        for name in round_gestures:
            schedule.append(TrialSpec(
                gesture_name=name,
                hold_seconds=hold_seconds,
                rest_seconds=rest_seconds,
            ))

    return schedule