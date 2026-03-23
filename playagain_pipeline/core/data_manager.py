"""
Data management for gesture pipeline.

Handles loading, saving, and organizing recording sessions and datasets.
"""

from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Callable
from datetime import datetime
import json
import re
import shutil
import numpy as np

from playagain_pipeline.core.session import RecordingSession
from playagain_pipeline.core.gesture import GestureSet
from playagain_pipeline.models.classifier import EMGFeatureExtractor
from sklearn.model_selection import train_test_split


class DataManager:
    """
    Manages storage and retrieval of recording sessions and datasets.

    Provides a unified interface for:
    - Saving and loading recording sessions
    - Organizing data by subject and session
    - Creating ML-ready datasets
    """

    def __init__(self, data_dir: Path):
        """
        Initialize the data manager.

        Args:
            data_dir: Root directory for all data storage
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Subdirectories
        self.sessions_dir = self.data_dir / "sessions"
        self.datasets_dir = self.data_dir / "datasets"
        self.calibrations_dir = self.data_dir / "calibrations"
        self.models_dir = self.data_dir / "models"
        self.participants_dir = self.data_dir / "Participant_Info"

        for d in [self.sessions_dir, self.datasets_dir,
                  self.calibrations_dir, self.models_dir,
                  self.participants_dir]:
            d.mkdir(exist_ok=True)

    def get_session_path(self, subject_id: str, session_id: str) -> Path:
        """Get the path for a specific session."""
        return self.sessions_dir / subject_id / session_id

    def save_session(self, session: RecordingSession) -> Path:
        """
        Save a recording session.

        Args:
            session: The session to save

        Returns:
            Path where the session was saved
        """
        path = self.get_session_path(
            session.metadata.subject_id,
            session.metadata.session_id
        )
        session.save(path)
        return path

    def load_session(self, subject_id: str, session_id: str) -> RecordingSession:
        """
        Load a recording session.

        Args:
            subject_id: ID of the subject
            session_id: ID of the session

        Returns:
            Loaded RecordingSession
        """
        path = self.get_session_path(subject_id, session_id)
        return RecordingSession.load(path)

    def list_subjects(self) -> List[str]:
        """List all subjects with recorded data, sorted in natural rising order."""
        subjects = set()

        for d in self.sessions_dir.iterdir():
            if d.is_dir():
                subjects.add(d.name)

        if self.participants_dir.exists():
            for participant_file in self.participants_dir.glob("*.json"):
                subjects.add(participant_file.stem)

        return sorted(subjects, key=self._natural_sort_key)

    def get_participant_info_path(self, subject_id: str) -> Path:
        """Get the JSON file path for a participant's info record."""
        return self.participants_dir / f"{subject_id}.json"

    def has_participant_info(self, subject_id: str) -> bool:
        """Return True if a participant info file exists for the subject."""
        return self.get_participant_info_path(subject_id).exists()

    def load_participant_info(self, subject_id: str) -> Optional[Dict[str, Any]]:
        """Load stored participant information for a subject, if available."""
        path = self.get_participant_info_path(subject_id)
        if not path.exists():
            return None

        with open(path, 'r') as f:
            return json.load(f)

    def save_participant_info(self, subject_id: str, info: Dict[str, Any]) -> Path:
        """Save participant information to `Participant_Info/<subject_id>.json`."""
        self.participants_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "subject_id": subject_id,
            "saved_at": datetime.now().isoformat(),
            "participant": info,
        }

        path = self.get_participant_info_path(subject_id)
        with open(path, 'w') as f:
            json.dump(payload, f, indent=2, default=str)
        return path

    def get_next_subject_id(self, prefix: str = "VP_", digits: int = 2) -> str:
        """Allocate the next available `VP_##` subject identifier."""
        pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
        max_index = 0

        for subject_id in self.list_subjects():
            match = pattern.match(subject_id)
            if match:
                max_index = max(max_index, int(match.group(1)))

        next_index = max_index + 1
        width = max(digits, len(str(next_index)))
        return f"{prefix}{next_index:0{width}d}"

    @staticmethod
    def _natural_sort_key(text: str):
        """Sort key for natural ordering: VP_00 < VP_01 < VP_02 < VP_10."""
        return [int(c) if c.isdigit() else c.lower()
                for c in re.split(r'(\d+)', text)]

    def list_sessions(self, subject_id: str) -> List[str]:
        """List all sessions for a subject, sorted by recording order (creation time)."""
        subject_dir = self.sessions_dir / subject_id
        if not subject_dir.exists():
            return []

        sessions = []
        for d in subject_dir.iterdir():
            if d.is_dir():
                metadata_path = d / "metadata.json"
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r') as f:
                            data = json.load(f)
                        created_at = datetime.fromisoformat(data["metadata"]["created_at"])
                        sessions.append((d.name, created_at))
                    except (KeyError, ValueError):
                        # If can't parse, use directory name as fallback
                        sessions.append((d.name, datetime.min))
                else:
                    sessions.append((d.name, datetime.min))

        # Sort by creation time, then by name for stable ordering
        sessions.sort(key=lambda x: (x[1], self._natural_sort_key(x[0])))
        return [name for name, _ in sessions]

    def get_all_sessions(self, subject_id: Optional[str] = None) -> List[RecordingSession]:
        """
        Get all sessions, optionally filtered by subject.

        Args:
            subject_id: If provided, only get sessions for this subject

        Returns:
            List of RecordingSession objects
        """
        sessions = []
        subjects = [subject_id] if subject_id else self.list_subjects()

        for sid in subjects:
            for sess_id in self.list_sessions(sid):
                try:
                    sessions.append(self.load_session(sid, sess_id))
                except Exception as e:
                    print(f"Warning: Could not load session {sid}/{sess_id}: {e}")

        return sessions

    def create_dataset(
        self,
        name: str,
        sessions: Optional[List[RecordingSession]] = None,
        subject_ids: Optional[List[str]] = None,
        window_size_ms: int = 200,
        window_stride_ms: int = 50,
        include_invalid: bool = False,
        preprocessing_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        calibration=None,
        use_per_session_rotation: bool = False,
        feature_config: Optional[Dict[str, Any]] = None,
        bad_channels: Optional[Dict[str, List[int]]] = None,
    ) -> Dict[str, Any]:
        """
        Create a machine learning dataset from recording sessions.

        Args:
            name: Name for the dataset
            sessions: List of sessions to include (if None, use subject_ids)
            subject_ids: List of subjects to include (if sessions is None)
            window_size_ms: Window size in milliseconds
            window_stride_ms: Window stride in milliseconds
            include_invalid: Whether to include invalid trials
            preprocessing_fn: Optional function to preprocess EMG data before windowing
            calibration: Optional CalibrationResult to apply a single global channel
                reordering. Ignored when use_per_session_rotation is True.
            use_per_session_rotation: If True, each session's own rotation_offset
                (stored in its metadata) is used to rotate channels into the
                canonical reference layout. This is the recommended mode when
                combining recordings from different bracelet placements, because
                each session gets its own correction rather than a single global one.
            feature_config: Optional dict with keys 'mode' and 'features' for
                pre-extracting features at dataset creation time. If provided and
                mode is not 'raw', features are computed and stored as 2D X.
            bad_channels: Optional dict mapping session_id -> list of bad channel
                indices (0-based). Bad channels are zeroed out per session.

        Returns:
            Dictionary containing:
            - X: Feature array (windows x samples x channels) or (windows x features)
            - y: Label array
            - metadata: Dataset metadata
        """
        if sessions is None:
            if subject_ids is None:
                subject_ids = self.list_subjects()
            sessions = []
            for sid in subject_ids:
                sessions.extend(self.get_all_sessions(sid))

        per_session_rotations = {}  # session_id -> rotation_offset applied

        # ── Helper: prepare one session's data (rotation, bad-ch, preproc) ──
        def _prepare_session_data(session: RecordingSession):
            """Return processed data array (as float32) for a single session."""
            data = np.array(session.get_data(), dtype=np.float32, copy=True)

            # Zero out bad channels for this session
            session_bad_chs: list = []
            if bad_channels is not None:
                session_bad_chs = list(bad_channels.get(session.metadata.session_id, []))
            if hasattr(session.metadata, 'bad_channels') and session.metadata.bad_channels:
                session_bad_chs = list(set(session_bad_chs) | set(session.metadata.bad_channels))
            if session_bad_chs:
                n_ch = data.shape[1]
                for ch_idx in session_bad_chs:
                    if ch_idx < n_ch:
                        left = (ch_idx - 1) % n_ch
                        right = (ch_idx + 1) % n_ch
                        data[:, ch_idx] = 0.5 * (data[:, left] + data[:, right])

            # Determine rotation to apply
            if use_per_session_rotation:
                rot = session.metadata.rotation_offset
                # Use the pre-computed channel_mapping stored in session metadata
                # when available — it was built by create_channel_mapping() and
                # handles the 32-ch split-ring topology correctly.
                # Recomputing from rotation_offset with (i - rot) would apply the
                # inverse rotation and silently mis-align all channels.
                mapping = getattr(session.metadata, 'channel_mapping', None)
                if mapping and len(mapping) == data.shape[1]:
                    data = data[:, mapping]
                elif rot != 0 and data.shape[1] >= session.metadata.num_channels:
                    # Fallback: no stored mapping — reconstruct with correct sign
                    n_ch = session.metadata.num_channels
                    mapping = [(i + rot) % n_ch for i in range(n_ch)]
                    data = data[:, mapping]
                per_session_rotations[session.metadata.session_id] = rot
            elif calibration is not None:
                try:
                    data = calibration.apply_to_data(data)
                except ValueError:
                    pass

            if preprocessing_fn is not None:
                data = preprocessing_fn(data)

            return data

        # ── Pass 1: count total windows so we can pre-allocate ──────────
        window_counts = []   # (session_idx, n_windows)
        expected_shape = None
        total_windows = 0

        for sess_idx, session in enumerate(sessions):
            trials = session.trials if include_invalid else session.get_valid_trials()
            window_samples = int(window_size_ms * session.metadata.sampling_rate / 1000)
            stride_samples = int(window_stride_ms * session.metadata.sampling_rate / 1000)
            n_wins = 0
            for trial in trials:
                trial_len = trial.end_sample - trial.start_sample
                if trial_len >= window_samples:
                    n_wins += (trial_len - window_samples) // stride_samples + 1
            window_counts.append(n_wins)
            total_windows += n_wins

            if expected_shape is None and n_wins > 0:
                n_channels = session.metadata.num_channels
                expected_shape = (window_samples, n_channels)

        if total_windows == 0 or expected_shape is None:
            raise ValueError("No data windows extracted")

        # ── Pre-allocate output arrays (float32 saves ~50% memory) ──────
        X = np.empty((total_windows, expected_shape[0], expected_shape[1]),
                      dtype=np.float32)
        y = np.empty(total_windows, dtype=np.int64)
        trial_id_list: List[str] = []
        write_pos = 0
        skipped_windows = 0

        # ── Pass 2: fill arrays one session at a time ───────────────────
        for sess_idx, session in enumerate(sessions):
            if window_counts[sess_idx] == 0:
                continue

            data = _prepare_session_data(session)

            trials = session.trials if include_invalid else session.get_valid_trials()
            window_samples = int(window_size_ms * session.metadata.sampling_rate / 1000)
            stride_samples = int(window_stride_ms * session.metadata.sampling_rate / 1000)

            for trial in trials:
                trial_data = data[trial.start_sample:trial.end_sample]

                for start in range(0, len(trial_data) - window_samples + 1, stride_samples):
                    window = trial_data[start:start + window_samples]
                    # Validate shape
                    if window.shape != expected_shape:
                        skipped_windows += 1
                        continue
                    X[write_pos] = window
                    y[write_pos] = trial.gesture_label
                    trial_id_list.append(
                        f"{session.metadata.session_id}_{trial.trial_id}")
                    write_pos += 1

            # Release the session data we just used
            del data

        # Trim if we skipped any windows
        if skipped_windows > 0:
            print(f"  Skipped {skipped_windows} windows with inconsistent shape.")
        if write_pos < total_windows:
            X = X[:write_pos]
            y = y[:write_pos]

        if write_pos == 0:
            raise ValueError("No valid windows after filtering inconsistent shapes")

        # Store original shape info before potential feature extraction
        raw_window_samples = X.shape[1] if X.ndim > 1 else len(X)
        raw_num_channels = X.shape[2] if X.ndim > 2 else (sessions[0].metadata.num_channels if sessions else 0)

        # Pre-extract features if requested
        features_extracted = False
        feature_dim = 0
        if feature_config is not None and feature_config.get("mode") != "raw" and X.ndim == 3:
            extractor = EMGFeatureExtractor(feature_config)
            X = extractor.extract_features(X)
            features_extracted = True
            feature_dim = X.shape[1] if X.ndim == 2 else 0

        # Get gesture names from first session (use display_name for nicer UI)
        gesture_set = sessions[0].gesture_set
        label_names = {g.label_id: g.display_name for g in gesture_set.gestures}

        metadata = {
            "name": name,
            "created_at": datetime.now().isoformat(),
            "num_samples": len(X),
            "num_classes": len(np.unique(y)),
            "window_size_ms": window_size_ms,
            "window_stride_ms": window_stride_ms,
            "sampling_rate": sessions[0].metadata.sampling_rate if sessions else 2000,
            "num_channels": raw_num_channels,
            "window_samples": raw_window_samples,
            "label_names": label_names,
            "sessions_used": [s.metadata.session_id for s in sessions],
            "calibration_applied": calibration is not None or use_per_session_rotation,
            "calibration_rotation_offset": (
                calibration.rotation_offset if calibration is not None else 0
            ),
            "per_session_rotation": use_per_session_rotation,
            "session_rotation_offsets": per_session_rotations,
            "features_extracted": features_extracted,
            "feature_config": feature_config if feature_config else None,
            "feature_dim": feature_dim,
        }

        dataset = {
            "X": X,
            "y": y,
            "trial_ids": np.array(trial_id_list),
            "metadata": metadata
        }

        return dataset

    def save_dataset(self, dataset: Dict[str, Any]) -> Path:
        """
        Save a dataset to disk.

        Args:
            dataset: Dataset dictionary from create_dataset

        Returns:
            Path where the dataset was saved
        """
        name = dataset["metadata"]["name"]
        dataset_path = self.datasets_dir / name
        dataset_path.mkdir(exist_ok=True)

        np.save(dataset_path / "X.npy", dataset["X"])
        np.save(dataset_path / "y.npy", dataset["y"])
        np.save(dataset_path / "trial_ids.npy", dataset["trial_ids"])

        with open(dataset_path / "metadata.json", 'w') as f:
            json.dump(dataset["metadata"], f, indent=2)

        return dataset_path

    def load_dataset(
        self,
        name: str,
        mmap: bool = False,
    ) -> Dict[str, Any]:
        """
        Load a dataset from disk.

        Args:
            name: Name of the dataset
            mmap: If True, memory-map the X array so only the pages
                  that are actually accessed are loaded into RAM.
                  Useful for very large datasets.

        Returns:
            Dataset dictionary
        """
        dataset_path = self.datasets_dir / name

        with open(dataset_path / "metadata.json", 'r') as f:
            metadata = json.load(f)

        mmap_mode = "r" if mmap else None

        return {
            "X": np.load(dataset_path / "X.npy", mmap_mode=mmap_mode),
            "y": np.load(dataset_path / "y.npy"),
            "trial_ids": np.load(dataset_path / "trial_ids.npy"),
            "metadata": metadata
        }

    def list_datasets(self) -> List[str]:
        """List all available datasets."""
        return [d.name for d in self.datasets_dir.iterdir() if d.is_dir()]

    def delete_dataset(self, name: str) -> bool:
        """
        Delete a dataset by name.

        Args:
            name: Name of the dataset to delete

        Returns:
            True if successfully deleted
        """
        dataset_path = self.datasets_dir / name
        if dataset_path.exists() and dataset_path.is_dir():
            shutil.rmtree(dataset_path)
            return True
        return False

    def get_train_test_split(
        self,
        dataset: Dict[str, Any],
        test_ratio: float = 0.2,
        stratify: bool = True,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split dataset into training and test sets.

        Args:
            dataset: Dataset dictionary
            test_ratio: Ratio of test samples
            stratify: Whether to stratify by class
            random_state: Random seed for reproducibility

        Returns:
            X_train, X_test, y_train, y_test
        """


        X = dataset["X"]
        y = dataset["y"]

        stratify_param = y if stratify else None

        return train_test_split(
            X, y,
            test_size=test_ratio,
            stratify=stratify_param,
            random_state=random_state
        )
