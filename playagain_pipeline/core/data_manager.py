"""
Data management for gesture pipeline.

Handles loading, saving, and organizing recording sessions and datasets.
"""

from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Callable
from datetime import datetime
import json
import numpy as np

from playagain_pipeline.core.session import RecordingSession
from playagain_pipeline.core.gesture import GestureSet


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

        for d in [self.sessions_dir, self.datasets_dir,
                  self.calibrations_dir, self.models_dir]:
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
        """List all subjects with recorded data."""
        return [d.name for d in self.sessions_dir.iterdir() if d.is_dir()]

    def list_sessions(self, subject_id: str) -> List[str]:
        """List all sessions for a subject."""
        subject_dir = self.sessions_dir / subject_id
        if not subject_dir.exists():
            return []
        return [d.name for d in subject_dir.iterdir() if d.is_dir()]

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
        preprocessing_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None
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

        Returns:
            Dictionary containing:
            - X: Feature array (windows x samples x channels)
            - y: Label array
            - metadata: Dataset metadata
        """
        if sessions is None:
            if subject_ids is None:
                subject_ids = self.list_subjects()
            sessions = []
            for sid in subject_ids:
                sessions.extend(self.get_all_sessions(sid))

        windows = []
        labels = []
        trial_ids = []

        for session in sessions:
            data = session.get_data()

            # Apply preprocessing if provided
            if preprocessing_fn is not None:
                data = preprocessing_fn(data)

            trials = session.trials if include_invalid else session.get_valid_trials()

            window_samples = int(window_size_ms * session.metadata.sampling_rate / 1000)
            stride_samples = int(window_stride_ms * session.metadata.sampling_rate / 1000)

            for trial in trials:
                trial_data = data[trial.start_sample:trial.end_sample]

                # Extract windows
                for start in range(0, len(trial_data) - window_samples + 1, stride_samples):
                    window = trial_data[start:start + window_samples]
                    windows.append(window)
                    labels.append(trial.gesture_label)
                    trial_ids.append(f"{session.metadata.session_id}_{trial.trial_id}")

        if not windows:
            raise ValueError("No data windows extracted")

        X = np.array(windows)
        y = np.array(labels)

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
            "num_channels": X.shape[2] if X.ndim > 2 else 1,
            "window_samples": X.shape[1] if X.ndim > 1 else len(X),
            "label_names": label_names,
            "sessions_used": [s.metadata.session_id for s in sessions]
        }

        dataset = {
            "X": X,
            "y": y,
            "trial_ids": np.array(trial_ids),
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

    def load_dataset(self, name: str) -> Dict[str, Any]:
        """
        Load a dataset from disk.

        Args:
            name: Name of the dataset

        Returns:
            Dataset dictionary
        """
        dataset_path = self.datasets_dir / name

        with open(dataset_path / "metadata.json", 'r') as f:
            metadata = json.load(f)

        return {
            "X": np.load(dataset_path / "X.npy"),
            "y": np.load(dataset_path / "y.npy"),
            "trial_ids": np.load(dataset_path / "trial_ids.npy"),
            "metadata": metadata
        }

    def list_datasets(self) -> List[str]:
        """List all available datasets."""
        return [d.name for d in self.datasets_dir.iterdir() if d.is_dir()]

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
        from sklearn.model_selection import train_test_split

        X = dataset["X"]
        y = dataset["y"]

        stratify_param = y if stratify else None

        return train_test_split(
            X, y,
            test_size=test_ratio,
            stratify=stratify_param,
            random_state=random_state
        )

