#!/usr/bin/env python3
"""
Advanced dataset creation with custom preprocessing.

This script demonstrates how to edit EMG data during the sessions-to-datasets
transformation process using custom preprocessing functions.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from pathlib import Path
from playagain_pipeline.core.data_manager import DataManager
from scipy import signal
from sklearn.preprocessing import StandardScaler
import json
from typing import List, Dict, Any


def bandpass_filter(data: np.ndarray, low_freq: float = 20, high_freq: float = 500, fs: float = 2000) -> np.ndarray:
    """Apply bandpass filter to EMG data."""
    nyquist = fs / 2
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    return signal.filtfilt(b, a, data, axis=0)


def rectify_and_smooth(data: np.ndarray, window_size: int = 50) -> np.ndarray:
    """Rectify and apply moving average smoothing."""
    rectified = np.abs(data)
    kernel = np.ones(window_size) / window_size
    smoothed = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='same'), axis=0, arr=rectified)
    return smoothed


def normalize_channels(data: np.ndarray) -> np.ndarray:
    """Z-score normalize each channel independently."""
    scaler = StandardScaler()
    # Reshape to (samples * channels,) for fitting, then transform back
    original_shape = data.shape
    data_2d = data.reshape(-1, data.shape[-1])
    normalized = scaler.fit_transform(data_2d)
    return normalized.reshape(original_shape)


def extract_rms_features(data: np.ndarray, window_size: int = 50) -> np.ndarray:
    """Extract RMS features from EMG data."""
    def rms_window(arr):
        return np.sqrt(np.mean(arr**2, axis=0))

    rms_features = []
    for i in range(0, len(data) - window_size + 1, window_size // 2):
        window = data[i:i + window_size]
        rms = rms_window(window)
        rms_features.append(rms)

    return np.array(rms_features)


def add_rest_periods_to_session(session_dir: Path) -> Dict[str, Any]:
    """
    Update a session's metadata to include rest periods between gesture trials.

    Rest periods are automatically inserted between the end of one gesture trial
    and the start of the next gesture trial.

    Args:
        session_dir: Path to the session directory containing metadata.json

    Returns:
        Updated metadata dictionary with rest periods added
    """
    metadata_path = session_dir / "metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(f"No metadata.json found in {session_dir}")

    with open(metadata_path, 'r') as f:
        data = json.load(f)

    original_trials = data["trials"]
    new_trials = []
    trial_id = 0

    # Add rest trials between gesture trials
    for i, trial in enumerate(original_trials):
        # Insert rest period before this gesture if not the first trial
        if i > 0:
            prev_trial = original_trials[i - 1]
            rest_start_sample = prev_trial["end_sample"]
            rest_end_sample = trial["start_sample"]
            rest_start_time = prev_trial["end_time"]
            rest_end_time = trial["start_time"]

            # Only add if there's a gap
            if rest_end_sample > rest_start_sample:
                new_trials.append({
                    "trial_id": trial_id,
                    "gesture_name": "rest",
                    "gesture_label": 0,  # Rest is label 0
                    "start_sample": rest_start_sample,
                    "end_sample": rest_end_sample,
                    "start_time": rest_start_time,
                    "end_time": rest_end_time,
                    "is_valid": True,
                    "notes": "Rest period (auto-generated)"
                })
                trial_id += 1

        # Update gesture trial ID and adjust gesture label (shift by 1 since rest is 0)
        updated_trial = trial.copy()
        updated_trial["trial_id"] = trial_id
        # Shift gesture labels: rest=0, fist=1, index_thumb=2, three_finger_thumb=3
        if updated_trial["gesture_name"] == "fist":
            updated_trial["gesture_label"] = 1
        elif updated_trial["gesture_name"] == "index_thumb":
            updated_trial["gesture_label"] = 2
        elif updated_trial["gesture_name"] == "three_finger_thumb":
            updated_trial["gesture_label"] = 3

        new_trials.append(updated_trial)
        trial_id += 1

    # Update metadata
    data["trials"] = new_trials
    notes = data["metadata"].get("notes", "")
    if "[Rest periods added]" not in notes:
        data["metadata"]["notes"] = (notes + " [Rest periods added]").strip()

    # Save updated metadata
    with open(metadata_path, 'w') as f:
        json.dump(data, f, indent=2)

    return data


def update_all_sessions_with_rest(sessions_root: Path) -> None:
    """
    Update all sessions in a sessions directory to include rest periods.

    Args:
        sessions_root: Path to root sessions directory (e.g., gesture_pipeline/data/sessions)
    """
    sessions_root = Path(sessions_root)

    if not sessions_root.exists():
        raise FileNotFoundError(f"Sessions directory not found: {sessions_root}")

    updated_count = 0
    error_count = 0

    # Iterate through all subject directories
    for subject_dir in sessions_root.iterdir():
        if not subject_dir.is_dir():
            continue

        print(f"\nProcessing subject: {subject_dir.name}")

        # Iterate through all session directories
        for session_dir in subject_dir.iterdir():
            if not session_dir.is_dir():
                continue

            try:
                updated_data = add_rest_periods_to_session(session_dir)
                num_rest_trials = sum(1 for t in updated_data["trials"] if t["gesture_name"] == "rest")
                num_gesture_trials = len(updated_data["trials"]) - num_rest_trials
                print(f"  ✓ {session_dir.name}: {num_gesture_trials} gestures + {num_rest_trials} rest periods")
                updated_count += 1
            except Exception as e:
                print(f"  ✗ {session_dir.name}: Error - {e}")
                error_count += 1

    print(f"\n{'='*60}")
    print(f"Summary: {updated_count} sessions updated, {error_count} errors")
    print(f"{'='*60}")


def main():
    """Demonstrate advanced dataset creation with preprocessing."""

    # Initialize data manager
    data_dir = Path("gesture_pipeline_data")
    data_manager = DataManager(data_dir)

    print("Available subjects:", data_manager.list_subjects())

    # Example 1: Basic preprocessing (filtering + rectification)
    print("\n=== Example 1: Filtered and Rectified Dataset ===")

    def basic_preprocessing(data):
        # Bandpass filter
        filtered = bandpass_filter(data, fs=2000)
        # Rectify and smooth
        processed = rectify_and_smooth(filtered)
        return processed

    dataset1 = data_manager.create_dataset(
        name="filtered_rectified_dataset",
        subject_ids=["VP_00", "VP_01"],  # Specify subjects
        window_size_ms=200,
        window_stride_ms=50,
        preprocessing_fn=basic_preprocessing
    )

    data_manager.save_dataset(dataset1)
    print(f"Created dataset with {dataset1['metadata']['num_samples']} samples")

    # Example 2: Feature extraction (RMS features)
    print("\n=== Example 2: RMS Feature Dataset ===")

    def rms_preprocessing(data):
        # First apply basic preprocessing
        filtered = bandpass_filter(data, fs=2000)
        rectified = np.abs(filtered)
        # Extract RMS features (this changes the data structure)
        return extract_rms_features(rectified, window_size=40)  # 40 samples at 2000Hz = 20ms

    dataset2 = data_manager.create_dataset(
        name="rms_features_dataset",
        subject_ids=["VP_00"],  # Only one subject
        window_size_ms=100,     # Larger windows for RMS
        window_stride_ms=50,
        preprocessing_fn=rms_preprocessing
    )

    data_manager.save_dataset(dataset2)
    print(f"Created RMS dataset with {dataset2['metadata']['num_samples']} samples")
    print(f"Feature shape: {dataset2['X'].shape}")

    # Example 3: Multi-stage preprocessing pipeline
    print("\n=== Example 3: Advanced Preprocessing Pipeline ===")

    def advanced_preprocessing(data):
        # Stage 1: Filtering
        filtered = bandpass_filter(data, low_freq=30, high_freq=400, fs=2000)

        # Stage 2: Rectification
        rectified = np.abs(filtered)

        # Stage 3: Normalization per channel
        normalized = normalize_channels(rectified)

        # Stage 4: Moving average smoothing
        smoothed = rectify_and_smooth(normalized, window_size=30)

        return smoothed

    dataset3 = data_manager.create_dataset(
        name="advanced_preprocessing_dataset",
        window_size_ms=250,     # Longer windows
        window_stride_ms=100,   # Overlapping windows
        include_invalid=False,  # Only valid trials
        preprocessing_fn=advanced_preprocessing
    )

    data_manager.save_dataset(dataset3)
    print(f"Created advanced dataset with {dataset3['metadata']['num_samples']} samples")

    # Example 4: Custom session selection and editing
    print("\n=== Example 4: Custom Session Selection ===")

    # Load specific sessions
    sessions = []
    sessions.append(data_manager.load_session("VP_00", "session_20260204_101826"))
    sessions.append(data_manager.load_session("VP_01", "session_20260204_102914"))

    # You can modify sessions here before creating dataset
    for session in sessions:
        print(f"Session: {session.metadata.session_id}")
        print(f"  Trials: {len(session.trials)}")
        print(f"  Valid trials: {len(session.get_valid_trials())}")
        print(f"  Duration: {session.duration_seconds:.1f}s")

    dataset4 = data_manager.create_dataset(
        name="custom_sessions_dataset",
        sessions=sessions,  # Use specific sessions
        window_size_ms=150,
        window_stride_ms=75,
        preprocessing_fn=basic_preprocessing
    )

    data_manager.save_dataset(dataset4)
    print(f"Created custom dataset with {dataset4['metadata']['num_samples']} samples")

    print("\n=== Available Datasets ===")
    for dataset_name in data_manager.list_datasets():
        dataset = data_manager.load_dataset(dataset_name)
        meta = dataset['metadata']
        print(f"- {dataset_name}: {meta['num_samples']} samples, {meta['num_classes']} classes")


if __name__ == "__main__":
    main()
