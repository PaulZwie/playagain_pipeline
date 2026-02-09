"""
Machine learning models for gesture classification.

This module provides a modular framework for training and using
different ML models for EMG gesture recognition.
"""

import json
import pickle
import numpy as np

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from catboost import CatBoostClassifier as CatBoostWrapper
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class ModelMetadata:
    """Metadata for a trained model."""
    name: str
    model_type: str
    created_at: datetime
    num_classes: int
    num_channels: int
    window_size_ms: int
    sampling_rate: int
    training_accuracy: float = 0.0
    validation_accuracy: float = 0.0
    class_names: Dict[int, str] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    training_history: Dict[str, List[float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model_type": self.model_type,
            "created_at": self.created_at.isoformat(),
            "num_classes": self.num_classes,
            "num_channels": self.num_channels,
            "window_size_ms": self.window_size_ms,
            "sampling_rate": self.sampling_rate,
            "training_accuracy": self.training_accuracy,
            "validation_accuracy": self.validation_accuracy,
            "class_names": self.class_names,
            "hyperparameters": self.hyperparameters,
            "training_history": self.training_history
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        data = data.copy()
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        # Convert string keys back to int for class_names
        data["class_names"] = {int(k): v for k, v in data["class_names"].items()}
        return cls(**data)


class BaseClassifier(ABC):
    """
    Abstract base class for gesture classifiers.

    Provides a unified interface for different ML models.
    """

    def __init__(self, name: str = "base_classifier"):
        self.name = name
        self.metadata: Optional[ModelMetadata] = None
        self._model = None
        self._is_trained = False

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @abstractmethod
    def extract_features(self, X: np.ndarray) -> np.ndarray:
        """
        Extract features from raw EMG windows.

        Args:
            X: Raw EMG data (windows, samples, channels)

        Returns:
            Feature array
        """
        pass

    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            X_train: Training data
            y_train: Training labels
            X_val: Validation data
            y_val: Validation labels
            **kwargs: Additional training parameters

        Returns:
            Training results/history
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Input data

        Returns:
            Predicted labels
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.

        Args:
            X: Input data

        Returns:
            Class probabilities
        """
        pass

    def save(self, path: Path) -> None:
        """Save model to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save metadata
        if self.metadata:
            with open(path / "metadata.json", 'w') as f:
                json.dump(self.metadata.to_dict(), f, indent=2)

        # Save model
        with open(path / "model.pkl", 'wb') as f:
            pickle.dump(self._model, f)

    def load(self, path: Path) -> None:
        """Load model from disk."""
        path = Path(path)

        # Load metadata
        with open(path / "metadata.json", 'r') as f:
            self.metadata = ModelMetadata.from_dict(json.load(f))

        # Load model
        with open(path / "model.pkl", 'rb') as f:
            self._model = pickle.load(f)

        self._is_trained = True


class EMGFeatureExtractor:
    """
    Extracts time-domain and frequency-domain features from EMG signals.
    """

    @staticmethod
    def compute_rms(data: np.ndarray, axis: int = 1) -> np.ndarray:
        """Compute Root Mean Square."""
        return np.sqrt(np.mean(data ** 2, axis=axis))

    @staticmethod
    def compute_mav(data: np.ndarray, axis: int = 1) -> np.ndarray:
        """Compute Mean Absolute Value."""
        return np.mean(np.abs(data), axis=axis)

    @staticmethod
    def compute_var(data: np.ndarray, axis: int = 1) -> np.ndarray:
        """Compute Variance."""
        return np.var(data, axis=axis)

    @staticmethod
    def compute_wl(data: np.ndarray, axis: int = 1) -> np.ndarray:
        """Compute Waveform Length."""
        return np.sum(np.abs(np.diff(data, axis=axis)), axis=axis)

    @staticmethod
    def compute_zc(data: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """Compute Zero Crossings."""
        # For each window and channel
        if data.ndim == 3:
            zc = np.zeros((data.shape[0], data.shape[2]))
            for i in range(data.shape[0]):
                for j in range(data.shape[2]):
                    signal = data[i, :, j]
                    zc[i, j] = np.sum(
                        (np.abs(np.diff(np.sign(signal))) > 0) &
                        (np.abs(np.diff(signal)) > threshold)
                    )
            return zc
        else:
            signal = data
            return np.sum(
                (np.abs(np.diff(np.sign(signal), axis=0)) > 0) &
                (np.abs(np.diff(signal, axis=0)) > threshold),
                axis=0
            )

    @staticmethod
    def compute_ssc(data: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """Compute Slope Sign Changes."""
        if data.ndim == 3:
            ssc = np.zeros((data.shape[0], data.shape[2]))
            for i in range(data.shape[0]):
                for j in range(data.shape[2]):
                    signal = data[i, :, j]
                    diff = np.diff(signal)
                    ssc[i, j] = np.sum(
                        (diff[:-1] * diff[1:] < 0) &
                        (np.abs(diff[:-1] - diff[1:]) > threshold)
                    )
            return ssc
        else:
            diff = np.diff(data, axis=0)
            return np.sum(
                (diff[:-1] * diff[1:] < 0) &
                (np.abs(diff[:-1] - diff[1:]) > threshold),
                axis=0
            )

    @classmethod
    def extract_all_features(cls, data: np.ndarray) -> np.ndarray:
        """
        Extract all time-domain features.

        Args:
            data: EMG data (windows, samples, channels)

        Returns:
            Feature array (windows, features)
        """
        if data.ndim != 3:
            raise ValueError("Expected 3D array (windows, samples, channels)")

        features = []

        # Time-domain features
        features.append(cls.compute_rms(data))
        features.append(cls.compute_mav(data))
        features.append(cls.compute_var(data))
        features.append(cls.compute_wl(data))
        features.append(cls.compute_zc(data))
        features.append(cls.compute_ssc(data))

        # Stack features
        return np.hstack(features)


class SVMClassifier(BaseClassifier):
    """
    Support Vector Machine classifier for gesture recognition.
    """

    def __init__(self, name: str = "svm_classifier", **kwargs):
        super().__init__(name)
        self.hyperparameters = {
            "kernel": kwargs.get("kernel", "rbf"),
            "C": kwargs.get("C", 1.0),
            "gamma": kwargs.get("gamma", "scale"),
            "probability": True
        }
        self._feature_extractor = EMGFeatureExtractor()

    def extract_features(self, X: np.ndarray) -> np.ndarray:
        """Extract features from raw EMG."""
        return self._feature_extractor.extract_all_features(X)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Train the SVM model."""
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler
        import time

        start_time = time.time()

        # Extract features
        X_train_features = self.extract_features(X_train)
        feature_count = X_train_features.shape[1]

        # Normalize features
        self._scaler = StandardScaler()
        X_train_scaled = self._scaler.fit_transform(X_train_features)

        # Create and train model
        self._model = SVC(**self.hyperparameters)
        self._model.fit(X_train_scaled, y_train)

        # Compute training accuracy
        train_pred = self._model.predict(X_train_scaled)
        train_acc = np.mean(train_pred == y_train)

        # Compute validation accuracy if provided
        val_acc = 0.0
        if X_val is not None and y_val is not None:
            X_val_features = self.extract_features(X_val)
            X_val_scaled = self._scaler.transform(X_val_features)
            val_pred = self._model.predict(X_val_scaled)
            val_acc = np.mean(val_pred == y_val)

        training_time = time.time() - start_time
        self._is_trained = True

        # Create metadata
        self.metadata = ModelMetadata(
            name=self.name,
            model_type="SVM",
            created_at=datetime.now(),
            num_classes=len(np.unique(y_train)),
            num_channels=X_train.shape[2],
            window_size_ms=kwargs.get("window_size_ms", 200),
            sampling_rate=kwargs.get("sampling_rate", 2000),
            training_accuracy=train_acc,
            validation_accuracy=val_acc,
            hyperparameters=self.hyperparameters
        )

        return {
            "training_accuracy": train_acc,
            "validation_accuracy": val_acc,
            "feature_count": feature_count,
            "training_time": training_time
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self._is_trained:
            raise ValueError("Model not trained")

        X_features = self.extract_features(X)
        X_scaled = self._scaler.transform(X_features)
        return self._model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if not self._is_trained:
            raise ValueError("Model not trained")

        X_features = self.extract_features(X)
        X_scaled = self._scaler.transform(X_features)
        return self._model.predict_proba(X_scaled)

    def save(self, path: Path) -> None:
        """Save model and scaler."""
        super().save(path)
        with open(Path(path) / "scaler.pkl", 'wb') as f:
            pickle.dump(self._scaler, f)

    def load(self, path: Path) -> None:
        """Load model and scaler."""
        super().load(path)
        with open(Path(path) / "scaler.pkl", 'rb') as f:
            self._scaler = pickle.load(f)


class RandomForestClassifier(BaseClassifier):
    """
    Random Forest classifier for gesture recognition.
    """

    def __init__(self, name: str = "rf_classifier", **kwargs):
        super().__init__(name)
        self.hyperparameters = {
            "n_estimators": kwargs.get("n_estimators", 100),
            "max_depth": kwargs.get("max_depth", None),
            "min_samples_split": kwargs.get("min_samples_split", 2),
            "min_samples_leaf": kwargs.get("min_samples_leaf", 1),
            "random_state": kwargs.get("random_state", 42)
        }
        self._feature_extractor = EMGFeatureExtractor()

    def extract_features(self, X: np.ndarray) -> np.ndarray:
        """Extract features from raw EMG."""
        return self._feature_extractor.extract_all_features(X)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Train the Random Forest model."""
        from sklearn.ensemble import RandomForestClassifier as RFClassifier
        import time

        start_time = time.time()

        # Extract features
        X_train_features = self.extract_features(X_train)
        feature_count = X_train_features.shape[1]

        # Create and train model
        self._model = RFClassifier(**self.hyperparameters)
        self._model.fit(X_train_features, y_train)

        # Compute training accuracy
        train_pred = self._model.predict(X_train_features)
        train_acc = np.mean(train_pred == y_train)

        # Compute validation accuracy if provided
        val_acc = 0.0
        if X_val is not None and y_val is not None:
            X_val_features = self.extract_features(X_val)
            val_pred = self._model.predict(X_val_features)
            val_acc = np.mean(val_pred == y_val)

        training_time = time.time() - start_time
        self._is_trained = True

        # Create metadata
        self.metadata = ModelMetadata(
            name=self.name,
            model_type="RandomForest",
            created_at=datetime.now(),
            num_classes=len(np.unique(y_train)),
            num_channels=X_train.shape[2],
            window_size_ms=kwargs.get("window_size_ms", 200),
            sampling_rate=kwargs.get("sampling_rate", 2000),
            training_accuracy=train_acc,
            validation_accuracy=val_acc,
            hyperparameters=self.hyperparameters
        )

        return {
            "training_accuracy": train_acc,
            "validation_accuracy": val_acc,
            "feature_count": feature_count,
            "training_time": training_time,
            "feature_importances": self._model.feature_importances_.tolist()
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self._is_trained:
            raise ValueError("Model not trained")

        X_features = self.extract_features(X)
        return self._model.predict(X_features)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if not self._is_trained:
            raise ValueError("Model not trained")

        X_features = self.extract_features(X)
        return self._model.predict_proba(X_features)


class LDAClassifier(BaseClassifier):
    """
    Linear Discriminant Analysis classifier for gesture recognition.
    """

    def __init__(self, name: str = "lda_classifier", **kwargs):
        super().__init__(name)
        self.hyperparameters = {
            "solver": kwargs.get("solver", "svd"),
            "shrinkage": kwargs.get("shrinkage", None)
        }
        self._feature_extractor = EMGFeatureExtractor()

    def extract_features(self, X: np.ndarray) -> np.ndarray:
        """Extract features from raw EMG."""
        return self._feature_extractor.extract_all_features(X)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Train the LDA model."""
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.preprocessing import StandardScaler
        import time

        start_time = time.time()

        # Extract features
        X_train_features = self.extract_features(X_train)
        feature_count = X_train_features.shape[1]

        # Normalize features
        self._scaler = StandardScaler()
        X_train_scaled = self._scaler.fit_transform(X_train_features)

        # Create and train model
        self._model = LinearDiscriminantAnalysis(**self.hyperparameters)
        self._model.fit(X_train_scaled, y_train)

        # Compute training accuracy
        train_pred = self._model.predict(X_train_scaled)
        train_acc = np.mean(train_pred == y_train)

        # Compute validation accuracy if provided
        val_acc = 0.0
        if X_val is not None and y_val is not None:
            X_val_features = self.extract_features(X_val)
            X_val_scaled = self._scaler.transform(X_val_features)
            val_pred = self._model.predict(X_val_scaled)
            val_acc = np.mean(val_pred == y_val)

        training_time = time.time() - start_time
        self._is_trained = True

        # Create metadata
        self.metadata = ModelMetadata(
            name=self.name,
            model_type="LDA",
            created_at=datetime.now(),
            num_classes=len(np.unique(y_train)),
            num_channels=X_train.shape[2],
            window_size_ms=kwargs.get("window_size_ms", 200),
            sampling_rate=kwargs.get("sampling_rate", 2000),
            training_accuracy=train_acc,
            validation_accuracy=val_acc,
            hyperparameters=self.hyperparameters
        )

        return {
            "training_accuracy": train_acc,
            "validation_accuracy": val_acc,
            "feature_count": feature_count,
            "training_time": training_time
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self._is_trained:
            raise ValueError("Model not trained")

        X_features = self.extract_features(X)
        X_scaled = self._scaler.transform(X_features)
        return self._model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if not self._is_trained:
            raise ValueError("Model not trained")

        X_features = self.extract_features(X)
        X_scaled = self._scaler.transform(X_features)
        return self._model.predict_proba(X_scaled)

    def save(self, path: Path) -> None:
        """Save model and scaler."""
        super().save(path)
        with open(Path(path) / "scaler.pkl", 'wb') as f:
            pickle.dump(self._scaler, f)

    def load(self, path: Path) -> None:
        """Load model and scaler."""
        super().load(path)
        with open(Path(path) / "scaler.pkl", 'rb') as f:
            self._scaler = pickle.load(f)


class MLPClassifier(BaseClassifier):
    """
    Multi-Layer Perceptron (Neural Network) classifier.
    Supports advanced training options like epochs, early stopping, and various optimizers.
    """

    def __init__(self, name: str = "mlp_classifier", **kwargs):
        super().__init__(name)
        self.hyperparameters = {
            "hidden_layers": kwargs.get("hidden_layers", (128, 64)),
            "activation": kwargs.get("activation", "relu"),
            "dropout": kwargs.get("dropout", 0.2),
            "optimizer": kwargs.get("optimizer", "adam"),  # adam, rmsprop, sgd
            "learning_rate": kwargs.get("learning_rate", 0.001),
            "batch_size": kwargs.get("batch_size", 32),
            "epochs": kwargs.get("epochs", 100),
            "early_stopping": kwargs.get("early_stopping", True),
            "patience": kwargs.get("patience", 10),
            "device": kwargs.get("device", "mps")  # "cpu" or "cuda"
        }
        self._feature_extractor = EMGFeatureExtractor()
        self._scaler = None
        self._model = None
        self._input_dim = 0
        self._output_dim = 0

    def extract_features(self, X: np.ndarray) -> np.ndarray:
        """Extract features from raw EMG."""
        return self._feature_extractor.extract_all_features(X)

    def _build_model(self, input_dim: int, output_dim: int):
        """Build PyTorch model architecture."""
        layers = []
        in_dim = input_dim

        # Hidden layers
        for hidden_dim in self.hyperparameters["hidden_layers"]:
            layers.append(nn.Linear(in_dim, hidden_dim))

            if self.hyperparameters["activation"] == "relu":
                layers.append(nn.ReLU())
            elif self.hyperparameters["activation"] == "tanh":
                layers.append(nn.Tanh())

            if self.hyperparameters["dropout"] > 0:
                layers.append(nn.Dropout(self.hyperparameters["dropout"]))

            in_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(in_dim, output_dim))

        return nn.Sequential(*layers)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Train the MLP model."""
        import time

        start_time = time.time()

        # Callback for progress reporting
        callback = kwargs.get("callback", None)

        # Extract features
        X_train_features = self.extract_features(X_train)
        feature_count = X_train_features.shape[1]

        # Normalize features
        self._scaler = StandardScaler()
        X_train_scaled = self._scaler.fit_transform(X_train_features)

        # Prepare validation data
        if X_val is not None:
            X_val_features = self.extract_features(X_val)
            X_val_scaled = self._scaler.transform(X_val_features)
        else:
            X_val_scaled = None

        # Convert to PyTorch tensors
        device = torch.device(self.hyperparameters["device"] if torch.cuda.is_available() else "cpu")

        X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
        y_train_tensor = torch.LongTensor(y_train).to(device)

        X_val_tensor = None
        y_val_tensor = None

        if X_val_scaled is not None:
            X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
            y_val_tensor = torch.LongTensor(y_val).to(device)

        self._input_dim = X_train_scaled.shape[1]
        self._output_dim = len(np.unique(y_train))
        self._model = self._build_model(self._input_dim, self._output_dim).to(device)

        # Setup optimizer
        opt_name = self.hyperparameters["optimizer"].lower()
        lr = self.hyperparameters["learning_rate"]

        if opt_name == "adam":
            optimizer = optim.Adam(self._model.parameters(), lr=lr)
        elif opt_name == "rmsprop":
            optimizer = optim.RMSprop(self._model.parameters(), lr=lr)
        else:
            optimizer = optim.SGD(self._model.parameters(), lr=lr, momentum=0.9)

        criterion = nn.CrossEntropyLoss()

        # Training loop
        batch_size = self.hyperparameters["batch_size"]
        epochs = self.hyperparameters["epochs"]
        dataset = TensorDataset(X_train_tensor, y_train_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        best_val_loss = float('inf')
        patience_counter = 0
        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

        for epoch in range(epochs):
            self._model.train()
            train_loss = 0.0
            correct = 0
            total = 0

            for inputs, targets in dataloader:
                optimizer.zero_grad()
                outputs = self._model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            train_loss /= len(dataloader.dataset)
            train_acc = correct / total

            # Validation
            val_loss = 0.0
            val_acc = 0.0

            if X_val_scaled is not None:
                self._model.eval()
                with torch.no_grad():
                    outputs = self._model(X_val_tensor)
                    loss = criterion(outputs, y_val_tensor)
                    val_loss = loss.item()
                    _, predicted = outputs.max(1)
                    val_acc = predicted.eq(y_val_tensor).sum().item() / y_val_tensor.size(0)

            # Store history
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            # Report progress
            if callback:
                callback(epoch + 1, train_loss, val_loss, train_acc, val_acc)

            # Early stopping
            if self.hyperparameters["early_stopping"] and X_val_scaled is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model state
                    # self._best_state = self._model.state_dict()
                else:
                    patience_counter += 1
                    if patience_counter >= self.hyperparameters["patience"]:
                        print(f"Early stopping at epoch {epoch}")
                        break

        training_time = time.time() - start_time
        self._is_trained = True

        # Create metadata
        self.metadata = ModelMetadata(
            name=self.name,
            model_type="MLP",
            created_at=datetime.now(),
            num_classes=self._output_dim,
            num_channels=X_train.shape[2],
            window_size_ms=kwargs.get("window_size_ms", 200),
            sampling_rate=kwargs.get("sampling_rate", 2000),
            training_accuracy=history["train_acc"][-1],
            validation_accuracy=history["val_acc"][-1] if history["val_acc"] else 0.0,
            hyperparameters=self.hyperparameters,
            training_history=history
        )

        return {
            "training_accuracy": history["train_acc"][-1],
            "validation_accuracy": history["val_acc"][-1] if history["val_acc"] else 0.0,
            "feature_count": feature_count,
            "training_time": training_time,
            "epochs_trained": len(history["train_loss"])
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self._is_trained:
            raise ValueError("Model not trained")

        X_features = self.extract_features(X)
        X_scaled = self._scaler.transform(X_features)

        device = torch.device(self.hyperparameters["device"] if torch.cuda.is_available() else "cpu")
        X_tensor = torch.FloatTensor(X_scaled).to(device)

        self._model.eval()
        with torch.no_grad():
            outputs = self._model(X_tensor)
            _, predicted = outputs.max(1)

        return predicted.cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if not self._is_trained:
            raise ValueError("Model not trained")

        X_features = self.extract_features(X)
        X_scaled = self._scaler.transform(X_features)

        device = torch.device(self.hyperparameters["device"] if torch.cuda.is_available() else "cpu")
        X_tensor = torch.FloatTensor(X_scaled).to(device)

        self._model.eval()
        with torch.no_grad():
            outputs = self._model(X_tensor)
            probs = torch.softmax(outputs, dim=1)

        return probs.cpu().numpy()

    def save(self, path: Path) -> None:
        """Save model and scaler."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save metadata
        if self.metadata:
            with open(path / "metadata.json", 'w') as f:
                json.dump(self.metadata.to_dict(), f, indent=2)

        # Save pytorch model
        torch.save(self._model.state_dict(), path / "model.pt")

        # Save scaler
        with open(path / "scaler.pkl", 'wb') as f:
            pickle.dump(self._scaler, f)

        # Save params to reconstruct model structure
        with open(path / "params.json", 'w') as f:
            json.dump({
                "input_dim": self._input_dim,
                "output_dim": self._output_dim,
                "hyperparameters": self.hyperparameters
            }, f)

    def load(self, path: Path) -> None:
        """Load model and scaler."""
        path = Path(path)

        # Load metadata
        with open(path / "metadata.json", 'r') as f:
            self.metadata = ModelMetadata.from_dict(json.load(f))

        # Load params
        with open(path / "params.json", 'r') as f:
            params = json.load(f)
            self._input_dim = params["input_dim"]
            self._output_dim = params["output_dim"]
            self.hyperparameters = params["hyperparameters"]

        # Load scaler
        with open(path / "scaler.pkl", 'rb') as f:
            self._scaler = pickle.load(f)

        # Rebuild and load model
        self._model = self._build_model(self._input_dim, self._output_dim)

        device = torch.device(self.hyperparameters["device"] if torch.cuda.is_available() else "cpu")
        self._model.load_state_dict(torch.load(path / "model.pt", map_location=device))
        self._model.to(device)

        self._is_trained = True


class CNNClassifier(BaseClassifier):
    def __init__(self, name: str = "cnn_classifier", **kwargs):
        super().__init__(name)
        self.hyperparameters = {
            "filters": kwargs.get("filters", [32, 64, 128]),
            "kernel_sizes": kwargs.get("kernel_sizes", [5, 3, 3]),
            "fc_layers": kwargs.get("fc_layers", (128,)),
            "activation": kwargs.get("activation", "relu"),
            "dropout": kwargs.get("dropout", 0.3),
            "optimizer": kwargs.get("optimizer", "adam"),
            "learning_rate": kwargs.get("learning_rate", 0.0005),
            "batch_size": kwargs.get("batch_size", 32),
            "epochs": kwargs.get("epochs", 100),
            "early_stopping": kwargs.get("early_stopping", True),
            "patience": kwargs.get("patience", 10),
            "device": kwargs.get("device", "cpu")
        }
        self._scaler = None
        self._model = None
        self._input_shape = None
        self._output_dim = 0

    def extract_features(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 3:
            return np.transpose(X, (0, 2, 1))
        return X

    def _build_model(self, input_channels: int, input_length: int, output_dim: int):
        layers = []
        in_channels = input_channels

        for out_channels, k_size in zip(self.hyperparameters["filters"], self.hyperparameters["kernel_sizes"]):
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=k_size, padding=k_size // 2))
            layers.append(nn.BatchNorm1d(out_channels))

            if self.hyperparameters["activation"] == "relu":
                layers.append(nn.ReLU())
            elif self.hyperparameters["activation"] == "tanh":
                layers.append(nn.Tanh())

            layers.append(nn.MaxPool1d(2))

            if self.hyperparameters["dropout"] > 0:
                layers.append(nn.Dropout(self.hyperparameters["dropout"]))

            in_channels = out_channels

        layers.append(nn.AdaptiveAvgPool1d(1))
        layers.append(nn.Flatten())

        current_dim = in_channels
        for hidden_dim in self.hyperparameters["fc_layers"]:
            layers.append(nn.Linear(current_dim, hidden_dim))
            if self.hyperparameters["activation"] == "relu":
                layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.hyperparameters["dropout"]))
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, output_dim))
        return nn.Sequential(*layers)

    def train(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            **kwargs
    ) -> Dict[str, Any]:
        import time
        start_time = time.time()
        callback = kwargs.get("callback", None)

        X_train_proc = self.extract_features(X_train)
        N, C, T = X_train_proc.shape

        # Scale per channel
        self._scaler = StandardScaler()
        X_train_flat = X_train_proc.transpose(0, 2, 1).reshape(-1, C)
        X_train_scaled_flat = self._scaler.fit_transform(X_train_flat)
        X_train_scaled = X_train_scaled_flat.reshape(N, T, C).transpose(0, 2, 1)

        X_val_scaled = None
        if X_val is not None:
            X_val_proc = self.extract_features(X_val)
            Nv, Cv, Tv = X_val_proc.shape
            X_val_flat = X_val_proc.transpose(0, 2, 1).reshape(-1, Cv)
            X_val_scaled_flat = self._scaler.transform(X_val_flat)
            X_val_scaled = X_val_scaled_flat.reshape(Nv, Tv, Cv).transpose(0, 2, 1)

        device = torch.device(self.hyperparameters["device"] if torch.cuda.is_available() else "cpu")

        X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
        y_train_tensor = torch.LongTensor(y_train).to(device)

        X_val_tensor = None
        y_val_tensor = None
        if X_val_scaled is not None:
            X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
            y_val_tensor = torch.LongTensor(y_val).to(device)

        self._input_shape = (C, T) # Ensure input shape is saved
        self._output_dim = len(np.unique(y_train))
        self._model = self._build_model(C, T, self._output_dim).to(device)

        lr = self.hyperparameters["learning_rate"]
        opt_name = self.hyperparameters["optimizer"]
        if opt_name == "adam":
            optimizer = optim.Adam(self._model.parameters(), lr=lr)
        elif opt_name == "rmsprop":
            optimizer = optim.RMSprop(self._model.parameters(), lr=lr)
        else:
            optimizer = optim.SGD(self._model.parameters(), lr=lr, momentum=0.9)

        criterion = nn.CrossEntropyLoss()
        dataset = TensorDataset(X_train_tensor, y_train_tensor)
        dataloader = DataLoader(dataset, batch_size=self.hyperparameters["batch_size"], shuffle=True)

        best_val_loss = float('inf')
        patience_counter = 0
        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

        for epoch in range(self.hyperparameters["epochs"]):
            self._model.train()
            train_loss = 0.0
            correct = 0
            total = 0

            for inputs, targets in dataloader:
                optimizer.zero_grad()
                outputs = self._model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            train_loss /= len(dataloader.dataset)
            train_acc = correct / total

            val_loss = 0.0
            val_acc = 0.0

            if X_val_scaled is not None:
                self._model.eval()
                with torch.no_grad():
                    outputs = self._model(X_val_tensor)
                    loss = criterion(outputs, y_val_tensor)
                    val_loss = loss.item()
                    _, predicted = outputs.max(1)
                    val_acc = predicted.eq(y_val_tensor).sum().item() / y_val_tensor.size(0)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            if callback:
                callback(epoch + 1, train_loss, val_loss, train_acc, val_acc)

            if self.hyperparameters["early_stopping"] and X_val_scaled is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.hyperparameters["patience"]:
                        print(f"Early stopping at epoch {epoch}")
                        break

        training_time = time.time() - start_time
        self._is_trained = True

        self.metadata = ModelMetadata(
            name=self.name,
            model_type="CNN",
            created_at=datetime.now(),
            num_classes=self._output_dim,
            num_channels=X_train.shape[2] if X_train.ndim > 2 else X_train.shape[1],
            window_size_ms=kwargs.get("window_size_ms", 200),
            sampling_rate=kwargs.get("sampling_rate", 2000),
            training_accuracy=history["train_acc"][-1],
            validation_accuracy=history["val_acc"][-1] if history["val_acc"] else 0.0,
            hyperparameters=self.hyperparameters,
            training_history=history
        )

        return {
            "training_accuracy": history["train_acc"][-1],
            "validation_accuracy": history["val_acc"][-1] if history["val_acc"] else 0.0,
            "feature_count": C,
            "training_time": training_time,
            "epochs_trained": len(history["train_loss"])
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_trained:
            raise ValueError("Model not trained")

        X_proc = self.extract_features(X)
        N, C, T = X_proc.shape
        X_flat = X_proc.transpose(0, 2, 1).reshape(-1, C)
        X_scaled_flat = self._scaler.transform(X_flat)
        X_scaled = X_scaled_flat.reshape(N, T, C).transpose(0, 2, 1)

        device = torch.device(self.hyperparameters["device"] if torch.cuda.is_available() else "cpu")
        X_tensor = torch.FloatTensor(X_scaled).to(device)

        self._model.eval()
        with torch.no_grad():
            outputs = self._model(X_tensor)
            _, predicted = outputs.max(1)

        return predicted.cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._is_trained:
            raise ValueError("Model not trained")

        X_proc = self.extract_features(X)
        N, C, T = X_proc.shape
        X_flat = X_proc.transpose(0, 2, 1).reshape(-1, C)
        X_scaled_flat = self._scaler.transform(X_flat)
        X_scaled = X_scaled_flat.reshape(N, T, C).transpose(0, 2, 1)

        device = torch.device(self.hyperparameters["device"] if torch.cuda.is_available() else "cpu")
        X_tensor = torch.FloatTensor(X_scaled).to(device)

        self._model.eval()
        with torch.no_grad():
            outputs = self._model(X_tensor)
            probs = torch.softmax(outputs, dim=1)

        return probs.cpu().numpy()

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self.metadata:
            with open(path / "metadata.json", 'w') as f:
                json.dump(self.metadata.to_dict(), f, indent=2)

        torch.save(self._model.state_dict(), path / "model.pt")

        with open(path / "scaler.pkl", 'wb') as f:
            pickle.dump(self._scaler, f)

        with open(path / "params.json", 'w') as f:
            json.dump({
                "input_channels": self._input_shape[0] if self._input_shape else 0,
                "input_length": self._input_shape[1] if self._input_shape else 0,
                "output_dim": self._output_dim,
                "hyperparameters": self.hyperparameters
            }, f)

    def load(self, path: Path) -> None:
        path = Path(path)

        with open(path / "metadata.json", 'r') as f:
            self.metadata = ModelMetadata.from_dict(json.load(f))

        with open(path / "params.json", 'r') as f:
            params = json.load(f)
            self._input_shape = (params["input_channels"], params["input_length"])
            self._output_dim = params["output_dim"]
            self.hyperparameters = params["hyperparameters"]

        with open(path / "scaler.pkl", 'rb') as f:
            self._scaler = pickle.load(f)

        self._model = self._build_model(self._input_shape[0], self._input_shape[1], self._output_dim)
        device = torch.device(self.hyperparameters["device"] if torch.cuda.is_available() else "cpu")
        self._model.load_state_dict(torch.load(path / "model.pt", map_location=device))
        self._model.to(device)

        self._is_trained = True


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, pool_proj):
        super().__init__()
        self.branch1 = nn.Sequential(nn.Conv1d(in_channels, out_1x1, kernel_size=1), nn.ReLU())

        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, red_3x3, kernel_size=1), nn.ReLU(),
            nn.Conv1d(red_3x3, out_3x3, kernel_size=3, padding=1), nn.ReLU()
        )

        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, red_5x5, kernel_size=1), nn.ReLU(),
            nn.Conv1d(red_5x5, out_5x5, kernel_size=5, padding=2), nn.ReLU()
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, pool_proj, kernel_size=1), nn.ReLU()
        )

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)


class InceptionAttentionClassifier(CNNClassifier):
    def __init__(self, name: str = "inception_att_classifier", **kwargs):
        super().__init__(name, **kwargs)
        self.hyperparameters.update({
            "reduction_ratio": kwargs.get("reduction_ratio", 8),
            "inception_channels": kwargs.get("inception_channels", 32)  # Base width
        })

    def _build_model(self, input_channels: int, input_length: int, output_dim: int):
        base = self.hyperparameters["inception_channels"]

        # Initial Feature Extraction
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, base, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(base),
            nn.ReLU()
        )

        # Inception Module
        # Total output channels = 16 (br1) + 32 (br2) + 8 (br3) + 8 (br4) = 64
        self.inception = InceptionBlock(base, 16, 16, 32, 4, 8, 8)

        # Attention
        self.attention = ChannelAttention(64, reduction=self.hyperparameters["reduction_ratio"])

        # Classification Head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(self.hyperparameters["dropout"]),
            nn.Linear(64, output_dim)
        )

        return nn.Sequential(self.stem, self.inception, self.attention, self.head)


class CatBoostClassifier(BaseClassifier):
    """
    CatBoost classifier for gesture recognition.
    """

    def __init__(self, name: str = "catboost_classifier", **kwargs):
        super().__init__(name)
        # Default hyperparameters optimized for multi-class classification
        self.hyperparameters = {
            "iterations": kwargs.get("iterations", 1000),
            "learning_rate": kwargs.get("learning_rate", 0.03),
            "depth": kwargs.get("depth", 6),
            "l2_leaf_reg": kwargs.get("l2_leaf_reg", 3),
            "loss_function": kwargs.get("loss_function", "MultiClass"),
            "verbose": kwargs.get("verbose", False),
            "task_type": kwargs.get("task_type", "CPU"),  # Use "GPU" if available
            "early_stopping_rounds": kwargs.get("early_stopping_rounds", 50)
        }
        self._feature_extractor = EMGFeatureExtractor()
        # Note: CatBoost is generally scale-invariant, but we maintain the scaler
        # to ensure strict API consistency with the SVM implementation.
        self._scaler = None

    def extract_features(self, X: np.ndarray) -> np.ndarray:
        """Extract features from raw EMG."""
        return self._feature_extractor.extract_all_features(X)

    def train(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            **kwargs
    ) -> Dict[str, Any]:
        """Train the CatBoost model."""
        import time

        start_time = time.time()

        # Extract features
        X_train_features = self.extract_features(X_train)
        feature_count = X_train_features.shape[1]

        # Normalize features
        self._scaler = StandardScaler()
        X_train_scaled = self._scaler.fit_transform(X_train_features)

        # Prepare validation set if available (Crucial for CatBoost early stopping)
        eval_set = None
        X_val_scaled = None
        if X_val is not None and y_val is not None:
            X_val_features = self.extract_features(X_val)
            X_val_scaled = self._scaler.transform(X_val_features)
            eval_set = (X_val_scaled, y_val)

        # Create model
        self._model = CatBoostWrapper(**self.hyperparameters)

        # Fit model
        self._model.fit(
            X_train_scaled,
            y_train,
            eval_set=eval_set,
            verbose=self.hyperparameters.get("verbose", False)
        )

        # Compute training accuracy
        # Note: CatBoost predict returns a column vector, so we flatten it
        train_pred = self._model.predict(X_train_scaled).flatten()
        train_acc = np.mean(train_pred == y_train)

        # Compute validation accuracy if provided
        val_acc = 0.0
        if eval_set is not None:
            # We already computed X_val_scaled in the eval_set block
            val_pred = self._model.predict(X_val_scaled).flatten()
            val_acc = np.mean(val_pred == y_val)

        training_time = time.time() - start_time
        self._is_trained = True

        # Create metadata
        self.metadata = ModelMetadata(
            name=self.name,
            model_type="CatBoost",
            created_at=datetime.now(),
            num_classes=len(np.unique(y_train)),
            num_channels=X_train.shape[2],
            window_size_ms=kwargs.get("window_size_ms", 200),
            sampling_rate=kwargs.get("sampling_rate", 2000),
            training_accuracy=train_acc,
            validation_accuracy=val_acc,
            hyperparameters=self.hyperparameters
        )

        return {
            "training_accuracy": train_acc,
            "validation_accuracy": val_acc,
            "feature_count": feature_count,
            "training_time": training_time
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self._is_trained:
            raise ValueError("Model not trained")

        X_features = self.extract_features(X)
        X_scaled = self._scaler.transform(X_features)
        # Flatten is required because CatBoost returns (N, 1) rather than (N,)
        return self._model.predict(X_scaled).flatten()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if not self._is_trained:
            raise ValueError("Model not trained")

        X_features = self.extract_features(X)
        X_scaled = self._scaler.transform(X_features)
        return self._model.predict_proba(X_scaled)




class ModelManager:
    """
    Manages training, loading, and using gesture classification models.
    """

    AVAILABLE_MODELS = {
        "svm": SVMClassifier,
        "random_forest": RandomForestClassifier,
        "lda": LDAClassifier,
        "catboost": CatBoostClassifier,
        "mlp": MLPClassifier,
        "cnn": CNNClassifier,
        "inceptionattn": InceptionAttentionClassifier,
    }

    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._current_model: Optional[BaseClassifier] = None

    def create_model(self, model_type: str, name: str = None, **kwargs) -> BaseClassifier:
        """
        Create a new model instance.

        Args:
            model_type: Type of model ("svm", "random_forest", "lda")
            name: Optional name for the model
            **kwargs: Model-specific hyperparameters

        Returns:
            Model instance
        """
        if model_type not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Available: {list(self.AVAILABLE_MODELS.keys())}"
            )

        if name is None:
            name = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        model = self.AVAILABLE_MODELS[model_type](name=name, **kwargs)
        self._current_model = model
        return model

    def train_model(
        self,
        model: BaseClassifier,
        dataset: Dict[str, Any],
        test_ratio: float = 0.2,
        save: bool = True
    ) -> Dict[str, Any]:
        """
        Train a model on a dataset.

        Args:
            model: Model instance to train
            dataset: Dataset dictionary from DataManager
            test_ratio: Ratio for validation split
            save: Whether to save the trained model

        Returns:
            Training results
        """
        from sklearn.model_selection import train_test_split

        X = dataset["X"]
        y = dataset["y"]

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_ratio, stratify=y, random_state=42
        )

        # Train model
        results = model.train(
            X_train, y_train,
            X_val, y_val,
            window_size_ms=dataset["metadata"]["window_size_ms"],
            sampling_rate=dataset["metadata"].get("sampling_rate", 2000)
        )

        # Add class names to metadata
        if model.metadata and "label_names" in dataset["metadata"]:
            model.metadata.class_names = dataset["metadata"]["label_names"]

        # Save model
        if save:
            model_path = self.models_dir / model.name
            model.save(model_path)

        self._current_model = model
        return results

    def load_model(self, name: str, model_type: str = None) -> BaseClassifier:
        """
        Load a trained model.

        Args:
            name: Name of the model
            model_type: Type of model (if known)

        Returns:
            Loaded model
        """
        model_path = self.models_dir / name

        # Try to determine model type from metadata
        if model_type is None:
            with open(model_path / "metadata.json", 'r') as f:
                metadata = json.load(f)
            model_type = metadata["model_type"].lower()
            if model_type == "randomforest":
                model_type = "random_forest"

        model = self.create_model(model_type, name=name)
        model.load(model_path)
        self._current_model = model
        return model

    def list_models(self) -> List[str]:
        """List all saved models."""
        return [d.name for d in self.models_dir.iterdir() if d.is_dir()]

    @property
    def current_model(self) -> Optional[BaseClassifier]:
        """Get the current model."""
        return self._current_model
