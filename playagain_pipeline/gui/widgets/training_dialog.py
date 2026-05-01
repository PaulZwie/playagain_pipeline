"""
Training Progress Dialog for detailed model training visualization.

Provides hyperparameter editing, real-time progress monitoring,
and training result visualization.
"""

from datetime import datetime as _dt
from pathlib import Path
import threading
from typing import Optional, Dict, Any, List

from sklearn.model_selection import train_test_split
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QLabel, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox,
    QPushButton, QProgressBar, QTabWidget, QWidget, QTextEdit,
    QTableWidget, QTableWidgetItem, QHeaderView, QRadioButton,
    QButtonGroup, QListWidget, QListWidgetItem, QMessageBox
)
from PySide6.QtCore import Signal, Slot, QThread, Qt
import pyqtgraph as pg

from playagain_pipeline.config.config import PipelineConfig
from playagain_pipeline.models.classifier import ModelManager
from playagain_pipeline.models.feature_pipeline import get_registered_features


class TrainingWorker(QThread):
    """Worker thread for model training."""

    progress = Signal(int, str)  # progress percentage, message
    iteration_update = Signal(int, float, float, float, float)  # iteration, train_loss, val_loss, train_acc, val_acc
    finished = Signal(dict)  # results
    error = Signal(str)

    def __init__(self, model, X_train, y_train, X_val, y_val, kwargs):
        super().__init__()
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.kwargs = kwargs
        self._should_stop = False

    def stop(self):
        """Request training to stop."""
        self._should_stop = True

    def run(self):
        """Run the training in background thread."""
        try:
            self.progress.emit(5, "Loading data...")

            # Determine total epochs for iterative models
            total_epochs = self.kwargs.get("epochs", 0)
            if total_epochs == 0:
                # Try to get from the model's hyperparameters
                if hasattr(self.model, 'hyperparameters'):
                    total_epochs = self.model.hyperparameters.get("epochs", 0)

            self.progress.emit(10, "Extracting features & preparing data...")

            # Define callback for iterative models (e.g. MLP, CNN)
            def progress_callback(epoch, train_loss, val_loss, train_acc=0.0, val_acc=0.0):
                self.iteration_update.emit(epoch, train_loss, val_loss, train_acc, val_acc)
                # Scale progress from 20% to 95% based on epoch/total_epochs
                if total_epochs > 0:
                    pct = 20 + int(75 * epoch / total_epochs)
                    pct = min(pct, 95)
                    self.progress.emit(pct, f"Training epoch {epoch}/{total_epochs}...")

            # Add callback to kwargs
            self.kwargs['callback'] = progress_callback

            if self.kwargs.get("auto_learning_rate"):
                self.progress.emit(12, "Running learning-rate finder...")

            # For non-iterative models, use a background timer to simulate progress
            is_iterative = hasattr(self.model, 'hyperparameters') and \
                           self.model.hyperparameters.get("epochs", 0) > 0

            if not is_iterative:
                self.progress.emit(15, "Extracting features...")
                # Start a simulated progress thread that advances 20% → 90%
                _sim_stop = threading.Event()

                def _simulate_progress():
                    pct = 20
                    while not _sim_stop.is_set() and pct < 90:
                        self.progress.emit(pct, "Training model...")
                        _sim_stop.wait(timeout=0.5)
                        pct = min(pct + 3, 90)

                sim_thread = threading.Thread(target=_simulate_progress, daemon=True)
                sim_thread.start()

            # Train model
            results = self.model.train(
                self.X_train, self.y_train,
                self.X_val, self.y_val,
                **self.kwargs
            )

            if not is_iterative:
                _sim_stop.set()
                sim_thread.join(timeout=1.0)

            self.progress.emit(100, "Training complete!")
            self.finished.emit(results)

        except Exception as e:
            if not is_iterative and '_sim_stop' in dir():
                _sim_stop.set()
            self.error.emit(str(e))


class HyperparameterWidget(QWidget):
    """Widget for editing model hyperparameters."""

    def __init__(self, model_type: str, config: PipelineConfig, parent=None):
        super().__init__(parent)
        self.model_type = model_type
        self.config = config
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        if self.model_type == "svm":
            self._setup_svm_params(layout)
        elif self.model_type == "catboost":
            self._setup_catboost_params(layout)
        elif self.model_type == "random_forest":
            self._setup_rf_params(layout)
        elif self.model_type == "lda":
            self._setup_lda_params(layout)
        elif self.model_type == "mlp":
            self._setup_mlp_params(layout)
        elif self.model_type == "cnn":
            self._setup_cnn_params(layout)
        elif self.model_type == "attention_net":
            self._setup_inception_params(layout)
        elif self.model_type == "mstnet":
            self._setup_mstnet_params(layout)
        else:
            layout.addWidget(QLabel("No hyperparameters for this model type"))

        layout.addStretch()

    def _setup_svm_params(self, layout):
        grid = QGridLayout()

        # Kernel
        grid.addWidget(QLabel("Kernel:"), 0, 0)
        self.kernel_combo = QComboBox()
        self.kernel_combo.addItems(["rbf", "linear", "poly", "sigmoid"])
        self.kernel_combo.setCurrentText(self.config.model.svm_kernel)
        grid.addWidget(self.kernel_combo, 0, 1)

        # C parameter
        grid.addWidget(QLabel("C (Regularization):"), 1, 0)
        self.c_spin = QDoubleSpinBox()
        self.c_spin.setRange(0.001, 1000)
        self.c_spin.setValue(self.config.model.svm_c)
        self.c_spin.setDecimals(3)
        grid.addWidget(self.c_spin, 1, 1)

        # Gamma
        grid.addWidget(QLabel("Gamma:"), 2, 0)
        self.gamma_combo = QComboBox()
        self.gamma_combo.addItems(["scale", "auto"])
        self.gamma_combo.setCurrentText(self.config.model.svm_gamma)
        grid.addWidget(self.gamma_combo, 2, 1)

        layout.addLayout(grid)

    def _setup_catboost_params(self, layout):
        grid = QGridLayout()

        # Iterations
        grid.addWidget(QLabel("Iterations:"), 0, 0)
        self.iterations_spin = QSpinBox()
        self.iterations_spin.setRange(10, 10000)
        self.iterations_spin.setValue(self.config.model.catboost_iterations)
        grid.addWidget(self.iterations_spin, 0, 1)

        # Learning rate
        grid.addWidget(QLabel("Learning Rate:"), 1, 0)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.001, 1.0)
        self.lr_spin.setValue(self.config.model.catboost_learning_rate)
        self.lr_spin.setDecimals(4)
        grid.addWidget(self.lr_spin, 1, 1)

        # Depth
        grid.addWidget(QLabel("Depth:"), 2, 0)
        self.depth_spin = QSpinBox()
        self.depth_spin.setRange(1, 16)
        self.depth_spin.setValue(self.config.model.catboost_depth)
        grid.addWidget(self.depth_spin, 2, 1)

        # L2 regularization
        grid.addWidget(QLabel("L2 Regularization:"), 3, 0)
        self.l2_spin = QDoubleSpinBox()
        self.l2_spin.setRange(0, 100)
        self.l2_spin.setValue(self.config.model.catboost_l2_leaf_reg)
        grid.addWidget(self.l2_spin, 3, 1)

        # Early stopping
        self.early_stop_check = QCheckBox("Early Stopping")
        self.early_stop_check.setChecked(self.config.model.catboost_early_stopping)
        grid.addWidget(self.early_stop_check, 4, 0, 1, 2)

        layout.addLayout(grid)

    def _setup_rf_params(self, layout):
        grid = QGridLayout()

        # Number of estimators
        grid.addWidget(QLabel("N Estimators:"), 0, 0)
        self.n_estimators_spin = QSpinBox()
        self.n_estimators_spin.setRange(10, 1000)
        self.n_estimators_spin.setValue(self.config.model.rf_n_estimators)
        grid.addWidget(self.n_estimators_spin, 0, 1)

        # Max depth
        grid.addWidget(QLabel("Max Depth:"), 1, 0)
        self.max_depth_spin = QSpinBox()
        self.max_depth_spin.setRange(1, 100)
        self.max_depth_spin.setValue(self.config.model.rf_max_depth or 10)
        self.max_depth_spin.setSpecialValueText("None")
        grid.addWidget(self.max_depth_spin, 1, 1)

        # Min samples split
        grid.addWidget(QLabel("Min Samples Split:"), 2, 0)
        self.min_samples_spin = QSpinBox()
        self.min_samples_spin.setRange(2, 100)
        self.min_samples_spin.setValue(self.config.model.rf_min_samples_split)
        grid.addWidget(self.min_samples_spin, 2, 1)

        layout.addLayout(grid)

    def _setup_lda_params(self, layout):
        grid = QGridLayout()

        # Solver
        grid.addWidget(QLabel("Solver:"), 0, 0)
        self.solver_combo = QComboBox()
        self.solver_combo.addItems(["svd", "lsqr", "eigen"])
        self.solver_combo.setCurrentText(self.config.model.lda_solver)
        grid.addWidget(self.solver_combo, 0, 1)

        # Shrinkage
        grid.addWidget(QLabel("Shrinkage:"), 1, 0)
        self.shrinkage_combo = QComboBox()
        self.shrinkage_combo.addItems(["None", "auto"])
        self.shrinkage_combo.setCurrentText("None" if self.config.model.lda_shrinkage is None else self.config.model.lda_shrinkage)
        grid.addWidget(self.shrinkage_combo, 1, 1)

        layout.addLayout(grid)

    def _setup_mlp_params(self, layout):
        grid = QGridLayout()

        # Hidden Layers
        grid.addWidget(QLabel("Hidden Layers:"), 0, 0)
        self.hidden_layers_edit = QComboBox() # Using combo for simplicity, or line edit
        self.hidden_layers_edit.setEditable(True)
        self.hidden_layers_edit.addItems([",".join(map(str, self.config.model.mlp_hidden_layers)), "128, 64", "256, 128", "64, 32", "128"])
        self.hidden_layers_edit.setCurrentText(",".join(map(str, self.config.model.mlp_hidden_layers)))
        grid.addWidget(self.hidden_layers_edit, 0, 1)

        # Epochs
        grid.addWidget(QLabel("Epochs:"), 1, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 10000)
        self.epochs_spin.setValue(self.config.model.mlp_epochs)
        grid.addWidget(self.epochs_spin, 1, 1)

        # Batch Size
        grid.addWidget(QLabel("Batch Size:"), 2, 0)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 1024)
        self.batch_spin.setValue(self.config.model.mlp_batch_size)
        grid.addWidget(self.batch_spin, 2, 1)

        # Learning Rate
        grid.addWidget(QLabel("Learning Rate:"), 3, 0)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(5)
        self.lr_spin.setRange(0.00001, 1.0)
        self.lr_spin.setValue(self.config.model.mlp_learning_rate)
        grid.addWidget(self.lr_spin, 3, 1)

        # Optimizer
        grid.addWidget(QLabel("Optimizer:"), 4, 0)
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["Adam", "AdamW", "RMSprop", "SGD"])
        self.optimizer_combo.setCurrentText(self.config.model.mlp_optimizer.capitalize())
        grid.addWidget(self.optimizer_combo, 4, 1)

        # Early Stopping
        self.early_stop_check = QCheckBox("Early Stopping")
        self.early_stop_check.setChecked(self.config.model.mlp_early_stopping)
        grid.addWidget(self.early_stop_check, 5, 0)

        # Patience
        grid.addWidget(QLabel("Patience:"), 5, 1) # Reuse row if possible or new row
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(1, 100)
        self.patience_spin.setValue(self.config.model.mlp_patience)
        grid.addWidget(self.patience_spin, 5, 2)

        # Weight Decay
        grid.addWidget(QLabel("Weight Decay:"), 6, 0)
        self.weight_decay_spin = QDoubleSpinBox()
        self.weight_decay_spin.setDecimals(5)
        self.weight_decay_spin.setRange(0.0, 1.0)
        self.weight_decay_spin.setValue(self.config.model.mlp_weight_decay)
        self.weight_decay_spin.setSingleStep(0.001)
        grid.addWidget(self.weight_decay_spin, 6, 1)

        # LR Scheduler
        grid.addWidget(QLabel("LR Scheduler:"), 7, 0)
        self.scheduler_combo = QComboBox()
        self.scheduler_combo.addItems(["None", "Cosine Annealing", "Reduce on Plateau", "Step"])
        self.scheduler_combo.setCurrentText(
            {"none": "None", "cosine": "Cosine Annealing", "plateau": "Reduce on Plateau", "step": "Step"}
            .get(self.config.model.mlp_scheduler, "None")
        )
        grid.addWidget(self.scheduler_combo, 7, 1)

        self._setup_auto_lr_controls(grid, 8)

        layout.addLayout(grid)

    def _setup_cnn_params(self, layout):
        grid = QGridLayout()

        # Convolutional Layers (Filters)
        grid.addWidget(QLabel("Filters per Layer:"), 0, 0)
        self.cnn_filters_edit = QComboBox()
        self.cnn_filters_edit.setEditable(True)
        self.cnn_filters_edit.addItems(["32, 64, 128", "16, 32, 64", "64, 128", "32, 32"])
        self.cnn_filters_edit.setCurrentText(self.config.model.cnn_filters)
        grid.addWidget(self.cnn_filters_edit, 0, 1)

        # Kernel Sizes
        grid.addWidget(QLabel("Kernel Sizes:"), 1, 0)
        self.cnn_kernels_edit = QComboBox()
        self.cnn_kernels_edit.setEditable(True)
        self.cnn_kernels_edit.addItems(["5, 3, 3", "3, 3, 3", "7, 5, 3", "5, 5"])
        self.cnn_kernels_edit.setCurrentText(self.config.model.cnn_kernels)
        grid.addWidget(self.cnn_kernels_edit, 1, 1)

        # FC Layers
        grid.addWidget(QLabel("FC Layers:"), 2, 0)
        self.cnn_fc_edit = QComboBox()
        self.cnn_fc_edit.setEditable(True)
        self.cnn_fc_edit.addItems(["128", "256, 128", "64"])
        self.cnn_fc_edit.setCurrentText(self.config.model.cnn_fc_layers)
        grid.addWidget(self.cnn_fc_edit, 2, 1)

        # Epochs
        grid.addWidget(QLabel("Epochs:"), 3, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 10000)
        self.epochs_spin.setValue(self.config.model.cnn_epochs)
        grid.addWidget(self.epochs_spin, 3, 1)

        # Batch Size
        grid.addWidget(QLabel("Batch Size:"), 4, 0)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 1024)
        self.batch_spin.setValue(self.config.model.cnn_batch_size)
        grid.addWidget(self.batch_spin, 4, 1)

        # Learning Rate
        grid.addWidget(QLabel("Learning Rate:"), 5, 0)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(5)
        self.lr_spin.setRange(0.00001, 1.0)
        self.lr_spin.setValue(self.config.model.cnn_learning_rate)
        grid.addWidget(self.lr_spin, 5, 1)

        # Optimizer
        grid.addWidget(QLabel("Optimizer:"), 6, 0)
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["Adam", "AdamW", "RMSprop", "SGD"])
        self.optimizer_combo.setCurrentText(self.config.model.cnn_optimizer.capitalize())
        grid.addWidget(self.optimizer_combo, 6, 1)

        # Early Stopping Row
        row_es = 7
        self.early_stop_check = QCheckBox("Early Stopping")
        self.early_stop_check.setChecked(self.config.model.cnn_early_stopping)
        grid.addWidget(self.early_stop_check, row_es, 0)

        # Patience (on same row)
        patience_label = QLabel("Patience:")
        grid.addWidget(patience_label, row_es, 1)
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(1, 100)
        self.patience_spin.setValue(self.config.model.cnn_patience)
        grid.addWidget(self.patience_spin, row_es, 2)

        # Weight Decay
        grid.addWidget(QLabel("Weight Decay:"), 8, 0)
        self.weight_decay_spin = QDoubleSpinBox()
        self.weight_decay_spin.setDecimals(5)
        self.weight_decay_spin.setRange(0.0, 1.0)
        self.weight_decay_spin.setValue(self.config.model.cnn_weight_decay)
        self.weight_decay_spin.setSingleStep(0.001)
        grid.addWidget(self.weight_decay_spin, 8, 1)

        # LR Scheduler
        grid.addWidget(QLabel("LR Scheduler:"), 9, 0)
        self.scheduler_combo = QComboBox()
        self.scheduler_combo.addItems(["None", "Cosine Annealing", "Reduce on Plateau", "Step"])
        self.scheduler_combo.setCurrentText(
            {"none": "None", "cosine": "Cosine Annealing", "plateau": "Reduce on Plateau", "step": "Step"}
            .get(self.config.model.cnn_scheduler, "None")
        )
        grid.addWidget(self.scheduler_combo, 9, 1)

        self._setup_auto_lr_controls(grid, 10)

        layout.addLayout(grid)

    def _setup_inception_params(self, layout):
        grid = QGridLayout()

        # Inception Base Channels
        grid.addWidget(QLabel("Base Channels:"), 0, 0)
        self.inc_base_spin = QSpinBox()
        self.inc_base_spin.setRange(8, 256)
        try:
            default_base = int(self.config.model.inception_filters.split(",")[0].strip())
        except (ValueError, IndexError):
            default_base = 32
        self.inc_base_spin.setValue(default_base)
        grid.addWidget(self.inc_base_spin, 0, 1)

        # Inception Branch Kernel Sizes
        grid.addWidget(QLabel("Branch Kernels:"), 1, 0)
        self.inc_kernels_edit = QComboBox()
        self.inc_kernels_edit.setEditable(True)
        self.inc_kernels_edit.addItems([
            self.config.model.inception_kernels,
            "3, 5", "3, 7", "5, 9", "3, 3",
            "3, 5, 7", "3, 5, 9", "3, 7, 15", "5, 9, 15",
            "3", "5", "7", "3, 15, 39"
        ])
        self.inc_kernels_edit.setCurrentText(self.config.model.inception_kernels)
        grid.addWidget(self.inc_kernels_edit, 1, 1)

        # Attention Reduction Ratio
        grid.addWidget(QLabel("Attention Reduction:"), 2, 0)
        self.att_red_spin = QSpinBox()
        self.att_red_spin.setRange(1, 32)
        self.att_red_spin.setValue(8)
        grid.addWidget(self.att_red_spin, 2, 1)

        # FC Layers
        grid.addWidget(QLabel("FC Layers:"), 3, 0)
        self.inc_fc_edit = QComboBox()
        self.inc_fc_edit.setEditable(True)
        self.inc_fc_edit.addItems([self.config.model.inception_fc_layers, "128", "256, 128", "64"])
        self.inc_fc_edit.setCurrentText(self.config.model.inception_fc_layers)
        grid.addWidget(self.inc_fc_edit, 3, 1)

        # Epochs
        grid.addWidget(QLabel("Epochs:"), 4, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 10000)
        self.epochs_spin.setValue(self.config.model.inception_epochs)
        grid.addWidget(self.epochs_spin, 4, 1)

        # Batch Size
        grid.addWidget(QLabel("Batch Size:"), 5, 0)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 1024)
        self.batch_spin.setValue(self.config.model.inception_batch_size)
        grid.addWidget(self.batch_spin, 5, 1)

        # Learning Rate
        grid.addWidget(QLabel("Learning Rate:"), 6, 0)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(5)
        self.lr_spin.setRange(0.00001, 1.0)
        self.lr_spin.setValue(self.config.model.inception_learning_rate)
        grid.addWidget(self.lr_spin, 6, 1)

        # Optimizer
        grid.addWidget(QLabel("Optimizer:"), 7, 0)
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["Adam", "AdamW", "RMSprop", "SGD"])
        self.optimizer_combo.setCurrentText(self.config.model.inception_optimizer.capitalize())
        grid.addWidget(self.optimizer_combo, 7, 1)

        # Early Stopping Row
        row_es = 8
        self.early_stop_check = QCheckBox("Early Stopping")
        self.early_stop_check.setChecked(self.config.model.inception_early_stopping)
        grid.addWidget(self.early_stop_check, row_es, 0)

        # Patience
        patience_label = QLabel("Patience:")
        grid.addWidget(patience_label, row_es, 1)
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(1, 100)
        self.patience_spin.setValue(self.config.model.inception_patience)
        grid.addWidget(self.patience_spin, row_es, 2)

        # Weight Decay
        grid.addWidget(QLabel("Weight Decay:"), 9, 0)
        self.weight_decay_spin = QDoubleSpinBox()
        self.weight_decay_spin.setDecimals(5)
        self.weight_decay_spin.setRange(0.0, 1.0)
        self.weight_decay_spin.setValue(self.config.model.inception_weight_decay)
        self.weight_decay_spin.setSingleStep(0.001)
        grid.addWidget(self.weight_decay_spin, 9, 1)

        # LR Scheduler
        grid.addWidget(QLabel("LR Scheduler:"), 10, 0)
        self.scheduler_combo = QComboBox()
        self.scheduler_combo.addItems(["None", "Cosine Annealing", "Reduce on Plateau", "Step"])
        self.scheduler_combo.setCurrentText(
            {"none": "None", "cosine": "Cosine Annealing", "plateau": "Reduce on Plateau", "step": "Step"}
            .get(self.config.model.inception_scheduler, "None")
        )
        grid.addWidget(self.scheduler_combo, 10, 1)

        self._setup_auto_lr_controls(grid, 11)

        layout.addLayout(grid)

    def _setup_mstnet_params(self, layout):
        grid = QGridLayout()

        # Base Filters
        grid.addWidget(QLabel("Base Filters:"), 0, 0)
        self.mst_base_spin = QSpinBox()
        self.mst_base_spin.setRange(8, 256)
        self.mst_base_spin.setValue(self.config.model.mstnet_base_filters)
        grid.addWidget(self.mst_base_spin, 0, 1)

        # Multi-Scale Kernels
        grid.addWidget(QLabel("MS Kernels:"), 1, 0)
        self.mst_kernels_edit = QComboBox()
        self.mst_kernels_edit.setEditable(True)
        self.mst_kernels_edit.addItems([
            self.config.model.mstnet_kernels,
            "3, 7, 15", "3, 5, 11", "5, 11, 21", "3, 7"
        ])
        self.mst_kernels_edit.setCurrentText(self.config.model.mstnet_kernels)
        grid.addWidget(self.mst_kernels_edit, 1, 1)

        # Number of Blocks
        grid.addWidget(QLabel("Num Blocks:"), 2, 0)
        self.mst_blocks_spin = QSpinBox()
        self.mst_blocks_spin.setRange(1, 6)
        self.mst_blocks_spin.setValue(self.config.model.mstnet_num_blocks)
        grid.addWidget(self.mst_blocks_spin, 2, 1)

        # Epochs
        grid.addWidget(QLabel("Epochs:"), 3, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 10000)
        self.epochs_spin.setValue(self.config.model.mstnet_epochs)
        grid.addWidget(self.epochs_spin, 3, 1)

        # Batch Size
        grid.addWidget(QLabel("Batch Size:"), 4, 0)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 1024)
        self.batch_spin.setValue(self.config.model.mstnet_batch_size)
        grid.addWidget(self.batch_spin, 4, 1)

        # Learning Rate
        grid.addWidget(QLabel("Learning Rate:"), 5, 0)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setDecimals(5)
        self.lr_spin.setRange(0.00001, 1.0)
        self.lr_spin.setValue(self.config.model.mstnet_learning_rate)
        grid.addWidget(self.lr_spin, 5, 1)

        # Optimizer
        grid.addWidget(QLabel("Optimizer:"), 6, 0)
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["Adam", "AdamW", "RMSprop", "SGD"])
        self.optimizer_combo.setCurrentText(self.config.model.mstnet_optimizer.capitalize())
        grid.addWidget(self.optimizer_combo, 6, 1)

        # Early Stopping
        self.early_stop_check = QCheckBox("Early Stopping")
        self.early_stop_check.setChecked(self.config.model.mstnet_early_stopping)
        grid.addWidget(self.early_stop_check, 7, 0)

        # Patience
        grid.addWidget(QLabel("Patience:"), 7, 1) # Reuse row if possible or new row
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(1, 100)
        self.patience_spin.setValue(self.config.model.mstnet_patience)
        grid.addWidget(self.patience_spin, 7, 2)

        # Weight Decay
        grid.addWidget(QLabel("Weight Decay:"), 8, 0)
        self.weight_decay_spin = QDoubleSpinBox()
        self.weight_decay_spin.setDecimals(5)
        self.weight_decay_spin.setRange(0.0, 1.0)
        self.weight_decay_spin.setValue(self.config.model.mstnet_weight_decay)
        self.weight_decay_spin.setSingleStep(0.001)
        grid.addWidget(self.weight_decay_spin, 8, 1)

        # LR Scheduler
        grid.addWidget(QLabel("LR Scheduler:"), 9, 0)
        self.scheduler_combo = QComboBox()
        self.scheduler_combo.addItems(["None", "Cosine Annealing", "Reduce on Plateau", "Step"])
        self.scheduler_combo.setCurrentText(
            {"none": "None", "cosine": "Cosine Annealing", "plateau": "Reduce on Plateau", "step": "Step"}
            .get(self.config.model.mstnet_scheduler, "None")
        )
        grid.addWidget(self.scheduler_combo, 9, 1)

        self._setup_auto_lr_controls(grid, 10)

        layout.addLayout(grid)

    def _setup_auto_lr_controls(self, grid: QGridLayout, row: int):
        """Add the Auto Learning Rate option shared by deep-learning models."""
        self._manual_epochs_value = None
        self._manual_early_stopping_value = None

        self.auto_lr_check = QCheckBox("Auto Learning Rate (LR Finder)")
        self.auto_lr_check.setToolTip(
            "Run an LR range test before training, then train for 100 epochs with early stopping disabled."
        )
        self.auto_lr_check.toggled.connect(self._on_auto_lr_toggled)
        grid.addWidget(self.auto_lr_check, row, 0, 1, 2)

        self.auto_lr_info_label = QLabel("Uses a learning-rate search before training.")
        self.auto_lr_info_label.setStyleSheet("color: #666; font-size: 10px;")
        grid.addWidget(self.auto_lr_info_label, row + 1, 0, 1, 3)

    def _on_auto_lr_toggled(self, enabled: bool):
        """Lock or unlock training controls when Auto LR is enabled."""
        if enabled:
            if hasattr(self, "epochs_spin"):
                self._manual_epochs_value = self.epochs_spin.value()
                self.epochs_spin.setValue(100)
                self.epochs_spin.setEnabled(False)
            if hasattr(self, "early_stop_check"):
                self._manual_early_stopping_value = self.early_stop_check.isChecked()
                self.early_stop_check.setChecked(False)
                self.early_stop_check.setEnabled(False)
            if hasattr(self, "patience_spin"):
                self.patience_spin.setEnabled(False)
            if hasattr(self, "lr_spin"):
                self.lr_spin.setEnabled(False)
        else:
            if hasattr(self, "epochs_spin"):
                if self._manual_epochs_value is not None:
                    self.epochs_spin.setValue(self._manual_epochs_value)
                self.epochs_spin.setEnabled(True)
            if hasattr(self, "early_stop_check"):
                if self._manual_early_stopping_value is not None:
                    self.early_stop_check.setChecked(self._manual_early_stopping_value)
                self.early_stop_check.setEnabled(True)
            if hasattr(self, "patience_spin"):
                self.patience_spin.setEnabled(True)
            if hasattr(self, "lr_spin"):
                self.lr_spin.setEnabled(True)

    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get the configured hyperparameters."""
        params = {}

        if self.model_type == "svm":
            params = {
                "kernel": self.kernel_combo.currentText(),
                "C": self.c_spin.value(),
                "gamma": self.gamma_combo.currentText()
            }
        elif self.model_type == "catboost":
            params = {
                "iterations": self.iterations_spin.value(),
                "learning_rate": self.lr_spin.value(),
                "depth": self.depth_spin.value(),
                "l2_leaf_reg": self.l2_spin.value(),
                "early_stopping_rounds": 50 if self.early_stop_check.isChecked() else None
            }
        elif self.model_type == "random_forest":
            params = {
                "n_estimators": self.n_estimators_spin.value(),
                "max_depth": self.max_depth_spin.value() if self.max_depth_spin.value() > 1 else None,
                "min_samples_split": self.min_samples_spin.value()
            }
        elif self.model_type == "lda":
            shrinkage = self.shrinkage_combo.currentText()
            params = {
                "solver": self.solver_combo.currentText(),
                "shrinkage": None if shrinkage == "None" else shrinkage
            }
        elif self.model_type == "mlp":
            # Parse hidden layers
            try:
                layers_str = self.hidden_layers_edit.currentText()
                layers = [int(x.strip()) for x in layers_str.split(",")]
            except ValueError:
                layers = [128, 64] # Fallback

            params = {
                "hidden_layers": tuple(layers),
                "epochs": 100 if getattr(self, "auto_lr_check", None) and self.auto_lr_check.isChecked() else self.epochs_spin.value(),
                "batch_size": self.batch_spin.value(),
                "learning_rate": self.lr_spin.value(),
                "optimizer": self.optimizer_combo.currentText().lower(),
                "early_stopping": False if getattr(self, "auto_lr_check", None) and self.auto_lr_check.isChecked() else self.early_stop_check.isChecked(),
                "patience": self.patience_spin.value(),
                "dropout": 0.2,
                "weight_decay": self.weight_decay_spin.value(),
                "scheduler": {"None": "none", "Cosine Annealing": "cosine",
                              "Reduce on Plateau": "plateau", "Step": "step"}
                             .get(self.scheduler_combo.currentText(), "none"),
                "auto_learning_rate": getattr(self, "auto_lr_check", None) and self.auto_lr_check.isChecked(),
            }
        elif self.model_type == "cnn":
            # Parse filters
            try:
                filters_str = self.cnn_filters_edit.currentText()
                filters = [int(x.strip()) for x in filters_str.split(",")]
            except ValueError:
                filters = [32, 64, 128]

            # Parse kernel sizes
            try:
                kernels_str = self.cnn_kernels_edit.currentText()
                kernels = [int(x.strip()) for x in kernels_str.split(",")]
            except ValueError:
                kernels = [5, 3, 3]

            # Parse FC layers
            try:
                fc_str = self.cnn_fc_edit.currentText()
                fc_layers = [int(x.strip()) for x in fc_str.split(",")]
            except ValueError:
                fc_layers = [128]

            params = {
                "filters": filters,
                "kernel_sizes": kernels,
                "fc_layers": tuple(fc_layers),
                "epochs": 100 if getattr(self, "auto_lr_check", None) and self.auto_lr_check.isChecked() else self.epochs_spin.value(),
                "batch_size": self.batch_spin.value(),
                "learning_rate": self.lr_spin.value(),
                "optimizer": self.optimizer_combo.currentText().lower(),
                "early_stopping": False if getattr(self, "auto_lr_check", None) and self.auto_lr_check.isChecked() else self.early_stop_check.isChecked(),
                "patience": self.patience_spin.value(),
                "weight_decay": self.weight_decay_spin.value(),
                "scheduler": {"None": "none", "Cosine Annealing": "cosine",
                              "Reduce on Plateau": "plateau", "Step": "step"}
                             .get(self.scheduler_combo.currentText(), "none"),
                "auto_learning_rate": getattr(self, "auto_lr_check", None) and self.auto_lr_check.isChecked(),
            }

        elif self.model_type == "attention_net":
            # Parse branch kernel sizes (variable number supported)
            try:
                kernels_str = self.inc_kernels_edit.currentText()
                kernels = [int(x.strip()) for x in kernels_str.split(",") if x.strip()]
                if not kernels:
                    kernels = [3, 5]
            except ValueError:
                kernels = [3, 5]

            params = {
                "inception_channels": self.inc_base_spin.value(),
                "branch_kernels": kernels,
                "reduction_ratio": self.att_red_spin.value(),
                "epochs": 100 if getattr(self, "auto_lr_check", None) and self.auto_lr_check.isChecked() else self.epochs_spin.value(),
                "batch_size": self.batch_spin.value(),
                "learning_rate": self.lr_spin.value(),
                "optimizer": self.optimizer_combo.currentText().lower(),
                "early_stopping": False if getattr(self, "auto_lr_check", None) and self.auto_lr_check.isChecked() else self.early_stop_check.isChecked(),
                "patience": self.patience_spin.value(),
                "weight_decay": self.weight_decay_spin.value(),
                "scheduler": {"None": "none", "Cosine Annealing": "cosine",
                              "Reduce on Plateau": "plateau", "Step": "step"}
                             .get(self.scheduler_combo.currentText(), "none"),
                "auto_learning_rate": getattr(self, "auto_lr_check", None) and self.auto_lr_check.isChecked(),
            }

        elif self.model_type == "mstnet":
            # Parse multi-scale kernels
            try:
                kernels_str = self.mst_kernels_edit.currentText()
                ms_kernels = [int(x.strip()) for x in kernels_str.split(",") if x.strip()]
                if not ms_kernels:
                    ms_kernels = [3, 7, 15]
            except ValueError:
                ms_kernels = [3, 7, 15]

            params = {
                "base_filters": self.mst_base_spin.value(),
                "ms_kernels": ms_kernels,
                "num_blocks": self.mst_blocks_spin.value(),
                "epochs": 100 if getattr(self, "auto_lr_check", None) and self.auto_lr_check.isChecked() else self.epochs_spin.value(),
                "batch_size": self.batch_spin.value(),
                "learning_rate": self.lr_spin.value(),
                "optimizer": self.optimizer_combo.currentText().lower(),
                "early_stopping": False if getattr(self, "auto_lr_check", None) and self.auto_lr_check.isChecked() else self.early_stop_check.isChecked(),
                "patience": self.patience_spin.value(),
                "dropout": self.config.model.mstnet_dropout,
                "weight_decay": self.weight_decay_spin.value(),
                "scheduler": {"None": "none", "Cosine Annealing": "cosine",
                              "Reduce on Plateau": "plateau", "Step": "step"}
                             .get(self.scheduler_combo.currentText(), "none"),
                "auto_learning_rate": getattr(self, "auto_lr_check", None) and self.auto_lr_check.isChecked(),
            }

        return params


class TrainingProgressDialog(QDialog):
    """
    Dialog for monitoring and controlling model training.

    Features:
    - Dataset and model type selection
    - Hyperparameter editing per model type
    - Real-time loss/accuracy visualization
    - Confusion matrix display
    - Training log
    """

    training_complete = Signal(dict)  # Emitted when training finishes

    def __init__(self, model_type: Optional[str] = None, dataset: Optional[Dict[str, Any]] = None,
                 config: Optional[PipelineConfig] = None, parent=None,
                 available_datasets: Optional[List[str]] = None, available_models: Optional[List[str]] = None):
        super().__init__(parent)
        self.model_type = model_type
        self.dataset = dataset
        self.config = config or PipelineConfig()
        self.available_datasets = available_datasets or []
        self.available_models = available_models or ["SVM", "CatBoost", "Random Forest", "LDA", "MLP", "CNN"]
        self._worker: Optional[TrainingWorker] = None

        # Flag used to short-circuit handlers when the dialog is being
        # torn down. Qt may deliver queued signals after the C++ objects
        # have started destruction which raises RuntimeError; using this
        # flag makes handlers no-op in that window.
        self._tearing_down = False
        # Also listen for QObject.destroyed in case teardown happens
        # from outside closeEvent; the signal is queued as a safe hook.
        try:
            self.destroyed.connect(self._on_destroyed)
        except Exception:
            # Be defensive: some PySide versions may behave oddly during
            # shutdown; ignore if connecting fails.
            pass

        # Training history for plotting
        self._train_losses: List[float] = []
        self._val_losses: List[float] = []
        self._train_accs: List[float] = []
        self._val_accs: List[float] = []
        self._iterations: List[int] = []

        self._setup_ui()
        if model_type:
            self.setWindowTitle(f"Train {model_type.upper()} Model")
        else:
            self.setWindowTitle("Advanced Model Training")
        self.setMinimumSize(1000, 700)

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Selection group (only show if dataset/model not pre-selected)
        if not self.dataset or not self.model_type:
            selection_group = QGroupBox("Training Configuration")
            selection_layout = QGridLayout(selection_group)

            # Model type selection
            if not self.model_type:
                selection_layout.addWidget(QLabel("Model Type:"), 0, 0)
                self.model_type_combo = QComboBox()
                # Block signals while populating so currentTextChanged
                # doesn't fire before self.tabs exists (it's created
                # further down in _setup_ui).
                self.model_type_combo.blockSignals(True)
                self.model_type_combo.addItems(self.available_models)
                self.model_type_combo.blockSignals(False)
                self.model_type_combo.currentTextChanged.connect(self._on_model_type_changed)
                selection_layout.addWidget(self.model_type_combo, 0, 1)
                # Convert first available model name to internal type name
                first_model = self.available_models[0]
                if first_model == "AttentionNet":
                    self.model_type = "attention_net"
                else:
                    self.model_type = first_model.lower().replace(" ", "_")
            else:
                self.model_type_combo = None

            # Dataset selection
            if not self.dataset:
                selection_layout.addWidget(QLabel("Dataset:"), 1, 0)
                self.dataset_combo = QComboBox()
                self.dataset_combo.addItems(self.available_datasets)
                self.dataset_combo.currentTextChanged.connect(self._on_dataset_selected)
                selection_layout.addWidget(self.dataset_combo, 1, 1)
            else:
                self.dataset_combo = None

            self.seed_spin = QSpinBox()
            self.seed_spin.setRange(0, 2 ** 31 - 1)
            self.seed_spin.setValue(int(getattr(self.config, "seed", 42) or 42))
            self.seed_spin.setToolTip(
                "Random seed for the train/val split. Keep constant across runs "
                "for reproducibility; change it to see how sensitive the numbers "
                "are to the split."
            )
            selection_layout.addWidget(QLabel("Random seed:"), 2, 0)
            selection_layout.addWidget(self.seed_spin, 2, 1)

            layout.addWidget(selection_group)

        # Dataset info group
        info_group = QGroupBox("Dataset Information")
        info_layout = QGridLayout(info_group)
        self.info_group = info_group
        self.info_layout = info_layout

        if self.dataset:
            metadata = self.dataset.get("metadata", {})
            info_layout.addWidget(QLabel(f"Name: {metadata.get('name', 'Unknown')}"), 0, 0)
            info_layout.addWidget(QLabel(f"Samples: {metadata.get('num_samples', 0)}"), 0, 1)
            info_layout.addWidget(QLabel(f"Classes: {metadata.get('num_classes', 0)}"), 1, 0)
            info_layout.addWidget(QLabel(f"Channels: {metadata.get('num_channels', 0)}"), 1, 1)
        else:
            info_layout.addWidget(QLabel("Select a dataset above to view information"), 0, 0)

        layout.addWidget(info_group)

        # Tabs for different sections
        self.tabs = QTabWidget()

        # Hyperparameters tab
        self.hyperparam_widget = HyperparameterWidget(self.model_type, self.config)
        self.tabs.addTab(self.hyperparam_widget, "Hyperparameters")

        # Feature Selection Tab
        self.features_widget = QWidget()
        self._setup_features_tab()
        self.tabs.addTab(self.features_widget, "Features")

        # Progress tab
        progress_widget = QWidget()
        progress_layout = QVBoxLayout(progress_widget)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        progress_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready to train")
        progress_layout.addWidget(self.status_label)

        # Loss and Accuracy plot
        self.loss_plot = pg.PlotWidget()
        self.loss_plot.setBackground('w')
        self.loss_plot.setLabel('left', 'Loss / Accuracy')
        self.loss_plot.setLabel('bottom', 'Iteration')
        self.loss_plot.addLegend()

        self.train_loss_curve = self.loss_plot.plot(pen=pg.mkPen('b', width=2), name='Train Loss')
        self.val_loss_curve = self.loss_plot.plot(pen=pg.mkPen('r', width=2), name='Validation Loss')
        self.train_acc_curve = self.loss_plot.plot(pen=pg.mkPen('g', width=2), name='Train Accuracy')
        self.val_acc_curve = self.loss_plot.plot(pen=pg.mkPen('orange', width=2), name='Validation Accuracy')

        progress_layout.addWidget(self.loss_plot)

        self.tabs.addTab(progress_widget, "Progress")

        # Results tab
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)

        # Summary table for epoch results
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels(["Epoch", "Train Loss", "Val Loss", "Train Acc", "Val Acc"])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        results_layout.addWidget(QLabel("Training Results:"), 0)
        results_layout.addWidget(self.results_table)

        self.tabs.addTab(results_widget, "Results")

        # Log tab
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.tabs.addTab(self.log_text, "Log")

        layout.addWidget(self.tabs)

        # Buttons
        button_layout = QHBoxLayout()

        self.train_btn = QPushButton("Start Training")
        self.train_btn.clicked.connect(self._on_start_training)
        button_layout.addWidget(self.train_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop_training)
        button_layout.addWidget(self.stop_btn)

        button_layout.addStretch()

        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.close_btn)

        self.save_btn = QPushButton("Save Model")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._on_save_model)
        button_layout.addWidget(self.save_btn)

        layout.addLayout(button_layout)

    def _log(self, message: str):
        """Add message to log."""
        self.log_text.append(message)

    def _on_model_type_changed(self, model_name: str):
        """Handle model type change.

        Defensively guard against late signals during dialog teardown.
        Qt's destruction order can fire ``currentTextChanged`` after
        ``self.tabs`` has already been deleted, raising
        ``RuntimeError: wrapped C/C++ object already deleted``. That's
        the "Internal C++ object (PySide6.QtWidgets.QTabWidget)
        already deleted" error users were seeing 2-6 times per training
        run. Set ``self.tabs = None`` on the first such failure so
        subsequent late signals bail out cheaply.
        """
        # Bail out quickly if we're being torn down — avoids acting on
        # queued signals that arrive after C++ children were deleted.
        if getattr(self, "_tearing_down", False):
            return

        # Convert UI display names to internal model type names
        if model_name == "AttentionNet":
            self.model_type = "attention_net"
        else:
            self.model_type = model_name.lower().replace(" ", "_")

        tabs = getattr(self, "tabs", None)
        if tabs is None:
            return
        try:
            tab_count = tabs.count()
        except RuntimeError:
            self.tabs = None
            return

        # Update hyperparameter widget — find its tab.
        for i in range(tab_count):
            try:
                if tabs.tabText(i) != "Hyperparameters":
                    continue
                old_widget = tabs.widget(i)
                # Remove the tab first so insertTab doesn't add a
                # duplicate. removeTab() re-parents the widget to None
                # so we must also delete it explicitly; deleteLater()
                # is safe here because we're on the main thread.
                tabs.removeTab(i)
                if old_widget is not None:
                    old_widget.deleteLater()
                self.hyperparam_widget = HyperparameterWidget(self.model_type, self.config)
                tabs.insertTab(i, self.hyperparam_widget, "Hyperparameters")
                tabs.setCurrentIndex(i)
            except RuntimeError:
                # Race during teardown — silently abandon this update.
                self.tabs = None
                return
            break

    # NOTE: closeEvent merged further below to ensure single definitive
    # cleanup implementation (stops worker, blocks signals, and marks
    # tabs as gone). The real closeEvent is defined later in this file.

    def _on_destroyed(self, *a, **k):
        """Slot for QObject.destroyed to mark teardown and attempt to
        defensively block remaining combo signals."""
        try:
            self._tearing_down = True
            for combo_attr in ("model_type_combo", "dataset_combo"):
                combo = getattr(self, combo_attr, None)
                if combo is not None:
                    try:
                        combo.blockSignals(True)
                        combo.currentTextChanged.disconnect()
                    except Exception:
                        pass
        except Exception:
            pass

    def _on_dataset_selected(self, dataset_name: str):
        """Handle dataset selection to update info."""
        if dataset_name and hasattr(self.parent(), 'data_manager'):
            try:
                self.dataset = self.parent().data_manager.load_dataset(dataset_name)
                self._update_dataset_info()
            except Exception as e:
                self._log(f"Error loading dataset: {e}")

    def _update_dataset_info(self):
        """Update the dataset information display."""
        if self.dataset and hasattr(self, 'info_layout'):
            metadata = self.dataset.get("metadata", {})
            # Clear old layout
            while self.info_layout.count():
                child = self.info_layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()

            # Add new info
            self.info_layout.addWidget(QLabel(f"Name: {metadata.get('name', 'Unknown')}"), 0, 0)
            self.info_layout.addWidget(QLabel(f"Samples: {metadata.get('num_samples', 0)}"), 0, 1)
            self.info_layout.addWidget(QLabel(f"Classes: {metadata.get('num_classes', 0)}"), 1, 0)
            self.info_layout.addWidget(QLabel(f"Channels: {metadata.get('num_channels', 0)}"), 1, 1)

    def _setup_features_tab(self):
        """Setup the features selection tab."""
        layout = QVBoxLayout(self.features_widget)

        # Mode Selection
        mode_group = QGroupBox("Feature Mode")
        mode_layout = QVBoxLayout(mode_group)

        self.btn_group = QButtonGroup(self)
        self.btn_group.setExclusive(True)

        self.radio_default = QRadioButton("Use All Features (Default)")
        self.radio_default.setChecked(True)
        self.btn_group.addButton(self.radio_default)
        mode_layout.addWidget(self.radio_default)

        self.radio_raw = QRadioButton("Use Raw EMG Only")
        self.btn_group.addButton(self.radio_raw)
        mode_layout.addWidget(self.radio_raw)

        self.radio_custom = QRadioButton("Custom Selection")
        self.btn_group.addButton(self.radio_custom)
        mode_layout.addWidget(self.radio_custom)

        self.btn_group.buttonClicked.connect(self._on_feature_mode_changed)

        layout.addWidget(mode_group)

        # Feature List
        list_group = QGroupBox("Available Features")
        list_layout = QVBoxLayout(list_group)

        self.feature_list = QListWidget()
        self.feature_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.feature_list.setEnabled(False)

        # Populate features
        features = get_registered_features()
        for name in sorted(features.keys()):
            item = QListWidgetItem(name)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.feature_list.addItem(item)

        list_layout.addWidget(self.feature_list)

        # Selection buttons
        btn_layout = QHBoxLayout()
        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.clicked.connect(self._on_select_all_features)
        self.select_all_btn.setEnabled(False)
        btn_layout.addWidget(self.select_all_btn)

        self.deselect_all_btn = QPushButton("Deselect All")
        self.deselect_all_btn.clicked.connect(self._on_deselect_all_features)
        self.deselect_all_btn.setEnabled(False)
        btn_layout.addWidget(self.deselect_all_btn)

        list_layout.addLayout(btn_layout)
        layout.addWidget(list_group)

    def _on_feature_mode_changed(self, button):
        """Handle feature mode change."""
        is_custom = self.radio_custom.isChecked()
        self.feature_list.setEnabled(is_custom)
        self.select_all_btn.setEnabled(is_custom)
        self.deselect_all_btn.setEnabled(is_custom)

    def _on_select_all_features(self):
        """Select all features in the list."""
        for i in range(self.feature_list.count()):
            self.feature_list.item(i).setCheckState(Qt.CheckState.Checked)

    def _on_deselect_all_features(self):
        """Deselect all features in the list."""
        for i in range(self.feature_list.count()):
            self.feature_list.item(i).setCheckState(Qt.CheckState.Unchecked)

    @Slot()
    def _on_start_training(self):
        """Start the training process."""
        self._log("Starting training...")
        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.save_btn.setEnabled(False)

        # Clear previous results
        self._train_losses = []
        self._val_losses = []
        self._train_accs = []
        self._val_accs = []
        self._iterations = []
        self.results_table.setRowCount(0)

        try:
            # Load dataset if not provided
            if not self.dataset:
                if hasattr(self, 'dataset_combo') and self.dataset_combo:
                    dataset_name = self.dataset_combo.currentText()
                    if not dataset_name:
                        raise ValueError("Please select a dataset")
                    # Load from data manager
                    if hasattr(self.parent(), 'data_manager'):
                        self._log(f"Loading dataset '{dataset_name}'...")
                        self.dataset = self.parent().data_manager.load_dataset(dataset_name)
                    else:
                        raise ValueError("Data manager not available")
                else:
                    raise ValueError("Please select a dataset")

            # Create model with hyperparameters
            # Use parent window's model_manager if available, or create with proper path
            if hasattr(self.parent(), 'model_manager'):
                model_manager = self.parent().model_manager
            else:
                # Fallback: create with default path
                pipeline_dir = Path(__file__).parent.parent.parent
                models_dir = pipeline_dir / "data" / "models"
                model_manager = ModelManager(models_dir)
            hyperparams = self.hyperparam_widget.get_hyperparameters()

            # Add feature configuration to hyperparameters
            feature_config = self.get_feature_config()
            hyperparams["feature_config"] = feature_config

            model = model_manager.create_model(self.model_type, **hyperparams)

            if feature_config["mode"] == "raw":
                self._log("Using RAW EMG signals (no feature extraction)")
            elif feature_config["mode"] == "custom":
                self._log(f"Using custom features: {', '.join(feature_config['features'])}")
            else:
                self._log("Using default feature set")

            self._log(f"Training {self.model_type} model...")

            # Split data
            X = self.dataset["X"]
            y = self.dataset["y"]

            # Seed from config if present, else preserve v1 default of 42.
            # Using getattr keeps PipelineConfig definitions backward compatible
            # — older configs without a `seed` field still work.
            random_state = int(getattr(self.config, "seed", 42) or 42)
            self._log(f"  split seed: {random_state}")

            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=random_state
            )

            self._log(f"Dataset split: Train={len(X_train)}, Validation={len(X_val)}")

            # Store model for later
            self._model = model
            self._model_manager = model_manager

            # Start training in background
            kwargs = {
                "window_size_ms": self.dataset["metadata"].get("window_size_ms", 200),
                "sampling_rate": self.dataset["metadata"].get("sampling_rate", 2000),
                "num_channels": self.dataset["metadata"].get("num_channels", 0),
                "random_state": random_state,
            }

            self._worker = TrainingWorker(model, X_train, y_train, X_val, y_val, kwargs)
            self._worker.progress.connect(self._on_progress)
            self._worker.iteration_update.connect(self._on_iteration_update)
            self._worker.finished.connect(self._on_training_finished)
            self._worker.error.connect(self._on_training_error)
            self._worker.start()

        except Exception as e:
            self._log(f"Error: {e}")
            self.train_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)

    @Slot()
    def _on_stop_training(self):
        """Stop the training process."""
        if self._worker:
            self._worker.stop()
            self._log("Stopping training...")
            if self._worker.isRunning():
                self._worker.wait(3000)
            if self._worker.isRunning():
                self._worker.terminate()
                self._worker.wait(2000)
            self._worker = None

    @Slot(int, str)
    def _on_progress(self, progress: int, message: str):
        """Handle progress update."""
        self.progress_bar.setValue(progress)
        self.status_label.setText(message)
        self._log(message)

    @Slot(int, float, float, float, float)
    def _on_iteration_update(self, iteration: int, train_loss: float, val_loss: float, train_acc: float = 0.0, val_acc: float = 0.0):
        """Handle iteration update for plotting."""
        self._iterations.append(iteration)
        self._train_losses.append(train_loss)
        self._val_losses.append(val_loss)
        self._train_accs.append(train_acc)
        self._val_accs.append(val_acc)

        # Update plots
        self.train_loss_curve.setData(self._iterations, self._train_losses)
        self.val_loss_curve.setData(self._iterations, self._val_losses)
        self.train_acc_curve.setData(self._iterations, self._train_accs)
        self.val_acc_curve.setData(self._iterations, self._val_accs)

        # Add row to results table
        row_position = self.results_table.rowCount()
        self.results_table.insertRow(row_position)
        self.results_table.setItem(row_position, 0, QTableWidgetItem(str(iteration)))
        self.results_table.setItem(row_position, 1, QTableWidgetItem(f"{train_loss:.4f}"))
        self.results_table.setItem(row_position, 2, QTableWidgetItem(f"{val_loss:.4f}"))
        self.results_table.setItem(row_position, 3, QTableWidgetItem(f"{train_acc:.4f}"))
        self.results_table.setItem(row_position, 4, QTableWidgetItem(f"{val_acc:.4f}"))

    @Slot(dict)
    def _on_training_finished(self, results: Dict[str, Any]):
        """Handle training completion."""
        self._log("=" * 50)
        self._log("Training Complete!")
        self._log(f"Final Training Accuracy: {results.get('training_accuracy', 0):.4f}")
        self._log(f"Final Validation Accuracy: {results.get('validation_accuracy', 0):.4f}")
        self._log("=" * 50)

        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.save_btn.setEnabled(True)

        # Add class names to model metadata
        if self._model.metadata and "label_names" in self.dataset["metadata"]:
            self._model.metadata.class_names = self.dataset["metadata"]["label_names"]

        # Store feature extraction info so prediction path knows what to expect
        if self._model.metadata:
            self._model.metadata.features_extracted = self.dataset["metadata"].get("features_extracted", False)
            self._model.metadata.feature_config = self.dataset["metadata"].get("feature_config", None)
            self._model.metadata.bad_channel_mode = self.dataset["metadata"].get("bad_channel_mode", "interpolate")

        self._results = results

        # -- New: set model name to the requested scheme: model_<datasetname>
        try:
            # Prefer dataset metadata name, fallback to dataset_combo text
            dataset_name = None
            if self.dataset and "metadata" in self.dataset:
                dataset_name = self.dataset["metadata"].get("name")
            if not dataset_name and hasattr(self, 'dataset_combo') and self.dataset_combo:
                dataset_name = self.dataset_combo.currentText()
            if not dataset_name:
                dataset_name = "unknown"

            # Sanitize dataset name for filesystem: replace colons and spaces
            safe_name = dataset_name.replace(":", "-").replace(" ", "_")
            # Also remove any path separators just in case
            safe_name = safe_name.replace("/", "_").replace("\\\\", "_")

            # Use model type if available
            prefix = self.model_type if self.model_type else "model"
            # Add time stamp so re-training doesn't overwrite previous runs
            ts = _dt.now().strftime("%H%M%S")
            model_name = f"{prefix}_{safe_name}_{ts}"

            if hasattr(self, '_model') and self._model:
                # Set model.name so callers that save by model.name use the desired scheme
                try:
                    self._model.name = model_name
                except Exception:
                    pass

                # Also set metadata name if present
                try:
                    if hasattr(self._model, 'metadata'):
                        self._model.metadata.name = model_name
                except Exception:
                    pass

            self._log(f"Model will be saved as: {model_name}")
        except Exception as e:
            self._log(f"Warning: could not set model name automatically: {e}")

        if self._worker and self._worker.isRunning():
            self._worker.wait(5000)
        self._worker = None

    @Slot(str)
    def _on_training_error(self, error: str):
        """Handle training error."""
        self._log(f"Error: {error}")
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText(f"Error: {error}")
        if self._worker and self._worker.isRunning():
            self._worker.wait(3000)
        self._worker = None

    def closeEvent(self, event):
        """Ensure worker thread is stopped and disconnect signals before dialog destruction.

        This method performs four defensive steps:
        1) Mark teardown so queued slots bail out early.
        2) Disconnect ALL signals on every child combo/widget before Qt
           starts destroying children. This is the critical step: it
           must happen *before* super().closeEvent() so that no queued
           currentTextChanged can fire after C++ children are gone.
        3) Null self.tabs so any signal that races past step 2 is a
           cheap no-op rather than a RuntimeError.
        4) Stop any background worker thread and then call base closeEvent.
        """
        # 1) Mark teardown immediately — slots check this flag first.
        self._tearing_down = True

        # 2) Aggressively sever all Qt signals on interactive child widgets
        #    BEFORE super().closeEvent() triggers C++ child destruction.
        #    Using a broad try/except per widget so one failure doesn't
        #    prevent the others from being disconnected.
        for combo_attr in ("model_type_combo", "dataset_combo"):
            combo = getattr(self, combo_attr, None)
            if combo is None:
                continue
            try:
                combo.blockSignals(True)
            except Exception:
                pass
            try:
                combo.currentTextChanged.disconnect()
            except Exception:
                pass

        # Also block the hyperparameter widget's internal combos if present.
        hp = getattr(self, "hyperparam_widget", None)
        if hp is not None:
            try:
                hp.blockSignals(True)
            except Exception:
                pass

        # 3) Null the tabs reference so any signal that slipped through
        #    the blockSignals() window hits the `if tabs is None: return`
        #    guard in _on_model_type_changed instead of crashing.
        self.tabs = None

        # 4) Stop worker thread before handing control back to Qt.
        try:
            if self._worker and self._worker.isRunning():
                self._worker.stop()
                self._worker.wait(3000)
                if self._worker.isRunning():
                    self._worker.terminate()
                    self._worker.wait(2000)
        except Exception:
            pass
        finally:
            self._worker = None

        super().closeEvent(event)

    def get_trained_model(self):
        """Get the trained model."""
        return getattr(self, '_model', None)

    def get_results(self) -> Dict[str, Any]:
        """Get training results."""
        return getattr(self, '_results', {})

    @Slot()
    def _on_save_model(self):
        """Save the trained model without closing."""
        if self._model:
            try:
                # Use parent window's model_manager if available
                if hasattr(self.parent(), 'model_manager'):
                    model_manager = self.parent().model_manager
                    # The model name is already set in _on_training_finished
                    model_manager._current_model = self._model
                    self._model.save(model_manager.models_dir / self._model.name)
                    self._log(f"Model saved: {self._model.name}")

                    # Refresh parent model list if possible
                    if hasattr(self.parent(), '_refresh_models'):
                        self.parent()._refresh_models()

                    QMessageBox.information(self, "Success", f"Model saved as '{self._model.name}'")
                else:
                    # Fallback save
                    save_dir = Path("data/models") / self._model.name
                    self._model.save(save_dir)
                    self._log(f"Model saved to: {save_dir}")
                    QMessageBox.information(self, "Success", f"Model saved to '{save_dir}'")
            except Exception as e:
                self._log(f"Error saving model: {e}")
                QMessageBox.critical(self, "Error", f"Failed to save model: {e}")

    def get_feature_config(self) -> Dict[str, Any]:
        """Get the feature configuration based on UI selection."""
        config = {
            "mode": "default",
            "features": []
        }

        if self.radio_raw.isChecked():
            config["mode"] = "raw"
        elif self.radio_default.isChecked():
            config["mode"] = "default"
        elif self.radio_custom.isChecked():
            config["mode"] = "custom"
            selected_features = []
            for i in range(self.feature_list.count()):
                item = self.feature_list.item(i)
                if item.checkState() == Qt.CheckState.Checked:
                    selected_features.append(item.text())
            config["features"] = selected_features

        return config