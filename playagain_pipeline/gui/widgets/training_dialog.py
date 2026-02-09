"""
Training Progress Dialog for detailed model training visualization.

Provides hyperparameter editing, real-time progress monitoring,
and training result visualization.
"""

from typing import Optional, Dict, Any, List
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QLabel, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox,
    QPushButton, QProgressBar, QTabWidget, QWidget, QTextEdit,
    QTableWidget, QTableWidgetItem, QHeaderView
)
from PySide6.QtCore import Signal, Slot, QThread
import pyqtgraph as pg

from playagain_pipeline.config.config import PipelineConfig, ModelConfig


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
            self.progress.emit(10, "Extracting features...")

            # Define callback for iterative models (e.g. MLP)
            def progress_callback(epoch, train_loss, val_loss, train_acc=0.0, val_acc=0.0):
                self.iteration_update.emit(epoch, train_loss, val_loss, train_acc, val_acc)
                # Keep progress between 10 and 90 during training
                # This is approximate since we don't know total epochs easily here without parsing kwargs
                # Removed explicit log emission as requested, data is in plots and table

            # Add callback to kwargs
            self.kwargs['callback'] = progress_callback

            # Train model
            results = self.model.train(
                self.X_train, self.y_train,
                self.X_val, self.y_val,
                **self.kwargs
            )

            self.progress.emit(100, "Training complete!")
            self.finished.emit(results)

        except Exception as e:
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
        self.optimizer_combo.addItems(["Adam", "RMSprop", "SGD"])
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
        self.optimizer_combo.addItems(["Adam", "RMSprop", "SGD"])
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

        layout.addLayout(grid)

    def _setup_inception_params(self, layout):
        grid = QGridLayout()

        # Base Inception Channels
        grid.addWidget(QLabel("Base Channels:"), 0, 0)
        self.inc_base_spin = QSpinBox()
        self.inc_base_spin.setRange(8, 256)
        self.inc_base_spin.setValue(32)
        grid.addWidget(self.inc_base_spin, 0, 1)

        # Attention Reduction Ratio
        grid.addWidget(QLabel("Attention Reduction:"), 1, 0)
        self.att_red_spin = QSpinBox()
        self.att_red_spin.setRange(1, 32)
        self.att_red_spin.setValue(8)
        grid.addWidget(self.att_red_spin, 1, 1)

        # Learning Rate (Inherited logic)
        grid.addWidget(QLabel("Learning Rate:"), 2, 0)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.00001, 0.1)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setDecimals(5)
        grid.addWidget(self.lr_spin, 2, 1)

        layout.addLayout(grid)

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
                "epochs": self.epochs_spin.value(),
                "batch_size": self.batch_spin.value(),
                "learning_rate": self.lr_spin.value(),
                "optimizer": self.optimizer_combo.currentText().lower(),
                "early_stopping": self.early_stop_check.isChecked(),
                "patience": self.patience_spin.value(),
                "dropout": 0.2
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
                "epochs": self.epochs_spin.value(),
                "batch_size": self.batch_spin.value(),
                "learning_rate": self.lr_spin.value(),
                "optimizer": self.optimizer_combo.currentText().lower(),
                "early_stopping": self.early_stop_check.isChecked(),
                "patience": self.patience_spin.value()
            }

        elif self.model_type == "inception":
            params = {
                "base_channels": self.inc_base_spin.value(),
                "attention_reduction": self.att_red_spin.value(),
                "learning_rate": self.lr_spin.value()
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
                self.model_type_combo.addItems(self.available_models)
                self.model_type_combo.currentTextChanged.connect(self._on_model_type_changed)
                selection_layout.addWidget(self.model_type_combo, 0, 1)
                self.model_type = self.available_models[0].lower().replace(" ", "_")
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
        self.save_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.save_btn)

        layout.addLayout(button_layout)

    def _log(self, message: str):
        """Add message to log."""
        self.log_text.append(message)

    def _on_model_type_changed(self, model_name: str):
        """Handle model type change."""
        self.model_type = model_name.lower().replace(" ", "_")
        # Update hyperparameter widget
        # Find the tab index for hyperparameters
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i) == "Hyperparameters":
                # Remove old widget
                old_widget = self.tabs.widget(i)
                if old_widget:
                    old_widget.deleteLater()
                # Create new widget
                self.hyperparam_widget = HyperparameterWidget(self.model_type, self.config)
                self.tabs.insertTab(i, self.hyperparam_widget, "Hyperparameters")
                self.tabs.setCurrentIndex(i)
                break

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

    @Slot()
    def _on_start_training(self):
        """Start the training process."""
        from playagain_pipeline.models.classifier import ModelManager
        from sklearn.model_selection import train_test_split

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
            from pathlib import Path
            # Use parent window's model_manager if available, or create with proper path
            if hasattr(self.parent(), 'model_manager'):
                model_manager = self.parent().model_manager
            else:
                # Fallback: create with default path
                pipeline_dir = Path(__file__).parent.parent.parent
                models_dir = pipeline_dir / "data" / "models"
                model_manager = ModelManager(models_dir)
            hyperparams = self.hyperparam_widget.get_hyperparameters()
            model = model_manager.create_model(self.model_type, **hyperparams)

            self._log(f"Training {self.model_type} model...")

            # Split data
            X = self.dataset["X"]
            y = self.dataset["y"]
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )

            self._log(f"Dataset split: Train={len(X_train)}, Validation={len(X_val)}")

            # Store model for later
            self._model = model
            self._model_manager = model_manager

            # Start training in background
            kwargs = {
                "window_size_ms": self.dataset["metadata"].get("window_size_ms", 200),
                "sampling_rate": self.dataset["metadata"].get("sampling_rate", 2000)
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
            model_name = f"{prefix}_{safe_name}"

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

    @Slot(str)
    def _on_training_error(self, error: str):
        """Handle training error."""
        self._log(f"Error: {error}")
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText(f"Error: {error}")

    def get_trained_model(self):
        """Get the trained model."""
        return getattr(self, '_model', None)

    def get_results(self) -> Dict[str, Any]:
        """Get training results."""
        return getattr(self, '_results', {})
