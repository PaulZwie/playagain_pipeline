"""GUI widgets for gesture pipeline."""

from playagain_pipeline.gui.widgets.emg_plot import EMGPlotWidget, EMGPlotWindow
from playagain_pipeline.gui.widgets.protocol_widget import (
    ProtocolWidget,
    GestureDisplayWidget,
    ProtocolProgressWidget
)
from playagain_pipeline.gui.widgets.config_dialog import (
    ConfigurationDialog,
    BraceletVisualizationWidget
)
from playagain_pipeline.gui.widgets.training_dialog import (
    TrainingProgressDialog,
    HyperparameterWidget
)
from playagain_pipeline.gui.widgets.feature_selection import FeatureSelectionDialog
from playagain_pipeline.gui.widgets.calibration_dialog import CalibrationDialog
from playagain_pipeline.gui.widgets.emg_plot import VispyBiosignalPlot

__all__ = [
    "EMGPlotWidget",
    "EMGPlotWindow",
    "ProtocolWidget",
    "GestureDisplayWidget",
    "ProtocolProgressWidget",
    "ConfigurationDialog",
    "BraceletVisualizationWidget",
    "TrainingProgressDialog",
    "HyperparameterWidget",
    "FeatureSelectionDialog",
    "CalibrationDialog",
    "VispyBiosignalPlot"
]
