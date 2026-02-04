"""GUI widgets for gesture pipeline."""

from playagain_pipeline.gui.widgets.emg_plot import EMGPlotWidget
from playagain_pipeline.gui.widgets.protocol_widget import (
    ProtocolWidget,
    GestureDisplayWidget,
    ProtocolProgressWidget
)

__all__ = [
    "EMGPlotWidget",
    "ProtocolWidget",
    "GestureDisplayWidget",
    "ProtocolProgressWidget"
]
