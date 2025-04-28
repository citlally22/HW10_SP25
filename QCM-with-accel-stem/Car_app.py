#region imports
from Car_GUI import Ui_Form
import sys
from PyQt5 import QtCore as qtc
from PyQt5 import QtWidgets as qtw
from PyQt5 import QtGui as qtg
from QuarterCarModel import CarController

# Matplotlib imports needed for embedding graphs into the PyQt GUI
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
#endregion

class MainWindow(qtw.QWidget, Ui_Form):
    """
    Main application window class for the Quarter Car Model Simulator.
    Handles GUI setup, signal-slot connections, and delegates logic to CarController.
    """

    def __init__(self):
        """
        Initialize the main window and set up all widgets and signal connections.
        """
        super().__init__()
        self.setupUi(self)

        # Set up the car controller with input and output widgets
        input_widgets = (self.le_m1, self.le_v, self.le_k1, self.le_c1, self.le_m2, self.le_k2,
                         self.le_ang, self.le_tmax, self.chk_IncludeAccel)
        display_widgets = (self.gv_Schematic, self.chk_LogX, self.chk_LogY, self.chk_LogAccel,
                           self.chk_ShowAccel, self.lbl_MaxMinInfo, self.layout_horizontal_main)

        # Instantiate the controller
        self.controller = CarController((input_widgets, display_widgets))

        # Connect GUI buttons and checkboxes to their respective functions
        self.btn_calculate.clicked.connect(self.controller.calculate)
        self.pb_Optimize.clicked.connect(self.doOptimize)
        self.chk_LogX.stateChanged.connect(self.controller.doPlot)
        self.chk_LogY.stateChanged.connect(self.controller.doPlot)
        self.chk_LogAccel.stateChanged.connect(self.controller.doPlot)
        self.chk_ShowAccel.stateChanged.connect(self.controller.doPlot)

        self.show()

    def doOptimize(self):
        """
        Slot to trigger the optimization routine when the 'Optimize Suspension' button is pressed.
        """
        app.setOverrideCursor(qtc.Qt.WaitCursor)
        self.controller.OptimizeSuspension()
        app.restoreOverrideCursor()

if __name__ == '__main__':
    # Standard PyQt application startup
    app = qtw.QApplication(sys.argv)
    mw = MainWindow()
    mw.setWindowTitle('Quarter Car Model')
    sys.exit(app.exec())