# region imports
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np
import math
from PyQt5 import QtWidgets as qtw
from PyQt5 import QtCore as qtc
from PyQt5 import QtGui as qtg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


# endregion

# region class definitions
# region specialized graphic items
class MassBlock(qtw.QGraphicsItem):
    """A rectangular graphical item representing a mass block (e.g., car body or wheel mass)."""

    def __init__(self, CenterX, CenterY, width=30, height=10, parent=None, pen=None, brush=None, name='CarBody',
                 mass=10):
        """Initialize the MassBlock with position, size, and properties.

        Args:
            CenterX (float): X-coordinate of the block's center.
            CenterY (float): Y-coordinate of the block's center.
            width (float, optional): Width of the block. Defaults to 30.
            height (float, optional): Height of the block. Defaults to 10.
            parent (QGraphicsItem, optional): Parent item. Defaults to None.
            pen (QPen, optional): Pen for drawing outline. Defaults to None.
            brush (QBrush, optional): Brush for filling the block. Defaults to None.
            name (str, optional): Name of the block. Defaults to 'CarBody'.
            mass (float, optional): Mass of the block. Defaults to 10.
        """
        super().__init__(parent)
        # Set center coordinates
        self.x = CenterX
        self.y = CenterY
        # Set drawing properties
        self.pen = pen
        self.brush = brush
        # Set dimensions
        self.width = width
        self.height = height
        # Calculate top-left corner
        self.top = self.y - self.height / 2
        self.left = self.x - self.width / 2
        # Define bounding rectangle
        self.rect = qtc.QRectF(self.left, self.top, self.width, self.height)
        # Set name and mass
        self.name = name
        self.mass = mass
        # Initialize transformation
        self.transformation = qtg.QTransform()
        # Set tooltip with position and mass info
        stTT = self.name + "\nx={:0.3f}, y={:0.3f}\nmass = {:0.3f}".format(self.x, self.y, self.mass)
        self.setToolTip(stTT)

    def boundingRect(self):
        """Return the bounding rectangle of the item.

        Returns:
            QRectF: The bounding rectangle after applying transformations.
        """
        bounding_rect = self.transformation.mapRect(self.rect)
        return bounding_rect

    def paint(self, painter, option, widget=None):
        """Paint the mass block on the graphics scene.

        Args:
            painter (QPainter): The painter object.
            option (QStyleOptionGraphicsItem): Style options.
            widget (QWidget, optional): The widget being painted on.
        """
        # Reset transformation
        self.transformation.reset()
        # Set pen if provided
        if self.pen is not None:
            painter.setPen(self.pen)
        # Set brush if provided
        if self.brush is not None:
            painter.setBrush(self.brush)
        # Update rectangle position
        self.top = -self.height / 2
        self.left = -self.width / 2
        self.rect = qtc.QRectF(self.left, self.top, self.width, self.height)
        # Draw the rectangle
        painter.drawRect(self.rect)
        # Apply translation to center
        self.transformation.translate(self.x, self.y)
        self.setTransform(self.transformation)
        # Reset transformation for next paint
        self.transformation.reset()


class Wheel(qtw.QGraphicsItem):
    """A circular graphical item representing a wheel with an embedded mass block."""

    def __init__(self, CenterX, CenterY, radius=10, parent=None, pen=None, wheelBrush=None, massBrush=None,
                 name='Wheel', mass=10):
        """Initialize the Wheel with position, radius, and properties.

        Args:
            CenterX (float): X-coordinate of the wheel's center.
            CenterY (float): Y-coordinate of the wheel's center.
            radius (float, optional): Radius of the wheel. Defaults to 10.
            parent (QGraphicsItem, optional): Parent item. Defaults to None.
            pen (QPen, optional): Pen for drawing outline. Defaults to None.
            wheelBrush (QBrush, optional): Brush for filling the wheel. Defaults to None.
            massBrush (QBrush, optional): Brush for the embedded mass block. Defaults to None.
            name (str, optional): Name of the wheel. Defaults to 'Wheel'.
            mass (float, optional): Mass of the wheel. Defaults to 10.
        """
        super().__init__(parent)
        # Set center coordinates
        self.x = CenterX
        self.y = CenterY
        # Set drawing properties
        self.pen = pen
        self.brush = wheelBrush
        # Set radius
        self.radius = radius
        # Define bounding rectangle
        self.rect = qtc.QRectF(self.x - self.radius, self.y - self.radius, self.radius * 2, self.radius * 2)
        # Set name and mass
        self.name = name
        self.mass = mass
        # Initialize transformation
        self.transformation = qtg.QTransform()
        # Set tooltip with position and mass info
        stTT = self.name + "\nx={:0.3f}, y={:0.3f}\nmass = {:0.3f}".format(self.x, self.y, self.mass)
        self.setToolTip(stTT)
        # Create embedded mass block
        self.massBlock = MassBlock(CenterX, CenterY, width=2 * radius * 0.85, height=radius / 3, pen=pen,
                                   brush=massBrush, name="Wheel Mass", mass=mass)

    def boundingRect(self):
        """Return the bounding rectangle of the wheel.

        Returns:
            QRectF: The bounding rectangle after applying transformations.
        """
        bounding_rect = self.transformation.mapRect(self.rect)
        return bounding_rect

    def addToScene(self, scene):
        """Add the wheel and its mass block to the graphics scene.

        Args:
            scene (QGraphicsScene): The scene to add the items to.
        """
        # Add wheel to scene
        scene.addItem(self)
        # Add mass block to scene
        scene.addItem(self.massBlock)

    def paint(self, painter, option, widget=None):
        """Paint the wheel on the graphics scene.

        Args:
            painter (QPainter): The painter object.
            option (QStyleOptionGraphicsItem): Style options.
            widget (QWidget, optional): The widget being painted on.
        """
        # Reset transformation
        self.transformation.reset()
        # Set pen if provided
        if self.pen is not None:
            painter.setPen(self.pen)
        # Set brush if provided
        if self.brush is not None:
            painter.setBrush(self.brush)
        # Update ellipse rectangle
        self.rect = qtc.QRectF(-self.radius, -self.radius, self.radius * 2, self.radius * 2)
        # Draw the ellipse
        painter.drawEllipse(self.rect)
        # Apply translation to center
        self.transformation.translate(self.x, self.y)
        self.setTransform(self.transformation)
        # Reset transformation for next paint
        self.transformation.reset()


class SpringItem(qtw.QGraphicsItem):
    """A graphical item representing a spring as a zigzag line."""

    def __init__(self, x1, y1, x2, y2, parent=None, pen=None):
        """Initialize the SpringItem with start and end points.

        Args:
            x1 (float): X-coordinate of the start point.
            y1 (float): Y-coordinate of the start point.
            x2 (float): X-coordinate of the end point.
            y2 (float): Y-coordinate of the end point.
            parent (QGraphicsItem, optional): Parent item. Defaults to None.
            pen (QPen, optional): Pen for drawing the spring. Defaults to None.
        """
        super().__init__(parent)
        # Set start and end coordinates
        self.x1, self.y1 = x1, y1
        self.x2, self.y2 = x2, y2
        # Set drawing pen
        self.pen = pen
        # Calculate spring length
        self.length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        # Set zigzag width
        self.width = 20
        # Initialize transformation
        self.transformation = qtg.QTransform()

    def boundingRect(self):
        """Return the bounding rectangle of the spring.

        Returns:
            QRectF: The bounding rectangle encompassing the spring.
        """
        return qtc.QRectF(min(self.x1, self.x2) - self.width / 2, min(self.y1, self.y2),
                          abs(self.x2 - self.x1) + self.width, abs(self.y2 - self.y1) + self.width)

    def paint(self, painter, option, widget=None):
        """Paint the spring as a zigzag line.

        Args:
            painter (QPainter): The painter object.
            option (QStyleOptionGraphicsItem): Style options.
            widget (QWidget, optional): The widget being painted on.
        """
        print("Painting SpringItem")
        # Set pen if provided
        if self.pen is not None:
            painter.setPen(self.pen)
        # Disable brush for outline-only drawing
        painter.setBrush(qtc.Qt.NoBrush)
        # Calculate step sizes for zigzag
        dx = (self.x2 - self.x1) / 8
        dy = (self.y2 - self.y1) / 8
        points = []
        # Add start point
        points.append(qtc.QPointF(self.x1, self.y1))
        # Generate zigzag points
        for i in range(1, 8):
            x = self.x1 + i * dx
            y = self.y1 + i * dy + (self.width / 2 if i % 2 == 0 else -self.width / 2)
            points.append(qtc.QPointF(x, y))
        # Add end point
        points.append(qtc.QPointF(self.x2, self.y2))
        # Draw polyline
        painter.drawPolyline(points)


class DashpotItem:
    """A composite item representing a dashpot (damper) with lines and a central rectangle."""

    def __init__(self, x1, y1, x2, y2, parent=None, pen=None):
        """Initialize the DashpotItem with start and end points.

        Args:
            x1 (float): X-coordinate of the start point.
            y1 (float): Y-coordinate of the start point.
            x2 (float): X-coordinate of the end point.
            y2 (float): Y-coordinate of the end point.
            parent (QGraphicsItem, optional): Parent item. Defaults to None.
            pen (QPen, optional): Pen for drawing the dashpot. Defaults to None.
        """
        # Initialize list to store graphic items
        self.items = []
        # Set start and end coordinates
        self.x1, self.y1 = x1, y1
        self.x2, self.y2 = x2, y2
        # Set drawing pen
        self.pen = pen
        # Calculate dashpot length
        self.length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        # Set rectangle width
        self.width = 10

    def addToScene(self, scene):
        """Add the dashpot components to the graphics scene.

        Args:
            scene (QGraphicsScene): The scene to add the items to.
        """
        # Calculate midpoint
        mid_x = (self.x1 + self.x2) / 2
        mid_y = (self.y1 + self.y2) / 2
        # Create two lines
        line1 = qtw.QGraphicsLineItem(self.x1, self.y1, mid_x, mid_y)
        line2 = qtw.QGraphicsLineItem(mid_x, mid_y, self.x2, self.y2)
        # Create central rectangle
        rect = qtw.QGraphicsRectItem(mid_x - self.width / 4, mid_y - self.width / 4, self.width / 2, self.width / 2)
        # Set pen for all items
        if self.pen is not None:
            line1.setPen(self.pen)
            line2.setPen(self.pen)
            rect.setPen(self.pen)
        # Store items
        self.items = [line1, line2, rect]
        # Add items to scene
        for item in self.items:
            scene.addItem(item)


# endregion

# region MVC for quarter car model
class CarModel:
    """Model class for the quarter car simulation, storing parameters and results."""

    def __init__(self):
        """Initialize the CarModel with default parameters and constraints."""
        # Initialize results storage
        self.results = []
        # Set simulation time parameters
        self.tmax = 3.0
        self.t = np.linspace(0, self.tmax, 200)
        self.tramp = 1.0
        # Set road profile parameters
        self.angrad = 0.1
        self.ymag = 6.0 / (12 * 3.3)
        self.yangdeg = 45.0
        self.results = None

        # Set physical parameters
        self.m1 = 450.0  # Sprung mass (car body)
        self.m2 = 20.0  # Unsprung mass (wheel)
        self.c1 = 4500.0  # Damping coefficient
        self.k1 = 15000.0  # Suspension spring constant
        self.k2 = 90000.0  # Tire spring constant
        self.v = 120.0  # Vehicle velocity (km/h)

        # Calculate spring constant constraints
        g = 9.81
        self.mink1 = (self.m1 * g) / 0.1524
        self.maxk1 = (self.m1 * g) / 0.0762
        total_mass = self.m1 + self.m2
        self.mink2 = (total_mass * g) / 0.0381
        self.maxk2 = (total_mass * g) / 0.01905
        # Initialize acceleration and error metrics
        self.accel = None
        self.accelMax = 0.0
        self.accelLim = 0.2
        self.SSE = 0.0


class CarView:
    """View class for the quarter car model, handling GUI and visualization."""

    def __init__(self, args):
        """Initialize the CarView with input and display widgets.

        Args:
            args (tuple): Tuple containing input_widgets and display_widgets.
        """
        # Unpack input and display widgets
        self.input_widgets, self.display_widgets = args
        self.le_m1, self.le_v, self.le_k1, self.le_c1, self.le_m2, self.le_k2, self.le_ang, \
            self.le_tmax, self.chk_IncludeAccel = self.input_widgets

        self.gv_Schematic, self.chk_LogX, self.chk_LogY, self.chk_LogAccel, \
            self.chk_ShowAccel, self.lbl_MaxMinInfo, self.layout_horizontal_main = self.display_widgets

        # Initialize Matplotlib figure and canvas
        self.figure = Figure(tight_layout=True, frameon=True, facecolor='none')
        self.canvas = FigureCanvasQTAgg(self.figure)
        # Add canvas to layout
        self.layout_horizontal_main.addWidget(self.canvas)

        # Create plot axes
        self.ax = self.figure.add_subplot()
        if self.ax is not None:
            self.ax1 = self.ax.twinx()

        # Build the graphics scene
        self.buildScene()

    def updateView(self, model=None):
        """Update the GUI with model parameters and results.

        Args:
            model (CarModel, optional): The model containing updated parameters.
        """
        # Update input fields with model values
        self.le_m1.setText("{:0.2f}".format(model.m1))
        self.le_k1.setText("{:0.2f}".format(model.k1))
        self.le_c1.setText("{:0.2f}".format(model.c1))
        self.le_m2.setText("{:0.2f}".format(model.m2))
        self.le_k2.setText("{:0.2f}".format(model.k2))
        self.le_ang.setText("{:0.2f}".format(model.yangdeg))
        self.le_tmax.setText("{:0.2f}".format(model.tmax))
        # Update info label with spring constraints and SSE
        stTmp = "k1_min = {:0.2f}, k1_max = {:0.2f}\nk2_min = {:0.2f}, k2_max = {:0.2f}\n".format(model.mink1,
                                                                                                  model.maxk1,
                                                                                                  model.mink2,
                                                                                                  model.maxk2)
        stTmp += "SSE = {:0.2f}".format(model.SSE)
        self.lbl_MaxMinInfo.setText(stTmp)
        # Update plot
        self.doPlot(model)

    def buildScene(self):
        """Build the graphics scene with schematic components."""
        print("Building scene")
        # Create and configure scene
        self.scene = qtw.QGraphicsScene()
        self.scene.setObjectName("MyScene")
        self.scene.setSceneRect(-200, -200, 400, 400)

        # Set scene to graphics view
        self.gv_Schematic.setScene(self.scene)
        # Initialize pens and brushes
        self.setupPensAndBrushes()
        print("Adding wheel")
        # Create and add wheel
        self.Wheel = Wheel(0, 50, 50, pen=self.penWheel, wheelBrush=self.brushWheel, massBrush=self.brushMass,
                           name="Wheel", mass=20)
        self.Wheel.addToScene(self.scene)
        print("Adding car body")
        # Create and add car body
        self.CarBody = MassBlock(0, -70, 100, 30, pen=self.penWheel, brush=self.brushMass, name="Car Body", mass=450)
        self.scene.addItem(self.CarBody)

        print("Adding spring")
        # Create and add suspension spring
        spring = SpringItem(0, -40, 0, 50, pen=self.penWheel)
        self.scene.addItem(spring)
        print("Adding dashpot")
        # Create and add dashpot
        dashpot = DashpotItem(20, -40, 20, 50, pen=self.penWheel)
        dashpot.addToScene(self.scene)
        print("Adding tire spring")
        # Create and add tire spring
        tire_spring = SpringItem(0, 100, 0, 150, pen=self.penWheel)
        self.scene.addItem(tire_spring)
        print("Adding ground")
        # Create and add ground line
        ground = qtw.QGraphicsLineItem(-150, 150, 150, 150)
        ground.setPen(self.penWheel)
        self.scene.addItem(ground)
        print("Scene built successfully")

    def setupPensAndBrushes(self):
        """Initialize pens and brushes for drawing schematic components."""
        # Set pen for outlines
        self.penWheel = qtg.QPen(qtg.QColor("orange"))
        self.penWheel.setWidth(1)
        # Set brush for wheel fill
        self.brushWheel = qtg.QBrush(qtg.QColor.fromHsv(35, 255, 255, 64))
        # Set brush for mass fill
        self.brushMass = qtg.QBrush(qtg.QColor(200, 200, 200, 128))

    def doPlot(self, model=None):
        """Plot simulation results using Matplotlib.

        Args:
            model (CarModel, optional): The model containing simulation results.
        """
        if model.results is None:
            return
        # Get axes
        ax = self.ax
        ax1 = self.ax1
        # Clear previous plots
        ax.clear()
        ax1.clear()
        # Get time and position data
        t = model.t
        ycar = model.results[:, 0]
        ywheel = model.results[:, 2]
        accel = model.accel

        # Calculate road profile
        yroad = np.zeros_like(t)
        for i in range(len(t)):
            if t[i] < model.tramp:
                yroad[i] = model.ymag * (t[i] / model.tramp)
            else:
                yroad[i] = model.ymag

        # Set x-axis limits and scale
        if self.chk_LogX.isChecked():
            ax.set_xlim(max(1e-6, t.min()), t.max())
            ax.set_xscale('log')
        else:
            ax.set_xlim(0.0, model.tmax)
            ax.set_xscale('linear')

        # Set y-axis limits and scale for positions
        ycar_min = np.min(ycar[ycar > 0]) if np.any(ycar > 0) else 1e-6
        ywheel_min = np.min(ywheel[ywheel > 0]) if np.any(ywheel > 0) else 1e-6
        yroad_min = np.min(yroad[yroad > 0]) if np.any(yroad > 0) else 1e-6
        y_max = max(ycar.max(), ywheel.max(), yroad.max())
        y_min = min(ycar.min(), ywheel.min(), yroad.min())

        if self.chk_LogY.isChecked():
            ax.set_ylim(ycar_min / 1.05, y_max * 1.05)
            ax.set_yscale('log')
        else:
            ax.set_ylim(y_min / 1.05, y_max * 1.05)
            ax.set_yscale('linear')

        # Plot position data
        ax.plot(t, ycar, 'b-', label='Body Position')
        ax.plot(t, ywheel, 'r-', label='Wheel Position')
        ax.plot(t, yroad, 'k--', label='Road Profile')

        # Plot acceleration if checked
        if self.chk_ShowAccel.isChecked() and accel is not None:
            accel_min = np.min(accel[accel > 0]) if np.any(accel > 0) else 1e-6
            accel_max = np.max(accel) if np.any(accel) else 1e-6
            if self.chk_LogAccel.isChecked():
                ax1.set_ylim(accel_min / 1.05, max(accel_max, model.accelLim) * 1.05)
                ax1.set_yscale('log')
            else:
                ax1.set_ylim(min(accel.min(), -model.accelLim) / 1.05, max(accel_max, model.accelLim) * 1.05)
                ax1.set_yscale('linear')
            ax1.plot(t, accel, 'g-', label='Body Accel')
            ax1.axhline(y=accel.max(), color='orange', linestyle='--')
            ax1.set_ylabel("Y'' (g)", fontsize='large')
            ax1.legend(loc='upper right')

        # Set labels and legend
        ax.set_ylabel("Vertical Position (m)", fontsize='large')
        ax.set_xlabel("time (s)", fontsize='large')
        ax.legend(loc='upper left')

        # Add reference lines
        ax.axvline(x=model.tramp)
        ax.axhline(y=model.ymag)

        # Configure tick parameters
        ax.tick_params(axis='both', which='both', direction='in', top=True, labelsize='large')
        ax1.tick_params(axis='both', which='both', direction='in', right=True, labelsize='large')

        # Draw the plot
        self.canvas.draw()


class CarController:
    """Controller class for the quarter car model, handling calculations and updates."""

    def __init__(self, args):
        """Initialize the CarController with input and display widgets.

        Args:
            args (tuple): Tuple containing input_widgets and display_widgets.
        """
        # Unpack input and display widgets
        self.input_widgets, self.display_widgets = args
        self.le_m1, self.le_v, self.le_k1, self.le_c1, self.le_m2, self.le_k2, self.le_ang, \
            self.le_tmax, self.chk_IncludeAccel = self.input_widgets

        self.gv_Schematic, self.chk_LogX, self.chk_LogY, self.chk_LogAccel, \
            self.chk_ShowAccel, self.lbl_MaxMinInfo, self.layout_horizontal_main = self.display_widgets

        # Initialize model and view
        self.model = CarModel()
        self.view = CarView(args)

        # Initialize acceleration checkbox
        self.chk_IncludeAccel = qtw.QCheckBox()

    def ode_system(self, X, t):
        """Define the ODE system for the quarter car model.

        Args:
            X (list): State vector [x1, x1dot, x2, x2dot].
            t (float): Time point.

        Returns:
            list: Derivatives [x1dot, x1ddot, x2dot, x2ddot].
        """
        # Calculate road profile at time t
        if t < self.model.tramp:
            y = self.model.ymag * (t / self.model.tramp)
        else:
            y = self.model.ymag

        # Unpack state variables
        x1 = X[0]
        x1dot = X[1]
        x2 = X[2]
        x2dot = X[3]

        # Get model parameters
        m1 = self.model.m1
        m2 = self.model.m2
        c1 = self.model.c1
        k1 = self.model.k1
        k2 = self.model.k2

        # Calculate accelerations
        x1ddot = (-k1 * (x1 - x2) - c1 * (x1dot - x2dot)) / m1
        x2ddot = (k1 * (x1 - x2) + c1 * (x1dot - x2dot) - k2 * (x2 - y)) / m2

        return [x1dot, x1ddot, x2dot, x2ddot]

    def calculate(self, doCalc=True):
        """Update model parameters from GUI inputs and perform calculations.

        Args:
            doCalc (bool, optional): Whether to perform simulation. Defaults to True.
        """
        # Update model parameters from input fields
        self.model.m1 = float(self.le_m1.text())
        self.model.v = float(self.le_v.text())
        self.model.k1 = float(self.le_k1.text())
        self.model.c1 = float(self.le_c1.text())
        self.model.m2 = float(self.le_m2.text())
        self.model.k2 = float(self.le_k2.text())

        # Enforce positive parameter values
        if self.model.k1 <= 0:
            self.model.k1 = 15000
        if self.model.k2 <= 0:
            self.model.k2 = 90000
        if self.model.m1 <= 0:
            self.model.m1 = 450
        if self.model.m2 <= 0:
            self.model.m2 = 20
        if self.model.c1 <= 0:
            self.model.c1 = 4500

        # Calculate spring constant constraints
        g = 9.81
        self.model.mink1 = (self.model.m1 * g) / 0.1524
        self.model.maxk1 = (self.model.m1 * g) / 0.0762
        total_mass = self.model.m1 + self.model.m2
        self.model.mink2 = (total_mass * g) / 0.0381
        self.model.maxk2 = (total_mass * g) / 0.01905

        # Set road profile amplitude
        ymag = 6.0 / (12.0 * 3.3)
        if ymag is not None:
            self.model.ymag = ymag

        # Update angle and simulation time
        self.model.yangdeg = float(self.le_ang.text())
        self.model.tmax = float(self.le_tmax.text())

        # Perform simulation if requested
        if doCalc:
            self.doCalc()

        # Calculate SSE and update view
        self.SSE((self.model.k1, self.model.c1, self.model.k2), optimizing=False)
        self.view.updateView(self.model)

    def doCalc(self, doPlot=True, doAccel=True):
        """Perform the simulation by solving ODEs.

        Args:
            doPlot (bool, optional): Whether to update the plot. Defaults to True.
            doAccel (bool, optional): Whether to calculate acceleration. Defaults to True.
        """
        # Convert velocity to m/s
        v = 1000 * self.model.v / 3600
        # Convert angle to radians
        self.model.angrad = self.model.yangdeg * math.pi / 180.0

        # Enforce minimum velocity and angle
        if v <= 0:
            v = 1.0
        if abs(math.sin(self.model.angrad)) <= 1e-6:
            self.model.angrad = math.radians(5)

        # Calculate ramp time
        self.model.tramp = self.model.ymag / (math.sin(self.model.angrad) * v)

        # Set time array
        self.model.t = np.linspace(0, self.model.tmax, 2000)
        # Set initial conditions
        ic = [0, 0, 0, 0]
        # Solve ODE system
        self.model.results = odeint(self.ode_system, ic, self.model.t)

        # Calculate acceleration if requested
        if doAccel:
            self.calcAccel()
        # Update plot if requested
        if doPlot:
            self.doPlot()

    def calcAccel(self):
        """Calculate car body acceleration from velocity data.

        Returns:
            bool: True if calculation is successful.
        """
        # Initialize acceleration array
        N = len(self.model.t)
        self.model.accel = np.zeros(shape=N)
        vel = self.model.results[:, 1]
        # Calculate acceleration using finite differences
        for i in range(N):
            if i == N - 1:
                h = self.model.t[i] - self.model.t[i - 1]
                self.model.accel[i] = (vel[i] - vel[i - 1]) / (9.81 * h) + 1e-6
            else:
                h = self.model.t[i + 1] - self.model.t[i]
                self.model.accel[i] = (vel[i + 1] - vel[i]) / (9.81 * h) + 1e-6
        # Store maximum acceleration
        self.model.accelMax = self.model.accel.max()
        return True

    def OptimizeSuspension(self):
        """Optimize suspension parameters (k1, c1, k2) to minimize SSE."""
        # Update model parameters without simulation
        self.calculate(doCalc=False)
        print(f"Initial values: k1 = {self.model.k1}, c1 = {self.model.c1}, k2 = {self.model.k2}")
        print(
            f"Optimization bounds: k1 = ({self.model.mink1}, {self.model.maxk1}), c1 = (10, 10000), k2 = ({self.model.mink2}, {self.model.maxk2})")

        # Normalize parameters to [0, 1]
        k1_range = self.model.maxk1 - self.model.mink1
        c1_range = 10000 - 10
        k2_range = self.model.maxk2 - self.model.mink2

        # Set initial guess in normalized space
        x0 = np.array([
            (self.model.k1 - self.model.mink1) / k1_range,
            (self.model.c1 - 10) / c1_range,
            (self.model.k2 - self.model.mink2) / k2_range
        ])
        bounds = [(0, 1), (0, 1), (0, 1)]

        def normalized_SSE(x):
            """Calculate SSE for normalized parameters."""
            k1 = self.model.mink1 + x[0] * k1_range
            c1 = 10 + x[1] * c1_range
            k2 = self.model.mink2 + x[2] * k2_range
            return self.SSE((k1, c1, k2), optimizing=True)

        def callback(x):
            """Print optimization progress."""
            k1 = self.model.mink1 + x[0] * k1_range
            c1 = 10 + x[1] * c1_range
            k2 = self.model.mink2 + x[2] * k2_range
            print(f"Optimization step: k1 = {k1}, c1 = {c1}, k2 = {k2}")

        print(f"Starting optimization with normalized x0 = {x0}")
        # Perform optimization
        answer = minimize(normalized_SSE, x0, method='SLSQP', bounds=bounds, callback=callback,
                          options={'maxiter': 1000, 'disp': True})
        print(f"Optimization result: success = {answer.success}, message = {answer.message}")

        # Denormalize results
        self.model.k1 = self.model.mink1 + answer.x[0] * k1_range
        self.model.c1 = 10 + answer.x[1] * c1_range
        self.model.k2 = self.model.mink2 + answer.x[2] * k2_range

        # Clamp values to bounds
        self.model.k1 = max(self.model.mink1, min(self.model.maxk1, self.model.k1))
        self.model.k2 = max(self.model.mink2, min(self.model.maxk2, self.model.k2))
        self.model.c1 = max(10, min(10000, self.model.c1))
        print(f"Final optimized values: k1 = {self.model.k1}, c1 = {self.model.c1}, k2 = {self.model.k2}")
        # Perform final simulation
        self.doCalc()
        # Update view
        self.view.updateView(self.model)

    def SSE(self, vals, optimizing=True):
        """Calculate the sum of squared errors (SSE) for the simulation.

        Args:
            vals (tuple): Parameters (k1, c1, k2).
            optimizing (bool, optional): Whether called during optimization. Defaults to True.

        Returns:
            float: The calculated SSE.
        """
        # Unpack parameters
        k1, c1, k2 = vals
        # Update model parameters
        self.model.k1 = k1
        self.model.c1 = c1
        self.model.k2 = k2
        # Perform simulation
        self.doCalc(doPlot=False)

        # Calculate SSE
        SSE = 0
        for i in range(len(self.model.results[:, 0])):
            t = self.model.t[i]
            y = self.model.results[:, 0][i]
            if t < self.model.tramp:
                ytarget = self.model.ymag * (t / self.model.tramp)
            else:
                ytarget = self.model.ymag
            SSE += (y - ytarget) ** 2

        # Apply penalties during optimization
        if optimizing:
            if self.model.accelMax > self.model.accelLim and self.chk_IncludeAccel.isChecked():
                SSE += 1e12 * (self.model.accelMax - self.model.accelLim) ** 2
            # Penalize bounds violations
            if k1 < self.model.mink1:
                SSE += 1e15 * (self.model.mink1 - k1) ** 2
            if k1 > self.model.maxk1:
                SSE += 1e15 * (k1 - self.model.maxk1) ** 2
            if k2 < self.model.mink2:
                SSE += 1e15 * (self.model.mink2 - k2) ** 2
            if k2 > self.model.maxk2:
                SSE += 1e15 * (k2 - self.model.maxk2) ** 2

        # Store SSE in model
        self.model.SSE = SSE
        return SSE

    def doPlot(self):
        """Update the plot with current simulation results."""
        self.view.doPlot(self.model)

# endregion
# endregion