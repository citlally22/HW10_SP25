#region imports
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
#endregion

#region class definitions
#region specialized graphic items
class MassBlock(qtw.QGraphicsItem):
    def __init__(self, CenterX, CenterY, width=30, height=10, parent=None, pen=None, brush=None, name='CarBody', mass=10):
        super().__init__(parent)
        self.x = CenterX
        self.y = CenterY
        self.pen = pen
        self.brush = brush
        self.width = width
        self.height = height
        self.top = self.y - self.height/2
        self.left = self.x - self.width/2
        self.rect = qtc.QRectF(self.left, self.top, self.width, self.height)
        self.name = name
        self.mass = mass
        self.transformation = qtg.QTransform()
        stTT = self.name +"\nx={:0.3f}, y={:0.3f}\nmass = {:0.3f}".format(self.x, self.y, self.mass)
        self.setToolTip(stTT)

    def boundingRect(self):
        bounding_rect = self.transformation.mapRect(self.rect)
        return bounding_rect

    def paint(self, painter, option, widget=None):
        self.transformation.reset()
        if self.pen is not None:
            painter.setPen(self.pen)
        if self.brush is not None:
            painter.setBrush(self.brush)
        self.top = -self.height/2
        self.left = -self.width/2
        self.rect = qtc.QRectF(self.left, self.top, self.width, self.height)
        painter.drawRect(self.rect)
        self.transformation.translate(self.x, self.y)
        self.setTransform(self.transformation)
        self.transformation.reset()

class Wheel(qtw.QGraphicsItem):
    def __init__(self, CenterX, CenterY, radius=10, parent=None, pen=None, wheelBrush=None, massBrush=None, name='Wheel', mass=10):
        super().__init__(parent)
        self.x = CenterX
        self.y = CenterY
        self.pen = pen
        self.brush = wheelBrush
        self.radius = radius
        self.rect = qtc.QRectF(self.x - self.radius, self.y - self.radius, self.radius*2, self.radius*2)
        self.name = name
        self.mass = mass
        self.transformation = qtg.QTransform()
        stTT = self.name +"\nx={:0.3f}, y={:0.3f}\nmass = {:0.3f}".format(self.x, self.y, self.mass)
        self.setToolTip(stTT)
        self.massBlock = MassBlock(CenterX, CenterY, width=2*radius*0.85, height=radius/3, pen=pen, brush=massBrush, name="Wheel Mass", mass=mass)

    def boundingRect(self):
        bounding_rect = self.transformation.mapRect(self.rect)
        return bounding_rect

    def addToScene(self, scene):
        scene.addItem(self)
        scene.addItem(self.massBlock)

    def paint(self, painter, option, widget=None):
        self.transformation.reset()
        if self.pen is not None:
            painter.setPen(self.pen)
        if self.brush is not None:
            painter.setBrush(self.brush)
        self.rect = qtc.QRectF(-self.radius, -self.radius, self.radius*2, self.radius*2)
        painter.drawEllipse(self.rect)
        self.transformation.translate(self.x, self.y)
        self.setTransform(self.transformation)
        self.transformation.reset()

class SpringItem(qtw.QGraphicsItem):
    def __init__(self, x1, y1, x2, y2, parent=None, pen=None):
        super().__init__(parent)
        self.x1, self.y1 = x1, y1
        self.x2, self.y2 = x2, y2
        self.pen = pen
        self.length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        self.width = 20
        self.transformation = qtg.QTransform()

    def boundingRect(self):
        return qtc.QRectF(min(self.x1, self.x2) - self.width/2, min(self.y1, self.y2),
                         abs(self.x2 - self.x1) + self.width, abs(self.y2 - self.y1) + self.width)

    def paint(self, painter, option, widget=None):
        if self.pen is not None:
            painter.setPen(self.pen)
        painter.setBrush(qtc.Qt.NoBrush)
        dx = (self.x2 - self.x1) / 8
        dy = (self.y2 - self.y1) / 8
        points = []
        points.append(qtc.QPointF(self.x1, self.y1))
        for i in range(1, 8):
            x = self.x1 + i * dx
            y = self.y1 + i * dy + (self.width / 2 if i % 2 == 0 else -self.width / 2)
            points.append(qtc.QPointF(x, y))
        points.append(qtc.QPointF(self.x2, self.y2))
        painter.drawPolyline(points)

class DashpotItem(qtw.QGraphicsItem):
    def __init__(self, x1, y1, x2, y2, parent=None, pen=None):
        super().__init__(parent)
        self.x1, self.y1 = x1, y1
        self.x2, self.y2 = x2, y2
        self.pen = pen
        self.length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        self.width = 10
        self.transformation = qtg.QTransform()

    def boundingRect(self):
        return qtc.QRectF(min(self.x1, self.x2) - self.width, min(self.y1, self.y2),
                         abs(self.x2 - self.x1) + 2*self.width, abs(self.y2 - self.y1) + self.width)

    def paint(self, painter, option, widget=None):
        if self.pen is not None:
            painter.setPen(self.pen)
        painter.setBrush(qtc.Qt.NoBrush)
        # Simplified to avoid crash
        painter.drawLine(self.x1, self.y1, self.x2, self.y2)

#endregion

#region MVC for quarter car model
class CarModel:
    def __init__(self):
        self.results = []
        self.tmax = 3.0
        self.t = np.linspace(0, self.tmax, 200)
        self.tramp = 1.0
        self.angrad = 0.1
        self.ymag = 6.0 / (12 * 3.3)
        self.yangdeg = 45.0
        self.results = None

        self.m1 = 450.0  # Default, but user may change
        self.m2 = 20.0
        self.c1 = 4500.0
        self.k1 = 15000.0
        self.k2 = 90000.0
        self.v = 120.0

        g = 9.81
        self.mink1 = (self.m1 * g) / 0.1524
        self.maxk1 = (self.m1 * g) / 0.0762
        total_mass = self.m1 + self.m2
        self.mink2 = (total_mass * g) / 0.0381
        self.maxk2 = (total_mass * g) / 0.01905
        self.accel = None
        self.accelMax = 0.0
        self.accelLim = 0.2
        self.SSE = 0.0

class CarView:
    def __init__(self, args):
        self.input_widgets, self.display_widgets = args
        self.le_m1, self.le_v, self.le_k1, self.le_c1, self.le_m2, self.le_k2, self.le_ang, \
         self.le_tmax, self.chk_IncludeAccel = self.input_widgets

        self.gv_Schematic, self.chk_LogX, self.chk_LogY, self.chk_LogAccel, \
        self.chk_ShowAccel, self.lbl_MaxMinInfo, self.layout_horizontal_main = self.display_widgets

        self.figure = Figure(tight_layout=True, frameon=True, facecolor='none')
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.layout_horizontal_main.addWidget(self.canvas)

        self.ax = self.figure.add_subplot()
        if self.ax is not None:
            self.ax1 = self.ax.twinx()

        self.buildScene()

    def updateView(self, model=None):
        self.le_m1.setText("{:0.2f}".format(model.m1))
        self.le_k1.setText("{:0.2f}".format(model.k1))
        self.le_c1.setText("{:0.2f}".format(model.c1))
        self.le_m2.setText("{:0.2f}".format(model.m2))
        self.le_k2.setText("{:0.2f}".format(model.k2))
        self.le_ang.setText("{:0.2f}".format(model.yangdeg))
        self.le_tmax.setText("{:0.2f}".format(model.tmax))
        stTmp="k1_min = {:0.2f}, k1_max = {:0.2f}\nk2_min = {:0.2f}, k2_max = {:0.2f}\n".format(model.mink1, model.maxk1, model.mink2, model.maxk2)
        stTmp+="SSE = {:0.2f}".format(model.SSE)
        self.lbl_MaxMinInfo.setText(stTmp)
        self.doPlot(model)

    def buildScene(self):
        print("Building scene")
        self.scene = qtw.QGraphicsScene()
        self.scene.setObjectName("MyScene")
        self.scene.setSceneRect(-200, -200, 400, 400)

        self.gv_Schematic.setScene(self.scene)
        self.setupPensAndBrushes()
        print("Adding wheel")
        self.Wheel = Wheel(0, 50, 50, pen=self.penWheel, wheelBrush=self.brushWheel, massBrush=self.brushMass, name="Wheel", mass=20)
        self.Wheel.addToScene(self.scene)
        print("Adding car body")
        self.CarBody = MassBlock(0, -70, 100, 30, pen=self.penWheel, brush=self.brushMass, name="Car Body", mass=450)
        self.scene.addItem(self.CarBody)

        print("Adding spring")
        spring = SpringItem(0, -40, 0, 50, pen=self.penWheel)
        self.scene.addItem(spring)
        print("Adding dashpot")
        dashpot = DashpotItem(20, -40, 20, 50, pen=self.penWheel)
        self.scene.addItem(dashpot)
        print("Adding tire spring")
        tire_spring = SpringItem(0, 100, 0, 150, pen=self.penWheel)
        self.scene.addItem(tire_spring)
        print("Adding ground")
        ground = qtw.QGraphicsLineItem(-150, 150, 150, 150)
        ground.setPen(self.penWheel)
        self.scene.addItem(ground)
        print("Scene built successfully")

    def setupPensAndBrushes(self):
        self.penWheel = qtg.QPen(qtg.QColor("orange"))
        self.penWheel.setWidth(1)
        self.brushWheel = qtg.QBrush(qtg.QColor.fromHsv(35,255,255, 64))
        self.brushMass = qtg.QBrush(qtg.QColor(200,200,200, 128))

    def doPlot(self, model=None):
        if model.results is None:
            return
        ax = self.ax
        ax1 = self.ax1
        ax.clear()
        ax1.clear()
        t = model.t
        ycar = model.results[:, 0]
        ywheel = model.results[:, 2]
        accel = model.accel

        # Plot road profile
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

        ax.set_ylabel("Vertical Position (m)", fontsize='large')
        ax.set_xlabel("time (s)", fontsize='large')
        ax.legend(loc='upper left')

        ax.axvline(x=model.tramp)
        ax.axhline(y=model.ymag)

        ax.tick_params(axis='both', which='both', direction='in', top=True, labelsize='large')
        ax1.tick_params(axis='both', which='both', direction='in', right=True, labelsize='large')

        self.canvas.draw()

class CarController:
    def __init__(self, args):
        self.input_widgets, self.display_widgets = args
        self.le_m1, self.le_v, self.le_k1, self.le_c1, self.le_m2, self.le_k2, self.le_ang, \
         self.le_tmax, self.chk_IncludeAccel = self.input_widgets

        self.gv_Schematic, self.chk_LogX, self.chk_LogY, self.chk_LogAccel, \
        self.chk_ShowAccel, self.lbl_MaxMinInfo, self.layout_horizontal_main = self.display_widgets

        self.model = CarModel()
        self.view = CarView(args)

        self.chk_IncludeAccel = qtw.QCheckBox()

    def ode_system(self, X, t):
        if t < self.model.tramp:
            y = self.model.ymag * (t / self.model.tramp)
        else:
            y = self.model.ymag

        x1 = X[0]
        x1dot = X[1]
        x2 = X[2]
        x2dot = X[3]

        m1 = self.model.m1
        m2 = self.model.m2
        c1 = self.model.c1
        k1 = self.model.k1
        k2 = self.model.k2

        x1ddot = (-k1 * (x1 - x2) - c1 * (x1dot - x2dot)) / m1
        x2ddot = (k1 * (x1 - x2) + c1 * (x1dot - x2dot) - k2 * (x2 - y)) / m2

        return [x1dot, x1ddot, x2dot, x2ddot]

    def calculate(self, doCalc=True):
        self.model.m1 = float(self.le_m1.text())
        self.model.v = float(self.le_v.text())
        self.model.k1 = float(self.le_k1.text())
        self.model.c1 = float(self.le_c1.text())
        self.model.m2 = float(self.le_m2.text())
        self.model.k2 = float(self.le_k2.text())

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

        g = 9.81
        self.model.mink1 = (self.model.m1 * g) / 0.1524
        self.model.maxk1 = (self.model.m1 * g) / 0.0762
        total_mass = self.model.m1 + self.model.m2
        self.model.mink2 = (total_mass * g) / 0.0381
        self.model.maxk2 = (total_mass * g) / 0.01905

        ymag = 6.0 / (12.0 * 3.3)
        if ymag is not None:
            self.model.ymag = ymag

        self.model.yangdeg = float(self.le_ang.text())
        self.model.tmax = float(self.le_tmax.text())

        if doCalc:
            self.doCalc()

        self.SSE((self.model.k1, self.model.c1, self.model.k2), optimizing=False)
        self.view.updateView(self.model)

    def doCalc(self, doPlot=True, doAccel=True):
        v = 1000 * self.model.v / 3600
        self.model.angrad = self.model.yangdeg * math.pi / 180.0

        if v <= 0:
            v = 1.0
        if abs(math.sin(self.model.angrad)) <= 1e-6:
            self.model.angrad = math.radians(5)

        self.model.tramp = self.model.ymag / (math.sin(self.model.angrad) * v)

        self.model.t = np.linspace(0, self.model.tmax, 2000)
        ic = [0, 0, 0, 0]
        self.model.results = odeint(self.ode_system, ic, self.model.t)

        if doAccel:
            self.calcAccel()
        if doPlot:
            self.doPlot()

    def calcAccel(self):
        N = len(self.model.t)
        self.model.accel = np.zeros(shape=N)
        vel = self.model.results[:, 1]
        for i in range(N):
            if i == N-1:
                h = self.model.t[i] - self.model.t[i-1]
                self.model.accel[i] = (vel[i] - vel[i-1]) / (9.81 * h) + 1e-6
            else:
                h = self.model.t[i + 1] - self.model.t[i]
                self.model.accel[i] = (vel[i + 1] - vel[i]) / (9.81 * h) + 1e-6
        self.model.accelMax = self.model.accel.max()
        return True

    def OptimizeSuspension(self):
        self.calculate(doCalc=False)
        x0 = np.array([(self.model.mink1 + self.model.maxk1) / 2,  # Middle of k1 range
                      max(10, self.model.c1),
                      (self.model.mink2 + self.model.maxk2) / 2])  # Middle of k2 range
        bounds = [(self.model.mink1, self.model.maxk1), (10, None), (self.model.mink2, self.model.maxk2)]
        answer = minimize(self.SSE, x0, method='L-BFGS-B', bounds=bounds)
        self.model.k1, self.model.c1, self.model.k2 = answer.x
        self.doCalc()
        self.view.updateView(self.model)

    def SSE(self, vals, optimizing=True):
        k1, c1, k2 = vals
        self.model.k1 = k1
        self.model.c1 = c1
        self.model.k2 = k2
        self.doCalc(doPlot=False)

        SSE = 0
        for i in range(len(self.model.results[:, 0])):
            t = self.model.t[i]
            y = self.model.results[:, 0][i]
            if t < self.model.tramp:
                ytarget = self.model.ymag * (t / self.model.tramp)
            else:
                ytarget = self.model.ymag
            SSE += (y - ytarget) ** 2

        if optimizing:
            if self.model.accelMax > self.model.accelLim and self.chk_IncludeAccel.isChecked():
                SSE += 1e10 * (self.model.accelMax - self.model.accelLim) ** 2

        self.model.SSE = SSE
        return SSE

    def doPlot(self):
        self.view.doPlot(self.model)

#endregion
#endregion
