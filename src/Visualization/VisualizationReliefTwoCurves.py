import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Helpers.Parameterizations.BasicSE3Transformations import rho_y
from SE3Parameterizations.Parameterizations.Helix import Helix
from SE3Parameterizations.Parameterizations.Torus import Torus
from Visualization.Parameterizations.Surfaces.TorusVisualization import TorusVisualization
from Visualization.Parameterizations.Surfaces.CylinderVisualization import CylinderVisualization
from Visualization.Parameterizations.Curves.HelixVisualization import HelixVisualization


def plotSE3Element(ax, phi, color='r', style='line', frame = None):
    o = phi @ np.array([0, 0, 0, 1])
    x = phi @ np.array([1, 0, 0, 1])
    y = phi @ np.array([0, 1, 0, 1])
    z = phi @ np.array([0, 0, 1, 1])

    if style == "arrow":
        ax.quiver(
            o[0],
            o[1],
            o[2],
            x[0] - o[0],
            x[1] - o[1],
            x[2] - o[2],
            arrow_length_ratio=0.2,
            linewidth=None,
            facecolor=color,
            edgecolor=color,
        )
        ax.quiver(
            o[0],
            o[1],
            o[2],
            y[0] - o[0],
            y[1] - o[1],
            y[2] - o[2],
            arrow_length_ratio=0.2,
            linewidth=None,
            facecolor=color,
            edgecolor=color,
        )
        ax.quiver(
            o[0],
            o[1],
            o[2],
            z[0] - o[0],
            z[1] - o[1],
            z[2] - o[2],
            arrow_length_ratio=0.2,
            linewidth=None,
            facecolor=color,
            edgecolor=color,
        )

        # plot some points
        #  invisible point at the end of each arrow to allow auto-scaling to work
        ax.scatter(
            xs=[o[0], x[0], y[0], z[0]],
            ys=[o[1], x[1], y[1], z[1]],
            zs=[o[2], x[2], y[2], z[2]],
            s=[0, 0, 0, 0],
        )
    elif style == "line":
        width = None
        ax.plot(
            [o[0], x[0]], [o[1], x[1]], [o[2], x[2]], color=color[0], linewidth=width
        )
        ax.plot(
            [o[0], y[0]], [o[1], y[1]], [o[2], y[2]], color=color[0], linewidth=width
        )
        ax.plot(
            [o[0], z[0]], [o[1], z[1]], [o[2], z[2]], color=color[0], linewidth=width
        )

    d2 = 1.15
    x = (x - o) * d2 + o
    y = (y - o) * d2 + o
    z = (z - o) * d2 + o
    labels = ['X', 'Y', 'Z']
    textcolor = 'b'

    ax.text(
        x[0],
        x[1],
        x[2],
        "$%c_{%s}$" % (labels[0], frame),
        color=textcolor,
        horizontalalignment="center",
        verticalalignment="center",
    )
    ax.text(
        y[0],
        y[1],
        y[2],
        "$%c_{%s}$" % (labels[1], frame),
        color=textcolor,
        horizontalalignment="center",
        verticalalignment="center",
    )
    ax.text(
        z[0],
        z[1],
        z[2],
        "$%c_{%s}$" % (labels[2], frame),
        color=textcolor,
        horizontalalignment="center",
        verticalalignment="center",
    )

class SubplotAnimationReliefTwoCurves(animation.TimedAnimation):
    def __init__(self, results, parameterSpaceMetrics, perturbationMagnitudes, targetParameter, positioningProblem):
        self.solutions = [S[:len(S)] for (a, S) in results]
        self.parameters = [S[-1] for (a, S) in results]

        self.perturbationMagnitudes = perturbationMagnitudes

        self.parameterSpaceMetrics = parameterSpaceMetrics

        fig = plt.figure(figsize=(12, 10), dpi=100)

        self.ax1 = plt.axes(projection="3d")

        self.ax1.set_xlabel('x')
        self.ax1.set_ylabel('y')
        self.ax1.set_zlabel('z')

        self.ax1.set_xlim(-20., 20.)
        self.ax1.set_ylim(-20., 20.)
        self.ax1.set_zlim(-20., 20.)

        self.targetParameter = targetParameter

        self.positioningProblem = positioningProblem

        self.draw_results(0)

        animation.TimedAnimation.__init__(self, fig, interval=2000, blit=False, repeat=True)

    def _draw_frame(self, framedata):
        self.draw_results(framedata)

        self.ax1.clear()

        self.ax1.set_xlabel('x')
        self.ax1.set_ylabel('y')
        self.ax1.set_zlabel('z')

        self.ax1.set_xlim(-20., 20.)
        self.ax1.set_ylim(-20., 20.)
        self.ax1.set_zlim(-20., 20.)

        #plot cylinder
        self.cylinderAx = self.ax1.plot_surface(self.Cylinder.X, self.Cylinder.Y, self.Cylinder.Z, antialiased=True,
                              color=self.Cylinder.color, rstride=self.Cylinder.RStride,
                             cstride=self.Cylinder.CStride,
                             alpha=self.Cylinder.opacity)

        #plot wheel
        self.wheelAx = self.ax1.plot_surface(self.Wheel.X, self.Wheel.Y, self.Wheel.Z, antialiased=True,
                              color=self.Wheel.color, rstride=self.Wheel.RStride,
                             cstride=self.Wheel.CStride,
                             alpha=self.Wheel.opacity)
        # plot helices
        self.rakeAx = self.ax1.plot3D(self.RakeHelix.X, self.RakeHelix.Y, self.RakeHelix.Z, color=self.RakeHelix.color)
        self.backAx = self.ax1.plot3D(self.BackHelix.X, self.BackHelix.Y, self.BackHelix.Z, color=self.BackHelix.color)

        print("Delta parameter = " + str(self.targetParameter - self.parameters[framedata]))

    def new_frame_seq(self):
        return iter(range(len(self.parameters)))

    def _init_draw(self):
        pass

    def draw_results(self, framedata):
        solution = self.solutions[framedata]
        parameter = self.parameters[framedata]
        Rt, rt, helixAngle1, helixRadius1, helixAngle2, helixRadius2, offsetAngle, trajectoryParameter = parameter

        phi = np.block([[solution[0], np.reshape(solution[1], (3, 1))], [np.zeros((1, 3)), 1.]])
        phiTilde = np.block(
            [[solution[0] @ rho_y(-0.5 * np.pi - helixAngle1)[:3, :3], np.reshape(solution[1], (3, 1))], [np.zeros((1, 3)), 1.]])

        self.Cylinder = CylinderVisualization(self.ax1, self.positioningProblem.helixLength, helixRadius1, np.array([0., 0.]), 'grey', opacity=.5)

        helix = Helix(helixRadius1, helixAngle1, self.positioningProblem.helixLength, offsetAngle=0.0)
        secondHelix = Helix(helixRadius2, helixAngle2, self.positioningProblem.helixLength, offsetAngle=offsetAngle)
        torus = Torus(rt, Rt, self.positioningProblem.offsetWheel)

        self.RakeHelix = HelixVisualization(self.ax1, helix, 'red')
        self.BackHelix = HelixVisualization(self.ax1, secondHelix, 'blue')

        self.Wheel = TorusVisualization(self.ax1, torus, SE3Transformation=phiTilde, color='orange')

        # plotSE3Element(ax, phi @ torus.Evaluate(0, 0) @ rho_z(-0.5 * np.pi - helixAngle), color='g', frame='W')
        # plotSE3Element(ax, helix.Evaluate(0.1) @ rho_x(np.arctan(np.tan(X0[2][0]) / np.cos(helixAngle))), color='k',
        #                frame='H')

        # self.ax1.plot(xRot, yRot, 'r-')

    # def draw_pose(self, phi):
    #     o = phi @ np.array([0, 0, 0, 1])
    #     x = phi @ np.array([1, 0, 0, 1])
    #     y = phi @ np.array([0, 1, 0, 1])
    #     z = phi @ np.array([0, 0, 1, 1])
    #
    #     width = None
    #     self.ax1
    #         [o[0], x[0]], [o[1], x[1]], [o[2], x[2]], color=color[0], linewidth=width
    #     )
    #     ax.plot(
    #         [o[0], y[0]], [o[1], y[1]], [o[2], y[2]], color=color[0], linewidth=width
    #     )
    #     ax.plot(
    #         [o[0], z[0]], [o[1], z[1]], [o[2], z[2]], color=color[0], linewidth=width
    #     )

    def show(self):
        plt.show()
