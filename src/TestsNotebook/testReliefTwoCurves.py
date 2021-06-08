import autograd.numpy as np

from pymanopt.manifolds import Product
from pymanopt.manifolds import Rotations
from pymanopt.manifolds import Euclidean

# Dimension of the sphere
solutionSpaceDimension = 12

# Instantiate the SE(3) manifold
R3 = Euclidean(3)
R6 = Euclidean(6)
SO3 = Rotations(3)
R33 = Euclidean(3, 3)
specialEuclideanGroup = Product((SO3, R3))

# Instantiate the solution space
solutionSpace = Product((SO3, R3, R6))

# Dimension of the parameter space
parameterSpaceDimension = 2

# Instantiate the parameter space
parameterSpace = Euclidean(2)

# Instantiate the global manifold
productAmbient = Product((R33, R3, R6, parameterSpace))
product = Product((SO3, R3, R6, parameterSpace))

from SE3Parameterizations.Helpers.SE3Representation import matrixRepresentationOfSE3Element
from SE3Parameterizations.Helpers.BasicSE3Transformations import rho_x, rho_z
from SE3Parameterizations.Parameterizations.Helix import Helix
from SE3Parameterizations.Parameterizations.Torus import Torus

# Define constants
rt, Rt = 5, 30
offsetWheel = np.array([0., 40., 0.])
wheelProfileParameter = 0.
# exceptionally, fix the trajectory parameter
trajectoryParameter = 0.

helixLength = 10
offsetAngle = 30. * np.pi / 180.

def cost(S):
    phi = matrixRepresentationOfSE3Element(S[0], S[1])
    t, mu = S[2], S[3]

    grindingMark = -0.5 * np.pi - mu[1]

    W = Torus(rt, Rt, offsetWheel)
    C1 = Helix(mu[0], mu[1], helixLength)
    C2 = Helix(mu[0], mu[1], helixLength, offsetAngle)

    # u = phi @ Tore(0., wheelProfileParameter, rt, Rt, offsetWheel) @ rho_z(grindingMark) \
    #     - Helix(trajectoryParameter, mu[0], mu[1], helixLength) \
    #     @ rho_x(np.arctan(np.tan(t[0]) / np.cos(mu[1])))
    #
    # v = phi @ Tore(t[1], t[2], rt, Rt, offsetWheel) @ rho_z(t[3]) \
    #     - Helix(t[4], mu[0], mu[1], helixLength, offsetAngle) \
    #     @ rho_x(np.arctan(np.tan(t[5]) / np.cos(mu[1])))

    u = phi @ W.Evaluate(0., wheelProfileParameter) @ rho_z(grindingMark) \
         - C1.Evaluate(trajectoryParameter) \
         @ rho_x(np.arctan(np.tan(t[0]) / np.cos(mu[1])))

    v = phi @ W.Evaluate(t[1], t[2]) @ rho_z(t[3]) \
         - C2.Evaluate(t[4]) \
         @ rho_x(np.arctan(np.tan(t[5]) / np.cos(mu[1])))

    return np.trace(u.T @ u) + np.trace(v.T @ v)

# Define initial solution and initial parameter
initialParameter = np.array([2., 30. * np.pi / 180.])
secondWheelRevolutionAngle = np.arccos(1 - (initialParameter[0] / (Rt + rt)) ** 2 * (1 - np.cos(offsetAngle)))


def FindInitialCondition():
    grindingMark = -0.5 * np.pi - initialParameter[1]
    reliefAngle = 0. * np.pi / 180.

    W = Torus(rt, Rt, offsetWheel)
    C1 = Helix(initialParameter[0], initialParameter[1], helixLength)

    initialPhi = C1.Evaluate(trajectoryParameter) \
                 @ rho_x(np.arctan(np.tan(reliefAngle) / np.cos(initialParameter[1])))\
                 @ rho_z(- grindingMark) \
                 @ np.linalg.inv(W.Evaluate(0, 0))

    # initialPhi = Helix(trajectoryParameter, initialParameter[0], initialParameter[1], helixLength) \
    #              @ rho_x(np.arctan(np.tan(reliefAngle) / np.cos(initialParameter[1])))\
    #              @ rho_z(- grindingMark) \
    #              @ np.linalg.inv(Tore(0, 0, rt, Rt, offsetWheel))

    initialScalars = np.array([0., 0., 0., grindingMark, trajectoryParameter, 0.])

    initialGuess = [initialPhi[:3, :3],
                       initialPhi[:3, 3],
                       initialScalars]

    def costForFixedParameter(S):
        A = list(S)
        A.append(initialParameter)

        return cost(A)

    def Skew(A):
        return 0.5 * (A - A.T)

    def diffProj(x, z, v):
        return solutionSpace.proj(x, [x[0] @ Skew(z[0].T @ v[0]) + z[0] @ Skew(x[0].T @ v[0]), np.zeros(x[1].shape),
                                np.zeros(x[2].shape)])

    def hessForFixedParameter(x, z):
        egrad = correctorProblem.egrad(x)
        ehess = correctorProblem.ehess(x, [SO3.tangent2ambient(x[0], z[0]), z[1], z[2]])
        return solutionSpace.proj(x, ehess) + diffProj(x, [SO3.tangent2ambient(x[0], z[0]), z[1], z[2]], egrad)

    correctorProblem = Problem(solutionSpace, costForFixedParameter, hess=hessForFixedParameter)

    #corrector = SolverRBFGS(correctorProblem, False)
    from pymanopt.solvers.trust_regions import TrustRegions

    solver = TrustRegions()
    return solver.solve(correctorProblem, x = initialGuess)
    #return corrector.SearchSolution(initialGuess, np.eye(solutionSpaceDimension))


#X0 = FindInitialCondition()

def Skew(A):
    return 0.5 * (A - A.T)

def SkewMat(w):
    return np.array([[0., -w[2], w[1]], [w[2], 0., -w[0]], [-w[1], w[0], 0.]], dtype=float)

def diffProj(x, z, v):
    return product.proj(x, [x[0] @ Skew(z[0].T @ v[0]) + z[0] @ Skew(x[0].T @ v[0]), np.zeros(x[1].shape), np.zeros(x[2].shape), np.zeros(x[3].shape)])

def hess(x, z):
    egrad = problem.egrad(x)
    ehess = problem.ehess(x, [SO3.tangent2ambient(x[0], z[0]), z[1], z[2], z[3]])
    return product.proj(x, ehess) + diffProj(x, [SO3.tangent2ambient(x[0], z[0]), z[1], z[2], z[3]], egrad)


from pymanopt.core.problem import Problem

# Instantiate the problem
problem = Problem(product, cost=cost, hess=hess)

from Continuation.PositioningProblem.PathAdaptiveContinuationSecondOrderApproximation import PathAdaptiveContinuationSecondOrderApproximation

# Instantiate continuation object

# continuation = StepSizeAdaptiveContinuation(problem,
#                                             initialSolution,
#                                             initialParameter,
#                                             targetParameter)
#
# continuation3 = PathAdaptiveContinuationApproximateLength(problem,
#                                                           initialSolution,
#                                                           initialParameter,
#                                                           targetParameter)

initialSolution = [np.array([[ 0.14700258,  0.9854882 ,  0.08487199],
                            [-0.98911615,  0.14700258,  0.0062838 ],
                            [-0.00628379, -0.08487199,  0.99637205]]),
                   np.array([-2.92744102, -0.73501289,  0.42435993]),
                   np.array([ 0.14807544, -0.00251379,  0.02705792, -2.09305339, -0.08459128,
                            -0.14802329])]

targetParameter = np.array([3.0, 30 * np.pi / 180.])

continuation2 = PathAdaptiveContinuationSecondOrderApproximation(problem,
                                                                initialSolution,
                                                                initialParameter,
                                                                targetParameter)

#results, parameterSpaceMetrics, perturbationMagnitudes, iterations, solved = continuation.Traverse()
#results3, parameterSpaceMetrics3, perturbationMagnitudes3, iterations3, solved3 = continuation3.Traverse()
results2, parameterSpaceMetrics2, perturbationMagnitudes2, iterations2, solved2 = continuation2.Traverse()



##########################################################################
##########################################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse

plt.ion()

class SubplotAnimation(animation.TimedAnimation):
    def __init__(self, results, parameterSpaceMetrics, perturbationMagnitudes, targetParameter):
        self.solutions = [S[:len(S)] for (a, S) in results]
        self.parameters = [S[-1] for (a, S) in results]

        self.perturbationMagnitudes = perturbationMagnitudes

        self.parameterSpaceMetrics = parameterSpaceMetrics

        self.xParameters = [x for (x, y) in self.parameters]
        self.yParameters = [y for (x, y) in self.parameters]

        fig = plt.figure(figsize=(10, 8), dpi=100)
        self.ax1 = fig.add_subplot(1, 2, 2)
        self.ax2 = fig.add_subplot(1, 2, 1)

        self.ax1.set_xlabel('x')
        self.ax1.set_ylabel('y')
        self.ax1.set_xlim(-2., 2.)
        self.ax1.set_ylim(-2., 2.)

        self.ax2.set_xlabel('x')
        self.ax2.set_ylabel('y')
        self.ax2.set_xlim(-7, 7)
        self.ax2.set_ylim(-7, 7)

        self.ax1.grid()
        self.ax2.grid()

        self.ax2.plot(targetParameter[0], targetParameter[1], 'ro')

        self.ax2.plot(np.linspace(-10, 0.5 * np.pi, 100), np.zeros(100), 'g-')
        self.ax2.plot(0.5 * np.pi * np.ones(100), np.linspace(-10, 0, 100), 'g-')

        self.linePoints, = self.ax1.plot([], [], 'ko', lw=2)
        self.lineCurve, = self.ax1.plot([], [], 'r-', lw=2)
        self.line, = self.ax2.plot([], [], 'o', lw=2)
        self.time_template = 'iteration = %.1i'
        self.time_text = self.ax2.text(0.05, 0.9, '', transform=self.ax2.transAxes)

        self.targetParameter = targetParameter

        self.draw_curve(0)

        if self.perturbationMagnitudes[0] is None:
            pass
        else:
            self.ellipse = self.determineEllipse(0)
            self.ax2.add_patch(self.ellipse)

        animation.TimedAnimation.__init__(self, fig, interval=1800, blit=False, repeat=True)

    def _draw_frame(self, framedata):

        # draw ellipse
        if framedata < len(self.xParameters) - 1:
            if self.perturbationMagnitudes[framedata] is None:
                pass
            else:
                self.ellipse = self.determineEllipse(framedata)
                self.ax2.patches = []
                self.ax2.add_patch(self.ellipse)

        else:
            self.ax2.patches = []

        # draw perturbation vector
        if framedata + 1 < len(self.xParameters):
            dx = self.xParameters[framedata + 1] - self.xParameters[framedata]
            dy = self.yParameters[framedata + 1] - self.yParameters[framedata]
            self.ax2.arrow(self.xParameters[framedata], self.yParameters[framedata], dx, dy)

        self.draw_curve(framedata)

        # draw new parameter value in parameter space
        thisx = self.xParameters[framedata]
        thisy = self.yParameters[framedata]

        self.line.set_data(thisx, thisy)

        # draw number of iteration
        self.time_text.set_text(self.time_template % framedata)

        print("Delta parameter = " + str(self.targetParameter - self.parameters[framedata]))

    def new_frame_seq(self):
        return iter(range(len(self.xParameters)))

    def _init_draw(self):
        self.line.set_data([], [])
        self.lineCurve.set_data([], [])
        self.linePoints.set_data([], [])
        self.time_text.set_text('')

    def draw_curve(self, framedata):
        a = self.yParameters[framedata]

        solution = self.solutions[framedata]

        rotationMatrix = solution[0]
        translationVector = solution[1]
        t = solution[2]

        x = np.linspace(-3, 3, 100)
        y = a * x ** 2

        xRotAndTrans = rotationMatrix[0, 0] * x + rotationMatrix[0, 1] * y + translationVector[0]
        yRotAndTrans = rotationMatrix[1, 0] * x + rotationMatrix[1, 1] * y + translationVector[1]

        self.lineCurve.set_data(xRotAndTrans, yRotAndTrans)

        self.linePoints.set_data(
            [rotationMatrix[0, 0] * t[0] + rotationMatrix[0, 1] * a * t[0] ** 2 + translationVector[0],
             rotationMatrix[0, 0] * t[1] + rotationMatrix[0, 1] * a * t[1] ** 2 + translationVector[0]],
            [rotationMatrix[1, 0] * t[0] + rotationMatrix[1, 1] * a * t[0] ** 2 + translationVector[1],
             rotationMatrix[1, 0] * t[1] + rotationMatrix[1, 1] * a * t[1] ** 2 + translationVector[1]])
        # self.ax1.plot(xRot, yRot, 'r-')

    def determineEllipse(self, framedata):

        metricMatrix = self.parameterSpaceMetrics[framedata]

        deltaParameter = self.targetParameter - self.parameters[framedata]

        eigvals, eigvecs = np.linalg.eig(metricMatrix)

        smallestIndex = np.argmin(eigvals)
        largestIndex = np.argmax(eigvals)

        print(eigvals)

        slope = eigvecs[1, smallestIndex] / eigvecs[0, smallestIndex]
        angle = 180.0 * np.arctan(slope) / np.pi

        if np.inner(deltaParameter, metricMatrix @ deltaParameter) < self.perturbationMagnitudes[framedata] ** 2:
            _perturbationMagnitude = np.sqrt(np.inner(deltaParameter, metricMatrix @ deltaParameter))
        else:
            _perturbationMagnitude = self.perturbationMagnitudes[framedata]

        return Ellipse(self.parameters[framedata], width=2 * _perturbationMagnitude / np.sqrt(eigvals[smallestIndex]),
                       height=2 * _perturbationMagnitude / np.sqrt(eigvals[largestIndex]), angle=angle, fill=False)

    def show(self):
        plt.show()


ani = SubplotAnimation(results2,
                       parameterSpaceMetrics2,
                       perturbationMagnitudes2,
                       targetParameter)

ani.show()