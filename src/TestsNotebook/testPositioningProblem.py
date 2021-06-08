import autograd.numpy as np

from pymanopt.manifolds import Product
from pymanopt.manifolds import Rotations
from pymanopt.manifolds import Euclidean

# Dimension of the sphere
solutionSpaceDimension = 9

# Instantiate the SE(3) manifold
euclideanSpace = Euclidean(3)
SO3 = Rotations(3)
R33 = Euclidean(3, 3)
specialEuclideanGroup = Product((SO3, euclideanSpace))

# Instantiate the solution space
solutionSpace = Product((SO3, euclideanSpace, euclideanSpace))

# Dimension of the parameter space
parameterSpaceDimension = 2

# Instantiate the parameter space
parameterSpace = Euclidean(2)

# Instantiate the global manifold
n = 3
Rn = Euclidean(n)
productAmbient = Product((R33, euclideanSpace, Rn, parameterSpace))
product = Product((SO3, euclideanSpace, Rn, parameterSpace))

def rho_z(theta):
    rotation = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

    return np.block([[rotation, np.zeros((3, 1), dtype='float64')], [np.zeros((1, 3), dtype='float64'), 1.]])


def tau_x(x):
    translation = np.reshape(np.array([x, 0., 0.]), (3, 1))

    return np.block([[np.eye(3), translation], [np.zeros((1, 3), dtype='float64'), 1.]])


def tau_y(y):
    translation = np.reshape(np.array([0., y, 0.]), (3, 1))

    return np.block([[np.eye(3), translation], [np.zeros((1, 3), dtype='float64'), 1.]])


def C(t, a):
    return tau_y(a * t ** 2) @ tau_x(t) @ rho_z(np.arctan(2 * a * t))

startingTheta = (45.) * np.pi / 180.
startingCoefficient = 1.

finalTheta =  0.24235328104536874 #1.1#(70) * np.pi / 180.
finalCoefficient = 0.16422448979591836

p1 = tau_x(-0.5)
p2 = tau_x(0.5)

p1Inv = np.linalg.inv(p1)
p2Inv = np.linalg.inv(p2)

initialSolution = [np.eye(3),
                   np.array([0., -0.25, 0.]),
                   np.array([-0.5, 0.5, 45. * np.pi / 180.])]

#initialSolution = solutionSpace.exp(_initialSolution, 1e-7 * solutionSpace.randvec(_initialSolution))

#initialSolution = [np.array([[ 0.9332623559555006, -0.35919545508871464,  0.        ],
#                     [ 0.35919545508871464,  0.9332623559555006,  0.        ],
#                     [ 0.        ,  0.        ,  1.        ]]),
#                     np.array([ 0.2528757691106841, -0.38914481919722627,  0.        ]),
#                     np.array([-0.5628515635950829,  0.3704107923604174,  1.3445029385189626])]

initialParameter = np.array([startingTheta, startingCoefficient])
#initialParameter = _initialParameter + 1e-7 * parameterSpace.randvec(_initialParameter)
targetParameter = np.array([finalTheta, finalCoefficient])

def matrixRepresentationOfSE3Element(rotation, translation):
    return np.block([[rotation, np.reshape(translation, (3, 1))], [np.zeros((1, 3), dtype='float64'), 1]])

def matrixRepresentationOfse3Element(element):
    return np.block([[element[0], np.reshape(element[1], (3, 1))], [np.zeros((1, 4), dtype='float64')]])

def tupleRepresentationOfSE3Element(element):
    return element[:3, :3], element[:3, 3]


# def cost(S):
#     C1 = C(S[2][0], S[3][1])
#     C2 = C(S[2][1], S[3][1])
#     rho_z1 = rho_z(0.5 * np.pi - S[3][0])
#     rho_z2 = rho_z(- S[2][2])
#
#     A1 = S[0] @ C1[:3, :3] @ rho_z1[:3, :3] @ p1Inv[:3, :3]
#     A2 = S[0] @ C2[:3, :3] @ rho_z2[:3, :3] @ p2Inv[:3, :3]
#
#     tau1 = S[1] + S[0] @ C1[:3, 3] + S[0] @ C1[:3, :3] @ rho_z1[:3, :3] @ p1Inv[:3, 3]
#     tau2 = S[1] + S[0] @ C2[:3, 3] + S[0] @ C2[:3, :3] @ rho_z2[:3, :3] @ p2Inv[:3, 3]
#
#     return - 2 * np.trace(A1 + A2 - 2 * np.eye(3)) + np.inner(tau1, tau1) + np.inner(tau2, tau2)
#
# cost(list(initialSolution) + [initialParameter])

currentPoint = list(initialSolution) + [initialParameter]

def cost(S):
    I = np.eye(4)
    phi = matrixRepresentationOfSE3Element(S[0], S[1])
    u = phi @ C(S[2][0], S[3][1]) @ rho_z(0.5 * np.pi - S[3][0]) @ p1Inv - I
    v = phi @ C(S[2][1], S[3][1]) @ rho_z(- S[2][2]) @ p2Inv - I

    return np.trace(u.T @ u) + np.trace(v.T @ v)

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

from Continuation.PositioningProblem.StepSizeAdaptiveContinuationStraightLine import StepSizeAdaptiveContinuation
from Continuation.PositioningProblem.PathAdaptiveContinuationApproximateLength import PathAdaptiveContinuationApproximateLength
from Continuation.PositioningProblem.PathAdaptiveContinuationSecondOrderApproximation import PathAdaptiveContinuationSecondOrderApproximation

# Instantiate continuation object

continuation = StepSizeAdaptiveContinuation(problem,
                                            initialSolution,
                                            initialParameter,
                                            targetParameter)

continuation3 = PathAdaptiveContinuationApproximateLength(problem,
                                                          initialSolution,
                                                          initialParameter,
                                                          targetParameter)

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