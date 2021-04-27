import sys
sys.path.append('../')

import autograd.numpy as np

from pymanopt.manifolds import Sphere

# Dimension of the sphere
solutionSpaceDimension = 3

# Instantiate the unit sphere manifold
unitSphere = Sphere(solutionSpaceDimension)

from pymanopt.manifolds import Euclidean

# Dimension of the parameter space
parameterSpaceDimension = 2

# Instantiate the parameter space
parameterSpace = Euclidean(parameterSpaceDimension)

from pymanopt.manifolds import Product

productManifold = Product([unitSphere, parameterSpace])

def A(mu):
    sigma = 100
    lambda0 = 1 + np.exp((mu[0] - mu[1]) / sigma)
    lambda1 = 2 + np.exp((3 * mu[0] + mu[1]) / sigma)
    lambda2 = 3 + np.exp((mu[0] + mu[1]) / sigma)

    return np.array([[lambda0, -1, 0], [-1, lambda1, -1], [0, -1, lambda2]])


def DA(mu, v):
    sigma = 100
    Dlambda0 = (v[0] - v[1]) / sigma * np.exp((mu[0] - mu[1]) / sigma)
    Dlambda1 = (3 * v[0] + v[1]) / sigma * np.exp((3 * mu[0] + mu[1]) / sigma)
    Dlambda2 = (v[0] + v[1]) / sigma * np.exp((mu[0] + mu[1]) / sigma)

    return np.array([[Dlambda0, 0, 0], [0, Dlambda1, 0], [0, 0, Dlambda2]])

# Define derivatives of gradient wrt solution/parameter
def differentialSolutionAlongV(x, mu, xi):
    return A(mu) @ xi - 2 * (x.T @ A(mu) @ xi) * x - (x.T @ A(mu) @ x) * xi

def differentialParameterAlongV(x, mu, v):
    return (np.eye(solutionSpaceDimension) - np.outer(x, x)) @ DA(mu, v) @ x


from pymanopt.core.problem import Problem

def cost(S):
    return np.inner(S[0], A(S[1]) @ S[0])

problem = Problem(productManifold, cost=cost)

from Continuation.PathAdaptiveContinuation import PathAdaptiveMultiParameterContinuation

initialParameter = np.array([1, 2])
targetParameter = np.array([23., 35.])

B = A(initialParameter)
C = A(targetParameter)

wB, vB = np.linalg.eig(B)
wC, vC = np.linalg.eig(C)

initialSolution = vB[:, np.argmin(wB)]
finalSolution = vC[:, np.argmin(wC)]

parameterMagnitude = 0.5 * 1e-1

# Instantiate continuation object
continuation = PathAdaptiveMultiParameterContinuation(problem,
                                                      initialSolution,
                                                      initialParameter,
                                                      targetParameter,
                                                      differentialSolutionAlongV,
                                                      differentialParameterAlongV,
                                                      A,
                                                      parameterMagnitude)
results = continuation.Traverse()


##########################################################################################
##########################################################################################
################################## VISUALIZATION #########################################
##########################################################################################
##########################################################################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse

plt.ion()

class SubplotAnimation(animation.TimedAnimation):
    def __init__(self, results, parameterMagnitude):
        self.parameters = [p for (a, (x, p)) in results]
        self.solutions = [x for (a, (x, p)) in results]

        self.parameterMagnitude = parameterMagnitude
        self.data = [np.reshape(solution, (1, 3)) for solution in self.solutions]

        self.xParameters = [x for (x, y) in self.parameters]
        self.yParameters = [y for (x, y) in self.parameters]

        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 2, projection="3d")
        self.ax2 = fig.add_subplot(1, 2, 1)

        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('z')

        ax1.set_xlim(-1.2, 1.2)
        ax1.set_ylim(-1.2, 1.2)
        ax1.set_zlim(-1.2, 1.2)

        self.ax2.set_xlabel('x')
        self.ax2.set_ylabel('y')
        self.ax2.set_xlim(-40, 50)
        self.ax2.set_ylim(-40, 50)

        self.ax2.grid()

        # draw sphere
        u, v = np.mgrid[0: 2 * np.pi: 15j, 0: np.pi: 15j]
        x = np.cos(u) * np.sin(v)
        y = np.sin(u) * np.sin(v)
        z = np.cos(v)

        ax1.plot_wireframe(x, y, z, color="y")

        ax1.scatter(finalSolution[0], finalSolution[1], finalSolution[2])

        ax1.view_init(5, 225)

        self.scatters = [ax1.scatter(self.data[0][i, 0:1], self.data[0][i, 1:2], self.data[0][i, 2:], 'o-') for i in
                         range(self.data[0].shape[0])]

        self.line, = self.ax2.plot([], [], 'o', lw=2)
        self.time_template = 'iteration = %.1i'
        self.time_text = self.ax2.text(0.05, 0.9, '', transform=self.ax2.transAxes)

        self.ellipse = self.determineEllipse(0)
        self.ax2.add_patch(self.ellipse)

        animation.TimedAnimation.__init__(self, fig, interval=1800, blit=False, repeat=True)

    def _draw_frame(self, framedata):

        #draw ellipse
        self.ellipse = self.determineEllipse(framedata)

        self.ax2.patches = []
        self.ax2.add_patch(self.ellipse)

        #draw perturbation vector
        if framedata + 1 < len(self.xParameters):
            dx = self.xParameters[framedata + 1] - self.xParameters[framedata]
            dy = self.yParameters[framedata + 1] - self.yParameters[framedata]
            self.ax2.arrow(self.xParameters[framedata], self.yParameters[framedata], dx, dy)

        # draw solution point on sphere
        for i in range(self.data[0].shape[0]):
            self.scatters[i]._offsets3d = (
            self.data[framedata][i, 0:1], self.data[framedata][i, 1:2], self.data[framedata][i, 2:])

        # draw new parameter value in parameter space
        thisx = [self.xParameters[framedata], targetParameter[0]]
        thisy = [self.yParameters[framedata], targetParameter[1]]

        self.line.set_data(thisx, thisy)

        # draw number of iteration
        self.time_text.set_text(self.time_template % framedata)

        print("Delta parameter = " + str(targetParameter - self.parameters[framedata]))

    def new_frame_seq(self):
        return iter(range(len(self.xParameters)))

    def _init_draw(self):
        self.line.set_data([], [])
        self.time_text.set_text('')

    def reconstructMetricMatrix(self, framedata):
        currentSolution = self.solutions[framedata]
        currentParameter = self.parameters[framedata]

        differentialParameterMatrix = self.WriteMatrixInEuclideanBasisAtGivenPoint(differentialParameterAlongV,
                                                                                   currentSolution,
                                                                                   currentParameter,
                                                                                   len(currentParameter))

        differentialSolutionMatrix = self.WriteMatrixInEuclideanBasisAtGivenPoint(differentialSolutionAlongV,
                                                                                  currentSolution,
                                                                                  currentParameter,
                                                                                  len(currentSolution))

        inverseDifferentialSolutionMatrix = np.linalg.inv(differentialSolutionMatrix)

        parameterSpaceMetricMatrix = differentialParameterMatrix.T \
                                     @ inverseDifferentialSolutionMatrix.T \
                                     @ np.eye(len(currentSolution)) \
                                     @ inverseDifferentialSolutionMatrix \
                                     @ differentialParameterMatrix

        return parameterSpaceMetricMatrix

    def WriteMatrixInEuclideanBasisAtGivenPoint(self, matrixVectorFunction, x, mu, dimension):
        spaceDimension = len(x)
        indices = np.arange(dimension)
        A = np.zeros((spaceDimension, dimension))

        for index in indices:
            v = np.zeros(dimension)
            v[index] = 1

            A[:, index] = matrixVectorFunction(x, mu, v)

        return A

    def determineEllipse(self, framedata):

        metricMatrix = self.reconstructMetricMatrix(framedata)

        deltaParameter = targetParameter - self.parameters[framedata]

        eigvals, eigvecs = np.linalg.eig(metricMatrix)

        smallestIndex = np.argmin(eigvals)
        largestIndex = np.argmax(eigvals)

        print(eigvals)

        slope = eigvecs[1, smallestIndex] / eigvecs[0, smallestIndex]
        angle = 180.0 * np.arctan(slope) / np.pi

        if np.inner(deltaParameter, metricMatrix @ deltaParameter) < self.parameterMagnitude ** 2:
            _perturbationMagnitude = np.sqrt(np.inner(deltaParameter, metricMatrix @ deltaParameter))
        else:
            _perturbationMagnitude = self.parameterMagnitude

        return Ellipse(self.parameters[framedata], width= 2 * _perturbationMagnitude / np.sqrt(eigvals[smallestIndex]),
                               height=2 * _perturbationMagnitude / np.sqrt(eigvals[largestIndex]), angle=angle, fill=False)

ani = SubplotAnimation(results, parameterMagnitude)
# ani.save('test_sub.mp4')
plt.show()

print(results)