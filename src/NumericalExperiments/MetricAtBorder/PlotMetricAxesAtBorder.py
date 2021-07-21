import numpy as np
import pandas as pd
import math

fileName = 'bla'
nrows, ncols = 50, 40

def DetectEdge(file, nrows, ncols):
    data = pd.read_csv(file, sep=" ")
    x = np.array([float(data.values[i][0]) for i in range(ncols)])
    y = np.sort(np.array(list(set([float(data.values[i][1]) for i in range(data.shape[0])]))))
    Iterations = np.array([int(data.values[i][2]) for i in range(data.shape[0])])
    Iterations = Iterations.reshape((nrows, ncols))
    IterationsWithNaN = np.where(Iterations < 0, np.nan, Iterations)

    Edge = np.zeros(Iterations.shape)
    edgePositions = []
    for i in range(1, IterationsWithNaN.shape[0]-1):
        for j in range(1, IterationsWithNaN.shape[1]-1):
            if not math.isnan(IterationsWithNaN[i, j]) and (math.isnan(IterationsWithNaN[i+1, j]) or math.isnan(IterationsWithNaN[i-1, j])
                    or math.isnan(IterationsWithNaN[i, j-1]) or math.isnan(IterationsWithNaN[i, j+1])):
                Edge[i, j] = 1
                edgePositions.append(np.array([x[j], y[i]]))

    return Edge, edgePositions

Edge, edgePositions = DetectEdge(fileName, nrows, ncols)

####################################################################################################
################################## Instantiate the problem #########################################
####################################################################################################
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

# Instantiate the parameter space
parameterSpaceDimension = 2
parameterSpace = Euclidean(parameterSpaceDimension)

# Instantiate the solution manifold
n = 3
Rn = Euclidean(n)
productAmbient = Product((R33, euclideanSpace, Rn, parameterSpace))
product = Product((SO3, euclideanSpace, Rn, parameterSpace))

# Instantiate the constrained manifold
SE3Squared = Product((SO3, Rn, SO3, Rn))

# Define positioning problem
from PositioningProblems.Parabola.TwoDimensionalParameterSpace.ParabolaPositioningProblem import ParabolaPositioningProblem

startingTheta, startingCoefficient = 45. * np.pi / 180., 1.
finalTheta, finalCoefficient = 150 * np.pi / 180., -7.

initialSolution = [np.eye(3),
                   np.array([0., -0.25, 0.]),
                   np.array([-0.5, 0.5, 45. * np.pi / 180.])]

initialParameter = np.array([startingTheta, startingCoefficient])
targetParameter = np.array([finalTheta, finalCoefficient])

initialCondition = list(initialSolution) + [initialParameter]

distanceBetweenContactPoints = 1.

positioningProblem = ParabolaPositioningProblem(solutionSpace, SE3Squared, parameterSpace, distanceBetweenContactPoints)

# Compute hessian

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

###################################################################################################
###################################################################################################
# Solve problem

from pymanopt.core.problem import Problem

# Instantiate the problem
problem = Problem(product, cost=positioningProblem.Cost, hess=hess)

from Continuation.PositioningProblem.PerturbationWithDifferential.LinearContinuation import LinearMultiParameterContinuation
from Helpers.AssembleMatrices import RepresentRectangularOperatorFromParameterSpaceToSE32, RepresentRectangularOperatorFromSolutionSpaceToSE32

with open('MetricsAtBorder.npy', 'wb') as f:
    for edgePosition in edgePositions:
        continuation = LinearMultiParameterContinuation(problem,
                                                        positioningProblem,
                                                        initialSolution,
                                                        initialParameter,
                                                        edgePosition,
                                                        500)

        results, parameterSpaceMetrics, perturbationMagnitudes, iterations, solved = continuation.Traverse()

        if solved:
            currentPoint = results[-1][1]
            FPoint = F(currentPoint)
            DF_S_Mat = RepresentRectangularOperatorFromSolutionSpaceToSE32(positioningProblem.DifferentialSolution,
                                                                           solutionSpace,
                                                                           SE3Squared,
                                                                           currentPoint,
                                                                           FPoint)

            DF_S_Mat_PInv = np.linalg.pinv(DF_S_Mat)

            DF_mu_Mat = RepresentRectangularOperatorFromParameterSpaceToSE32(positioningProblem.DifferentialParameter,
                                                                             parameterSpace,
                                                                             SE3Squared,
                                                                             currentPoint,
                                                                             FPoint)

            metric = DF_mu_Mat.T \
                     @ DF_S_Mat_PInv.T \
                     @ np.diag([2., 2., 2., 1., 1., 1., 1., 1., 1.]) \
                     @ DF_S_Mat_PInv \
                     @ DF_mu_Mat

            np.save(f, metric)


#############################################################################################
#############################################################################################
# PLot metric at border


with open('MetricsAtBorder.npy', 'rb') as f:
    metrics = []
    for i in range(len(edgePositions)):
        a = np.load(f, allow_pickle=True)
        metrics.append(a)


from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 8), dpi=100)

ax2 = fig.add_subplot(1, 2, 1)

ax2.grid()
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_xlim(0, 2)
ax2.set_ylim(0, 4)
radius = 0.2
for i in range(len(metrics)):
    if i % 3:
        metricMatrix = metrics[i]

        eigvals, eigvecs = np.linalg.eig(metricMatrix)

        smallestIndex = np.argmin(eigvals)
        largestIndex = np.argmax(eigvals)

        print(eigvals)

        slope = eigvecs[1, smallestIndex] / eigvecs[0, smallestIndex]
        angle = 180.0 * np.arctan(slope) / np.pi

        ax2.add_patch(Ellipse(edgePositions[i], width=2 * radius / np.sqrt(eigvals[smallestIndex]),
                       height=2 * radius / np.sqrt(eigvals[largestIndex]), angle=angle, fill=False))
