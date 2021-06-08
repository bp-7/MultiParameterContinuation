import numpy as np
import pandas as pd

nrows, ncols = 40, 30
nonSolvedCoeff = 500

dataPos = pd.read_csv("H:\MasterThesis\Code\src\TestsNotebook\FeasableMap\Ellipsoid\dataPos.txt", sep=" ")

xPos = np.array([float(dataPos.values[i][0]) for i in range(ncols)])
yPos = np.sort(np.array(list(set([float(dataPos.values[i][1]) for i in range(dataPos.shape[0])]))))
IterationsPos = np.array([int(dataPos.values[i][2]) for i in range(dataPos.shape[0])])
IterationsPos = IterationsPos.reshape((nrows, ncols))
iterMaxPos = np.max(IterationsPos)
testPos = np.where(IterationsPos < 0, IterationsPos * nonSolvedCoeff, IterationsPos)

Edge = np.zeros(IterationsPos.shape)
edgePositions = []
for i in range(1, testPos.shape[0]-1):
    for j in range(1, testPos.shape[1]-1):
        if testPos[i, j] > 0 and ((testPos[i, j] - testPos[i, j-1] > 500 and testPos[i, j] - testPos[i, j+1] < 200)
                                  or (testPos[i, j] - testPos[i, j-1] < 200 and testPos[i, j] - testPos[i, j+1] > 500)):
            Edge[i, j] = 1
            edgePositions.append(np.array([xPos[j], yPos[i]]))

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

startingTheta = 45. * np.pi / 180.
startingCoefficient = 1.

p1 = tau_x(-0.5)
p2 = tau_x(0.5)

p1Inv = np.linalg.inv(p1)
p2Inv = np.linalg.inv(p2)

initialSolution = [np.eye(3),
                   np.array([0., -0.25, 0.]),
                   np.array([-0.5, 0.5, 45. * np.pi / 180.])]

initialParameter = np.array([startingTheta, startingCoefficient])

def matrixRepresentationOfSE3Element(rotation, translation):
    return np.block([[rotation, np.reshape(translation, (3, 1))], [np.zeros((1, 3), dtype='float64'), 1]])

def matrixRepresentationOfse3Element(element):
    return np.block([[element[0], np.reshape(element[1], (3, 1))], [np.zeros((1, 4), dtype='float64')]])

def tupleRepresentationOfSE3Element(element):
    return element[:3, :3], element[:3, 3]

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

from Continuation.PositioningProblem.PathAdaptiveContinuationSecondOrderApproximation import PathAdaptiveContinuationSecondOrderApproximation
from Continuation.PositioningProblem.PathAdaptiveContinuationApproximateLength import PathAdaptiveContinuationApproximateLength

from Continuation.Helpers.AssembleMatrices import RepresentSquareOperatorInTotalBergerBasis

with open('Metrics1AtBorder.npy', 'wb') as f:
    for edgePosition in edgePositions:

        #continuation = PathAdaptiveContinuationSecondOrderApproximation(problem,
        #                                                                 initialSolution,
        #                                                                 initialParameter,
        #                                                                 edgePosition)

        continuation = PathAdaptiveContinuationApproximateLength(problem,
                                                                  initialSolution,
                                                                  initialParameter,
                                                                  edgePosition)

        results, parameterSpaceMetrics, perturbationMagnitudes, iterations, solved = continuation.Traverse()

        if solved:
            hessianMatrix = RepresentSquareOperatorInTotalBergerBasis(hess,
                                                                      product,
                                                                      results[-1][1])

            solutionHessianMatrix = hessianMatrix[:solutionSpaceDimension, :solutionSpaceDimension]
            inverseSolutionHessianMatrix = np.linalg.inv(solutionHessianMatrix)
            hessianMixteMatrix = hessianMatrix[:solutionSpaceDimension, solutionSpaceDimension:]

            metric1 = hessianMixteMatrix.T @ inverseSolutionHessianMatrix.T @ np.diag(np.array([2., 2., 2., 1., 1., 1., 1., 1., 1]))@ inverseSolutionHessianMatrix @ hessianMixteMatrix
            metric = hessianMatrix[solutionSpaceDimension:, solutionSpaceDimension:]
            np.save(f, metric)


###############################
###############################
###############################

with open('Metrics1AtBorder.npy', 'rb') as f:
    metrics = []
    for i in range(len(edgePositions)):
        a = np.load(f)
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
    if i < 24:
        metricMatrix = metrics[i]

        eigvals, eigvecs = np.linalg.eig(metricMatrix)

        smallestIndex = np.argmin(eigvals)
        largestIndex = np.argmax(eigvals)

        print(eigvals)

        slope = eigvecs[1, smallestIndex] / eigvecs[0, smallestIndex]
        angle = 180.0 * np.arctan(slope) / np.pi

        ax2.add_patch(Ellipse(edgePositions[i], width=2 * radius / np.sqrt(eigvals[smallestIndex]),
                       height=2 * radius / np.sqrt(eigvals[largestIndex]), angle=angle, fill=False))
    else:
        metricMatrix = metrics[i]

        eigvals, eigvecs = np.linalg.eig(metricMatrix)

        smallestIndex = np.argmin(eigvals)
        largestIndex = np.argmax(eigvals)

        print(eigvals)

        slope = eigvecs[1, smallestIndex] / eigvecs[0, smallestIndex]
        angle = 180.0 * np.arctan(slope) / np.pi

        ax2.add_patch(Ellipse(edgePositions[i + 1], width=2 * radius / np.sqrt(eigvals[smallestIndex]),
                              height=2 * radius / np.sqrt(eigvals[largestIndex]), angle=angle, fill=False))