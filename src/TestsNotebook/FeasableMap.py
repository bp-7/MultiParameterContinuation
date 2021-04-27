import autograd.numpy as np

from pymanopt.manifolds import Product
from pymanopt.manifolds import Rotations
from pymanopt.manifolds import Euclidean

# Dimension of the sphere
solutionSpaceDimension = 9

# Instantiate the SE(3) manifold
euclideanSpace = Euclidean(3)
rotationSpace = Rotations(3)
specialEuclideanGroup = Product((rotationSpace, euclideanSpace))

# Instantiate the solution space
solutionSpace = Product((rotationSpace, euclideanSpace, euclideanSpace))

# Dimension of the parameter space
parameterSpaceDimension = 2

# Instantiate the parameter space
parameterSpace = Euclidean(2)

# Instantiate the global manifold
n = 3
Rn = Euclidean(n)
product = Product((rotationSpace, euclideanSpace, Rn, parameterSpace))


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

ncols = 30
nrows = 40
finalThetas = np.linspace(0. * np.pi + 1e-3, 0.5 * np.pi - 1e-3, ncols)
finalCoefficients = np.linspace(1e-3, 4, nrows)

p1 = tau_x(-0.5)
p2 = tau_x(0.5)

p1Inv = np.linalg.inv(p1)
p2Inv = np.linalg.inv(p2)

initialSolution = (np.eye(3),
                   np.array([0., -0.25, 0.]),
                   np.array([-0.5, 0.5, 45. * np.pi / 180.]))

initialParameter = np.array([startingTheta, startingCoefficient])

def matrixRepresentationOfSE3Element(element):
    return np.block([[element[0], np.reshape(element[1], (3, 1))], [np.zeros((1, 3), dtype='float64'), 1]])

def matrixRepresentationOfse3Element(element):
    return np.block([[element[0], np.reshape(element[1], (3, 1))], [np.zeros((1, 4), dtype='float64')]])

def tupleRepresentationOfSE3Element(element):
    return element[:3, :3], element[:3, 3]


def cost(S):
    I = np.eye(4)
    phi = matrixRepresentationOfSE3Element((S[0], S[1]))
    u = phi @ C(S[2][0], S[3][1]) @ rho_z(0.5 * np.pi - S[3][0]) @ p1Inv - I
    v = phi @ C(S[2][1], S[3][1]) @ rho_z(- S[2][2]) @ p2Inv - I

    return np.trace(u.T @ u) + np.trace(v.T @ v)

from pymanopt.core.problem import Problem

# Instantiate the problem
problem = Problem(product, cost=cost)

from Continuation.PositioningProblem.PathAdaptiveStepSizeAdaptiveContinuation import PathAdaptiveMultiParameterContinuation

x, y, Solved, Iterations, data = [], [], [], [], []


for finalCoefficient in finalCoefficients:
    for finalTheta in finalThetas:

        print("\n\n\nFINAL THETA = " + str(finalTheta * 180. / np.pi) + "\n")
        print("FINAL COEFFICIENT = " + str(finalCoefficient) + "\n\n\n")

        # Instantiate continuation object
        continuation = PathAdaptiveMultiParameterContinuation(problem,
                                                          initialSolution,
                                                          initialParameter,
                                                          np.array([finalTheta, finalCoefficient]),
                                                          'ellipsoid',
                                                          'naive')

        results, parameterSpaceMetrics, perturbationMagnitudes, iterations, solved = continuation.Traverse()

        data.append(([finalTheta, finalCoefficient], iterations, solved))
        x.append(finalTheta)
        y.append(finalCoefficient)
        Solved.append(solved)
        Iterations.append(iterations)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd

with open('data2.txt', 'w') as fp:
    fp.write('\n'.join('{} {} {}'.format(x[0],x[1],x[2]) for x in data))

x, y, Iterations = np.array(x), np.array(y), np.array(Iterations)
Iterations = Iterations.reshape((nrows, ncols))
iterMax = np.max(Iterations)
test = np.where(Iterations < 0, iterMax * 2, Iterations)

plt.imshow(test, extent=(x.min(), x.max(), y.min(), y.max()), interpolation='nearest', cmap=cm.gist_rainbow)
plt.colorbar()
plt.show()

cd H:\MasterThesis\Code\src\TestsNotebook

data = pd.read_csv("dataPos.txt", sep=" ")
x = np.array([float(data.values[i][0]) for i in range(ncols)])
y = np.array(list(set([float(data.values[i][1]) for i in range(data.shape[0])])))
Iterations = np.array([int(data.values[i][2]) for i in range(data.shape[0])])
Iterations = Iterations.reshape((nrows, ncols))
iterMax = np.max(Iterations)
test = np.where(Iterations < 0, iterMax * 2, Iterations)

plt.imshow(test[::-1], extent=(x.min(), x.max(), y.min(), y.max()), interpolation='nearest')
plt.colorbar()
plt.show()

cd H:\MasterThesis\Code\src\TestsNotebook\FeasableMap\Ellipsoid
dataPos = pd.read_csv("dataPos.txt", sep=" ")
dataNeg = pd.read_csv("dataNeg.txt", sep=" ")

nrows, ncols = 40, 30

nonSolvedCoeff = 250

xPos = np.array([float(dataPos.values[i][0]) for i in range(ncols)])
yPos = np.array(list(set([float(dataPos.values[i][1]) for i in range(dataPos.shape[0])])))
IterationsPos = np.array([int(dataPos.values[i][2]) for i in range(dataPos.shape[0])])
IterationsPos = IterationsPos.reshape((nrows, ncols))
iterMaxPos = np.max(IterationsPos)
testPos = np.where(IterationsPos < 0, IterationsPos * nonSolvedCoeff, IterationsPos)
plt.imshow(testPos[::-1], extent=(xPos.min(), xPos.max(), yPos.min(), yPos.max()), interpolation='nearest', cmap=cm.gist_rainbow)
plt.colorbar()
plt.show()

xNeg = np.array([float(dataNeg.values[i][0]) for i in range(ncols)])
yNeg = np.array(list(set([float(dataNeg.values[i][1]) for i in range(dataNeg.shape[0])])))
IterationsNeg = np.array([int(dataNeg.values[i][2]) for i in range(dataNeg.shape[0])])
IterationsNeg = IterationsNeg.reshape((nrows, ncols))
iterMaxNeg = np.max(IterationsNeg)
testNeg = np.where(IterationsNeg < 0, IterationsNeg * nonSolvedCoeff, IterationsNeg)
plt.imshow(testNeg, extent=(xNeg.min(), xNeg.max(), yNeg.min(), yNeg.max()), interpolation='nearest', cmap=cm.gist_rainbow)
plt.colorbar()
plt.show()


IterationsPosExtent = [-1 * np.ones(shape=(ncols)) for i in range(nrows)]
IterationsPos = list(IterationsPos)
testIters = np.concatenate((IterationsPos, IterationsPosExtent), axis=1)

testPos = np.where(testIters < 0,testIters * nonSolvedCoeff, testIters)

xPos = [float(dataPos.values[i][0]) for i in range(ncols)]
xPosExtention = np.linspace(90. * np.pi / 180., np.pi, 30)

for i in range(30):
    xPos.append(xPosExtention[i])

xPos = np.array(xPos)

plt.imshow(testPos[::-1], extent=(xPos.min(), xPos.max(), yPos.min(), yPos.max()), interpolation='nearest', cmap=cm.gist_rainbow)
plt.colorbar()
plt.show()


testItersNeg = np.concatenate((IterationsPosExtent, IterationsNeg), axis=1)
testNeg = np.where(testItersNeg < 0,testItersNeg * nonSolvedCoeff, testItersNeg)

xNeg = [float(dataNeg.values[i][0]) for i in range(ncols)]
xNegExtention = np.linspace(0.0, 90. * np.pi / 180., 30)
for i in range(30):
    xNeg.append(xNegExtention[i])

xNeg = np.array(xNeg)

plt.imshow(testNeg, extent=(xNeg.min(), xNeg.max(), yNeg.min(), yNeg.max()), interpolation='nearest', cmap=cm.gist_rainbow)
plt.colorbar()
plt.show()

test = np.concatenate((testPos[::-1],testNeg), axis=0)

plt.imshow(test, extent=(0, np.pi, yNeg.min(), yPos.max()), interpolation='nearest', cmap=cm.gist_rainbow)
plt.colorbar()
plt.show()