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

from Continuation.StandardContinuation.StepSizeAdaptiveContinuation import StepSizeAdaptiveContinuation

x, y, Solved, Iterations, data = [], [], [], [], []


for finalCoefficient in finalCoefficients:
    for finalTheta in finalThetas:

        print("\n\n\nFINAL THETA = " + str(finalTheta * 180. / np.pi) + "\n")
        print("FINAL COEFFICIENT = " + str(finalCoefficient) + "\n\n\n")

        # Instantiate continuation object
        continuation = StepSizeAdaptiveContinuation(problem,
                                                    initialSolution,
                                                    initialParameter,
                                                    np.array([finalTheta, finalCoefficient]))

        results, iterations, solved = continuation.Traverse()

        data.append(([finalTheta, finalCoefficient], iterations, solved))
        x.append(finalTheta)
        y.append(finalCoefficient)
        Solved.append(solved)
        Iterations.append(iterations)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd

with open('dataPosLine.txt', 'w') as fp:
    fp.write('\n'.join('{} {} {}'.format(x[0],x[1],x[2]) for x in data))

x, y, Iterations = np.array(x), np.array(y), np.array(Iterations)
Iterations = Iterations.reshape((nrows, ncols))
iterMax = np.max(Iterations)
test = np.where(Iterations < 0, iterMax * 2, Iterations)

plt.imshow(test, extent=(x.min(), x.max(), y.min(), y.max()), interpolation='nearest', cmap=cm.gist_rainbow)
plt.colorbar()
plt.show()