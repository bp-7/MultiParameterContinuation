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

initialSolution = [np.eye(3),
                   np.array([0., -0.25, 0.]),
                   np.array([-0.5, 0.5, 45. * np.pi / 180.])]

initialParameter = np.array([startingTheta, startingCoefficient])

ncols = 40
nrows = 100
finalThetas = np.linspace(0. * np.pi + 1e-3, 0.5 * np.pi - 1e-3, ncols)
finalCoefficients = np.linspace(0, 10, nrows)

distanceBetweenContactPoints = 1.

positioningProblem = ParabolaPositioningProblem(solutionSpace, SE3Squared, parameterSpace, distanceBetweenContactPoints)

################################################################
################################################################
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

################################################################
################################################################

from pymanopt.core.problem import Problem

# Instantiate the problem
problem = Problem(product, cost=positioningProblem.Cost, hess=hess)

from Continuation.PositioningProblem.PerturbationWithDifferential.PathAdaptiveContinuation import PathAdaptiveMultiParameterContinuation


x, y, Solved, Iterations, data = [], [], [], [], []

for finalCoefficient in finalCoefficients:
    for finalTheta in finalThetas:

        print("\n\n\nFINAL THETA = " + str(finalTheta * 180. / np.pi) + "\n")
        print("FINAL COEFFICIENT = " + str(finalCoefficient) + "\n\n\n")

        targetParameter = np.array([finalTheta, finalCoefficient])
        continuation = PathAdaptiveMultiParameterContinuation(problem,
                                                               positioningProblem,
                                                               initialSolution,
                                                               initialParameter,
                                                               targetParameter,
                                                               500)

        results, parameterSpaceMetrics, perturbationMagnitudes, iterations, solved = continuation.Traverse()

        data.append(([finalTheta, finalCoefficient], iterations, solved))
        x.append(finalTheta)
        y.append(finalCoefficient)
        Solved.append(solved)
        Iterations.append(iterations)

############################################################
# Save results and plot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

fileName = 'feasibleMap'
with open(fileName + '.txt', 'w') as fp:
    fp.write('\n'.join('{} {} {}'.format(x[0],x[1],x[2]) for x in data))

x, y, Iterations = np.array(x), np.array(y), np.array(Iterations)
Iterations = Iterations.reshape((nrows, ncols))
iterMax = np.max(Iterations)
test = np.where(Iterations < 0, np.nan, Iterations)

plt.imshow(test, extent=(x.min(), x.max(), y.min(), y.max()), interpolation='nearest', cmap=cm.turbo)
plt.colorbar()
plt.show()

