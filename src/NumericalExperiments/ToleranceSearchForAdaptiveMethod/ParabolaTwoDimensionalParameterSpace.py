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
from pymanopt.core.problem import Problem

# Instantiate the problem
problem = Problem(product, cost=positioningProblem.Cost, hess=hess)

numberOfEpsilonValues = 10
coefficients = np.linspace(0, 10, 102)
thetas = np.linspace(0. * np.pi, 0.5 * np.pi, 102)
epsilons = np.logspace(0, 4, numberOfEpsilonValues)

import random
from Continuation.PositioningProblem.PerturbationWithDifferential.PathAdaptiveContinuation import PathAdaptiveMultiParameterContinuation
from Continuation.PositioningProblem.PerturbationWithDifferential.LinearContinuation import LinearMultiParameterContinuation

numberOfPointsComputed = 100


StraightLineSolved, StraightLineIterations = [], []
EllipsoidSolved, EllipsoidIterations = [], []

for i in range(numberOfPointsComputed):

    straightLineSolved, straightLineIterations = [], []
    ellipsoidSolved, ellipsoidIterations = [], []

    indexTheta = random.randint(1, 99)
    indexCoefficients = random.randint(0, 99)

    if thetas[indexTheta] < 0.5 * np.pi:
        while coefficients[indexCoefficients] < 0:
            indexCoefficients = random.randint(0, 99)
    else:
        while coefficients[indexCoefficients] > 0:
            indexCoefficients = random.randint(0, 99)

    targetParameter = np.array([thetas[indexTheta], coefficients[indexCoefficients]])

    print("#################################################\n TARGET PARAMETER = " + str(targetParameter) + "\n\n")
    print("#################################################\n i = " + str(i) + "\n\n")

    for eps in epsilons:
        print("#################################################\n EPSILON = " + str(eps) + "\n\n")
        continuation1 = PathAdaptiveMultiParameterContinuation(problem,
                                                               positioningProblem,
                                                               initialSolution,
                                                               initialParameter,
                                                               targetParameter,
                                                               eps)

        continuation2 = LinearMultiParameterContinuation(problem,
                                                         positioningProblem,
                                                         initialSolution,
                                                         initialParameter,
                                                         targetParameter,
                                                         eps)

        results1, parameterSpaceMetrics1, perturbationMagnitudes1, iterations1, solved1 = continuation1.Traverse()
        results2, parameterSpaceMetrics2, perturbationMagnitudes2, iterations2, solved2 = continuation2.Traverse()


        ellipsoidIterations.append(iterations1)
        straightLineIterations.append(iterations2)
        ellipsoidSolved.append(solved1)
        straightLineSolved.append(solved2)

    EllipsoidIterations.append(ellipsoidIterations)
    StraightLineIterations.append(straightLineIterations)
    EllipsoidSolved.append(ellipsoidSolved)
    StraightLineSolved.append(straightLineSolved)

################################
StraightLineIterationsArr = np.reshape(StraightLineIterations, (numberOfPointsComputed, numberOfEpsilonValues))
EllipsoidIterationsArr = np.reshape(EllipsoidIterations, (numberOfPointsComputed, numberOfEpsilonValues))

StraightLineSolvedArr = np.reshape(StraightLineSolved, (numberOfPointsComputed, numberOfEpsilonValues))
EllipsoidSolvedArr = np.reshape(EllipsoidSolved, (numberOfPointsComputed, numberOfEpsilonValues))

StraightLineIterationsArr = np.where(StraightLineIterationsArr < 0, np.nan, StraightLineIterationsArr)
EllipsoidIterationsArr = np.where(EllipsoidIterationsArr < 0, np.nan, EllipsoidIterationsArr)

StraightLineSolvedArr = np.where(StraightLineSolvedArr == False, 0, 1)
EllipsoidSolvedArr = np.where(EllipsoidSolvedArr == False, 0, 1)

StraightLineNumberOfSolved = np.sum(StraightLineSolvedArr, axis=0)
EllipsoidNumberOfSolved = np.sum(EllipsoidSolvedArr, axis=0)

StraightLineAverageIterations = np.nanmean(StraightLineIterationsArr, axis=0)
EllipsoidAverageIterations = np.nanmean(EllipsoidIterationsArr, axis=0)

################################
# fig = plt.figure(figsize=(14, 7), dpi=100)
#
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "sans-serif",
#     "font.sans-serif": ["Helvetica"]})
# plt.rcParams['font.size'] = '16'
#
# ax1 = fig.add_subplot(1, 2, 2)
# ax2 = fig.add_subplot(1, 2, 1)
#
# ax1.set_ylabel(r"$N_{RBFGS}$")
# ax1.set_xlabel(r"$\varepsilon$")
# ax2.set_ylabel(r"$N_{Solved}$")
# ax2.set_xlabel(r"$\varepsilon$")
#
# ax1.semilogx(epsilons, EllipsoidAverageIterations, 'b+', markersize=5, label="Ellipsoid method")
# ax1.semilogx(epsilons, StraightLineAverageIterations, 'r+', markersize=5, label="Straight line method")
#
# ax2.semilogx(epsilons, EllipsoidNumberOfSolved, 'b+', markersize=5, label="Ellipsoid method")
# ax2.semilogx(epsilons, StraightLineNumberOfSolved, 'r+', markersize=5, label="Straight line method")
#
# ax1.legend()
# ax2.legend()