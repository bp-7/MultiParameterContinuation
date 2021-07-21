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
parameterSpaceDimension = 3
parameterSpace = Euclidean(parameterSpaceDimension)

# Instantiate the solution manifold
n = 3
Rn = Euclidean(n)
productAmbient = Product((R33, euclideanSpace, Rn, parameterSpace))
product = Product((SO3, euclideanSpace, Rn, parameterSpace))

# Instantiate the constrained manifold
SE3Squared = Product((SO3, Rn, SO3, Rn))

# Define positioning problem
from PositioningProblems.Parabola.ThreeDimensionalParameterSpace.ParabolaPositioningProblem import ParabolaPositioningProblem

startingTheta, startingCoefficient, startingH = 45. * np.pi / 180., 1., 1.
finalTheta, finalCoefficient, finalH = 1.8, -4., 1.

initialSolution = [np.eye(3),
                   np.array([0., -0.25, 0.]),
                   np.array([-0.5, 0.5, 45. * np.pi / 180.])]

initialParameter = np.array([startingTheta, startingCoefficient, startingH])
targetParameter = np.array([finalTheta, finalCoefficient, finalH])

initialCondition = list(initialSolution) + [initialParameter]

positioningProblem = ParabolaPositioningProblem(solutionSpace, SE3Squared, parameterSpace)

# Compute hessian

def Skew(A):
    return 0.5 * (A - A.T)


def SkewMat(w):
    return np.array([[0., -w[2], w[1]], [w[2], 0., -w[0]], [-w[1], w[0], 0.]], dtype=float)


def diffProj(x, z, v):
    return product.proj(x, [x[0] @ Skew(z[0].T @ v[0]) + z[0] @ Skew(x[0].T @ v[0]), np.zeros(x[1].shape),
                            np.zeros(x[2].shape), np.zeros(x[3].shape)])


def hess(x, z):
    egrad = problem.egrad(x)
    ehess = problem.ehess(x, [SO3.tangent2ambient(x[0], z[0]), z[1], z[2], z[3]])
    return product.proj(x, ehess) + diffProj(x, [SO3.tangent2ambient(x[0], z[0]), z[1], z[2], z[3]], egrad)

###################################################################################################
###################################################################################################
from pymanopt.core.problem import Problem

# Instantiate the problem
problem = Problem(product, cost=positioningProblem.Cost, hess=hess)


hs = np.linspace(0, 5, 102)
coefficients = np.linspace(-10, 10, 102)
thetas = np.linspace(0, np.pi, 102)

import random
from Continuation.PositioningProblem.PerturbationWithDifferential.PathAdaptiveContinuation import PathAdaptiveMultiParameterContinuation
from Continuation.PositioningProblem.PerturbationWithDifferential.LinearContinuation import LinearMultiParameterContinuation

indexTheta = random.randint(1, 99)
indexH = random.randint(1, 100)
indexCoefficients = random.randint(0, 99)
epsLine = 500
epsEll = 15

if thetas[indexTheta] < 0.5 * np.pi:
    while coefficients[indexCoefficients] < 0:
        indexCoefficients = random.randint(0, 99)
else:
    while coefficients[indexCoefficients] > 0:
        indexCoefficients = random.randint(0, 99)

targetParameter = np.array([thetas[indexTheta], coefficients[indexCoefficients], hs[indexH]])

continuation1 = PathAdaptiveMultiParameterContinuation(problem,
                                                       positioningProblem,
                                                       initialSolution,
                                                       initialParameter,
                                                       targetParameter,
                                                       epsEll)

results1, parameterSpaceMetrics1, perturbationMagnitudes1, iterations1, solved1 = continuation1.Traverse()

initialSolution = results1[-1][1][:-1]
initialParameter = targetParameter
print("initial Parameter = " + str(targetParameter))
numberOfPointsComputed = 300

straightLineSolved, straightLineIterations = [], []
ellipsoidSolved, ellipsoidIterations = [], []

for i in range(numberOfPointsComputed):

    indexTheta = random.randint(1, 99)
    indexH = random.randint(1, 100)
    indexCoefficients = random.randint(0, 99)
    epsLine = 500
    epsEll = 15

    if thetas[indexTheta] < 0.5 * np.pi:
        while coefficients[indexCoefficients] < 0:
            indexCoefficients = random.randint(0, 99)
    else:
        while coefficients[indexCoefficients] > 0:
            indexCoefficients = random.randint(0, 99)

    targetParameter = np.array([thetas[indexTheta], coefficients[indexCoefficients], hs[indexH]])

    print("#################################################\n TARGET PARAMETER = " + str(targetParameter) + "\n\n")
    print("\n\n i = " + str(i) + "\n")
    continuation1 = PathAdaptiveMultiParameterContinuation(problem,
                                                           positioningProblem,
                                                           initialSolution,
                                                           initialParameter,
                                                           targetParameter,
                                                           epsEll)

    continuation2 = LinearMultiParameterContinuation(problem,
                                                     positioningProblem,
                                                     initialSolution,
                                                     initialParameter,
                                                     targetParameter,
                                                     epsLine)

    results1, parameterSpaceMetrics1, perturbationMagnitudes1, iterations1, solved1 = continuation1.Traverse()
    results2, parameterSpaceMetrics2, perturbationMagnitudes2, iterations2, solved2 = continuation2.Traverse()

    ellipsoidIterations.append(iterations1)
    straightLineIterations.append(iterations2)
    ellipsoidSolved.append(solved1)
    straightLineSolved.append(solved2)

    

