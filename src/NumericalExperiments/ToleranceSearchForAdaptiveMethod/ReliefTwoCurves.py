import autograd.numpy as np

from pymanopt.manifolds import Product
from pymanopt.manifolds import Rotations
from pymanopt.manifolds import Euclidean

from Solver.SolverRBFGSPositioning import SolverRBFGS

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
parameterSpaceDimension = 8

# Instantiate the parameter space
parameterSpace = Euclidean(parameterSpaceDimension)

# Instantiate the global manifold
productAmbient = Product((R33, R3, R6, parameterSpace))
product = Product((SO3, R3, R6, parameterSpace))

# Instantiate the constrained manifold
SE3Squared = Product((SO3, R3, SO3, R3))

# Define constants
offsetWheel = np.array([0., -30., 0.])
wheelProfileParameter = 0.
helixLength = 10

# Define initial solution and initial parameter
initialHelixAngle, initialHelixRadius = 20. * np.pi / 180., 2.
initialOffsetAngle, initialTrajectoryParameter = 30. * np.pi / 180., 0.
initialTorusLargeRadius, initialTorusSmallRadius = 20, 5.

initialParameter = np.array([20., 5., initialHelixAngle, initialHelixRadius, initialHelixAngle, initialHelixRadius, initialOffsetAngle, initialTrajectoryParameter])

# Define positioning problem
from PositioningProblems.ReliefTwoCurves.ReliefTwoCurvesPositioningProblem import ReliefTwoCurvesPositioningProblem

positioningProblem = ReliefTwoCurvesPositioningProblem(solutionSpace,
                                                       SE3Squared,
                                                       parameterSpace,
                                                       wheelProfileParameter,
                                                       helixLength,
                                                       offsetWheel)

initialSolution = positioningProblem.FindInitialCondition(initialParameter)

######################################################################################################################
######################################################################################################################
# Define hessian

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

###################################################################################
###################################################################################

from pymanopt.core.problem import Problem

# Instantiate the problem
problem = Problem(product, cost=positioningProblem.Cost, hess=hess)

numberOfEpsilonValues = 10
Rts = np.linspace(10, 25, 100)
rts = np.linspace(0.1, 5, 100)
helixAngles1 = np.linspace(0, 70 * np.pi / 180., 100)
helixAngles2 = np.linspace(0, 70 * np.pi / 180., 100)
helixRadius1 = np.linspace(2, 4, 100)
helixRadius2 = np.linspace(2, 4, 100)
offsetAngles = np.linspace(1e-3, 0.5 * np.pi, 100)
trajectoryParameters = np.linspace(0, 1, 100)

epsilons = np.logspace(0, 4, numberOfEpsilonValues)

import random
from Continuation.PositioningProblem.DifferentialF.PathAdaptiveContinuation import PathAdaptiveMultiParameterContinuation
from Continuation.PositioningProblem.DifferentialF.LinearContinuation import LinearMultiParameterContinuation

numberOfPointsComputed = 100


StraightLineSolved, StraightLineIterations = [], []
EllipsoidSolved, EllipsoidIterations = [], []
targetParameters = []
for i in range(numberOfPointsComputed):

    straightLineSolved, straightLineIterations = [], []
    ellipsoidSolved, ellipsoidIterations = [], []

    indices = np.random.randint(0, 99, 8)
    indexCoefficients = random.randint(0, 99)

    if helixAngles1[indices[2]] > helixAngles2[indices[4]]:
        while offsetAngles[indices[7]] < 40. * np.pi / 180:
            indices[7] = np.random.randint(1, 99)

    while np.abs(helixAngles1[indices[2]] - helixAngles2[indices[4]]) > 40 * np.pi / 180:
        indices[2] = np.random.randint(1, 99)


    targetParameter = np.array([Rts[indices[0]],
                                rts[indices[1]],
                                helixAngles1[indices[2]],
                                helixRadius1[indices[3]],
                                helixAngles2[indices[4]],
                                helixRadius2[indices[5]],
                                offsetAngles[indices[6]],
                                trajectoryParameters[indices[7]]])

    targetParameters.append(targetParameter)
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