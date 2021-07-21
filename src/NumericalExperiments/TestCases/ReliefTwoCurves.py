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

# Instantiate the optimization problem
problem = Problem(product, cost=positioningProblem.Cost, hess=hess)

from Continuation.PositioningProblem.PerturbationWithDifferential.PathAdaptiveContinuation import PathAdaptiveMultiParameterContinuation

# Instantiate continuation object
targetParameter = np.array([23.18181818,  3.66363636,  40 * np.pi / 180,  3.3,  0.9,  3, 0.5,  0.7])

tolerance = 300
continuation = PathAdaptiveMultiParameterContinuation(problem,
                                                       positioningProblem,
                                                       initialSolution,
                                                       initialParameter,
                                                       targetParameter,
                                                       tolerance)

results, parameterSpaceMetrics, perturbationMagnitudes, iterations, solved = continuation.Traverse()


##########################################################################
##########################################################################

import matplotlib.pyplot as plt
from Visualization.VisualizationReliefTwoCurves import SubplotAnimationReliefTwoCurves

plt.ion()

ani = SubplotAnimationReliefTwoCurves(results,
                                      parameterSpaceMetrics,
                                      perturbationMagnitudes,
                                      targetParameter,
                                      positioningProblem)

ani.show()