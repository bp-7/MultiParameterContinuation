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

from Continuation.PositioningProblem.PerturbationWithDifferential.PathAdaptiveContinuation import \
    PathAdaptiveMultiParameterContinuation
from Continuation.PositioningProblem.PerturbationWithDifferential.LinearContinuation import LinearMultiParameterContinuation

tolerance = 10
continuation = PathAdaptiveMultiParameterContinuation(problem,
                                                      positioningProblem,
                                                      initialSolution,
                                                      initialParameter,
                                                      targetParameter,
                                                      tolerance)

results, parameterSpaceMetrics, perturbationMagnitudes, iterations, solved = continuation.Traverse()

##########################################################################
##########################################################################
# Test if the angle is good
from Helpers.Parameterizations.SE3Representation import matrixRepresentationOfSE3Element
from SE3Parameterizations.Parameterizations.Parabola import Parabola

finalSolution = results[-1][1]
R, r, t, mu = finalSolution[0], finalSolution[1], finalSolution[2], finalSolution[3]
phi = matrixRepresentationOfSE3Element(R, r)

movingCurve = phi @ Parabola(finalSolution[2][0], finalSolution[3][1])
angle = np.arctan(movingCurve[1, 0] / movingCurve[0, 1]) + 0.5 * np.pi
if np.abs(angle - targetParameter[0]):
    print("Final angle is true.")
else:
    print("Final angle is wrong.")

##########################################################################
##########################################################################
########################### VISUALIZATION ################################
##########################################################################
##########################################################################
import matplotlib.pyplot as plt
from Visualization.VisualizationParabola import SubplotAnimationTwoDimensionalParameterSpace


plt.ion()

ani = SubplotAnimationTwoDimensionalParameterSpace(results,
                                                    parameterSpaceMetrics,
                                                    perturbationMagnitudes,
                                                    targetParameter)

ani.show()
