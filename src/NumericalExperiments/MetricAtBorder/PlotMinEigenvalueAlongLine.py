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

nPoints = 150
finalThetas = np.linspace(1.3, 0.5 * np.pi, nPoints)
finalCoefficient = 3

startingTheta, startingCoefficient = 45. * np.pi / 180., 1.

initialSolution = [np.eye(3),
                   np.array([0., -0.25, 0.]),
                   np.array([-0.5, 0.5, 45. * np.pi / 180.])]

initialParameter = np.array([startingTheta, startingCoefficient])

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

from Continuation.PositioningProblem.PerturbationWithDifferential.LinearContinuation import LinearMultiParameterContinuation
from Helpers.AssembleMatrices import RepresentRectangularOperatorFromParameterSpaceToSE32, RepresentRectangularOperatorFromSolutionSpaceToSE32

x, y, Solved, Iterations, data = [], [], [], [], []

lambdas_min_DF_S, lambdas_min_metric, conds = [], [], []

for finalTheta in finalThetas:

    print("\n\n\nFINAL THETA = " + str(finalTheta * 180. / np.pi) + "\n")
    print("FINAL COEFFICIENT = " + str(finalCoefficient) + "\n\n\n")

    targetParameter = np.array([finalTheta, finalCoefficient])

    continuation = LinearMultiParameterContinuation(problem,
                                                    positioningProblem,
                                                    initialSolution,
                                                    initialParameter,
                                                    targetParameter,
                                                    500)

    results, parameterSpaceMetrics, perturbationMagnitudes, iterations, solved = continuation.Traverse()

    currentPoint = results[-1][1]
    FPoint = positioningProblem.SystemEvaluation(currentPoint)

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

    U, D, V = np.linalg.svd(DF_S_Mat)
    lambdas_min_DF_S.append(np.min(D))
    conds.append(np.linalg.cond(metric))
    lambdas_min_metric.append(np.max(np.linalg.eigvals(metric)))

