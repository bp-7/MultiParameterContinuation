import autograd.numpy as np

from pymanopt.manifolds import Product
from pymanopt.manifolds import Rotations
from pymanopt.manifolds import Euclidean

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
parameterSpaceDimension = 2

# Instantiate the parameter space
parameterSpace = Euclidean(2)

# Instantiate the global manifold
productAmbient = Product((R33, R3, R6, parameterSpace))
product = Product((SO3, R3, R6, parameterSpace))

from SE3Parameterizations.Helpers.SE3Representation import matrixRepresentationOfSE3Element
from SE3Parameterizations.Helpers.BasicSE3Transformations import rho_x, rho_z
from SE3Parameterizations.Parameterizations.Helix import Helix
from SE3Parameterizations.Parameterizations.Torus import Tore

# Define constants
rt, Rt = 3, 20
offsetWheel = np.array([0., 40., 0.])
wheelProfileParameter = 0.
# exceptionally, fix the trajectory parameter
trajectoryParameter = 0.

helixLength = 10
offsetAngle = 30. * np.pi / 180.
reliefAngle = 20. * np.pi / 180.

def cost(S):
    phi = matrixRepresentationOfSE3Element(S[0], S[1])
    t, mu = S[2], S[3]

    grindingMark = -0.5 * np.pi - mu[1]

    u = phi @ Tore(0., wheelProfileParameter, rt, Rt, offsetWheel) @ rho_z(grindingMark) \
        - Helix(trajectoryParameter, mu[0], mu[1], helixLength) @ rho_x(-np.arctan(np.tan(reliefAngle) / np.cos(mu[1])))

    return np.trace(u.T @ u)

# Define initial solution and initial parameter
initialParameter = np.array([2., 0.])#30. * np.pi / 180.])
secondWheelRevolutionAngle = np.arccos(1 - (initialParameter[0] / (Rt + rt)) ** 2 * (1 - np.cos(offsetAngle)))

#initialPhi = tau_x(-offsetWheel[0]) @ tau_z(-offsetWheel[2]) @ tau_x(initialParameter[0] - offsetWheel[1] + Rt + rt) \
#             @ rho_z(- 0.5 * np.pi)
#initialSolution = [initialPhi[:3, :3],
#                   initialPhi[:3, 3],
#                   np.array([0.1, secondWheelRevolutionAngle, wheelProfileParameter, np.pi, 0., -0.1])]

#currentPoint = list(initialSolution) + [initialParameter]
#cost(currentPoint)
#costReliefOneCurve(currentPoint)

from Solver.SolverRBFGSPositioning import SolverRBFGS


def FindInitialCondition():
    grindingMark = -0.5 * np.pi - initialParameter[1]
    reliefAngle = 0. * np.pi / 180.

    initialPhi = Helix(trajectoryParameter, initialParameter[0], initialParameter[1], helixLength) \
                 @ rho_x(np.arctan(np.tan(reliefAngle) / np.cos(initialParameter[1])))\
                 @ rho_z(- grindingMark) \
                 @ np.linalg.inv(Tore(0, 0, rt, Rt, offsetWheel))

    initialScalars = np.array([reliefAngle, secondWheelRevolutionAngle, 0., grindingMark, trajectoryParameter, reliefAngle])

    initialGuess = [initialPhi[:3, :3],
                       initialPhi[:3, 3],
                       initialScalars]

    def costForFixedParameter(S):
        A = list(S)
        A.append(initialParameter)

        return cost(A)

    correctorProblem = Problem(solutionSpace, costForFixedParameter)

    corrector = SolverRBFGS(correctorProblem, False)

    return corrector.SearchSolution(initialGuess, np.eye(solutionSpaceDimension))

X0 = FindInitialCondition()

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

from Continuation.PositioningProblem.StepSizeAdaptiveContinuationStraightLine import StepSizeAdaptiveContinuation
from Continuation.PositioningProblem.PathAdaptiveContinuationApproximateLength import PathAdaptiveContinuationApproximateLength
from Continuation.PositioningProblem.PathAdaptiveContinuationSecondOrderApproximation import PathAdaptiveContinuationSecondOrderApproximation

# Instantiate continuation object

continuation = StepSizeAdaptiveContinuation(problem,
                                            initialSolution,
                                            initialParameter,
                                            targetParameter)

continuation3 = PathAdaptiveContinuationApproximateLength(problem,
                                                          initialSolution,
                                                          initialParameter,
                                                          targetParameter)

continuation2 = PathAdaptiveContinuationSecondOrderApproximation(problem,
                                                                initialSolution,
                                                                initialParameter,
                                                                targetParameter)

#results, parameterSpaceMetrics, perturbationMagnitudes, iterations, solved = continuation.Traverse()
#results3, parameterSpaceMetrics3, perturbationMagnitudes3, iterations3, solved3 = continuation3.Traverse()
results2, parameterSpaceMetrics2, perturbationMagnitudes2, iterations2, solved2 = continuation2.Traverse()


