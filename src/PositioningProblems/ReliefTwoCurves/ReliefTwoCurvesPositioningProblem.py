import autograd.numpy as np

from Helpers.MathHelpers import InvSE3
from Helpers.AssembleMatrices import RepresentRectangularOperatorFromSolutionSpaceToSE32, \
    RepresentRectangularOperatorFromParameterSpaceToSE32

from pymanopt.manifolds.rotations import Rotations

from PositioningProblems.PositioningProblem import PositioningProblem
from PositioningProblems.ReliefTwoCurves.Cost import cost
from PositioningProblems.ReliefTwoCurves.Differential import differentialSolution, differentialParameter, systemEvaluation

from SE3Parameterizations.Parameterizations.Torus import torus
from SE3Parameterizations.Parameterizations.Helix import helix

from Helpers.Parameterizations.BasicSE3Transformations import rho_x, rho_z

from Solver.SolverRBFGSPositioning import SolverRBFGS


class ReliefTwoCurvesPositioningProblem(PositioningProblem):

    def __init__(self, solutionSpace, constrainedManifold, parameterSpace, wheelProfileParameter, helixLength, offsetWheel):
        super().__init__(solutionSpace, constrainedManifold, parameterSpace)

        self.wheelProfileParameter = wheelProfileParameter
        self.helixLength = helixLength
        self.offsetWheel = offsetWheel
        self.SO3 = Rotations(3)

    def Cost(self, S):
        return cost(S, self.wheelProfileParameter, self.helixLength, self.offsetWheel)

    def SystemEvaluation(self, S):
        return systemEvaluation(S, self.wheelProfileParameter, self.helixLength, self.offsetWheel)

    def DifferentialSolution(self, S, zeta):
        return differentialSolution(S, zeta, self.wheelProfileParameter, self.helixLength, self.offsetWheel)

    def DifferentialParameter(self, S, v):
        return differentialParameter(S, v, self.wheelProfileParameter, self.helixLength, self.offsetWheel)

    def RepresentDifferentialSolutionInSuitableBasis(self, currentPoint):
        return RepresentRectangularOperatorFromSolutionSpaceToSE32(self.DifferentialSolution,
                                                                   self._solutionSpace,
                                                                   self.ConstrainedManifold,
                                                                   currentPoint,
                                                                   self.SystemEvaluation(currentPoint))

    def RepresentDifferentialParameterInSuitableBasis(self, currentPoint):
        return RepresentRectangularOperatorFromParameterSpaceToSE32(self.DifferentialParameter,
                                                                    self._parameterSpace,
                                                                    self.ConstrainedManifold,
                                                                    currentPoint,
                                                                    self.SystemEvaluation(currentPoint))

    def FindInitialCondition(self, initialParameter):
        Rt, rt, helixAngle1, helixRadius1, helixAngle2, helixRadius2, offsetAngle, trajectoryParameter = initialParameter

        grindingMark = -0.5 * np.pi - helixAngle1

        reliefAngle = 20.
        W1 = torus(0, self.wheelProfileParameter, rt, Rt, self.offsetWheel)
        C1 = helix(trajectoryParameter, helixRadius1, helixAngle1, self.helixLength)

        initialPhi = C1 \
                     @ rho_x(np.arctan(np.tan(reliefAngle) / np.cos(helixAngle1))) \
                     @ rho_z(- grindingMark) \
                     @ InvSE3(W1)

        initialScalars = np.array([reliefAngle, 0., 0, grindingMark, trajectoryParameter, -reliefAngle])

        initialGuess = [initialPhi[:3, :3],
                        initialPhi[:3, 3],
                        initialScalars]

        def costForFixedParameter(S):
            A = list(S)
            A.append(initialParameter)

            return self.Cost(A)

        def Skew(A):
            return 0.5 * (A - A.T)

        def diffProj(x, z, v):
            return self._solutionSpace.proj(x, [x[0] @ Skew(z[0].T @ v[0]) + z[0] @ Skew(x[0].T @ v[0]), np.zeros(x[1].shape),
                                          np.zeros(x[2].shape)])

        def hessForFixedParameter(x, z):
            egrad = correctorProblem.egrad(x)
            ehess = correctorProblem.ehess(x, [self.SO3.tangent2ambient(x[0], z[0]), z[1], z[2]])
            return self._solutionSpace.proj(x, ehess) + diffProj(x, [self.SO3.tangent2ambient(x[0], z[0]), z[1], z[2]], egrad)

        from pymanopt.core.problem import Problem

        correctorProblem = Problem(self._solutionSpace, costForFixedParameter, hess=hessForFixedParameter)

        corrector = SolverRBFGS(correctorProblem, False)

        return corrector.SearchSolution(initialGuess, np.eye(int(self._solutionSpace.dim)))[0]

