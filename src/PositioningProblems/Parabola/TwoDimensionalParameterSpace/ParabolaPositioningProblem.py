from Helpers.AssembleMatrices import RepresentRectangularOperatorFromSolutionSpaceToSE32, \
    RepresentRectangularOperatorFromParameterSpaceToSE32

from PositioningProblems.PositioningProblem import PositioningProblem
from PositioningProblems.Parabola.TwoDimensionalParameterSpace.Cost import cost
from PositioningProblems.Parabola.TwoDimensionalParameterSpace.Differential import differentialSolution, differentialParameter, systemEvaluation

class ParabolaPositioningProblem(PositioningProblem):

    def __init__(self, solutionSpace, constrainedManifold, parameterSpace, distanceBetweenContactPoints):
        super().__init__(solutionSpace, constrainedManifold, parameterSpace)

        self.distanceBetweenContactPoints = distanceBetweenContactPoints


    def Cost(self, S):
        return cost(S, self.distanceBetweenContactPoints)


    def SystemEvaluation(self, S):
        return systemEvaluation(S, self.distanceBetweenContactPoints)

    def DifferentialSolution(self, S, zeta):
        return differentialSolution(S, zeta, self.distanceBetweenContactPoints)

    def DifferentialParameter(self, S, v):
        return differentialParameter(S, v, self.distanceBetweenContactPoints)


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
