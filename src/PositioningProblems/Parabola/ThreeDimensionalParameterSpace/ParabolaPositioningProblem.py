from Helpers.AssembleMatrices import RepresentRectangularOperatorFromSolutionSpaceToSE32, \
    RepresentRectangularOperatorFromParameterSpaceToSE32

from PositioningProblems.PositioningProblem import PositioningProblem
from PositioningProblems.Parabola.ThreeDimensionalParameterSpace.Cost import cost
from PositioningProblems.Parabola.ThreeDimensionalParameterSpace.Differential import differentialSolution, differentialParameter, systemEvaluation

class ParabolaPositioningProblem(PositioningProblem):

    def __init__(self, solutionSpace, constrainedManifold, parameterSpace):
        super().__init__(solutionSpace, constrainedManifold, parameterSpace)

    def Cost(self, S):
        return cost(S)

    def SystemEvaluation(self, S):
        return systemEvaluation(S)

    def DifferentialSolution(self, S, zeta):
        return differentialSolution(S, zeta)

    def DifferentialParameter(self, S, v):
        return differentialParameter(S, v)

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
