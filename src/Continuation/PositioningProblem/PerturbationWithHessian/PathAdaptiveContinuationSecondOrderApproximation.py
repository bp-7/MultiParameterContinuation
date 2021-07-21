import autograd.numpy as np

from Continuation.PositioningProblem.PerturbationWithHessian.PathAdaptiveStepSizeAdaptiveContinuation import PathAdaptiveMultiParameterContinuation
from Helpers.AssembleMatrices import RepresentSquareOperatorInTotalBergerBasis
from Helpers.MathHelpers import Skew

class PathAdaptiveContinuationSecondOrderApproximation(PathAdaptiveMultiParameterContinuation):

    def __init__(self,
                 problem,
                 initialSolution,
                 initialParameter,
                 targetParameter):

        super().__init__(problem, initialSolution, initialParameter, targetParameter)

        self.IsBasisNormalized = False

    def DetermineMetricMatrix(self):

        solutionSpaceDimension = int(self.SolutionSpace.dim)
        hessianParameterMatrix = self.hessianMatrix[solutionSpaceDimension:, solutionSpaceDimension:]

        return hessianParameterMatrix - sym(self.hessianMatrix[solutionSpaceDimension:, :solutionSpaceDimension] @ self.inverseHessianSolutionMatrix @ self.hessianMixteMatrix)

    def DeterminePerturbationsInTangentSpaces(self):
        self._currentParameterPerturbation = self.FinalParameter - self._currentParameter

        self._currentSolutionPerturbation = self.ConvertToTangentVectorOnSolutionSpace(self._currentSolution,
                                                                                       self.DeterminePerturbationInSolutionSpace(
                                                                                           self._currentParameterPerturbation))
        self._currentStepSize = 1.0

        if self._objectiveFunction(
                list(self.SolutionSpace.exp(self._currentSolution, self._currentSolutionPerturbation)) \
                + [self._currentParameter + self._currentParameterPerturbation]) > self.ObjectivePredictionTolerance:

            self._currentStepSize = self.ChooseLargestStepSizeSatisfyingRequirements()
            if np.sum(np.linalg.eigvals(self._parameterSpaceMetricMatrix) > 0) < 2:
                potentialParameterPerturbation = self._currentParameterPerturbation
            else:
                self._perturbationMagnitude = np.sqrt(self.ObjectivePredictionTolerance)
                potentialParameterPerturbation = self.chooseParameterPerturbationOnEllipsoid(
                    self._parameterSpaceMetricMatrix,
                    self._perturbationMagnitude)


            if np.linalg.norm(
                    self._currentParameter + self._currentStepSize * self._currentParameterPerturbation - self.FinalParameter) \
                    > np.linalg.norm(
                    self._currentParameter +  potentialParameterPerturbation - self.FinalParameter):

                potentialSolutionPerturbation = self.ConvertToTangentVectorOnSolutionSpace(self._currentSolution,
                                                                                           self.DeterminePerturbationInSolutionSpace(
                                                                                               potentialParameterPerturbation))

                if self._objectiveFunction(
                        list(self.SolutionSpace.exp(self._currentSolution, potentialSolutionPerturbation)) \
                        + [
                            self._currentParameter + potentialParameterPerturbation]) > self.ObjectivePredictionTolerance:

                    tempParameterPerturbation = self._currentParameterPerturbation
                    tempSolutionPerturbation = self._currentSolutionPerturbation

                    self._currentParameterPerturbation = potentialParameterPerturbation
                    self._currentSolutionPerturbation = potentialSolutionPerturbation

                    newStepSize = self.ChooseLargestStepSizeSatisfyingRequirements()

                    if np.linalg.norm(
                            self._currentParameter + newStepSize * self._currentParameterPerturbation - self.FinalParameter) \
                            <= np.linalg.norm(
                        self._currentParameter + self._currentStepSize * tempParameterPerturbation - self.FinalParameter):
                        self._currentStepSize = newStepSize
                    else:
                        self._currentParameterPerturbation = tempParameterPerturbation
                        self._currentSolutionPerturbation = tempSolutionPerturbation
                else:
                    self._currentParameterPerturbation = potentialParameterPerturbation
                    self._currentSolutionPerturbation = potentialSolutionPerturbation
                    self._currentStepSize = 1.0
            else:
                pass

    def ConvertToTangentVectorOnSolutionSpace(self, currentPoint, vector):
        return self.SolutionSpace.proj(currentPoint, (
            Skew(vector[:3]) , np.array(vector[3:6]), np.array(vector[6:])))

    def ConvertToTangentVector(self, currentPoint, vector):
        return self.ProductManifold.proj(currentPoint, (
            Skew(vector[:3]) , np.array(vector[3:6]), np.array(vector[6:9]), np.array(vector[9:])))

    def ExpressHessianMatrixInSuitableBasis(self):
        return RepresentSquareOperatorInTotalBergerBasis(self._hessian,
                                                           self.ProductManifold,
                                                           self.currentPoint)