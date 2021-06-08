import autograd.numpy as np

from Continuation.PositioningProblem.PathAdaptiveStepSizeAdaptiveContinuation import PathAdaptiveMultiParameterContinuation
from Continuation.Helpers.AssembleMatrices import RepresentSquareOperatorInTotalBergerBasis
from Continuation.Helpers.MathHelpers import Skew

class PathAdaptiveContinuationApproximateLength(PathAdaptiveMultiParameterContinuation):

    def __init__(self,
                 problem,
                 initialSolution,
                 initialParameter,
                 targetParameter):

        super().__init__(problem, initialSolution, initialParameter, targetParameter)

        self.IsBasisNormalized = False
        self._solutionSpaceMetricMatrix = np.diag(np.array([2., 2., 2., 1., 1., 1., 1., 1., 1.]))

    def DetermineMetricMatrix(self):
        temp = self.inverseHessianSolutionMatrix @ self.hessianMixteMatrix

        return  temp.T @ self._solutionSpaceMetricMatrix @ temp

    def DeterminePerturbationsInTangentSpaces(self):
        self._currentParameterPerturbation = self.FinalParameter - self._currentParameter

        self._currentSolutionPerturbation = self.ConvertToTangentVectorOnSolutionSpace(self._currentSolution,
                                                                                       self.DeterminePerturbationInSolutionSpace(
                                                                                           self._currentParameterPerturbation))
        self._currentStepSize = 1.0

        if self._objectiveFunction(
                tuple(self.SolutionSpace.exp(self._currentSolution, self._currentSolutionPerturbation)) \
                + (self._currentParameter + self._currentParameterPerturbation,)) > self.ObjectivePredictionTolerance:

            potentialStepSize = self.ChooseLargestStepSizeSatisfyingRequirements()
            self._perturbationMagnitude = potentialStepSize \
                                          * self.SolutionSpace.norm(self._currentSolution,
                                                                    self._currentSolutionPerturbation)

            potentialParameterPerturbation = self.chooseParameterPerturbationOnEllipsoid(
                self._parameterSpaceMetricMatrix, self._perturbationMagnitude)
            potentialSolutionPerturbation = self.ConvertToTangentVectorOnSolutionSpace(self._currentSolution,
                                                                                       self.DeterminePerturbationInSolutionSpace(
                                                                                           potentialParameterPerturbation))

            if self._objectiveFunction(
                    tuple(self.SolutionSpace.exp(self._currentSolution, potentialSolutionPerturbation)) \
                    + (self._currentParameter + potentialParameterPerturbation,)) > self.ObjectivePredictionTolerance:

                tempParameterPerturbation = self._currentParameterPerturbation
                tempSolutionPerturbation = self._currentSolutionPerturbation
                tempStepSize = potentialStepSize

                self._currentParameterPerturbation = potentialParameterPerturbation
                self._currentSolutionPerturbation = potentialSolutionPerturbation

                newStepSize = self.ChooseLargestStepSizeSatisfyingRequirements()

                if np.linalg.norm(
                        self._currentParameter + newStepSize * self._currentParameterPerturbation - self.FinalParameter) \
                        <= np.linalg.norm(
                    self._currentParameter + tempStepSize * tempParameterPerturbation - self.FinalParameter):
                    self._currentStepSize = newStepSize
                else:
                    self._currentParameterPerturbation = tempParameterPerturbation
                    self._currentSolutionPerturbation = tempSolutionPerturbation
                    self._currentStepSize = tempStepSize
            else:
                self._currentParameterPerturbation = potentialParameterPerturbation
                self._currentSolutionPerturbation = potentialSolutionPerturbation
                self._currentStepSize = 1.0

    def ConvertToTangentVectorOnSolutionSpace(self, currentPoint, vector):
        return self.SolutionSpace.proj(currentPoint, (
            Skew(vector[:3]), np.array(vector[3:6]), np.array(vector[6:])))

    def ConvertToTangentVector(self, currentPoint, vector):
        return self.ProductManifold.proj(currentPoint, (
            Skew(vector[:3]), np.array(vector[3:6]), np.array(vector[6:9]), np.array(vector[9:])))

    def ExpressHessianMatrixInSuitableBasis(self):
        return RepresentSquareOperatorInTotalBergerBasis(self._hessian,
                                                         self.ProductManifold,
                                                         self.currentPoint)