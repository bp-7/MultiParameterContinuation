import autograd.numpy as np

from Continuation.PositioningProblem.PerturbationWithHessian.PathAdaptiveStepSizeAdaptiveContinuation import PathAdaptiveMultiParameterContinuation
from Helpers.AssembleMatrices import RepresentSquareOperatorInTotalBergerBasis
from Helpers.MathHelpers import Skew

class PathAdaptiveContinuationApproximateLength(PathAdaptiveMultiParameterContinuation):
    SE3Dimension = 6

    def __init__(self,
                 problem,
                 initialSolution,
                 initialParameter,
                 targetParameter,
                 objectiveFunctionTolerance):

        super().__init__(problem, initialSolution, initialParameter, targetParameter, objectiveFunctionTolerance)

        self.IsBasisNormalized = False
        self._solutionSpaceMetricMatrix = np.diag(np.concatenate((np.array([2., 2., 2., 1., 1., 1.]),
                                                   np.ones(int(self.SolutionSpace.dim) - self.SE3Dimension))))

    def DetermineMetricMatrix(self):
        temp = self.inverseHessianSolutionMatrix @ self.hessianMixteMatrix

        return  temp.T @ self._solutionSpaceMetricMatrix @ temp

    def DeterminePerturbationsInTangentSpaces(self):
        self._currentParameterPerturbation = self.FinalParameter - self._currentParameter

        self._currentSolutionPerturbation = self.ConvertToTangentVectorOnSolutionSpace(self._currentSolution,
                                                                                       self.DeterminePerturbationInSolutionSpace(
                                                                                           self._currentParameterPerturbation))
        self._currentStepSize = 1.0

        self.alternativeDirectionTaken = False

        if self._objectiveFunction(
                tuple(self.SolutionSpace.exp(self._currentSolution, self._currentSolutionPerturbation)) \
                + (
                self._currentParameter + self._currentParameterPerturbation,)) > self.ObjectivePredictionTolerance:

            potentialStepSize = self.ChooseLargestStepSizeSatisfyingRequirements()

            if potentialStepSize < self.MinimumStepSize:
                potentialStepSize = self.MinimumStepSize

            straightLineMethodTolerance = self._objectiveFunction(
                tuple(self.SolutionSpace.exp(self._currentSolution,
                                             potentialStepSize * self._currentSolutionPerturbation)) \
                + (self._currentParameter + potentialStepSize * self._currentParameterPerturbation,))

            print("StraightLineTolerance = " + str(straightLineMethodTolerance) + "\n")

            self.currentParameterPerturbationNorm = potentialStepSize * np.linalg.norm(
                self._currentParameterPerturbation)

            self._perturbationMagnitude = potentialStepSize \
                                          * self.SolutionSpace.norm(self._currentSolution,
                                                                    self._currentSolutionPerturbation)

            potentialParameterPerturbation = self.chooseParameterPerturbationOnEllipsoid(
                self._parameterSpaceMetricMatrix,
                self._perturbationMagnitude)

            potentialSolutionPerturbation = self.ConvertToTangentVectorOnSolutionSpace(
                self._currentSolution,
                self.DeterminePerturbationInSolutionSpace(potentialParameterPerturbation))

            ellipsoidMethodTolerance = self._objectiveFunction(
                tuple(self.SolutionSpace.exp(self._currentSolution, potentialSolutionPerturbation)) \
                + (self._currentParameter + potentialParameterPerturbation,))

            if ellipsoidMethodTolerance > self.ObjectivePredictionTolerance:
                # or ellipsoidMethodTolerance > straightLineMethodTolerance:

                potentialPerturbationStepSize = self.ChoosePotentialLargestStepSizeSatisfyingRequirements(
                    potentialParameterPerturbation,
                    potentialSolutionPerturbation,
                    self.currentParameterPerturbationNorm,
                    straightLineMethodTolerance
                )

                print("ellipsoidTolerance = " + str(self._objectiveFunction(
                    tuple(self.SolutionSpace.exp(self._currentSolution,
                                                 potentialPerturbationStepSize * potentialSolutionPerturbation)) \
                    + (
                    self._currentParameter + potentialPerturbationStepSize * potentialParameterPerturbation,))) + "\n")
                print("straightLineTolerance = " + str(straightLineMethodTolerance) + "\n\n\n")

                if potentialPerturbationStepSize < 1e-14:
                    self._currentStepSize = potentialStepSize
                else:
                    self._currentStepSize = potentialPerturbationStepSize
                    self._currentParameterPerturbation = potentialParameterPerturbation
                    self._currentSolutionPerturbation = potentialSolutionPerturbation
                    self.alternativeDirectionTaken = True

                # if potentialPerturbationStepSize is None:
                #     self._currentStepSize = potentialStepSize

            else:
                self._currentParameterPerturbation = potentialParameterPerturbation
                self._currentSolutionPerturbation = potentialSolutionPerturbation
                self._currentStepSize = 1.0
                self.alternativeDirectionTaken = True

    def ChoosePotentialLargestStepSizeSatisfyingRequirements(self,
                                                             potentialParameterPerturbation,
                                                             potentialSolutionPerturbation,
                                                             currentPerturbationNorm,
                                                             straightLineMethodTolerance):
        print("----- Step Size Selection-----\n")
        print("Initial stepSize = " + str(self._currentStepSize) + "\n")

        minTolerance = np.min([np.inf, self.ObjectivePredictionTolerance])

        lowBound, highBound = 0.0, 1.0
        newStepSize = 1.0

        potentialPerturbationNorm = np.linalg.norm(potentialParameterPerturbation)

        def potentialSolutionCurvePoint(newStepSize):
            return list(self.SolutionSpace.exp(self._currentSolution, newStepSize * potentialSolutionPerturbation)) + \
                   [self._currentParameter + newStepSize * potentialParameterPerturbation]

        def potentialObjectiveFunctionValue(newStepSize):
            return self._objectiveFunction(potentialSolutionCurvePoint(newStepSize))

        def potentialGradientNorm(newStepSize):
            return self.ProductManifold.norm(potentialSolutionCurvePoint(newStepSize),
                                           self._gradient(potentialSolutionCurvePoint(newStepSize)))

        print("NextObj = " + str(potentialObjectiveFunctionValue(newStepSize)) + "\n")
        print("NextGrad = " + str(potentialGradientNorm(newStepSize)) + "\n")

        if not potentialObjectiveFunctionValue(newStepSize) < minTolerance:
               # and potentialGradientNorm(newStepSize) < self.GradientNormPredictionTolerance):

            while newStepSize > self.MinimumStepSize \
                    and potentialObjectiveFunctionValue(newStepSize) > minTolerance:
                    #and newStepSize * potentialPerturbationNorm > currentPerturbationNorm:

                newStepSize = 0.95 * newStepSize

                print("\nPotential new step size = " + str(newStepSize) + "\n")
                print("NextObj = " + str(potentialObjectiveFunctionValue(newStepSize)) + "\n")
                print("NextGrad = " + str(potentialGradientNorm(newStepSize)) + "\n")

        return newStepSize

    def ConvertToTangentVectorOnSolutionSpace(self, currentPoint, vector):
        return self.SolutionSpace.proj(currentPoint, (
            currentPoint[0] @ Skew(vector[:3]), np.array(vector[3:6]), np.array(vector[6:])))

    def ConvertToTangentVector(self, currentPoint, vector):
        return self.ProductManifold.proj(currentPoint, (
            currentPoint[0] @ Skew(vector[:3]), np.array(vector[3:6]), np.array(vector[6:9]), np.array(vector[9:])))

    def ExpressHessianMatrixInSuitableBasis(self):
        return RepresentSquareOperatorInTotalBergerBasis(self._hessian,
                                                         self.ProductManifold,
                                                         self.currentPoint)