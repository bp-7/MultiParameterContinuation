import autograd.numpy as np

from pymanopt.core.problem import Problem

from Helpers.AssembleMatrices import RepresentSquareOperatorInTotalBergerBasis

from Continuation.PositioningProblem.ContinuationPositioning import AbstractContinuationPositioning
from Helpers.MathHelpers import Skew

from Solver.SolverRBFGSPositioning import SolverRBFGS



class StepSizeAdaptiveContinuation(AbstractContinuationPositioning):
    StepSizeContraction = 0.75
    InitialStepSize = 0.1
    MaximumStepSize = 1
    MaximumNumberOfContinuationSteps = 150
    MinimumStepSize = 1.0 / MaximumNumberOfContinuationSteps

    def __init__(self,
                 problem,
                 initialSolution,
                 initialParameter,
                 targetParameter):

        super().__init__(problem, initialSolution, initialParameter, targetParameter)

        self._currentStepSize = self.InitialStepSize

        self.InitialParameter = initialParameter

        self._currentPoint = list(self._currentSolution) + [self._currentParameter]

        self._currentParameterPerturbation = None
        self._currentSolutionPerturbation = None

        self.hessianMatrix = None
        self.hessianSolutionMatrix = None
        self.hessianMixteMatrix = None
        self.inverseHessianSolutionMatrix = None

        self._modifiedPerturbationMagnitude = None

    def GetNextParameter(self):
        return self.parameterSpaceCurve(self._currentContinuationArgument + self._currentStepSize)

    def GetNextApproximate(self):
        return self.SolutionSpace.exp(self._currentSolution, self._currentStepSize * self._currentSolutionPerturbation)

    def GetNextContinuationArgument(self):

        self._currentPoint = list(self._currentSolution) + [self._currentParameter]

        self.hessianMatrix = RepresentSquareOperatorInTotalBergerBasis(self._hessian,
                                                                  self.ProductManifold,
                                                                  self._currentPoint)
        solutionSpaceDimension = int(self.SolutionSpace.dim)

        self.hessianSolutionMatrix = self.hessianMatrix[:solutionSpaceDimension, :solutionSpaceDimension]

        self.hessianMixteMatrix = self.hessianMatrix[:solutionSpaceDimension, solutionSpaceDimension:]

        self.inverseHessianSolutionMatrix = np.linalg.inv(self.hessianSolutionMatrix)



        self._currentParameterPerturbation = self.FinalParameter - self._currentParameter

        self._currentSolutionPerturbation = self.ConvertToTangentVectorOnSolutionSpace(self._currentSolution,
                                                                                       self.DeterminePerturbationInSolutionSpace(
                                                                                           self._currentParameterPerturbation))

        potentialNewStepSize = self.ChooseLargestStepSizeSatisfyingRequirements()

        if self._currentContinuationArgument + potentialNewStepSize - 1 > - self.MinimumStepSize:
            self._currentStepSize = 1 - self._currentContinuationArgument
            print(" --> End of continuation exceeded")
            print("\nNext step size = " + str(self._currentStepSize) + "\n\n")

            return 1

        self._currentStepSize = potentialNewStepSize
        print("\nNext step size = " + str(self._currentStepSize) + "\n\n")

        return self._currentContinuationArgument + self._currentStepSize

    def ChooseLargestStepSizeSatisfyingRequirements(self):
        print("----- Step Size Selection-----\n")
        print("Initial stepSize = " + str(self._currentStepSize) + "\n")

        lowBound, highBound = 0.0, 1.0
        newStepSize = 1.0

        def potentialSolutionCurvePoint(newStepSize):
            return list(
                self.SolutionSpace.exp(self._currentSolution, newStepSize * self._currentSolutionPerturbation)) + \
                   [self._currentParameter + newStepSize * self._currentParameterPerturbation]

        def potentialObjectiveFunctionValue(newStepSize):
            return self._objectiveFunction(potentialSolutionCurvePoint(newStepSize))

        def potentialGradientNorm(newStepSize):
            return self.ProductManifold.norm(potentialSolutionCurvePoint(newStepSize),
                                             self._gradient(potentialSolutionCurvePoint(newStepSize)))

        print("NextObj = " + str(potentialObjectiveFunctionValue(newStepSize)) + "\n")
        print("NextGrad = " + str(potentialGradientNorm(newStepSize)) + "\n")

        if not (potentialObjectiveFunctionValue(newStepSize) < self.ObjectivePredictionTolerance
                and potentialGradientNorm(newStepSize) < self.GradientNormPredictionTolerance):

            while np.abs(highBound - lowBound) >= 2 * self.MinimumStepSize:
                newStepSize = (lowBound + highBound) / 2

                print("\nPotential new step size = " + str(newStepSize) + "\n")
                print("NextObj = " + str(potentialObjectiveFunctionValue(newStepSize)) + "\n")
                print("NextGrad = " + str(potentialGradientNorm(newStepSize)) + "\n")

                if (potentialObjectiveFunctionValue(newStepSize) < self.ObjectivePredictionTolerance
                        and potentialGradientNorm(newStepSize) < self.GradientNormPredictionTolerance):
                    lowBound = newStepSize
                else:
                    highBound = newStepSize

        return newStepSize

    def DoSomethingUponFailureOrAcceptFailure(self):
        print("############### ULTIMATUM ###############\n")
        totalRBFGSIters = 0

        while self._currentStepSize * self.StepSizeContraction > self.MinimumStepSize:

            self._currentStepSize *= self.StepSizeContraction
            print("\nStep size = " + str(self._currentStepSize) + "\n")
            potentialNextContinuationArgument = self._currentContinuationArgument + self._currentStepSize
            potentialNextParameter = self.GetNextParameter()
            potentialNextApproximate = self.GetNextApproximate()

            def costForFixedParameter(S):
                A = list(S)
                A.append(potentialNextParameter)

                return self._objectiveFunction(A)

            correctorProblem = Problem(self.SolutionSpace, costForFixedParameter)

            corrector = SolverRBFGS(correctorProblem)

            (potentialNextSolution, self._approximateInverseHessian, RBFGSIters) = corrector.SearchSolution(
                potentialNextApproximate, self._approximateInverseHessian)

            totalRBFGSIters += RBFGSIters
            if self._objectiveFunction(tuple(potentialNextSolution) + (potentialNextParameter,)) <= 1e-7:
                print("######## ULTIMATUM SUCCESS ########\n")
                self._currentContinuationArgument = potentialNextContinuationArgument

                return potentialNextContinuationArgument, tuple(potentialNextSolution) + (
                potentialNextParameter,), totalRBFGSIters

        print("### ULTIMATUM FAILED : MIN STEP SIZE REACHED ###\n")

        return None

    def DeterminePerturbationInSolutionSpace(self, perturbationInParameterSpace):
        return - self.inverseHessianSolutionMatrix \
               @ self.hessianMixteMatrix \
               @ perturbationInParameterSpace

    def parameterSpaceCurve(self, tau):
        return (1 - tau) * self.InitialParameter + tau * self.FinalParameter

    def ConvertToTangentVectorOnSolutionSpace(self, currentPoint, vector):
        return self.SolutionSpace.proj(currentPoint, [Skew(vector[:3]), np.array(vector[3:6]), np.array(vector[6:])])

