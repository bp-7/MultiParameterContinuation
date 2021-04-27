import autograd.numpy as np

from pymanopt.core.problem import Problem

from Continuation.Helpers.AssembleMatrices import RepresentSquareOperatorInBergerBasis, \
    ConstructMetricMatrixForBergerManifold, \
    RepresentRectangularOperatorFromEuclideanToBergerManifold

from Continuation.StandardContinuation.Continuation import AbstractContinuationPositioning
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

        self._currentParameterPerturbation = None
        self._currentSolutionPerturbation = None

        # For the moment, but need to change
        self._solutionSpaceMetricMatrix = ConstructMetricMatrixForBergerManifold(int(self.SolutionSpace.dim) - 6)

        self._modifiedPerturbationMagnitude = None

    def GetNextParameter(self):
        return self.parameterSpaceCurve(self._currentContinuationArgument + self._currentStepSize)

    def GetNextApproximate(self):
        return self.SolutionSpace.retr(self._currentSolution, self._currentStepSize * self._currentSolutionPerturbation)

    def GetNextContinuationArgument(self):

        def differentialSolution(S, xi):
            modifiedTangentVector = list(xi)
            modifiedTangentVector.append(np.zeros(2))

            return self._hessian(S, modifiedTangentVector)[:3]

        def differentialParameter(S, v):
            modifiedTangentVector = list(self.SolutionSpace.zerovec(S[:3]))
            modifiedTangentVector.append(v)

            return self._hessian(S, modifiedTangentVector)[:3]

        differentialSolutionMatrix = RepresentSquareOperatorInBergerBasis(differentialSolution,
                                                                          self.SolutionSpace,
                                                                          tuple(self._currentSolution) + (self._currentParameter,))

        differentialParameterMatrix = RepresentRectangularOperatorFromEuclideanToBergerManifold(differentialParameter,
                                                                                                self.SolutionSpace,
                                                                                                tuple(self._currentSolution) + (self._currentParameter,),
                                                                                                len(self._currentParameter))

        inverseDifferentialSolutionMatrix = np.linalg.inv(differentialSolutionMatrix)

        self._currentParameterPerturbation = self.FinalParameter - self._currentParameter

        self._currentSolutionPerturbation = - self.ConvertToTangentVectorOnSolutionSpace(self._currentSolution, inverseDifferentialSolutionMatrix \
                                                                                       @ differentialParameterMatrix \
                                                                                       @ self._currentParameterPerturbation)

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

        newStepSize = self._currentStepSize

        def potentialSolutionCurvePoint(newStepSize):
            return tuple(
                self.SolutionSpace.retr(self._currentSolution, newStepSize * self._currentSolutionPerturbation)) + \
                   (self.parameterSpaceCurve(self._currentContinuationArgument + newStepSize),)

        def potentialObjectiveFunctionValue(newStepSize):
            return self._objectiveFunction(potentialSolutionCurvePoint(newStepSize))

        def potentialGradientNorm(newStepSize):
            return self.ProductManifold.norm(potentialSolutionCurvePoint(newStepSize),
                                             self._gradient(potentialSolutionCurvePoint(newStepSize)))

        print("NextObj = " + str(potentialObjectiveFunctionValue(newStepSize)) + "\n")
        print("NextGrad = " + str(potentialGradientNorm(newStepSize)) + "\n")

        if (potentialObjectiveFunctionValue(newStepSize) < self.ObjectivePredictionTolerance
                and potentialGradientNorm(newStepSize) < self.GradientNormPredictionTolerance):

            while (potentialObjectiveFunctionValue(newStepSize) < self.ObjectivePredictionTolerance
                   and potentialGradientNorm(newStepSize) < self.GradientNormPredictionTolerance
                   and newStepSize < self.MaximumStepSize):
                newStepSize = newStepSize / self.StepSizeContraction

                print("\nPotential new step size = " + str(newStepSize) + "\n")
                print("NextObj = " + str(potentialObjectiveFunctionValue(newStepSize)) + "\n")
                print("NextGrad = " + str(potentialGradientNorm(newStepSize)) + "\n")

                # if DEBUG
            if potentialObjectiveFunctionValue(newStepSize) >= self.ObjectivePredictionTolerance:
                print(" --> Objective function tolerance exceeded.\n")

            if potentialGradientNorm(newStepSize) >= self.GradientNormPredictionTolerance:
                print(" --> Gradient norm tolerance exceeded.\n")

            if newStepSize > self.MaximumStepSize:
                print(" --> Maximum step size exceeded.\n")

            # endif
            newStepSize *= self.StepSizeContraction

        else:
            while (not (potentialObjectiveFunctionValue(newStepSize) < self.ObjectivePredictionTolerance
                        and potentialGradientNorm(newStepSize) < self.GradientNormPredictionTolerance)
                   and newStepSize > self.MinimumStepSize):
                newStepSize *= self.StepSizeContraction
                print("\nPotential new step size = " + str(newStepSize) + "\n")
                print("NextObj = " + str(potentialObjectiveFunctionValue(newStepSize)) + "\n")
                print("NextGrad = " + str(potentialGradientNorm(newStepSize)) + "\n")

            if newStepSize < self.MinimumStepSize:
                print(" --> Minimum step size exceeded.\n")
                newStepSize /= self.StepSizeContraction

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

            #from pymanopt.solvers.steepest_descent import SteepestDescent
            #corrector = SteepestDescent()

            #potentialNextSolution = corrector.solve(correctorProblem, x=potentialNextApproximate)

            corrector = SolverRBFGS(correctorProblem)

            (potentialNextSolution, self._approximateInverseHessian, RBFGSIters) = corrector.SearchSolution(
                potentialNextApproximate, self._approximateInverseHessian)

            totalRBFGSIters += RBFGSIters
            if self._objectiveFunction(tuple(potentialNextSolution) + (potentialNextParameter,)) <= 1e-7:
                print("######## ULTIMATUM SUCCESS ########\n")
                self._currentContinuationArgument = potentialNextContinuationArgument

                return potentialNextContinuationArgument, tuple(potentialNextSolution) +  (potentialNextParameter, ), totalRBFGSIters

        print("### ULTIMATUM FAILED : MIN STEP SIZE REACHED ###\n")

        return None

    def parameterSpaceCurve(self, tau):
        return (1 - tau) * self.InitialParameter + tau * self.FinalParameter

    def ConvertToTangentVectorOnSolutionSpace(self, currentPoint, vector):
        return self.SolutionSpace.proj(currentPoint, (
            self.Skew(vector[:3]), np.array(vector[3:6]), np.array(vector[6:])))

    def Skew(self, w):
        return np.array([[0., -w[2], w[1]], [w[2], 0., -w[0]], [-w[1], w[0], 0.]])
