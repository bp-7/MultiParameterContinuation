import autograd.numpy as np

from Continuation.PositioningProblem.ContinuationPositioning import AbstractContinuationPositioning
from Helpers.AssembleMatrices import RepresentRectangularOperatorFromSolutionSpaceToSE32, \
    RepresentRectangularOperatorFromParameterSpaceToSE32
from Helpers.MathHelpers import Skew

from pymanopt.manifolds.euclidean import Euclidean
from pymanopt.core.problem import Problem
from pymanopt.manifolds import Sphere
from pymanopt.solvers import SteepestDescent

from Solver.SolverRBFGSPositioning import SolverRBFGS


class LinearMultiParameterContinuation(AbstractContinuationPositioning):
    StepSizeContraction = 0.75
    InitialStepSize = 0.1
    MaximumStepSize = 1
    MaximumNumberOfContinuationSteps = 15
    MinimumStepSize = 1e-2#1.0 / MaximumNumberOfContinuationSteps
    SE3Dimension = 6

    def __init__(self,
                 problem,
                 positioningProblem,
                 initialSolution,
                 initialParameter,
                 targetParameter,
                 ObjectivePredictionTolerance):

        super().__init__(problem, initialSolution, initialParameter, targetParameter, ObjectivePredictionTolerance)

        self._currentStepSize = self.InitialStepSize

        self.currentPoint = list(self._currentSolution) + [self._currentParameter]

        self.InitialParameter = initialParameter
        self._currentParameterPerturbation = None
        self._currentSolutionPerturbation = None
        self.differentialSolutionMatrix = None
        self.differentialParameterMatrix = None
        self.differentialSolutionPInv = None

        self.differentialSolution = positioningProblem.DifferentialSolution
        self.differentialParameter = positioningProblem.DifferentialParameter
        self.F = positioningProblem.SystemEvaluation
        self.FCodomain = positioningProblem.ConstrainedManifold
        self.parameterSpace = Euclidean(targetParameter.shape)
        self._solutionSpaceMetricMatrix = np.diag(np.concatenate((np.array([2., 2., 2., 1., 1., 1.]),
                                                                  np.ones(int(
                                                                      self.SolutionSpace.dim) - self.SE3Dimension))))

    def GetNextParameter(self):
        return self.parameterSpaceCurve(self._currentContinuationArgument + self._currentStepSize)

    def GetNextApproximate(self):
        return self.SolutionSpace.exp(self._currentSolution, self._currentStepSize * self._currentSolutionPerturbation)

    def parameterSpaceCurve(self, tau):
        return (1 - tau) * self.InitialParameter + tau * self.FinalParameter

    def GetNextContinuationArgument(self):

        self.currentPoint = list(self._currentSolution) + [self._currentParameter]
        FPoint = self.F(self.currentPoint)

        self.differentialSolutionMatrix = RepresentRectangularOperatorFromSolutionSpaceToSE32(
            self.differentialSolution,
            self.SolutionSpace,
            self.FCodomain,
            self.currentPoint,
            FPoint)

        self.differentialSolutionPInv = np.linalg.pinv(self.differentialSolutionMatrix)

        self.differentialParameterMatrix = RepresentRectangularOperatorFromParameterSpaceToSE32(
            self.differentialParameter,
            self.parameterSpace,
            self.FCodomain,
            self.currentPoint,
            FPoint)

        self.DeterminePerturbationsInTangentSpaces()

        potentialNewStepSize = self.ChooseLargestStepSizeSatisfyingRequirements()

        if potentialNewStepSize < self.MinimumStepSize:
            if self._objectiveFunction(
                    tuple(self.SolutionSpace.exp(self._currentSolution, self.MinimumStepSize * self._currentSolutionPerturbation)) \
                    + (
                    self._currentParameter + self.MinimumStepSize * self._currentParameterPerturbation,)) > self.ObjectivePredictionTolerance:
                return None
            else:
                potentialNewStepSize = self.MinimumStepSize

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
                    newStepSize = lowBound

        return newStepSize

    def DeterminePerturbationInSolutionSpace(self, perturbationInParameterSpace):
        return - self.differentialSolutionPInv \
               @ self.differentialParameterMatrix \
               @ perturbationInParameterSpace

    def DoSomethingUponFailureOrAcceptFailure(self):
        print("############### ULTIMATUM ###############\n")
        totalRBFGSIters = 0
        iterUltimatum = 0
        while self._currentStepSize * self.StepSizeContraction > self.MinimumStepSize and iterUltimatum < 7:
            iterUltimatum = iterUltimatum + 1
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
            if self._objectiveFunction(list(potentialNextSolution) + [potentialNextParameter]) <= 1e-9:
                print("######## ULTIMATUM SUCCESS ########\n")

                return potentialNextContinuationArgument, tuple(potentialNextSolution) + (
                potentialNextParameter,), totalRBFGSIters

        print("### ULTIMATUM FAILED : MIN STEP SIZE REACHED ###\n")

        return None

    def chooseParameterPerturbationOnEllipsoid(self, metric, magnitude):
        # take Cholesky factor
        choleskyFactor = np.linalg.cholesky(metric)
        inverseCholeskyFactor = np.linalg.inv(choleskyFactor)

        deltaParameter = self.FinalParameter - self._currentParameter

        if np.inner(deltaParameter, metric @ deltaParameter) < magnitude ** 2:
            perturbationMagnitude = np.sqrt(np.inner(deltaParameter, metric @ deltaParameter))
        else:
            perturbationMagnitude = magnitude

        def cost(y):
            u = perturbationMagnitude * inverseCholeskyFactor.T @ y - deltaParameter

            return 0.5 * np.inner(u, u)

        unitSphereDimension = len(self._currentParameter)
        unitSphere = Sphere(unitSphereDimension)

        # Instantiate a minimization problem
        problem = Problem(manifold=unitSphere, cost=cost)

        # Instantiate a Pymanopt solver
        solver = SteepestDescent()

        xStart = (1.0 / perturbationMagnitude) * choleskyFactor.T @ deltaParameter
        xStartNormalized = xStart / np.linalg.norm(xStart)

        if unitSphere.norm(xStartNormalized, problem.grad(xStartNormalized)) <= 1e-15:
            return perturbationMagnitude * inverseCholeskyFactor.T @ xStartNormalized

        v = solver.solve(problem, x=xStartNormalized)

        print("v_steepest = " + str(v) + "\n")

        newPerturbation = perturbationMagnitude * inverseCholeskyFactor.T @ v

        print("\n New perturbation : " + str(newPerturbation) + "\n")

        return newPerturbation

    def DeterminePerturbationsInTangentSpaces(self):
        self._currentParameterPerturbation = self.FinalParameter - self._currentParameter

        self._currentSolutionPerturbation = self.ConvertToTangentVectorOnSolutionSpace(self._currentSolution,
                                                                                       self.DeterminePerturbationInSolutionSpace(
                                                                                           self._currentParameterPerturbation))

    def ConvertToTangentVectorOnSolutionSpace(self, currentPoint, vector):
        return self.SolutionSpace.proj(currentPoint, (
            currentPoint[0] @ Skew(vector[:3]), np.array(vector[3:6]), np.array(vector[6:])))

    def ConvertToTangentVector(self, currentPoint, vector):
        return self.ProductManifold.proj(currentPoint, (
            currentPoint[0] @ Skew(vector[:3]), np.array(vector[3:6]), np.array(vector[6:9]), np.array(vector[9:])))
