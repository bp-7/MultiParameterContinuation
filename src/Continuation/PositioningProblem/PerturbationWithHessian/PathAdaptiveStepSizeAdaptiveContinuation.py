import autograd.numpy as np

from Continuation.PositioningProblem.ContinuationPositioning import AbstractContinuationPositioning

from pymanopt.core.problem import Problem
from pymanopt.manifolds import Sphere
from pymanopt.solvers import SteepestDescent

from Solver.SolverRBFGSPositioning import SolverRBFGS


class PathAdaptiveMultiParameterContinuation(AbstractContinuationPositioning):
    StepSizeContraction = 0.75
    InitialStepSize = 0.1
    MaximumStepSize = 1
    MaximumNumberOfContinuationSteps = 50
    MinimumStepSize = 1.0 / MaximumNumberOfContinuationSteps

    def __init__(self,
                 problem,
                 initialSolution,
                 initialParameter,
                 targetParameter,
                 objectiveFunctionTolerance):

        super().__init__(problem, initialSolution, initialParameter, targetParameter, objectiveFunctionTolerance)

        self._currentStepSize = self.InitialStepSize

        self.currentPoint = list(self._currentSolution) + [self._currentParameter]

        self._currentParameterPerturbation = None
        self._currentSolutionPerturbation = None

        self.hessianMatrix = None
        self.hessianSolutionMatrix = None
        self.hessianMixteMatrix = None
        self.inverseHessianSolutionMatrix = None

    def GetNextParameter(self):
        return self._currentParameter + self._currentStepSize * self._currentParameterPerturbation

    def GetNextApproximate(self):
        return self.SolutionSpace.exp(self._currentSolution, self._currentStepSize * self._currentSolutionPerturbation)

    def GetNextContinuationArgument(self):

        self.currentPoint = list(self._currentSolution) + [self._currentParameter]

        self.hessianMatrix = self.ExpressHessianMatrixInSuitableBasis()

        solutionSpaceDimension = int(self.SolutionSpace.dim)

        self.hessianSolutionMatrix = self.hessianMatrix[:solutionSpaceDimension, :solutionSpaceDimension]

        self.hessianMixteMatrix = self.hessianMatrix[:solutionSpaceDimension, solutionSpaceDimension:]

        self.inverseHessianSolutionMatrix = np.linalg.inv(self.hessianSolutionMatrix)

        self._parameterSpaceMetricMatrix = self.DetermineMetricMatrix()

        self.DeterminePerturbationsInTangentSpaces()

        print("\nNext step size = " + str(self._currentStepSize) + "\n\n")

        return self._currentContinuationArgument + self._currentStepSize

    def ChooseLargestStepSizeSatisfyingRequirements(self):
        print("----- Step Size Selection-----\n")
        print("Initial stepSize = " + str(self._currentStepSize) + "\n")

        lowBound, highBound = 0.0, 1.0
        newStepSize = 1.0

        def potentialSolutionCurvePoint(newStepSize):
            return list(self.SolutionSpace.exp(self._currentSolution, newStepSize * self._currentSolutionPerturbation)) + \
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
        return - self.inverseHessianSolutionMatrix \
               @ self.hessianMixteMatrix \
               @ perturbationInParameterSpace

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

            (potentialNextSolution, self._approximateInverseHessian, RBFGSIters) = corrector.SearchSolution(potentialNextApproximate, self._approximateInverseHessian)

            totalRBFGSIters += RBFGSIters
            if self._objectiveFunction(list(potentialNextSolution) +  [potentialNextParameter]) <= 1e-9:
                print("######## ULTIMATUM SUCCESS ########\n")

                return potentialNextContinuationArgument, tuple(potentialNextSolution) +  (potentialNextParameter, ), totalRBFGSIters

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

    def DetermineMetricMatrix(self):
        raise NotImplementedError

    def DeterminePerturbationsInTangentSpaces(self):
        raise NotImplementedError

    def ConvertToTangentVectorOnSolutionSpace(self, currentPoint, vector):
        raise NotImplementedError

    def ConvertToTangentVector(self, currentPoint, vector):
        raise NotImplementedError

    def ExpressHessianMatrixInSuitableBasis(self):
        raise NotImplementedError