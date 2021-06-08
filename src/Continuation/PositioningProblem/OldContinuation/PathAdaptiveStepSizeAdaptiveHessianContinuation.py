import autograd.numpy as np

from Continuation.PositioningProblem.ContinuationPositioning import AbstractContinuationPositioning
from Continuation.Helpers.AssembleMatrices import RepresentSquareOperatorInTotalNormalizedBergerBasis, \
    ConstructMetricMatrixForBergerManifold

from pymanopt.core.problem import Problem
from pymanopt.manifolds import Sphere
from pymanopt.solvers import SteepestDescent

from Solver.SolverRBFGSPositioning import SolverRBFGS


class PathAdaptiveMultiParameterContinuation(AbstractContinuationPositioning):
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

        self._currentParameterPerturbation = None
        self._currentSolutionPerturbation = None

        self.IsBasisNormalized = False

    def GetNextParameter(self):
        return self._currentParameter + self._currentStepSize * self._currentParameterPerturbation

    def GetNextApproximate(self):
        return self.SolutionSpace.exp(self._currentSolution, self._currentStepSize * self._currentSolutionPerturbation)

    def GetNextContinuationArgument(self):

        currentPoint = list(self._currentSolution) + [self._currentParameter]

        hessianMatrix = RepresentSquareOperatorInTotalNormalizedBergerBasis(self._hessian,
                                                                            self.ProductManifold,
                                                                            currentPoint)
        solutionSpaceDimension = int(self.SolutionSpace.dim)

        self.hessianSolutionMatrix = hessianMatrix[:solutionSpaceDimension, :solutionSpaceDimension]

        self.hessianMixteMatrix = hessianMatrix[:solutionSpaceDimension, solutionSpaceDimension:]

        self.hessianParameterMatrix = hessianMatrix[solutionSpaceDimension:, solutionSpaceDimension:]

        self.inverseHessianSolutionMatrix = np.linalg.inv(self.hessianSolutionMatrix)

        self.hessianG = self.hessianParameterMatrix + self.hessianMixteMatrix.T @ self.inverseHessianSolutionMatrix @ self.hessianMixteMatrix

        self._parameterSpaceMetricMatrix = self.hessianG
        self._perturbationMagnitude = self.ObjectivePredictionTolerance

        self.DeterminePerturbationsInTangentSpaces()

        print("\nNext step size = " + str(self._currentStepSize) + "\n\n")

        return self._currentContinuationArgument + self._currentStepSize

    def ChooseLargestStepSizeSatisfyingRequirements(self):
        print("----- Step Size Selection-----\n")
        print("Initial stepSize = " + str(self._currentStepSize) + "\n")

        newStepSize = self._currentStepSize
        lowBound, highBound = 0.0, self._currentStepSize

        def potentialSolutionCurvePoint(newStepSize):
            currentSolutionPerturbation = self.ChangeInNonNormalizedBasis(self._currentSolutionPerturbation)
            return list(self.SolutionSpace.exp(self._currentSolution, newStepSize * currentSolutionPerturbation)) + \
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

            corrector = SolverRBFGS(correctorProblem, self.IsBasisNormalized)

            (potentialNextSolution, self._approximateInverseHessian, RBFGSIters) = corrector.SearchSolution(potentialNextApproximate, self._approximateInverseHessian)

            totalRBFGSIters += RBFGSIters
            if self._objectiveFunction(tuple(potentialNextSolution) +  (potentialNextParameter,)) <= 1e-7:
                print("######## ULTIMATUM SUCCESS ########\n")

                return potentialNextContinuationArgument, tuple(potentialNextSolution) +  (potentialNextParameter, ), totalRBFGSIters

        print("### ULTIMATUM FAILED : MIN STEP SIZE REACHED ###\n")

        return None

    def chooseParameterPerturbationOnEllipsoid(self, metric, magnitude):
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
                                                                                         self.DeterminePerturbationInSolutionSpace(self._currentParameterPerturbation))
        self._currentStepSize = 1.0

        if self._objectiveFunction(
                tuple(self.SolutionSpace.exp(self._currentSolution, self._currentSolutionPerturbation)) \
                + (self._currentParameter + self._currentParameterPerturbation,)) > self.ObjectivePredictionTolerance:

            potentialStepSize = self.ChooseLargestStepSizeSatisfyingRequirements()
            potentialParameterPerturbation = self.chooseParameterPerturbationOnEllipsoid(self.hessianG, np.sqrt(self.ObjectivePredictionTolerance))
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
            self.Skew(vector[:3]) / np.sqrt(2.), np.array(vector[3:6]), np.array(vector[6:])))

    def ConvertToTangentVector(self, currentPoint, vector):
        return self.ProductManifold.proj(currentPoint, (
        self.Skew(vector[:3]) / np.sqrt(2.), np.array(vector[3:6]), np.array(vector[6:9]), np.array(vector[9:])))

    def Skew(self, w):
        #return np.array([np.cross(w, np.array([1., 0, 0])), np.cross(w, np.array([0, 1., 0])), np.cross(w, np.array([0, 0, 1.]))], dtype=object).T
        return np.array([[0., -w[2], w[1]], [w[2], 0., -w[0]], [-w[1], w[0], 0.]], dtype=float)

    def ChangeInNonNormalizedBasis(self, vector):
        return [vector[0] * np.sqrt(2), vector[1], vector[2], vector[3]]