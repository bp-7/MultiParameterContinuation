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


class PathAdaptiveMultiParameterContinuation(AbstractContinuationPositioning):
    StepSizeContraction = 0.75
    StepSizeContractionPerturbation = 0.95
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
                 ObjectiveFunctionTolerance):

        super().__init__(problem, initialSolution, initialParameter, targetParameter, ObjectiveFunctionTolerance)

        self._currentStepSize = self.InitialStepSize

        self.currentPoint = list(self._currentSolution) + [self._currentParameter]

        self._currentParameterPerturbation = None
        self._currentSolutionPerturbation = None
        self.differentialSolutionMatrix = None
        self.differentialParameterMatrix = None
        self.differentialSolutionPInv = None
        self.currentParameterPerturbationNorm = None

        self.differentialSolution = positioningProblem.DifferentialSolution
        self.differentialParameter = positioningProblem.DifferentialParameter
        self.F = positioningProblem.SystemEvaluation
        self.FCodomain = positioningProblem.ConstrainedManifold
        self.parameterSpace = Euclidean(targetParameter.shape)
        self._solutionSpaceMetricMatrix = np.diag(np.concatenate((np.array([2., 2., 2., 1., 1., 1.]),
                                                                  np.ones(int(
                                                                      self.SolutionSpace.dim) - self.SE3Dimension))))
        
    def GetNextParameter(self):
        return self._currentParameter + self._currentStepSize * self._currentParameterPerturbation

    def GetNextApproximate(self):
        return self.SolutionSpace.exp(self._currentSolution, self._currentStepSize * self._currentSolutionPerturbation)

    def GetNextContinuationArgument(self):

        self.currentPoint = list(self._currentSolution) + [self._currentParameter]
        FPoint = self.F(self.currentPoint)

        self.differentialSolutionMatrix = RepresentRectangularOperatorFromSolutionSpaceToSE32(
            self.differentialSolution,
            self.SolutionSpace,
            self.FCodomain,
            self.currentPoint,
            FPoint)

        if (self.SolutionSpace.dim == self.FCodomain.dim):
            self.differentialSolutionPInv = np.linalg.inv(self.differentialSolutionMatrix)

        else:
            self.differentialSolutionPInv = np.linalg.pinv(self.differentialSolutionMatrix)

        self.differentialParameterMatrix = RepresentRectangularOperatorFromParameterSpaceToSE32(
            self.differentialParameter,
            self.parameterSpace,
            self.FCodomain,
            self.currentPoint,
            FPoint)

        self._parameterSpaceMetricMatrix = self.DetermineMetricMatrix()

        # if np.linalg.cond(self._parameterSpaceMetricMatrix) > 1e6:
        #     print("Metric highly ill-conditioned, we reached the border and can not go further.")
        #     return None

        self.DeterminePerturbationsInTangentSpaces()

        print("\nNext step size = " + str(self._currentStepSize) + "\n\n")

        if self._currentStepSize < self.MinimumStepSize:
            return None

        return self._currentContinuationArgument + self._currentStepSize

    def DeterminePerturbationsInTangentSpaces(self):
        self._currentParameterPerturbation = self.FinalParameter - self._currentParameter

        self._currentSolutionPerturbation = self.ConvertToTangentVectorOnSolutionSpace(self._currentSolution,
                                                                                       self.DeterminePerturbationInSolutionSpace(
                                                                                           self._currentParameterPerturbation))
        self._currentStepSize = 1.0

        self.alternativeDirectionTaken = False

        if self._objectiveFunction(
                tuple(self.SolutionSpace.exp(self._currentSolution, self._currentSolutionPerturbation)) \
                + (self._currentParameter + self._currentParameterPerturbation,)) > self.ObjectivePredictionTolerance:

            potentialStepSize = self.ChooseLargestStepSizeSatisfyingRequirements()

            if potentialStepSize < self.MinimumStepSize:
                potentialStepSize = 0.1#self.MinimumStepSize

            straightLineMethodTolerance = self._objectiveFunction(
                tuple(self.SolutionSpace.exp(self._currentSolution, potentialStepSize * self._currentSolutionPerturbation)) \
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
                    #or ellipsoidMethodTolerance > straightLineMethodTolerance:

                potentialPerturbationStepSize = self.ChoosePotentialLargestStepSizeSatisfyingRequirements(
                    potentialParameterPerturbation,
                    potentialSolutionPerturbation,
                    self.currentParameterPerturbationNorm,
                    straightLineMethodTolerance
                )

                print("ellipsoidTolerance = " + str(self._objectiveFunction(
                     tuple(self.SolutionSpace.exp(self._currentSolution, potentialPerturbationStepSize * potentialSolutionPerturbation)) \
                     + (self._currentParameter + potentialPerturbationStepSize * potentialParameterPerturbation,))) + "\n")
                print("straightLineTolerance = " + str(straightLineMethodTolerance) + "\n\n\n")

                if potentialPerturbationStepSize < 1e-14: #or np.linalg.norm(
    #                     self._currentParameter + newStepSize * self._currentParameterPerturbation - self.FinalParameter) \
    #                     <= np.linalg.norm(
    #                 self._currentParameter + tempStepSize * tempParameterPerturbation - self.FinalParameter):
                    self._currentStepSize = potentialStepSize
                else:
                    self._currentStepSize = potentialPerturbationStepSize
                    self._currentParameterPerturbation = potentialParameterPerturbation
                    self._currentSolutionPerturbation = potentialSolutionPerturbation
                    self.alternativeDirectionTaken = True
            else:
                self._currentParameterPerturbation = potentialParameterPerturbation
                self._currentSolutionPerturbation = potentialSolutionPerturbation
                self._currentStepSize = 1.0
                self.alternativeDirectionTaken = True

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

                if potentialObjectiveFunctionValue(newStepSize) < self.ObjectivePredictionTolerance:
                   #and potentialGradientNorm(newStepSize) < self.GradientNormPredictionTolerance):
                    lowBound = newStepSize
                else:
                    highBound = newStepSize
                    newStepSize = lowBound

        return newStepSize

    def ChoosePotentialLargestStepSizeSatisfyingRequirements(self,
                                                             potentialParameterPerturbation,
                                                             potentialSolutionPerturbation,
                                                             currentPerturbationNorm,
                                                             straightLineMethodTolerance):
        print("----- Step Size Selection-----\n")
        print("Initial stepSize = " + str(self._currentStepSize) + "\n")

        minTolerance = np.min([straightLineMethodTolerance, self.ObjectivePredictionTolerance])

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

        if not (potentialObjectiveFunctionValue(newStepSize) < self.ObjectivePredictionTolerance
                and potentialGradientNorm(newStepSize) < self.GradientNormPredictionTolerance):

            while newStepSize > self.MinimumStepSize \
                    and potentialObjectiveFunctionValue(newStepSize) > self.ObjectivePredictionTolerance:
                    #and newStepSize * potentialPerturbationNorm > currentPerturbationNorm:

                newStepSize = self.StepSizeContractionPerturbation * newStepSize

                print("\nPotential new step size = " + str(newStepSize) + "\n")
                print("NextObj = " + str(potentialObjectiveFunctionValue(newStepSize)) + "\n")
                print("NextGrad = " + str(potentialGradientNorm(newStepSize)) + "\n")

        return newStepSize

    def DeterminePerturbationInSolutionSpace(self, perturbationInParameterSpace):
        return - self.differentialSolutionPInv \
               @ self.differentialParameterMatrix \
               @ perturbationInParameterSpace

    def DoSomethingUponFailureOrAcceptFailure(self):
        print("############### ULTIMATUM ###############\n")

        return self.ClassicUltimatum()

        # if self.alternativeDirectionTaken:
        #     return self.AlternativeUltimatum()
        # else:
        #     return self.ClassicUltimatum()

    def ClassicUltimatum(self):
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

            (potentialNextSolution, self._approximateInverseHessian, RBFGSIters) = corrector.SearchSolution(potentialNextApproximate, self._approximateInverseHessian)

            totalRBFGSIters += RBFGSIters
            if self._objectiveFunction(list(potentialNextSolution) +  [potentialNextParameter]) <= 1e-9:
                print("######## ULTIMATUM SUCCESS ########\n")

                return potentialNextContinuationArgument, tuple(potentialNextSolution) +  (potentialNextParameter, ), totalRBFGSIters

        print("### ULTIMATUM FAILED : MIN STEP SIZE REACHED ###\n")

        return None

    def AlternativeUltimatum(self):
        totalRBFGSIters = 0

        currentPerturbationNorm = np.linalg.norm(self._currentParameterPerturbation)

        while self._currentStepSize * self.StepSizeContraction > self.MinimumStepSize \
                and self._currentStepSize * self.StepSizeContraction * currentPerturbationNorm > self.currentParameterPerturbationNorm:

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

        from pymanopt.solvers.trust_regions import TrustRegions
        solver = TrustRegions()

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
        temp = self.differentialSolutionPInv @ self.differentialParameterMatrix

        return temp.T @ self._solutionSpaceMetricMatrix @ temp

    def ConvertToTangentVectorOnSolutionSpace(self, currentPoint, vector):
        return self.SolutionSpace.proj(currentPoint, (
              currentPoint[0] @ Skew(vector[:3]), np.array(vector[3:6]), np.array(vector[6:])))

    def ConvertToTangentVector(self, currentPoint, vector):
        return self.ProductManifold.proj(currentPoint, (
             currentPoint[0] @ Skew(vector[:3]), np.array(vector[3:6]), np.array(vector[6:9]), np.array(vector[9:])))
