import autograd.numpy as np

from Continuation.PositioningProblem.ContinuationPositioning import AbstractContinuationPositioning
from Continuation.Helpers.AssembleMatrices import RepresentSquareOperatorInBergerBasis, \
    ConstructMetricMatrixForBergerManifold, \
    RepresentRectangularOperatorFromEuclideanToBergerManifold

from pymanopt.core.problem import Problem
from pymanopt.manifolds import Sphere
from pymanopt.solvers import SteepestDescent

from Solver.SolverRBFGSPositioning import SolverRBFGS


class PathAdaptiveMultiParameterContinuation(AbstractContinuationPositioning):
    StepSizeContraction = 0.75
    InitialStepSize = 0.1
    MaximumStepSize = 1
    MaximumNumberOfContinuationSteps = 250
    MinimumStepSize = 1.0 / MaximumNumberOfContinuationSteps

    def __init__(self,
                 problem,
                 initialSolution,
                 initialParameter,
                 targetParameter,
                 perturbationMagnitude,
                 typeOfConstraint):

        super().__init__(problem, initialSolution, initialParameter, targetParameter)

        self._perturbationMagnitude = perturbationMagnitude
        self._currentStepSize = self.InitialStepSize

        self._typeOfConstraint = typeOfConstraint
        self._currentParameterPerturbation = None
        self._currentSolutionPerturbation = None

        # For the moment, but need to change
        self._solutionSpaceMetricMatrix = ConstructMetricMatrixForBergerManifold(int(self.SolutionSpace.dim) - 6)

        self._modifiedPerturbationMagnitude = None

    def GetNextParameter(self):
        return self._currentParameter + self._currentStepSize * self._currentParameterPerturbation

    def GetNextApproximate(self):
        return self.SolutionSpace.exp(self._currentSolution, self._currentStepSize * self._currentSolutionPerturbation)

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
                                                                          self._currentSolution + (self._currentParameter,))

        differentialParameterMatrix = RepresentRectangularOperatorFromEuclideanToBergerManifold(differentialParameter,
                                                                                                self.SolutionSpace,
                                                                                                self._currentSolution + (self._currentParameter,),
                                                                                                len(self._currentParameter))

        inverseDifferentialSolutionMatrix = np.linalg.inv(differentialSolutionMatrix)

        self._parameterSpaceMetricMatrix = differentialParameterMatrix.T \
                                     @ inverseDifferentialSolutionMatrix.T \
                                     @ self._solutionSpaceMetricMatrix \
                                     @ inverseDifferentialSolutionMatrix \
                                     @ differentialParameterMatrix

        self._currentParameterPerturbation = self.chooseParameterPerturbationOnEllipsoid(self._parameterSpaceMetricMatrix)

        self._currentSolutionPerturbation = inverseDifferentialSolutionMatrix \
                                     @ differentialParameterMatrix \
                                     @ self._currentParameterPerturbation

        potentialNewStepSize = self.ChooseLargestStepSizeSatisfyingRequirements()

        if self._currentContinuationArgument + potentialNewStepSize - 1 > - self.MinimumStepSize:
            self._currentStepSize = 1 #- self._currentContinuationArgument
            print(" --> End of continuation exceeded")
            print("\nNext step size = " + str(self._currentStepSize) + "\n\n")

            return 1

        self._currentStepSize = potentialNewStepSize
        print("\nNext step size = " + str(self._currentStepSize) + "\n\n")

        return self._currentContinuationArgument + self._currentStepSize

    def ChooseLargestStepSizeSatisfyingRequirements(self):
        print("----- Step Size Selection-----\n")
        print("Initial stepSize = " + str(self._currentStepSize) + "\n")

        newStepSize = self.InitialStepSize

        def potentialSolutionCurvePoint(newStepSize):
            return tuple(self.SolutionSpace.exp(self._currentSolution, newStepSize * self._currentSolutionPerturbation)) + \
                   (self._currentParameter + self._currentStepSize * self._currentParameterPerturbation, )

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

            (potentialNextSolution, self._approximateInverseHessian) = corrector.SearchSolution(potentialNextApproximate, self._approximateInverseHessian)

            if self._objectiveFunction(potentialNextSolution +  (potentialNextParameter,)) <= 1e-7:
                print("######## ULTIMATUM SUCCESS ########\n")

                return potentialNextContinuationArgument, potentialNextSolution +  (potentialNextParameter, )

        print("### ULTIMATUM FAILED : MIN STEP SIZE REACHED ###\n")

        return None

    def chooseParameterPerturbationOnEllipsoid(self, metricMatrix):
        choleskyFactor = np.linalg.cholesky(metricMatrix)
        inverseCholeskyFactor = np.linalg.inv(choleskyFactor)

        deltaParameter = self.FinalParameter - self._currentParameter
        if self._typeOfConstraint == 'sphere':
            if np.inner(deltaParameter, deltaParameter) < self._perturbationMagnitude ** 2:
                perturbationMagnitude = np.sqrt(np.inner(deltaParameter, deltaParameter))
            else:
                perturbationMagnitude = self._perturbationMagnitude

            def cost(y):
                u = perturbationMagnitude * y - deltaParameter

                return 0.5 * np.inner(u, u)
        else:
            if np.inner(deltaParameter, metricMatrix @ deltaParameter) < self._perturbationMagnitude ** 2:
                perturbationMagnitude = np.sqrt(np.inner(deltaParameter, metricMatrix @ deltaParameter))
            else:
                perturbationMagnitude = self._perturbationMagnitude

            def cost(y):
                u = perturbationMagnitude * inverseCholeskyFactor.T @ y - deltaParameter

                return 0.5 * np.inner(u, u)

        unitSphereDimension = len(self._currentParameter)
        unitSphere = Sphere(unitSphereDimension)

        # Instantiate a minimization problem
        problem = Problem(manifold=unitSphere, cost=cost)

        # Instantiate a Pymanopt solver
        solver = SteepestDescent()

        if self._typeOfConstraint == 'sphere':
            xStart = deltaParameter
            xStartNormalized = xStart / np.linalg.norm(xStart)
            if unitSphere.norm(xStartNormalized, problem.grad(xStartNormalized)) <= 1e-15:
                return perturbationMagnitude * xStartNormalized
        else:
            xStart = (1.0 / perturbationMagnitude) * choleskyFactor.T @ deltaParameter
            xStartNormalized = xStart / np.linalg.norm(xStart)
            if unitSphere.norm(xStartNormalized, problem.grad(xStartNormalized)) <= 1e-15:
                return perturbationMagnitude * inverseCholeskyFactor.T @ xStartNormalized

        v = solver.solve(problem, x=xStartNormalized)

        print("v_steepest = " + str(v) + "\n")

        if self._typeOfConstraint == 'sphere':
            newPerturbation = perturbationMagnitude * v
        else:
            newPerturbation = perturbationMagnitude * inverseCholeskyFactor.T @ v

        print("\n New perturbation : " + str(newPerturbation) + "\n")

        return newPerturbation
