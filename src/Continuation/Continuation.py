import numpy as np

from pymanopt.core.problem import Problem
from pymanopt.manifolds.product import Product

from Solver.SolverRBFGS import SolverRBFGS

class AbstractContinuation:
    ParameterTolerance = 1e-9
    ObjectivePredictionTolerance = 2
    GradientNormPredictionTolerance = 500
    MaximalContinuationIterations = 300

    def __init__(self,
                 problem,
                 initialSolution,
                 initialParameter,
                 targetParameter,
                 A):
        if not isinstance(problem, Problem):
            raise ValueError('The problem must be an instance of pymanopt.core.problem.Problem')

        if not isinstance(problem.manifold, Product):
            raise ValueError('The problem manifold should be a product manifold')

        self.A = A
        self.ProductManifold = problem.manifold
        self.SolutionSpace = problem.manifold._manifolds[0]
        self.FinalParameter = targetParameter

        self._objectiveFunction = problem.cost
        self._gradient = problem.grad
        self._approximateInverseHessian = np.eye(self.SolutionSpace.dim + 1)

        self._currentContinuationArgument = 0.0
        self._nextContinuationArgument = self._currentContinuationArgument

        self._currentSolution = initialSolution
        self._currentParameter = initialParameter

        self.distance = 0

        self._currentApproximate = None
        self._nextApproximate = None
        self._nextParameter = None
        self._nextInitialGuess = None

    def UpdateContinuationInformation(self, newlyFoundApproximate):
        self._currentContinuationArgument = self._nextContinuationArgument
        self._currentApproximate = newlyFoundApproximate[0]
        self._currentParameter = newlyFoundApproximate[1]

        self._nextContinuationArgument = self.GetNextContinuationArgument()
        self._nextParameter = self.GetNextParameter()
        self._nextApproximate = self.GetNextApproximate()

    def GetNextContinuationArgument(self):
        raise NotImplementedError

    def GetNextParameter(self):
        raise NotImplementedError

    def GetNextApproximate(self):
        raise NotImplementedError

    def AssertIfContinuationIsFinished(self, iterations):
        return np.linalg.norm(self._nextParameter - self.FinalParameter) <= self.ParameterTolerance \
               or iterations >= self.MaximalContinuationIterations

    def Traverse(self):

        iterations = 0
        solutionCurve = [(0, (self._currentSolution, self._currentParameter))]

        continuationFinished = False
        print("\n\n\n========== New continuation problem ==========\n")

        while not continuationFinished:

            iterations = iterations + 1
            self.UpdateContinuationInformation(solutionCurve[-1][1])

            print("----- Step " + str(iterations) + ", L = " + str(self._nextContinuationArgument) + " -----\n")

            def costForFixedParameter(X):
                return self._objectiveFunction((X, self._nextParameter))

            correctorProblem = Problem(self.SolutionSpace, costForFixedParameter)

            corrector = SolverRBFGS(correctorProblem)

            (self._currentSolution, self._approximateInverseHessian) = corrector.SearchSolution(self._nextApproximate, self._approximateInverseHessian)

            def modifiedCostFixedParameter(X):
                return costForFixedParameter(X) - np.min(np.linalg.eigvals(self.A(self._nextParameter)))

            if modifiedCostFixedParameter(self._currentSolution) >= 1.0e-9:
                potentialNewSolutionCurvePoint = self.DoSomethingUponFailureOrAcceptFailure()

                if potentialNewSolutionCurvePoint is None:
                    solutionCurve.append((self._nextContinuationArgument, (self._currentSolution, self._nextParameter)))

                else:
                    solutionCurve.append(potentialNewSolutionCurvePoint)
            else:
                solutionCurve.append((self._nextContinuationArgument, (self._currentSolution, self._nextParameter)))

            self.distance += self.SolutionSpace.dist(solutionCurve[-1][1][0], solutionCurve[-2][1][0])

            continuationFinished = self.AssertIfContinuationIsFinished(iterations)

        return solutionCurve

class CostForFixedParameterValue:
    def __init__(self, costFunctionOnProductManifold, currentParameter):
        self.costFunctionOnProductManifold = costFunctionOnProductManifold
        self.currentParameter = currentParameter

    def cost(self, X):
        return self.costFunctionOnProductManifold((X, self.currentParameter))

