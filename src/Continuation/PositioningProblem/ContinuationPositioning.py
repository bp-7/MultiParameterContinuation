import autograd.numpy as np

from pymanopt.core.problem import Problem
from pymanopt.manifolds.product import Product

from Solver.SolverRBFGSPositioning import SolverRBFGS

class AbstractContinuationPositioning:
    ParameterTolerance = 1e-9
    GradientNormPredictionTolerance = 1000
    MaximalContinuationIterations = 15

    def __init__(self,
                 problem,
                 initialSolution,
                 initialParameter,
                 targetParameter,
                 ObjectivePredictionTolerance):
        if not isinstance(problem, Problem):
            raise ValueError('The problem must be an instance of pymanopt.core.problem.TestCases')

        if not isinstance(problem.manifold, Product):
            raise ValueError('The problem manifold should be a product manifold')

        if len(problem.manifold._manifolds[:-1]) > 1:
            self.SolutionSpace = Product(problem.manifold._manifolds[:-1])
        else:
            self.SolutionSpace = problem.manifold._manifolds[0]

        self.ProductManifold = problem.manifold
        self.FinalParameter = targetParameter
        self.Problem = problem

        self.ObjectivePredictionTolerance = ObjectivePredictionTolerance

        self._objectiveFunction = problem.cost
        self._gradient = problem.grad
        self._hessian = problem.hess
        self._approximateInverseHessian = np.eye(int(self.SolutionSpace.dim))

        self._currentContinuationArgument = 0.0
        self._nextContinuationArgument = self._currentContinuationArgument

        self._currentSolution = initialSolution
        self._currentParameter = initialParameter

        self.IsBasisNormalized = False

        def costFunctionForFixedParameter(S):
            A = list(S)
            A.append(self._currentParameter)
            return problem.cost(A)

        self.problemForFixedParameter = Problem(self.SolutionSpace, cost=costFunctionForFixedParameter)

        self._objectiveFunctionForFixedParameter = self.problemForFixedParameter.cost
        self._gradientForFixedParameter = self.problemForFixedParameter.grad
        self._hessianForFixedParameter = self.problemForFixedParameter.hess

        self.distance = 0

        self._parameterSpaceMetricMatrix = None
        self._perturbationMagnitude = None

        self._currentApproximate = None
        self._nextApproximate = None
        self._nextParameter = None
        self._nextInitialGuess = None

    def UpdateContinuationInformation(self, newlyFoundApproximate):
        self._currentContinuationArgument = self._nextContinuationArgument
        self._currentSolution = newlyFoundApproximate[:3]
        self._currentParameter = newlyFoundApproximate[-1]

        self._nextContinuationArgument = self.GetNextContinuationArgument()

        if self._nextContinuationArgument is not None:
            self._nextParameter = self.GetNextParameter()
            self._nextApproximate = self.GetNextApproximate()
        else:
            pass

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
        solutionCurve = [(0, list(self._currentSolution) + [self._currentParameter])]

        solved = True
        totalRBFGSIterations = 0
        parameterSpaceMetricMatrices = []
        perturbationMagnitudes = []

        continuationFinished = False
        print("\n\n\n========== New continuation problem ==========\n")

        while not continuationFinished:

            iterations = iterations + 1
            self.UpdateContinuationInformation(solutionCurve[-1][1])

            if self._nextContinuationArgument is None:
                solved = False
                print("\n\n ======= NOT SOLVED =======\n\n")
                totalRBFGSIterations = -1
                break


            print("----- Step " + str(iterations) + ", L = " + str(self._nextContinuationArgument) + " -----\n")

            parameterSpaceMetricMatrices.append(self._parameterSpaceMetricMatrix)
            perturbationMagnitudes.append(self._perturbationMagnitude)
            def costForFixedParameter(S):
                A = list(S)
                A.append(self._nextParameter)

                return self._objectiveFunction(A)

            correctorProblem = Problem(self.SolutionSpace, costForFixedParameter)
            #from pymanopt.solvers.trust_regions import TrustRegions

            #corrector = TrustRegions()

            #self._currentSolution = corrector.solve(correctorProblem, x=self._nextApproximate)
            corrector = SolverRBFGS(correctorProblem, self.IsBasisNormalized)
            (potentialSolution, self._approximateInverseHessian, RBFGSIters) = corrector.SearchSolution(self._nextApproximate, self._approximateInverseHessian)

            totalRBFGSIterations += RBFGSIters

            if costForFixedParameter(potentialSolution) >= 1.0e-9:
                potentialNewSolutionCurvePoint = self.DoSomethingUponFailureOrAcceptFailure()

                if potentialNewSolutionCurvePoint is None:
                    solutionCurve.append((self._nextContinuationArgument, (self._currentSolution, self._nextParameter)))
                    solved = False
                    print("\n\n ======= NOT SOLVED =======\n\n")
                    totalRBFGSIterations = -1
                    break

                else:
                    solutionCurve.append(potentialNewSolutionCurvePoint[:2])
                    self._nextContinuationArgument = potentialNewSolutionCurvePoint[0]
                    self._nextApproximate = potentialNewSolutionCurvePoint[1][:3]
                    self._nextParameter = potentialNewSolutionCurvePoint[1][-1]
                    solved = True
                    totalRBFGSIterations += potentialNewSolutionCurvePoint[-1]
            else:
                self._currentSolution = potentialSolution
                solutionCurve.append((self._nextContinuationArgument, tuple(self._currentSolution) + (self._nextParameter, )))
                solved = True

            #self.distance += self.SolutionSpace.dist(solutionCurve[-1][1][0], solutionCurve[-2][1][0])
            print("Delta parameter = " + str(self.FinalParameter - solutionCurve[-1][1][-1]))
            continuationFinished = self.AssertIfContinuationIsFinished(iterations)

        if iterations >= self.MaximalContinuationIterations and np.linalg.norm(self._nextParameter - self.FinalParameter) >= self.ParameterTolerance:
            solved = False
            totalRBFGSIterations = -1

        return solutionCurve, parameterSpaceMetricMatrices, perturbationMagnitudes, totalRBFGSIterations, solved

