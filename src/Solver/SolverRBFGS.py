import numpy as np
from pymanopt.core.problem import Problem

class SolverRBFGS:
    MaxIterations = 250
    ArmijoPromisedDecreaseScaling = 1e-4
    ArmijoStepSizeContraction = 0.7
    ArmijoInitialStepSize = 1
    ObjectiveFunctionTolerance = 1e-12
    GradientNormTolerance = 1e-6

    def __init__(self, problem):
        if not isinstance(problem, Problem):
            raise ValueError('The problem must be an instance of pymanopt.core.problem.Problem')

        self._solutionSpace = problem.manifold
        self._objectiveFunction = problem.cost
        self._gradient = problem.grad

    def ContinueOptimization(self, currentPoint, iterations):
        return (self._solutionSpace.norm(currentPoint, self._gradient(currentPoint)) > self.GradientNormTolerance
                and self._objectiveFunction(currentPoint) > self.ObjectiveFunctionTolerance) \
               and iterations < self.MaxIterations #before was or then and

    def ArmijoLineSearch(self, currentPoint, currentGradient, searchDirection):
        fCurrentPoint = self._objectiveFunction(currentPoint)
        promisedDecrease = self._solutionSpace.inner(currentPoint, currentGradient, searchDirection)
        currentStepSize = self.ArmijoInitialStepSize

        def armijoCondition(stepSize):
            return self._objectiveFunction(self._solutionSpace.retr(currentPoint, stepSize * searchDirection)) > fCurrentPoint \
                   + stepSize * self.ArmijoPromisedDecreaseScaling * promisedDecrease

        while armijoCondition(currentStepSize) and currentStepSize > 1e-10:
            currentStepSize = self.ArmijoStepSizeContraction * currentStepSize

        return currentStepSize

    def SearchSolution(self, initialGuess, initialApproximateInverseHessian):
        iterations = 0

        currentPoint = initialGuess
        currentGradient = self._gradient(currentPoint)

        updateDirection = - currentGradient

        approximateInverseHessian = initialApproximateInverseHessian

        print("f_" + str(iterations) + " = " + str(self._objectiveFunction(currentPoint)))
        print("|gradf_" + str(iterations) + "| = " + str(self._solutionSpace.norm(currentPoint, currentGradient)))

        while self.ContinueOptimization(currentPoint, iterations):
            iterations = iterations + 1

            stepSize = self.ArmijoLineSearch(currentPoint, currentGradient, updateDirection)
            newPoint = self._solutionSpace.retr(currentPoint, stepSize * updateDirection)

            previousGradient = currentGradient
            newGradient = self._gradient(newPoint)

            approximateInverseHessian = self.UpdateApproximateInverseHessian(
                currentPoint,
                approximateInverseHessian,
                newGradient,
                previousGradient,
                stepSize * updateDirection)

            updateDirection = -approximateInverseHessian @ newGradient

            currentPoint = newPoint

            currentGradient = newGradient

            print("f_" + str(iterations) + " = " + str(self._objectiveFunction(currentPoint)))
            print("|gradf_" + str(iterations) + "| = " + str(self._solutionSpace.norm(currentPoint, currentGradient)))

        return currentPoint, approximateInverseHessian

    def UpdateApproximateInverseHessian(self,
                                        currentPoint,
                                        oldInverseHessian,
                                        currentGradient,
                                        previousGradient,
                                        previousSearchDirection):
        yk = currentGradient - previousGradient
        sk = previousSearchDirection

        #For the moment
        metricMatrix = np.eye(self._solutionSpace.dim + 1)

        def inner(G, H):
            return self._solutionSpace.inner(currentPoint, G, H)

        skTGyk = inner(sk, yk)

        if not skTGyk > 0:
            return oldInverseHessian

        intermediateScalar = inner(sk, yk) + inner(yk , oldInverseHessian @ yk)

        return oldInverseHessian \
               + np.outer(sk, sk.T @ metricMatrix) * intermediateScalar / (skTGyk * skTGyk)\
               - (np.outer(oldInverseHessian @ yk, sk.T @ metricMatrix)
                  + np.outer(sk, yk.T @ metricMatrix) @ oldInverseHessian) / skTGyk

