import numpy as np
from pymanopt.core.problem import Problem

class SolverRBFGS:
    MaxIterations = 150
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
                approximateInverseHessian,
                newGradient,
                previousGradient,
                stepSize * updateDirection)

            updateDirection = self.ConvertToTangentVector(currentPoint, -approximateInverseHessian @ self.ConvertToVectorInRn(newGradient))

            currentPoint = newPoint

            currentGradient = newGradient

            #print("f_" + str(iterations) + " = " + str(self._objectiveFunction(currentPoint)))
            #print("|gradf_" + str(iterations) + "| = " + str(self._solutionSpace.norm(currentPoint, currentGradient)))

        print("f_" + str(iterations) + " = " + str(self._objectiveFunction(currentPoint)))
        print("|gradf_" + str(iterations) + "| = " + str(self._solutionSpace.norm(currentPoint, currentGradient)))

        return tuple(currentPoint), approximateInverseHessian, iterations

    def UpdateApproximateInverseHessian(self,
                                        oldInverseHessian,
                                        currentGradient,
                                        previousGradient,
                                        previousSearchDirection):

        yk = self.ConvertToVectorInRn(currentGradient) - self.ConvertToVectorInRn(previousGradient)
        sk = self.ConvertToVectorInRn(previousSearchDirection)

        #For the moment
        metricMatrix = np.diag(np.array([2., 2., 2., 1., 1., 1., 1., 1., 1.]))

        def inner(u, v):
            return np.inner(u, metricMatrix @ v)

        skTGyk = inner(sk, yk)

        if not skTGyk > 0:
            return oldInverseHessian

        intermediateScalar = inner(sk, yk) + inner(yk , oldInverseHessian @ yk)

        #if skTGyk < 1e-9 and np.linalg.norm(oldInverseHessian - np.eye(9)) >= 1e-14 :
        #    return np.eye(9)

        return oldInverseHessian \
               + np.outer(sk, sk.T @ metricMatrix) * intermediateScalar / (skTGyk * skTGyk)\
               - (np.outer(oldInverseHessian @ yk, sk.T @ metricMatrix)
                  + np.outer(sk, yk.T @ metricMatrix) @ oldInverseHessian) / skTGyk

    def ConvertToVectorInRn(self, tangentVector):
        vector = [self.InvSkew(tangentVector[0]) , tangentVector[1], tangentVector[2]]

        return np.array(vector).flatten()

    def ConvertToTangentVector(self, currentPoint, vector):
        return self._solutionSpace.proj(currentPoint, (self.Skew(vector[:3]), vector[3:6], vector[6:]))

    def Skew(self, w):
        return np.array([[0., -w[2], w[1]], [w[2], 0., -w[0]], [-w[1], w[0], 0.]])

    def InvSkew(self, S):
            return np.array([S[2, 1], S[0, 2], S[1, 0]])
