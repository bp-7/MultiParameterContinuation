import numpy as np
import scipy.linalg


class GLTR:
    def __init__(self, A, Dmu, g, radius):
        self.A, self.Dmu = A, Dmu
        self.g = g
        self.radius = radius

        self.sk, self.skMsk = 0., 0.
        self.gk = g
        self.vk = self.EvaluateInverseMetric(self.gk)
        self.pk = - self.vk
        self.gammak = np.sqrt(np.dot(self.vk, self.gk))
        self.INTERIOR = True

        self.alphak = None
        self.betak = None
        self.betaNext = None
        self.sNext = None
        self.Tk = None

        self.pkMpkPrev, self.skMpkPrev = None, None

        self.Tprev = None
        self.alphaPrev = None
        self.betaPrev = 0

        self.iter = None

        self.maxInnerIterations = np.sum(self.Dmu.shape)
        self.MaximumIteration = np.sum(self.Dmu.shape)

    def EvaluateInverseMetric(self, v):
        parameterSpaceDimension = self.Dmu.shape[-1]
        B = np.block([[self.Dmu, self.A],
                      [np.zeros((parameterSpaceDimension, parameterSpaceDimension), dtype='float64'), self.Dmu.T]])

        rhs = np.concatenate(np.zeros(parameterSpaceDimension, v))
        lu_and_piv = scipy.linalg.lu_factor(B)

        X = scipy.linalg.lu_solve(lu_and_piv, rhs)

        return X[:parameterSpaceDimension]

    def pcgOneIteration(self):
        alphak = np.dot(self.gk, self.vk) / np.dot(self.pk, self.pk)
        gNext = self.gk + alphak * self.pk
        vNext = self.EvaluateInverseMetric(gNext)
        betak = np.dot(gNext, gNext) / np.dot(self.gk, self.vk)
        pNext = - vNext + betak * self.vk

        return gNext, vNext, pNext

    def EvaluateTridiagonalRecurrence(self):

        if self.iter == 0:
            return np.array([1. / self.alphak])
        else:
            return np.block([[self.Tprev, np.concatenate((np.zeros(self.iter - 1), np.array([np.sqrt(self.betaPrev) / self.alphaPrev])))],
                            [np.concatenate((np.zeros(self.iter - 1), np.array([np.sqrt(self.betaPrev) / self.alphaPrev,
                                                                               1. / self.alphak + self.betaPrev / self.alphaPrev])))]])

    def SolveTridiagonalTRS(self):
        maxiter = 200
        #Need to find suitable initial value
        lambdak = 0
        I = np.eye(self.iter + 1)
        res = np.inf
        innerIteration = 0

        while res >= 1.e-10 and innerIteration < self.maxInnerIterations:
            B, eigvals = np.linalg.eig(self.Tk + lambdak * I)
            D = np.diag(eigvals)
            Dinv = np.linalg.inv(D)

            h = - self.gammak * B @ Dinv @ B.T @ I[0]
            w = B.T @ h

            lambdak = lambdak - 1. / (self.radius * w.T @ Dinv @ w.T) * (self.radius - np.linalg.norm(h)) * np.linalg.norm(h) ** 2

            phi_lambdak = 1. / np.norm(h) - 1. / self.radius

            res = np.abs(phi_lambdak)

            innerIteration = innerIteration + 1

    def EvaluateMetricNormTerm(self):
        pkMpk = np.dot(self.gk, self.vk) if self.iter == 0 else np.dot(self.gk, self.vk) + self.betaPrev ** 2 * self.pkMpkPrev
        skMpk = 0 if self.iter == 0 else self.betaPrev * (self.skMpkPrev + self.alphaPrev * self.pkMpkPrev)

        self.pkMpkPrev, self.skMpkPrev = pkMpk, skMpk

        skMskNext = self.skMsk + 2 * self.alphak * skMpk + self.alphak ** 2 * pkMpk
        self.skMsk = skMskNext

        return skMskNext


    def SearchDirection(self):
        self.iter = 0

        while self.ContinueOptimization():
            self.alphak = np.dot(self.gk, self.vk) / np.dot(self.pk, self.pk)
            self.Tk = self.EvaluateTridiagonalRecurrence()

            if self.INTERIOR and (np.abs(self.alphak) <= 1e-15 or self.EvaluateMetricNormTerm() >= self.radius):
                self.INTERIOR = False
            elif self.INTERIOR:
                self.sNext = self.sk + self.alphak * self.pk
            else:
                self.SolveTridiagonalTRS()

            gNext = self.gk + self.alphak * self.pk
            vNext = self.EvaluateInverseMetric(gNext)

            if self.INTERIOR:
                self.TestConvergence()
            else:
                self.TestConvergence()

            self.betak = np.dot(gNext, vNext) / np.dot(self.gk, self.vk)
            pNext = - vNext + self.betak * self.pk

            self.alphaPrev, self.betaPrev, self.Tprev = self.alphak, self.betak, self.Tk

            self.pk, self.gk, self.vk = pNext, gNext, vNext

            self.iter += self.iter

        if not self.INTERIOR:
            self.sk = self.Qk @ self.hk

    def ContinueOptimization(self):
        return True

    def TestConvergence(self):
        raise NotImplementedError



