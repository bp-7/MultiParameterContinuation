import numpy as np
import scipy.linalg

from GLTR import GLTR


class TrustRegionAlgorithm:
    GradientTolerance = 1.0e-5
    LowerBoundRatio = 1.0e-2
    UpperBoundRatio = 0.95
    ContractionFactor = 0.5
    DilationFactor = 2.

    def __init__(self, A, Dmu, g, x0, radius, cost):
        self.A, self.Dmu = A, Dmu
        self.g = g
        self.currentIterate = x0
        self.radius = radius
        self.cost = cost

        self.GLTR = GLTR(A, Dmu, g, radius)

    def ContinueOptimization(self):
        return np.linalg.norm(self.currentIterate - g) <= self.GradientTolerance

    def SearchSolution(self):
        self.iteration = 0

        while self.ContinueOptimization():
            currentDirection = self.GLTR.SearchDirection()

            self.nextIterate = self.currentIterate + currentDirection











