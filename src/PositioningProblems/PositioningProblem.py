class PositioningProblem:

    def __init__(self, solutionSpace, constrainedManifold, parameterSpace):
        self._solutionSpace = solutionSpace
        self._parameterSpace = parameterSpace

        self.ConstrainedManifold = constrainedManifold


    def Cost(self, S):
        raise NotImplementedError

    def DifferentialSolution(self, S, zeta):
        raise NotImplementedError

    def DifferentialParameter(self, S, v):
        raise NotImplementedError
