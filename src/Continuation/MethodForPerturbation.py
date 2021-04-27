from enum import Enum

class MethodForPerturbation(Enum):
    naive = 1
    controlledPerturbation = 2
    controlledPerturbationFromSecondOrderEstimate = 3