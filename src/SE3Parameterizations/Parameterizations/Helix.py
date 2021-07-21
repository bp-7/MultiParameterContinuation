import autograd.numpy as np
from Helpers.Parameterizations.BasicSE3Transformations import tau_x, tau_y, tau_z, rho_z, rho_y

class Helix:
    def __init__(self, helixRadius, helixAngle, helixLength, offsetAngle=0.0, offsetYAxis=0.0):
        self.radius = helixRadius
        self.helixAngle = helixAngle
        self.length = helixLength
        self.offsetAngle = offsetAngle
        self.offsetYAxis = offsetYAxis
        self.helixStep = 2 * np.pi * helixRadius / np.tan(helixAngle) if helixAngle >= 1e-14 else 0.


    def Evaluate(self, t):
        revolutionAngle = self.length * t * np.tan(self.helixAngle) / self.radius
        totalAngle = revolutionAngle + self.offsetAngle

        return tau_y(self.offsetYAxis) \
               @ rho_z(totalAngle) \
               @ tau_x(self.radius) \
               @ tau_z(self.length * t) \
               @ rho_y(0.5 * np.pi) \
               @ rho_z(np.pi  - self.helixAngle)

def helix(t, helixRadius, helixAngle, helixLength, offsetAngle=0.):
    revolutionAngle = helixLength * t * np.tan(helixAngle) / helixRadius
    totalAngle = revolutionAngle + offsetAngle

    return rho_z(totalAngle) \
           @ tau_x(helixRadius) \
           @ tau_z(helixLength * t) \
           @ rho_y(0.5 * np.pi) \
           @ rho_z(np.pi - helixAngle)
