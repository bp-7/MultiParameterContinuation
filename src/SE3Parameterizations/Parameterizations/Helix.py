import autograd.numpy as np
from SE3Parameterizations.Helpers.BasicSE3Transformations import tau_x, tau_y, tau_z, rho_z, rho_y

class Helix:
    def __init__(self, helixRadius, helixAngle, helixLength, offsetAngle=0.0, offsetYAxis=0.0):
        self.radius = helixRadius
        self.helixAngle = helixAngle
        self.length = helixLength
        self.offsetAngle = offsetAngle
        self.offsetYAxis = offsetYAxis
        self.helixStep = 2 * np.pi * helixRadius / np.tan(helixAngle) if helixAngle <= 1e-14 else 0.


    def Evaluate(self, t):
        revolutionAngle = self.length * t * np.tan(self.helixAngle) / self.radius
        totalAngle = revolutionAngle + self.offsetAngle

        if np.abs(self.helixAngle) <= 1.e-14:
            # return tau_z(helixLength * t) @ tau_x(helixRadius * np.cos(totalAngle)) @ tau_y(helixRadius * np.sin(totalAngle)) \
            #       @ rho_y(0.5 * np.pi) @ rho_z(np.pi)
            return tau_y(-self.offsetYAxis) @ tau_z(self.length * t) @ rho_z(totalAngle) @ tau_x(self.radius) @ rho_y(0.5 * np.pi) @ rho_z(np.pi)
        else:
            helixStep = 2 * np.pi * self.radius / np.tan(self.helixAngle)
            # return tau_z(helixStep * t) @ tau_x(helixRadius * np.cos(totalAngle)) @ tau_y(helixRadius * np.sin(totalAngle)) @ rho_y(0.5 * np.pi) \
            #       @ rho_z(np.pi - helixAngle)
            return tau_y(-self.offsetYAxis) @ tau_z(self.length * t) @ rho_z(totalAngle) @ tau_x(self.radius) @ rho_y(0.5 * np.pi) @ rho_z(
                np.pi - self.helixAngle)


# def Helix(t, helixRadius, helixAngle, helixLength, offsetAngle = 0.):
#     revolutionAngle = helixLength * t * np.tan(helixAngle) / helixRadius
#     totalAngle = revolutionAngle + offsetAngle
#
#     if np.abs(helixAngle) <= 1.e-14:
#         #return tau_z(helixLength * t) @ tau_x(helixRadius * np.cos(totalAngle)) @ tau_y(helixRadius * np.sin(totalAngle)) \
#         #       @ rho_y(0.5 * np.pi) @ rho_z(np.pi)
#         return tau_z(helixLength * t) @ rho_z(totalAngle) @ tau_x(helixRadius) @ rho_y(0.5 * np.pi) @ rho_z(np.pi)
#     else:
#         helixStep = 2 * np.pi * helixRadius / np.tan(helixAngle)
#         #return tau_z(helixStep * t) @ tau_x(helixRadius * np.cos(totalAngle)) @ tau_y(helixRadius * np.sin(totalAngle)) @ rho_y(0.5 * np.pi) \
#         #       @ rho_z(np.pi - helixAngle)
#         return tau_z(helixStep * t) @ rho_z(totalAngle) @ tau_x(helixRadius) @ rho_y(0.5 * np.pi) @ rho_z(np.pi - helixAngle)
