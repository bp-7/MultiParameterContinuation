import autograd.numpy as np
from SE3Parameterizations.Helpers.BasicSE3Transformations import tau, tau_y, rho_x, rho_z

class Torus:
    def __init__(self, smallRadius, largeRadius, offset=np.array([0., 0., 0.])):
        self.smallRadius = smallRadius
        self.largeRadius = largeRadius
        self.offset = offset

    def Evaluate(self, s, t):
        return tau(self.offset) @ rho_z(2 * np.pi * s) @ tau_y(-self.largeRadius) @ rho_x(2 * np.pi * t) @ tau_y(
            - self.smallRadius) @ rho_x(-0.5 * np.pi)


# def Tore(s, t, smallRadius, largeRadius, offset):
#     #return tau(offset) @ rho_z(2 * np.pi * s) @ tau_x(largeRadius) @ rho_y(2 * np.pi * t) @ tau_z(smallRadius) @ rho_y(- 0.5 * np.pi)
#     return tau(offset) @ rho_z(2 * np.pi * s) @ tau_y(-largeRadius) @ rho_x(2 * np.pi * t) @ tau_y(- smallRadius) @ rho_x(-0.5 * np.pi)