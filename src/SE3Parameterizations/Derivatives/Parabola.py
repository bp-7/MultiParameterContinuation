import autograd.numpy as np
from Helpers.Parameterizations.BasicSE3Transformations import tau_x, tau_y, rho_z
from Helpers.Parameterizations.BasicSe3Transformation import se3Basis

def ParabolaDerivative(t, a):
    return 2 * a * t * tau_y(a * t ** 2) @ se3Basis.G5 @ tau_x(t) @ rho_z(np.arctan(2 * a * t)) \
           + tau_y(a * t ** 2) @ tau_x(t) @ se3Basis.G4 @ rho_z(np.arctan(2 * a * t)) \
           + 2 * a / (1 + 4 * (a * t) ** 2) * tau_y(a * t ** 2) @ tau_x(t) @ rho_z(np.arctan(2 * a * t)) @ se3Basis.G3

def ParabolaParameterDerivative(t, a):
    return t ** 2 * tau_y(a * t ** 2) @ se3Basis.G5 @ tau_x(t) @ rho_z(np.arctan(2 * a * t)) \
           + 2 * t / (1 + 4 * (a * t) ** 2) * tau_y(a * t ** 2) @ tau_x(t) @ rho_z(np.arctan(2 * a * t)) @ se3Basis.G3