import autograd.numpy as np
from Helpers.Parameterizations.BasicSE3Transformations import tau, tau_x, tau_y, rho_x, rho_z
from Helpers.Parameterizations.BasicSe3Transformation import se3Basis

def TorusDerivativeS(s, t, rt, Rt, offset):
    return 2 * np.pi \
           * tau(offset) \
           @ rho_z(2 * np.pi * s) \
           @ se3Basis.G3 \
           @ tau_y(Rt) \
           @ rho_x(2 * np.pi * t) \
           @ tau_y(rt) \
           @ rho_x(0.5 * np.pi)

def TorusDerivativeT(s, t, rt, Rt, offset):
    return 2 * np.pi \
           * tau(offset) \
           @ rho_z(2 * np.pi * s) \
           @ tau_y(Rt) \
           @ rho_x(2 * np.pi * t) \
           @ se3Basis.G1 \
           @ tau_y(rt) \
           @ rho_x(0.5 * np.pi)

def TorusDerivativeLargeRadius(s, t, rt, Rt, offset):
    return tau(offset) \
           @ rho_z(2 * np.pi * s) \
           @ tau_y(Rt) \
           @ se3Basis.G5 \
           @ rho_x(2 * np.pi * t) \
           @ tau_y(rt) \
           @ rho_x(0.5 * np.pi)

def TorusDerivativeSmallRadius(s, t, rt, Rt, offset):
    return tau(offset) \
           @ rho_z(2 * np.pi * s) \
           @ tau_y(Rt) \
           @ rho_x(2 * np.pi * t) \
           @ tau_y(rt) \
           @ se3Basis.G5 \
           @ rho_x(0.5 * np.pi)