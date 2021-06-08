import autograd.numpy as np
from SE3Parameterizations.Helpers.BasicSE3Transformations import tau_x, tau_y, rho_z

def Parabola(t, a):
    return tau_y(a * t ** 2) @ tau_x(t) @ rho_z(np.arctan(2 * a * t))