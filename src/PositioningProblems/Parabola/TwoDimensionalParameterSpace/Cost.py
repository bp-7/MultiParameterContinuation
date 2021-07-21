import autograd.numpy as np

from Helpers.Parameterizations.SE3Representation import matrixRepresentationOfSE3Element
from SE3Parameterizations.Parameterizations.Parabola import Parabola
from Helpers.Parameterizations.BasicSE3Transformations import rho_z, tau_x

def cost(S, h):
    R, r, t, mu = S[0], S[1], S[2], S[3]

    # Here h is fixed to 1
    h = 1
    I = np.eye(4)
    phi = matrixRepresentationOfSE3Element(R, r)
    u = phi @ Parabola(t[0], mu[1]) @ rho_z(0.5 * np.pi - mu[0]) @ tau_x(0.5 * h) - I
    v = phi @ Parabola(t[1], mu[1]) @ rho_z(- t[2]) @ tau_x(-0.5 * h) - I

    return np.trace(u.T @ u) + np.trace(v.T @ v)