import autograd.numpy as np

from Helpers.MathHelpers import Skew, InvSkew
from Helpers.Parameterizations.SE3Representation import matrixRepresentationOfSE3Element

from Helpers.Parameterizations.BasicSE3Transformations import tau_x, rho_z
from Helpers.Parameterizations.BasicSe3Transformation import se3Basis
from SE3Parameterizations.Parameterizations.Parabola import Parabola
from SE3Parameterizations.Derivatives.Parabola import ParabolaDerivative, ParabolaParameterDerivative

def systemEvaluation(S):
    R, r, t, mu = S[0], S[1], S[2], S[3]
    phi = matrixRepresentationOfSE3Element(R, r)

    u = phi @ Parabola(t[0], mu[1]) @ rho_z(0.5 * np.pi - mu[0]) @ tau_x(0.5 * mu[2])
    v = phi @ Parabola(t[1], mu[1]) @ rho_z(- S[2][2]) @ tau_x(-0.5 * mu[2])

    return [u[:3, :3], u[:3, 3], v[:3, :3], v[:3, 3]]

def E1(t1, mu):
    return Parabola(t1, mu[1]) @ rho_z(-mu[0] + 0.5 * np.pi) @ tau_x(0.5 * mu[2])

def E2(t2, t3, mu):
    return Parabola(t2, mu[1]) @ rho_z(-t3) @ tau_x(-0.5 * mu[2])

def Adjoint(phi):
    R = phi[:3, :3]
    r = phi[:3, 3]

    return np.block([[R, Skew(r) @ R], [np.zeros((3, 3)), R]])

def differentialSE3Fi(E, xi):
    return Adjoint(np.linalg.inv(E)) @ xi

def differentialSE3F(E1, E2, xi):
    d0 = differentialSE3Fi(E1, xi)
    d1 = differentialSE3Fi(E2, xi)

    return [Skew(d0[:3]), d0[3:6], Skew(d1[:3]), d1[3:6]]

def differentialRnF(S, v):
    R, r, t, mu = S[0], S[1], S[2], S[3]
    phi = matrixRepresentationOfSE3Element(R, r)
    d0 = v[0] * phi @ ParabolaDerivative(t[0], mu[1]) @ rho_z(-mu[0] + 0.5 * np.pi) @ tau_x(0.5 * mu[2])
    d1 = v[1] * phi @ ParabolaDerivative(t[1], mu[1]) @ rho_z(-t[2]) @ tau_x(-0.5 * mu[2]) \
         - v[2] * phi @ Parabola(t[1], mu[1]) @ rho_z(-t[2]) @ se3Basis.G3 @ tau_x(-0.5 * mu[2])

    return [d0[:3, :3], d0[:3, 3], d1[:3, :3], d1[:3, 3]]

def differentialSolution(S, zeta):
    R, r, t, mu = S[0], S[1], S[2], S[3]
    xi = np.concatenate((InvSkew(zeta[0]), zeta[1]))
    E_1, E_2 = E1(t[0], mu), E2(t[1], t[2], mu)

    d0 = differentialSE3F(E_1, E_2, xi)
    d1 = differentialRnF(S, zeta[2])

    return [d0[i] + d1[i] for i in range(len(d0))]


def differentialParameter(S, v):
    R, r, t, mu = S[0], S[1], S[2], S[3]
    phi = matrixRepresentationOfSE3Element(R, r)
    d0 = v[1] * phi @ ParabolaParameterDerivative(t[0], mu[1]) @ rho_z(-mu[0] + 0.5 * np.pi) @ tau_x(0.5 * mu[2])  \
         - v[0] * phi @ Parabola(t[0], mu[1]) @ rho_z(-mu[0] + 0.5 * np.pi) @ se3Basis.G3 @ tau_x(0.5 * mu[2]) \
         + 0.5 * v[2] * phi @ Parabola(t[0], mu[1]) @ rho_z(-mu[0] + 0.5 * np.pi) @ tau_x(0.5 * mu[2]) @ se3Basis.G4

    d1 = v[1] * phi @ ParabolaParameterDerivative(t[1], mu[1]) @ rho_z(-t[2]) @ tau_x(-0.5 * mu[2]) \
         - 0.5 * v[2] * phi @ Parabola(t[1], mu[1]) @ rho_z(-t[2]) @ tau_x(-0.5 * mu[2]) @ se3Basis.G4

    return [d0[:3, :3], d0[:3, 3], d1[:3, :3], d1[:3, 3]]
