import autograd.numpy as np

from Helpers.MathHelpers import Skew, InvSE3
from Helpers.Parameterizations.SE3Representation import matrixRepresentationOfSE3Element
from SE3Parameterizations.Parameterizations.Helix import helix
from SE3Parameterizations.Parameterizations.Torus import torus
from Helpers.Parameterizations.BasicSE3Transformations import rho_z, rho_x
from Helpers.Parameterizations.BasicSe3Transformation import se3Basis
from SE3Parameterizations.Derivatives.Helix import InverseHelixDerivativeOffsetAngle, \
    InverseHelixDerivativeHelixRadius, \
    InverseHelixDerivativeHelixAngle, \
    InverseHelixDerivativeTrajectoryParameter, \
    CuttingAngleDerivativeHelixAngle, \
    CuttingAngleDerivativeT

from SE3Parameterizations.Derivatives.Torus import TorusDerivativeS, \
    TorusDerivativeT, \
    TorusDerivativeLargeRadius, \
    TorusDerivativeSmallRadius

def systemEvaluation(S, wheelProfileParameter, helixLength, offsetWheel):
    R, r, t, mu = S[0], S[1], S[2], S[3]
    phi = matrixRepresentationOfSE3Element(R, r)

    Rt, rt, helixAngle1, helixRadius1, helixAngle2, helixRadius2, offsetAngle, trajectoryParameter = mu

    grindingMark = -0.5 * np.pi - helixAngle1

    W1 = torus(0, wheelProfileParameter, rt, Rt, offsetWheel)
    W2 = torus(t[1], t[2], rt, Rt, offsetWheel)
    C1 = helix(trajectoryParameter, helixRadius1, helixAngle1, helixLength)
    C2 = helix(t[4], helixRadius2, helixAngle2, helixLength, offsetAngle)

    u = phi @ W1 @ rho_z(grindingMark) @ rho_x(-np.arctan(np.tan(t[0]) / np.cos(helixAngle1))) @ InvSE3(C1)

    v = phi @ W2 @ rho_z(t[3]) @ rho_x(-np.arctan(np.tan(t[5]) / np.cos(helixAngle1))) @ InvSE3(C2)

    return [u[:3, :3], u[:3, 3], v[:3, :3], v[:3, 3]]

def InvSkew(S):
    return np.array([S[2, 1], S[0, 2], S[1, 0]])

def E1(t, mu, wheelProfileParameter, helixLength, offsetWheel):
    Rt, rt, helixAngle, helixRadius, trajectoryParameter = mu[0], mu[1], mu[2], mu[3], mu[7]
    grindingMark = -0.5 * np.pi - helixAngle

    return torus(0., wheelProfileParameter, rt, Rt, offsetWheel) \
           @ rho_z(grindingMark) \
           @ rho_x(-np.arctan(np.tan(t[0]) / np.cos(helixAngle))) \
           @ InvSE3(helix(trajectoryParameter, helixRadius, helixAngle, helixLength))

def E2(t, mu, wheelProfileParameter, helixLength, offsetWheel):
    Rt, rt, helixAngle, helixRadius, offsetAngle = mu[0], mu[1], mu[4], mu[5], mu[6]

    return torus(t[1], t[2], rt, Rt, offsetWheel) \
           @ rho_z(t[3]) \
           @ rho_x(-np.arctan(np.tan(t[5]) / np.cos(helixAngle))) \
           @ InvSE3(helix(t[4], helixRadius, helixAngle, helixLength, offsetAngle))

def Adjoint(phi):
    R = phi[:3, :3]
    r = phi[:3, 3]

    return np.block([[R, Skew(r) @ R], [np.zeros((3, 3)), R]])

def differentialSE3Fi(E, xi):
    return Adjoint(InvSE3(E)) @ xi

def differentialSE3F(E1, E2, xi):
    d0 = differentialSE3Fi(E1, xi)
    d1 = differentialSE3Fi(E2, xi)

    return [Skew(d0[:3]), d0[3:6], Skew(d1[:3]), d1[3:6]]

def differentialRnF(S, v, wheelProfileParameter, helixLength, offsetWheel):
    R, r, t, mu = S[0], S[1], S[2], S[3]
    Rt, rt, helixAngle1, helixRadius1, helixAngle2, helixRadius2, offsetAngle, trajectoryParameter = mu

    phi = matrixRepresentationOfSE3Element(R, r)

    grindingMark = -0.5 * np.pi - helixAngle1

    d0 = -v[0] * CuttingAngleDerivativeT(t[0], helixAngle1) * phi @ torus(0, wheelProfileParameter, rt, Rt, offsetWheel) @ rho_z(grindingMark) \
         @ rho_x(-np.arctan(np.tan(t[0]) / np.cos(helixAngle1))) @ se3Basis.G1 @ InvSE3(helix(trajectoryParameter, helixAngle1, helixRadius1, helixLength))

    d1 = v[1] * phi @ TorusDerivativeS(t[1], t[2], rt, Rt, offsetWheel) @ rho_z(t[3]) \
         @ rho_x(-np.arctan(np.tan(t[5]) / np.cos(helixAngle2))) @ InvSE3(helix(t[4], helixRadius2, helixAngle2, helixLength, offsetAngle)) \
         + v[2] * phi @ TorusDerivativeT(t[1], t[2], rt, Rt, offsetWheel) @ rho_z(t[3]) \
         @ rho_x(-np.arctan(np.tan(t[5]) / np.cos(helixAngle2))) @ InvSE3(helix(t[4], helixRadius2, helixAngle2, helixLength, offsetAngle)) \
         + v[3] * phi @ torus(t[1], t[2], rt, Rt, offsetWheel) @ rho_z(t[3]) \
         @ se3Basis.G3 @ rho_x(-np.arctan(t[5] / np.cos(helixAngle2))) @ InvSE3(helix(t[4], helixRadius2, helixAngle2, helixLength, offsetAngle)) \
         + v[4] * phi @ torus(t[1], t[2], rt, Rt, offsetWheel) @ rho_z(t[3]) \
         @ rho_x(-np.arctan(np.tan(t[5]) / np.cos(helixAngle2))) @ InverseHelixDerivativeTrajectoryParameter(t[4], helixRadius2, helixAngle2, helixLength, offsetAngle) \
         - v[5] * CuttingAngleDerivativeT(t[5], helixAngle2) * phi @ torus(t[1], t[2], rt, Rt, offsetWheel) @ rho_z(t[3]) \
         @ rho_x(-np.arctan(np.tan(t[5]) / np.cos(helixAngle2))) @ se3Basis.G1 @ helix(t[4], helixRadius2, helixAngle2, helixLength, offsetAngle)

    return [d0[:3, :3], d0[:3, 3], d1[:3, :3], d1[:3, 3]]

def differentialSolution(S, zeta, wheelProfileParameter, helixLength, offsetWheel):
    R, r, t, mu = S[0], S[1], S[2], S[3]
    xi = np.concatenate((InvSkew(zeta[0]), zeta[1]))
    E_1, E_2 = E1(t, mu, wheelProfileParameter, helixLength, offsetWheel), E2(t, mu, wheelProfileParameter, helixLength, offsetWheel)

    d0 = differentialSE3F(E_1, E_2, xi)
    d1 = differentialRnF(S, zeta[2], wheelProfileParameter, helixLength, offsetWheel)

    return [d0[i] + d1[i] for i in range(len(d0))]


def differentialParameter(S, v, wheelProfileParameter, helixLength, offsetWheel):
    R, r, t, mu = S[0], S[1], S[2], S[3]
    Rt, rt, helixAngle1, helixRadius1, helixAngle2, helixRadius2, offsetAngle, trajectoryParameter = mu

    phi = matrixRepresentationOfSE3Element(R, r)

    grindingMark = -0.5 * np.pi - helixAngle1

    W1 = torus(0, wheelProfileParameter, rt, Rt, offsetWheel)
    W2 = torus(t[1], t[2], rt, Rt, offsetWheel)
    C1 = helix(trajectoryParameter, helixRadius1, helixAngle1, helixLength)
    C2 = helix(t[4], helixRadius2, helixAngle2, helixLength, offsetAngle)

    d0 = v[0] * phi @ TorusDerivativeLargeRadius(0, wheelProfileParameter, rt, Rt, offsetWheel) @ rho_z(grindingMark) @ rho_x(-np.arctan(np.tan(t[0]) / np.cos(helixAngle1))) @ InvSE3(C1)  \
         + v[1] * phi @ TorusDerivativeSmallRadius(0, wheelProfileParameter, rt, Rt, offsetWheel) @ rho_z(grindingMark) @ rho_x(-np.arctan(np.tan(t[0]) / np.cos(helixAngle1))) @ InvSE3(C1) \
         + v[2] * phi @ W1 @ rho_z(grindingMark) @ rho_x(-np.arctan(np.tan(t[0]) / np.cos(helixAngle1))) @ InverseHelixDerivativeHelixAngle(trajectoryParameter, helixRadius1, helixAngle1, helixLength) \
         - v[2] * CuttingAngleDerivativeHelixAngle(t[0], helixAngle1) * phi @ W1 @ rho_z(grindingMark) @ rho_x(-np.arctan(np.tan(t[0]) / np.cos(helixAngle1))) @ se3Basis.G1 @ InvSE3(C1) \
         + v[3] * phi @ W1 @ rho_z(grindingMark) @ rho_x(-np.arctan(np.tan(t[0]) / np.cos(helixAngle1))) @ InverseHelixDerivativeHelixRadius(trajectoryParameter, helixRadius1, helixAngle1, helixLength) \
         + v[7] * phi @ W1 @ rho_z(grindingMark) @ rho_x(-np.arctan(np.tan(t[0]) / np.cos(helixAngle1))) @ InverseHelixDerivativeTrajectoryParameter(trajectoryParameter, helixRadius1, helixAngle1, helixLength)

    d1 = v[0] * phi @ TorusDerivativeLargeRadius(t[1], t[2], rt, Rt, offsetWheel) @ rho_z(grindingMark) @ rho_x(-np.arctan(np.tan(t[5]) / np.cos(helixAngle2))) @ InvSE3(C2)  \
         + v[1] * phi @ TorusDerivativeSmallRadius(t[1], t[2], rt, Rt, offsetWheel) @ rho_z(grindingMark) @ rho_x(-np.arctan(np.tan(t[5]) / np.cos(helixAngle2))) @ InvSE3(C2) \
         + v[4] * phi @ W2 @ rho_z(t[3]) @ rho_x(-np.arctan(np.tan(t[5]) / np.cos(helixAngle2))) @ InverseHelixDerivativeHelixAngle(t[4], helixRadius2, helixAngle2, helixLength, offsetAngle) \
         - v[4] * CuttingAngleDerivativeHelixAngle(t[5], helixAngle2) * phi @ W2 @ rho_z(t[3]) @ rho_x(-np.arctan(np.tan(t[5]) / np.cos(helixAngle2))) @ se3Basis.G1 @ InvSE3(C2) \
         + v[5] * phi @ W2 @ rho_z(t[3]) @ rho_x(-np.arctan(np.tan(t[5]) / np.cos(helixAngle2))) @ InverseHelixDerivativeHelixRadius(t[4], helixRadius2, helixAngle2, helixLength, offsetAngle) \
         + v[6] * phi @ W2 @ rho_z(t[3]) @ rho_x(-np.arctan(np.tan(t[5]) / np.cos(helixAngle2))) @ InverseHelixDerivativeOffsetAngle(t[4], helixRadius2, helixAngle2, helixLength, offsetAngle)

    return [d0[:3, :3], d0[:3, 3], d1[:3, :3], d1[:3, 3]]