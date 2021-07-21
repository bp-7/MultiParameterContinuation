import autograd.numpy as np

from Helpers.MathHelpers import InvSE3
from Helpers.Parameterizations.SE3Representation import matrixRepresentationOfSE3Element
from SE3Parameterizations.Parameterizations.Helix import helix
from SE3Parameterizations.Parameterizations.Torus import torus
from Helpers.Parameterizations.BasicSE3Transformations import rho_z, rho_x

def cost(S, wheelProfileParameter, helixLength, offsetWheel):
    phi = matrixRepresentationOfSE3Element(S[0], S[1])
    t, mu = S[2], S[3]

    Rt, rt, helixAngle1, helixRadius1, helixAngle2, helixRadius2, offsetAngle, trajectoryParameter = mu


    grindingMark = -0.5 * np.pi - helixAngle1

    W1 = torus(0, wheelProfileParameter, rt, Rt, offsetWheel)
    W2 = torus(t[1], t[2], rt, Rt, offsetWheel)
    C1 = helix(trajectoryParameter, helixRadius1, helixAngle1, helixLength)
    C2 = helix(t[4], helixRadius2, helixAngle2, helixLength, offsetAngle)

    I = np.eye(4)

    u = phi @ W1 @ rho_z(grindingMark) @ rho_x(-np.arctan(np.tan(t[0]) / np.cos(helixAngle1))) @ InvSE3(C1) - I

    v = phi @ W2 @ rho_z(t[3]) @ rho_x(-np.arctan(np.tan(t[5]) / np.cos(helixAngle2))) @ InvSE3(C2) - I

    return np.trace(u.T @ u) + np.trace(v.T @ v)