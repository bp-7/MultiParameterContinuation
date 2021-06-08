import numpy as np
from SE3Parameterizations.Derivatives.Parabola import ParabolaDerivative

def Skew(w):
    return np.array([[0., -w[2], w[1]], [w[2], 0., -w[0]], [-w[1], w[0], 0.]], dtype=float)

def E1(t1, a, theta):

def Adjoint(phi):
    R = phi[:3, :3]
    r = phi[:3, 3]

    return np.block([[R, Skew(r) @ R], [np.zeros((3, 3)), R]])

def differentialSE3Fi(E, xi):
    return Adjoint(np.linalg.inv(E)) @ xi

def differentialSE3F(E1, E2, xi):
    return np.array([differentialSE3Fi(E1, xi), differentialSE3Fi(E2, xi)])

def differentialRnF(S, v):
    phi, t, mu = S[0], S[1], S[2]


    return np.array([v[0] * phi @ ParabolaDerivative(t[0], mu[1]) ])