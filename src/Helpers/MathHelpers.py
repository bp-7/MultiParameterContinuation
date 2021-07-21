import autograd.numpy as np

def Skew(w):
    return np.array([[0., -w[2], w[1]], [w[2], 0., -w[0]], [-w[1], w[0], 0.]], dtype=float)

def InvSkew(S):
    return np.array([S[2, 1], S[0, 2], S[1, 0]])

def InvSE3(phi):
    rotation, translation = phi[:3, :3], phi[:3, 3]
    return np.block([[rotation.T, np.reshape(- rotation.T @ translation, (3, 1))], [np.zeros((1, 3), dtype='float64'), 1]])

