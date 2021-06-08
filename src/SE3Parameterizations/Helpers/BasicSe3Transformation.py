import autograd.numpy as np

def Skew(w):
    return np.array([[0., -w[2], w[1]], [w[2], 0., -w[0]], [-w[1], w[0], 0.]], dtype=float)

class se3Basis:
    G1 = np.block(
        [[Skew(np.array([1., 0., 0.])), np.reshape(np.zeros(3), (3, 1))], [np.zeros((1, 4), dtype='float64')]])
    G2 = np.block(
        [[Skew(np.array([0., 1., 0.])), np.reshape(np.zeros(3), (3, 1))], [np.zeros((1, 4), dtype='float64')]])
    G3 = np.block(
        [[Skew(np.array([0., 0., 1.])), np.reshape(np.zeros(3), (3, 1))], [np.zeros((1, 4), dtype='float64')]])
    G4 = np.block(
        [[Skew(np.array([0., 0., 0.])), np.reshape(np.array([1., 0., 0.]), (3, 1))],
         [np.zeros((1, 4), dtype='float64')]])
    G5 = np.block(
        [[Skew(np.array([0., 0., 0.])), np.reshape(np.array([0., 1., 0.]), (3, 1))],
         [np.zeros((1, 4), dtype='float64')]])
    G6 = np.block(
        [[Skew(np.array([0., 0., 0.])), np.reshape(np.array([0., 0., 1.]), (3, 1))],
         [np.zeros((1, 4), dtype='float64')]])