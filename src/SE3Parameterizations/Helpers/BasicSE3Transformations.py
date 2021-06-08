import autograd.numpy as np

def rho_z(theta):
    rotation = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

    return np.block([[rotation, np.zeros((3, 1), dtype='float64')], [np.zeros((1, 3), dtype='float64'), 1.]])

def rho_y(theta):
    rotation = np.array([[np.cos(theta), 0., np.sin(theta)], [0., 1., 0.], [-np.sin(theta), 0, np.cos(theta)]])

    return np.block([[rotation, np.zeros((3, 1), dtype='float64')], [np.zeros((1, 3), dtype='float64'), 1.]])

def rho_x(theta):
    rotation = np.array([[1., 0., 0], [0., np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])

    return np.block([[rotation, np.zeros((3, 1), dtype='float64')], [np.zeros((1, 3), dtype='float64'), 1.]])

def tau_z(z):
    translation = np.reshape(np.array([0., 0., z]), (3, 1))

    return np.block([[np.eye(3), translation], [np.zeros((1, 3), dtype='float64'), 1.]])

def tau_y(y):
    translation = np.reshape(np.array([0., y, 0.]), (3, 1))

    return np.block([[np.eye(3), translation], [np.zeros((1, 3), dtype='float64'), 1.]])

def tau_x(x):
    translation = np.reshape(np.array([x, 0., 0.]), (3, 1))

    return np.block([[np.eye(3), translation], [np.zeros((1, 3), dtype='float64'), 1.]])

def tau(point):
    translation = np.reshape(point, (3, 1))

    return np.block([[np.eye(3), translation], [np.zeros((1, 3), dtype='float64'), 1.]])