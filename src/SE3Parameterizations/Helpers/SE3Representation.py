import autograd.numpy as np

def matrixRepresentationOfSE3Element(rotation, translation):
    return np.block([[rotation, np.reshape(translation, (3, 1))], [np.zeros((1, 3), dtype='float64'), 1]])

def matrixRepresentationOfse3Element(element):
    return np.block([[element[0], np.reshape(element[1], (3, 1))], [np.zeros((1, 4), dtype='float64')]])

def tupleRepresentationOfSE3Element(element):
    return element[:3, :3], element[:3, 3]