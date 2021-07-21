import numpy as np
from Helpers.MathHelpers import Skew

def WriteMatrixInEuclideanBasisAtGivenPoint(matrixVectorFunction, x, mu, dimension):
    spaceDimension = len(x)
    indices = np.arange(dimension)
    A = np.zeros((spaceDimension, dimension))

    for index in indices:
        v = np.zeros(dimension)
        v[index] = 1

        A[:, index] = matrixVectorFunction(x, mu, v)

    return A

def ConstructNormalizedTotalBasisForBergerManifold(N):
    I3 = np.eye(3)
    IN = np.eye(N)
    IP = np.eye(2)

    SO3BasisPart = [[Skew(w) / np.sqrt(2.), np.zeros(3), np.zeros(N), np.zeros(2)] for w in I3]

    R3BasisPart = [[np.zeros((3, 3)), w, np.zeros(N), np.zeros(2)] for w in I3]

    RNBasisPart = [[np.zeros((3, 3)), np.zeros(3), w, np.zeros(2)] for w in IN]

    RPBasisPart = [[np.zeros((3, 3)), np.zeros(3), np.zeros(3), w] for w in IP]

    basis = [SO3BasisPart, R3BasisPart, RNBasisPart, RPBasisPart]

    basisRearranged = [item for sublist in basis for item in sublist]

    return basisRearranged

def RepresentSquareOperatorInTotalNormalizedBergerBasis(operator, manifold, point):
    dimension = int(manifold.dim)
    basis = ConstructNormalizedTotalBasisForBergerManifold(3)
    A = np.zeros((dimension, dimension))
    operatorResults = [operator(point, basis[i]) for i in range(dimension)]

    for i in range(dimension):
        temp = manifold.inner(point, basis[i], basis[i])
        for j in range(dimension):
            A[i, j] = manifold.inner(point, operatorResults[j], basis[i]) / temp

    return A

def RepresentSquareOperatorInTotalBergerBasis(operator, manifold, point):
    dimension = int(manifold.dim)
    basis = ConstructTotalBasisForBergerManifold(int(dimension - 6 - point[-1].shape[0]))
    A = np.zeros((dimension, dimension))
    operatorResults = [operator(point, basis[i]) for i in range(dimension)]

    for i in range(dimension):
        temp = manifold.inner(point, basis[i], basis[i])
        for j in range(dimension):
            A[i, j] = manifold.inner(point, operatorResults[j], basis[i]) / temp

    return A

def ConstructMetricMatrixForBergerManifold(N):
    basis = ConstructBasisForBergerManifold(N)
    G = np.zeros((6 + N, 6 + N))

    for i in range(6 + N):
        bi = basis[i]
        for j in range(6 + N):
            bj = basis[j]
            G[i, j] = np.trace(bi[0].T @ bj[0]) + np.dot(bi[1], bj[1]) + np.dot(bi[2], bj[2])

    return G

def RepresentSquareOperatorInBergerBasis(operator, manifold, point):
    dimension = int(manifold.dim)
    basis = ConstructBasisForBergerManifold(dimension - 6)
    A = np.zeros((dimension, dimension))
    operatorResults = [operator(point, basis[i]) for i in range(dimension)]

    for i in range(dimension):
        temp = manifold.inner(point, basis[i], basis[i])
        for j in range(dimension):
            A[i, j] = manifold.inner(point, operatorResults[j], basis[i]) / temp

    return A

def RepresentRectangularOperatorFromEuclideanToBergerManifold(operator, manifold, point, euclideanDimension):
    dimension = int(manifold.dim)
    euclideanBasis = np.eye(euclideanDimension)
    bergerBasis = ConstructBasisForBergerManifold(dimension - 6)
    A = np.empty((dimension, euclideanDimension))
    operatorResults = [operator(point, w) for w in euclideanBasis]
    for i in range(dimension):
        temp = manifold.inner(point, bergerBasis[i], bergerBasis[i])
        for j in range(euclideanDimension):
            A[i, j] = manifold.inner(point, operatorResults[j], bergerBasis[i]) / temp

    return A

############################################################

def ConstructBasisSE3Manifold2():
    I3 = np.eye(3)

    SO3BasisPart = [[Skew(w), np.zeros(3), np.zeros((3, 3)), np.zeros(3)] for w in I3]

    R3BasisPart = [[np.zeros((3, 3)), w, np.zeros((3, 3)), np.zeros(3)] for w in I3]

    SO3BasisPart2 = [[np.zeros((3, 3)), np.zeros(3), Skew(w), np.zeros(3)] for w in I3]

    R3BasisPart2 = [[np.zeros((3, 3)), np.zeros(3), np.zeros((3, 3)), w] for w in I3]

    basis = [SO3BasisPart, R3BasisPart, SO3BasisPart2, R3BasisPart2]

    basisRearranged = [item for sublist in basis for item in sublist]

    return basisRearranged

def ConstructTotalBasisForBergerManifold(N, parameterSpaceDimension):
    I3 = np.eye(3)
    IN = np.eye(N)
    IP = np.eye(parameterSpaceDimension)

    SO3BasisPart = [[Skew(w), np.zeros(3), np.zeros(N), np.zeros(parameterSpaceDimension)] for w in I3]

    R3BasisPart = [[np.zeros((3, 3)), w, np.zeros(N), np.zeros(parameterSpaceDimension)] for w in I3]

    RNBasisPart = [[np.zeros((3, 3)), np.zeros(3), w, np.zeros(parameterSpaceDimension)] for w in IN]

    RPBasisPart = [[np.zeros((3, 3)), np.zeros(3), np.zeros(N), w] for w in IP]

    basis = [SO3BasisPart, R3BasisPart, RNBasisPart, RPBasisPart]

    basisRearranged = [item for sublist in basis for item in sublist]

    return basisRearranged

def ConstructBasisForBergerManifold(N):
    I3 = np.eye(3)
    IN = np.eye(N)

    SO3BasisPart = [[Skew(w), np.zeros(3), np.zeros(N)] for w in I3]

    R3BasisPart = [[np.zeros((3, 3)), w, np.zeros(N)] for w in I3]

    RNBasisPart = [[np.zeros((3, 3)), np.zeros(3), w] for w in IN]

    basis = [SO3BasisPart, R3BasisPart, RNBasisPart]

    basisRearranged = [item for sublist in basis for item in sublist]

    return basisRearranged

def RepresentRectangularOperatorFromSolutionSpaceToSE32(operator, startingManifold, arrivalManifold, point, Fpoint):
    startingDimension = int(startingManifold.dim)
    arrivalDimension = int(arrivalManifold.dim)
    arrivalBasis = ConstructBasisSE3Manifold2()
    bergerBasis = ConstructBasisForBergerManifold(startingDimension - 6)
    A = np.empty((arrivalDimension, startingDimension))
    operatorResults = [operator(point, w) for w in bergerBasis]
    for i in range(arrivalDimension):
        temp = arrivalManifold.inner(Fpoint, arrivalBasis[i], arrivalBasis[i])
        for j in range(startingDimension):
            A[i, j] = arrivalManifold.inner(Fpoint, operatorResults[j], arrivalBasis[i]) / temp

    return A

def RepresentRectangularOperatorFromTotalSpaceToSE32(operator, startingManifold, arrivalManifold, point, Fpoint, parameterSpaceDimension):
    startingDimension = int(startingManifold.dim)
    arrivalDimension = int(arrivalManifold.dim)
    arrivalBasis = ConstructBasisSE3Manifold2()
    bergerBasis = ConstructTotalBasisForBergerManifold(startingDimension - 6 - parameterSpaceDimension, parameterSpaceDimension)
    A = np.empty((arrivalDimension, startingDimension))
    operatorResults = [operator(point, w) for w in bergerBasis]
    for i in range(arrivalDimension):
        temp = arrivalManifold.inner(Fpoint, arrivalBasis[i], arrivalBasis[i])
        for j in range(startingDimension):
            A[i, j] = arrivalManifold.inner(Fpoint, operatorResults[j], arrivalBasis[i]) / temp

    return A

def RepresentRectangularOperatorFromParameterSpaceToSE32(operator, startingManifold, arrivalManifold, point, Fpoint):
    startingDimension = int(startingManifold.dim)
    arrivalDimension = int(arrivalManifold.dim)
    arrivalBasis = ConstructBasisSE3Manifold2()
    startingBasis = np.eye(startingDimension)
    A = np.empty((arrivalDimension, startingDimension))
    operatorResults = [operator(point, w) for w in startingBasis]
    for i in range(arrivalDimension):
        temp = arrivalManifold.inner(Fpoint, arrivalBasis[i], arrivalBasis[i])
        for j in range(startingDimension):
            A[i, j] = arrivalManifold.inner(Fpoint, operatorResults[j], arrivalBasis[i]) / temp

    return A
