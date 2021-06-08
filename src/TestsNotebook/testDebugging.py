import sys
sys.path.append('../')

import autograd.numpy as np

from pymanopt.manifolds import Sphere

# Dimension of the sphere
solutionSpaceDimension = 3

# Instantiate the unit sphere manifold
unitSphere = Sphere(solutionSpaceDimension)

from pymanopt.manifolds import Euclidean

# Dimension of the parameter space
parameterSpaceDimension = 2

# Instantiate the parameter space
parameterSpace = Euclidean(parameterSpaceDimension)

from pymanopt.manifolds import Product

productManifold = Product([unitSphere, parameterSpace])

def A(mu):
    sigma = 100
    lambda0 = 1 + np.exp((mu[0] - mu[1]) / sigma)
    lambda1 = 2 + np.exp((3 * mu[0] + mu[1]) / sigma)
    lambda2 = 3 + np.exp((mu[0] + mu[1]) / sigma)

    return np.array([[lambda0, -1, 0], [-1, lambda1, -1], [0, -1, lambda2]])


def DA(mu, v):
    sigma = 100
    Dlambda0 = (v[0] - v[1]) / sigma * np.exp((mu[0] - mu[1]) / sigma)
    Dlambda1 = (3 * v[0] + v[1]) / sigma * np.exp((3 * mu[0] + mu[1]) / sigma)
    Dlambda2 = (v[0] + v[1]) / sigma * np.exp((mu[0] + mu[1]) / sigma)

    return np.array([[Dlambda0, 0, 0], [0, Dlambda1, 0], [0, 0, Dlambda2]])

# Define derivatives of gradient wrt solution/parameter
def differentialSolutionAlongV(x, mu, xi):
    return A(mu) @ xi - 2 * (x.T @ A(mu) @ xi) * x - (x.T @ A(mu) @ x) * xi

def differentialParameterAlongV(x, mu, v):
    return (np.eye(solutionSpaceDimension) - np.outer(x, x)) @ DA(mu, v) @ x


from pymanopt.core.problem import Problem

def cost(S):
    return np.inner(S[0], A(S[1]) @ S[0])

problem = Problem(productManifold, cost=cost)

x = productManifold.rand()
v = productManifold.randvec(x)

print(problem.hess())

def ConstructBasisForEigenvalueProblem(N):
    IN = np.eye(N)
    IP = np.eye(2)

    RNBasisPart = [[w, np.zeros(2)] for w in IN]

    RPBasisPart = [[np.zeros(N), w] for w in IP]

    basis = [RNBasisPart, RPBasisPart]

    basisRearranged = [item for sublist in basis for item in sublist]

    return basisRearranged

def RepresentSquareOperatorInEigenvalueProblemBasis(operator, manifold, point):
    dimension = int(manifold.dim)
    basis = ConstructBasisForEigenvalueProblem(3)
    A = np.zeros((dimension, dimension))
    operatorResults = [operator(point, basis[i]) for i in range(dimension)]

    for i in range(dimension):
        temp = manifold.inner(point, basis[i], basis[i])
        for j in range(dimension):
            A[i, j] = manifold.inner(point, operatorResults[j], basis[i]) / temp

    return A