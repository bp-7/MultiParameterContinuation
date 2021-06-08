import autograd.numpy as np

from pymanopt.manifolds import Rotations
from pymanopt.manifolds.product import Product
from pymanopt.manifolds import Euclidean
from pymanopt.core.problem import Problem

import matplotlib.pyplot as plt


# Instantiate the SO(3) manifold
SO3 = Euclidean(3)

randSPDMat = np.array([[3, 1, 1], [1, 2, -0.5], [1, -0.5, 2]])

def A(mu):
    return 2 * mu * randSPDMat

def cost(S):
    #return S[0].T @ A(S[1]) @ S[0]
    v = S[0] - 3 * S[1]
    return v.T @ v

solutionSpace = Euclidean(3)
parameterSpace = Euclidean(3)
product = Product((solutionSpace, parameterSpace))

problem = Problem(product, cost=cost)

def DetermineBasisR():
    I3 = np.eye(3)
    R3Basis = [[w, np.array([0])] for w in I3]
    RBasis = [[np.array([0, 0, 0]), np.array([1.])]]

    basis = [R3Basis, RBasis]

    basisRearranged = [item for sublist in basis for item in sublist]

    return basisRearranged

def DetermineBasis():
    I3 = np.eye(3)
    R3Basis = [[w, np.array([0, 0, 0])] for w in I3]
    RpBasis = [[np.array([0, 0, 0]), w] for w in I3]

    basis = [R3Basis, RpBasis]

    basisRearranged = [item for sublist in basis for item in sublist]

    return basisRearranged

def RepresentSquareOperatorInBasis(operator, manifold, point):
    dimension = int(manifold.dim)
    basis = DetermineBasis()
    A = np.zeros((dimension, dimension))
    operatorResults = [operator(point, basis[i]) for i in range(dimension)]

    for i in range(dimension):
        temp = manifold.inner(point, basis[i], basis[i])
        for j in range(dimension):
            A[i, j] = manifold.inner(point, operatorResults[j], basis[i]) / temp

    return A

x = Euclidean(3).rand()
X = [x, x]
H = RepresentSquareOperatorInBasis(problem.hess, product, X)


def g(v):
    hessianSolution = H[:3, :3]
    hessianMixte = H[:3, 3:]

    xi = - np.linalg.inv(hessianSolution) @ hessianMixte @ v

    return cost([x + xi, x + v])
# Define cost function
R1Fixed = SO3.rand()

def alpha(mu):
    if np.linalg.norm(mu) <= 1e-15:
        return 1
    else:
        return  np.linalg.norm(mu)

def beta(mu):
    if np.linalg.norm(mu) <= 1e-15:
        return 1
    else:
        return 2 * np.linalg.norm(mu)

def gamma(mu):
    if np.linalg.norm(mu) <= 1e-15:
        return 1
    else:
        return 3 * np.linalg.norm(mu)

def R1(mu):
    return np.array([alpha(mu) * R1Fixed[0], beta(mu) * R1Fixed[1], gamma(mu) * R1Fixed[2]])

def costSO3(S):
    u = S[0] - R1(S[1])

    return np.trace(u.T @ u)

# Instantiate the problem
problemR33 = Problem(R33, costSO3)
problemSO3 = Problem(SO3, costSO3)

def Skew(A):
    return 0.5 * (A - A.T)

def Sym(A):
    return 0.5 * (A + A.T)

def diffProj(x, z, v):
    #return Skew(z.T @ v)
    return SO3.proj(x, x @ Skew(z.T @ v) + z @ Skew(x.T @ v))

def hess(x, z):
    egrad = problemSO3.egrad(x)
    ehess = problemSO3.ehess(x, SO3.tangent2ambient(x, z))
    #return Skew(x.T @ ehess - z @ Sym(x.T @ egrad))
    #return SO3.proj(x, ehess) - diffProj(x, SO3.tangent2ambient(x, z), egrad - SO3.proj(x, egrad))
    return SO3.proj(x, ehess) + diffProj(x, SO3.tangent2ambient(x, z), egrad)

