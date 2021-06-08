import autograd.numpy as np

from pymanopt.manifolds import Rotations
from pymanopt.manifolds import Euclidean
from pymanopt.core.problem import Problem

import matplotlib.pyplot as plt


# Instantiate the SO(3) manifold
SO3 = Rotations(3)
R33 = Euclidean(3, 3)
parameterSpaceDimension = 2
parameterSpace = Euclidean(parameterSpaceDimension)

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
    Mu = mu / np.linalg.norm(mu)
    return np.array([alpha(Mu) * R1Fixed[0], alpha(Mu) * R1Fixed[1], alpha(Mu) * R1Fixed[2]])

def rho_z(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

def costSO3(S):
    u = S[0] - rho_z(S[1][0]) @ rho_z(S[1][1])

    return np.trace(u.T @ u)

def Skew(A):
    return 0.5 * (A - A.T)

def SkewMat(w):
    return np.array([[0., -w[2], w[1]], [w[2], 0., -w[0]], [-w[1], w[0], 0.]], dtype=float)

def diffProj(x, z, v):
    return product.proj(x, [x[0] @ Skew(z[0].T @ v[0]) + z[0] @ Skew(x[0].T @ v[0]), np.zeros(x[1].shape)])

def hess(x, z):
    egrad = problem.egrad(x)
    ehess = problem.ehess(x, [SO3.tangent2ambient(x[0], z[0]), z[1]])
    return product.proj(x, ehess) + diffProj(x, [SO3.tangent2ambient(x[0], z[0]), z[1]], egrad)


from pymanopt.manifolds.product import Product
product = Product((SO3, parameterSpace))

problem = Problem(product, cost=costSO3)

def ConstructNormalizedTotalBasisForBergerManifold(N):
    I3 = np.eye(3)
    IP = np.eye(parameterSpaceDimension)

    SO3BasisPart = [[SkewMat(w) / np.sqrt(2.), np.zeros(parameterSpaceDimension)] for w in I3]

    RPBasisPart = [[np.zeros((3, 3)), w] for w in IP]

    basis = [SO3BasisPart, RPBasisPart]

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

x = product.rand()
H = RepresentSquareOperatorInTotalNormalizedBergerBasis(hess, product, [R1(x[1]), x[1]])
Hg = H[3:, 3:] - H[3:, :3] @ np.linalg.inv(H[:3, :3]) @ H[:3, 3:]

x = product.rand()
v = product.randvec(x)
t = np.logspace(-8,1, 100)
firstOrderTerm = product.inner(x, v, problem.grad(x))
secondOrderTerm = product.inner(x, v, hess(x, v))
err = [np.abs(costSO3(product.exp(x, ti * v)) -  costSO3(x) - ti * firstOrderTerm - 0.5 * ti ** 2 * secondOrderTerm) for ti in t]

import matplotlib.pyplot as plt
plt.loglog(t, err, 'b-', t, t** 3, 'r-', t, t ** 2, 'g-')



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


# Plot second-order error approximation
x = SO3.rand()
u = SO3.randvec(x)
t = np.logspace(-8, 1, 100)
firstOrderTerm = SO3.inner(x, u, problemSO3.grad(x))
secondOrderTerm = 0.5 * SO3.inner(x, u, problemSO3.hess(x, u))
err = [np.abs(costSO3(SO3.retr(x, ti * u)) - costSO3(x) - ti * firstOrderTerm - ti ** 2 * secondOrderTerm) for ti in t]
plt.loglog(t, err, 'b-', t, t ** 3, 'r-', t, t ** 2, 'g-')

x = SO3.rand()
u = SO3.randvec(x)
t = np.logspace(-8, 1, 100)
firstOrderTerm = SO3.inner(x, u, problemSO3.grad(x))
secondOrderTerm = 0.5 * SO3.inner(x, u, problemSO3.hess(x, u))
err = [np.abs(costSO3(SO3.retr(x, ti * u)) - costSO3(x) - ti * firstOrderTerm - ti ** 2 * secondOrderTerm) for ti in t]
plt.loglog(t, err, 'b-', t, t ** 3, 'r-', t, t ** 2, 'g-')

x = R33.rand()
u = R33.randvec(x)
t = np.logspace(-8, 1, 100)
firstOrderTerm = R33.inner(x, u, problemR33.grad(x))
secondOrderTerm = 0.5 * R33.inner(x, u, problemR33.hess(x, u))
err = [np.abs(costSO3(R33.retr(x, ti * u)) - costSO3(x) - ti * firstOrderTerm - ti ** 2 * secondOrderTerm) for ti in t]
plt.loglog(t, err, 'b-', t, t ** 3, 'r-', t, t ** 2, 'g-')

import matplotlib.pyplot as plt
x = productMani.rand()#tuple(initialSolution) + (initialParameter, )
u = productMani.randvec(x)
t = np.logspace(-8, 1, 100)
firstOrderTerm = productMani.inner(x, u, problemMani.grad(x))
secondOrderTerm = 0.5 * productMani.inner(x, u, hess(x, u))
err = [np.abs(cost(productMani.retr(x, ti * u)) - cost(x) - ti * firstOrderTerm - ti ** 2 * secondOrderTerm) for ti in t]
plt.loglog(t, err, 'b-', t, t ** 3, 'r-', t, t ** 2, 'g-')
plt.show()
