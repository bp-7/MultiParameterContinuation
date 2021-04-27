import autograd.numpy as np

from pymanopt.manifolds import Rotations
from pymanopt.manifolds import Euclidean
from pymanopt.core.problem import Problem

import matplotlib.pyplot as plt


# Instantiate the SO(3) manifold
SO3 = Rotations(3)
R33 = Euclidean(3, 3)

# Define cost function
R1 = SO3.rand()

def costSO3(R):
    u = R - R1

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
