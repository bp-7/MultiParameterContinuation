import numpy as np
from scipy.linalg import hilbert

for j in range(3, 12):
    for k in range(2, j + 3):
        euclideanSolutionSpaceDimension = j
        parameterSpaceDimension = k

        G_x = np.diag(np.concatenate(
            (np.array([2., 2., 2.]), np.ones(3 + euclideanSolutionSpaceDimension))))

        n = G_x.shape[0]
        A = hilbert(n)
        B = 0.5 * (A + A.transpose())
        solutionHessianMatrix = B #+ n * np.eye(n)
        condHess = np.linalg.cond(solutionHessianMatrix)

        Dmu = np.random.rand(n, parameterSpaceDimension)

        M = Dmu.T @ np.linalg.inv(solutionHessianMatrix) @ G_x @ np.linalg.inv(solutionHessianMatrix) @ Dmu
        condM = np.linalg.cond(M)

        Mat = solutionHessianMatrix.T @ np.linalg.inv(G_x) @ solutionHessianMatrix
        BigMat = np.block([[Dmu, Mat],
                      [np.zeros((parameterSpaceDimension, parameterSpaceDimension), dtype='float64'), Dmu.T]])

        BigMatCond = np.linalg.cond(BigMat)

        print('=====================\n=====================\n=====================')
        print('j = ' + str(j) + ' , k = ' + str(k) +'\n')
        print('Cond. Hess = ' + str(condHess))
        print('BigMatCond = ' + str(BigMatCond))
        print('M cond = ' + str(condM))