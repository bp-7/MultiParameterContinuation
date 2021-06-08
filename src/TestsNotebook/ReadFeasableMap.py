cd H:\MasterThesis\Code\src\TestsNotebook

data = pd.read_csv("dataPos.txt", sep=" ")
x = np.array([float(data.values[i][0]) for i in range(ncols)])
y = np.array(list(set([float(data.values[i][1]) for i in range(data.shape[0])])))
Iterations = np.array([int(data.values[i][2]) for i in range(data.shape[0])])
Iterations = Iterations.reshape((nrows, ncols))
iterMax = np.max(Iterations)
test = np.where(Iterations < 0, iterMax * 2, Iterations)

plt.imshow(test[::-1], extent=(x.min(), x.max(), y.min(), y.max()), interpolation='nearest')
plt.colorbar()
plt.show()

cd H:\MasterThesis\Code\src\TestsNotebook\FeasableMap\Ellipsoid
dataPos = pd.read_csv("dataPos.txt", sep=" ")
dataNeg = pd.read_csv("dataNeg.txt", sep=" ")

nrows, ncols = 40, 30

nonSolvedCoeff = 250

xPos = np.array([float(dataPos.values[i][0]) for i in range(ncols)])
yPos = np.sort(np.array(list(set([float(dataPos.values[i][1]) for i in range(dataPos.shape[0])]))))
IterationsPos = np.array([int(dataPos.values[i][2]) for i in range(dataPos.shape[0])])
IterationsPos = IterationsPos.reshape((nrows, ncols))
iterMaxPos = np.max(IterationsPos)
testPos = np.where(IterationsPos < 0, IterationsPos * nonSolvedCoeff, IterationsPos)
plt.imshow(testPos[::-1], extent=(xPos.min(), xPos.max(), yPos.min(), yPos.max()), interpolation='nearest', cmap=cm.gist_rainbow)
plt.colorbar()
plt.show()

xNeg = np.array([float(dataNeg.values[i][0]) for i in range(ncols)])
yNeg = np.array(list(set([float(dataNeg.values[i][1]) for i in range(dataNeg.shape[0])])))
IterationsNeg = np.array([int(dataNeg.values[i][2]) for i in range(dataNeg.shape[0])])
IterationsNeg = IterationsNeg.reshape((nrows, ncols))
iterMaxNeg = np.max(IterationsNeg)
testNeg = np.where(IterationsNeg < 0, IterationsNeg * nonSolvedCoeff, IterationsNeg)
plt.imshow(testNeg, extent=(xNeg.min(), xNeg.max(), yNeg.min(), yNeg.max()), interpolation='nearest', cmap=cm.gist_rainbow)
plt.colorbar()
plt.show()


IterationsPosExtent = [-1 * np.ones(shape=(ncols)) for i in range(nrows)]
IterationsPos = list(IterationsPos)
testIters = np.concatenate((IterationsPos, IterationsPosExtent), axis=1)

testPos = np.where(testIters < 0,testIters * nonSolvedCoeff, testIters)

xPos = [float(dataPos.values[i][0]) for i in range(ncols)]
xPosExtention = np.linspace(90. * np.pi / 180., np.pi, 30)

for i in range(30):
    xPos.append(xPosExtention[i])

xPos = np.array(xPos)

plt.imshow(testPos[::-1], extent=(xPos.min(), xPos.max(), yPos.min(), yPos.max()), interpolation='nearest', cmap=cm.gist_rainbow)
plt.colorbar()
plt.show()


testItersNeg = np.concatenate((IterationsPosExtent, IterationsNeg), axis=1)
testNeg = np.where(testItersNeg < 0,testItersNeg * nonSolvedCoeff, testItersNeg)

xNeg = [float(dataNeg.values[i][0]) for i in range(ncols)]
xNegExtention = np.linspace(0.0, 90. * np.pi / 180., 30)
for i in range(30):
    xNeg.append(xNegExtention[i])

xNeg = np.array(xNeg)

plt.imshow(testNeg, extent=(xNeg.min(), xNeg.max(), yNeg.min(), yNeg.max()), interpolation='nearest', cmap=cm.gist_rainbow)
plt.colorbar()
plt.show()

test = np.concatenate((testPos[::-1],testNeg), axis=0)

plt.imshow(test, extent=(0, np.pi, yNeg.min(), yPos.max()), interpolation='nearest', cmap=cm.gist_rainbow)
plt.colorbar()
plt.show()

//////////////////////////////
///////////////////////////
Edge = np.zeros(testPos.shape)
edgePositions = []
testPosBis = testPos
for i in range(1, testPosBis.shape[0]-1):
    for j in range(1, testPosBis.shape[1]-1):
        if testPos[i, j] > 0 and ((testPosBis[i, j] - testPosBis[i, j-1] > 500 and testPosBis[i, j] - testPosBis[i, j+1] < 200) or (testPosBis[i, j] - testPosBis[i, j-1] < 200 and testPosBis[i, j] - testPosBis[i, j+1] > 500)):
            Edge[i, j] = 1
            edgePositions.append((xPos[j], yPos[i]))



self._currentPoint = list(self._currentSolution) + [self._nextParameter]

self.hessianMatrix = RepresentSquareOperatorInTotalBergerBasis(self._hessian,
                                                          self.ProductManifold,
                                                          self._currentPoint)
solutionSpaceDimension = int(self.SolutionSpace.dim)

self.hessianSolutionMatrix = self.hessianMatrix[:solutionSpaceDimension, :solutionSpaceDimension]

self.hessianMixteMatrix = self.hessianMatrix[:solutionSpaceDimension, solutionSpaceDimension:]
np.sort(np.linalg.eigvals(self.hessianSolutionMatrix))