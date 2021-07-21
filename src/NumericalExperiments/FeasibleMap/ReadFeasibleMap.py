import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd

fileNameForPositiveCoefficients = 'bla'
fileNameForNegativeCoefficients = 'bla'

dataPos = pd.read_csv(fileNameForPositiveCoefficients + ".txt", sep=" ")
dataNeg = pd.read_csv(fileNameForNegativeCoefficients + ".txt", sep=" ")
nrows, ncols = 50, 40

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})

IterationsPos = np.array([int(dataPos.values[i][2]) for i in range(dataPos.shape[0])])
IterationsPos = IterationsPos.reshape((nrows, ncols))
yPos = np.sort(np.array(list(set([float(dataPos.values[i][1]) for i in range(dataPos.shape[0])]))))

IterationsNeg = np.array([int(dataNeg.values[i][2]) for i in range(dataNeg.shape[0])])
IterationsNeg = IterationsNeg.reshape((nrows, ncols))
yNeg = np.array(list(set([float(dataNeg.values[i][1]) for i in range(dataNeg.shape[0])])))

IterationsPosExtent = [-1 * np.ones(shape=(ncols)) for i in range(nrows)]
IterationsPos = list(IterationsPos)

testIters = np.concatenate((IterationsPos, IterationsPosExtent), axis=1)
testPos = np.where(testIters < 0, np.nan, testIters)

xPos = [float(dataPos.values[i][0]) for i in range(ncols)]
xPosExtention = np.linspace(90. * np.pi / 180., np.pi, ncols)
for i in range(ncols):
    xPos.append(xPosExtention[i])
xPos = np.array(xPos)

testItersNeg = np.concatenate((IterationsPosExtent, IterationsNeg), axis=1)
testNeg = np.where(testItersNeg < 0, np.nan, testItersNeg)

xNeg = [float(dataNeg.values[i][0]) for i in range(ncols)]
xNegExtention = np.linspace(0.0, 90. * np.pi / 180., ncols)
for i in range(ncols):
    xNeg.append(xNegExtention[i])
xNeg = np.array(xNeg)

test = np.concatenate((testPos[::-1],testNeg), axis=0)

plt.figure(figsize=(5, 8), dpi=100)
plt.rcParams['font.size'] = '18'
plt.xlabel(r"$\theta$")
plt.ylabel(r"$a$")

plt.imshow(test, extent=(0, np.pi, yNeg.min(), yPos.max()), interpolation='nearest', cmap=cm.turbo)
plt.colorbar()
plt.show()
