import numpy as np
from Visualization.Parameterizations.Curves.CurveVisualization import CurveVisualization
from SE3Parameterizations.Parameterizations.Helix import Helix

class HelixVisualization(CurveVisualization):
    def __init__(self, ax, helix, color, SE3Transformation = np.eye(4)):
        super().__init__(ax, SE3Transformation, color)

        if not isinstance(helix, Helix):
            raise ValueError('The helix must be an instance of SE3Parameterizations.Helix')

        self.helix = helix

        self.numberOfPoints = int(5 * self.helix.length + 7 * self.helix.helixStep)

        self.generateData()

    def generateData(self):
        t = np.linspace(0., 1., self.numberOfPoints)

        posesHelixTranslation = [list((self.SE3Transformation @ self.helix.Evaluate(ti))[:3, 3]) for ti in t]

        unzipList = list(zip(*posesHelixTranslation))

        self.X, self.Y, self.Z = unzipList[0], unzipList[1], unzipList[2]