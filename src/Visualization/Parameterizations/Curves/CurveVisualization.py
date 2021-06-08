class CurveVisualization:
    def __init__(self, ax, SE3Transformation, color):
        self.ax = ax
        self.SE3Transformation = SE3Transformation
        self.color = color

        self.rotationTransformation = SE3Transformation[:3, :3]
        self.translationTransformation = SE3Transformation[:3, 3]

        self.X, self.Y, self.Z = None, None, None

    def visualize(self):
        self.ax.plot3D(self.X, self.Y, self.Z, color=self.color)

    def generateData(self):
        raise NotImplementedError





