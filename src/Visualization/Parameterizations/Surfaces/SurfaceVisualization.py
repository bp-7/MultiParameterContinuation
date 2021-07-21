class SurfaceVisualization:
    DiscretizationParameter = 100
    RStride = 5
    CStride = 5

    def __init__(self, ax, color, SE3Transformation, opacity):
        self.ax = ax
        self.color = color
        self.opacity = opacity

        self.rotationTransformation = SE3Transformation[:3, :3]
        self.translationTransformation = SE3Transformation[:3, 3]

        self.X, self.Y, self.Z = None, None, None

    def visualize(self):
        self.ax.plot_surface(self.X, self.Y, self.Z, antialiased=True, color=self.color, rstride=self.RStride, cstride=self.CStride,
                             alpha=self.opacity)

    def generateMesh(self):
        raise NotImplementedError
