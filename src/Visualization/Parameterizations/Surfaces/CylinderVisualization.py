import numpy as np
from Visualization.Parameterizations.Surfaces.SurfaceVisualization import SurfaceVisualization

class CylinderVisualization(SurfaceVisualization):
    def __init__(self, ax, height, radius, center, color, SE3Transformation = np.eye(4), opacity=0.5):
        super().__init__(ax, color, SE3Transformation, opacity)

        self.height = height
        self.radius = radius
        self.centerX, self.centerY = center[0], center[1]

        self.generateMesh()

    def generateMesh(self):
        angle = np.linspace(0, 2 * np.pi, self.DiscretizationParameter)
        z = np.linspace(0, self.height, self.DiscretizationParameter)

        phi, Zt = np.meshgrid(angle, z)

        Xt = self.radius * np.cos(phi) + self.centerX
        Yt = self.radius * np.sin(phi) + self.centerY

        self.X = self.translationTransformation[0] \
                 + self.rotationTransformation[0, 0] * Xt \
                 + self.rotationTransformation[0, 1] * Yt \
                 + self.rotationTransformation[0, 2] * Zt

        self.Y = self.translationTransformation[1] \
                 + self.rotationTransformation[1, 0] * Xt \
                 + self.rotationTransformation[1, 1] * Yt \
                 + self.rotationTransformation[1, 2] * Zt

        self.Z = self.translationTransformation[2] \
                 + self.rotationTransformation[2, 0] * Xt \
                 + self.rotationTransformation[2, 1] * Yt \
                 + self.rotationTransformation[2, 2] * Zt