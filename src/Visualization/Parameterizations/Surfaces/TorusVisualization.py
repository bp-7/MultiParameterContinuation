import numpy as np
from Visualization.Parameterizations.Surfaces.SurfaceVisualization import SurfaceVisualization

class TorusVisualization(SurfaceVisualization):
    def __init__(self, ax, torus, color, SE3Transformation = np.eye(4), opacity=0.5):
        super().__init__(ax, color, SE3Transformation, opacity)

        self.largeRadius = torus.largeRadius
        self.smallRadius = torus.smallRadius
        self.offset = torus.offset

        self.generateMesh()

    def generateMesh(self):
        angle = np.linspace(0, 2 * np.pi, self.DiscretizationParameter)
        theta, phi = np.meshgrid(angle, angle)

        Xt = (self.largeRadius + self.smallRadius * np.cos(phi)) * np.cos(theta) + self.offset[0]
        Yt = (self.largeRadius + self.smallRadius * np.cos(phi)) * np.sin(theta) + self.offset[1]
        Zt = self.smallRadius * np.sin(phi) + self.offset[2]

        self.X = self.translationTransformation[0] \
                 + self.rotationTransformation[0, 0] * Xt \
                 + self.rotationTransformation[0, 1] * Yt \
                 + self.rotationTransformation[0, 2] * Zt\

        self.Y = self.translationTransformation[1] \
                 + self.rotationTransformation[1, 0] * Xt \
                 + self.rotationTransformation[1, 1] * Yt \
                 + self.rotationTransformation[1, 2] * Zt\

        self.Z = self.translationTransformation[2] \
                 + self.rotationTransformation[2, 0] * Xt \
                 + self.rotationTransformation[2, 1] * Yt \
                 + self.rotationTransformation[2, 2] * Zt\
