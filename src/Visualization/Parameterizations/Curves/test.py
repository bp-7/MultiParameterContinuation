from Visualization.Parameterizations.Surfaces.TorusVisualization import TorusVisualization
from Visualization.Parameterizations.Surfaces.CylinderVisualization import CylinderVisualization
from Visualization.Parameterizations.Curves.HelixVisualization import HelixVisualization
from SE3Parameterizations.Parameterizations.Helix import Helix
from SE3Parameterizations.Parameterizations.Torus import Torus

import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$z$')
helixRadius, helixLength, helixAngle = 2., 10., 30. * np.pi / 180.
rt, Rt = 5., 30.
offsetWheel = np.array([0., 40., 0.])

X0 = [np.array([[ 0.14700258,  0.9854882 ,  0.08487199],
        [-0.98911615,  0.14700258,  0.0062838 ],
        [-0.00628379, -0.08487199,  0.99637205]]),
      np.array([-2.92744102, -0.73501289,  0.42435993]),
      np.array([ 0.14807544, -0.00251379,  0.02705792, -2.09305339, -0.08459128,
        -0.14802329])]

phi = np.block([[X0[0], np.reshape(X0[1], (3, 1))], [np.zeros((1, 3)), 1.]])

#h1 = HelixVisualization(ax, helixRadius, helixAngle, helixLength, 'red')
#h2 = HelixVisualization(ax, helixRadius, helixAngle + 20 * np.pi / 180., helixLength, 'blue', 30 * np.pi / 180.)
c = CylinderVisualization(ax, helixLength, helixRadius, np.array([0., 0.]), 'grey', opacity=0.5)

helix = Helix(helixRadius, helixAngle, helixLength, offsetAngle=0.0)
secondHelix = Helix(helixRadius, helixAngle, helixLength, offsetAngle=30. * np.pi / 180.)
torus = Torus(rt, Rt, offsetWheel)

helixVisualization = HelixVisualization(ax, helix, 'red')
secondHelixVisualization = HelixVisualization(ax, secondHelix, 'blue')
torusVisualization = TorusVisualization(ax, torus, SE3Transformation=phi, color='orange')



helixVisualization.visualize()
secondHelixVisualization.visualize()
# h2.visualize()
c.visualize()
torusVisualization.visualize()