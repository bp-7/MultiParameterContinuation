import autograd.numpy as np
from Helpers.Parameterizations.BasicSE3Transformations import tau_x, tau_y, tau_z, rho_z, rho_y
from Helpers.Parameterizations.BasicSe3Transformation import se3Basis

def sec(t):
    return 1. / np.cos(t)

def CuttingAngleDerivativeHelixAngle(t, helixAngle):
    return np.tan(t) * np.tan(helixAngle) * sec(helixAngle) / ((np.tan(t) ** 2 * sec(helixAngle)) ** 2 + 1)

def CuttingAngleDerivativeT(t, helixAngle):
    return sec(t) ** 2 * sec(helixAngle) / ((np.tan(t) * sec(helixAngle)) ** 2 + 1)

def TotalAngleDerivativeTrajectoryParameter(helixRadius, helixAngle, helixLength):
    return helixLength * np.tan(helixAngle) / helixRadius

def TotalAngleDerivativeHelixAngle(t, helixRadius, helixAngle, helixLength):
    return helixLength * t / (np.cos(helixAngle) ** 2 * helixRadius)

def TotalAngleDerivativeHelixRadius(t, helixRadius, helixAngle, helixLength):
    return - helixLength * t * np.tan(helixAngle) / (helixRadius ** 2)

def InverseHelixDerivativeTrajectoryParameter(t, helixRadius, helixAngle, helixLength, offsetAngle=0.):
    revolutionAngle = helixLength * t * np.tan(helixAngle) / helixRadius
    totalAngle = revolutionAngle + offsetAngle

    return - helixLength \
           * rho_z(helixAngle - np.pi) \
           @ rho_y(- 0.5 * np.pi) \
           @ tau_z(- helixLength * t) \
           @ se3Basis.G6 \
           @ tau_x(- helixRadius) \
           @ rho_z(- totalAngle) \
           - TotalAngleDerivativeTrajectoryParameter(helixRadius, helixAngle, helixLength) \
           * rho_z(helixAngle - np.pi) \
           @ rho_y(- 0.5 * np.pi) \
           @ tau_z(- helixLength) \
           @ tau_x(- helixRadius) \
           @ rho_z(-totalAngle) \
           @ se3Basis.G3

def InverseHelixDerivativeHelixAngle(t, helixRadius, helixAngle, helixLength, offsetAngle=0.):
    revolutionAngle = helixLength * t * np.tan(helixAngle) / helixRadius
    totalAngle = revolutionAngle + offsetAngle

    return rho_z(helixAngle - np.pi) \
           @ se3Basis.G3 \
           @ rho_y(- 0.5 * np.pi) \
           @ tau_z(- helixLength * t) \
           @ tau_x(- helixRadius) \
           @ rho_z(-totalAngle) \
           - TotalAngleDerivativeHelixAngle(t, helixRadius, helixAngle, helixLength) \
           * rho_z(helixAngle - np.pi) \
           @ rho_y(- 0.5 * np.pi) \
           @ tau_z(- helixLength * t) \
           @ tau_x(- helixRadius) \
           @ rho_z(-totalAngle) \
           @ se3Basis.G3

def InverseHelixDerivativeHelixRadius(t, helixRadius, helixAngle, helixLength, offsetAngle=0.):
    revolutionAngle = helixLength * t * np.tan(helixAngle) / helixRadius
    totalAngle = revolutionAngle + offsetAngle

    return - rho_z(helixAngle - np.pi) \
           @ rho_y(- 0.5 * np.pi) \
           @ tau_z(- helixLength * t) \
           @ tau_x(- helixRadius) \
           @ se3Basis.G4 \
           @ rho_z(-totalAngle) \
           - TotalAngleDerivativeHelixRadius(t, helixRadius, helixAngle, helixLength) \
           * rho_z(helixAngle - np.pi) \
           @ rho_y(- 0.5 * np.pi) \
           @ tau_z(- helixLength * t) \
           @ tau_x(- helixRadius) \
           @ rho_z(-totalAngle) \
           @ se3Basis.G3

def InverseHelixDerivativeOffsetAngle(t, helixRadius, helixAngle, helixLength, offsetAngle):
    revolutionAngle = helixLength * t * np.tan(helixAngle) / helixRadius
    totalAngle = revolutionAngle + offsetAngle

    return - rho_z(helixAngle - np.pi) \
           @ rho_y(- 0.5 * np.pi) \
           @ tau_z(- helixLength) \
           @ tau_x(- helixRadius) \
           @ rho_z(-totalAngle) \
           @ se3Basis.G3