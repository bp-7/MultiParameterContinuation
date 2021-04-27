import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse
from matplotlib.patches import Circle
from matplotlib.patches import Rectangle

class SubplotAnimation(animation.TimedAnimation):
    def __init__(self, results, parameterMagnitude, targetParameter, finalSolution, differentialParameterAlongV, differentialSolutionAlongV):
        self.parameters = [p for (a, (x, p)) in results]
        self.solutions = [x for (a, (x, p)) in results]

        self.parameterMagnitude = parameterMagnitude
        self.data = [np.reshape(solution, (1, 3)) for solution in self.solutions]

        self.xParameters = [x for (x, y) in self.parameters]
        self.yParameters = [y for (x, y) in self.parameters]

        fig = plt.figure(figsize=(10,8), dpi= 100)
        ax1 = fig.add_subplot(1, 2, 2, projection="3d")
        self.ax2 = fig.add_subplot(1, 2, 1)

        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('z')

        ax1.set_xlim(-1.2, 1.2)
        ax1.set_ylim(-1.2, 1.2)
        ax1.set_zlim(-1.2, 1.2)

        self.ax2.set_xlabel('x')
        self.ax2.set_ylabel('y')
        self.ax2.set_xlim(-40, 50)
        self.ax2.set_ylim(-40, 50)

        self.ax2.grid()

        # draw sphere
        u, v = np.mgrid[0: 2 * np.pi: 15j, 0: np.pi: 15j]
        x = np.cos(u) * np.sin(v)
        y = np.sin(u) * np.sin(v)
        z = np.cos(v)

        ax1.plot_wireframe(x, y, z, color="y")

        ax1.scatter(finalSolution[0], finalSolution[1], finalSolution[2], c='r')

        ax1.view_init(5, 225)
        
        self.ax2.plot(targetParameter[0], targetParameter[1], 'ro')

        self.scatters = [ax1.scatter(self.data[0][i, 0:1], self.data[0][i, 1:2], self.data[0][i, 2:], 'o-') for i in
                         range(self.data[0].shape[0])]

        self.line, = self.ax2.plot([], [], 'o', lw=2)
        self.time_template = 'iteration = %.1i'
        self.time_text = self.ax2.text(0.05, 0.9, '', transform=self.ax2.transAxes)
        
        self.targetParameter = targetParameter
        self.differentialParameterAlongV = differentialParameterAlongV
        self.differentialSolutionAlongV = differentialSolutionAlongV
        
        self.ellipse = self.determineEllipse(0)
        self.ax2.add_patch(self.ellipse)
        
        animation.TimedAnimation.__init__(self, fig, interval=1800, blit=False, repeat=True)

    def _draw_frame(self, framedata):

        #draw ellipse
        self.ellipse = self.determineEllipse(framedata)

        self.ax2.patches = []
        self.ax2.add_patch(self.ellipse)

        #draw perturbation vector
        if framedata + 1 < len(self.xParameters):
            dx = self.xParameters[framedata + 1] - self.xParameters[framedata]
            dy = self.yParameters[framedata + 1] - self.yParameters[framedata]
            self.ax2.arrow(self.xParameters[framedata], self.yParameters[framedata], dx, dy)

        # draw solution point on sphere
        for i in range(self.data[0].shape[0]):
            self.scatters[i]._offsets3d = (
            self.data[framedata][i, 0:1], self.data[framedata][i, 1:2], self.data[framedata][i, 2:])

        # draw new parameter value in parameter space
        thisx = self.xParameters[framedata]
        thisy = self.yParameters[framedata]

        self.line.set_data(thisx, thisy)

        # draw number of iteration
        self.time_text.set_text(self.time_template % framedata)

        print("Delta parameter = " + str(self.targetParameter - self.parameters[framedata]))

    def new_frame_seq(self):
        return iter(range(len(self.xParameters)))

    def _init_draw(self):
        self.line.set_data([], [])
        self.time_text.set_text('')

    def reconstructMetricMatrix(self, framedata):
        currentSolution = self.solutions[framedata]
        currentParameter = self.parameters[framedata]

        differentialParameterMatrix = self.WriteMatrixInEuclideanBasisAtGivenPoint(self.differentialParameterAlongV,
                                                                                   currentSolution,
                                                                                   currentParameter,
                                                                                   len(currentParameter))

        differentialSolutionMatrix = self.WriteMatrixInEuclideanBasisAtGivenPoint(self.differentialSolutionAlongV,
                                                                                  currentSolution,
                                                                                  currentParameter,
                                                                                  len(currentSolution))

        inverseDifferentialSolutionMatrix = np.linalg.inv(differentialSolutionMatrix)

        parameterSpaceMetricMatrix = differentialParameterMatrix.T \
                                     @ inverseDifferentialSolutionMatrix.T \
                                     @ np.eye(len(currentSolution)) \
                                     @ inverseDifferentialSolutionMatrix \
                                     @ differentialParameterMatrix

        return parameterSpaceMetricMatrix

    def WriteMatrixInEuclideanBasisAtGivenPoint(self, matrixVectorFunction, x, mu, dimension):
        spaceDimension = len(x)
        indices = np.arange(dimension)
        A = np.zeros((spaceDimension, dimension))

        for index in indices:
            v = np.zeros(dimension)
            v[index] = 1

            A[:, index] = matrixVectorFunction(x, mu, v)

        return A

    def determineEllipse(self, framedata):

        metricMatrix = self.reconstructMetricMatrix(framedata)

        deltaParameter = self.targetParameter - self.parameters[framedata]

        eigvals, eigvecs = np.linalg.eig(metricMatrix)

        smallestIndex = np.argmin(eigvals)
        largestIndex = np.argmax(eigvals)

        print(eigvals)

        slope = eigvecs[1, smallestIndex] / eigvecs[0, smallestIndex]
        angle = 180.0 * np.arctan(slope) / np.pi

        if np.inner(deltaParameter, metricMatrix @ deltaParameter) < self.parameterMagnitude ** 2:
            _perturbationMagnitude = np.sqrt(np.inner(deltaParameter, metricMatrix @ deltaParameter))
        else:
            _perturbationMagnitude = self.parameterMagnitude

        return Ellipse(self.parameters[framedata], width= 2 * _perturbationMagnitude / np.sqrt(eigvals[smallestIndex]),
                               height=2 * _perturbationMagnitude / np.sqrt(eigvals[largestIndex]), angle=angle, fill=False)
    
    def show(self):
        plt.show()
 

class SubplotAnimationWithDiscontinuities(animation.TimedAnimation):
    def __init__(self, results, parameterMagnitude, targetParameter, finalSolution, differentialParameterAlongV, differentialSolutionAlongV):
        self.parameters = [p for (a, (x, p)) in results]
        self.solutions = [x for (a, (x, p)) in results]

        self.parameterMagnitude = parameterMagnitude
        self.data = [np.reshape(solution, (1, 3)) for solution in self.solutions]

        self.xParameters = [x for (x, y) in self.parameters]
        self.yParameters = [y for (x, y) in self.parameters]

        fig = plt.figure(figsize=(10,8), dpi= 100)
        ax1 = fig.add_subplot(1, 2, 2, projection="3d")
        self.ax2 = fig.add_subplot(1, 2, 1)

        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('z')

        ax1.set_xlim(-1.2, 1.2)
        ax1.set_ylim(-1.2, 1.2)
        ax1.set_zlim(-1.2, 1.2)

        self.ax2.set_xlabel('x')
        self.ax2.set_ylabel('y')
        self.ax2.set_xlim(-40, 50)
        self.ax2.set_ylim(-40, 50)

        self.ax2.grid()

        # draw sphere
        u, v = np.mgrid[0: 2 * np.pi: 15j, 0: np.pi: 15j]
        x = np.cos(u) * np.sin(v)
        y = np.sin(u) * np.sin(v)
        z = np.cos(v)

        ax1.plot_wireframe(x, y, z, color="y")

        ax1.scatter(finalSolution[0], finalSolution[1], finalSolution[2], c='r')

        ax1.view_init(5, 225)
        
        self.ax2.plot(targetParameter[0], targetParameter[1], 'ro')
        
        self.scatters = [ax1.scatter(self.data[0][i, 0:1], self.data[0][i, 1:2], self.data[0][i, 2:], 'o-') for i in
                         range(self.data[0].shape[0])]

        self.line, = self.ax2.plot([], [], 'o', lw=2)
        self.time_template = 'iteration = %.1i'
        self.time_text = self.ax2.text(0.05, 0.9, '', transform=self.ax2.transAxes)
        
        self.targetParameter = targetParameter
        self.differentialParameterAlongV = differentialParameterAlongV
        self.differentialSolutionAlongV = differentialSolutionAlongV
        
        self.ellipse = self.determineEllipse(0)
        self.ax2.add_patch(self.ellipse)
        
        self.ax2.plot(np.linspace(-100, 100, 100), np.linspace(-100, 100, 100), '-')
        self.ax2.plot(-np.linspace(-100, 100, 100), np.linspace(-100, 100, 100), '-')
        self.ax2.plot(- 3 * np.linspace(-100, 100, 100), np.linspace(-100, 100, 100), '-')
        
        animation.TimedAnimation.__init__(self, fig, interval=1800, blit=False, repeat=True)

    def _draw_frame(self, framedata):

        #draw ellipse
        self.ellipse = self.determineEllipse(framedata)

        self.ax2.patches = []
        self.ax2.add_patch(self.ellipse)

        #draw perturbation vector
        if framedata + 1 < len(self.xParameters):
            dx = self.xParameters[framedata + 1] - self.xParameters[framedata]
            dy = self.yParameters[framedata + 1] - self.yParameters[framedata]
            self.ax2.arrow(self.xParameters[framedata], self.yParameters[framedata], dx, dy)

        # draw solution point on sphere
        for i in range(self.data[0].shape[0]):
            self.scatters[i]._offsets3d = (
            self.data[framedata][i, 0:1], self.data[framedata][i, 1:2], self.data[framedata][i, 2:])

        # draw new parameter value in parameter space
        thisx = self.xParameters[framedata]
        thisy = self.yParameters[framedata]

        self.line.set_data(thisx, thisy)

        # draw number of iteration
        self.time_text.set_text(self.time_template % framedata)

        print("Delta parameter = " + str(self.targetParameter - self.parameters[framedata]))

    def new_frame_seq(self):
        return iter(range(len(self.xParameters)))

    def _init_draw(self):
        self.line.set_data([], [])
        self.time_text.set_text('')

    def reconstructMetricMatrix(self, framedata):
        currentSolution = self.solutions[framedata]
        currentParameter = self.parameters[framedata]

        differentialParameterMatrix = self.WriteMatrixInEuclideanBasisAtGivenPoint(self.differentialParameterAlongV,
                                                                                   currentSolution,
                                                                                   currentParameter,
                                                                                   len(currentParameter))

        differentialSolutionMatrix = self.WriteMatrixInEuclideanBasisAtGivenPoint(self.differentialSolutionAlongV,
                                                                                  currentSolution,
                                                                                  currentParameter,
                                                                                  len(currentSolution))

        inverseDifferentialSolutionMatrix = np.linalg.inv(differentialSolutionMatrix)

        parameterSpaceMetricMatrix = differentialParameterMatrix.T \
                                     @ inverseDifferentialSolutionMatrix.T \
                                     @ np.eye(len(currentSolution)) \
                                     @ inverseDifferentialSolutionMatrix \
                                     @ differentialParameterMatrix

        return parameterSpaceMetricMatrix

    def WriteMatrixInEuclideanBasisAtGivenPoint(self, matrixVectorFunction, x, mu, dimension):
        spaceDimension = len(x)
        indices = np.arange(dimension)
        A = np.zeros((spaceDimension, dimension))

        for index in indices:
            v = np.zeros(dimension)
            v[index] = 1

            A[:, index] = matrixVectorFunction(x, mu, v)

        return A

    def determineEllipse(self, framedata):

        metricMatrix = self.reconstructMetricMatrix(framedata)

        deltaParameter = self.targetParameter - self.parameters[framedata]

        eigvals, eigvecs = np.linalg.eig(metricMatrix)

        smallestIndex = np.argmin(eigvals)
        largestIndex = np.argmax(eigvals)

        print(eigvals)

        slope = eigvecs[1, smallestIndex] / eigvecs[0, smallestIndex]
        angle = 180.0 * np.arctan(slope) / np.pi

        if np.inner(deltaParameter, metricMatrix @ deltaParameter) < self.parameterMagnitude ** 2:
            _perturbationMagnitude = np.sqrt(np.inner(deltaParameter, metricMatrix @ deltaParameter))
        else:
            _perturbationMagnitude = self.parameterMagnitude

        return Ellipse(self.parameters[framedata], width= 2 * _perturbationMagnitude / np.sqrt(eigvals[smallestIndex]),
                               height=2 * _perturbationMagnitude / np.sqrt(eigvals[largestIndex]), angle=angle, fill=False)
    
    def show(self):
        plt.show()
        
class SubplotAnimationWithDiscontinuityRegions(animation.TimedAnimation):
    def __init__(self, results, parameterMagnitude, targetParameter, finalSolution, differentialParameterAlongV, differentialSolutionAlongV, epsilon):
        self.epsilon = epsilon
        
        self.parameters = [p for (a, (x, p)) in results]
        self.solutions = [x for (a, (x, p)) in results]

        self.parameterMagnitude = parameterMagnitude
        self.data = [np.reshape(solution, (1, 3)) for solution in self.solutions]

        self.xParameters = [x for (x, y) in self.parameters]
        self.yParameters = [y for (x, y) in self.parameters]

        fig = plt.figure(figsize=(10,8), dpi= 100)
        ax1 = fig.add_subplot(1, 2, 2, projection="3d")
        self.ax2 = fig.add_subplot(1, 2, 1)

        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('z')

        ax1.set_xlim(-1.2, 1.2)
        ax1.set_ylim(-1.2, 1.2)
        ax1.set_zlim(-1.2, 1.2)

        self.ax2.set_xlabel('x')
        self.ax2.set_ylabel('y')
        self.ax2.set_xlim(-40, 50)
        self.ax2.set_ylim(-40, 50)

        self.ax2.grid()

        # draw sphere
        u, v = np.mgrid[0: 2 * np.pi: 15j, 0: np.pi: 15j]
        x = np.cos(u) * np.sin(v)
        y = np.sin(u) * np.sin(v)
        z = np.cos(v)

        ax1.plot_wireframe(x, y, z, color="y")

        ax1.scatter(finalSolution[0], finalSolution[1], finalSolution[2], c='r')

        ax1.view_init(5, 225)
        
        self.ax2.plot(targetParameter[0], targetParameter[1], 'ro')

        self.scatters = [ax1.scatter(self.data[0][i, 0:1], self.data[0][i, 1:2], self.data[0][i, 2:], 'o-') for i in
                         range(self.data[0].shape[0])]

        self.line, = self.ax2.plot([], [], 'o', lw=2)
        self.time_template = 'iteration = %.1i'
        self.time_text = self.ax2.text(0.05, 0.9, '', transform=self.ax2.transAxes)
        
        self.targetParameter = targetParameter
        self.differentialParameterAlongV = differentialParameterAlongV
        self.differentialSolutionAlongV = differentialSolutionAlongV

        rectangle1 = Rectangle((70 - epsilon / np.sqrt(2), -70 - epsilon / np.sqrt(2)), 2 * epsilon, 2000, 45, color='g')
        rectangle2 = Rectangle((-70 - epsilon / np.sqrt(2), -70 + epsilon / np.sqrt(2)), 2 * epsilon, 2000, -45, color='g')
        rectangle3 = Rectangle((30 - 3.0 * epsilon / np.sqrt(10.0), -90 - epsilon / np.sqrt(10.0)), 2 * epsilon, 2000,
                            np.rad2deg(np.arctan(1.0 / 3)), color='g')

        self.ax2.add_artist(rectangle1)
        self.ax2.add_artist(rectangle2)
        self.ax2.add_artist(rectangle3)

        self.ellipse = self.determineEllipse(0)
        self.ax2.add_patch(self.ellipse)
        
        animation.TimedAnimation.__init__(self, fig, interval=1800, blit=False, repeat=True)

    def _draw_frame(self, framedata):

        #draw ellipse
        self.ellipse = self.determineEllipse(framedata)

        self.ax2.patches = []
        self.ax2.add_patch(self.ellipse)

        #draw perturbation vector
        if framedata + 1 < len(self.xParameters):
            dx = self.xParameters[framedata + 1] - self.xParameters[framedata]
            dy = self.yParameters[framedata + 1] - self.yParameters[framedata]
            self.ax2.arrow(self.xParameters[framedata], self.yParameters[framedata], dx, dy)

        # draw solution point on sphere
        for i in range(self.data[0].shape[0]):
            self.scatters[i]._offsets3d = (
            self.data[framedata][i, 0:1], self.data[framedata][i, 1:2], self.data[framedata][i, 2:])

        # draw new parameter value in parameter space
        thisx = self.xParameters[framedata]
        thisy = self.yParameters[framedata]

        self.line.set_data(thisx, thisy)

        # draw number of iteration
        self.time_text.set_text(self.time_template % framedata)

        print("Delta parameter = " + str(self.targetParameter - self.parameters[framedata]))

    def new_frame_seq(self):
        return iter(range(len(self.xParameters)))

    def _init_draw(self):
        self.line.set_data([], [])
        self.time_text.set_text('')

    def reconstructMetricMatrix(self, framedata):
        currentSolution = self.solutions[framedata]
        currentParameter = self.parameters[framedata]

        differentialParameterMatrix = self.WriteMatrixInEuclideanBasisAtGivenPoint(self.differentialParameterAlongV,
                                                                                   currentSolution,
                                                                                   currentParameter,
                                                                                   len(currentParameter))

        differentialSolutionMatrix = self.WriteMatrixInEuclideanBasisAtGivenPoint(self.differentialSolutionAlongV,
                                                                                  currentSolution,
                                                                                  currentParameter,
                                                                                  len(currentSolution))

        inverseDifferentialSolutionMatrix = np.linalg.inv(differentialSolutionMatrix)

        parameterSpaceMetricMatrix = differentialParameterMatrix.T \
                                     @ inverseDifferentialSolutionMatrix.T \
                                     @ np.eye(len(currentSolution)) \
                                     @ inverseDifferentialSolutionMatrix \
                                     @ differentialParameterMatrix

        return parameterSpaceMetricMatrix

    def WriteMatrixInEuclideanBasisAtGivenPoint(self, matrixVectorFunction, x, mu, dimension):
        spaceDimension = len(x)
        indices = np.arange(dimension)
        A = np.zeros((spaceDimension, dimension))

        for index in indices:
            v = np.zeros(dimension)
            v[index] = 1

            A[:, index] = matrixVectorFunction(x, mu, v)

        return A

    def determineEllipse(self, framedata):

        metricMatrix = self.reconstructMetricMatrix(framedata)

        deltaParameter = self.targetParameter - self.parameters[framedata]

        eigvals, eigvecs = np.linalg.eig(metricMatrix)

        smallestIndex = np.argmin(eigvals)
        largestIndex = np.argmax(eigvals)

        print(eigvals)

        slope = eigvecs[1, smallestIndex] / eigvecs[0, smallestIndex]
        angle = 180.0 * np.arctan(slope) / np.pi

        if np.inner(deltaParameter, metricMatrix @ deltaParameter) < self.parameterMagnitude ** 2:
            _perturbationMagnitude = np.sqrt(np.inner(deltaParameter, metricMatrix @ deltaParameter))
        else:
            _perturbationMagnitude = self.parameterMagnitude

        return Ellipse(self.parameters[framedata], width= 2 * _perturbationMagnitude / np.sqrt(eigvals[smallestIndex]),
                               height=2 * _perturbationMagnitude / np.sqrt(eigvals[largestIndex]), angle=angle, fill=False)
    
    def show(self):
        plt.show()


class SubplotAnimationCircleWithDiscontinuities(animation.TimedAnimation):
    def __init__(self, results, parameterMagnitude, targetParameter, finalSolution, differentialParameterAlongV, differentialSolutionAlongV):
        self.parameters = [p for (a, (x, p)) in results]
        self.solutions = [x for (a, (x, p)) in results]

        self.parameterMagnitude = parameterMagnitude
        self.data = [np.reshape(solution, (1, 3)) for solution in self.solutions]

        self.xParameters = [x for (x, y) in self.parameters]
        self.yParameters = [y for (x, y) in self.parameters]

        fig = plt.figure(figsize=(10,8), dpi= 100)
        ax1 = fig.add_subplot(1, 2, 2, projection="3d")
        self.ax2 = fig.add_subplot(1, 2, 1)

        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('z')

        ax1.set_xlim(-1.2, 1.2)
        ax1.set_ylim(-1.2, 1.2)
        ax1.set_zlim(-1.2, 1.2)

        self.ax2.set_xlabel('x')
        self.ax2.set_ylabel('y')
        self.ax2.set_xlim(-40, 50)
        self.ax2.set_ylim(-40, 50)

        self.ax2.grid()

        # draw sphere
        u, v = np.mgrid[0: 2 * np.pi: 15j, 0: np.pi: 15j]
        x = np.cos(u) * np.sin(v)
        y = np.sin(u) * np.sin(v)
        z = np.cos(v)

        ax1.plot_wireframe(x, y, z, color="y")

        ax1.scatter(finalSolution[0], finalSolution[1], finalSolution[2], c='r')

        ax1.view_init(5, 225)
        
        self.ax2.plot(targetParameter[0], targetParameter[1], 'ro')

        self.scatters = [ax1.scatter(self.data[0][i, 0:1], self.data[0][i, 1:2], self.data[0][i, 2:], 'o-') for i in
                         range(self.data[0].shape[0])]

        self.line, = self.ax2.plot([], [], 'o', lw=2)
        self.time_template = 'iteration = %.1i'
        self.time_text = self.ax2.text(0.05, 0.9, '', transform=self.ax2.transAxes)
        
        self.targetParameter = targetParameter
        self.differentialParameterAlongV = differentialParameterAlongV
        self.differentialSolutionAlongV = differentialSolutionAlongV
        
        self.ellipse = self.determineCircle(0)
        self.ax2.add_patch(self.ellipse)
        
        self.ax2.plot(np.linspace(-100, 100, 100), np.linspace(-100, 100, 100), '-')
        self.ax2.plot(-np.linspace(-100, 100, 100), np.linspace(-100, 100, 100), '-')
        self.ax2.plot(- 3 * np.linspace(-100, 100, 100), np.linspace(-100, 100, 100), '-')
        
        animation.TimedAnimation.__init__(self, fig, interval=1800, blit=False, repeat=True)

    def _draw_frame(self, framedata):

        #draw ellipse
        self.ellipse = self.determineCircle(framedata)

        self.ax2.patches = []
        self.ax2.add_patch(self.ellipse)

        #draw perturbation vector
        if framedata + 1 < len(self.xParameters):
            dx = self.xParameters[framedata + 1] - self.xParameters[framedata]
            dy = self.yParameters[framedata + 1] - self.yParameters[framedata]
            self.ax2.arrow(self.xParameters[framedata], self.yParameters[framedata], dx, dy)

        # draw solution point on sphere
        for i in range(self.data[0].shape[0]):
            self.scatters[i]._offsets3d = (
            self.data[framedata][i, 0:1], self.data[framedata][i, 1:2], self.data[framedata][i, 2:])

        # draw new parameter value in parameter space
        thisx = self.xParameters[framedata]
        thisy = self.yParameters[framedata]

        self.line.set_data(thisx, thisy)

        # draw number of iteration
        self.time_text.set_text(self.time_template % framedata)

        print("Delta parameter = " + str(self.targetParameter - self.parameters[framedata]))

    def new_frame_seq(self):
        return iter(range(len(self.xParameters)))

    def _init_draw(self):
        self.line.set_data([], [])
        self.time_text.set_text('')

    def reconstructMetricMatrix(self, framedata):
        currentSolution = self.solutions[framedata]
        currentParameter = self.parameters[framedata]

        differentialParameterMatrix = self.WriteMatrixInEuclideanBasisAtGivenPoint(self.differentialParameterAlongV,
                                                                                   currentSolution,
                                                                                   currentParameter,
                                                                                   len(currentParameter))

        differentialSolutionMatrix = self.WriteMatrixInEuclideanBasisAtGivenPoint(self.differentialSolutionAlongV,
                                                                                  currentSolution,
                                                                                  currentParameter,
                                                                                  len(currentSolution))

        inverseDifferentialSolutionMatrix = np.linalg.inv(differentialSolutionMatrix)

        parameterSpaceMetricMatrix = differentialParameterMatrix.T \
                                     @ inverseDifferentialSolutionMatrix.T \
                                     @ np.eye(len(currentSolution)) \
                                     @ inverseDifferentialSolutionMatrix \
                                     @ differentialParameterMatrix

        return parameterSpaceMetricMatrix

    def WriteMatrixInEuclideanBasisAtGivenPoint(self, matrixVectorFunction, x, mu, dimension):
        spaceDimension = len(x)
        indices = np.arange(dimension)
        A = np.zeros((spaceDimension, dimension))

        for index in indices:
            v = np.zeros(dimension)
            v[index] = 1

            A[:, index] = matrixVectorFunction(x, mu, v)

        return A

    def determineCircle(self, framedata):
        deltaParameter = self.targetParameter - self.parameters[framedata]

        if np.inner(deltaParameter, deltaParameter) < self.parameterMagnitude ** 2:
            _perturbationMagnitude = np.sqrt(np.inner(deltaParameter, deltaParameter))
        else:
            _perturbationMagnitude = self.parameterMagnitude

        return Circle(self.parameters[framedata], _perturbationMagnitude, fill=False)
    
    def show(self):
        plt.show()



####################################################################################
####################################################################################

class SubplotAnimationPositioning(animation.TimedAnimation):
    def __init__(self, results, parameterMagnitude, targetParameter):
        self.solutions = [S[:len(S)] for (a, S) in results]
        self.parameters = [S[-1] for (a, S) in results]

        self.parameterMagnitude = parameterMagnitude

        self.xParameters = [x for (x, y) in self.parameters]
        self.yParameters = [y for (x, y) in self.parameters]

        fig = plt.figure(figsize=(10, 8), dpi=100)
        self.ax1 = fig.add_subplot(1, 2, 2)
        self.ax2 = fig.add_subplot(1, 2, 1)

        self.ax1.set_xlabel('x')
        self.ax1.set_ylabel('y')
        self.ax1.set_xlim(-2., 2.)
        self.ax1.set_ylim(-2., 2.)

        self.ax2.set_xlabel('x')
        self.ax2.set_ylabel('y')
        self.ax2.set_xlim(-40, 50)
        self.ax2.set_ylim(-40, 50)

        self.ax1.grid()
        self.ax2.grid()

        self.ax2.plot(targetParameter[0], targetParameter[1], 'ro')

        self.line2, = self.ax1.plot([], [], 'r-', lw=2)
        self.line, = self.ax2.plot([], [], 'o', lw=2)
        self.time_template = 'iteration = %.1i'
        self.time_text = self.ax2.text(0.05, 0.9, '', transform=self.ax2.transAxes)

        self.targetParameter = targetParameter

        self.draw_curve(0)

        # self.ellipse = self.determineEllipse(0)
        # self.ax2.add_patch(self.ellipse)

        animation.TimedAnimation.__init__(self, fig, interval=200, blit=False, repeat=True)

    def _draw_frame(self, framedata):

        # draw ellipse
        # self.ellipse = self.determineEllipse(framedata)

        # self.ax2.patches = []
        # self.ax2.add_patch(self.ellipse)

        # draw perturbation vector
        if framedata + 1 < len(self.xParameters):
            dx = self.xParameters[framedata + 1] - self.xParameters[framedata]
            dy = self.yParameters[framedata + 1] - self.yParameters[framedata]
            self.ax2.arrow(self.xParameters[framedata], self.yParameters[framedata], dx, dy)

        self.draw_curve(self, framedata)

        # draw new parameter value in parameter space
        thisx = self.xParameters[framedata]
        thisy = self.yParameters[framedata]

        self.line.set_data(thisx, thisy)

        # draw number of iteration
        self.time_text.set_text(self.time_template % framedata)

        print("Delta parameter = " + str(self.targetParameter - self.parameters[framedata]))

    def new_frame_seq(self):
        return iter(range(len(self.xParameters)))

    def _init_draw(self):
        self.line.set_data([], [])
        self.line2.set_data([], [])
        self.time_text.set_text('')

    def reconstructMetricMatrix(self, framedata):
        currentSolution = self.solutions[framedata]
        currentParameter = self.parameters[framedata]

        differentialParameterMatrix = self.WriteMatrixInEuclideanBasisAtGivenPoint(self.differentialParameterAlongV,
                                                                                   currentSolution,
                                                                                   currentParameter,
                                                                                   len(currentParameter))

        differentialSolutionMatrix = self.WriteMatrixInEuclideanBasisAtGivenPoint(self.differentialSolutionAlongV,
                                                                                  currentSolution,
                                                                                  currentParameter,
                                                                                  len(currentSolution))

        inverseDifferentialSolutionMatrix = np.linalg.inv(differentialSolutionMatrix)

        parameterSpaceMetricMatrix = differentialParameterMatrix.T \
                                     @ inverseDifferentialSolutionMatrix.T \
                                     @ np.eye(len(currentSolution)) \
                                     @ inverseDifferentialSolutionMatrix \
                                     @ differentialParameterMatrix

        return parameterSpaceMetricMatrix

    def WriteMatrixInEuclideanBasisAtGivenPoint(self, matrixVectorFunction, x, mu, dimension):
        spaceDimension = len(x)
        indices = np.arange(dimension)
        A = np.zeros((spaceDimension, dimension))

        for index in indices:
            v = np.zeros(dimension)
            v[index] = 1

            A[:, index] = matrixVectorFunction(x, mu, v)

        return A

    def draw_curve(self, framedata):
        a = self.yParameters[framedata]

        if framedata == 0:
            solution = self.solutions[0][0]
        else:
            solution = self.solutions[framedata]

        rotationMatrix = solution[0]

        x = np.linspace(-3, 3, 100)
        y = a * x ** 2

        xRot = rotationMatrix[0, 0] * x + rotationMatrix[0, 1] * y
        yRot = rotationMatrix[1, 0] * x + rotationMatrix[1, 1] * y

        self.line2.set_data(xRot, yRot)
        # self.ax1.plot(xRot, yRot, 'r-')

    def determineEllipse(self, framedata):

        metricMatrix = self.reconstructMetricMatrix(framedata)

        deltaParameter = self.targetParameter - self.parameters[framedata]

        eigvals, eigvecs = np.linalg.eig(metricMatrix)

        smallestIndex = np.argmin(eigvals)
        largestIndex = np.argmax(eigvals)

        print(eigvals)

        slope = eigvecs[1, smallestIndex] / eigvecs[0, smallestIndex]
        angle = 180.0 * np.arctan(slope) / np.pi

        if np.inner(deltaParameter, metricMatrix @ deltaParameter) < self.parameterMagnitude ** 2:
            _perturbationMagnitude = np.sqrt(np.inner(deltaParameter, metricMatrix @ deltaParameter))
        else:
            _perturbationMagnitude = self.parameterMagnitude

        return Ellipse(self.parameters[framedata], width=2 * _perturbationMagnitude / np.sqrt(eigvals[smallestIndex]),
                       height=2 * _perturbationMagnitude / np.sqrt(eigvals[largestIndex]), angle=angle, fill=False)

    def show(self):
        plt.show()

class SubplotAnimationPositioning(animation.TimedAnimation):
    def __init__(self, results, parameterSpaceMetrics, parameterMagnitude, targetParameter):
        self.solutions = [S[:len(S)] for (a, S) in results]
        self.parameters = [S[-1] for (a, S) in results]

        self.parameterMagnitude = parameterMagnitude

        self.parameterSpaceMetrics = parameterSpaceMetrics

        self.xParameters = [x for (x, y) in self.parameters]
        self.yParameters = [y for (x, y) in self.parameters]

        fig = plt.figure(figsize=(10, 8), dpi=100)
        self.ax1 = fig.add_subplot(1, 2, 2)
        self.ax2 = fig.add_subplot(1, 2, 1)

        self.ax1.set_xlabel('x')
        self.ax1.set_ylabel('y')
        self.ax1.set_xlim(-2., 2.)
        self.ax1.set_ylim(-2., 2.)

        self.ax2.set_xlabel('x')
        self.ax2.set_ylabel('y')
        self.ax2.set_xlim(-4, 4)
        self.ax2.set_ylim(-4, 4)

        self.ax1.grid()
        self.ax2.grid()

        self.ax2.plot(targetParameter[0], targetParameter[1], 'ro')

        self.linePoints, = self.ax1.plot([], [], 'ko', lw=2)
        self.lineCurve, = self.ax1.plot([], [], 'r-', lw=2)
        self.line, = self.ax2.plot([], [], 'o', lw=2)
        self.time_template = 'iteration = %.1i'
        self.time_text = self.ax2.text(0.05, 0.9, '', transform=self.ax2.transAxes)

        self.targetParameter = targetParameter

        self.draw_curve(0)

        self.ellipse = self.determineEllipse(0)
        self.ax2.add_patch(self.ellipse)

        animation.TimedAnimation.__init__(self, fig, interval=1200, blit=False, repeat=True)

    def _draw_frame(self, framedata):

        # draw ellipse
        if framedata < len(self.parameters):
            self.ellipse = self.determineEllipse(framedata)
            self.ax2.patches = []
            self.ax2.add_patch(self.ellipse)

        # draw perturbation vector
        if framedata + 1 < len(self.xParameters):
            dx = self.xParameters[framedata + 1] - self.xParameters[framedata]
            dy = self.yParameters[framedata + 1] - self.yParameters[framedata]
            self.ax2.arrow(self.xParameters[framedata], self.yParameters[framedata], dx, dy)

        self.draw_curve(framedata)

        # draw new parameter value in parameter space
        thisx = self.xParameters[framedata]
        thisy = self.yParameters[framedata]

        self.line.set_data(thisx, thisy)

        # draw number of iteration
        self.time_text.set_text(self.time_template % framedata)

        print("Delta parameter = " + str(self.targetParameter - self.parameters[framedata]))

    def new_frame_seq(self):
        return iter(range(len(self.xParameters)))

    def _init_draw(self):
        self.line.set_data([], [])
        self.lineCurve.set_data([], [])
        self.linePoints.set_data([], [])
        self.time_text.set_text('')

    def draw_curve(self, framedata):
        a = self.yParameters[framedata]

        if framedata == 0:
            solution = self.solutions[0][0]
        else:
            solution = self.solutions[framedata]

        rotationMatrix = solution[0]
        translationVector = solution[1]
        t = solution[2]

        x = np.linspace(-3, 3, 100)
        y = a * x ** 2

        xRotAndTrans = rotationMatrix[0, 0] * x + rotationMatrix[0, 1] * y + translationVector[0]
        yRotAndTrans = rotationMatrix[1, 0] * x + rotationMatrix[1, 1] * y + translationVector[1]

        self.lineCurve.set_data(xRotAndTrans, yRotAndTrans)

        self.linePoints.set_data(
            [rotationMatrix[0, 0] * t[0] + rotationMatrix[0, 1] * a * t[0] ** 2 + translationVector[0],
             rotationMatrix[0, 0] * t[1] + rotationMatrix[0, 1] * a * t[1] ** 2 + translationVector[0]],
            [rotationMatrix[1, 0] * t[0] + rotationMatrix[1, 1] * a * t[0] ** 2 + translationVector[1],
             rotationMatrix[1, 0] * t[1] + rotationMatrix[1, 1] * a * t[1] ** 2 + translationVector[1]])

    def determineEllipse(self, framedata):

        metricMatrix = self.parameterSpaceMetrics[framedata]

        deltaParameter = self.targetParameter - self.parameters[framedata]

        eigvals, eigvecs = np.linalg.eig(metricMatrix)

        smallestIndex = np.argmin(eigvals)
        largestIndex = np.argmax(eigvals)

        slope = eigvecs[1, smallestIndex] / eigvecs[0, smallestIndex]
        angle = 180.0 * np.arctan(slope) / np.pi

        if np.inner(deltaParameter, metricMatrix @ deltaParameter) < self.parameterMagnitude ** 2:
            _perturbationMagnitude = np.sqrt(np.inner(deltaParameter, metricMatrix @ deltaParameter))
        else:
            _perturbationMagnitude = self.parameterMagnitude

        return Ellipse(self.parameters[framedata], width=2 * _perturbationMagnitude / np.sqrt(eigvals[smallestIndex]),
                       height=2 * _perturbationMagnitude / np.sqrt(eigvals[largestIndex]), angle=angle, fill=False)

    def show(self):
        plt.show()