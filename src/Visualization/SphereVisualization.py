import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

def animate_scatters(iteration, data, scatters):
    """
    Update the data held by the scatter plot and therefore animates it.
    Args:
        iteration (int): Current iteration of the animation
        data (list): List of the data positions at each iteration.
        scatters (list): List of all the scatters (One per element)
    Returns:
        list: List of scatters (One per element) with new coordinates
    """
    for i in range(data[0].shape[0]):
        scatters[i]._offsets3d = (data[iteration][i,0:1], data[iteration][i,1:2], data[iteration][i,2:])
    return scatters


"""
Creates the 3D figure and animates it with the input data.
Args:
    data (list): List of the data positions at each iteration.
    save (bool): Whether to save the recording of the animation. (Default to False).
"""

solutions = [x for (a,(x,p)) in results]
data = [np.reshape(solution, (1, 3)) for solution in solutions]

# Attaching 3D axis to the figure
fig = plt.figure()
ax = p3.Axes3D(fig)

# draw sphere
u, v = np.mgrid[0 : 2* np.pi : 15j, 0 : np.pi : 15j]
x = np.cos(u) * np.sin(v)
y = np.sin(u) * np.sin(v)
z = np.cos(v)

ax.plot_wireframe(x, y, z, color="y")

ax.scatter(finalSolution[0], finalSolution[1], finalSolution[2])
# Initialize scatters
scatters = [ ax.scatter(data[0][i,0:1], data[0][i,1:2], data[0][i,2:], 'o-') for i in range(data[0].shape[0]) ]

# Number of iterations
iterations = len(data)

# Setting the axes properties
ax.set_xlim3d([-1.5, 1.5])
ax.set_xlabel('X')

ax.set_ylim3d([-1.5, 1.5])
ax.set_ylabel('Y')

ax.set_zlim3d([-1.5, 1.5])
ax.set_zlabel('Z')

ax.set_title('3D Animated Scatter Example')

# Provide starting angle for the view.
ax.view_init(5, 225)

ani = animation.FuncAnimation(fig, animate_scatters, iterations, fargs=(data, scatters),
                                   interval=500, blit=False, repeat=True)

plt.show()