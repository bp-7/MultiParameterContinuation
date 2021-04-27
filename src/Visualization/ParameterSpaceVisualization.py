import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

parameters = [p for (a,(x,p)) in results]

x = [x for (x,y) in parameters]
y = [y for (x,y) in parameters]

fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=True, xlim=(-30, 45), ylim=(-30, 55))
ax.grid()

line, = ax.plot([], [], 'o', lw=2)
time_template = 'iteration = %.1i'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

#Other stuff
ax.grid()

def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text


def animate(i):
    thisx = [x[i], targetParameter[0]]
    thisy = [y[i], targetParameter[1]]

    line.set_data(thisx, thisy)
    #line.set_colors(['b', 'r'])
    time_text.set_text(time_template % (i))
    return line, time_text

ani = animation.FuncAnimation(fig, animate, np.arange(1, len(y)),
                              interval=500, blit=True, init_func=init)