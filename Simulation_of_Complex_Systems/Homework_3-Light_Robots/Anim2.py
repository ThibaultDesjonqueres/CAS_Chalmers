import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from LightRobots_wrapper import Pos

#%%


r = np.sin(np.linspace(0,3.14,100))
t = np.linspace(0, 10, 100)
sample_path = np.c_[r*(np.sin(t)+np.cos(t)), r*(np.cos(t)-np.sin(t))]/1.5

fig, ax = plt.subplots()

line, = ax.plot(Pos[0,0], Pos[0,1], "ro-")

def connect(i):
    start=max((i-5,0))
    line.set_data(Pos[start:i,0],Pos[start:i,1])
    return line,

ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
ani = animation.FuncAnimation(fig, connect, np.arange(1, 100), interval=200)
plt.show()