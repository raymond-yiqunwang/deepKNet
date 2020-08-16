import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

u = np.linspace(0, 2*np.pi, 100)
v = np.linspace(0, np.pi, 100)

r_max = 2. / 1.54184
rx = r_max * np.outer(np.cos(u), np.sin(v))
ry = r_max * np.outer(np.sin(u), np.sin(v))
rz = r_max * np.outer(np.ones(np.size(u)), np.cos(v))

data = np.load('mp-981386.npy')

xs = data[:,0]
ys = data[:,1]
zs = data[:,2]
Is = data[:,3]

fig = plt.figure()
ax = plt.axes(projection='3d')
axis_min = -1*r_max
axis_min *= 1
axis_max = r_max
axis_max *= 1
ax.set_xlim(axis_min, axis_max)
ax.set_ylim(axis_min, axis_max)
ax.set_zlim(axis_min, axis_max)
surf = ax.scatter(xs, ys, zs, c=Is)
fig.colorbar(surf)

ax.plot_surface(rx, ry, rz, rstride=5, cstride=5, color='k', alpha=0.05)
plt.axis('off')

plt.show()
