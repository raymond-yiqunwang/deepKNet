# plot XRD diffraction intensity in the Ewald sphere
import numpy as np
import ast
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

u = np.linspace(0, 2*np.pi, 100)
v = np.linspace(0, np.pi, 100)

r_max = 2. / 1.54184
rx = r_max * np.outer(np.cos(u), np.sin(v))
ry = r_max * np.outer(np.sin(u), np.sin(v))
rz = r_max * np.outer(np.ones(np.size(u)), np.cos(v))

data = pd.read_csv("../gen_data/data_raw/compute_xrd.csv", sep=';', header=0, index_col=None)
for idx, irow in data.iterrows():
    recip_latt = ast.literal_eval(irow['recip_latt'])
    hkl_list = ast.literal_eval(irow['hkl'])
    recip_xyz = [np.dot(np.array(recip_latt).T, np.array(hkl_list[idx])) \
                    for idx in range(len(hkl_list))]
    intensity_list = ast.literal_eval(irow['intensity_hkl'])
    xs = []
    ys = []
    zs = []
    Is = []
    for pid in range(len(hkl_list)):
        xyz = recip_xyz[pid]
        x, y, z= xyz[0], xyz[1], xyz[2]
        intensity = intensity_list[pid]
        xs.append(x)
        ys.append(y)
        zs.append(z)
        Is.append(intensity)
    c = Is

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    axis_min = -1*r_max
    axis_min *= 1
    axis_max = r_max
    axis_max *= 1
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    ax.set_zlim(axis_min, axis_max)
    surf = ax.scatter(xs, ys, zs, c=c)
    fig.colorbar(surf)
    
    ax.plot_surface(rx, ry, rz, rstride=5, cstride=5, color='k', alpha=0.05)
    plt.axis('off')

    plt.show()
