# plot XRD diffraction intensity in the Ewald sphere
import numpy as np
import ast
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

data = pd.read_csv("../gen_data/data_raw/compute_xrd.csv", sep=';', header=0, index_col=None)
ddict = {}
for idx, irow in data.iterrows():
    hkl_list = ast.literal_eval(irow['hkl'])
    intensity_list = ast.literal_eval(irow['intensity_hkl'])
    for idx in range(len(hkl_list)):
        ddict[tuple(hkl_list[idx])] = intensity_list[idx]
    for key, val in sorted(ddict.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        if max(np.abs(key)) < 3:
            print(key, val)
