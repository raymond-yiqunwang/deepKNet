import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymatgen.core.structure import Structure

with open("../gen_data/XRD_simulator/atomic_scattering_params.json") as f1:
    ATOMIC_SCATTERING_PARAMS = json.load(f1)

with open("./Na_prim.cif") as f2:
    cif = f2.read()
struct = Structure.from_str(cif, fmt="cif")
latt = struct.lattice

xs = []
ys = []

#wavelength = 1.54184 # CuKa wavelength
wavelengths = np.linspace(0.5, 1.5, 10)
for wavelength in wavelengths:
    max_r = 2 / wavelength
    recip_latt = latt.reciprocal_lattice_crystallographic
    recip_pts = recip_latt.get_points_in_sphere(
        [[0, 0, 0]], [0, 0, 0], max_r)
    print(len(recip_pts))
    zs = []
    coeffs = []
    fcoords = []
    for site in struct:
        sp = site.specie
        zs.append(sp.Z)
        c = ATOMIC_SCATTERING_PARAMS[sp.symbol]
        coeffs.append(c)
        fcoords.append(site.frac_coords)
    zs = np.array(zs)
    coeffs = np.array(coeffs)
    fcoords = np.array(fcoords)
    F_sum = 0
    for hkl, g_hkl, _, _ in recip_pts:
        if g_hkl < 1e-12: continue
        hkl = [int(round(i)) for i in hkl]
        d_hkl = 1 / g_hkl
        s = g_hkl / 2
        theta = math.asin(wavelength * s)
        s2 = s ** 2
        g_dot_r = np.dot(fcoords, np.transpose([hkl])).T[0]
        fs = []
        for site in struct:
            el = site.specie
            coeff = ATOMIC_SCATTERING_PARAMS[el.symbol]
            f = el.Z - 41.78214 * s2 * np.sum(
                [d[0] * np.exp(-d[1] * s2) for d in coeff])
            fs.append(f)
        fs = np.array(fs)
        atomic_f_hkl = fs * np.exp(2j * math.pi * g_dot_r)
        f_hkl = np.sum(atomic_f_hkl)
        intensity_hkl = (f_hkl * f_hkl.conjugate()).real
        F_sum += intensity_hkl
    xs.append(max_r)
    ys.append(F_sum/len(recip_pts))

plt.xlabel('max R')
plt.ylabel('sum(F^2)')
plt.scatter(xs, ys)
plt.show()
