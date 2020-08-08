# plot XRD diffraction intensity in the Ewald sphere
import math
import numpy as np
import json
import pandas as pd
from pymatgen.core.structure import Structure

with open("./NaCl_cubic.cif") as f1:
    cif = f1.read()
struct = Structure.from_str(cif, fmt="cif")
latt = struct.lattice
print(latt)
wavelength = 1.54184 # CuKa wavelength
max_r = 2 / wavelength
recip_latt = latt.reciprocal_lattice_crystallographic
recip_pts = recip_latt.get_points_in_sphere(
    [[0, 0, 0]], [0, 0, 0], max_r)
with open("../gen_data/XRD_simulator/atomic_scattering_params.json") as f2:
    ATOMIC_SCATTERING_PARAMS = json.load(f2)
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
print('fcoords')
print(fcoords)
ddict = {}
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
    if intensity_hkl < 1e-8: intensity_hkl = 0.
    ddict[tuple(hkl)] = intensity_hkl

for key, val in sorted(ddict.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
    if max(np.abs(key)) < 3:
        print(key, val)
