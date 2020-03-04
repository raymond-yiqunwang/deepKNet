import os
import sys
import ast
import math
import json
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from pymatgen import MPRester
from pymatgen.core.structure import Structure


# input -- material_id
# output -- dict[hkl]: (lorentz_factor, i_hkl, atomic_form_factor)
def debug_xrd(material_id):
    my_API_key = "gxTAyXSm2GvCdWer"
    m = MPRester(api_key=my_API_key)
    mp_data = m.query(criteria={
        "material_id" : {'$eq' : material_id },
    }, properties=['cif'])
    
    struct = Structure.from_str(mp_data[0]['cif'], fmt="cif")
    latt = struct.lattice
    wavelength = 1.54184 # CuKa wavelength
    max_r = 2 / wavelength
    recip_latt = latt.reciprocal_lattice_crystallographic
    recip_basis = recip_latt.matrix
    recip_pts = recip_latt.get_points_in_sphere(
        [[0, 0, 0]], [0, 0, 0], max_r)
    npoints = 512
    recip_pts = sorted(recip_pts, key=lambda i: i[1])
    if len(recip_pts) > npoints:
        recip_pts = recip_pts[:npoints]
    
    with open("./XRD_simulator/atomic_scattering_params.json") as f:
        ATOMIC_SCATTERING_PARAMS = json.load(f)
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
    ddict = dict()
    for hkl, g_hkl, _, _ in recip_pts:
        if g_hkl < 1e-12: continue
        d_hkl = 1 / g_hkl
        theta = math.asin(wavelength * g_hkl / 2)
        s = g_hkl / 2
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
        i_hkl = (f_hkl * f_hkl.conjugate()).real
        lorentz = (1 + math.cos(2 * theta) ** 2) / \
            (math.sin(theta) ** 2 * math.cos(theta))
        aff = [0]*94
        for idx in range(zs.size):
            atom_f = atomic_f_hkl[idx]
            Z = zs[idx]
            atomic_intensity = (atom_f * atom_f.conjugate()).real
            aff[Z-1] += atomic_intensity
        ddict[tuple(hkl)] = (i_hkl, lorentz, aff)

    return ddict


# input -- material_id
# output -- band_gap, energy_per_atom, formation_energy_per_atom
def debug_feat(material_id):
    my_API_key = "gxTAyXSm2GvCdWer"
    m = MPRester(api_key=my_API_key)
    mp_data = m.query(criteria={
        "material_id" : {'$eq' : material_id },
    }, properties=["band_gap", "formation_energy_per_atom", "energy_per_atom"])
    
    return mp_data[0]["band_gap"], \
           mp_data[0]["energy_per_atom"], \
           mp_data[0]["formation_energy_per_atom"]


def main():
    # read stored data
    filename = "./data_raw/compute_xrd.csv"
    if not os.path.isfile(filename):
        print("{} file does not exist, please generate it first..".format(filename))
        sys.exit(1)
    data_all = pd.read_csv(filename, sep=';', header=0, index_col=None, chunksize=1000)
    
    # loop over chunks
    for xrd_data in data_all:
        # randomly select 1 material data
        sample = xrd_data.iloc[np.random.randint(xrd_data.shape[0])]
        # properties
        material_id = sample['material_id']
        band_gap = sample['band_gap']
        energy_per_atom = sample['energy_per_atom']
        formation_energy_per_atom = sample['formation_energy_per_atom']
        # calculated point-specific features as dictionary
        hkl = ast.literal_eval(sample['hkl'])
        i_hkl = ast.literal_eval(sample['i_hkl'])
        lorentz_factor = ast.literal_eval(sample['lorentz_factor'])
        atomic_form_factor = ast.literal_eval(sample['atomic_form_factor'])
        ddict = dict()
        for idx, ihkl in enumerate(hkl):
            ddict[tuple(ihkl)] = (i_hkl[idx], lorentz_factor[idx],
                          atomic_form_factor[idx])

        # threshold for numerical comparison
        threshold = 1e-8
        
        # debug property 
        gap_debug, energy_debug, fenergy_debug = debug_feat(material_id)
        assert(abs(gap_debug-band_gap) < threshold)
        assert(abs(energy_debug-energy_per_atom) < threshold)
        assert(abs(fenergy_debug-formation_energy_per_atom) < threshold)
        
        # debug primitive features
        debug_dict = debug_xrd(material_id)
        for key, val in ddict.items():
            intensity, lorentz, aff = val
            debug_intensity = debug_dict[key][0]
            debug_lorentz = debug_dict[key][1]
            debug_aff = debug_dict[key][2]
            assert(abs(intensity-debug_intensity) < threshold)
            assert(abs(lorentz-debug_lorentz) < threshold)
            for idx, ival in enumerate(aff):
                assert(abs(ival-debug_aff[idx]) < threshold)
        print('>> passing one chunk..')

    print("All test cases passed, good to go training your model..")


if __name__ == "__main__":
    main()


