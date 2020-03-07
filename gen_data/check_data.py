import os
import sys
import ast
import math
import json
import argparse
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from pymatgen import MPRester
from pymatgen.core.structure import Structure

parser = argparse.ArgumentParser()
parser.add_argument('--debug', dest='debug', action='store_true')
args = parser.parse_args()

# input -- material_id
# output -- recip_latt, dict{(hkl): (i_hkl, atomic_form_factor)}
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
    recip_pts = recip_latt.get_points_in_sphere(
        [[0, 0, 0]], [0, 0, 0], max_r)
    
    # the lattice and recip_lattice basis should be orthogonal
    orthog = np.dot(latt.matrix, np.transpose(recip_latt.matrix))
    orthog -= np.eye(3)
    assert(max(np.abs(orthog.flatten())) < 1e-12)

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
        i_hkl = (f_hkl * f_hkl.conjugate()).real
        aff = [0]*94
        for idx, Z in enumerate(zs):
            atom_f = atomic_f_hkl[idx]
            atomic_intensity = (atom_f * atom_f.conjugate()).real
            aff[Z-1] += atomic_intensity
        ddict[tuple(hkl)] = (i_hkl, aff)

    return recip_latt.matrix, ddict


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


def debug_compute_xrd(chunksize, xrd_filename):
    if not os.path.isfile(xrd_filename):
        print("{} file does not exist, please generate it first..".format(xrd_filename))
        sys.exit(1)
    data_all = pd.read_csv(xrd_filename, sep=';', header=0, index_col=None, chunksize=chunksize)
    
    # loop over chunks
    for xrd_data in data_all:
        # randomly select 1 material data
        sample = xrd_data.iloc[np.random.randint(xrd_data.shape[0])]
        # properties
        material_id = sample['material_id']
        band_gap = sample['band_gap']
        energy_per_atom = sample['energy_per_atom']
        formation_energy_per_atom = sample['formation_energy_per_atom']
        recip_latt = ast.literal_eval(sample['recip_latt'])
        # calculated point-specific features as dictionary
        hkl = ast.literal_eval(sample['hkl'])
        i_hkl = ast.literal_eval(sample['i_hkl'])
        atomic_form_factor = ast.literal_eval(sample['atomic_form_factor'])
        ddict = dict()
        for idx, ihkl in enumerate(hkl):
            ddict[tuple(ihkl)] = (i_hkl[idx], atomic_form_factor[idx])

        # threshold for numerical comparison
        threshold = 1e-12

        # debug property 
        gap_debug, energy_debug, fenergy_debug = debug_feat(material_id)
        assert(abs(gap_debug-band_gap) < threshold)
        assert(abs(energy_debug-energy_per_atom) < threshold)
        assert(abs(fenergy_debug-formation_energy_per_atom) < threshold)

        # debug primitive features
        debug_recip_latt, debug_dict = debug_xrd(material_id)
        diff = np.hstack(recip_latt) - np.hstack(debug_recip_latt)
        assert(max(np.abs(diff)) < threshold)
        for key, val in ddict.items():
            intensity, aff = val
            debug_intensity = debug_dict[key][0]
            debug_aff = debug_dict[key][1]
            assert(abs(intensity-debug_intensity) < threshold)
            for idx, ival in enumerate(aff):
                assert(abs(ival-debug_aff[idx]) < threshold)
        print('>> passing one chunk of xrd data..')

    print("All test cases passed for compute_xrd.csv\n")


def read_training(material_id, train_data_root):
    features = pd.read_csv(train_data_root+'features/'+material_id+'.csv', \
                           sep=';', header=None, index_col=None)
    target = pd.read_csv(train_data_root+'target/'+material_id+'.csv', \
                          sep=';', header=0, index_col=None)
    return features, target
           

def cart2sphere(xyz):
    x, y, z = xyz[0], xyz[1], xyz[2]
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos([z/r])[0]
    phi = np.arctan2(y, x)
    r = r / (2/1.54184)
    theta /= math.pi
    phi = 0.5*(phi/math.pi)+0.5
    return [r, theta, phi]


def debug_training_features(chunksize, xrd_filename, train_data_root):
    if not os.path.isfile(xrd_filename):
        print("{} file does not exist, please generate it first..".format(xrd_filename))
        sys.exit(1)
    data_all = pd.read_csv(xrd_filename, sep=';', header=0, index_col=None, chunksize=chunksize)

    npoints = 512
    
    # loop over chunks
    for xrd_data in data_all:
        # randomly select 1 material data
        sample = xrd_data.iloc[np.random.randint(xrd_data.shape[0])]
        # properties
        material_id = sample['material_id']
        band_gap = sample['band_gap']
        energy_per_atom = sample['energy_per_atom']
        formation_energy_per_atom = sample['formation_energy_per_atom']
        recip_latt = ast.literal_eval(sample['recip_latt'])
        # calculated point-specific features as dictionary
        hkl = ast.literal_eval(sample['hkl'])
        while len(hkl) < npoints:
            hkl.extend(hkl)
        hkl = hkl[:npoints]
        i_hkl = ast.literal_eval(sample['i_hkl'])
        while len(i_hkl) < npoints:
            i_hkl.extend(i_hkl)
        i_hkl = i_hkl[:npoints]
        atomic_form_factor = ast.literal_eval(sample['atomic_form_factor'])
        while len(atomic_form_factor) < npoints:
            atomic_form_factor.extend(atomic_form_factor)
        atomic_form_factor = atomic_form_factor[:npoints]
        # process primitive features
        recip_xyz = [np.dot(np.transpose(recip_latt), hkl[idx]) for idx in range(len(hkl))]
        recip_spherical = [cart2sphere(recip_xyz[idx]) for idx in range(len(recip_xyz))]
        intensity = np.array(i_hkl) / max(i_hkl)
        for idx in range(len(atomic_form_factor)):
            imax = max(1, max(atomic_form_factor[idx]))
            atomic_form_factor[idx] = (np.array(atomic_form_factor[idx])/imax).tolist()

        # read training features
        features, target = read_training(material_id, train_data_root)
        
        # threshold for numerical comparison
        threshold = 1e-12
        
        # debug targets
        assert(abs(target['band_gap'][0]-band_gap) < threshold)
        assert(abs(target['energy_per_atom'][0]-energy_per_atom) < threshold)
        assert(abs(target['formation_energy_per_atom'][0]-formation_energy_per_atom) < threshold)

        # debug features
        for i in range(len(hkl)):
            # check intensity
            assert(abs(intensity[i]-features.iloc[3, i]) < threshold)
            for j in range(len(recip_spherical[i])):
                # check spherical coord
                assert(abs(recip_spherical[i][j]-features.iloc[j, i]) < threshold)
            for k in range(len(atomic_form_factor[i])):
                # check aff
                assert(abs(atomic_form_factor[i][k]-features.iloc[4+k, i]) < threshold)
        print('>> passing one chunk in the training set..')

    print("All test cases passed for training data, good to go start training!")


def main():
    global args

    if not args.debug:
        chunksize = 100
        xrd_filename = "./data_raw/compute_xrd.csv"
        train_data_root = "../data/"
    else:
        chunksize = 10
        xrd_filename = "./data_raw/debug_compute_xrd.csv"
        train_data_root = "./data_raw/debug_data/"

    # debug xrd data
    debug_compute_xrd(chunksize, xrd_filename)
    
    # debug training data
    debug_training_features(chunksize, xrd_filename, train_data_root)


if __name__ == "__main__":
    main()


