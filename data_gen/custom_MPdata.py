import os
import sys
import ast
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import multiprocessing
import matplotlib.pyplot as plt
from multiprocessing import Pool
from collections import defaultdict
from pymatgen.core.structure import Structure
from pymatgen.core.periodic_table import Element

""" properties
        "material_id", "icsd_ids",
        "unit_cell_formula", "pretty_formula",
        "spacegroup", "cif",
        "volume", "nsites", "elements", "nelements",
        "energy", "energy_per_atom", "formation_energy_per_atom", "e_above_hull",
        "band_gap", "density", "total_magnetization", "elasticity",
        "is_hubbard", "hubbards",
        "warnings", "tags", "crystal_system"
"""

def show_statistics(data, plot=False):
    # size of database
    print('>> Total number of materials: {:d}, number of properties: {:d}'\
            .format(data.shape[0], data.shape[1]))

    # compounds in ICSD
    ICSD_ids = set()
    for ids in data['icsd_ids']:
        ICSD_ids.update(ast.literal_eval(ids))
    non_ICSD = data[data['icsd_ids'] == '[]'].shape[0]
    print('>> Number of ICSD IDs: {:d}, number of compounds not in ICSD: {:d}'\
            .format(len(ICSD_ids), non_ICSD))

    # space group
    sg_set = set()
    for sg in data['spacegroup']:
        sg_dict = ast.literal_eval(sg)
        sg_set.add(sg_dict['number'])
    print('>> Number of unique space groups: {:d}'.format(len(sg_set)))
    
    # crystal system
    syst_set = set()
    for syst in data['crystal_system']:
        syst_set.add(syst)
    print('>> Number of unique crystal systems: {:d}'.format(len(syst_set)))

    # volume
    vol = data['volume']
    print('>> Cell volume (A^3): mean = {:.2f}, median = {:.2f}, '
                'std = {:.2f}, min = {:.2f}, max = {:.2f}' \
                .format(vol.mean(), vol.median(), vol.std(), 
                        vol.min(), vol.max()))
    
    # number of sites
    nsites = data['nsites']
    print('>> Number of sites: mean = {:.1f}, median = {:.1f}, '
                'std = {:.1f}, min = {:d}, max = {:d}' \
                .format(nsites.mean(), nsites.median(), nsites.std(), \
                        nsites.min(), nsites.max()))

    # elements
    elements = data['elements']
    elem_dict = defaultdict(int)
    max_Z = -1
    for compound in elements:
        for elem in ast.literal_eval(compound):
            max_Z = max(Element(elem).Z, max_Z)
            elem_dict[elem] += 1
    min_key = min(elem_dict, key=elem_dict.get)
    max_key = max(elem_dict, key=elem_dict.get)
    print('>> Number of unique elements: {:d}, min: {}({:d}), max: {}({:d})' \
            .format(len(elem_dict), min_key, elem_dict[min_key], \
                                    max_key, elem_dict[max_key]))
    print('>> Max Z: {}'.format(max_Z))

    # energy per atom
    energy_atom = data['energy_per_atom']
    print('>> Energy per atom (eV): mean = {:.2f}, median = {:.2f}, '
                'std = {:.2f}, min = {:.2f}, max = {:.2f}' \
                .format(energy_atom.mean(), energy_atom.median(), energy_atom.std(), \
                        energy_atom.min(), energy_atom.max()))

    # formation energy per atom
    formation_atom = data['formation_energy_per_atom']
    print('>> Formation energy per atom (eV): mean = {:.2f}, median = {:.2f}, '
                'std = {:.2f}, min = {:.2f}, max = {:.2f}' \
                .format(formation_atom.mean(), formation_atom.median(),\
                        formation_atom.std(), formation_atom.min(), formation_atom.max()))

    # energy above hull
    e_above_hull = data['e_above_hull']
    print('>> Energy above hull (eV): mean = {:.2f}, median = {:.2f}, '
                'std = {:.2f}, min = {:.2f}, max = {:.2f}' \
                .format(e_above_hull.mean(), e_above_hull.median(),\
                     e_above_hull.std(), e_above_hull.min(), e_above_hull.max()))

    # band gap 
    gap_threshold = 1E-6
    metals = data[data['band_gap'] <= gap_threshold]['band_gap']
    insulators = data[data['band_gap'] > gap_threshold]['band_gap']
    print('>> Number of metals: {:d}, number of insulators: {:d}' \
                .format(metals.size, insulators.size))
    print('     band gap of all dataset: mean = {:.2f}, median = {:.2f}, '
                 'std = {:.2f}, min = {:.5f}, max = {:.2f}' \
                 .format(data['band_gap'].mean(), data['band_gap'].median(),\
                         data['band_gap'].std(), data['band_gap'].min(), data['band_gap'].max()))
    print('     band gap of insulators: mean = {:.2f}, median = {:.2f}, '
                 'std = {:.2f}, min = {:.5f}, max = {:.2f}' \
                 .format(insulators.mean(), insulators.median(),\
                         insulators.std(), insulators.min(), insulators.max()))

    # warnings
    no_warnings = data[data['warnings'] == '[]']
    print('>> Number of entries with no warnings: {:d}'.format(no_warnings.shape[0]))

    # elasticity
    elastic = data['elasticity'].dropna()
    print('>> Number of elastic data: {:d}'.format(elastic.size))
    Gs, Ks, Ps = [], [], []
    for imat in elastic:
        shear_mod = ast.literal_eval(imat)['G_Voigt_Reuss_Hill']
        if shear_mod > -1E-6: Gs.append(shear_mod)
        bulk_mod = ast.literal_eval(imat)['K_Voigt_Reuss_Hill']
        if bulk_mod > -1E-6: Ks.append(bulk_mod)
        poisson_ratio = ast.literal_eval(imat)['poisson_ratio']
        if poisson_ratio > -1E-6: Ps.append(poisson_ratio)

    print('Shear modulus > 100: {:d}'.format((np.array(Gs)>100).sum()))
    print('Bulk modulus > 200: {:d}'.format((np.array(Ks)>200).sum()))
    print('Shear modulus: mean = {:.2f}, median = {:.2f}, std = {:.2f}, '
                         'min = {:.5f}, max = {:.2f}' \
           .format(np.mean(Gs), np.median(Gs), np.std(Gs), np.min(Gs), np.max(Gs)))
    print('Bulk modulus: mean = {:.2f}, median = {:.2f}, std = {:.2f}, '
                         'min = {:.5f}, max = {:.2f}' \
           .format(np.mean(Ks), np.median(Ks), np.std(Ks), np.min(Ks), np.max(Ks)))
    print('Poisson ratio: mean = {:.2f}, median = {:.2f}, std = {:.2f}, '
                         'min = {:.5f}, max = {:.2f}' \
           .format(np.mean(Ps), np.median(Ps), np.std(Ps), np.min(Ps), np.max(Ps)))


def customize_data(raw_data):
    data_custom = raw_data.copy()
    
    # get rid of rare elements
    if True:
        # identify rare elements
        elem_dict = defaultdict(int)
        for entry in data_custom['elements']:
            for elem in ast.literal_eval(entry):
                elem_dict[elem] += 1
        rare_dict = {key: val for key, val in elem_dict.items() if val < 60}
        print('\n>> Rare elements: ')
        print(rare_dict)
        rare_elements = set(rare_dict.keys())
        # drop entries containing rare elements
        drop_instance = []
        for idx, value in data_custom['elements'].iteritems():
            if rare_elements & set(ast.literal_eval(value)):
                drop_instance.append(idx)
        data_custom = data_custom.drop(drop_instance)


    # only take no-warning entries
    if True:
        data_custom = data_custom[data_custom['warnings'] == '[]']

    # only take crystals in ICSD
    if True:
        data_custom = data_custom[data_custom['icsd_ids'] != '[]']

    return data_custom


def main():
    filename = "./raw_data/fetch_MPdata.csv"
    if not os.path.isfile(filename):
        print("{} file does not exist, please generate it first..".format(filename))
        sys.exit(1)
    # get raw data from Materials Project
    raw_data = pd.read_csv(filename, sep=';', header=0, index_col=None)

    # show statistics of raw data
    print('\nShowing raw data:')
    show_statistics(data=raw_data, plot=False)

    # custom data
    data_custom = customize_data(raw_data)

    # show statistics of customized data
    print('\nShowing customized data:')
    show_statistics(data=data_custom, plot=False)

    # write customized data
    out_file = "./raw_data/custom_MPdata.csv"
    data_custom.to_csv(out_file, sep=';', columns=None, mode='w', \
                       header=data_custom.columns, index=None)


if __name__ == "__main__":
    main()


