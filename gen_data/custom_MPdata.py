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

""" properties
        "material_id", "icsd_ids",
        "unit_cell_formula", "pretty_formula",
        "spacegroup", "cif",
        "volume", "nsites", "elements", "nelements",
        "energy", "energy_per_atom", "formation_energy_per_atom", "e_above_hull",
        "band_gap", "density", "total_magnetization", "elasticity",
        "is_hubbard", "hubbards",
        "warnings", "tags",
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
    for compound in elements:
        for elem in ast.literal_eval(compound):
            elem_dict[elem] += 1
    min_key = min(elem_dict, key=elem_dict.get)
    max_key = max(elem_dict, key=elem_dict.get)
    print('>> Number of unique elements: {:d}, min: {}({:d}), max: {}({:d})' \
            .format(len(elem_dict), min_key, elem_dict[min_key], \
                                    max_key, elem_dict[max_key]))

    # energy per atom
    energy_atom = data['energy_per_atom']
    print('>> Energy per atom (eV): mean = {:.2f}, median = {:.2f}, '
                'std = {:.2f}, min = {:.2f}, max = {:.2f}' \
                .format(energy_atom.mean(), energy_atom.median(), energy_atom.std(), \
                        energy_atom.min(), energy_atom.max()))
    if plot:
        energy_atom.plot.hist(bins=20)
        plt.show()

    # formation energy per atom
    formation_atom = data['formation_energy_per_atom']
    print('>> Formation energy per atom (eV): mean = {:.2f}, median = {:.2f}, '
                'std = {:.2f}, min = {:.2f}, max = {:.2f}' \
                .format(formation_atom.mean(), formation_atom.median(),\
                        formation_atom.std(), formation_atom.min(), formation_atom.max()))
    if plot:
        formation_atom.plot.hist(bins=20)
        plt.show()

    # energy above hull
    e_above_hull = data['e_above_hull']
    print('>> Energy above hull (eV): mean = {:.2f}, median = {:.2f}, '
                'std = {:.2f}, min = {:.2f}, max = {:.2f}' \
                .format(e_above_hull.mean(), e_above_hull.median(),\
                     e_above_hull.std(), e_above_hull.min(), e_above_hull.max()))

    # band gap TODO determine threshold
    gap_threshold = 1E-3
    metals = data[data['band_gap'] <= gap_threshold]['band_gap']
    insulators = data[data['band_gap'] > gap_threshold]['band_gap']
    if plot:
        data['band_gap'].plot.hist(bins=20)
        plt.show()
        insulators.plot.hist(bins=20)
        plt.show()
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


def customize_data(data_raw):
    data_custom = data_raw.copy()

    # only take no-warning entries
    if True:
        data_custom = data_custom[data_custom['warnings'] == '[]']

    # only take crystals in ICSD
    if True:
        data_custom = data_custom[data_custom['icsd_ids'] != '[]']

    # get rid of extreme volumes
    if False:
        data_custom = data_custom[data_custom['volume'] > 100]
        data_custom = data_custom[data_custom['volume'] < 800]

    # get rid of rare elements
    if True:
        # identify rare elements
        elem_dict = defaultdict(int)
        for entry in data_custom['elements']:
            for elem in ast.literal_eval(entry):
                elem_dict[elem] += 1
        rare_dict = {key: val for key, val in elem_dict.items() if val < 60}
        print('>> Rare elements: ')
        print(rare_dict)
        rare_elements = set(rare_dict.keys())
        # drop entries containing rare elements
        drop_instance = []
        for idx, value in data_custom['elements'].iteritems():
            if rare_elements & set(ast.literal_eval(value)):
                drop_instance.append(idx)
        data_custom = data_custom.drop(drop_instance)

    return data_custom


def main():
    filename = "./data_raw/fetch_MPdata.csv"
    if not os.path.isfile(filename):
        print("{} file does not exist, please generate it first..".format(filename))
        sys.exit(1)
    # get raw data from Materials Project
    data_raw = pd.read_csv(filename, sep=';', header=0, index_col=None)

    # show statistics of raw data
    print('\nShowing raw data:')
    show_statistics(data=data_raw, plot=False)

    # custom data
    data_custom = customize_data(data_raw)

    # show statistics of customized data
    print('\nShowing customized data:')
    show_statistics(data=data_custom, plot=False)

    # write customized data
    out_file = "./data_raw/custom_MPdata.csv"
    data_custom.to_csv(out_file, sep=';', columns=None, mode='w', \
                       header=data_custom.columns, index=None)


if __name__ == "__main__":
    main()


