import pandas as pd
import ast
from collections import defaultdict


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
    print('>> Cell volume (A^3): mean = {:.2f}, median = {:.2f}, standard deviation = {:.2f}, '
                    'min = {:.2f}, max = {:.2f}'.format(vol.mean(), vol.median(),\
                     vol.std(), vol.min(), vol.max()))

    # number of sites
    nsites = data['nsites']
    print('>> Number of sites: mean = {:.1f}, median = {:.1f}, standard deviation = {:.1f}, '
                    'min = {:d}, max = {:d}'.format(nsites.mean(), nsites.median(),\
                     nsites.std(), nsites.min(), nsites.max()))

    # elements
    elements = data['elements']
    elem_dict = defaultdict(int)
    for compound in elements:
        for elem in ast.literal_eval(compound):
            elem_dict[elem] += 1
    min_key = min(elem_dict, key=elem_dict.get)
    max_key = max(elem_dict, key=elem_dict.get)
    print('>> Number of unique elements: {:d}, min: {:d}, max: {:d}'\
            .format(len(elem_dict), elem_dict[min_key], elem_dict[max_key]))

    # energy per atom
    energy_atom = data['energy_per_atom']
    print('>> Energy per atom (eV): mean = {:.2f}, median = {:.2f}, standard deviation = {:.2f}, '
                    'min = {:.2f}, max = {:.2f}'.format(energy_atom.mean(), energy_atom.median(),\
                     energy_atom.std(), energy_atom.min(), energy_atom.max()))

    # formation energy per atom
    formation_atom = data['formation_energy_per_atom']
    print('>> Formation energy per atom (eV): mean = {:.2f}, median = {:.2f}, standard deviation = {:.2f}, '
                    'min = {:.2f}, max = {:.2f}'.format(formation_atom.mean(), formation_atom.median(),\
                     formation_atom.std(), formation_atom.min(), formation_atom.max()))

    # energy above hull
    e_above_hull = data['e_above_hull']
    print('>> Energy above hull (eV): mean = {:.2f}, median = {:.2f}, standard deviation = {:.2f}, '
                    'min = {:.2f}, max = {:.2f}'.format(e_above_hull.mean(), e_above_hull.median(),\
                     e_above_hull.std(), e_above_hull.min(), e_above_hull.max()))

    # band gap TODO determine threshold
    metals = data[data['band_gap'] <= 1E-3]['band_gap']
    insulators = data[data['band_gap'] > 1E-3]['band_gap']
    print('>> Number of metals: {:d}, number of insulators: {:d}'.format(metals.size, insulators.size))
    print('     band gap of insulators: mean = {:.2f}, median = {:.2f}, standard deviation = {:.2f}, '
                        'min = {:.5f}, max = {:.2f}'.format(insulators.mean(), insulators.median(),\
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

    # get rid of extreme volumes TODO determine threshold
    if True:
        data_custom = data_custom[data_custom['volume'] < 600]

    # get rid of rare elements
    if True:
        rare_elements = []
        elements = data_custom['elements']
        elem_dict = defaultdict(int)
        for compound in elements:
            for elem in ast.literal_eval(compound):
                elem_dict[elem] += 1
        print(elem_dict)

    """
    # rare elements
    rare_elements = []
    elem_dict = defaultdict(int)
    for elements in data_origin['elements']:
        for elem in elements:
            elem_dict[Element(elem).Z] += 1
    for elem, count in elem_dict.items():
        if (count < int(0.01*data_origin.shape[0])):
            rare_elements.append(elem)
            print("Element No. {} has a count of {}.".format(elem, count))

    # customize data
    data_custom = data_origin[['band_gap', 'nsites', 'volume', 'cif', 'elements']].copy()
    drop_instance = []
    for idx, irow in data_custom.iterrows():
        if (True in [ (Element(elem).Z in rare_elements) for elem in irow['elements']]) \
            or (irow['nsites'] > 50) or (irow['volume'] > 800.):
            drop_instance.append(idx)
    data_custom = data_custom.drop(drop_instance)


    # number of points
    npoint_list = []
    xrdcalc = xrd.XRDCalculator(wavelength=wavelength)
    for _, irow in data_custom.iterrows():
        struct = Structure.from_str(irow['cif'], fmt="cif")
        npoint_list.append(xrdcalc.get_npoint(struct))
    
    npoint_array = np.asarray(npoint_list)
    print(" npoint: mean = {:.2f}, median = {:.2f}, standard deviation = {:.2f}, min = {:.2f}, max = {:.2f}"
          .format(np.mean(npoint_array), np.median(npoint_array), np.std(npoint_array), np.min(npoint_array), np.max(npoint_array)))
    """

    return data_custom


def main():
    # get raw data from Materials Project
    data_raw = pd.read_csv("../data/MP_data_has_band.csv", sep=';', header=0, index_col=None)

    # show statistics of raw data
    print('\nShowing raw data:')
    show_statistics(data=data_raw, plot=False)

    # custom data
    data_custom = customize_data(data_raw)

    # show statistics of customized data
    print('\nShowing customized data:')
    show_statistics(data=data_custom, plot=False)


if __name__ == "__main__":
    main()


