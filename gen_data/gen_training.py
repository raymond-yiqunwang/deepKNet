import pandas as pd
import XRD_simulator.xrd_simulator as xrd
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


def generate_xrd(data_raw):

    xrd_data = []
    xrd_simulator = xrd.XRDSimulator(wavelength='CuKa')
    for idx, irow in data_raw.iterrows():
        if (idx+1)%500 == 0: print('>> Processed materials: {}'.format(idx+1))
        # obtain xrd features
        struct = Structure.from_str(irow['cif'], fmt="cif")
        _, features = xrd_simulator.get_pattern(struct)
        """
          features: nrow = (the number of reciprocal kpoints in that material)
          features: ncol = (hkl  , recip_xyz, recip_spherical, i_hkl_corrected, atomic_form_factor)
                            [1x3], [1x3]    , [1x3]          , [1x1](scalar)  , [1x120]
        """
        flat_features = [ifeat for sublist in features for ifeat in sublist]
        # properties of interest
        band_gap = irow['band_gap']
        energy_per_atom = irow['energy_per_atom']
        formation_energy_per_atom = irow['formation_energy_per_atom']
        # finish collecting one material
        xrd_data.append(flat_features + [band_gap, energy_per_atom, formation_energy_per_atom])
    
    xrd_data = pd.DataFrame(xrd_data)
    return xrd_data


def main():
    # read customized data
    MP_data = pd.read_csv("./data_raw/custom_MPdata.csv", sep=';', header=0, index_col=None)

    # generate xrd point cloud representations
    xrd_data = generate_xrd(MP_data)

    # write customized data
    xrd_data.to_csv("./data_raw/compute_xrd.csv", sep=';', header=None, index=False)


if __name__ == "__main__":
    main()


