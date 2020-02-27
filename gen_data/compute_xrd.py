import os
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


def compute_xrd(data_raw):
    # specify output
    out_file = "./data_raw/compute_xrd.csv"
    # remove existing csv file
    if os.path.exists(out_file):
        os.remove(out_file)

    # define batch size and npoints
    chunksize = 500
    npoints = 512
    header = ['material_id', 'band_gap', 'energy_per_atom', 'formation_energy_per_atom', \
        'hkl', 'recip_xyz', 'recip_spherical', 'intensity', 'atomic_form_factor']

    xrd_data_batch = []
    xrd_simulator = xrd.XRDSimulator(wavelength='AgKa')
    for idx, irow in data_raw.iterrows():
        # obtain xrd features
        struct = Structure.from_str(irow['cif'], fmt="cif")
        _, features = xrd_simulator.get_pattern(struct, npoints=npoints)
        assert(len(features) == npoints)
        """
          features: nrow = number of reciprocal kpoints (npoints)
          features: ncol = (hkl  , recip_xyz, recip_spherical, i_hkl_corrected, atomic_form_factor)
                            [1x3], [1x3]    , [1x3]          , [1x1](scalar)  , [1x120]
        """
        # regroup features
        hkl = [ipoint[0] for ipoint in features]
        recip_xyz = [ipoint[1] for ipoint in features]
        recip_spherical = [ipoint[2] for ipoint in features]
        i_hkl_corrected = [ipoint[3] for ipoint in features]
        atomic_form_factor = [ipoint[4] for ipoint in features]

        # properties of interest
        material_id = irow['material_id']
        band_gap = irow['band_gap']
        energy_per_atom = irow['energy_per_atom']
        formation_energy_per_atom = irow['formation_energy_per_atom']

        # finish collecting one material
        ifeat = [material_id, band_gap, energy_per_atom, formation_energy_per_atom] 
        ifeat.extend([hkl, recip_xyz, recip_spherical, i_hkl_corrected, atomic_form_factor])
        xrd_data_batch.append(ifeat)

        # process batch
        if (idx+1)%chunksize == 0:
            print('>> Processed materials: {}'.format(idx+1))
            # write to file
            xrd_data_batch = pd.DataFrame(xrd_data_batch)
            xrd_data_batch.to_csv(out_file, sep=';', header=header, index=False, mode='a')
            # clear data list
            xrd_data_batch = [] 


def main():
    # read customized data
    MP_data = pd.read_csv("./data_raw/custom_MPdata.csv", sep=';', header=0, index_col=None)

    # generate xrd point cloud representations
    compute_xrd(MP_data)


if __name__ == "__main__":
    main()


