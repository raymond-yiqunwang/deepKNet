import os
import shutil
import ast
import pandas as pd


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


def generate_point_cloud(xrd_data, out_dir_root):
    # safeguard
    _ = input("Attention, all existing training data will be deleted and regenerated.. \
        \n>> Hit Enter to continue, Ctrl+c to terminate..")
    print("Started generating dataset..")

    # remove existing csv files
    target_dir = out_dir_root + '/target/'
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir, ignore_errors=True)
    os.mkdir(target_dir)
    features_dir = out_dir_root + '/features/'
    if os.path.exists(features_dir):
        shutil.rmtree(features_dir, ignore_errors=True)
    os.mkdir(features_dir)

    # store point cloud representation for each material
    for idx, irow in xrd_data.iterrows():
        if (idx+1)%500 == 0:
            print('>> Generated dataset size: {}'.format(idx+1))
        # unique ID
        material_id = irow['material_id']
        filename = str(material_id) + '.csv'
        if os.path.exists(features_dir+filename):
            print('duplicate material_id detected, check data source..')
            sys.exit(1)

        # features
        hkl = ast.literal_eval(irow['hkl'])
        recip_xyz = ast.literal_eval(irow['recip_xyz'])
        recip_spherical = ast.literal_eval(irow['recip_spherical'])
        i_hkl_lorentz = ast.literal_eval(irow['i_hkl_lorentz'])
        atomic_form_factor = ast.literal_eval(irow['atomic_form_factor'])
        max_r = float(irow['max_r'])
        # pick features wisely
        features = [recip_spherical[i]+atomic_form_factor[i] for i in range(len(hkl))]
        features = pd.DataFrame(features)
        # normalize features
        features.iloc[:, 0] = features.iloc[:, 0] / max_r
        features.iloc[:, 3:-1] = features.iloc[:, 3:-1] / max(max(atomic_form_factor))
        # transpose features to accommodate PyTorch tensor style
        features_T = features.transpose()
        assert(features_T.shape[0] == 3+94)
        assert(features_T.shape[1] == 512)
        # write features_T
        features_T.to_csv(features_dir+filename, sep=';', header=None, index=False)

        # target properties
        band_gap = irow['band_gap']
        energy_per_atom = irow['energy_per_atom']
        formation_energy_per_atom = irow['formation_energy_per_atom']
        # write target
        properties = [[band_gap, energy_per_atom, formation_energy_per_atom]]
        header = ['band_gap', 'energy_per_atom', 'formation_energy_per_atom']
        properties = pd.DataFrame(properties)
        properties.to_csv(target_dir+filename, sep=';', header=header, index=False)


def main():
    # read xrd raw data
    xrd_data = pd.read_csv("./data_raw/compute_xrd.csv", sep=';', header=0, index_col=None)

    out_dir_root = "../data/"
    # generate training data
    generate_point_cloud(xrd_data, out_dir_root)


if __name__ == "__main__":
    main()


