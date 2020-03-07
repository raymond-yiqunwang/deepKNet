import os
import pandas as pd
from pymatgen import MPRester

def fetch_materials_data(out_file):
    # properties of interest
    properties = [ 
        "material_id", "icsd_ids",
        "unit_cell_formula", "pretty_formula",
        "spacegroup", "cif",
        "volume", "nsites", "elements", "nelements",
        "energy", "energy_per_atom", "formation_energy_per_atom", "e_above_hull",
        "band_gap", "density", "total_magnetization", "elasticity",
        "is_hubbard", "hubbards",
        "warnings", "tags",
    ]
    
    # MaterialsProject API settings
    my_API_key = "gxTAyXSm2GvCdWer"
    m = MPRester(api_key=my_API_key)
    
    # query data with calculated band structures
    mp_data = m.query(criteria={
        "has": "bandstructure",
    }, properties=properties)
    
    data_origin = []
    for entry in mp_data:
        plist = []
        for _, val in entry.items():
            plist.append(val)
        data_origin.append(plist)

    data_origin = pd.DataFrame(data_origin)
    data_origin.to_csv(out_file, sep=';', index=False, header=properties, mode='w')


def main():
    # specify IO
    root_dir = "./data_raw/"
    if not os.path.exists(root_dir):
        print("{} folder does not exist, making directory..".format(root_dir))
        os.mkdir(root_dir)
    out_file = root_dir + "fetch_MPdata.csv"

    # fetch raw data from the Materials Project
    fetch_materials_data(out_file)


if __name__ == "__main__":
    main()


