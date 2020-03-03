import os
import pandas as pd
from pymatgen import MPRester


def fetch_materials_data():
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
    if not os.path.exists("./data_raw/"):
        print("./data_raw/ folder does not exist, making directory..")
        os.mkdir("./data_raw/")
    data_origin.to_csv("./data_raw/fetch_MPdata.csv", sep=';', index=False, header=properties)


def main():
    # fetch raw data from the Materials Project
    fetch_materials_data()


if __name__ == "__main__":
    main()


