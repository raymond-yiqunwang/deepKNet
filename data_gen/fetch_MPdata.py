import os
import shutil
import pandas as pd
from pymatgen import MPRester

def fetch_materials_data():
    # specify IO
    out_dir = "./MPdata_all/"
    out_file = os.path.join(out_dir, "MPdata_all.csv")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)

    # properties of interest
    properties = [ 
        "material_id", "icsd_ids", "cif",
        "unit_cell_formula", "pretty_formula",
        "spacegroup", "crystal_system"
        "volume", "nsites", "elements", "nelements",
        "energy", "energy_per_atom", "formation_energy_per_atom", "e_above_hull",
        "band_gap", "elasticity", "density", "total_magnetization",
        "warnings", "tags"
    ]
    
    # MaterialsProject API settings
    my_API_key = "gxTAyXSm2GvCdWer"
    m = MPRester(api_key=my_API_key)
    
    # query all materials data
    query_all = m.query(criteria={"volume": {"$lt": 100}}, properties=properties)
    MPdata_all = pd.DataFrame(entry.values() for entry in query_all)
    MPdata_all.columns = properties

    # write cif to file
    for _, irow in MPdata_all[["material_id", "cif"]].iterrows():
        cif_file = os.path.join(out_dir, irow["material_id"] + ".cif")
        with open(cif_file, 'w') as f:
            f.write(irow["cif"])
    MPdata_all = MPdata_all.drop(columns=["cif"])

    # materials with calculated band structures
    query_band = m.query(criteria={"has": "bandstructure", "volume": {"$lt": 100}}, 
                         properties=["material_id"])
    band_filenames = [list(entry.values())[0] for entry in query_band]

    MPdata_all['has_band_structure'] = MPdata_all["material_id"].isin(band_filenames)
    MPdata_all.to_csv(out_file, sep=';', index=False, header=MPdata_all.columns, mode='w')


if __name__ == "__main__":
    fetch_materials_data()


