#! /home/raymondw/.conda/envs/pymatgen/bin/python
from pymatgen import MPRester
import pandas as pd

# relevant properties:
prop_list = [ 
    "material_id", "icsd_id", "icsd_ids", "pretty_formula", "unit_cell_formula", 
    "elements",  "nelements", "nsites", "spacegroup", "cif", "volume",
    "energy", "energy_per_atom", "e_above_hull", "formation_energy_per_atom",
    "density", "elasticity", "piezo",  "diel", "band_gap", "total_magnetization",
    "is_hubbard",  "hubbards",  "is_compatible",
    "oxide_type", "tags"
]

# fetch data
my_API_key = "gxTAyXSm2GvCdWer"
m = MPRester(api_key=my_API_key)
mp_data = m.query(criteria={"volume": {"$gt": -1}}, properties=[ prop for prop in prop_list ])

# convert to DataFrame
data = []
for entry in mp_data:
    plist = []
    for prop, val in entry.items():
        plist.append(val)
    data.append(plist)

data = pd.DataFrame(data, index=None, columns=None)
data.to_csv("./data/MP_data.csv", sep=';', columns=None, header=prop_list, index=None)
