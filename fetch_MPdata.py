#! /home/raymondw/.conda/envs/pymatgen/bin/python
from pymatgen import MPRester
import pandas as pd

# relevant properties:
prop_list = [ 
    "material_id", "pretty_formula", "band_gap", "unit_cell_formula", 
    "elements",  "nelements", "nsites", "spacegroup", "cif", "volume",
    "energy", "energy_per_atom", "e_above_hull", "formation_energy_per_atom",
    "density", "elasticity", "piezo",  "diel", "total_magnetization",
    "is_hubbard",  "hubbards",  "is_compatible", "oxide_type"
]

# fetch data
my_API_key = "gxTAyXSm2GvCdWer"
m = MPRester(api_key=my_API_key)

# have band gap
mp_data1 = m.query(criteria={"band_gap": { "$ne" : None}}, properties=prop_list)
# have band gap, no warning
mp_data2 = m.query(criteria={"band_gap": { "$ne" : None}, "warnings": []}, properties=prop_list)
# have band structure, no warning, 50171 instances
mp_data3 = m.query(criteria={"band_structure": { "$ne" : None}, "warnings": []}, properties=prop_list)


# convert to DataFrame
data1 = []
for entry in mp_data1:
    plist = []
    for prop, val in entry.items():
        plist.append(val)
    data1.append(plist)

data1 = pd.DataFrame(data1, index=None, columns=None)
data1.to_csv("./data/MPdata_bandgap.csv", sep=';', columns=None, header=prop_list, index=None)

data2 = []
for entry in mp_data2:
    plist = []
    for prop, val in entry.items():
        plist.append(val)
    data2.append(plist)

data2 = pd.DataFrame(data2, index=None, columns=None)
data2.to_csv("./data/MPdata_bandgap_no_warning.csv", sep=';', columns=None, header=prop_list, index=None)

# convert to DataFrame
data3 = []
for entry in mp_data3:
    plist = []
    for prop, val in entry.items():
        plist.append(val)
    data3.append(plist)

data3 = pd.DataFrame(data3, index=None, columns=None)
data3.to_csv("./data/MPdata_bandstruct_no_warning.csv", sep=';', columns=None, header=prop_list, index=None)


