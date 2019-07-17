#! /home/raymondw/.conda/envs/tf-cpu/bin/python
from pymatgen import MPRester
import pandas as pd
import os


# relevant properties:
prop_list = [ 
    "material_id", "pretty_formula", "band_gap", 
    "elements",  "nelements", "nsites", "spacegroup", "cif", "volume",
    "energy", "energy_per_atom", "e_above_hull", "formation_energy_per_atom",
    "density", "elasticity", "piezo",  "diel", "total_magnetization"
]

# fetch data
my_API_key = "gxTAyXSm2GvCdWer"
m = MPRester(api_key=my_API_key)

# have band gap, 120612 instances
#mp_data = m.query(criteria={"band_gap": { "$ne" : None}}, properties=prop_list)
# have band gap, no warning, 97585 instances
#mp_data = m.query(criteria={"band_gap": { "$ne" : None}, "warnings": []}, properties=prop_list)
# have band structure, no warning, 50171 instances
#mp_data = m.query(criteria={"band_structure": { "$ne" : None}, "warnings": []}, properties=prop_list)
mp_data = m.query(criteria={"band_structure": { "$ne" : None}, "nsites": { "$lt" : 10 }, "volume": { "$lt" : 100 },"warnings": []}, properties=prop_list)

index = 0
data = []
for entry in mp_data:
    plist = [index]
    for _, val in entry.items():
        plist.append(val)
    data.append(plist)
    index += 1

out_dir = "../data/"
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

prop_list.insert(0, "my_id")
data = pd.DataFrame(data, index=None, columns=None)
data.to_csv(out_dir+"MPdata.csv", sep=';', columns=None, header=prop_list, index=None)


