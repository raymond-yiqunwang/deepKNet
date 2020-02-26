import pandas as pd
import XRD_simulator.xrd_simulator as xrd


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


def generate_point_cloud(data_raw):

    # number of points
#    npoint_list = []
    xrdcalc = xrd.XRDCalculator(wavelength='CuKa')
#    for _, irow in data_custom.iterrows():
#        struct = Structure.from_str(irow['cif'], fmt="cif")
#        npoint_list.append(xrdcalc.get_npoint(struct))
    
#    npoint_array = np.asarray(npoint_list)
#    print(" npoint: mean = {:.2f}, median = {:.2f}, standard deviation = {:.2f}, min = {:.2f}, max = {:.2f}"
#          .format(np.mean(npoint_array), np.median(npoint_array), np.std(npoint_array), np.min(npoint_array), np.max(npoint_array)))
#
    return data_raw


def main():
    # read customized data
    data = pd.read_csv("../custom_data_has_band.csv", sep=';', header=0, index_col=None)

    point_cloud = generate_point_cloud(data)

    # write customized data
    point_cloud.to_csv("../point_cloud.csv", sep=';', columns=None, header=None, index=None)


if __name__ == "__main__":
    main()


