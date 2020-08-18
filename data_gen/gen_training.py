import os
import sys
import ast
import shutil
import random
import numpy as np
import pandas as pd
from collections import defaultdict
import multiprocessing
from multiprocessing import Pool
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def gen_Xsys_data(data_custom):
    print("\ngenerate Xsys data..")
    print("size before customization:", data_custom.shape[0])
    
    # only take crystals in ICSD
    print('>> remove entries with no ICSD IDs')
    data_custom = data_custom[data_custom['icsd_ids'] != '[]']
    
    # only take no-warning entries
    print('>> remove entries with warnings')
    data_custom = data_custom[data_custom['warnings'] == '[]']

    print("statistics after customization:")
    show_statistics(data_custom)
    
    # output directory
    npoint = 343
    random_seed = 8
    out_dir = "./data_Xsys_P343/"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)
    properties = ['material_id', 'crystal_system']
    generate_train_valid_test(data_custom, out_dir, properties, npoint, random_seed)


def gen_tri_hex_cls_data(data_custom):
    print("\ngenerate tri_hex_cls data..")
    print("size before customization:", data_custom.shape[0])
    
    # only take trigonal and hexagonal materials
    print('only take trigonal and hexagonal materials')
    data_custom = data_custom[data_custom['crystal_system'].isin(['trigonal', 'hexagonal'])]

    # only take crystals in ICSD
#    print('>> remove entries with no ICSD IDs')
#    data_custom = data_custom[data_custom['icsd_ids'] != '[]']
    
    # only take no-warning entries
    print('>> remove entries with warnings')
    data_custom = data_custom[data_custom['warnings'] == '[]']
    
    print("statistics after customization:")
    show_statistics(data_custom)
    
    # output directory
    npoint = 27
    random_seed = 8
    out_dir = "./data_tri_hex_cls_P27/"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)
    properties = ['material_id', 'crystal_system']
    generate_train_valid_test(data_custom, out_dir, properties, npoint, random_seed)


def gen_MIC_data(data_custom):
    print("\ngenerate MIC data..")
    print("size before customization:", data_custom.shape[0])
    
    # only take materials with calculated band structures
    print('>> remove entries with no calculated band structures')
    data_custom = data_custom[data_custom['has_band_structure']]
    
    # only take crystals in ICSD
    print('>> remove entries with no ICSD IDs')
    data_custom = data_custom[data_custom['icsd_ids'] != '[]']
    
    # only take no-warning entries
    print('>> remove entries with warnings')
    data_custom = data_custom[data_custom['warnings'] == '[]']

    print("statistics after customization:")
    show_statistics(data_custom)
    
    # get rid of rare elements
    elem_dict = defaultdict(int)
    for entry in data_custom['elements']:
        for elem in ast.literal_eval(entry):
            elem_dict[elem] += 1
    rare_dict = {key: val for key, val in elem_dict.items() if val < 60}
    print('>> get rid of rare elements: ')
    print(rare_dict)
    rare_elements = set(rare_dict.keys())
    # drop entries containing rare elements
    drop_instance = []
    for idx, value in data_custom['elements'].iteritems():
        if rare_elements & set(ast.literal_eval(value)):
            drop_instance.append(idx)
    data_custom = data_custom.drop(drop_instance)
    
    # output directory
    npoint = 343
    random_seed = 8
    out_dir = "./data_MIC_P343/"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)
    properties = ['material_id', 'band_gap']
    generate_train_valid_test(data_custom, out_dir, properties, npoint, random_seed)


def gen_elasticity_data(data_custom):
    print("\ngenerate elasticity data..")
    print("size before customization:", data_custom.shape[0])
    
    # only take materials with elasticity data
    data_custom = data_custom[data_custom['elasticity'].notnull()]

    # only take crystals in ICSD
    print('>> remove entries with no ICSD IDs')
    data_custom = data_custom[data_custom['icsd_ids'] != '[]']
    
    # only take no-warning entries
    print('>> remove entries with warnings')
    data_custom = data_custom[data_custom['warnings'] == '[]']

    print("statistics after customization:")
    show_statistics(data_custom)
    
    # elasticity
    GKP = []
    for idx, irow in data_custom.iterrows():
        elasticity = ast.literal_eval(irow['elasticity'])
        shear_mod = elasticity['G_Voigt_Reuss_Hill']
        bulk_mod = elasticity['K_Voigt_Reuss_Hill']
        poisson_ratio = elasticity['poisson_ratio']
        GKP.append([shear_mod, bulk_mod, poisson_ratio])
    data_custom['elasticity_data'] = GKP

    # output directory
    npoint = 343
    random_seed = 8
    out_dir = "./data_elasticity_P343/"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)
    properties = ['material_id', 'elasticity_data']
    generate_train_valid_test(data_custom, out_dir, properties, npoint, random_seed)


def show_statistics(data):
    # size of database
    print('>> total number of materials: {:d}, number of properties: {:d}'\
            .format(data.shape[0], data.shape[1]))

    # compounds in ICSD
    ICSD_ids = data[data['icsd_ids'] != '[]']
    print('>> number of compounds that have ICSD IDs: {:d}'\
            .format(ICSD_ids.shape[0]))

    # space group
    sg_set = set()
    for sg in data['spacegroup']:
        sg_dict = ast.literal_eval(sg)
        sg_set.add(sg_dict['number'])
    print('>> number of unique space groups: {:d}'.format(len(sg_set)))

    # crystal system
    Xsys = data['crystal_system']
    print('>> crystal system value count:')
    print(Xsys.value_counts())

    # volume
    vol = data['volume']
    print('>> cell volume (A^3): mean = {:.2f}, median = {:.2f}, '
                'std = {:.2f}, min = {:.2f}, max = {:.2f}' \
                .format(vol.mean(), vol.median(), vol.std(),
                        vol.min(), vol.max()))

    # number of sites
    nsites = data['nsites']
    print('>> Number of sites: mean = {:.1f}, median = {:.1f}, '
                'std = {:.1f}, min = {:d}, max = {:d}' \
                .format(nsites.mean(), nsites.median(), nsites.std(), \
                        nsites.min(), nsites.max()))

    # elements
    elem_dict = defaultdict(int)
    for compound in data['elements']:
        for elem in ast.literal_eval(compound):
            elem_dict[elem] += 1
    min_key = min(elem_dict, key=elem_dict.get)
    max_key = max(elem_dict, key=elem_dict.get)
    print('>> Number of unique elements: {:d}, min: {}({:d}), max: {}({:d})' \
            .format(len(elem_dict), min_key, elem_dict[min_key], \
                                    max_key, elem_dict[max_key]))

    # energy per atom
    energy_atom = data['energy_per_atom']
    print('>> Energy per atom (eV): mean = {:.2f}, median = {:.2f}, '
                'std = {:.2f}, min = {:.2f}, max = {:.2f}' \
                .format(energy_atom.mean(), energy_atom.median(), energy_atom.std(), \
                        energy_atom.min(), energy_atom.max()))

    # formation energy per atom
    formation_atom = data['formation_energy_per_atom']
    print('>> Formation energy per atom (eV): mean = {:.2f}, median = {:.2f}, '
                'std = {:.2f}, min = {:.2f}, max = {:.2f}' \
                .format(formation_atom.mean(), formation_atom.median(),\
                        formation_atom.std(), formation_atom.min(), formation_atom.max()))

    # energy above hull
    e_above_hull = data['e_above_hull']
    print('>> Energy above hull (eV): mean = {:.2f}, median = {:.2f}, '
                'std = {:.2f}, min = {:.2f}, max = {:.2f}' \
                .format(e_above_hull.mean(), e_above_hull.median(),\
                     e_above_hull.std(), e_above_hull.min(), e_above_hull.max()))
    print('>> Energy above hull (eV) < 10meV: {:d}'.format( \
          e_above_hull[e_above_hull < 0.01].size))

    # band gap
    gap_threshold = 1E-6
    metals = data[data['band_gap'] <= gap_threshold]['band_gap']
    insulators = data[data['band_gap'] > gap_threshold]['band_gap']
    print('>> Number of metals: {:d}, number of insulators: {:d}' \
                .format(metals.size, insulators.size))
    print('     band gap of all dataset: mean = {:.2f}, median = {:.2f}, '
                 'std = {:.2f}, min = {:.5f}, max = {:.2f}' \
                 .format(data['band_gap'].mean(), data['band_gap'].median(),\
                         data['band_gap'].std(), data['band_gap'].min(), data['band_gap'].max()))
    print('     band gap of insulators: mean = {:.2f}, median = {:.2f}, '
                 'std = {:.2f}, min = {:.5f}, max = {:.2f}' \
                 .format(insulators.mean(), insulators.median(),\
                         insulators.std(), insulators.min(), insulators.max()))

    # warnings
    no_warnings = data[data['warnings'] == '[]']
    print('>> Number of entries with no warnings: {:d}'.format(no_warnings.shape[0]))

    # elasticity
    elasticity = data['elasticity'].dropna()
    print('>> Number of elasticity data: {:d}'.format(elasticity.size))
    Gs, Ks, Ps = [], [], []
    for imat in elasticity:
        shear_mod = ast.literal_eval(imat)['G_Voigt_Reuss_Hill']
        #if shear_mod > -1E-6: Gs.append(shear_mod)
        Gs.append(shear_mod)
        bulk_mod = ast.literal_eval(imat)['K_Voigt_Reuss_Hill']
        #if bulk_mod > -1E-6: Ks.append(bulk_mod)
        Ks.append(bulk_mod)
        poisson_ratio = ast.literal_eval(imat)['poisson_ratio']
        #if poisson_ratio > -1E-6: Ps.append(poisson_ratio)
        Ps.append(poisson_ratio)
    print('Shear modulus > 100: {:d}'.format((np.array(Gs)>100).sum()))
    print('Bulk modulus > 200: {:d}'.format((np.array(Ks)>200).sum()))
    print('Shear modulus: mean = {:.2f}, median = {:.2f}, std = {:.2f}, '
                         'min = {:.5f}, max = {:.2f}' \
           .format(np.mean(Gs), np.median(Gs), np.std(Gs), np.min(Gs), np.max(Gs)))
    print('Bulk modulus: mean = {:.2f}, median = {:.2f}, std = {:.2f}, '
                         'min = {:.5f}, max = {:.2f}' \
           .format(np.mean(Ks), np.median(Ks), np.std(Ks), np.min(Ks), np.max(Ks)))
    print('Poisson ratio: mean = {:.2f}, median = {:.2f}, std = {:.2f}, '
                         'min = {:.5f}, max = {:.2f}' \
           .format(np.mean(Ps), np.median(Ps), np.std(Ps), np.min(Ps), np.max(Ps)))


def check_crystal_system(data_input, sym_thresh):
    drop_list =[]
    cnt = 0
    for idx, irow in data_input.iterrows():
        cnt += 1
        if cnt % 2000 == 0:
            print('this worker finished processing {} materials..'\
                   .format(cnt), flush=True)
        material_id = irow['material_id']
        with open("./MPdata_all/"+material_id+".cif") as f:
            cif_file = f.read()
        cif_struct = Structure.from_str(cif_file, fmt="cif")
        try:
            sga = SpacegroupAnalyzer(cif_struct, symprec=sym_thresh)
            assert(sga.get_crystal_system() == irow['crystal_system'])
        except:
            print('0.1 failed', irow['material_id'], 'added to drop list')
            print('sga:', sga.get_crystal_system(), ', MP:', irow['crystal_system'])
            drop_list.append(idx)
            continue
        # get conventional cell
        conventional_struct = sga.get_conventional_standard_structure()
        latt = conventional_struct.lattice.matrix
        a, b, c = latt[0], latt[1], latt[2]
        lena = np.linalg.norm(a)
        lenb = np.linalg.norm(b)
        lenc = np.linalg.norm(c)
        theta_ab = np.arccos(np.dot(a,b)/(lena*lenb))/np.pi*180
        theta_bc = np.arccos(np.dot(b,c)/(lenb*lenc))/np.pi*180
        theta_ac = np.arccos(np.dot(a,c)/(lena*lenc))/np.pi*180
        if irow['crystal_system'] == 'hexagonal' or \
            irow['crystal_system'] == 'trigonal':
            # a==b!=c, ab==120, bc==ac==90
            try:
                assert(abs(lena - lenb) < 1E-2)
                assert(abs(theta_ab-120) < 1E-1)
                assert(abs(theta_bc-90) < 1E-1)
                assert(abs(theta_ac-90) < 1E-1)
            except:
                print(irow['material_id'])
                print(irow['crystal_system'])
                print(lena, lenb, lenc, theta_ab, theta_bc, theta_ac)
        elif irow['crystal_system'] == 'cubic':
            # a==b==c, ab==bc==ac==90
            try:
                assert(abs(lena - lenb) < 1E-2)
                assert(abs(lenb - lenc) < 1E-2)
                assert(abs(lena - lenc) < 1E-2)
                assert(abs(theta_ab-90) < 1E-1)
                assert(abs(theta_bc-90) < 1E-1)
                assert(abs(theta_ac-90) < 1E-1)
            except:
                print(irow['material_id'])
                print('cubic')
                print(lena, lenb, lenc, theta_ab, theta_bc, theta_ac)
        elif irow['crystal_system'] == 'tetragonal':
            # a==b!=c, ab==bc==ac==90
            try:
                assert(abs(lena - lenb) < 1E-2)
                assert(abs(theta_ab-90) < 1E-1)
                assert(abs(theta_bc-90) < 1E-1)
                assert(abs(theta_ac-90) < 1E-1)
            except:
                print(irow['material_id'])
                print('tetragonal')
                print(lena, lenb, lenc, theta_ab, theta_bc, theta_ac)
        elif irow['crystal_system'] == 'orthorhombic':
            # a!=b!=c, ab==bc==ac==90
            try:
                assert(abs(theta_ab-90) < 1E-1)
                assert(abs(theta_bc-90) < 1E-1)
                assert(abs(theta_ac-90) < 1E-1)
            except:
                print(irow['material_id'])
                print('orthorhombic')
                print(lena, lenb, lenc, theta_ab, theta_bc, theta_ac)
        elif irow['crystal_system'] == 'monoclinic':
            # a!=b!=c, ab==bc==90, ac!=90
            try:
                assert(abs(theta_ab-90) < 1E-1)
                assert(abs(theta_bc-90) < 1E-1)
            except:
                print(irow['material_id'])
                print('monoclinic')
                print(lena, lenb, lenc, theta_ab, theta_bc, theta_ac)
        else:
            try:
                assert(irow['crystal_system'] == 'triclinic')
            except:
                print(irow['material_id'])
                print('UNK --', irow['crystal_system'])
                print(lena, lenb, lenc, theta_ab, theta_bc, theta_ac)

    print('number of entries to drop in this batch:', len(drop_list))
    data_out = data_input.drop(drop_list)
    return data_out


def generate_train_valid_test(MPdata, out_dir, properties, npoint, random_seed):
    # id_prop file
    id_prop_all = MPdata[properties]
    print('size of all data:', id_prop_all.shape[0], 'npoint:', npoint)
    # random shuffle with seed
    id_prop_all = id_prop_all.sample(frac=1, random_state=random_seed)
    # split ratio
    train_ratio, valid_ratio = 0.7, 0.15
    # train
    train_dir = os.path.join(out_dir, "train/")
    os.mkdir(train_dir)
    train_file = os.path.join(train_dir, "id_prop.csv")
    train_split = int(np.floor(id_prop_all.shape[0] * train_ratio))
    train_data = id_prop_all.iloc[:train_split]
    train_data.to_csv(train_file, sep=',', header=properties, index=False, mode='w')
    # valid
    valid_dir = os.path.join(out_dir, "valid/")
    os.mkdir(valid_dir)
    valid_file = os.path.join(valid_dir, "id_prop.csv")
    valid_split = train_split + int(np.floor(id_prop_all.shape[0] * valid_ratio))
    valid_data = id_prop_all.iloc[train_split:valid_split]
    valid_data.to_csv(valid_file, sep=',', header=properties, index=False, mode='w')
    # test
    test_dir = os.path.join(out_dir, "test/")
    os.mkdir(test_dir)
    test_file = os.path.join(test_dir, "id_prop.csv")
    test_data = id_prop_all.iloc[valid_split:]
    test_data.to_csv(test_file, sep=',', header=properties, index=False, mode='w')
    # write to file
    for (save_data, save_dir) in [(train_data, train_dir), (valid_data, valid_dir), \
                             (test_data, test_dir)]:
        for mat_id in save_data['material_id']:
            hkl_feat = np.load("./MPdata_all/"+mat_id+".npy")
            # select hkl points
            if npoint == 3:
                conditions = np.where((np.sum(hkl_feat[:,:-1], axis=1)==1) & \
                                      (np.min(hkl_feat[:,:-1], axis=1)>-1))
                selected_hkl_feat = hkl_feat[conditions]
            elif npoint == 27:
                conditions = np.where((np.max(hkl_feat[:,:-1], axis=1)<1.1) & \
                                      (np.min(hkl_feat[:,:-1], axis=1)>-1.1))
                selected_hkl_feat = hkl_feat[conditions]
            elif npoint == 125:
                conditions = np.where((np.max(hkl_feat[:,:-1], axis=1)<2.1) & \
                                      (np.min(hkl_feat[:,:-1], axis=1)>-2.1))
                selected_hkl_feat = hkl_feat[conditions]
            elif npoint == 343:
                conditions = np.where((np.max(hkl_feat[:,:-1], axis=1)<3.1) & \
                                      (np.min(hkl_feat[:,:-1], axis=1)>-3.1))
                selected_hkl_feat = hkl_feat[conditions]
            elif npoint == 729:
                conditions = np.where((np.max(hkl_feat[:,:-1], axis=1)<4.1) & \
                                      (np.min(hkl_feat[:,:-1], axis=1)>-4.1))
                selected_hkl_feat = hkl_feat[conditions]
            else:
                raise NotImplementedError

            # convert to Cartesion
            recip_latt = np.load("./MPdata_all/"+mat_id+"_basis.npy")
            recip_pos = np.dot(selected_hkl_feat[:,:-1], recip_latt)
            # CuKa by default
            max_r = 2 / 1.54184
            recip_pos /= max_r
            assert(np.amax(recip_pos) <= 1.0)
            assert(np.amin(recip_pos) >= -1.0)
            # normalize diffraction intensity
            intensity = np.log(1+selected_hkl_feat[:,-1]) / 3
            intensity = intensity.reshape(-1, 1)
            assert(np.amax(intensity) <= 1.3)
            assert(np.amin(intensity) >= 0.)
            # generate point cloud and write to file
            point_cloud = np.concatenate((recip_pos, intensity), axis=1)
            np.save(os.path.join(save_dir, mat_id), point_cloud)


def main():
    # read all MPdata
    MPdata_all = pd.read_csv("./MPdata_all/MPdata_all.csv", sep=';', header=0, index_col=None)

    # show statistics of original data
    if True:
        print('statistics of original data')
        show_statistics(data=MPdata_all)

    # check crystal system match
    if True:
        print('\nchecking crystal system match on all {} data'.format(MPdata_all.shape[0]))
        sym_thresh = 0.1
        nworkers = multiprocessing.cpu_count()
        pool = Pool(processes=nworkers)
        df_split = np.array_split(MPdata_all, nworkers)
        args = [(data, sym_thresh) for data in df_split]
        MPdata_all = pd.concat(pool.starmap(check_crystal_system, args), axis=0)
        pool.close()
        pool.join()
        print('size of data with matched crystal system:', MPdata_all.shape[0])

    # crystal system classification
    if False:
        gen_Xsys_data(MPdata_all)

    # trigonal-hexagonal classification
    if False:
        gen_tri_hex_cls_data(MPdata_all)

    # metal-insulator classification
    if False:
        gen_MIC_data(MPdata_all)

    # elasticity classification
    if True:
        gen_elasticity_data(MPdata_all)


if __name__ == "__main__":
    main()


