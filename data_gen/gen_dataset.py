import os
import sys
import ast
import shutil
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import multiprocessing
from multiprocessing import Pool
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def gen_metallicity_dataset(data_custom, root_dir, dtype, ctype):
    print(f"\ngenerate {dtype} {ctype} metallicity data..")

    # only take materials with calculated band structures
    print('>> remove entries without calculated band structures')
    data_custom = data_custom[data_custom['has_band_structure']]
    
    # only take no-warning entries
    print('>> remove entries with warnings')
    data_custom = data_custom[data_custom['warnings'] == '[]']

    # filter out proper band gap values
    metals = data_custom[data_custom['band_gap'] < 1E-6]
    insulators = data_custom[data_custom['band_gap'] > 0.05]
    print('number of metals: {}, number of insulators: {}, max gap: {}' \
          .format(metals.shape[0], insulators.shape[0], max(insulators['band_gap'])))
    data_custom = pd.concat((metals, insulators), axis=0)

    # show statistics
    check_properties(data=data_custom)

    # check diffraction patterns
    check_diffraction_patterns_parallel(data=data_custom, dtype=dtype, ctype=ctype)

    print('size of dataset:', data_custom.shape[0])
    # output directory
    out_dir = os.path.join(root_dir, f"data_metallicity_{dtype}_{ctype}")
    if os.path.exists(out_dir):
        print('\ndeleting original data folder..')
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)
    # save to datasets dir
    metallicity_data = data_custom[['material_id', 'band_gap']]
    if dtype == 'ND':
        drop_list = [] # compounds containing Ac and Pu do not have simulated ND patterns
        for idx, props in metallicity_data.iterrows():
            mat_id = props['material_id']
            src = f'./MPdata_all/ND_data/'+mat_id+f"_ND_{ctype}.npy"
            if not os.path.isfile(src):
                drop_list.append(idx)
        metallicity_data = metallicity_data.drop(drop_list)
    id_prop_file = os.path.join(out_dir, 'id_prop.csv')
    metallicity_data.to_csv(id_prop_file, sep=',', header=metallicity_data.columns, index=False, mode='w')
    for mat_id in metallicity_data['material_id']:
        src = f'./MPdata_all/{dtype}_data/'+mat_id+f"_{dtype}_{ctype}.npy"
        assert os.path.isfile(src)
        shutil.copy(src, out_dir)


def gen_stability_dataset(data_custom, root_dir, dtype, ctype):
    print(f"\ngenerate {dtype} {ctype} stability data..")

    # only take no-warning entries
    print('>> remove entries with warnings')
    data_custom = data_custom[data_custom['warnings'] == '[]']

    # show statistics
    check_properties(data=data_custom)
    
    # check diffraction patterns
    check_diffraction_patterns_parallel(data=data_custom, dtype=dtype, ctype=ctype)

    print('size of dataset:', data_custom.shape[0])
    # output directory
    out_dir = os.path.join(root_dir, f"data_stability_{dtype}_{ctype}")
    if os.path.exists(out_dir):
        print('\ndeleting original data folder..')
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)
    # save to datasets dir
    stability_data = data_custom[['material_id', 'e_above_hull']]
    if dtype == 'ND':
        drop_list = [] # compounds containing Ac and Pu do not have simulated ND patterns
        for idx, props in stability_data.iterrows():
            mat_id = props['material_id']
            src = f'./MPdata_all/ND_data/'+mat_id+f"_ND_{ctype}.npy"
            if not os.path.isfile(src):
                drop_list.append(idx)
        stability_data = stability_data.drop(drop_list)
    id_prop_file = os.path.join(out_dir, 'id_prop.csv')
    stability_data.to_csv(id_prop_file, sep=',', header=stability_data.columns,
                          index=False, mode='w')
    for mat_id in stability_data['material_id']:
        src = f'./MPdata_all/{dtype}_data/'+mat_id+f"_{dtype}_{ctype}.npy"
        assert os.path.isfile(src)
        shutil.copy(src, out_dir)


def gen_elasticity_dataset(data_custom, root_dir, dtype, ctype):
    print(f"\ngenerate {dtype} {ctype} elasticity data..")

    # only take materials with elasticity data
    data_custom = data_custom[data_custom['elasticity'].notnull()]
    
    # only take no-warning entries
    print('>> remove entries with warnings')
    data_custom = data_custom[data_custom['warnings'] == '[]']

    # show statistics
    check_properties(data=data_custom)

    # check diffraction patterns
    check_diffraction_patterns_parallel(data=data_custom, dtype=dtype, ctype=ctype)

    # elasticity
    shear_mod = []
    bulk_mod = []
    poisson_ratio = []
    for idx, irow in data_custom.iterrows():
        elasticity = ast.literal_eval(irow['elasticity'])
        shear_mod.append(elasticity['G_Voigt_Reuss_Hill'])
        bulk_mod.append(elasticity['K_Voigt_Reuss_Hill'])
        poisson_ratio.append(elasticity['poisson_ratio'])
    data_custom['shear_modulus'] = shear_mod
    data_custom['bulk_modulus'] = bulk_mod
    data_custom['poisson_ratio'] = poisson_ratio

    print('size of dataset:', data_custom.shape[0])
    # output directory
    out_dir = os.path.join(root_dir, f"data_elasticity_{dtype}_{ctype}")
    if os.path.exists(out_dir):
        print('\ndeleting original data folder..')
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)

    # save to datasets dir
    elasticity_data = data_custom[['material_id', 'shear_modulus', 'bulk_modulus', 'poisson_ratio']]
    if dtype == 'ND':
        drop_list = [] # compounds containing Ac and Pu do not have simulated ND patterns
        for idx, props in elasticity_data.iterrows():
            mat_id = props['material_id']
            src = f'./MPdata_all/ND_data/'+mat_id+f"_ND_{ctype}.npy"
            if not os.path.isfile(src):
                drop_list.append(idx)
        elasticity_data = elasticity_data.drop(drop_list)
    id_prop_file = os.path.join(out_dir, 'id_prop.csv')
    elasticity_data.to_csv(id_prop_file, sep=',', header=elasticity_data.columns, index=False, mode='w')
    for mat_id in elasticity_data['material_id']:
        src = f'./MPdata_all/{dtype}_data/'+mat_id+f"_{dtype}_{ctype}.npy"
        assert os.path.isfile(src)
        shutil.copy(src, out_dir)


def check_properties(data):
    # size of database
    print('>> total number of materials: {:d}, number of properties: {:d}'\
            .format(data.shape[0], data.shape[1]), '\n')
    
    # properties
    print('>> properties:')
    print(data.columns.tolist(), '\n')

    # compounds in ICSD
    ICSD_ids = data[data['icsd_ids'] != '[]']
    print('>> number of compounds that have ICSD IDs: {:d}'\
            .format(ICSD_ids.shape[0]), '\n')

    # space group
    sg_set = set()
    for sg in data['spacegroup']:
        sg_dict = ast.literal_eval(sg)
        sg_set.add(sg_dict['number'])
    print('>> number of unique space groups: {:d}'.format(len(sg_set)), '\n')
    
    # crystal system
    Xsys = data['crystal_system']
    print('>> crystal system value count:')
    print(Xsys.value_counts(), '\n')

    # volume
    vol = data['volume']
    print('>> cell volume (A^3): mean = {:.2f}, median = {:.2f}, '
                'std = {:.2f}, min = {:.2f}, max = {:.2f}' \
                .format(vol.mean(), vol.median(), vol.std(),
                        vol.min(), vol.max()), '\n')

    # number of sites
    nsites = data['nsites']
    print('>> Number of sites: mean = {:.1f}, median = {:.1f}, '
                'std = {:.1f}, min = {:d}, max = {:d}' \
                .format(nsites.mean(), nsites.median(), nsites.std(), \
                        nsites.min(), nsites.max()), '\n')

    # elements
    elem_dict = defaultdict(int)
    for compound in data['elements']:
        for elem in ast.literal_eval(compound):
            elem_dict[elem] += 1
    min_key = min(elem_dict, key=elem_dict.get)
    max_key = max(elem_dict, key=elem_dict.get)
    print('>> Number of unique elements: {:d}, min: {}({:d}), max: {}({:d})' \
            .format(len(elem_dict), min_key, elem_dict[min_key], \
                                    max_key, elem_dict[max_key]), '\n')

    # energy per atom
    energy_atom = data['energy_per_atom']
    print('>> Energy per atom (eV): mean = {:.2f}, median = {:.2f}, '
                'std = {:.2f}, min = {:.2f}, max = {:.2f}' \
                .format(energy_atom.mean(), energy_atom.median(), energy_atom.std(), \
                        energy_atom.min(), energy_atom.max()), '\n')

    # formation energy per atom
    formation_atom = data['formation_energy_per_atom']
    print('>> Formation energy per atom (eV): mean = {:.2f}, median = {:.2f}, '
                'std = {:.2f}, min = {:.2f}, max = {:.2f}' \
                .format(formation_atom.mean(), formation_atom.median(),\
                        formation_atom.std(), formation_atom.min(), formation_atom.max()), '\n')

    # energy above hull
    e_above_hull = data['e_above_hull']
    print('>> Energy above hull (eV): mean = {:.2f}, median = {:.2f}, '
                'std = {:.2f}, min = {:.2f}, max = {:.2f}' \
                .format(e_above_hull.mean(), e_above_hull.median(),\
                     e_above_hull.std(), e_above_hull.min(), e_above_hull.max()))
    print('>> Energy above hull (eV) < 10 meV: {:d}'.format( \
          e_above_hull[e_above_hull < 0.01].size), '\n')

    # calculated band structure available
    has_band_structure = data[data['has_band_structure'] == True]
    print('>> Number of materials with calculated band structure: {}\n'.format(has_band_structure.shape[0]))

    # band gap
    gap_threshold = 1E-4
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
                         insulators.std(), insulators.min(), insulators.max()), '\n')

    # warnings
    no_warnings = data[data['warnings'] == '[]']
    print('>> Number of entries with no warnings: {:d}'.format(no_warnings.shape[0]), '\n')

    # elasticity
    elasticity = data['elasticity'].dropna()
    print('>> Number of elasticity data: {:d}'.format(elasticity.size))
    Gs, Ks, Ps = [], [], []
    for imat in elasticity:
        shear_mod = ast.literal_eval(imat)['G_Voigt_Reuss_Hill']
        Gs.append(shear_mod)
        bulk_mod = ast.literal_eval(imat)['K_Voigt_Reuss_Hill']
        Ks.append(bulk_mod)
        poisson_ratio = ast.literal_eval(imat)['poisson_ratio']
        Ps.append(poisson_ratio)
    print('Shear modulus > 50: {:d}'.format((np.array(Gs)>50).sum()))
    print('Bulk modulus > 100: {:d}'.format((np.array(Ks)>100).sum()))
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
    process_name = multiprocessing.current_process().name
    drop_list =[]
    for idx, irow in tqdm(data_input.iterrows()):
        cif_struct = Structure.from_file("./MPdata_all/MP_cifs/"+irow['material_id']+".cif")
        sga = SpacegroupAnalyzer(cif_struct, symprec=sym_thresh)
        sgb = ast.literal_eval(irow['spacegroup'])
        try:
            assert sga.get_crystal_system() == irow['crystal_system']
            assert int(sga.get_space_group_number()) == int(sgb['number'])
        except:
            print(sym_thresh, 'failed on', irow['material_id'], 'added to drop list')
            print('sga XSys: {}, MP XSys: {}, sga SG: {}, MP SG: {}'
                  .format(sga.get_crystal_system(), irow['crystal_system'], 
                          int(sga.get_space_group_number()), int(sgb['number'])))
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
                assert (abs(lena - lenb) < 1E-2)
                assert (abs(theta_ab-120) < 1E-1)
                assert (abs(theta_bc-90) < 1E-1)
                assert (abs(theta_ac-90) < 1E-1)
            except:
                print(irow['material_id'])
                print(irow['crystal_system'])
                print(lena, lenb, lenc, theta_ab, theta_bc, theta_ac)
        elif irow['crystal_system'] == 'cubic':
            # a==b==c, ab==bc==ac==90
            try:
                assert (abs(lena - lenb) < 1E-2)
                assert (abs(lenb - lenc) < 1E-2)
                assert (abs(lena - lenc) < 1E-2)
                assert (abs(theta_ab-90) < 1E-1)
                assert (abs(theta_bc-90) < 1E-1)
                assert (abs(theta_ac-90) < 1E-1)
            except:
                print(irow['material_id'])
                print('cubic')
                print(lena, lenb, lenc, theta_ab, theta_bc, theta_ac)
        elif irow['crystal_system'] == 'tetragonal':
            # a==b!=c, ab==bc==ac==90
            try:
                assert (abs(lena - lenb) < 1E-2)
                assert (abs(theta_ab-90) < 1E-1)
                assert (abs(theta_bc-90) < 1E-1)
                assert (abs(theta_ac-90) < 1E-1)
            except:
                print(irow['material_id'])
                print('tetragonal')
                print(lena, lenb, lenc, theta_ab, theta_bc, theta_ac)
        elif irow['crystal_system'] == 'orthorhombic':
            # a!=b!=c, ab==bc==ac==90
            try:
                assert (abs(theta_ab-90) < 1E-1)
                assert (abs(theta_bc-90) < 1E-1)
                assert (abs(theta_ac-90) < 1E-1)
            except:
                print(irow['material_id'])
                print('orthorhombic')
                print(lena, lenb, lenc, theta_ab, theta_bc, theta_ac)
        elif irow['crystal_system'] == 'monoclinic':
            # a!=b!=c, ab==bc==90, ac!=90
            try:
                assert (abs(theta_ab-90) < 1E-1)
                assert (abs(theta_bc-90) < 1E-1)
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

    print('number of entries to drop in process {}: {}'.format(process_name, len(drop_list)), flush=True)
    data_out = data_input.drop(drop_list)
    return data_out


def check_diffraction_patterns_serial(data, dtype, ctype):
    process_name = multiprocessing.current_process().name
    max_r, max_I = 0, 0
    material_ids = data['material_id']
    for mat_id in tqdm(material_ids):
        feat_file = "./MPdata_all/{0}_data/{1}_{2}_{3}.npy".format(dtype, mat_id, dtype, ctype)
        if not os.path.isfile(feat_file):
            assert dtype == 'ND'
            continue
        features = np.load(feat_file) # (npoint, 10)
        hkl = features[:,:3]
        xyz = features[:,3:6]
        I_hkl = features[:,-1]
        r = np.linalg.norm(xyz, axis=1)
        max_r = max(max(r), max_r)
        max_I = max(max(I_hkl), max_I)
    return max_r, max_I


def check_diffraction_patterns_parallel(data, dtype, ctype):
    print('\nshow diffraction pattern info')
    # take samples
    nworkers = multiprocessing.cpu_count()
    pool = Pool(processes=nworkers)
    df_split = np.array_split(data, nworkers)
    args = [(data, dtype, ctype) for data in df_split]
    patterns = pool.starmap_async(check_diffraction_patterns_serial, args)
    pool.close()
    pool.join()
    patterns = np.array(patterns.get())
    # post-processing
    max_r = max(patterns[:,0])
    max_I = max(patterns[:,1])
    print('\n {0} {1} pattern max_r: {2}, max_I: {3}'.format(dtype, ctype, max_r, max_I))


if __name__ == "__main__":
    # read all MPdata
    MPdata_all = pd.read_csv("./MPdata_all/MPdata_all.csv", sep=';', header=0, index_col=None)
    
    # show statistics of original data
    if False:
        print('show statistics of original data')
        check_properties(data=MPdata_all)
    else:
        print('size of original data:', MPdata_all.shape[0])

    # check crystal symmetry 
    if False:
        print('\nchecking crystal symmetry match on all {} data'.format(MPdata_all.shape[0]))
        sym_thresh = 0.1
        nworkers = multiprocessing.cpu_count()
        pool_Xsys = Pool(processes=nworkers)
        df_split = np.array_split(MPdata_all, nworkers)
        args = [(data, sym_thresh) for data in df_split]
        MPdata_all = pd.concat(pool_Xsys.starmap(check_crystal_system, args), axis=0)
        pool_Xsys.close()
        pool_Xsys.join()
        print('size of data with matched crystal symmetry:', MPdata_all.shape[0])
    else:
        print('\nskip checking..')
        drop_ids = ['mp-18828', 'mp-12843', 'mp-20811']
        MPdata_all = MPdata_all[~MPdata_all['material_id'].isin(drop_ids)]
        print('size of data with matched crystal system:', MPdata_all.shape[0])

    # make dataset directory
    root_dir = './datasets/'
    if not os.path.exists(root_dir):
        print("\n{} does not exist, making dir..".format(root_dir))
        os.mkdir(root_dir)

    # metallicity
    gen_metallicity_dataset(MPdata_all, root_dir, dtype='XRD', ctype='primitive')
    gen_metallicity_dataset(MPdata_all, root_dir, dtype='XRD', ctype='conventional')
    gen_metallicity_dataset(MPdata_all, root_dir, dtype='ND', ctype='primitive')
    gen_metallicity_dataset(MPdata_all, root_dir, dtype='ND', ctype='conventional')

    # stability
    gen_stability_dataset(MPdata_all, root_dir, dtype='XRD', ctype='primitive')
    gen_stability_dataset(MPdata_all, root_dir, dtype='ND', ctype='primitive')
    
    # elasticity
    gen_elasticity_dataset(MPdata_all, root_dir, dtype='XRD', ctype='primitive')
    gen_elasticity_dataset(MPdata_all, root_dir, dtype='ND', ctype='primitive')


