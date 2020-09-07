import numpy as np
import ast
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

MPdata_all = pd.read_csv("/home/raymondw/Research/deepKNet/data_gen/MPdata_all/MPdata_all.csv", \
                         sep=';', header=0, index_col=None)

# metal-semiconductor classification
if False:
    MIC_data = pd.read_csv("MIC_run2_misclass.csv", header=0, index_col=None)
    assert(MIC_data['id'].nunique() == MIC_data.shape[0])
    MIC_dict = {0: 'metal', 1: 'insulator'}
    results = []
    for _, irow in MIC_data.iterrows():
        mat_id = irow['id']
        pretty_formula = MPdata_all.loc[MPdata_all['material_id'] == mat_id]\
                                       ['pretty_formula'].values[0]
        pred_label = MIC_dict[irow['pred']]
        pred_score = irow['pred_score']
        true_label = MIC_dict[irow['true']]
        true_score = irow['true_score']
        band_gap = MPdata_all.loc[MPdata_all['material_id'] == mat_id]\
                                 ['band_gap'].values[0]
        out = [mat_id, pretty_formula, pred_label, pred_score, true_label, true_score, band_gap]
        results.append(out)

    results = pd.DataFrame(results)
    header_out = ['material_id', 'pretty_formula', 'pred_label', 'pred_score',
                                        'true_label', 'true_score', 'band_gap']
    results.to_csv("./metal-semiconductor_misclass.csv", header=header_out, index=False)


# crystal family classification
if False:
    Xsys_data = pd.read_csv("Xsys_run37_misclass.csv", header=0, index_col=None)
    assert(Xsys_data['id'].nunique() == Xsys_data.shape[0])
    Xsys_dict = {0: 'cubic', 1: 'orthorhombic', 2: 'tetragonal',
                 3: 'monoclinic', 4: 'triclinic', 5: 'trigonal/hexagonal'}
    results = []
    for _, irow in Xsys_data.iterrows():
        mat_id = irow['id']
        pretty_formula = MPdata_all.loc[MPdata_all['material_id'] == mat_id]\
                                       ['pretty_formula'].values[0]
        pred_label = Xsys_dict[irow['pred']]
        pred_score = irow['pred_score']
        true_label = Xsys_dict[irow['true']]
        true_score = irow['true_score']
        with open("/home/raymondw/Research/deepKNet/data_gen/MPdata_all/"+mat_id+".cif") as f:
            cif_file = f.read()
        cif_struct = Structure.from_str(cif_file, fmt="cif")
        sga = SpacegroupAnalyzer(cif_struct, symprec=0.1)
        conventional_struct = sga.get_conventional_standard_structure()
        basis = conventional_struct.lattice.matrix
        a, b, c = basis[0], basis[1], basis[2]                                 
        lena = np.linalg.norm(a)
        lenb = np.linalg.norm(b)
        lenc = np.linalg.norm(c)
        alpha = np.arccos(np.dot(a,b)/(lena*lenb))/np.pi*180
        beta = np.arccos(np.dot(b,c)/(lenb*lenc))/np.pi*180
        gamma = np.arccos(np.dot(a,c)/(lena*lenc))/np.pi*180
        out = [mat_id, pretty_formula, lena, lenb, lenc, alpha, beta, gamma, 
               pred_label, pred_score, true_label, true_score]
        results.append(out)
    
    results = pd.DataFrame(results)
    header_out = ['material_id', 'pretty_formula', 'lena', 'lenb', 'lenc', 
                  'alpha', 'beta', 'gamma', 
                  'pred_label', 'pred_score', 'true_label', 'true_score']
    results.to_csv("./crystal_family_misclass.csv", header=header_out, index=False)

# bulk modulus classification
if True:
    bulk_data = pd.read_csv("bulk_run2_misclass.csv", header=0, index_col=None)
    assert(bulk_data['id'].nunique() == bulk_data.shape[0])
    bulk_dict = {0: 'soft', 1: 'hard'}
    results = []
    for _, irow in bulk_data.iterrows():
        mat_id = irow['id']
        pretty_formula = MPdata_all.loc[MPdata_all['material_id'] == mat_id]\
                                       ['pretty_formula'].values[0]
        pred_label = bulk_dict[irow['pred']]
        pred_score = irow['pred_score']
        true_label = bulk_dict[irow['true']]
        true_score = irow['true_score']
        elasticity = MPdata_all.loc[MPdata_all['material_id'] == mat_id]\
                                   ['elasticity'].values[0]
        elasticity = ast.literal_eval(elasticity)
        bulk_modulus = elasticity['K_Voigt_Reuss_Hill']
        out = [mat_id, pretty_formula, pred_label, pred_score, true_label, true_score, bulk_modulus]
        results.append(out)

    results = pd.DataFrame(results)
    header_out = ['material_id', 'pretty_formula', 'pred_label', 'pred_score',
                                    'true_label', 'true_score', 'bulk_modulus']
    results.to_csv("./bulk_modulus_misclass.csv", header=header_out, index=False)
