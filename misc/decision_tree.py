from pymatgen import MPRester
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.structure import Structure
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
import sklearn.metrics as metrics
from sklearn.metrics import plot_precision_recall_curve
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import matplotlib.pyplot as plt

def fetch_data():
    my_API_key = "gxTAyXSm2GvCdWer"
    m = MPRester(api_key=my_API_key)

    props = ["material_id", "crystal_system", "band_gap", "icsd_ids", "warnings", "volume", "cif"]
    query = m.query(criteria={"has": "bandstructure"},
                    properties=props)

    MPdata = pd.DataFrame(entry.values() for entry in query)
    MPdata.to_csv('MPdata1.csv', sep=';', index=False, header=props, mode='w')


def post_process():
    data = pd.read_csv("./MPdata1.csv", sep=';', index_col=None, header=0)
    conventional_volumes = []
    primitive_volumes = []
    for idx, irow in data.iterrows():
        if (idx+1) % 2000 == 0:
            print('processed', idx)
        cif_struct = Structure.from_str(irow["cif"], fmt="cif")
        sga = SpacegroupAnalyzer(cif_struct, symprec=0.1)
        conventional_struct = sga.get_conventional_standard_structure()
        primitive_struct = sga.get_primitive_standard_structure()
        conventional_volumes.append(conventional_struct.volume)
        primitive_volumes.append(primitive_struct.volume)
    data['conventional_volume'] = conventional_volumes
    data['primitive_volume'] = primitive_volumes
    data.to_csv('MPdata2.csv', sep=';', index=False, header=data.columns, mode='w')


def model():
    data = pd.read_csv("./MPdata2.csv", sep=';', index_col=None, header=0)
    data = data[data['icsd_ids'] != '[]']
    data = data[data['warnings'] == '[]']
    print('data size:', data.shape[0])
    Xsys = pd.get_dummies(data['crystal_system'], prefix='Xsys')
    conventional_vol = data['conventional_volume'].values.reshape(-1, 1)
    primitive_vol = data['primitive_volume'].values.reshape(-1, 1)
    scaler = StandardScaler()
    conventional_vol = scaler.fit_transform(conventional_vol)
    primitive_vol = scaler.fit_transform(primitive_vol)

#    X = np.concatenate((Xsys.values, conventional_vol), axis=1)
    X = np.concatenate((Xsys.values, primitive_vol), axis=1)
    y = (data['band_gap'] < 1E-6).values.reshape(-1, 1)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, train_size=0.8, test_size=0.2)

    cls = DecisionTreeClassifier(max_depth=8)
    cls.fit(X_train, y_train.ravel())

    train_score_roc = cls.score(X_train, y_train)
    test_score_roc = cls.score(X_test, y_test)
    print('train score', train_score_roc)
    print('test score', test_score_roc)

    y_pred = cls.predict_proba(X_test)[:, 1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    print("ROC-AUC:", roc_auc)
    ap_score = metrics.average_precision_score(y_test, y_pred)
    print('AP:', ap_score)

    disp = plot_precision_recall_curve(cls, X_test, y_test)
    disp.ax_.set_title('2-class Precision-Recall curve: '
                       'AP={0:0.2f}'.format(ap_score))

    plt.savefig('AP.png')


if __name__ == "__main__":
#    fetch_data()
#    post_process()
    model()
