# follow the following steps 

# install pymatgen for XRD pattern calculation
conda install -c matsci pymatgen

# install tensorflow
pip install tensorflow(-gpu)

# pointcloud data is stored in ./data/ folder,
# the data generator functions are in data_gen/ folder

# to train the model
python train.py
