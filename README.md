# DeepKNet
Learning Materials Genome in the Momentum Space.


## Author info...
**TBD**


## Prerequisites

### Install PyTorch
Linux with GPU support:
```code
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```
Linux & Windows CPU-only: 
```code
conda install pytorch torchvision cpuonly -c pytorch
```
Mac CPU-only:
```code
conda install pytorch torchvision -c pytorch
```
See more instructions at the PyTorch [website](https://pytorch.org/get-started/locally/).

### Install pymatgen
```code
conda install --channel conda-forge pymatgen
```


## Materials data
Train, val, test data is by default stored in `./data/` folder.
The data generator packages are in `./gen_data/` folder.
In order to generate new dataset, follow these steps:
```code
(STEP0) cd ./gen_data
(STEP1)   python fetch_MPdata.py
(STEP2)   python custom_MPdata.py
(STEP3)   python compute_xrd.py
(STEP4)   python gen_training.py
```
This may take tens of minutes or more depending on your CPU power.


## Train the model
```python
python main.py 
```


