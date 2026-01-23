#! /bin/bash

python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip
pip3 install numpy
pip3 install scipy
pip3 install torch
pip3 install matplotlib
pip3 install einops     # for cleaner tensor operations
pip3 install seaborn    # for prettier plots
pip3 install ipython    # for interactive shell (especially for debugging)
pip3 install tqdm       # for progress bars
pip3 install pyyaml     # for configuration management
pip3 install h5py       # for storing datasets in HDF5 format