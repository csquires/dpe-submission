#! /bin/bash

python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip
pip3 install numpy
pip3 install matplotlib
pip3 install seaborn
pip3 install torch
pip3 install ipython