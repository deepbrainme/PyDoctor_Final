#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "ERROR! Illegal number of parameters. Usage: bash install.sh conda_install_path environment_name"
    exit 0
fi

conda_install_path=$1
conda_env_name=$2

source $conda_install_path/etc/profile.d/conda.sh
echo "****************** Creating conda environment ${conda_env_name} python=3.7 ******************"
conda create -y --name $conda_env_name

echo ""
echo ""
echo "****************** Activating conda environment ${conda_env_name} ******************"
conda activate $conda_env_name

echo ""
echo ""
echo "****************** Installing pytorch with cuda10 ******************"
conda install -y pytorch torchvision cudatoolkit=10.0 -c pytorch

echo ""
echo ""
echo "****************** Installing matplotlib ******************"
conda install -y matplotlib


echo ""
echo ""
echo "****************** Installing opencv ******************"
pip install opencv-python

echo ""
echo ""
echo "****************** Installing gpustat ******************"
pip install gpustat

echo ""
echo ""
echo "****************** Installing tensorboard ******************"
pip install tb-nightly

echo ""
echo ""
echo "****************** Installing visdom ******************"
pip install visdom

echo ""
echo ""
echo "****************** Installing pydicom ******************"
pip install pydicom

echo ""
echo ""
echo "****************** Installing scipy ******************"
pip install scipy

echo ""
echo ""
echo "****************** Installing SimpleITK ******************"
pip install SimpleITK


echo ""
echo ""
echo "****************** Setting up environment ******************"
python -c "from pydoctor.evaluation.environment import create_default_local_file; create_default_local_file()"
python -c "from ltr.admin.environment import create_default_local_file; create_default_local_file()"


echo "****************** Downloading networks ******************"
mkdir pydoctor/networks
echo ""
echo ""
echo "****************** Trained networks can be downloaded from the baidu drive folder in INSTALL.md showed! ******************"

echo ""
echo ""
echo "****************** Installation complete! ******************"

