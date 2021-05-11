# Installation 

This document contain detailed instructions for installing the necessary dependencies for PyDoctor. The instrustions have been test on an Ubuntu 18.04 system. We recommend using the [install script](install.sh) if you have not already tried that.

### Requirements
* Conda installation with Python3.7. If not already installed, install from https://www.anaconda.com/distribution/.
* Nvidia GPU

## Step-by-Step instructions
#### Create and activate a conda environment
```bash
conda create --name pydoctor python=3.7
conda activate pydoctor
```
#### Install PyTorch
Install PyTorch with cuda10
```bash
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
```

**Note:**
- It is possible to use any PyTorch supported version of CUDA(not necessarily v10)
- For more detials about PyTorch installation. see https://pytorch.org/get-started/previous-versions/.

#### Install matplotlib, SimpleITK, pydicom, opencv, visdom, tensorboard gpustat and scipy.
```bash
conda install matplotlib 
pip install opencv-python visdom tb-nightly scipy pydicom SimpleITK gpustat
```

#### Setup the environment
Create the default environment setting files.
```bash
# Environment settings for pydoctor. Saved at pydoctor/evaluation/local.py
python -c "from pydoctor.evaluation.environment import create_default_local_file; create_default_local_file() "

# Environment settings for ltr. Saved at ltr/admin/local.py
python -c "from ltr.admin.environment import create_default_local_file; create_default_local_file()"
```
You can modify local.py files to set the paths to datasets .results path etc.

















