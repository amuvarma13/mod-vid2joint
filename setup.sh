#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status.

# Clone the repository recursively
git clone https://github.com/zju3dv/GVHMR --recursive
cd GVHMR

# Initialize conda for bash so that "conda activate" works
eval "$(conda shell.bash hook)"

# Create and activate the conda environment
conda create -y -n gvhmr python=3.10
conda activate gvhmr

# Install GVHMR requirements and the package in editable mode
pip install -r requirements.txt
pip install -e .

# ---------------------------------------
# DPVO Setup
# ---------------------------------------
cd third-party/DPVO

# Download and unzip Eigen
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
unzip eigen-3.4.0.zip -d thirdparty && rm -rf eigen-3.4.0.zip

# Install additional dependencies
pip install torch-scatter -f "https://data.pyg.org/whl/torch-2.3.0+cu121.html"
pip install numba pypose

# Set CUDA environment variables
export CUDA_HOME=/usr/local/cuda-12.1/
export PATH=$PATH:/usr/local/cuda-12.1/bin/

# Install DPVO in editable mode
pip install -e .
