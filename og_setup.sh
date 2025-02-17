#!/bin/bash
set -e  # Exit immediately if any command exits with a non-zero status.


# --- Setup Conda Environment ---
echo "Initializing conda..."
# Initialize conda for bash so that "conda activate" works inside this script.
eval "$(conda shell.bash hook)"

echo "Creating and activating the 'gvhmr' conda environment..."
conda create -y -n gvhmr python=3.10
conda activate gvhmr

# --- Fix numpy Dependency Conflict ---
echo "Installing numpy==1.23.5 to satisfy ultralytics dependency..."
pip install numpy==1.23.5

# --- Install GVHMR Package and Dependencies ---
echo "Installing GVHMR requirements and package in editable mode..."
pip install -r requirements.txt
pip install -e .

# --- DPVO Setup ---
echo "Entering DPVO directory..."
cd third-party/DPVO

echo "Downloading Eigen 3.4.0..."
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip

echo "Unzipping Eigen and cleaning up..."
unzip eigen-3.4.0.zip -d thirdparty && rm -rf eigen-3.4.0.zip

# --- Install PyTorch and Other Dependencies ---
echo "Installing PyTorch (torch), torch-scatter, numba, and pypose..."
# Install PyTorch with CUDA 12.1 support
pip install torch==2.3.0+cu121 torchvision==0.14.1+cu121 torchaudio==2.0.1 --extra-index-url https://download.pytorch.org/whl/cu121

# Install torch-scatter using the corresponding torch version
pip install torch-scatter -f "https://data.pyg.org/whl/torch-2.3.0+cu121.html"

# Install additional dependencies
pip install numba pypose

echo "Setting CUDA environment variables..."
export CUDA_HOME=/usr/local/cuda-12.1/
export PATH=$PATH:/usr/local/cuda-12.1/bin/

echo "Installing DPVO in editable mode using PEP517..."
pip install -e . --use-pep517

echo "Setup complete!"
