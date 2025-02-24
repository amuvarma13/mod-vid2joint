#!/bin/bash
set -e  # Exit immediately if any command fails

echo "Initializing conda..."
# Initialize conda so that 'conda activate' works in this script.
eval "$(conda shell.bash hook)"

echo "Creating and activating the 'gvhmr' conda environment..."
conda create -y -n gvhmr python=3.10
conda activate gvhmr

echo "Installing numpy==1.23.5 to satisfy ultralytics dependency..."
pip install numpy==1.23.5

echo "Installing GVHMR requirements and package in editable mode..."
pip install -r requirements.txt
pip install -e .

echo "Entering DPVO directory..."
cd third-party/DPVO

echo "Downloading Eigen 3.4.0..."
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip

echo "Unzipping Eigen and cleaning up..."
unzip eigen-3.4.0.zip -d thirdparty && rm -rf eigen-3.4.0.zip

# Install PyTorch (with CUDA 12.1 support) BEFORE installing torch-scatter
echo "Installing PyTorch, torchvision, and torchaudio with CUDA 12.1 support..."
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121

echo "Installing torch-scatter, numba, and pypose..."
pip install torch-scatter -f "https://data.pyg.org/whl/torch-2.3.0+cu121.html"
pip install numba pypose

echo "Setting CUDA environment variables..."
export CUDA_HOME=/usr/local/cuda-12.1/
export PATH=$PATH:/usr/local/cuda-12.1/bin/

echo "Installing DPVO in editable mode ..."
pip install -e .

cd ../../
# Initial SMPL setup
mkdir -p inputs/checkpoints/body_models/smpl
wget "https://firebasestorage.googleapis.com/v0/b/human-audio-recording.appspot.com/o/smpl%2FSMPL_python_v.1.1.0%20(1).zip?alt=media&token=943d0328-3596-46b1-adaf-41f08100db56" -O inputs/checkpoints/body_models/smpl/smpl.zip
unzip inputs/checkpoints/body_models/smpl/smpl.zip -d inputs/checkpoints/body_models/smpl/
mv inputs/checkpoints/body_models/smpl/SMPL_python_v.1.1.0/smpl/models/* inputs/checkpoints/body_models/smpl/ && rm -r inputs/checkpoints/body_models/smpl/SMPL_python_v.1.1.0

# Rename SMPL model files
mv inputs/checkpoints/body_models/smpl/basicmodel_f_lbs_10_207_0_v1.1.0.pkl inputs/checkpoints/body_models/smpl/SMPL_FEMALE.pkl
mv inputs/checkpoints/body_models/smpl/basicmodel_m_lbs_10_207_0_v1.1.0.pkl inputs/checkpoints/body_models/smpl/SMPL_MALE.pkl
mv inputs/checkpoints/body_models/smpl/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl

# SMPLX setup
wget "https://firebasestorage.googleapis.com/v0/b/human-audio-recording.appspot.com/o/smpl%2Fmodels_smplx_v1_1.zip?alt=media&token=efb8f84c-63e7-4f01-9932-de1f8b52e9d1" -O inputs/checkpoints/body_models/smplx.zip
unzip inputs/checkpoints/body_models/smplx.zip -d inputs/checkpoints/body_models
mkdir -p inputs/checkpoints/body_models/smplx
mv inputs/checkpoints/body_models/models/smplx/* inputs/checkpoints/body_models/smplx/ && rm -r inputs/checkpoints/body_models/models

# Get and unzip dpvo
wget "https://firebasestorage.googleapis.com/v0/b/human-audio-recording.appspot.com/o/smpl%2Fotherfiles%2Fdpvo-20250214T201323Z-001.zip?alt=media&token=222b4ddc-94be-44ab-b8f4-3f4a463d31ff" -O inputs/checkpoints/dpvo.zip
unzip inputs/checkpoints/dpvo.zip -d inputs/checkpoints/

# Get and unzip gvhmr
wget "https://firebasestorage.googleapis.com/v0/b/human-audio-recording.appspot.com/o/smpl%2Fotherfiles%2Fgvhmr-20250214T201413Z-001.zip?alt=media&token=207dbe42-b92c-49df-b242-3d05188d48f2" -O inputs/checkpoints/gvhmr_siga24_release.zip
unzip inputs/checkpoints/gvhmr_siga24_release.zip -d inputs/checkpoints/

# Get vitpose
mkdir -p inputs/checkpoints/vitpose
wget "https://firebasestorage.googleapis.com/v0/b/human-audio-recording.appspot.com/o/smpl%2Fotherfiles%2Fvitpose-h-multi-coco-001.pth?alt=media&token=b3a203fd-5e52-4a5a-98da-6facfc34b3b3" -O inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth

# Get hmr2
mkdir -p inputs/checkpoints/hmr2
wget "https://firebasestorage.googleapis.com/v0/b/human-audio-recording.appspot.com/o/smpl%2Fotherfiles%2Fepoch%3D10-step%3D25000-001.ckpt?alt=media&token=7a1c3db5-1ef4-49eb-a2d8-d541d7c1c465" -O inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt

# Get and unzip yolo
wget "https://firebasestorage.googleapis.com/v0/b/human-audio-recording.appspot.com/o/smpl%2Fotherfiles%2Fyolo-20250214T204604Z-001.zip?alt=media&token=ad86a364-eda6-4bbb-9fd6-49e89102378b" -O inputs/checkpoints/yolo.zip
unzip inputs/checkpoints/yolo.zip -d inputs/checkpoints/