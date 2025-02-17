#!/bin/bash

echo "Creating directories for inputs and outputs..."
mkdir -p inputs/checkpoints
mkdir -p outputs

echo "Setting up SMPL model..."
mkdir -p inputs/checkpoints/body_models/smpl

echo "Downloading SMPL zip file..."
wget "https://firebasestorage.googleapis.com/v0/b/human-audio-recording.appspot.com/o/smpl%2FSMPL_python_v.1.1.0%20(1).zip?alt=media&token=943d0328-3596-46b1-adaf-41f08100db56" -O inputs/checkpoints/body_models/smpl/smpl.zip

echo "Unzipping SMPL zip file..."
unzip inputs/checkpoints/body_models/smpl/smpl.zip -d inputs/checkpoints/body_models/smpl/

echo "Moving SMPL model files and cleaning up..."
mv inputs/checkpoints/body_models/smpl/SMPL_python_v.1.1.0/smpl/models/* inputs/checkpoints/body_models/smpl/ && rm -r inputs/checkpoints/body_models/smpl/SMPL_python_v.1.1.0

echo "Renaming SMPL model files..."
mv inputs/checkpoints/body_models/smpl/basicmodel_f_lbs_10_207_0_v1.1.0.pkl inputs/checkpoints/body_models/smpl/SMPL_FEMALE.pkl
mv inputs/checkpoints/body_models/smpl/basicmodel_m_lbs_10_207_0_v1.1.0.pkl inputs/checkpoints/body_models/smpl/SMPL_MALE.pkl
mv inputs/checkpoints/body_models/smpl/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl


echo "Downloading and unzipping DPVO checkpoint files..."
wget "https://firebasestorage.googleapis.com/v0/b/human-audio-recording.appspot.com/o/smpl%2Fotherfiles%2Fdpvo-20250214T201323Z-001.zip?alt=media&token=222b4ddc-94be-44ab-b8f4-3f4a463d31ff" -O inputs/checkpoints/dpvo.zip
unzip inputs/checkpoints/dpvo.zip -d inputs/checkpoints/

echo "Downloading and unzipping GVHMR checkpoint files..."
wget "https://firebasestorage.googleapis.com/v0/b/human-audio-recording.appspot.com/o/smpl%2Fotherfiles%2Fgvhmr-20250214T201413Z-001.zip?alt=media&token=207dbe42-b92c-49df-b242-3d05188d48f2" -O inputs/checkpoints/gvhmr_siga24_release.zip
unzip inputs/checkpoints/gvhmr_siga24_release.zip -d inputs/checkpoints/

echo "Downloading Vitpose checkpoint..."
mkdir -p inputs/checkpoints/vitpose
wget "https://firebasestorage.googleapis.com/v0/b/human-audio-recording.appspot.com/o/smpl%2Fotherfiles%2Fvitpose-h-multi-coco-001.pth?alt=media&token=b3a203fd-5e52-4a5a-98da-6facfc34b3b3" -O inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth

echo "Downloading HMR2 checkpoint..."
mkdir -p inputs/checkpoints/hmr2
wget "https://firebasestorage.googleapis.com/v0/b/human-audio-recording.appspot.com/o/smpl%2Fotherfiles%2Fepoch%3D10-step%3D25000-001.ckpt?alt=media&token=7a1c3db5-1ef4-49eb-a2d8-d541d7c1c465" -O inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt

echo "Downloading and unzipping YOLO checkpoint files..."
wget "https://firebasestorage.googleapis.com/v0/b/human-audio-recording.appspot.com/o/smpl%2Fotherfiles%2Fyolo-20250214T204604Z-001.zip?alt=media&token=ad86a364-eda6-4bbb-9fd6-49e89102378b" -O inputs/checkpoints/yolo.zip
unzip inputs/checkpoints/yolo.zip -d inputs/checkpoints/

echo "All files have been downloaded and processed successfully."