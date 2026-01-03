#!/bin/bash

# Stop script on any error
set -e

echo ">>> Loading Conda module..."
module load conda

# check if environment exists to avoid errors
if conda info --envs | grep -q "c3"; then
    echo ">>> Environment 'c3' already exists. Skipping creation."
else
    echo ">>> Creating environment 'c3'..."
    conda create -n c3 -y
fi

# Activate the environment
echo ">>> Activating 'c3'..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate c3

echo ">>> Installing Python 3.10.14..."
conda install python=3.10.14 -y

echo ">>> Installing PyTorch ecosystem..."
# Installing pytorch, torchvision, and torchaudio together ensures version compatibility
conda install -c pytorch pytorch=2.4.0 torchvision torchaudio -y

echo ">>> Installing utility libraries..."
conda install nvitop tqdm matplotlib seaborn wandb -y

echo ">>> Installing Torchviz..."
conda install dnachun::torchviz -y

echo ">>> Setup complete. You can activate the environment using: conda activate c3"