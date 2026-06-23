#!/bin/bash
# Re-exec under bash if invoked via `sh`. This script uses bashisms
# (BASH_SOURCE, conda's bash hook); under dash BASH_SOURCE is empty and the
# repo-root detection below resolves to the parent of the repo, which breaks
# `pip install -r requirements.txt` and clones things outside the repo.
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi

eval "$(conda shell.bash hook)"

# This script lives in <repo>/sh_scripts/ ; operate on the repo root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Part 1: create the conda environment
# ---------------------------------------------------------------------------

# Remove any old env
conda env remove -n fusion_env -y

# Create fresh
conda create -n fusion_env python=3.11 -y
conda activate fusion_env

# Core Torch stack (stable)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"

pip install hatchling

# conda install -c conda-forge pytorch-gpu==2.1.2 -y
# conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
# conda install pytorch3d==0.7.6 -c conda-forge -y


# NumPy (let conda resolve, don’t force 2.0)
conda install numpy==1.26.4 -y

# Pip requirements
pip install --no-build-isolation -r requirements.txt

# prepare installing torch-mesh-isect
conda activate fusion_env

# get CUDA path for torch-mesh-isect installation from user
read -p "Enter CUDA_HOME path (e.g. /is/software/nvidia/cuda-12.1): " CUDA_HOME
export CUDA_HOME="$CUDA_HOME"


# clone external repos into external/ if they don't exist
mkdir -p external

if [ ! -d "external/torch-mesh-isect" ]; then
    git clone https://github.com/vchoutas/torch-mesh-isect.git external/torch-mesh-isect
fi

if [ ! -d "external/HMP" ]; then
    git clone https://github.com/enesduran/HMP.git external/HMP
fi

if [ ! -d "external/GrabNet" ]; then
    git clone https://github.com/otaheri/GrabNet.git external/GrabNet
fi

# install torch-mesh-isect
cd external/torch-mesh-isect
python setup.py install
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Part 2: unzip bundled data package if available
# ---------------------------------------------------------------------------

bash "$SCRIPT_DIR/unzip_data.sh"

echo "environment set up"
