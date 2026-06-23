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

# get CUDA path for torch-mesh-isect installation from user (auto-detect a default
# from nvcc so this is not tied to any one machine's layout)
NVCC_BIN="$(command -v nvcc 2>/dev/null)"
if [ -n "$NVCC_BIN" ]; then
    DEFAULT_CUDA_HOME="$(dirname "$(dirname "$NVCC_BIN")")"
else
    DEFAULT_CUDA_HOME="/usr/local/cuda"
fi
read -p "Enter CUDA_HOME path [$DEFAULT_CUDA_HOME]: " CUDA_HOME
export CUDA_HOME="${CUDA_HOME:-$DEFAULT_CUDA_HOME}"
export PATH="$CUDA_HOME/bin:$PATH"


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

# CUDA samples provide helper_math.h, required by torch-mesh-isect's CUDA kernels.
# CUDA >= 11 no longer ships the samples with the toolkit, so clone them here.
if [ ! -d "external/cuda-samples" ]; then
    git clone --depth 1 https://github.com/NVIDIA/cuda-samples.git external/cuda-samples
fi

# install torch-mesh-isect.
# This is 2019-era code (PyTorch 1.0 / CUDA 10) and does NOT build as-is against
# the modern fusion_env stack. We patch it WITHOUT modifying the upstream repo,
# purely via environment variables scoped to this single command:
#   CPPFLAGS           : AT_CHECK was removed from PyTorch -> alias it to TORCH_CHECK
#   NVCC_PREPEND_FLAGS : setup.py passes a *relative* -Iinclude that ninja cannot
#                        resolve; hand nvcc the absolute path so it finds
#                        double_vec_ops.h
#   CUDA_SAMPLES_INC   : dir containing helper_math.h (located by search, so it is
#                        agnostic to how a given cuda-samples version is laid out)
# TORCH_CUDA_ARCH_LIST builds a fat binary for several GPU generations so the same
# install works on machines whose GPU differs from the one that built it. Without
# it, setup.py targets only the build host's GPU. Edit the list to match your fleet
# (CUDA 12.x supports up to 9.0); fewer arches = faster build.
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.0;7.5;8.0;8.6;8.9;9.0}"
cd external/torch-mesh-isect
TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
CUDA_SAMPLES_INC="$(dirname "$(find "$REPO_ROOT/external/cuda-samples" -name helper_math.h | head -n1)")" \
NVCC_PREPEND_FLAGS="-I$PWD/include" \
CPPFLAGS="-DAT_CHECK=TORCH_CHECK" \
python setup.py install
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Part 2: unzip bundled data package if available
# ---------------------------------------------------------------------------

bash "$SCRIPT_DIR/unzip_data.sh"

# leave the user in the fusion_env environment
conda activate fusion_env

echo "environment set up"
