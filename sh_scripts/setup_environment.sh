#!/bin/bash
eval "$(conda shell.bash hook)"

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

# Unzip bundled data package if available
FUSION_ZIP="data.zip"
if [ -f "$FUSION_ZIP" ]; then
    if ! command -v unzip >/dev/null 2>&1; then
        echo "Error: unzip is required to extract $FUSION_ZIP."
        exit 1
    fi

    echo "Extracting $FUSION_ZIP ..."
    unzip -o "$FUSION_ZIP" -d .

    REQUIRED_PATHS=(
        "data/body_models/smplx"
        "data/body_models/mano"
        "data/body_models/watertight"
        "data/checkpoints/motionfix"
        "data/motion/statistics.npy"
        "data/body_models/smplx_parts_segm.pkl"
        "data/body_models/MANO_SMPLX_vertex_ids.pkl"
        "data/self_interaction"
        "data/sample_data"
        "data/sample_data_precomputed"
    )

    for path in "${REQUIRED_PATHS[@]}"; do
        if [ ! -e "$path" ]; then
            echo "Warning: expected extracted asset not found: $path"
        fi
    done
else
    echo "Warning: $FUSION_ZIP not found. Skipping data extraction."
fi

# prepare installing torch-mesh-isect 
conda activate fusion_env

# get CUDA path for torch-mesh-isect installation from user 
read -p "Enter CUDA_HOME path (e.g. /is/software/nvidia/cuda-12.1): " CUDA_HOME
export CUDA_HOME="$CUDA_HOME"


# clone torch-mesh-isect repo if it doesn't exist
if [ ! -d "external/torch-mesh-isect" ]; then
    git clone https://github.com/vchoutas/torch-mesh-isect.git external/torch-mesh-isect
fi

if [ ! -d "external/HMP" ]; then
    git clone https://github.com/enesduran/HMP.git
fi

if [ ! -d "external/GrabNet" ]; then
    git clone https://github.com/otaheri/GrabNet.git
fi

# install torch-mesh-isect
cd external/torch-mesh-isect
python setup.py install

echo "environment set up"