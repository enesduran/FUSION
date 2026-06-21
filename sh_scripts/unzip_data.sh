#!/bin/bash
# Extract the bundled data.zip into the repo root and verify expected assets.
#
# REQUIRED_PATHS below is the single source of truth for what the archive must
# contain — sh_scripts/build_data_zip.sh parses this array when bundling.
set -uo pipefail

# This script lives in <repo>/sh_scripts/ ; operate on the repo root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

FUSION_ZIP="data.zip"
if [ ! -f "$FUSION_ZIP" ]; then
    echo "Warning: $FUSION_ZIP not found. Skipping data extraction."
    exit 0
fi

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
    "data/motion/Hand_Raw/ARCTIC/data/arctic_data/data/meta/object_vtemplates"
    "data/motion/Hand_Raw/ARCTIC/data/arctic_data/data/meta/object_vtemplates_shifted"
    "data/motion/Hand_Raw/ARCTIC/data/arctic_data/data/meta/subject_vtemplates"
    "data/motion/Hand_Raw/ARCTIC/data/arctic_data/data/meta/hand_vtemplates"
    "data/motion/Body_Raw/OMOMO/captured_objects"
    "data/motion/Body_Raw/OMOMO/captured_objects_simplified"
    "data/motion/Hand_Raw/GRAB/grab/tools/subject_meshes"
    "data/motion/Hand_Raw/GRAB/grab/tools/object_meshes/contact_meshes"
    "data/motion/Hand_Raw/GRAB/grab/tools/object_meshes/contact_meshes_simplified"
)

for path in "${REQUIRED_PATHS[@]}"; do
    if [ ! -e "$path" ]; then
        echo "Warning: expected extracted asset not found: $path"
    fi
done
