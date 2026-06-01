#!/bin/bash
set -e

echo "===== Processing Body Datasets ====="

echo "[1/7] Processing AMASS body..."
python src/scripts/process/body/process_amass_body.py
echo "[1/7] AMASS body DONE"

echo "[2/7] Processing GRAB body..."
python src/scripts/process/body/process_grab_body.py
echo "[2/7] GRAB body DONE"

echo "[3/7] Processing ARCTIC body..."
python src/scripts/process/body/process_arctic_body.py
echo "[3/7] ARCTIC body DONE"

echo "[4/7] Processing BEAT2 body..."
python src/scripts/process/body/process_beat2_body.py
echo "[4/7] BEAT2 body DONE"

echo "[5/7] Processing OMOMO body..."
python src/scripts/process/body/process_omomo_body.py
echo "[5/7] OMOMO body DONE"

echo "[6/7] Processing SAMP body..."
python src/scripts/process/body/process_samp_body.py
echo "[6/7] SAMP body DONE"

echo "[7/7] Processing EMBODY3D body..."
python src/scripts/process/body/process_embody3d_body.py
echo "[7/7] EMBODY3D body DONE"

echo "===== All body datasets processed ====="
