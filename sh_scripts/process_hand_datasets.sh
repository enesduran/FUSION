#!/bin/bash
set -e

echo "===== Processing Hand Datasets ====="

echo "[1/9] Processing GRAB hand..."
python src/scripts/process/hand/process_grab_hand.py
echo "[1/9] GRAB hand DONE"

echo "[2/9] Processing ARCTIC hand..."
python src/scripts/process/hand/process_arctic_hand.py
echo "[2/9] ARCTIC hand DONE"

echo "[3/9] Processing InterHands hand..."
python src/scripts/process/hand/process_interhands_hand.py
echo "[3/9] InterHands hand DONE"

echo "[4/9] Processing ReInterHands hand..."
python src/scripts/process/hand/process_reinterhands_hand.py
echo "[4/9] ReInterHands hand DONE"

echo "[5/9] Processing EMBODY3D hand..."
python src/scripts/process/hand/process_embody3d_hand.py
echo "[5/9] EMBODY3D hand DONE"

echo "[6/9] Processing EMBODY3D hand (refactored)..."
python src/scripts/process/hand/process_embody3d_hand_refactored.py
echo "[6/9] EMBODY3D hand (refactored) DONE"

echo "[7/9] Processing HOT3D hand..."
python src/scripts/process/hand/process_hot3d_hand.py
echo "[7/9] HOT3D hand DONE"

echo "[8/9] Processing SAMP hand..."
python src/scripts/process/hand/process_samp_hand.py
echo "[8/9] SAMP hand DONE"

echo "[9/9] Processing MOYO hand..."
python src/scripts/process/hand/process_moyo_hand.py
echo "[9/9] MOYO hand DONE"

echo "===== All hand datasets processed ====="
