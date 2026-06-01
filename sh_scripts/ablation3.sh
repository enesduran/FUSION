#!/bin/bash 
export PYTHON="/is/cluster/eduran2/miniconda3/envs/omomo_test2/bin/python" 
export PATH=$PATH 
declare script_path="/lustre/fast/fast/eduran2/fusion/src/eval/ablate3.py" 
$PYTHON $script_path
echo DONE