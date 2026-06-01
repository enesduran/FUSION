#!/bin/bash 
export PYTHON="/is/cluster/eduran2/miniconda3/envs/fusion/bin/python" 
export PATH=$PATH 
declare script_path="/lustre/fast/fast/eduran2/fusion/src/eval/ablate1.py" 
$PYTHON $script_path
echo DONE