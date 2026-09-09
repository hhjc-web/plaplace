#!/bin/bash
#SBATCH -A yangzhijian
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -J DLOC
#SBATCH -o output.log
#SBATCH -e err.log
#SBATCH -p gpu

PYTHON_PATH=/project/cgduan/Miniconda3/envs/torch12/bin
$PYTHON_PATH/python -u example_optimal_source.py --config=example_optimal_source.yml