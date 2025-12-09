#!/bin/bash

#SBATCH -p batch
#SBATCH -A coreai_dlalgo_nemorl
#SBATCH -t 00:15:00
#SBATCH -N 1
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --job-name=coreai_dlalgo_nemorl-bignlp:import
#SBATCH --output=import%j.log


# Run command
enroot import -o nemo_rl_v0.4.sqsh docker://nvcr.io#nvidia/nemo-rl:v0.4.0