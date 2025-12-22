#!/bin/bash

#SBATCH -p gb200
#SBATCH -A coreai_dlalgo_llm
#SBATCH -t 00:15:00
#SBATCH -N 1
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=coreai_dlalgo_nemorl-bignlp:import
#SBATCH --output=import%j.log


# Run command
enroot import -o nemo_rl.sqsh docker://gitlab-master.nvidia.com/terryk/images/nemo-rl:tk-big-version-bump-a87f3e93-arm