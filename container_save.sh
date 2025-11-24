#!/bin/bash

#SBATCH -A coreai_dlalgo_llm 
#SBATCH -p interactive
#SBATCH -J coreai_dlalgo_llm-rl:automodel_container
#SBATCH -t 1:00:00 
#SBATCH -N 1 
#SBATCH --mem=0 
#SBATCH --gres=gpu:8
#SBATCH --dependency=singleton 
ACCOUNT="coreai_dlalgo_genai"

MOUNTS="\
/lustre:/lustre,\
$(pwd):$(pwd)"
ORIGINAL_CONTAINER_NAME="/lustre/fsw/portfolios/coreai/users/terryk/enroot-images/nvcr.io/nvidian/nemo-rl:bd2e645-37644239.squashfs"
NEW_CONTAINER_NAME="/lustre/fsw/portfolios/coreai/users/zhiyul/containers/nemo-rl-bd2e645-37644239-repro.sqsh"


srun -N1 \
 -n1 \
 -A ${ACCOUNT} \
 -J ${ACCOUNT}-rl:zhiyul-automodel-container-setup \
 -t 1:00:00 \
 -p interactive \
 --gres=gpu:1 \
 --export=ALL,HOME=/tmp \
 --no-container-mount-home \
 --container-mounts ${MOUNTS} \
 --container-image=${ORIGINAL_CONTAINER_NAME} \
 --container-writable \
 --container-workdir $(pwd) \
 --container-save=${NEW_CONTAINER_NAME} \
 --pty bash -c "
apt-get update && apt-get install -y libxrender1 libxext6 libsm6 libxrandr2 libxfixes3 libxi6
echo 'libX11 installation completed'
"