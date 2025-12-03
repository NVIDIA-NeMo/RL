COMMAND="uv run python examples/run_grpo_math.py --config examples/configs/grpo_math_8B_megatron.yaml" \
CONTAINER=gitlab-master.nvidia.com/terryk/images/nemo-rl:tk-big-version-bump-a87f3e93-arm  \
MOUNTS="${PWD}:${PWD},${PWD}/venvs:/opt/ray_venvs" \
sbatch \
--account coreai_dlalgo_nemorl \
--partition batch \
--job-name TestRL \
--gpus=4 \ 
--nodes=1 \
ray.sub

