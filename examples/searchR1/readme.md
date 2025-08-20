# Search-R1


## Set up Retrieval Server

We use a separate virtual environment for the retrieval server.

```bash
uv venv retrieval_env
source retrieval_env/bin/activate
uv pip install -r requirements.txt
```

### Install Faiss

We use `faiss`, a similarity search library for vectors, 

The following instructions install Faiss-GPU from source. Please refer to [this](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md) for more details.

```bash
uv pip install -U cmake
git clone https://github.com/facebookresearch/faiss.git thirdparty/faiss
cd thirdparty/faiss

# load the openblas library if necessary
module load openblas

uv run --active cmake -B build . \
    -DFAISS_ENABLE_GPU=ON \
    -DCMAKE_CUDA_ARCHITECTURES="90" \
    -DCMAKE_CUDA_COMPILER="/usr/local/cuda/bin/nvcc" \
    -DBUILD_TESTING=OFF 

uv run --active make -C build -j faiss

# Build the python binding
uv run --active make -C build -j swigfaiss
(cd build/faiss/python && uv run --active python setup.py install)

# Return to the example directory
cd ../../
```

### Download Dataset

```
uv run --active bash prepare.sh
```

### Launch Retrieval Server

The `wiki-18` dataset requires 2 80GB GPUs to run.

```
uv run --active sh retrieval_launch.sh 
```

## Launch Training Script
Run from the root of NeMo RL repo

```bash
NUM_ACTOR_NODES=1
SEARCH_URL=http://<retrieval_server_url>/retrieve

COMMAND=$(cat <<EOF
 uv run --active ./examples/searchR1/run_grpo_searchr1.py \
    env.search.search_url="${SEARCH_URL}" \
EOF
) \
CONTAINER=<container> \
MOUNTS="/lustre:/lustre:ro,$PWD:$PWD" \
sbatch \
    --nodes=${NUM_ACTOR_NODES} \
    --job-name="grpo_searchr1" \
    --time=4:0:0 \
    --gres=gpu:8 \
    --account=<your_account> \
    --partition=<your_partition> \
    ray.sub
```
