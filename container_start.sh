# 사용할 임시 컨테이너 이름을 지정합니다. 예: nemo-rl-check
CONTAINER_NAME="nemo-rl-check"

# .sqsh 파일의 전체 경로를 사용합니다.
IMAGE_PATH="/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/nemo-rl"

# 'enroot create' 명령으로 컨테이너 인스턴스를 생성합니다.
# -f 옵션은 파일(File)에서 이미지를 생성하도록 지시합니다.
enroot create -f ${IMAGE_PATH} ${CONTAINER_NAME}
