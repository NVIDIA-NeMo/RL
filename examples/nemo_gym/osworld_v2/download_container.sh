#!/bin/bash
set -euo pipefail

RELEASE_REPO="${CONTAINER_RELEASE_REPO:-jinglinglingling/nemo-rl-molt-osworld-backup}"
RELEASE_TAG="${CONTAINER_RELEASE_TAG:-osworld-v2-qualified-69764640}"
IMAGE_NAME=rl-osworld-v2-qualified-69764640.sqsh
IMAGE_SHA256=1b4edcdeac017210e6abe25885372e025ac111c45aeb038bda334659436eb74e
DESTINATION_DIR="${1:-$PWD}"

for command_name in gh sha256sum; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Required command is unavailable: $command_name" >&2
    exit 2
  fi
done

mkdir -p "$DESTINATION_DIR"
DESTINATION_DIR="$(cd "$DESTINATION_DIR" && pwd -P)"
destination="$DESTINATION_DIR/$IMAGE_NAME"

if [[ -f "$destination" ]]; then
  existing_sha256="$(sha256sum "$destination" | awk '{print $1}')"
  if [[ "$existing_sha256" == "$IMAGE_SHA256" ]]; then
    echo "Qualified container already present: $destination"
    exit 0
  fi
  echo "Refusing to overwrite container with unexpected checksum: $destination" >&2
  exit 2
fi

download_dir="$(mktemp -d "$DESTINATION_DIR/.osworld-container-parts.XXXXXX")"
partial_image="$DESTINATION_DIR/.$IMAGE_NAME.partial"
cleanup() {
  rm -rf "$download_dir"
  rm -f "$partial_image"
}
trap cleanup EXIT

gh release download "$RELEASE_TAG" \
  --repo "$RELEASE_REPO" \
  --pattern "$IMAGE_NAME.parts.sha256" \
  --pattern "$IMAGE_NAME.part-*" \
  --dir "$download_dir"

manifest="$download_dir/$IMAGE_NAME.parts.sha256"
test -s "$manifest"
mapfile -t part_names < <(awk '{print $2}' "$manifest")
if (( ${#part_names[@]} == 0 )); then
  echo "Container release has no parts: $RELEASE_REPO@$RELEASE_TAG" >&2
  exit 2
fi

for part_name in "${part_names[@]}"; do
  if [[ ! "$part_name" =~ ^rl-osworld-v2-qualified-69764640\.sqsh\.part-[0-9]{3}$ ]]; then
    echo "Unexpected part name in manifest: $part_name" >&2
    exit 2
  fi
done

(
  cd "$download_dir"
  sha256sum --check "$IMAGE_NAME.parts.sha256"
)

: > "$partial_image"
for part_name in "${part_names[@]}"; do
  cat "$download_dir/$part_name" >> "$partial_image"
done
actual_sha256="$(sha256sum "$partial_image" | awk '{print $1}')"
if [[ "$actual_sha256" != "$IMAGE_SHA256" ]]; then
  echo "Reconstructed container checksum mismatch" >&2
  echo "expected: $IMAGE_SHA256" >&2
  echo "actual:   $actual_sha256" >&2
  exit 2
fi

mv "$partial_image" "$destination"
trap - EXIT
rm -rf "$download_dir"
echo "Downloaded and verified: $destination"
