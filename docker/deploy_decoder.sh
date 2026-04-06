#!/bin/bash
# Deploy the speechBCI decoder to GCP Compute Engine.
#
# Prerequisites:
#   - gcloud CLI authenticated
#   - Docker installed locally
#   - Model files staged in model/ and vendor/ directories
#
# Usage:
#   ./docker/deploy_decoder.sh [build|push|deploy|all]

set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID:?Set GCP_PROJECT_ID}"
REGION="${GCP_REGION:-us-west1}"
ZONE="${GCP_ZONE:-us-west1-b}"
INSTANCE_NAME="bci-decoder"
IMAGE_NAME="bci-decoder"
REGISTRY="${REGION}-docker.pkg.dev/${PROJECT_ID}/clef/${IMAGE_NAME}"
MACHINE_TYPE="e2-highmem-16"

build() {
    echo "Building Docker image..."
    docker build -f docker/Dockerfile.decoder -t "${IMAGE_NAME}" .
    docker tag "${IMAGE_NAME}" "${REGISTRY}:latest"
}

push() {
    echo "Pushing to Artifact Registry..."
    gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet
    docker push "${REGISTRY}:latest"
}

deploy() {
    echo "Deploying to Compute Engine..."
    gcloud compute instances create-with-container "${INSTANCE_NAME}" \
        --project="${PROJECT_ID}" \
        --zone="${ZONE}" \
        --machine-type="${MACHINE_TYPE}" \
        --container-image="${REGISTRY}:latest" \
        --container-env="PYTHONUNBUFFERED=1" \
        --tags=bci-decoder \
        --boot-disk-size=50GB

    # Open port 8765
    gcloud compute firewall-rules create allow-bci-decoder \
        --project="${PROJECT_ID}" \
        --allow=tcp:8765 \
        --target-tags=bci-decoder \
        --description="Allow WebSocket connections to BCI decoder" \
        2>/dev/null || true

    # Get external IP
    IP=$(gcloud compute instances describe "${INSTANCE_NAME}" \
        --zone="${ZONE}" \
        --project="${PROJECT_ID}" \
        --format='get(networkInterfaces[0].accessConfigs[0].natIP)')
    echo ""
    echo "Deployed! Set decoder_url in io_config.yaml to:"
    echo "  ws://${IP}:8765/ws"
}

case "${1:-all}" in
    build) build ;;
    push) push ;;
    deploy) deploy ;;
    all) build && push && deploy ;;
    *) echo "Usage: $0 [build|push|deploy|all]" && exit 1 ;;
esac
