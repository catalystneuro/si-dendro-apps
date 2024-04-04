#!/bin/bash

# Docker image
IMAGE="ghcr.io/catalystneuro/dendro_si_mountainsort5"

# Command to be executed inside the container
ENTRYPOINT_CMD="dendro"
ARGS="test-app-processor --app-dir . --processor spikeinterface_pipeline_mountainsort5 --context sample_context_1.yaml"


# Run the Docker container, with conveninent volumes
docker run --gpus all \
    -v $(pwd)/results/output:/app/output \
    -v $(pwd)/results/results:/app/results \
    -v $(pwd)/results/scratch:/app/scratch \
    -v $(pwd)/sample_context_1.yaml:/app/sample_context_1.yaml \
    -v /mnt/shared_storage/Github/si-dendro-apps/common:/app/common \
    -v /mnt/shared_storage/Github/dendro/python:/src/dendro/python \
    -v /mnt/shared_storage/Github/spikeinterface_pipelines:/src/spikeinterface_pipelines \
    -v /mnt/shared_storage/Github/spikeinterface:/src/spikeinterface \
    -w /app \
    --entrypoint "$ENTRYPOINT_CMD" \
    $IMAGE \
    $ARGS
