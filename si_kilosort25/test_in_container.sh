#!/bin/bash

# Docker image
IMAGE="ghcr.io/catalystneuro/dendro_si_kilosort25"

# Command to be executed inside the container
ENTRYPOINT_CMD="dendro"
ARGS="test-app-processor --app-dir . --processor spikeinterface_pipeline_ks25 --context sample_context_1.yaml"


# Run the Docker container
docker run --gpus all \
    -v $(pwd):/app \
    -v /mnt/shared_storage/Github/dendro/python:/src/dendro/python \
    -v /mnt/shared_storage/Github/spikeinterface_pipelines:/src/spikeinterface_pipelines \
    -w /app \
    --entrypoint "$ENTRYPOINT_CMD" \
    $IMAGE \
    $ARGS
