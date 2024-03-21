# si-dendro-apps
SpikeInterface Apps for Dendro.

## Dev

For each Processor, create a symbolic link to the `common` folder.

```shell
ln -s ../common common
```

Create / update spec file:
```shell
dendro make-app-spec-file --app-dir . --spec-output-file spec.json
```

Build single App image:
```shell
DOCKER_BUILDKIT=1 docker build -t <tag-name> .
```

Examples:
```shell
DOCKER_BUILDKIT=1 docker build -f si_kilosort25/Dockerfile -t ghcr.io/catalystneuro/dendro_si_kilosort25:latest .
docker push ghcr.io/catalystneuro/dendro_si_kilosort25:latest

DOCKER_BUILDKIT=1 docker build -f si_kilosort3/Dockerfile -t ghcr.io/catalystneuro/dendro_si_kilosort3:latest .
docker push ghcr.io/catalystneuro/dendro_si_kilosort3:latest

DOCKER_BUILDKIT=1 docker build -f si_mountainsort5/Dockerfile -t ghcr.io/catalystneuro/dendro_si_mountainsort5:latest .
docker push ghcr.io/catalystneuro/dendro_si_mountainsort5:latest
```

## Test locally

Set up a bash script similar to this:
```shell
#!/bin/bash

# Docker image
IMAGE="ghcr.io/catalystneuro/dendro_si_kilosort25"

# Command to be executed inside the container
ENTRYPOINT_CMD="dendro"
ARGS="test-app-processor --app-dir . --processor spikeinterface_pipeline_ks25 --context sample_context_1.yaml"


# Run the Docker container, with hot-reload to local code versions
docker run --gpus all \
    -v $(pwd):/app \
    -v /mnt/shared_storage/Github/dendro/python:/src/dendro/python \
    -v /mnt/shared_storage/Github/spikeinterface_pipelines:/src/spikeinterface_pipelines \
    -w /app \
    --entrypoint "$ENTRYPOINT_CMD" \
    $IMAGE \
    $ARGS
```