# si-dendro-apps
SpikeInterface Apps for Dendro


## Dev

Build single App image:
```shell
DOCKER_BUILDKIT=1 docker build -t <tag-name> .
```

Examples:
```shell
DOCKER_BUILDKIT=1 docker build -t ghcr.io/catalystneuro/dendro_si_kilosort25:latest .
docker push ghcr.io/catalystneuro/dendro_si_kilosort25:latest
```