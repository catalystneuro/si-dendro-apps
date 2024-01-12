# si-dendro-apps
SpikeInterface Apps for Dendro


## Dev

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
DOCKER_BUILDKIT=1 docker build -t ghcr.io/catalystneuro/dendro_si_kilosort25:latest .
docker push ghcr.io/catalystneuro/dendro_si_kilosort25:latest
```