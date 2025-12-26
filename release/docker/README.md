# Docker images for basic testing

This directory contains a [`Dockerfile`](Dockerfile) for creating Ubuntu
containers with Python installed and very little else compared to the stock
Ubuntu images from https://hub.docker.com/_/ubuntu/. These can be used for
testing TensorFlow Quantum builds in relatively isolated environments.

The script [`create_docker_images.sh`](create_docker_images.sh) creates
separate images with Python 3.9&ndash;3.12 installed in Ubuntu 22.04 and 24.04.
The result is a total of eight images, with names like `ubuntu22-cp39`,
`ubuntu22-cp310`, etc. The script `create_docker_images.sh` is meant to be run
simply like this:

```shell
./create_docker_images.sh
```

The Dockerfile configuration runs a Bash shell as the last step if a container
is not started with any other command to run. When combined with Docker's `-v`
argument, you can easily run commands inside the containere environment while
accessing your TensorFlow Quantum source files. For example:

```shell
# The next cd command moves to the root of the source tree.
cd $(git rev-parse --show-toplevel)
docker run -it --rm --network host -v .:/tfq ubuntu24-cp312
```

will leave you with a shell prompt inside a basic Ubuntu 24.04 environment with
Python 3.12 preinstalled and your local TensorFlow Quantum source directory
accessible at `/tfq` from inside the container.
