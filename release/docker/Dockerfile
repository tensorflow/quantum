# Copyright 2025 The TensorFlow Quantum Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

# Summary: build file for making isolated Docker environments for testing.
# This accepts 2 optional build arguments to set the Ubuntu & Python versions.
# Usage example:
#
#   docker build --no-cache --build-arg PYTHON_VERSION=3.9 \
#     --build-arg UBUNTU_VERSION=24.04 -t my-image:latest .
#
# Note that the name and tag ("my-image" and "latest" in the example above) are
# yours to choose. They're not set using the arguments or linked to their values.

# Default values for build arguments:
ARG PYTHON_VERSION=3.9
ARG UBUNTU_VERSION=22.04

FROM ubuntu:${UBUNTU_VERSION}

# Make the Python version argument visible to the rest of this file after FROM.
ARG PYTHON_VERSION
ENV PYTHON_VERSION=${PYTHON_VERSION}

ENV TZ=UTC
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

# Ensure the shell is Bash.
SHELL ["/bin/bash", "-e", "-c"]

# Tell the Dockerfile linter not to warn about how we use apt.
# hadolint global ignore=DL3008,DL3009

# Add the deadsnakes Python package archive to access older Python versions.
RUN apt-get update -q && \
    apt-get install -yq --no-install-recommends software-properties-common \
        gnupg pkg-config git zip unzip zlib1g-dev && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update -q

# The following conditional is to handle Python 3.12's removal of distutils.
RUN distutils_pkg="python${PYTHON_VERSION}-distutils" && \
    if [[ "${PYTHON_VERSION#*.}" -ge 12 ]]; then \
        distutils_pkg="python3-setuptools"; \
    fi && \
    apt-get install -yq --no-install-recommends python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv ${distutils_pkg} \
        python3-pip python-is-python3

# Use update-alternatives to make the desired version be the default.
# hadolint ignore=SC3010
RUN python_path="" bin="/usr/bin" && \
    case "$(lsb_release -rs)" in \
        "24.04") python_path="${bin}/python3.12" ;; \
        "22.04") python_path="${bin}/python3.10" ;; \
        "20.04") python_path="${bin}/python3.8"  ;; \
        *) python_path=$(readlink -f ${bin}/python3) ;; \
    esac && \
    if [[ ! -f "${python_path}" ]]; then \
        echo "ERROR: ${python_path} does not exist." >&2; \
        exit 1; \
    fi && \
    update-alternatives --install ${bin}/python3 python3 "${python_path}" 1 && \
    update-alternatives --install ${bin}/python3 python3 "${bin}/python${PYTHON_VERSION}" 2

# Clean up before finishing.
# hadolint ignore=DL3027
RUN apt clean

CMD ["/bin/bash"]
