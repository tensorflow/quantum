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

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

# Ensure the shell is Bash.
SHELL ["/bin/bash", "-eo", "pipefail", "-c"]

# Tell the Dockerfile linter not to warn about how we use apt.
# hadolint global ignore=DL3008,DL3009

RUN apt-get -q update -q && \
    apt-get install -y --no-install-recommends ca-certificates \
         pkg-config gnupg curl lsb-release git zip unzip

# We avoid the use of "add-apt-repository" because it is only available from the
# "software-properties-common" package, and installing that package brings in a
# lot of other unrelated unnecessary packages. Instead, we fetch the GPG key for
# the deadsnakes package archive and install it where apt can find it.
ENV ID=0xF23C5A6CF475977595C89F51BA6932366A755776
ENV GPG_FILE="/etc/apt/trusted.gpg.d/deadsnakes.gpg"
RUN curl -sSL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=${ID}" | \
    gpg --dearmor -o ${GPG_FILE}

ENV PPA_URL="https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu"
ENV APT_FILE="/etc/apt/sources.list.d/deadsnakes.list"
RUN echo "deb [signed-by=${GPG_FILE}] ${PPA_URL} $(lsb_release -cs) main" > "${APT_FILE}"

RUN apt-get -q update -q && \
    apt-get install -yq --no-install-recommends make clang g++ zlib1g-dev \
        python${PYTHON_VERSION} python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-venv python-is-python3

# Use update-alternatives to make the desired version be the default.
# hadolint ignore=SC3010
RUN python_path="" bin="/usr/bin" && \
    case "$(lsb_release -rs)" in \
        "24.04") python_path="${bin}/python3.12" ;; \
        "22.04") python_path="${bin}/python3.10" ;; \
        "20.04") python_path="${bin}/python3.8"  ;; \
        *) python_path=$(readlink -f ${bin}/python3) ;; \
    esac && \
    update-alternatives --install ${bin}/python3 python3 "${python_path}" 1 && \
    update-alternatives --install ${bin}/python3 python3 "${bin}/python${PYTHON_VERSION}" 2

# Install pip.
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3

# Clean up before finishing.
RUN apt-get clean

CMD ["/bin/bash"]
