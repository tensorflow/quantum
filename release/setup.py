# Copyright 2020 The TensorFlow Quantum Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""TensorFlow Quantum adds quantum computing primitives to TensorFlow.

TensorFlow Quantum is an open source library for high performance batch
quantum computation on quantum simulators and quantum computers. The goal
of TensorFlow Quantum is to help researchers develop a deeper understanding
of quantum data and quantum systems via hybrid models.

TensorFlow Quantum was created in an ongoing collaboration between the
University of Waterloo and the Quantum AI team at Google along with help
from many other contributors within Google.
"""

from __future__ import absolute_import, division, print_function

import sys
from datetime import date

from setuptools import Extension, find_packages, setup
from setuptools.command.install import install
from setuptools.dist import Distribution

DOCLINES = __doc__.split("\n")


class InstallPlatlib(install):
    """Workaround so .so files in generated wheels are visible to auditwheel."""

    def finalize_options(self):
        install.finalize_options(self)
        if self.distribution.has_ext_modules():
            self.install_lib = self.install_platlib


REQUIRED_PACKAGES = [
    "cirq-core==1.3.0",
    "cirq-google==1.3.0",
    "sympy==1.14",
]

# Placed as extras to avoid overwriting existing nightly TF installs.
EXTRA_PACKAGES = ["tensorflow>=2.16,<2.17"]

CUR_VERSION = "0.7.4"


class BinaryDistribution(Distribution):
    """Create OS-specific wheels."""

    def has_ext_modules(self):
        """whether this has external modules."""
        return True


NIGHTLY_FLAG = False
if "--nightly" in sys.argv:
    NIGHTLY_FLAG = True
    sys.argv.remove("--nightly")

PROJECT_NAME = "tensorflow-quantum"
BUILD_VERSION = CUR_VERSION
if NIGHTLY_FLAG:
    PROJECT_NAME = "tfq-nightly"
    BUILD_VERSION = CUR_VERSION + ".dev" + str(date.today()).replace("-", "")

setup(
    name=PROJECT_NAME,
    version=BUILD_VERSION,
    description="Library for hybrid quantum-classical machine learning.",
    long_description="\n".join(DOCLINES[2:]),
    author="The TensorFlow Quantum Authors",
    author_email="tensorflow-quantum-team@google.com",
    url="https://github.com/tensorflow/quantum/",
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    extras_require={"extras": EXTRA_PACKAGES},
    include_package_data=True,
    # ext_modules=[Extension('_foo', ['stub.cc'])],
    zip_safe=False,
    distclass=BinaryDistribution,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Quantum Computing",
    ],
    license="Apache 2.0",
    keywords="tensorflow machine learning quantum qml",
    cmdclass={"install": InstallPlatlib},
)
