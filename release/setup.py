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

import sys
from datetime import date

from setuptools import find_packages, setup
from setuptools.command.install import install
from setuptools.dist import Distribution

CUR_VERSION = "0.7.7"

DOCLINES = __doc__.split("\n")


class InstallPlatlib(install):
    """Workaround so .so files in generated wheels are visible to auditwheel."""

    def finalize_options(self):
        install.finalize_options(self)
        if self.distribution.has_ext_modules():
            self.install_lib = self.install_platlib


REQUIRED_PACKAGES = [
    "cirq-core==1.5.0",
    "cirq-google==1.5.0",
    "numpy>=2,<3",
    "scipy>=1.15.3,<2",
    "sympy==1.14",
    "tf-keras>=2.18,<2.19",

    # The following are transitive dependencies that need to be constrained to
    # avoid incompatible versions or because some (e.g., contourpy 1.3.3)
    # require Python 3.11+ and we want to maintain Python 3.10 compatibility.
    # TODO: revisit after we reach compatibility with TensorFlow 2.19+.
    "jax>=0.5,<0.6",
    "contourpy<=1.3.2",
]

# TF requirement is placed as an extras to avoid overwriting existing nightly TF
# installations. Users can run "pip install tensorflow-quantum[and-tensorflow]"
# to get everything in one go (or "pip install tensorflow tensorflow-quantum").
EXTRA_PACKAGES = {}
EXTRA_PACKAGES["and-tensorflow"] = ["tensorflow>=2.18,<2.19"]
# "extras" was used before 0.7.4. Prefer "and-tensorflow" in 0.7.4+.
EXTRA_PACKAGES["extras"] = EXTRA_PACKAGES["and-tensorflow"]


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
    long_description_content_type="text/markdown",
    author="The TensorFlow Quantum Authors",
    author_email="tensorflow-quantum-team@google.com",
    url="https://github.com/tensorflow/quantum/",
    packages=find_packages(),
    python_requires='>=3.10',
    install_requires=REQUIRED_PACKAGES,
    extras_require=EXTRA_PACKAGES,
    include_package_data=True,
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
