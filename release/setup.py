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
"""TensorFlow Quantum adds qauntum computing primitives to TensorFlow.

TensorFlow Quantum is an open source library for high performance batch
quantum computation on quantum simulators and quantum computers. The goal
of TensorFlow Quantum is to help researchers develop a deeper understanding
of quantum data and quantum systems via hybrid models.

TensorFlow Quantum was created in an ongoing collaboration between the
University of Waterloo and the Quantum AI team at Google along with help from
many other contributors within Google.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from datetime import date
from setuptools import Extension
from setuptools import find_packages
from setuptools import setup
from setuptools.dist import Distribution
from setuptools.command.install import install


DOCLINES = __doc__.split('\n')


class InstallPlatlib(install):
    """Workaround so .so files in generated wheels
    can be seen by auditwheel."""

    def finalize_options(self):
        install.finalize_options(self)
        if self.distribution.has_ext_modules():
            self.install_lib = self.install_platlib


REQUIRED_PACKAGES = [
    'cirq-core==0.13.1', 'cirq-google==0.13.1', 'sympy == 1.8',
    'googleapis-common-protos==1.52.0', 'google-api-core==1.21.0',
    'google-auth==1.18.0', 'protobuf==3.19.5'
]

# placed as extra to not have required overwrite existing nightly installs if
# they exist.
EXTRA_PACKAGES = ['tensorflow == 2.11.0']
CUR_VERSION = '0.7.3'


class BinaryDistribution(Distribution):
    """This class is needed in order to create OS specific wheels."""

    def has_ext_modules(self):
        return True


nightly = False
if '--nightly' in sys.argv:
    nightly = True
    sys.argv.remove('--nightly')

project_name = 'tensorflow-quantum'
build_version = CUR_VERSION
if nightly:
    project_name = 'tfq-nightly'
    build_version = CUR_VERSION + '.dev' + str(date.today()).replace('-', '')

setup(
    name=project_name,
    version=build_version,
    description=
    'TensorFlow Quantum is a library for hybrid quantum-classical machine learning.',
    long_description='\n'.join(DOCLINES[2:]),
    author='Google Inc.',
    author_email='no-reply@google.com',
    url='https://github.com/tensorflow/quantum/',
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    extras_require={'extras': EXTRA_PACKAGES},
    # Add in any packaged data.
    include_package_data=True,
    #ext_modules=[Extension('_foo', ['stub.cc'])],
    zip_safe=False,
    distclass=BinaryDistribution,
    # PyPI package information.
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Physics',
    ],
    license='Apache 2.0',
    keywords='tensorflow machine learning quantum qml',
    cmdclass={'install': InstallPlatlib})
