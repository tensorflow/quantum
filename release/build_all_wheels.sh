#!/bin/bash
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
# Build python wheels for 3.6 and 3.7 and store them in wheels/
cd ..
git clone https://github.com/tensorflow/custom-op.git
cd custom-op
git checkout 994dc6bdd5b7c0c0c0ffb55bb0ac013d9d9268cd
cd ..

# Copy the toolchain config over from custom-op.

# Upgrade existing 3.6 pip.
python3 -m pip install --upgrade pip
python3 -m pip install tensorflow==2.1.0

cp -r custom-op/third_party/toolchains quantum/third_party/
cd /quantum
echo "Y\n" | ./configure.sh

bazel build -c opt --crosstool_top=//third_party/toolchains/preconfig/ubuntu16.04/gcc7_manylinux2010-nvcc-cuda10.0:toolchain \
 --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --cxxopt="-msse2" --cxxopt="-msse3" --cxxopt="-msse4" release:build_pip_package
bazel-bin/release/build_pip_package /tmp/tfquantum/

mkdir wheels

cp /tmp/tfquantum/tfquantum-0.2.0-cp36-cp36m-linux_x86_64.whl wheels/tfquantum-0.2.0-cp36-cp36m-linux_x86_64.whl
bazel clean

# Now build the 3.7 wheel.
cd /tmp
wget https://www.python.org/ftp/python/3.7.5/Python-3.7.5rc1.tar.xz
tar -xf Python-3.7.5rc1.tar.xz
cd Python-3.7.5rc1
./configure
make -j2 build_all
sudo make altinstall

cd /quantum

python3.7 -m pip install --upgrade pip setuptools
python3.7 -m pip install tensorflow==2.1.0
sed -i 's/python3/python3.7/g' configure.sh
sed -i 's/python3/python3.7/g' release/build_pip_package.sh

echo "Y\n" | ./configure.sh

bazel build -c opt --crosstool_top=//third_party/toolchains/preconfig/ubuntu16.04/gcc7_manylinux2010-nvcc-cuda10.0:toolchain \
 --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" --cxxopt="-msse2" --cxxopt="-msse3" --cxxopt="-msse4" release:build_pip_package
bazel-bin/release/build_pip_package /tmp/tfquantum/

cp /tmp/tfquantum/tfquantum-0.2.0-cp37-cp37m-linux_x86_64.whl wheels/tfquantum-0.2.0-cp37-cp37m-linux_x86_64.whl
bazel clean

sed -i 's/python3.7/python3/g' configure.sh
sed -i 's/python3.7/python3/g' release/build_pip_package.sh

rm -r /quantum/third_party/toolchains
exit
