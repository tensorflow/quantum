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
# Load up an unbuntu 16.04 docker for building compatable pip packages.
sudo docker pull tensorflow/tensorflow:custom-op-ubuntu16
sudo docker run -it -v ${PWD}:/quantum -w /quantum tensorflow/tensorflow:custom-op-ubuntu16

# Writing the permissions for the wheels directory inside of the docker doesn't work.
sudo chmod -R 777 wheels
