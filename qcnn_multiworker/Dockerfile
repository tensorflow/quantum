# Copyright 2021 The TensorFlow Quantum Authors. All Rights Reserved.
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

FROM tensorflow/tensorflow:2.4.1

# google-cloud-core is a dependency of google-cloud-storage
# Pinning google-cloud-core to 1.5.0 explicitly for now to bypass the
# requirements conflict between tensorflow-quantum 0.5.0 and and the latest
# google-cloud-core (1.6.0) over the package google-auth.
RUN pip install tensorflow-quantum==0.5.0 google-cloud-core==1.5.0 google-cloud-storage==1.38.0 tensorboard-plugin-profile==2.4.0

COPY training/qcnn.py .
COPY inference/qcnn_inference.py .
COPY common/qcnn_common.py .
