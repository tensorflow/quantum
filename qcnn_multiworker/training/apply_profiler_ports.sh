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

#!/usr/bin/env bash

kubectl get svc --no-headers -o custom-columns=":metadata.name" | grep qcnn-worker | while read -r svc; do
    kubectl patch svc $svc -p '[{"op":"add", "path":"/spec/ports/-", "value":{"name": "profiler-port", "port": 2223, "protocol": "TCP", "targetPort": 2223}}]' --type=json
done
