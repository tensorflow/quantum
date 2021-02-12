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

IMAGE_REGISTRY=<image_registry>

.PHONY: training inference build apply-common apply-training delete-training apply-inference delete-inference remove-images

training: build apply-training

inference: build apply-inference

build:
	docker build -t ${IMAGE_REGISTRY} .
	docker push ${IMAGE_REGISTRY}

apply-common:
	kubectl apply -f common/sa.yaml

apply-training: apply-common
	kubectl apply -f training/qcnn.yaml

delete-training:
	kubectl delete -f common/sa.yaml --ignore-not-found; kubectl delete -f training/qcnn.yaml

apply-tensorboard: apply-common
	kubectl apply -f training/tensorboard.yaml

delete-inference:
	kubectl delete -f common/sa.yaml --ignore-not-found; kubectl delete -f training/tensorboard.yaml

apply-inference: apply-common
	kubectl apply -f inference/inference.yaml

delete-inference:
	kubectl delete -f common/sa.yaml --ignore-not-found; kubectl delete -f inference/inference.yaml

remove-images:
	docker rmi $(shell docker images -q ${IMAGE_REGISTRY})
