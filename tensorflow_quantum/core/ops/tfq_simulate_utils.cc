/* Copyright 2020 The TensorFlow Quantum Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow_quantum/core/ops/tfq_simulate_utils.h"

#include <algorithm>
#include <cstdint>

#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tfq {

int GetBlockSize(tensorflow::OpKernelContext* context, const int output_size) {
  if (output_size <= 0 || context == nullptr || context->device() == nullptr) {
    return 1;
  }
  const auto* worker_threads =
      context->device()->tensorflow_cpu_worker_threads();
  if (worker_threads == nullptr || worker_threads->num_threads <= 0) {
    return 1;
  }
  return static_cast<int>(std::max<int64_t>(
      1, static_cast<int64_t>(output_size) /
             static_cast<int64_t>(worker_threads->num_threads)));
}

}  // namespace tfq
