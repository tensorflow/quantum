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

#ifndef TFQ_CORE_OPS_TFQ_SIMULATE_UTILS_H_
#define TFQ_CORE_OPS_TFQ_SIMULATE_UTILS_H_

#include <algorithm>
#include <cstdint>
#include <limits>

#include "tensorflow/core/framework/op_kernel.h"

namespace tfq {

// Returns a positive block size (>= 1) that distributes `output_size` work
// items across CPU worker threads, or 1 if `output_size <= 0` or `context` /
// CPU worker threads are unavailable or non-positive.
int GetBlockSize(tensorflow::OpKernelContext* context, const int output_size);

// Estimates per-item CPU cycles for ParallelFor when simulating circuits of up
// to `max_num_qubits` qubits, capping the shift so the result fits in signed
// int64_t (values near that limit already saturate thread scheduling).
inline constexpr int64_t EstimateSimulationCycles(uint64_t max_num_qubits) {
  constexpr int64_t kCycleMultiplier = 200;
  constexpr uint64_t kMaxCycleShift = 55;
  static_assert(kCycleMultiplier <=
                    (std::numeric_limits<int64_t>::max() >> kMaxCycleShift),
                "Cycle estimate overflows int64_t.");
  return kCycleMultiplier *
         (int64_t{1} << std::min(max_num_qubits, kMaxCycleShift));
}

}  // namespace tfq

#endif  // TFQ_CORE_OPS_TFQ_SIMULATE_UTILS_H_
