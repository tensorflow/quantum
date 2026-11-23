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
#include <limits>
#include <vector>

#include "absl/status/status.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tfq {

namespace internal {

absl::Status BufferAllocationError(const uint64_t num_qubits) {
  return tensorflow::errors::ResourceExhausted(
      "Failed to allocate simulation buffer for ", num_qubits, " qubits.");
}

}  // namespace internal

int GetBlockSize(tensorflow::OpKernelContext* context, const int output_size) {
  if (output_size <= 0 || context == nullptr || context->device() == nullptr) {
    return 1;
  }
  const tensorflow::DeviceBase::CpuWorkerThreads* worker_threads =
      context->device()->tensorflow_cpu_worker_threads();
  if (worker_threads == nullptr || worker_threads->num_threads <= 0) {
    return 1;
  }
  return std::max(1, output_size / worker_threads->num_threads);
}

absl::Status GetMaxNumQubits(const std::vector<int>& num_qubits,
                             const uint64_t max_allowed,
                             uint64_t* max_num_qubits) {
  if (max_num_qubits == nullptr) {
    return tensorflow::errors::InvalidArgument(
        "max_num_qubits output pointer must not be null.");
  }
  if (num_qubits.empty()) {
    *max_num_qubits = 0;
    return absl::OkStatus();
  }
  const auto [min_it, max_it] =
      std::minmax_element(num_qubits.begin(), num_qubits.end());
  if (*min_it < 0) {
    return tensorflow::errors::InvalidArgument(
        "Circuit qubit count must be non-negative. Got ", *min_it, ".");
  }
  const uint64_t max_val = static_cast<uint64_t>(*max_it);
  if (max_val > max_allowed) {
    return tensorflow::errors::InvalidArgument(
        "Circuit exceeds maximum supported qubit count of ", max_allowed,
        ". Got ", max_val, ".");
  }
  *max_num_qubits = max_val;
  return absl::OkStatus();
}

absl::Status ValidateMPS1DQubitsAndBondDim(const std::vector<int>& num_qubits,
                                           const int bond_dim,
                                           uint64_t* max_num_qubits) {
  if (max_num_qubits == nullptr) {
    return tensorflow::errors::InvalidArgument(
        "max_num_qubits output pointer must not be null.");
  }
  uint64_t current_max = 0;
  if (!num_qubits.empty()) {
    const auto [min_it, max_it] =
        std::minmax_element(num_qubits.begin(), num_qubits.end());
    if (*min_it < 0) {
      return tensorflow::errors::InvalidArgument(
          "Circuit qubit count must be non-negative. Got ", *min_it, ".");
    }
    if (*min_it <= 3) {
      // Preserve legacy error text even though MPS 1D requires >= 4 qubits.
      return tensorflow::errors::InvalidArgument(
          "All input circuits require minimum 3 qubits.");
    }
    current_max = static_cast<uint64_t>(*max_it);
  }
  if (bond_dim < 4) {
    return tensorflow::errors::InvalidArgument(
        "bond_dim must be at least 4. Got ", bond_dim, ".");
  }
  // Keep total floats (8*bd + 4*bd^2*(nq+1)) <= UINT_MAX / sizeof(float) so
  // qsim::mps::MPSStateSpace's 32-bit unsigned sizes and signed int offsets
  // cannot wrap.
  constexpr uint64_t kMaxFloats =
      std::numeric_limits<unsigned int>::max() / sizeof(float);
  constexpr uint64_t kMaxBondDim = kMaxFloats / 8;
  static_assert(
      kMaxBondDim > 0 &&
          kMaxBondDim <= std::numeric_limits<uint64_t>::max() / 4 / kMaxBondDim,
      "4 * bond_dim^2 could overflow uint64_t.");
  const uint64_t bd = static_cast<uint64_t>(bond_dim);
  // Apply the 4-qubit minimum uniformly so empty batches check bond_dim too.
  const uint64_t nq = std::max<uint64_t>(current_max, 4);
  if (bd > kMaxBondDim ||
      nq + 1 > (kMaxFloats - uint64_t{8} * bd) / (uint64_t{4} * bd * bd)) {
    return tensorflow::errors::InvalidArgument(
        "MPS simulation state size overflows 32-bit index bounds for ", nq,
        " qubits and bond_dim ", bond_dim, ".");
  }
  *max_num_qubits = current_max;
  return absl::OkStatus();
}

}  // namespace tfq
