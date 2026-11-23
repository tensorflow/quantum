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
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "absl/status/status.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"

namespace tfq {

namespace internal {

absl::Status BufferAllocationError(uint64_t num_qubits);

template <typename Factory, typename Buffer>
inline bool TryReallocateBuffer(const uint64_t nq, const Factory& create_buffer,
                                Buffer* buffer) {
  *buffer = create_buffer(nq);
  return buffer->get() != nullptr;
}

}  // namespace internal

static_assert(sizeof(size_t) >= sizeof(uint64_t),
              "64-bit size_t is required for qsim buffer-size bounds.");

// State-vector qubit limit: 8 * 2^59 = 2^62 bytes fits int64_t/size_t and
// keeps uint64_t bitmask shifts well-defined (n = 60 overflows int64_t).
inline constexpr uint64_t kMaxStateVectorQubits = 59;
static_assert(kMaxStateVectorQubits < 64 &&
                  sizeof(float) * 2 <=
                      static_cast<size_t>(std::numeric_limits<int64_t>::max() >>
                                          kMaxStateVectorQubits),
              "kMaxStateVectorQubits would overflow state-vector byte size.");

// Unitary qubit limit: 8 * 2^(2*29) = 2^61 bytes fits int64_t/size_t
// (n = 30 overflows int64_t; n = 32 wraps 2^n * 2^n).
inline constexpr uint64_t kMaxUnitaryQubits = 29;
static_assert(2 * kMaxUnitaryQubits < 64 &&
                  sizeof(float) * 2 <=
                      static_cast<size_t>(std::numeric_limits<int64_t>::max() >>
                                          (2 * kMaxUnitaryQubits)),
              "kMaxUnitaryQubits would overflow unitary byte size.");

// Shift clamp for ParallelFor cost_per_unit: 200 * 2^30 fits int64_t and exact
// double precision while already saturating Eigen thread sharding.
inline constexpr uint64_t kMaxCycleEstimateQubits = 30;
static_assert(kMaxCycleEstimateQubits < 63 &&
                  int64_t{200} <= (std::numeric_limits<int64_t>::max() >>
                                   kMaxCycleEstimateQubits),
              "kMaxCycleEstimateQubits would overflow int64_t cycle estimate.");

// Returns a positive block size (>= 1) that distributes `output_size` work
// items across CPU worker threads, or 1 if `output_size <= 0` or `context` /
// CPU worker threads are unavailable or non-positive.
int GetBlockSize(tensorflow::OpKernelContext* context, const int output_size);

// Estimates CPU cycles per ParallelFor work unit for a circuit batch whose
// largest circuit has `max_num_qubits` qubits, without shift or signed-int64
// overflow.
inline int64_t GetNumCycles(const uint64_t max_num_qubits) {
  const uint64_t clamped = std::min(max_num_qubits, kMaxCycleEstimateQubits);
  return int64_t{200} * (int64_t{1} << clamped);
}

// Validates that every entry in `num_qubits` is in [0, `max_allowed`] and
// writes the maximum (0 when `num_qubits` is empty) to `*max_num_qubits`.
absl::Status GetMaxNumQubits(const std::vector<int>& num_qubits,
                             uint64_t max_allowed, uint64_t* max_num_qubits);

// Validates that every entry in `num_qubits` is at least 4 (MPS 1D minimum),
// checks that `bond_dim` >= 4 and `(num_qubits, bond_dim)` do not overflow
// qsim::mps::MPSStateSpace's 32-bit unsigned size arithmetic, and writes the
// maximum qubit count (0 when `num_qubits` is empty) to `*max_num_qubits`.
absl::Status ValidateMPS1DQubitsAndBondDim(const std::vector<int>& num_qubits,
                                           int bond_dim,
                                           uint64_t* max_num_qubits);

// Returns ResourceExhausted if any qsim state/unitary allocation returned null.
template <typename... Buffers>
inline absl::Status CheckAllocated(const uint64_t num_qubits,
                                   const Buffers&... buffers) {
  static_assert(sizeof...(Buffers) > 0, "At least one buffer is required.");
  if (TF_PREDICT_FALSE(((buffers.get() == nullptr) || ...))) {
    return internal::BufferAllocationError(num_qubits);
  }
  return absl::OkStatus();
}

// Reallocates `buffers` via `create_buffer(nq)` when `nq > *largest_nq`,
// stopping on the first null allocation and returning ResourceExhausted.
// Updates `*largest_nq = nq` on success and resets `*largest_nq = 0` on
// failure so the invariant (`*largest_nq == 0` or every buffer holds at least
// `*largest_nq` qubits) is always preserved.
template <typename Factory, typename... Buffers>
inline absl::Status GrowBuffers(const uint64_t nq, uint64_t* largest_nq,
                                const Factory& create_buffer,
                                Buffers*... buffers) {
  static_assert(sizeof...(Buffers) > 0, "At least one buffer is required.");
  DCHECK(largest_nq != nullptr);
  if (nq <= *largest_nq) {
    return absl::OkStatus();
  }
  const bool ok =
      (internal::TryReallocateBuffer(nq, create_buffer, buffers) && ...);
  if (TF_PREDICT_FALSE(!ok)) {
    *largest_nq = 0;
    return internal::BufferAllocationError(nq);
  }
  *largest_nq = nq;
  return absl::OkStatus();
}

// Convenience wrapper around GrowBuffers for `ss.Create(nq)` state spaces.
template <typename StateSpace, typename... Buffers>
inline absl::Status GrowStateBuffers(const uint64_t nq, uint64_t* largest_nq,
                                     const StateSpace& ss,
                                     Buffers*... buffers) {
  return GrowBuffers(
      nq, largest_nq, [&ss](uint64_t n) { return ss.Create(n); }, buffers...);
}

}  // namespace tfq

#endif  // TFQ_CORE_OPS_TFQ_SIMULATE_UTILS_H_
