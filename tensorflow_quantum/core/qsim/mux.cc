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

#include "tensorflow_quantum/core/qsim/mux.h"

#ifdef __AVX2__
#include "tensorflow_quantum/core/qsim/state_space_avx.h"
#elif __SSE4_1__
#include "tensorflow_quantum/core/qsim/state_space_sse.h"
#endif

#include <memory>

#include "absl/memory/memory.h"
#include "tensorflow_quantum/core/qsim/state_space.h"
#include "tensorflow_quantum/core/qsim/state_space_slow.h"

namespace tfq {
namespace qsim {

std::unique_ptr<StateSpace> GetStateSpace(const uint64_t num_qubits,
                                          const uint64_t num_threads) {
  if (num_qubits <= 3) {
    return absl::make_unique<StateSpaceSlow>(num_qubits, num_threads);
  }

#ifdef __AVX2__
  return absl::make_unique<StateSpaceAVX>(num_qubits, num_threads);
#elif __SSE4_1__
  return absl::make_unique<StateSpaceSSE>(num_qubits, num_threads);
#else
  return absl::make_unique<StateSpaceSlow>(num_qubits, num_threads);
#endif
}

}  // namespace qsim
}  // namespace tfq
