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

#ifdef __SSE4_1__

#include "tensorflow_quantum/core/qsim/statespace_sse.h"

#include <immintrin.h>
#include <smmintrin.h>
#include <stdlib.h>

#include <complex>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <stdexcept>

#include "tensorflow_quantum/core/qsim/statespace.h"

namespace tfq {
namespace qsim {

StateSpaceSSE::StateSpaceSSE(const unsigned int num_qubits,
                             const unsigned int num_threads)
    : StateSpace(num_qubits, num_threads) {}

void StateSpaceSSE::CopyState(const State& src, State* dest) const {
  // TODO (zaqwerty): look into whether or not this could be made faster
  //  with sse instructions.
  for (uint64_t i = 0; i < size_; ++i) {
    dest->get()[i] = src.get()[i];
  }
}

void StateSpaceSSE::SetStateZero(State* state) const {
  uint64_t size2 = (size_ / 2) / 8;

  //__m256 val0 = _mm256_setzero_ps();
  __m128 val0 = _mm_setzero_ps();

  auto data = state->get();

  for (uint64_t i = 0; i < size2; ++i) {
    //_mm256_store_ps(state.get() + 16 * i, val0);
    //_mm256_store_ps(state.get() + 16 * i + 8, val0);
    _mm_store_ps(data + 16 * i, val0);
    _mm_store_ps(data + 16 * i + 4, val0);
    _mm_store_ps(data + 16 * i + 8, val0);
    _mm_store_ps(data + 16 * i + 12, val0);
  }

  state->get()[0] = 1;
}

}  // namespace qsim
}  // namespace tfq

#endif
