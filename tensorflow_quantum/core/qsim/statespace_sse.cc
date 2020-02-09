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

float StateSpaceSSE::GetRealInnerProduct(const State& a, const State& b) const {
  uint64_t size2 = (size_ / 2) / 4;
  __m128d expv_0 = _mm_setzero_pd();
  __m128d expv_1 = _mm_setzero_pd();
  __m128d temp = _mm_setzero_pd();
  __m128d rs_0, rs_1, is_0, is_1;

  auto statea = RawData(a);
  auto stateb = RawData(b);

  //#pragma omp parallel for num_threads(num_threads_)
  // Currently not a thread safe implementation of inner product!
  for (uint64_t i = 0; i < size2; ++i) {
    // rs = _mm256_cvtps_pd(_mm_load_ps(statea + 8 * i));
    rs_0 = _mm_cvtps_pd(_mm_load_ps(statea + 8 * i));
    rs_1 = _mm_cvtps_pd(_mm_load_ps(statea + 8 * i + 2));

    // is = _mm256_cvtps_pd(_mm_load_ps(stateb + 8 * i));
    is_0 = _mm_cvtps_pd(_mm_load_ps(stateb + 8 * i));
    is_1 = _mm_cvtps_pd(_mm_load_ps(stateb + 8 * i + 2));

    // expv = _mm256_fmadd_pd(rs, is, expv);
    temp = _mm_mul_pd(rs_0, is_0);
    expv_0 = _mm_add_pd(expv_0, temp);
    temp = _mm_mul_pd(rs_1, is_1);
    expv_1 = _mm_add_pd(expv_1, temp);

    // rs = _mm256_cvtps_pd(_mm_load_ps(statea + 8 * i + 4));
    rs_0 = _mm_cvtps_pd(_mm_load_ps(statea + 8 * i + 4));
    rs_1 = _mm_cvtps_pd(_mm_load_ps(statea + 8 * i + 6));

    // is = _mm256_cvtps_pd(_mm_load_ps(stateb + 8 * i + 4));
    is_0 = _mm_cvtps_pd(_mm_load_ps(stateb + 8 * i + 4));
    is_1 = _mm_cvtps_pd(_mm_load_ps(stateb + 8 * i + 6));

    // expv = _mm256_fmadd_pd(rs, is, expv);
    temp = _mm_mul_pd(rs_0, is_0);
    expv_0 = _mm_add_pd(expv_0, temp);
    temp = _mm_mul_pd(rs_1, is_1);
    expv_1 = _mm_add_pd(expv_1, temp);
  }
  double buffer[4];
  _mm_storeu_pd(buffer, expv_0);
  _mm_storeu_pd(buffer + 2, expv_1);
  return (float)(buffer[0] + buffer[1] + buffer[2] + buffer[3]);
}

std::complex<float> StateSpaceSSE::GetAmpl(const State& state,
                                           const uint64_t i) const {
  uint64_t p = (16 * (i / 8)) + (i % 8);
  return std::complex<float>(state.get()[p], state.get()[p + 8]);
}

void StateSpaceSSE::SetAmpl(State* state, const uint64_t i,
                            const std::complex<float>& val) const {
  uint64_t p = (16 * (i / 8)) + (i % 8);
  state->get()[p] = val.real();
  state->get()[p + 8] = val.imag();
}

}  // namespace qsim
}  // namespace tfq

#endif
