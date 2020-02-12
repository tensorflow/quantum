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

#include "tensorflow_quantum/core/qsim/simulator2_sse.h"

#include <immintrin.h>
#include <smmintrin.h>

#include <cmath>
#include <cstdint>

#include "tensorflow/core/platform/logging.h"
#include "tensorflow_quantum/core/qsim/simulator.h"

namespace tfq {
namespace qsim {

Simulator2SSE::Simulator2SSE(const unsigned int num_qubits,
                             const unsigned int num_threads)
    : Simulator(num_qubits, num_threads) {}

void Simulator2SSE::ApplyGate2(const unsigned int q0, const unsigned int q1,
                               const float* matrix, State* state) const {
  // Assume q0 < q1.

  if (q0 > 2) {
    ApplyGate2HH(q0, q1, matrix, state);
  } else if (q1 > 2) {
    ApplyGate2HL(q0, q1, matrix, state);
  } else {
    ApplyGate2LL(q0, q1, matrix, state);
  }
}

void Simulator2SSE::ApplyGate1(const float* matrix, State* state) const {
  CHECK(false) << "SSE simulator doesn't support small circuits.";
}

void Simulator2SSE::CopyState(const State& src, State* dest) const {
  // TODO (zaqwerty): look into whether or not this could be made faster
  //  with sse instructions.
  for (uint64_t i = 0; i < size_; ++i) {
    dest->get()[i] = src.get()[i];
  }
}

void Simulator2SSE::SetStateZero(State* state) const {
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

float Simulator2SSE::GetRealInnerProduct(const State& a, const State& b) const {
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

std::complex<float> Simulator2SSE::GetAmpl(const State& state,
                                           const uint64_t i) const {
  uint64_t p = (16 * (i / 8)) + (i % 8);
  return std::complex<float>(state.get()[p], state.get()[p + 8]);
}

void Simulator2SSE::SetAmpl(State* state, const uint64_t i,
                            const std::complex<float>& val) const {
  uint64_t p = (16 * (i / 8)) + (i % 8);
  state->get()[p] = val.real();
  state->get()[p + 8] = val.imag();
}

void Simulator2SSE::ApplyGate2HH(const unsigned int q0, const unsigned int q1,
                                 const float* matrix, State* state) const {
  uint64_t sizei = uint64_t(1) << (num_qubits_ + 1);
  uint64_t sizej = uint64_t(1) << (q1 + 1);
  uint64_t sizek = uint64_t(1) << (q0 + 1);

  auto rstate = RawData(state);

  for (uint64_t i = 0; i < sizei; i += 2 * sizej) {
    for (uint64_t j = 0; j < sizej; j += 2 * sizek) {
      for (uint64_t k = 0; k < sizek; k += 16) {
        uint64_t si = i | j | k;

        //__m256 r0, i0, r1, i1, r2, i2, r3, i3, ru, iu, rn, in;
        __m128 r0_0, i0_0, r1_0, i1_0, r2_0, i2_0, r3_0, i3_0, ru_0, iu_0, rn_0,
            in_0;
        __m128 r0_1, i0_1, r1_1, i1_1, r2_1, i2_1, r3_1, i3_1, ru_1, iu_1, rn_1,
            in_1;

        // use this for all fmadd and fnmadd replacements.
        __m128 temp;

        uint64_t p = si;
        // r0 = _mm256_load_ps(rstate + p);
        r0_0 = _mm_load_ps(rstate + p);
        r0_1 = _mm_load_ps(rstate + p + 4);

        // i0 = _mm256_load_ps(rstate + p + 8);
        i0_0 = _mm_load_ps(rstate + p + 8);
        i0_1 = _mm_load_ps(rstate + p + 12);

        // Can we get rid of ru duplicates ?
        // ru = _mm256_set1_ps(matrix[0]);
        ru_0 = _mm_set1_ps(matrix[0]);
        ru_1 = _mm_set1_ps(matrix[0]);

        // iu = _mm256_set1_ps(matrix[1]);
        iu_0 = _mm_set1_ps(matrix[1]);
        iu_1 = _mm_set1_ps(matrix[1]);

        // rn = _mm256_mul_ps(r0, ru);
        rn_0 = _mm_mul_ps(r0_0, ru_0);
        rn_1 = _mm_mul_ps(r0_1, ru_1);

        // in = _mm256_mul_ps(r0, iu);
        in_0 = _mm_mul_ps(r0_0, iu_0);
        in_1 = _mm_mul_ps(r0_1, iu_1);

        // rn = _mm256_fnmadd_ps(i0, iu, rn);
        temp = _mm_mul_ps(i0_0, iu_0);
        rn_0 = _mm_sub_ps(rn_0, temp);
        temp = _mm_mul_ps(i0_1, iu_1);
        rn_1 = _mm_sub_ps(rn_1, temp);

        // in = _mm256_fmadd_ps(i0, ru, in);
        temp = _mm_mul_ps(i0_0, ru_0);
        in_0 = _mm_add_ps(in_0, temp);
        temp = _mm_mul_ps(i0_1, ru_1);
        in_1 = _mm_add_ps(in_1, temp);

        p = si | sizek;

        // r1 = _mm256_load_ps(rstate + p);
        r1_0 = _mm_load_ps(rstate + p);
        r1_1 = _mm_load_ps(rstate + p + 4);

        // i1 = _mm256_load_ps(rstate + p + 8);
        i1_0 = _mm_load_ps(rstate + p + 8);
        i1_1 = _mm_load_ps(rstate + p + 12);

        // ru = _mm256_set1_ps(matrix[2]);
        ru_0 = _mm_set1_ps(matrix[2]);
        ru_1 = _mm_set1_ps(matrix[2]);

        // iu = _mm256_set1_ps(matrix[3]);
        iu_0 = _mm_set1_ps(matrix[3]);
        iu_1 = _mm_set1_ps(matrix[3]);

        // rn = _mm256_fmadd_ps(r1, ru, rn);
        temp = _mm_mul_ps(r1_0, ru_0);
        rn_0 = _mm_add_ps(rn_0, temp);
        temp = _mm_mul_ps(r1_1, ru_1);
        rn_1 = _mm_add_ps(rn_1, temp);

        // in = _mm256_fmadd_ps(r1, iu, in);
        temp = _mm_mul_ps(r1_0, iu_0);
        in_0 = _mm_add_ps(in_0, temp);
        temp = _mm_mul_ps(r1_1, iu_1);
        in_1 = _mm_add_ps(in_1, temp);

        // rn = _mm256_fnmadd_ps(i1, iu, rn);
        temp = _mm_mul_ps(i1_0, iu_0);
        rn_0 = _mm_sub_ps(rn_0, temp);
        temp = _mm_mul_ps(i1_1, iu_1);
        rn_1 = _mm_sub_ps(rn_1, temp);

        // in = _mm256_fmadd_ps(i1, ru, in);
        temp = _mm_mul_ps(i1_0, ru_0);
        in_0 = _mm_add_ps(in_0, temp);
        temp = _mm_mul_ps(i1_1, ru_1);
        in_1 = _mm_add_ps(in_1, temp);

        p = si | sizej;
        // r2 = _mm256_load_ps(rstate + p);
        r2_0 = _mm_load_ps(rstate + p);
        r2_1 = _mm_load_ps(rstate + p + 4);

        // i2 = _mm256_load_ps(rstate + p + 8);
        i2_0 = _mm_load_ps(rstate + p + 8);
        i2_1 = _mm_load_ps(rstate + p + 12);

        // ru = _mm256_set1_ps(matrix[4]);
        ru_0 = _mm_set1_ps(matrix[4]);
        ru_1 = _mm_set1_ps(matrix[4]);

        // iu = _mm256_set1_ps(matrix[5]);
        iu_0 = _mm_set1_ps(matrix[5]);
        iu_1 = _mm_set1_ps(matrix[5]);

        // rn = _mm256_fmadd_ps(r2, ru, rn);
        temp = _mm_mul_ps(r2_0, ru_0);
        rn_0 = _mm_add_ps(rn_0, temp);
        temp = _mm_mul_ps(r2_1, ru_1);
        rn_1 = _mm_add_ps(rn_1, temp);

        // in = _mm256_fmadd_ps(r2, iu, in);
        temp = _mm_mul_ps(r2_0, iu_0);
        in_0 = _mm_add_ps(in_0, temp);
        temp = _mm_mul_ps(r2_1, iu_1);
        in_1 = _mm_add_ps(in_1, temp);

        // rn = _mm256_fnmadd_ps(i2, iu, rn);
        temp = _mm_mul_ps(i2_0, iu_0);
        rn_0 = _mm_sub_ps(rn_0, temp);
        temp = _mm_mul_ps(i2_1, iu_1);
        rn_1 = _mm_sub_ps(rn_1, temp);

        // in = _mm256_fmadd_ps(i2, ru, in);
        temp = _mm_mul_ps(i2_0, ru_0);
        in_0 = _mm_add_ps(in_0, temp);
        temp = _mm_mul_ps(i2_1, ru_1);
        in_1 = _mm_add_ps(in_1, temp);

        p |= sizek;

        // r3 = _mm256_load_ps(rstate + p);
        r3_0 = _mm_load_ps(rstate + p);
        r3_1 = _mm_load_ps(rstate + p + 4);

        // i3 = _mm256_load_ps(rstate + p + 8);
        i3_0 = _mm_load_ps(rstate + p + 8);
        i3_1 = _mm_load_ps(rstate + p + 12);

        // ru = _mm256_set1_ps(matrix[6]);
        ru_0 = _mm_set1_ps(matrix[6]);
        ru_1 = _mm_set1_ps(matrix[6]);

        // iu = _mm256_set1_ps(matrix[7]);
        iu_0 = _mm_set1_ps(matrix[7]);
        iu_1 = _mm_set1_ps(matrix[7]);

        // rn = _mm256_fmadd_ps(r3, ru, rn);
        temp = _mm_mul_ps(r3_0, ru_0);
        rn_0 = _mm_add_ps(rn_0, temp);
        temp = _mm_mul_ps(r3_1, ru_1);
        rn_1 = _mm_add_ps(rn_1, temp);

        // in = _mm256_fmadd_ps(r3, iu, in);
        temp = _mm_mul_ps(r3_0, iu_0);
        in_0 = _mm_add_ps(in_0, temp);
        temp = _mm_mul_ps(r3_1, iu_1);
        in_1 = _mm_add_ps(in_1, temp);

        // rn = _mm256_fnmadd_ps(i3, iu, rn);
        temp = _mm_mul_ps(i3_0, iu_0);
        rn_0 = _mm_sub_ps(rn_0, temp);
        temp = _mm_mul_ps(i3_1, iu_1);
        rn_1 = _mm_sub_ps(rn_1, temp);

        // in = _mm256_fmadd_ps(i3, ru, in);
        temp = _mm_mul_ps(i3_0, ru_0);
        in_0 = _mm_add_ps(in_0, temp);
        temp = _mm_mul_ps(i3_1, ru_1);
        in_1 = _mm_add_ps(in_1, temp);

        p = si;

        //_mm256_store_ps(rstate + p, rn);
        _mm_store_ps(rstate + p, rn_0);
        _mm_store_ps(rstate + p + 4, rn_1);

        //_mm256_store_ps(rstate + p + 8, in);
        _mm_store_ps(rstate + p + 8, in_0);
        _mm_store_ps(rstate + p + 12, in_1);

        // ru = _mm256_set1_ps(matrix[8]);
        ru_0 = _mm_set1_ps(matrix[8]);
        ru_1 = _mm_set1_ps(matrix[8]);

        // iu = _mm256_set1_ps(matrix[9]);
        iu_0 = _mm_set1_ps(matrix[9]);
        iu_1 = _mm_set1_ps(matrix[9]);

        // rn = _mm256_mul_ps(r0, ru);
        rn_0 = _mm_mul_ps(r0_0, ru_0);
        rn_1 = _mm_mul_ps(r0_1, ru_1);

        // in = _mm256_mul_ps(r0, iu);
        in_0 = _mm_mul_ps(r0_0, iu_0);
        in_1 = _mm_mul_ps(r0_1, iu_1);

        // rn = _mm256_fnmadd_ps(i0, iu, rn);
        temp = _mm_mul_ps(i0_0, iu_0);
        rn_0 = _mm_sub_ps(rn_0, temp);
        temp = _mm_mul_ps(i0_1, iu_1);
        rn_1 = _mm_sub_ps(rn_1, temp);

        // in = _mm256_fmadd_ps(i0, ru, in);
        temp = _mm_mul_ps(i0_0, ru_0);
        in_0 = _mm_add_ps(in_0, temp);
        temp = _mm_mul_ps(i0_1, ru_1);
        in_1 = _mm_add_ps(in_1, temp);

        // ru = _mm256_set1_ps(matrix[10]);
        ru_0 = _mm_set1_ps(matrix[10]);
        ru_1 = _mm_set1_ps(matrix[10]);

        // iu = _mm256_set1_ps(matrix[11]);
        iu_0 = _mm_set1_ps(matrix[11]);
        iu_1 = _mm_set1_ps(matrix[11]);

        // rn = _mm256_fmadd_ps(r1, ru, rn);
        temp = _mm_mul_ps(r1_0, ru_0);
        rn_0 = _mm_add_ps(rn_0, temp);
        temp = _mm_mul_ps(r1_1, ru_1);
        rn_1 = _mm_add_ps(rn_1, temp);

        // in = _mm256_fmadd_ps(r1, iu, in);
        temp = _mm_mul_ps(r1_0, iu_0);
        in_0 = _mm_add_ps(in_0, temp);
        temp = _mm_mul_ps(r1_1, iu_1);
        in_1 = _mm_add_ps(in_1, temp);

        // rn = _mm256_fnmadd_ps(i1, iu, rn);
        temp = _mm_mul_ps(i1_0, iu_0);
        rn_0 = _mm_sub_ps(rn_0, temp);
        temp = _mm_mul_ps(i1_1, iu_1);
        rn_1 = _mm_sub_ps(rn_1, temp);

        // in = _mm256_fmadd_ps(i1, ru, in);
        temp = _mm_mul_ps(i1_0, ru_0);
        in_0 = _mm_add_ps(in_0, temp);
        temp = _mm_mul_ps(i1_1, ru_1);
        in_1 = _mm_add_ps(in_1, temp);

        // ru = _mm256_set1_ps(matrix[12]);
        ru_0 = _mm_set1_ps(matrix[12]);
        ru_1 = _mm_set1_ps(matrix[12]);

        // iu = _mm256_set1_ps(matrix[13]);
        iu_0 = _mm_set1_ps(matrix[13]);
        iu_1 = _mm_set1_ps(matrix[13]);

        // rn = _mm256_fmadd_ps(r2, ru, rn);
        temp = _mm_mul_ps(r2_0, ru_0);
        rn_0 = _mm_add_ps(rn_0, temp);
        temp = _mm_mul_ps(r2_1, ru_1);
        rn_1 = _mm_add_ps(rn_1, temp);

        // in = _mm256_fmadd_ps(r2, iu, in);
        temp = _mm_mul_ps(r2_0, iu_0);
        in_0 = _mm_add_ps(in_0, temp);
        temp = _mm_mul_ps(r2_1, iu_1);
        in_1 = _mm_add_ps(in_1, temp);

        // rn = _mm256_fnmadd_ps(i2, iu, rn);
        temp = _mm_mul_ps(i2_0, iu_0);
        rn_0 = _mm_sub_ps(rn_0, temp);
        temp = _mm_mul_ps(i2_1, iu_1);
        rn_1 = _mm_sub_ps(rn_1, temp);

        // in = _mm256_fmadd_ps(i2, ru, in);
        temp = _mm_mul_ps(i2_0, ru_0);
        in_0 = _mm_add_ps(in_0, temp);
        temp = _mm_mul_ps(i2_1, ru_1);
        in_1 = _mm_add_ps(in_1, temp);

        // ru = _mm256_set1_ps(matrix[14]);
        ru_0 = _mm_set1_ps(matrix[14]);
        ru_1 = _mm_set1_ps(matrix[14]);

        // iu = _mm256_set1_ps(matrix[15]);
        iu_0 = _mm_set1_ps(matrix[15]);
        iu_1 = _mm_set1_ps(matrix[15]);

        // rn = _mm256_fmadd_ps(r3, ru, rn);
        temp = _mm_mul_ps(r3_0, ru_0);
        rn_0 = _mm_add_ps(rn_0, temp);
        temp = _mm_mul_ps(r3_1, ru_1);
        rn_1 = _mm_add_ps(rn_1, temp);

        // in = _mm256_fmadd_ps(r3, iu, in);
        temp = _mm_mul_ps(r3_0, iu_0);
        in_0 = _mm_add_ps(in_0, temp);
        temp = _mm_mul_ps(r3_1, iu_1);
        in_1 = _mm_add_ps(in_1, temp);

        // rn = _mm256_fnmadd_ps(i3, iu, rn);
        temp = _mm_mul_ps(i3_0, iu_0);
        rn_0 = _mm_sub_ps(rn_0, temp);
        temp = _mm_mul_ps(i3_1, iu_1);
        rn_1 = _mm_sub_ps(rn_1, temp);

        // in = _mm256_fmadd_ps(i3, ru, in);
        temp = _mm_mul_ps(i3_0, ru_0);
        in_0 = _mm_add_ps(in_0, temp);
        temp = _mm_mul_ps(i3_1, ru_1);
        in_1 = _mm_add_ps(in_1, temp);

        p = si | sizek;
        //_mm256_store_ps(rstate + p, rn);
        _mm_store_ps(rstate + p, rn_0);
        _mm_store_ps(rstate + p + 4, rn_1);

        //_mm256_store_ps(rstate + p + 8, in);
        _mm_store_ps(rstate + p + 8, in_0);
        _mm_store_ps(rstate + p + 12, in_1);

        // ru = _mm256_set1_ps(matrix[16]);
        ru_0 = _mm_set1_ps(matrix[16]);
        ru_1 = _mm_set1_ps(matrix[16]);

        // iu = _mm256_set1_ps(matrix[17]);
        iu_0 = _mm_set1_ps(matrix[17]);
        iu_1 = _mm_set1_ps(matrix[17]);

        // rn = _mm256_mul_ps(r0, ru);
        rn_0 = _mm_mul_ps(r0_0, ru_0);
        rn_1 = _mm_mul_ps(r0_1, ru_1);

        // in = _mm256_mul_ps(r0, iu);
        in_0 = _mm_mul_ps(r0_0, iu_0);
        in_1 = _mm_mul_ps(r0_1, iu_1);

        // rn = _mm256_fnmadd_ps(i0, iu, rn);
        temp = _mm_mul_ps(i0_0, iu_0);
        rn_0 = _mm_sub_ps(rn_0, temp);
        temp = _mm_mul_ps(i0_1, iu_1);
        rn_1 = _mm_sub_ps(rn_1, temp);

        // in = _mm256_fmadd_ps(i0, ru, in);
        temp = _mm_mul_ps(i0_0, ru_0);
        in_0 = _mm_add_ps(in_0, temp);
        temp = _mm_mul_ps(i0_1, ru_1);
        in_1 = _mm_add_ps(in_1, temp);

        // ru = _mm256_set1_ps(matrix[18]);
        ru_0 = _mm_set1_ps(matrix[18]);
        ru_1 = _mm_set1_ps(matrix[18]);

        // iu = _mm256_set1_ps(matrix[19]);
        iu_0 = _mm_set1_ps(matrix[19]);
        iu_1 = _mm_set1_ps(matrix[19]);

        // rn = _mm256_fmadd_ps(r1, ru, rn);
        temp = _mm_mul_ps(r1_0, ru_0);
        rn_0 = _mm_add_ps(rn_0, temp);
        temp = _mm_mul_ps(r1_1, ru_1);
        rn_1 = _mm_add_ps(rn_1, temp);

        // in = _mm256_fmadd_ps(r1, iu, in);
        temp = _mm_mul_ps(r1_0, iu_0);
        in_0 = _mm_add_ps(in_0, temp);
        temp = _mm_mul_ps(r1_1, iu_1);
        in_1 = _mm_add_ps(in_1, temp);

        // rn = _mm256_fnmadd_ps(i1, iu, rn);
        temp = _mm_mul_ps(i1_0, iu_0);
        rn_0 = _mm_sub_ps(rn_0, temp);
        temp = _mm_mul_ps(i1_1, iu_1);
        rn_1 = _mm_sub_ps(rn_1, temp);

        // in = _mm256_fmadd_ps(i1, ru, in);
        temp = _mm_mul_ps(i1_0, ru_0);
        in_0 = _mm_add_ps(in_0, temp);
        temp = _mm_mul_ps(i1_1, ru_1);
        in_1 = _mm_add_ps(in_1, temp);

        // ru = _mm256_set1_ps(matrix[20]);
        ru_0 = _mm_set1_ps(matrix[20]);
        ru_1 = _mm_set1_ps(matrix[20]);

        // iu = _mm256_set1_ps(matrix[21]);
        iu_0 = _mm_set1_ps(matrix[21]);
        iu_1 = _mm_set1_ps(matrix[21]);

        // rn = _mm256_fmadd_ps(r2, ru, rn);
        temp = _mm_mul_ps(r2_0, ru_0);
        rn_0 = _mm_add_ps(rn_0, temp);
        temp = _mm_mul_ps(r2_1, ru_1);
        rn_1 = _mm_add_ps(rn_1, temp);

        // in = _mm256_fmadd_ps(r2, iu, in);
        temp = _mm_mul_ps(r2_0, iu_0);
        in_0 = _mm_add_ps(in_0, temp);
        temp = _mm_mul_ps(r2_1, iu_1);
        in_1 = _mm_add_ps(in_1, temp);

        // rn = _mm256_fnmadd_ps(i2, iu, rn);
        temp = _mm_mul_ps(i2_0, iu_0);
        rn_0 = _mm_sub_ps(rn_0, temp);
        temp = _mm_mul_ps(i2_1, iu_1);
        rn_1 = _mm_sub_ps(rn_1, temp);

        // in = _mm256_fmadd_ps(i2, ru, in);
        temp = _mm_mul_ps(i2_0, ru_0);
        in_0 = _mm_add_ps(in_0, temp);
        temp = _mm_mul_ps(i2_1, ru_1);
        in_1 = _mm_add_ps(in_1, temp);

        // ru = _mm256_set1_ps(matrix[22]);
        ru_0 = _mm_set1_ps(matrix[22]);
        ru_1 = _mm_set1_ps(matrix[22]);

        // iu = _mm256_set1_ps(matrix[23]);
        iu_0 = _mm_set1_ps(matrix[23]);
        iu_1 = _mm_set1_ps(matrix[23]);

        // rn = _mm256_fmadd_ps(r3, ru, rn);
        temp = _mm_mul_ps(r3_0, ru_0);
        rn_0 = _mm_add_ps(rn_0, temp);
        temp = _mm_mul_ps(r3_1, ru_1);
        rn_1 = _mm_add_ps(rn_1, temp);

        // in = _mm256_fmadd_ps(r3, iu, in);
        temp = _mm_mul_ps(r3_0, iu_0);
        in_0 = _mm_add_ps(in_0, temp);
        temp = _mm_mul_ps(r3_1, iu_1);
        in_1 = _mm_add_ps(in_1, temp);

        // rn = _mm256_fnmadd_ps(i3, iu, rn);
        temp = _mm_mul_ps(i3_0, iu_0);
        rn_0 = _mm_sub_ps(rn_0, temp);
        temp = _mm_mul_ps(i3_1, iu_1);
        rn_1 = _mm_sub_ps(rn_1, temp);

        // in = _mm256_fmadd_ps(i3, ru, in);
        temp = _mm_mul_ps(i3_0, ru_0);
        in_0 = _mm_add_ps(in_0, temp);
        temp = _mm_mul_ps(i3_1, ru_1);
        in_1 = _mm_add_ps(in_1, temp);

        p = si | sizej;
        //_mm256_store_ps(rstate + p, rn);
        _mm_store_ps(rstate + p, rn_0);
        _mm_store_ps(rstate + p + 4, rn_1);

        //_mm256_store_ps(rstate + p + 8, in);
        _mm_store_ps(rstate + p + 8, in_0);
        _mm_store_ps(rstate + p + 12, in_1);

        // ru = _mm256_set1_ps(matrix[24]);
        ru_0 = _mm_set1_ps(matrix[24]);
        ru_1 = _mm_set1_ps(matrix[24]);

        // iu = _mm256_set1_ps(matrix[25]);
        iu_0 = _mm_set1_ps(matrix[25]);
        iu_1 = _mm_set1_ps(matrix[25]);

        // rn = _mm256_mul_ps(r0, ru);
        rn_0 = _mm_mul_ps(r0_0, ru_0);
        rn_1 = _mm_mul_ps(r0_1, ru_1);

        // in = _mm256_mul_ps(r0, iu);
        in_0 = _mm_mul_ps(r0_0, iu_0);
        in_1 = _mm_mul_ps(r0_1, iu_1);

        // rn = _mm256_fnmadd_ps(i0, iu, rn);
        temp = _mm_mul_ps(i0_0, iu_0);
        rn_0 = _mm_sub_ps(rn_0, temp);
        temp = _mm_mul_ps(i0_1, iu_1);
        rn_1 = _mm_sub_ps(rn_1, temp);

        // in = _mm256_fmadd_ps(i0, ru, in);
        temp = _mm_mul_ps(i0_0, ru_0);
        in_0 = _mm_add_ps(in_0, temp);
        temp = _mm_mul_ps(i0_1, ru_1);
        in_1 = _mm_add_ps(in_1, temp);

        // ru = _mm256_set1_ps(matrix[26]);
        ru_0 = _mm_set1_ps(matrix[26]);
        ru_1 = _mm_set1_ps(matrix[26]);

        // iu = _mm256_set1_ps(matrix[27]);
        iu_0 = _mm_set1_ps(matrix[27]);
        iu_1 = _mm_set1_ps(matrix[27]);

        // rn = _mm256_fmadd_ps(r1, ru, rn);
        temp = _mm_mul_ps(r1_0, ru_0);
        rn_0 = _mm_add_ps(rn_0, temp);
        temp = _mm_mul_ps(r1_1, ru_1);
        rn_1 = _mm_add_ps(rn_1, temp);

        // in = _mm256_fmadd_ps(r1, iu, in);
        temp = _mm_mul_ps(r1_0, iu_0);
        in_0 = _mm_add_ps(in_0, temp);
        temp = _mm_mul_ps(r1_1, iu_1);
        in_1 = _mm_add_ps(in_1, temp);

        // rn = _mm256_fnmadd_ps(i1, iu, rn);
        temp = _mm_mul_ps(i1_0, iu_0);
        rn_0 = _mm_sub_ps(rn_0, temp);
        temp = _mm_mul_ps(i1_1, iu_1);
        rn_1 = _mm_sub_ps(rn_1, temp);

        // in = _mm256_fmadd_ps(i1, ru, in);
        temp = _mm_mul_ps(i1_0, ru_0);
        in_0 = _mm_add_ps(in_0, temp);
        temp = _mm_mul_ps(i1_1, ru_1);
        in_1 = _mm_add_ps(in_1, temp);

        // ru = _mm256_set1_ps(matrix[28]);
        ru_0 = _mm_set1_ps(matrix[28]);
        ru_1 = _mm_set1_ps(matrix[28]);

        // iu = _mm256_set1_ps(matrix[29]);
        iu_0 = _mm_set1_ps(matrix[29]);
        iu_1 = _mm_set1_ps(matrix[29]);

        // rn = _mm256_fmadd_ps(r2, ru, rn);
        temp = _mm_mul_ps(r2_0, ru_0);
        rn_0 = _mm_add_ps(rn_0, temp);
        temp = _mm_mul_ps(r2_1, ru_1);
        rn_1 = _mm_add_ps(rn_1, temp);

        // in = _mm256_fmadd_ps(r2, iu, in);
        temp = _mm_mul_ps(r2_0, iu_0);
        in_0 = _mm_add_ps(in_0, temp);
        temp = _mm_mul_ps(r2_1, iu_1);
        in_1 = _mm_add_ps(in_1, temp);

        // rn = _mm256_fnmadd_ps(i2, iu, rn);
        temp = _mm_mul_ps(i2_0, iu_0);
        rn_0 = _mm_sub_ps(rn_0, temp);
        temp = _mm_mul_ps(i2_1, iu_1);
        rn_1 = _mm_sub_ps(rn_1, temp);

        // in = _mm256_fmadd_ps(i2, ru, in);
        temp = _mm_mul_ps(i2_0, ru_0);
        in_0 = _mm_add_ps(in_0, temp);
        temp = _mm_mul_ps(i2_1, ru_1);
        in_1 = _mm_add_ps(in_1, temp);

        // ru = _mm256_set1_ps(matrix[30]);
        ru_0 = _mm_set1_ps(matrix[30]);
        ru_1 = _mm_set1_ps(matrix[30]);

        // iu = _mm256_set1_ps(matrix[31]);
        iu_0 = _mm_set1_ps(matrix[31]);
        iu_1 = _mm_set1_ps(matrix[31]);

        // rn = _mm256_fmadd_ps(r3, ru, rn);
        temp = _mm_mul_ps(r3_0, ru_0);
        rn_0 = _mm_add_ps(rn_0, temp);
        temp = _mm_mul_ps(r3_1, ru_1);
        rn_1 = _mm_add_ps(rn_1, temp);

        // in = _mm256_fmadd_ps(r3, iu, in);
        temp = _mm_mul_ps(r3_0, iu_0);
        in_0 = _mm_add_ps(in_0, temp);
        temp = _mm_mul_ps(r3_1, iu_1);
        in_1 = _mm_add_ps(in_1, temp);

        // rn = _mm256_fnmadd_ps(i3, iu, rn);
        temp = _mm_mul_ps(i3_0, iu_0);
        rn_0 = _mm_sub_ps(rn_0, temp);
        temp = _mm_mul_ps(i3_1, iu_1);
        rn_1 = _mm_sub_ps(rn_1, temp);

        // in = _mm256_fmadd_ps(i3, ru, in);
        temp = _mm_mul_ps(i3_0, ru_0);
        in_0 = _mm_add_ps(in_0, temp);
        temp = _mm_mul_ps(i3_1, ru_1);
        in_1 = _mm_add_ps(in_1, temp);

        p |= sizek;
        //_mm256_store_ps(rstate + p, rn);
        _mm_store_ps(rstate + p, rn_0);
        _mm_store_ps(rstate + p + 4, rn_1);

        //_mm256_store_ps(rstate + p + 8, in);
        _mm_store_ps(rstate + p + 8, in_0);
        _mm_store_ps(rstate + p + 12, in_1);
      }
    }
  }
}

void Simulator2SSE::ApplyGate2LL(const unsigned int q0, const unsigned int q1,
                                 const float* matrix, State* state) const {
  const unsigned int q = q0 + q1;

  //__m256 mb1, mb2, mb3;
  __m128 mb1, mb2, mb3;
  //__m256i ml1, ml2, ml3;

  uint64_t sizei = uint64_t(1) << (num_qubits_ + 1);
  auto rstate = RawData(state);

  switch (q) {
    case 1:
      // ml1 = _mm256_set_epi32(7, 6, 4, 5, 3, 2, 0, 1);
      // ml2 = _mm256_set_epi32(7, 4, 5, 6, 3, 0, 1, 2);
      // ml3 = _mm256_set_epi32(4, 6, 5, 7, 0, 2, 1, 3);
      mb1 = _mm_castsi128_ps(_mm_set_epi32(0, 0, -1, 0));
      mb2 = _mm_castsi128_ps(_mm_set_epi32(0, -1, 0, 0));
      mb3 = _mm_castsi128_ps(_mm_set_epi32(-1, 0, 0, 0));

      // mb2 = _mm256_castsi256_ps(_mm256_set_epi32(0, -1, 0, 0, 0, -1, 0,
      // 0)); mb3 = _mm256_castsi256_ps(_mm256_set_epi32(-1, 0, 0, 0, -1, 0,
      // 0, 0));

      for (uint64_t i = 0; i < sizei; i += 16) {
        //__m256 r0, i0, r1, i1, r2, i2, r3, i3, ru, iu, rn, in, rm, im;
        __m128 r0_0, i0_0, r1_0, i1_0, r2_0, i2_0, r3_0, i3_0, ru_0, iu_0, rn_0,
            in_0, rm_0, im_0;
        __m128 r0_1, i0_1, r1_1, i1_1, r2_1, i2_1, r3_1, i3_1, ru_1, iu_1, rn_1,
            in_1, rm_1, im_1;

        // holder for fnmadd and fmadd.
        __m128 temp;

        auto p = rstate + i;

        // r0 = _mm256_load_ps(p);
        r0_0 = _mm_load_ps(p);
        r0_1 = _mm_load_ps(p + 4);

        // i0 = _mm256_load_ps(p + 8);
        i0_0 = _mm_load_ps(p + 8);
        i0_1 = _mm_load_ps(p + 8 + 4);

        // r1 = _mm256_permutevar8x32_ps(r0, ml1);
        r1_0 = _mm_shuffle_ps(r0_0, r0_0, 225);
        r1_1 = _mm_shuffle_ps(r0_1, r0_1, 225);

        // i1 = _mm256_permutevar8x32_ps(i0, ml1);
        i1_0 = _mm_shuffle_ps(i0_0, i0_0, 225);
        i1_1 = _mm_shuffle_ps(i0_1, i0_1, 225);

        // r2 = _mm256_permutevar8x32_ps(r0, ml2);
        r2_0 = _mm_shuffle_ps(r0_0, r0_0, 198);
        r2_1 = _mm_shuffle_ps(r0_1, r0_1, 198);

        // i2 = _mm256_permutevar8x32_ps(i0, ml2);
        i2_0 = _mm_shuffle_ps(i0_0, i0_0, 198);
        i2_1 = _mm_shuffle_ps(i0_1, i0_1, 198);

        // r3 = _mm256_permutevar8x32_ps(r0, ml3);
        r3_0 = _mm_shuffle_ps(r0_0, r0_0, 39);
        r3_1 = _mm_shuffle_ps(r0_1, r0_1, 39);

        // i3 = _mm256_permutevar8x32_ps(i0, ml3);
        i3_0 = _mm_shuffle_ps(i0_0, i0_0, 39);
        i3_1 = _mm_shuffle_ps(i0_1, i0_1, 39);

        // ru = _mm256_set1_ps(matrix[0]);
        ru_0 = _mm_set1_ps(matrix[0]);
        ru_1 = _mm_set1_ps(matrix[0]);

        // iu = _mm256_set1_ps(matrix[1]);
        iu_0 = _mm_set1_ps(matrix[1]);
        iu_1 = _mm_set1_ps(matrix[1]);

        // rn = _mm256_mul_ps(r0, ru);
        rn_0 = _mm_mul_ps(r0_0, ru_0);
        rn_1 = _mm_mul_ps(r0_1, ru_1);

        // in = _mm256_mul_ps(r0, iu);
        in_0 = _mm_mul_ps(r0_0, iu_0);
        in_1 = _mm_mul_ps(r0_1, iu_1);

        // rn = _mm256_fnmadd_ps(i0, iu, rn);
        temp = _mm_mul_ps(i0_0, iu_0);
        rn_0 = _mm_sub_ps(rn_0, temp);
        temp = _mm_mul_ps(i0_1, iu_1);
        rn_1 = _mm_sub_ps(rn_1, temp);

        // in = _mm256_fmadd_ps(i0, ru, in);
        temp = _mm_mul_ps(i0_0, ru_0);
        in_0 = _mm_add_ps(in_0, temp);
        temp = _mm_mul_ps(i0_1, ru_1);
        in_1 = _mm_add_ps(in_1, temp);

        // ru = _mm256_set1_ps(matrix[2]);
        ru_0 = _mm_set1_ps(matrix[2]);
        ru_1 = _mm_set1_ps(matrix[2]);

        // iu = _mm256_set1_ps(matrix[3]);
        iu_0 = _mm_set1_ps(matrix[3]);
        iu_1 = _mm_set1_ps(matrix[3]);

        // rn = _mm256_fmadd_ps(r1, ru, rn);
        temp = _mm_mul_ps(r1_0, ru_0);
        rn_0 = _mm_add_ps(rn_0, temp);
        temp = _mm_mul_ps(r1_1, ru_1);
        rn_1 = _mm_add_ps(rn_1, temp);

        // in = _mm256_fmadd_ps(r1, iu, in);
        temp = _mm_mul_ps(r1_0, iu_0);
        in_0 = _mm_add_ps(in_0, temp);
        temp = _mm_mul_ps(r1_1, iu_1);
        in_1 = _mm_add_ps(in_1, temp);

        // rn = _mm256_fnmadd_ps(i1, iu, rn);
        temp = _mm_mul_ps(i1_0, iu_0);
        rn_0 = _mm_sub_ps(rn_0, temp);
        temp = _mm_mul_ps(i1_1, iu_1);
        rn_1 = _mm_sub_ps(rn_1, temp);

        // in = _mm256_fmadd_ps(i1, ru, in);
        temp = _mm_mul_ps(i1_0, ru_0);
        in_0 = _mm_add_ps(in_0, temp);
        temp = _mm_mul_ps(i1_1, ru_1);
        in_1 = _mm_add_ps(in_1, temp);

        // ru = _mm256_set1_ps(matrix[4]);
        ru_0 = _mm_set1_ps(matrix[4]);
        ru_1 = _mm_set1_ps(matrix[4]);

        // iu = _mm256_set1_ps(matrix[5]);
        iu_0 = _mm_set1_ps(matrix[5]);
        iu_1 = _mm_set1_ps(matrix[5]);

        // rn = _mm256_fmadd_ps(r2, ru, rn);
        temp = _mm_mul_ps(r2_0, ru_0);
        rn_0 = _mm_add_ps(rn_0, temp);
        temp = _mm_mul_ps(r2_1, ru_1);
        rn_1 = _mm_add_ps(rn_1, temp);

        // in = _mm256_fmadd_ps(r2, iu, in);
        temp = _mm_mul_ps(r2_0, iu_0);
        in_0 = _mm_add_ps(in_0, temp);
        temp = _mm_mul_ps(r2_1, iu_1);
        in_1 = _mm_add_ps(in_1, temp);

        // rn = _mm256_fnmadd_ps(i2, iu, rn);
        temp = _mm_mul_ps(i2_0, iu_0);
        rn_0 = _mm_sub_ps(rn_0, temp);
        temp = _mm_mul_ps(i2_1, iu_1);
        rn_1 = _mm_sub_ps(rn_1, temp);

        // in = _mm256_fmadd_ps(i2, ru, in);
        temp = _mm_mul_ps(i2_0, ru_0);
        in_0 = _mm_add_ps(in_0, temp);
        temp = _mm_mul_ps(i2_1, ru_1);
        in_1 = _mm_add_ps(in_1, temp);

        // ru = _mm256_set1_ps(matrix[6]);
        ru_0 = _mm_set1_ps(matrix[6]);
        ru_1 = _mm_set1_ps(matrix[6]);

        // iu = _mm256_set1_ps(matrix[7]);
        iu_0 = _mm_set1_ps(matrix[7]);
        iu_1 = _mm_set1_ps(matrix[7]);

        // rn = _mm256_fmadd_ps(r3, ru, rn);
        temp = _mm_mul_ps(r3_0, ru_0);
        rn_0 = _mm_add_ps(rn_0, temp);
        temp = _mm_mul_ps(r3_1, ru_1);
        rn_1 = _mm_add_ps(rn_1, temp);

        // in = _mm256_fmadd_ps(r3, iu, in);
        temp = _mm_mul_ps(r3_0, iu_0);
        in_0 = _mm_add_ps(in_0, temp);
        temp = _mm_mul_ps(r3_1, iu_1);
        in_1 = _mm_add_ps(in_1, temp);

        // rn = _mm256_fnmadd_ps(i3, iu, rn);
        temp = _mm_mul_ps(i3_0, iu_0);
        rn_0 = _mm_sub_ps(rn_0, temp);
        temp = _mm_mul_ps(i3_1, iu_1);
        rn_1 = _mm_sub_ps(rn_1, temp);

        // in = _mm256_fmadd_ps(i3, ru, in);
        temp = _mm_mul_ps(i3_0, ru_0);
        in_0 = _mm_add_ps(in_0, temp);
        temp = _mm_mul_ps(i3_1, ru_1);
        in_1 = _mm_add_ps(in_1, temp);

        // ru = _mm256_set1_ps(matrix[8]);
        ru_0 = _mm_set1_ps(matrix[8]);
        ru_1 = _mm_set1_ps(matrix[8]);

        // iu = _mm256_set1_ps(matrix[9]);
        iu_0 = _mm_set1_ps(matrix[9]);
        iu_1 = _mm_set1_ps(matrix[9]);

        // rm = _mm256_mul_ps(r0, ru);
        rm_0 = _mm_mul_ps(r0_0, ru_0);
        rm_1 = _mm_mul_ps(r0_1, ru_1);

        // im = _mm256_mul_ps(r0, iu);
        im_0 = _mm_mul_ps(r0_0, iu_0);
        im_1 = _mm_mul_ps(r0_1, iu_1);

        // rm = _mm256_fnmadd_ps(i0, iu, rm);
        temp = _mm_mul_ps(i0_0, iu_0);
        rm_0 = _mm_sub_ps(rm_0, temp);
        temp = _mm_mul_ps(i0_1, iu_1);
        rm_1 = _mm_sub_ps(rm_1, temp);

        // im = _mm256_fmadd_ps(i0, ru, im);
        temp = _mm_mul_ps(i0_0, ru_0);
        im_0 = _mm_add_ps(im_0, temp);
        temp = _mm_mul_ps(i0_1, ru_1);
        im_1 = _mm_add_ps(im_1, temp);

        // ru = _mm256_set1_ps(matrix[10]);
        ru_0 = _mm_set1_ps(matrix[10]);
        ru_1 = _mm_set1_ps(matrix[10]);

        // iu = _mm256_set1_ps(matrix[11]);
        iu_0 = _mm_set1_ps(matrix[11]);
        iu_1 = _mm_set1_ps(matrix[11]);

        // rm = _mm256_fmadd_ps(r1, ru, rm);
        temp = _mm_mul_ps(r1_0, ru_0);
        rm_0 = _mm_add_ps(rm_0, temp);
        temp = _mm_mul_ps(r1_1, ru_1);
        rm_1 = _mm_add_ps(rm_1, temp);

        // im = _mm256_fmadd_ps(r1, iu, im);
        temp = _mm_mul_ps(r1_0, iu_0);
        im_0 = _mm_add_ps(im_0, temp);
        temp = _mm_mul_ps(r1_1, iu_1);
        im_1 = _mm_add_ps(im_1, temp);

        // rm = _mm256_fnmadd_ps(i1, iu, rm);
        temp = _mm_mul_ps(i1_0, iu_0);
        rm_0 = _mm_sub_ps(rm_0, temp);
        temp = _mm_mul_ps(i1_1, iu_1);
        rm_1 = _mm_sub_ps(rm_1, temp);

        // im = _mm256_fmadd_ps(i1, ru, im);
        temp = _mm_mul_ps(i1_0, ru_0);
        im_0 = _mm_add_ps(im_0, temp);
        temp = _mm_mul_ps(i1_1, ru_1);
        im_1 = _mm_add_ps(im_1, temp);

        // ru = _mm256_set1_ps(matrix[12]);
        ru_0 = _mm_set1_ps(matrix[12]);
        ru_1 = _mm_set1_ps(matrix[12]);

        // iu = _mm256_set1_ps(matrix[13]);
        iu_0 = _mm_set1_ps(matrix[13]);
        iu_1 = _mm_set1_ps(matrix[13]);

        // rm = _mm256_fmadd_ps(r2, ru, rm);
        temp = _mm_mul_ps(r2_0, ru_0);
        rm_0 = _mm_add_ps(rm_0, temp);
        temp = _mm_mul_ps(r2_1, ru_1);
        rm_1 = _mm_add_ps(rm_1, temp);

        // im = _mm256_fmadd_ps(r2, iu, im);
        temp = _mm_mul_ps(r2_0, iu_0);
        im_0 = _mm_add_ps(im_0, temp);
        temp = _mm_mul_ps(r2_1, iu_1);
        im_1 = _mm_add_ps(im_1, temp);

        // rm = _mm256_fnmadd_ps(i2, iu, rm);
        temp = _mm_mul_ps(i2_0, iu_0);
        rm_0 = _mm_sub_ps(rm_0, temp);
        temp = _mm_mul_ps(i2_1, iu_1);
        rm_1 = _mm_sub_ps(rm_1, temp);

        // im = _mm256_fmadd_ps(i2, ru, im);
        temp = _mm_mul_ps(i2_0, ru_0);
        im_0 = _mm_add_ps(im_0, temp);
        temp = _mm_mul_ps(i2_1, ru_1);
        im_1 = _mm_add_ps(im_1, temp);

        // ru = _mm256_set1_ps(matrix[14]);
        ru_0 = _mm_set1_ps(matrix[14]);
        ru_1 = _mm_set1_ps(matrix[14]);

        // iu = _mm256_set1_ps(matrix[15]);
        iu_0 = _mm_set1_ps(matrix[15]);
        iu_1 = _mm_set1_ps(matrix[15]);

        // rm = _mm256_fmadd_ps(r3, ru, rm);
        temp = _mm_mul_ps(r3_0, ru_0);
        rm_0 = _mm_add_ps(rm_0, temp);
        temp = _mm_mul_ps(r3_1, ru_1);
        rm_1 = _mm_add_ps(rm_1, temp);

        // im = _mm256_fmadd_ps(r3, iu, im);
        temp = _mm_mul_ps(r3_0, iu_0);
        im_0 = _mm_add_ps(im_0, temp);
        temp = _mm_mul_ps(r3_1, iu_1);
        im_1 = _mm_add_ps(im_1, temp);

        // rm = _mm256_fnmadd_ps(i3, iu, rm);
        temp = _mm_mul_ps(i3_0, iu_0);
        rm_0 = _mm_sub_ps(rm_0, temp);
        temp = _mm_mul_ps(i3_1, iu_1);
        rm_1 = _mm_sub_ps(rm_1, temp);

        // im = _mm256_fmadd_ps(i3, ru, im);
        temp = _mm_mul_ps(i3_0, ru_0);
        im_0 = _mm_add_ps(im_0, temp);
        temp = _mm_mul_ps(i3_1, ru_1);
        im_1 = _mm_add_ps(im_1, temp);

        // rm = _mm256_permutevar8x32_ps(rm, ml1);
        rm_0 = _mm_shuffle_ps(rm_0, rm_0, 225);
        rm_1 = _mm_shuffle_ps(rm_1, rm_1, 225);

        // im = _mm256_permutevar8x32_ps(im, ml1);
        im_0 = _mm_shuffle_ps(im_0, im_0, 225);
        im_1 = _mm_shuffle_ps(im_1, im_1, 225);

        // rn = _mm256_blendv_ps(rn, rm, mb1);
        rn_0 = _mm_blendv_ps(rn_0, rm_0, mb1);
        rn_1 = _mm_blendv_ps(rn_1, rm_1, mb1);

        // in = _mm256_blendv_ps(in, im, mb1);
        in_0 = _mm_blendv_ps(in_0, im_0, mb1);
        in_1 = _mm_blendv_ps(in_1, im_1, mb1);

        // ru = _mm256_set1_ps(matrix[16]);
        ru_0 = _mm_set1_ps(matrix[16]);
        ru_1 = _mm_set1_ps(matrix[16]);

        // iu = _mm256_set1_ps(matrix[17]);
        iu_0 = _mm_set1_ps(matrix[17]);
        iu_1 = _mm_set1_ps(matrix[17]);

        // rm = _mm256_mul_ps(r0, ru);
        rm_0 = _mm_mul_ps(r0_0, ru_0);
        rm_1 = _mm_mul_ps(r0_1, ru_1);

        // im = _mm256_mul_ps(r0, iu);
        im_0 = _mm_mul_ps(r0_0, iu_0);
        im_1 = _mm_mul_ps(r0_1, iu_1);

        // rm = _mm256_fnmadd_ps(i0, iu, rm);
        temp = _mm_mul_ps(i0_0, iu_0);
        rm_0 = _mm_sub_ps(rm_0, temp);
        temp = _mm_mul_ps(i0_1, iu_1);
        rm_1 = _mm_sub_ps(rm_1, temp);

        // im = _mm256_fmadd_ps(i0, ru, im);
        temp = _mm_mul_ps(i0_0, ru_0);
        im_0 = _mm_add_ps(im_0, temp);
        temp = _mm_mul_ps(i0_1, ru_1);
        im_1 = _mm_add_ps(im_1, temp);

        // ru = _mm256_set1_ps(matrix[18]);
        ru_0 = _mm_set1_ps(matrix[18]);
        ru_1 = _mm_set1_ps(matrix[18]);

        // iu = _mm256_set1_ps(matrix[19]);
        iu_0 = _mm_set1_ps(matrix[19]);
        iu_1 = _mm_set1_ps(matrix[19]);

        // rm = _mm256_fmadd_ps(r1, ru, rm);
        temp = _mm_mul_ps(r1_0, ru_0);
        rm_0 = _mm_add_ps(rm_0, temp);
        temp = _mm_mul_ps(r1_1, ru_1);
        rm_1 = _mm_add_ps(rm_1, temp);

        // im = _mm256_fmadd_ps(r1, iu, im);
        temp = _mm_mul_ps(r1_0, iu_0);
        im_0 = _mm_add_ps(im_0, temp);
        temp = _mm_mul_ps(r1_1, iu_1);
        im_1 = _mm_add_ps(im_1, temp);

        // rm = _mm256_fnmadd_ps(i1, iu, rm);
        temp = _mm_mul_ps(i1_0, iu_0);
        rm_0 = _mm_sub_ps(rm_0, temp);
        temp = _mm_mul_ps(i1_1, iu_1);
        rm_1 = _mm_sub_ps(rm_1, temp);

        // im = _mm256_fmadd_ps(i1, ru, im);
        temp = _mm_mul_ps(i1_0, ru_0);
        im_0 = _mm_add_ps(im_0, temp);
        temp = _mm_mul_ps(i1_1, ru_1);
        im_1 = _mm_add_ps(im_1, temp);

        // ru = _mm256_set1_ps(matrix[20]);
        ru_0 = _mm_set1_ps(matrix[20]);
        ru_1 = _mm_set1_ps(matrix[20]);

        // iu = _mm256_set1_ps(matrix[21]);
        iu_0 = _mm_set1_ps(matrix[21]);
        iu_1 = _mm_set1_ps(matrix[21]);

        // rm = _mm256_fmadd_ps(r2, ru, rm);
        temp = _mm_mul_ps(r2_0, ru_0);
        rm_0 = _mm_add_ps(rm_0, temp);
        temp = _mm_mul_ps(r2_1, ru_1);
        rm_1 = _mm_add_ps(rm_1, temp);

        // im = _mm256_fmadd_ps(r2, iu, im);
        temp = _mm_mul_ps(r2_0, iu_0);
        im_0 = _mm_add_ps(im_0, temp);
        temp = _mm_mul_ps(r2_1, iu_1);
        im_1 = _mm_add_ps(im_1, temp);

        // rm = _mm256_fnmadd_ps(i2, iu, rm);
        temp = _mm_mul_ps(i2_0, iu_0);
        rm_0 = _mm_sub_ps(rm_0, temp);
        temp = _mm_mul_ps(i2_1, iu_1);
        rm_1 = _mm_sub_ps(rm_1, temp);

        // im = _mm256_fmadd_ps(i2, ru, im);
        temp = _mm_mul_ps(i2_0, ru_0);
        im_0 = _mm_add_ps(im_0, temp);
        temp = _mm_mul_ps(i2_1, ru_1);
        im_1 = _mm_add_ps(im_1, temp);

        // ru = _mm256_set1_ps(matrix[22]);
        ru_0 = _mm_set1_ps(matrix[22]);
        ru_1 = _mm_set1_ps(matrix[22]);

        // iu = _mm256_set1_ps(matrix[23]);
        iu_0 = _mm_set1_ps(matrix[23]);
        iu_1 = _mm_set1_ps(matrix[23]);

        // rm = _mm256_fmadd_ps(r3, ru, rm);
        temp = _mm_mul_ps(r3_0, ru_0);
        rm_0 = _mm_add_ps(rm_0, temp);
        temp = _mm_mul_ps(r3_1, ru_1);
        rm_1 = _mm_add_ps(rm_1, temp);

        // im = _mm256_fmadd_ps(r3, iu, im);
        temp = _mm_mul_ps(r3_0, iu_0);
        im_0 = _mm_add_ps(im_0, temp);
        temp = _mm_mul_ps(r3_1, iu_1);
        im_1 = _mm_add_ps(im_1, temp);

        // rm = _mm256_fnmadd_ps(i3, iu, rm);
        temp = _mm_mul_ps(i3_0, iu_0);
        rm_0 = _mm_sub_ps(rm_0, temp);
        temp = _mm_mul_ps(i3_1, iu_1);
        rm_1 = _mm_sub_ps(rm_1, temp);

        // im = _mm256_fmadd_ps(i3, ru, im);
        temp = _mm_mul_ps(i3_0, ru_0);
        im_0 = _mm_add_ps(im_0, temp);
        temp = _mm_mul_ps(i3_1, ru_1);
        im_1 = _mm_add_ps(im_1, temp);

        // rm = _mm256_permutevar8x32_ps(rm, ml2);
        rm_0 = _mm_shuffle_ps(rm_0, rm_0, 198);
        rm_1 = _mm_shuffle_ps(rm_1, rm_1, 198);

        // im = _mm256_permutevar8x32_ps(im, ml2);
        im_0 = _mm_shuffle_ps(im_0, im_0, 198);
        im_1 = _mm_shuffle_ps(im_1, im_1, 198);

        // rn = _mm256_blendv_ps(rn, rm, mb2);
        rn_0 = _mm_blendv_ps(rn_0, rm_0, mb2);
        rn_1 = _mm_blendv_ps(rn_1, rm_1, mb2);

        // in = _mm256_blendv_ps(in, im, mb2);
        in_0 = _mm_blendv_ps(in_0, im_0, mb2);
        in_1 = _mm_blendv_ps(in_1, im_1, mb2);

        // ru = _mm256_set1_ps(matrix[24]);
        ru_0 = _mm_set1_ps(matrix[24]);
        ru_1 = _mm_set1_ps(matrix[24]);

        // iu = _mm256_set1_ps(matrix[25]);
        iu_0 = _mm_set1_ps(matrix[25]);
        iu_1 = _mm_set1_ps(matrix[25]);

        // rm = _mm256_mul_ps(r0, ru);
        rm_0 = _mm_mul_ps(r0_0, ru_0);
        rm_1 = _mm_mul_ps(r0_1, ru_1);

        // im = _mm256_mul_ps(r0, iu);
        im_0 = _mm_mul_ps(r0_0, iu_0);
        im_1 = _mm_mul_ps(r0_1, iu_1);

        // rm = _mm256_fnmadd_ps(i0, iu, rm);
        temp = _mm_mul_ps(i0_0, iu_0);
        rm_0 = _mm_sub_ps(rm_0, temp);
        temp = _mm_mul_ps(i0_1, iu_1);
        rm_1 = _mm_sub_ps(rm_1, temp);

        // im = _mm256_fmadd_ps(i0, ru, im);
        temp = _mm_mul_ps(i0_0, ru_0);
        im_0 = _mm_add_ps(im_0, temp);
        temp = _mm_mul_ps(i0_1, ru_1);
        im_1 = _mm_add_ps(im_1, temp);

        // ru = _mm256_set1_ps(matrix[26]);
        ru_0 = _mm_set1_ps(matrix[26]);
        ru_1 = _mm_set1_ps(matrix[26]);

        // iu = _mm256_set1_ps(matrix[27]);
        iu_0 = _mm_set1_ps(matrix[27]);
        iu_1 = _mm_set1_ps(matrix[27]);

        // rm = _mm256_fmadd_ps(r1, ru, rm);
        temp = _mm_mul_ps(r1_0, ru_0);
        rm_0 = _mm_add_ps(rm_0, temp);
        temp = _mm_mul_ps(r1_1, ru_1);
        rm_1 = _mm_add_ps(rm_1, temp);

        // im = _mm256_fmadd_ps(r1, iu, im);
        temp = _mm_mul_ps(r1_0, iu_0);
        im_0 = _mm_add_ps(im_0, temp);
        temp = _mm_mul_ps(r1_1, iu_1);
        im_1 = _mm_add_ps(im_1, temp);

        // rm = _mm256_fnmadd_ps(i1, iu, rm);
        temp = _mm_mul_ps(i1_0, iu_0);
        rm_0 = _mm_sub_ps(rm_0, temp);
        temp = _mm_mul_ps(i1_1, iu_1);
        rm_1 = _mm_sub_ps(rm_1, temp);

        // im = _mm256_fmadd_ps(i1, ru, im);
        temp = _mm_mul_ps(i1_0, ru_0);
        im_0 = _mm_add_ps(im_0, temp);
        temp = _mm_mul_ps(i1_1, ru_1);
        im_1 = _mm_add_ps(im_1, temp);

        // ru = _mm256_set1_ps(matrix[28]);
        ru_0 = _mm_set1_ps(matrix[28]);
        ru_1 = _mm_set1_ps(matrix[28]);

        // iu = _mm256_set1_ps(matrix[29]);
        iu_0 = _mm_set1_ps(matrix[29]);
        iu_1 = _mm_set1_ps(matrix[29]);

        // rm = _mm256_fmadd_ps(r2, ru, rm);
        temp = _mm_mul_ps(r2_0, ru_0);
        rm_0 = _mm_add_ps(rm_0, temp);
        temp = _mm_mul_ps(r2_1, ru_1);
        rm_1 = _mm_add_ps(rm_1, temp);

        // im = _mm256_fmadd_ps(r2, iu, im);
        temp = _mm_mul_ps(r2_0, iu_0);
        im_0 = _mm_add_ps(im_0, temp);
        temp = _mm_mul_ps(r2_1, iu_1);
        im_1 = _mm_add_ps(im_1, temp);

        // rm = _mm256_fnmadd_ps(i2, iu, rm);
        temp = _mm_mul_ps(i2_0, iu_0);
        rm_0 = _mm_sub_ps(rm_0, temp);
        temp = _mm_mul_ps(i2_1, iu_1);
        rm_1 = _mm_sub_ps(rm_1, temp);

        // im = _mm256_fmadd_ps(i2, ru, im);
        temp = _mm_mul_ps(i2_0, ru_0);
        im_0 = _mm_add_ps(im_0, temp);
        temp = _mm_mul_ps(i2_1, ru_1);
        im_1 = _mm_add_ps(im_1, temp);

        // ru = _mm256_set1_ps(matrix[30]);
        ru_0 = _mm_set1_ps(matrix[30]);
        ru_1 = _mm_set1_ps(matrix[30]);

        // iu = _mm256_set1_ps(matrix[31]);
        iu_0 = _mm_set1_ps(matrix[31]);
        iu_1 = _mm_set1_ps(matrix[31]);

        // rm = _mm256_fmadd_ps(r3, ru, rm);
        temp = _mm_mul_ps(r3_0, ru_0);
        rm_0 = _mm_add_ps(rm_0, temp);
        temp = _mm_mul_ps(r3_1, ru_1);
        rm_1 = _mm_add_ps(rm_1, temp);

        // im = _mm256_fmadd_ps(r3, iu, im);
        temp = _mm_mul_ps(r3_0, iu_0);
        im_0 = _mm_add_ps(im_0, temp);
        temp = _mm_mul_ps(r3_1, iu_1);
        im_1 = _mm_add_ps(im_1, temp);

        // rm = _mm256_fnmadd_ps(i3, iu, rm);
        temp = _mm_mul_ps(i3_0, iu_0);
        rm_0 = _mm_sub_ps(rm_0, temp);
        temp = _mm_mul_ps(i3_1, iu_1);
        rm_1 = _mm_sub_ps(rm_1, temp);

        // im = _mm256_fmadd_ps(i3, ru, im);
        temp = _mm_mul_ps(i3_0, ru_0);
        im_0 = _mm_add_ps(im_0, temp);
        temp = _mm_mul_ps(i3_1, ru_1);
        im_1 = _mm_add_ps(im_1, temp);

        // rm = _mm256_permutevar8x32_ps(rm, ml3);
        rm_0 = _mm_shuffle_ps(rm_0, rm_0, 39);
        rm_1 = _mm_shuffle_ps(rm_1, rm_1, 39);

        // im = _mm256_permutevar8x32_ps(im, ml3);
        im_0 = _mm_shuffle_ps(im_0, im_0, 39);
        im_1 = _mm_shuffle_ps(im_1, im_1, 39);

        // rn = _mm256_blendv_ps(rn, rm, mb3);
        rn_0 = _mm_blendv_ps(rn_0, rm_0, mb3);
        rn_1 = _mm_blendv_ps(rn_1, rm_1, mb3);

        // in = _mm256_blendv_ps(in, im, mb3);
        in_0 = _mm_blendv_ps(in_0, im_0, mb3);
        in_1 = _mm_blendv_ps(in_1, im_1, mb3);

        //_mm256_store_ps(p, rn);
        _mm_store_ps(p, rn_0);
        _mm_store_ps(p + 4, rn_1);

        //_mm256_store_ps(p + 8, in);
        _mm_store_ps(p + 8, in_0);
        _mm_store_ps(p + 12, in_1);
      }

      break;
    case 2:
      mb1 = _mm_castsi128_ps(_mm_set_epi32(-1, 0, -1, 0));
      mb2 = _mm_castsi128_ps(_mm_set_epi32(0, -1, 0, -1));
      mb3 = _mm_castsi128_ps(_mm_set_epi32(-1, 0, -1, 0));

      for (uint64_t i = 0; i < sizei; i += 16) {
        //__m256 r0, i0, r1, i1, r2, i2, r3, i3, ru, iu, rn, in, rm, im;
        __m128 r0_0, i0_0, r1_0, i1_0, r2_0, i2_0, r3_0, i3_0, ru_0, iu_0, rn_0,
            in_0, rm_0, im_0;
        __m128 r0_1, i0_1, r1_1, i1_1, r2_1, i2_1, r3_1, i3_1, ru_1, iu_1, rn_1,
            in_1, rm_1, im_1;

        // holder for fnmadd and fmadd.
        __m128 temp, temp2;

        auto p = rstate + i;

        // r0 = _mm256_load_ps(p);
        r0_0 = _mm_load_ps(p);
        r0_1 = _mm_load_ps(p + 4);

        // i0 = _mm256_load_ps(p + 8);
        i0_0 = _mm_load_ps(p + 8);
        i0_1 = _mm_load_ps(p + 8 + 4);

        // r1 = _mm256_permutevar8x32_ps(r0, ml1);
        r1_0 = _mm_shuffle_ps(r0_0, r0_0, 177);
        r1_1 = r0_1;

        // i1 = _mm256_permutevar8x32_ps(i0, ml1);
        i1_0 = _mm_shuffle_ps(i0_0, i0_0, 177);
        i1_1 = i0_1;

        // r2 = _mm256_permutevar8x32_ps(r0, ml2);
        r2_0 = _mm_shuffle_ps(r0_0, r0_1, 141);
        r2_0 = _mm_shuffle_ps(r2_0, r2_0, 114);

        r2_1 = _mm_shuffle_ps(r0_0, r0_1, 216);
        r2_1 = _mm_shuffle_ps(r2_1, r2_1, 216);

        // i2 = _mm256_permutevar8x32_ps(i0, ml2);
        i2_0 = _mm_shuffle_ps(i0_0, i0_1, 141);
        i2_0 = _mm_shuffle_ps(i2_0, i2_0, 114);

        i2_1 = _mm_shuffle_ps(i0_1, i0_1, 216);
        i2_1 = _mm_shuffle_ps(i2_1, i2_1, 216);

        // r3 = _mm256_permutevar8x32_ps(r0, ml3);
        r3_0 = _mm_shuffle_ps(r0_0, r0_1, 221);
        r3_0 = _mm_shuffle_ps(r3_0, r3_0, 114);

        r3_1 = _mm_shuffle_ps(r0_0, r0_1, 136);
        r3_1 = _mm_shuffle_ps(r3_1, r3_1, 114);

        // i3 = _mm256_permutevar8x32_ps(i0, ml3);
        i3_0 = _mm_shuffle_ps(i0_0, i0_1, 221);
        i3_0 = _mm_shuffle_ps(i3_0, i3_0, 114);

        i3_1 = _mm_shuffle_ps(i0_0, i0_1, 136);
        i3_1 = _mm_shuffle_ps(i3_1, i3_1, 114);

        // ru = _mm256_set1_ps(matrix[0]);
        ru_0 = _mm_set1_ps(matrix[0]);
        ru_1 = _mm_set1_ps(matrix[0]);

        // iu = _mm256_set1_ps(matrix[1]);
        iu_0 = _mm_set1_ps(matrix[1]);
        iu_1 = _mm_set1_ps(matrix[1]);

        // rn = _mm256_mul_ps(r0, ru);
        rn_0 = _mm_mul_ps(r0_0, ru_0);
        rn_1 = _mm_mul_ps(r0_1, ru_1);

        // in = _mm256_mul_ps(r0, iu);
        in_0 = _mm_mul_ps(r0_0, iu_0);
        in_1 = _mm_mul_ps(r0_1, iu_1);

        // rn = _mm256_fnmadd_ps(i0, iu, rn);
        temp = _mm_mul_ps(i0_0, iu_0);
        rn_0 = _mm_sub_ps(rn_0, temp);
        temp = _mm_mul_ps(i0_1, iu_1);
        rn_1 = _mm_sub_ps(rn_1, temp);

        // in = _mm256_fmadd_ps(i0, ru, in);
        temp = _mm_mul_ps(i0_0, ru_0);
        in_0 = _mm_add_ps(in_0, temp);
        temp = _mm_mul_ps(i0_1, ru_1);
        in_1 = _mm_add_ps(in_1, temp);

        // ru = _mm256_set1_ps(matrix[2]);
        ru_0 = _mm_set1_ps(matrix[2]);
        ru_1 = _mm_set1_ps(matrix[2]);

        // iu = _mm256_set1_ps(matrix[3]);
        iu_0 = _mm_set1_ps(matrix[3]);
        iu_1 = _mm_set1_ps(matrix[3]);

        // rn = _mm256_fmadd_ps(r1, ru, rn);
        temp = _mm_mul_ps(r1_0, ru_0);
        rn_0 = _mm_add_ps(rn_0, temp);
        temp = _mm_mul_ps(r1_1, ru_1);
        rn_1 = _mm_add_ps(rn_1, temp);

        // in = _mm256_fmadd_ps(r1, iu, in);
        temp = _mm_mul_ps(r1_0, iu_0);
        in_0 = _mm_add_ps(in_0, temp);
        temp = _mm_mul_ps(r1_1, iu_1);
        in_1 = _mm_add_ps(in_1, temp);

        // rn = _mm256_fnmadd_ps(i1, iu, rn);
        temp = _mm_mul_ps(i1_0, iu_0);
        rn_0 = _mm_sub_ps(rn_0, temp);
        temp = _mm_mul_ps(i1_1, iu_1);
        rn_1 = _mm_sub_ps(rn_1, temp);

        // in = _mm256_fmadd_ps(i1, ru, in);
        temp = _mm_mul_ps(i1_0, ru_0);
        in_0 = _mm_add_ps(in_0, temp);
        temp = _mm_mul_ps(i1_1, ru_1);
        in_1 = _mm_add_ps(in_1, temp);

        // ru = _mm256_set1_ps(matrix[4]);
        ru_0 = _mm_set1_ps(matrix[4]);
        ru_1 = _mm_set1_ps(matrix[4]);

        // iu = _mm256_set1_ps(matrix[5]);
        iu_0 = _mm_set1_ps(matrix[5]);
        iu_1 = _mm_set1_ps(matrix[5]);

        // rn = _mm256_fmadd_ps(r2, ru, rn);
        temp = _mm_mul_ps(r2_0, ru_0);
        rn_0 = _mm_add_ps(rn_0, temp);
        temp = _mm_mul_ps(r2_1, ru_1);
        rn_1 = _mm_add_ps(rn_1, temp);

        // in = _mm256_fmadd_ps(r2, iu, in);
        temp = _mm_mul_ps(r2_0, iu_0);
        in_0 = _mm_add_ps(in_0, temp);
        temp = _mm_mul_ps(r2_1, iu_1);
        in_1 = _mm_add_ps(in_1, temp);

        // rn = _mm256_fnmadd_ps(i2, iu, rn);
        temp = _mm_mul_ps(i2_0, iu_0);
        rn_0 = _mm_sub_ps(rn_0, temp);
        temp = _mm_mul_ps(i2_1, iu_1);
        rn_1 = _mm_sub_ps(rn_1, temp);

        // in = _mm256_fmadd_ps(i2, ru, in);
        temp = _mm_mul_ps(i2_0, ru_0);
        in_0 = _mm_add_ps(in_0, temp);
        temp = _mm_mul_ps(i2_1, ru_1);
        in_1 = _mm_add_ps(in_1, temp);

        // ru = _mm256_set1_ps(matrix[6]);
        ru_0 = _mm_set1_ps(matrix[6]);
        ru_1 = _mm_set1_ps(matrix[6]);

        // iu = _mm256_set1_ps(matrix[7]);
        iu_0 = _mm_set1_ps(matrix[7]);
        iu_1 = _mm_set1_ps(matrix[7]);

        // rn = _mm256_fmadd_ps(r3, ru, rn);
        temp = _mm_mul_ps(r3_0, ru_0);
        rn_0 = _mm_add_ps(rn_0, temp);
        temp = _mm_mul_ps(r3_1, ru_1);
        rn_1 = _mm_add_ps(rn_1, temp);

        // in = _mm256_fmadd_ps(r3, iu, in);
        temp = _mm_mul_ps(r3_0, iu_0);
        in_0 = _mm_add_ps(in_0, temp);
        temp = _mm_mul_ps(r3_1, iu_1);
        in_1 = _mm_add_ps(in_1, temp);

        // rn = _mm256_fnmadd_ps(i3, iu, rn);
        temp = _mm_mul_ps(i3_0, iu_0);
        rn_0 = _mm_sub_ps(rn_0, temp);
        temp = _mm_mul_ps(i3_1, iu_1);
        rn_1 = _mm_sub_ps(rn_1, temp);

        // in = _mm256_fmadd_ps(i3, ru, in);
        temp = _mm_mul_ps(i3_0, ru_0);
        in_0 = _mm_add_ps(in_0, temp);
        temp = _mm_mul_ps(i3_1, ru_1);
        in_1 = _mm_add_ps(in_1, temp);

        // ru = _mm256_set1_ps(matrix[8]);
        ru_0 = _mm_set1_ps(matrix[8]);
        ru_1 = _mm_set1_ps(matrix[8]);

        // iu = _mm256_set1_ps(matrix[9]);
        iu_0 = _mm_set1_ps(matrix[9]);
        iu_1 = _mm_set1_ps(matrix[9]);

        // rm = _mm256_mul_ps(r0, ru);
        rm_0 = _mm_mul_ps(r0_0, ru_0);
        rm_1 = _mm_mul_ps(r0_1, ru_1);

        // im = _mm256_mul_ps(r0, iu);
        im_0 = _mm_mul_ps(r0_0, iu_0);
        im_1 = _mm_mul_ps(r0_1, iu_1);

        // rm = _mm256_fnmadd_ps(i0, iu, rm);
        temp = _mm_mul_ps(i0_0, iu_0);
        rm_0 = _mm_sub_ps(rm_0, temp);
        temp = _mm_mul_ps(i0_1, iu_1);
        rm_1 = _mm_sub_ps(rm_1, temp);

        // im = _mm256_fmadd_ps(i0, ru, im);
        temp = _mm_mul_ps(i0_0, ru_0);
        im_0 = _mm_add_ps(im_0, temp);
        temp = _mm_mul_ps(i0_1, ru_1);
        im_1 = _mm_add_ps(im_1, temp);

        // ru = _mm256_set1_ps(matrix[10]);
        ru_0 = _mm_set1_ps(matrix[10]);
        ru_1 = _mm_set1_ps(matrix[10]);

        // iu = _mm256_set1_ps(matrix[11]);
        iu_0 = _mm_set1_ps(matrix[11]);
        iu_1 = _mm_set1_ps(matrix[11]);

        // rm = _mm256_fmadd_ps(r1, ru, rm);
        temp = _mm_mul_ps(r1_0, ru_0);
        rm_0 = _mm_add_ps(rm_0, temp);
        temp = _mm_mul_ps(r1_1, ru_1);
        rm_1 = _mm_add_ps(rm_1, temp);

        // im = _mm256_fmadd_ps(r1, iu, im);
        temp = _mm_mul_ps(r1_0, iu_0);
        im_0 = _mm_add_ps(im_0, temp);
        temp = _mm_mul_ps(r1_1, iu_1);
        im_1 = _mm_add_ps(im_1, temp);

        // rm = _mm256_fnmadd_ps(i1, iu, rm);
        temp = _mm_mul_ps(i1_0, iu_0);
        rm_0 = _mm_sub_ps(rm_0, temp);
        temp = _mm_mul_ps(i1_1, iu_1);
        rm_1 = _mm_sub_ps(rm_1, temp);

        // im = _mm256_fmadd_ps(i1, ru, im);
        temp = _mm_mul_ps(i1_0, ru_0);
        im_0 = _mm_add_ps(im_0, temp);
        temp = _mm_mul_ps(i1_1, ru_1);
        im_1 = _mm_add_ps(im_1, temp);

        // ru = _mm256_set1_ps(matrix[12]);
        ru_0 = _mm_set1_ps(matrix[12]);
        ru_1 = _mm_set1_ps(matrix[12]);

        // iu = _mm256_set1_ps(matrix[13]);
        iu_0 = _mm_set1_ps(matrix[13]);
        iu_1 = _mm_set1_ps(matrix[13]);

        // rm = _mm256_fmadd_ps(r2, ru, rm);
        temp = _mm_mul_ps(r2_0, ru_0);
        rm_0 = _mm_add_ps(rm_0, temp);
        temp = _mm_mul_ps(r2_1, ru_1);
        rm_1 = _mm_add_ps(rm_1, temp);

        // im = _mm256_fmadd_ps(r2, iu, im);
        temp = _mm_mul_ps(r2_0, iu_0);
        im_0 = _mm_add_ps(im_0, temp);
        temp = _mm_mul_ps(r2_1, iu_1);
        im_1 = _mm_add_ps(im_1, temp);

        // rm = _mm256_fnmadd_ps(i2, iu, rm);
        temp = _mm_mul_ps(i2_0, iu_0);
        rm_0 = _mm_sub_ps(rm_0, temp);
        temp = _mm_mul_ps(i2_1, iu_1);
        rm_1 = _mm_sub_ps(rm_1, temp);

        // im = _mm256_fmadd_ps(i2, ru, im);
        temp = _mm_mul_ps(i2_0, ru_0);
        im_0 = _mm_add_ps(im_0, temp);
        temp = _mm_mul_ps(i2_1, ru_1);
        im_1 = _mm_add_ps(im_1, temp);

        // ru = _mm256_set1_ps(matrix[14]);
        ru_0 = _mm_set1_ps(matrix[14]);
        ru_1 = _mm_set1_ps(matrix[14]);

        // iu = _mm256_set1_ps(matrix[15]);
        iu_0 = _mm_set1_ps(matrix[15]);
        iu_1 = _mm_set1_ps(matrix[15]);

        // rm = _mm256_fmadd_ps(r3, ru, rm);
        temp = _mm_mul_ps(r3_0, ru_0);
        rm_0 = _mm_add_ps(rm_0, temp);
        temp = _mm_mul_ps(r3_1, ru_1);
        rm_1 = _mm_add_ps(rm_1, temp);

        // im = _mm256_fmadd_ps(r3, iu, im);
        temp = _mm_mul_ps(r3_0, iu_0);
        im_0 = _mm_add_ps(im_0, temp);
        temp = _mm_mul_ps(r3_1, iu_1);
        im_1 = _mm_add_ps(im_1, temp);

        // rm = _mm256_fnmadd_ps(i3, iu, rm);
        temp = _mm_mul_ps(i3_0, iu_0);
        rm_0 = _mm_sub_ps(rm_0, temp);
        temp = _mm_mul_ps(i3_1, iu_1);
        rm_1 = _mm_sub_ps(rm_1, temp);

        // im = _mm256_fmadd_ps(i3, ru, im);
        temp = _mm_mul_ps(i3_0, ru_0);
        im_0 = _mm_add_ps(im_0, temp);
        temp = _mm_mul_ps(i3_1, ru_1);
        im_1 = _mm_add_ps(im_1, temp);

        // rm = _mm256_permutevar8x32_ps(rm, ml1);
        rm_0 = _mm_shuffle_ps(rm_0, rm_0, 177);
        // rm_1 = nothing

        // im = _mm256_permutevar8x32_ps(im, ml1);
        im_0 = _mm_shuffle_ps(im_0, im_0, 177);
        // im_1 = nothing

        // rn = _mm256_blendv_ps(rn, rm, mb1);
        rn_0 = _mm_blendv_ps(rn_0, rm_0, mb1);
        // rn_1 = rn_1;

        // in = _mm256_blendv_ps(in, im, mb1);
        in_0 = _mm_blendv_ps(in_0, im_0, mb1);
        // in_1 = in_1;

        // ru = _mm256_set1_ps(matrix[16]);
        ru_0 = _mm_set1_ps(matrix[16]);
        ru_1 = _mm_set1_ps(matrix[16]);

        // iu = _mm256_set1_ps(matrix[17]);
        iu_0 = _mm_set1_ps(matrix[17]);
        iu_1 = _mm_set1_ps(matrix[17]);

        // rm = _mm256_mul_ps(r0, ru);
        rm_0 = _mm_mul_ps(r0_0, ru_0);
        rm_1 = _mm_mul_ps(r0_1, ru_1);

        // im = _mm256_mul_ps(r0, iu);
        im_0 = _mm_mul_ps(r0_0, iu_0);
        im_1 = _mm_mul_ps(r0_1, iu_1);

        // rm = _mm256_fnmadd_ps(i0, iu, rm);
        temp = _mm_mul_ps(i0_0, iu_0);
        rm_0 = _mm_sub_ps(rm_0, temp);
        temp = _mm_mul_ps(i0_1, iu_1);
        rm_1 = _mm_sub_ps(rm_1, temp);

        // im = _mm256_fmadd_ps(i0, ru, im);
        temp = _mm_mul_ps(i0_0, ru_0);
        im_0 = _mm_add_ps(im_0, temp);
        temp = _mm_mul_ps(i0_1, ru_1);
        im_1 = _mm_add_ps(im_1, temp);

        // ru = _mm256_set1_ps(matrix[18]);
        ru_0 = _mm_set1_ps(matrix[18]);
        ru_1 = _mm_set1_ps(matrix[18]);

        // iu = _mm256_set1_ps(matrix[19]);
        iu_0 = _mm_set1_ps(matrix[19]);
        iu_1 = _mm_set1_ps(matrix[19]);

        // rm = _mm256_fmadd_ps(r1, ru, rm);
        temp = _mm_mul_ps(r1_0, ru_0);
        rm_0 = _mm_add_ps(rm_0, temp);
        temp = _mm_mul_ps(r1_1, ru_1);
        rm_1 = _mm_add_ps(rm_1, temp);

        // im = _mm256_fmadd_ps(r1, iu, im);
        temp = _mm_mul_ps(r1_0, iu_0);
        im_0 = _mm_add_ps(im_0, temp);
        temp = _mm_mul_ps(r1_1, iu_1);
        im_1 = _mm_add_ps(im_1, temp);

        // rm = _mm256_fnmadd_ps(i1, iu, rm);
        temp = _mm_mul_ps(i1_0, iu_0);
        rm_0 = _mm_sub_ps(rm_0, temp);
        temp = _mm_mul_ps(i1_1, iu_1);
        rm_1 = _mm_sub_ps(rm_1, temp);

        // im = _mm256_fmadd_ps(i1, ru, im);
        temp = _mm_mul_ps(i1_0, ru_0);
        im_0 = _mm_add_ps(im_0, temp);
        temp = _mm_mul_ps(i1_1, ru_1);
        im_1 = _mm_add_ps(im_1, temp);

        // ru = _mm256_set1_ps(matrix[20]);
        ru_0 = _mm_set1_ps(matrix[20]);
        ru_1 = _mm_set1_ps(matrix[20]);

        // iu = _mm256_set1_ps(matrix[21]);
        iu_0 = _mm_set1_ps(matrix[21]);
        iu_1 = _mm_set1_ps(matrix[21]);

        // rm = _mm256_fmadd_ps(r2, ru, rm);
        temp = _mm_mul_ps(r2_0, ru_0);
        rm_0 = _mm_add_ps(rm_0, temp);
        temp = _mm_mul_ps(r2_1, ru_1);
        rm_1 = _mm_add_ps(rm_1, temp);

        // im = _mm256_fmadd_ps(r2, iu, im);
        temp = _mm_mul_ps(r2_0, iu_0);
        im_0 = _mm_add_ps(im_0, temp);
        temp = _mm_mul_ps(r2_1, iu_1);
        im_1 = _mm_add_ps(im_1, temp);

        // rm = _mm256_fnmadd_ps(i2, iu, rm);
        temp = _mm_mul_ps(i2_0, iu_0);
        rm_0 = _mm_sub_ps(rm_0, temp);
        temp = _mm_mul_ps(i2_1, iu_1);
        rm_1 = _mm_sub_ps(rm_1, temp);

        // im = _mm256_fmadd_ps(i2, ru, im);
        temp = _mm_mul_ps(i2_0, ru_0);
        im_0 = _mm_add_ps(im_0, temp);
        temp = _mm_mul_ps(i2_1, ru_1);
        im_1 = _mm_add_ps(im_1, temp);

        // ru = _mm256_set1_ps(matrix[22]);
        ru_0 = _mm_set1_ps(matrix[22]);
        ru_1 = _mm_set1_ps(matrix[22]);

        // iu = _mm256_set1_ps(matrix[23]);
        iu_0 = _mm_set1_ps(matrix[23]);
        iu_1 = _mm_set1_ps(matrix[23]);

        // rm = _mm256_fmadd_ps(r3, ru, rm);
        temp = _mm_mul_ps(r3_0, ru_0);
        rm_0 = _mm_add_ps(rm_0, temp);
        temp = _mm_mul_ps(r3_1, ru_1);
        rm_1 = _mm_add_ps(rm_1, temp);

        // im = _mm256_fmadd_ps(r3, iu, im);
        temp = _mm_mul_ps(r3_0, iu_0);
        im_0 = _mm_add_ps(im_0, temp);
        temp = _mm_mul_ps(r3_1, iu_1);
        im_1 = _mm_add_ps(im_1, temp);

        // rm = _mm256_fnmadd_ps(i3, iu, rm);
        temp = _mm_mul_ps(i3_0, iu_0);
        rm_0 = _mm_sub_ps(rm_0, temp);
        temp = _mm_mul_ps(i3_1, iu_1);
        rm_1 = _mm_sub_ps(rm_1, temp);

        // im = _mm256_fmadd_ps(i3, ru, im);
        temp = _mm_mul_ps(i3_0, ru_0);
        im_0 = _mm_add_ps(im_0, temp);
        temp = _mm_mul_ps(i3_1, ru_1);
        im_1 = _mm_add_ps(im_1, temp);

        // rm = _mm256_permutevar8x32_ps(rm, ml2);
        temp = _mm_shuffle_ps(rm_0, rm_1, 141);
        temp2 = _mm_shuffle_ps(temp, temp, 114);

        temp = _mm_shuffle_ps(rm_0, rm_1, 216);
        rm_1 = _mm_shuffle_ps(temp, temp, 216);
        rm_0 = temp2;

        // im = _mm256_permutevar8x32_ps(im, ml2);
        temp = _mm_shuffle_ps(im_0, im_1, 141);
        temp2 = _mm_shuffle_ps(temp, temp, 114);

        temp = _mm_shuffle_ps(im_0, im_1, 216);
        im_1 = _mm_shuffle_ps(temp, temp, 216);
        im_0 = temp2;

        // rn = _mm256_blendv_ps(rn, rm, mb2);
        // rn_0 = rn_0;
        rn_1 = _mm_blendv_ps(rn_1, rm_1, mb2);

        // in = _mm256_blendv_ps(in, im, mb2);
        // in_0 = in_0;
        in_1 = _mm_blendv_ps(in_1, im_1, mb2);

        // ru = _mm256_set1_ps(matrix[24]);
        ru_0 = _mm_set1_ps(matrix[24]);
        ru_1 = _mm_set1_ps(matrix[24]);

        // iu = _mm256_set1_ps(matrix[25]);
        iu_0 = _mm_set1_ps(matrix[25]);
        iu_1 = _mm_set1_ps(matrix[25]);

        // rm = _mm256_mul_ps(r0, ru);
        rm_0 = _mm_mul_ps(r0_0, ru_0);
        rm_1 = _mm_mul_ps(r0_1, ru_1);

        // im = _mm256_mul_ps(r0, iu);
        im_0 = _mm_mul_ps(r0_0, iu_0);
        im_1 = _mm_mul_ps(r0_1, iu_1);

        // rm = _mm256_fnmadd_ps(i0, iu, rm);
        temp = _mm_mul_ps(i0_0, iu_0);
        rm_0 = _mm_sub_ps(rm_0, temp);
        temp = _mm_mul_ps(i0_1, iu_1);
        rm_1 = _mm_sub_ps(rm_1, temp);

        // im = _mm256_fmadd_ps(i0, ru, im);
        temp = _mm_mul_ps(i0_0, ru_0);
        im_0 = _mm_add_ps(im_0, temp);
        temp = _mm_mul_ps(i0_1, ru_1);
        im_1 = _mm_add_ps(im_1, temp);

        // ru = _mm256_set1_ps(matrix[26]);
        ru_0 = _mm_set1_ps(matrix[26]);
        ru_1 = _mm_set1_ps(matrix[26]);

        // iu = _mm256_set1_ps(matrix[27]);
        iu_0 = _mm_set1_ps(matrix[27]);
        iu_1 = _mm_set1_ps(matrix[27]);

        // rm = _mm256_fmadd_ps(r1, ru, rm);
        temp = _mm_mul_ps(r1_0, ru_0);
        rm_0 = _mm_add_ps(rm_0, temp);
        temp = _mm_mul_ps(r1_1, ru_1);
        rm_1 = _mm_add_ps(rm_1, temp);

        // im = _mm256_fmadd_ps(r1, iu, im);
        temp = _mm_mul_ps(r1_0, iu_0);
        im_0 = _mm_add_ps(im_0, temp);
        temp = _mm_mul_ps(r1_1, iu_1);
        im_1 = _mm_add_ps(im_1, temp);

        // rm = _mm256_fnmadd_ps(i1, iu, rm);
        temp = _mm_mul_ps(i1_0, iu_0);
        rm_0 = _mm_sub_ps(rm_0, temp);
        temp = _mm_mul_ps(i1_1, iu_1);
        rm_1 = _mm_sub_ps(rm_1, temp);

        // im = _mm256_fmadd_ps(i1, ru, im);
        temp = _mm_mul_ps(i1_0, ru_0);
        im_0 = _mm_add_ps(im_0, temp);
        temp = _mm_mul_ps(i1_1, ru_1);
        im_1 = _mm_add_ps(im_1, temp);

        // ru = _mm256_set1_ps(matrix[28]);
        ru_0 = _mm_set1_ps(matrix[28]);
        ru_1 = _mm_set1_ps(matrix[28]);

        // iu = _mm256_set1_ps(matrix[29]);
        iu_0 = _mm_set1_ps(matrix[29]);
        iu_1 = _mm_set1_ps(matrix[29]);

        // rm = _mm256_fmadd_ps(r2, ru, rm);
        temp = _mm_mul_ps(r2_0, ru_0);
        rm_0 = _mm_add_ps(rm_0, temp);
        temp = _mm_mul_ps(r2_1, ru_1);
        rm_1 = _mm_add_ps(rm_1, temp);

        // im = _mm256_fmadd_ps(r2, iu, im);
        temp = _mm_mul_ps(r2_0, iu_0);
        im_0 = _mm_add_ps(im_0, temp);
        temp = _mm_mul_ps(r2_1, iu_1);
        im_1 = _mm_add_ps(im_1, temp);

        // rm = _mm256_fnmadd_ps(i2, iu, rm);
        temp = _mm_mul_ps(i2_0, iu_0);
        rm_0 = _mm_sub_ps(rm_0, temp);
        temp = _mm_mul_ps(i2_1, iu_1);
        rm_1 = _mm_sub_ps(rm_1, temp);

        // im = _mm256_fmadd_ps(i2, ru, im);
        temp = _mm_mul_ps(i2_0, ru_0);
        im_0 = _mm_add_ps(im_0, temp);
        temp = _mm_mul_ps(i2_1, ru_1);
        im_1 = _mm_add_ps(im_1, temp);

        // ru = _mm256_set1_ps(matrix[30]);
        ru_0 = _mm_set1_ps(matrix[30]);
        ru_1 = _mm_set1_ps(matrix[30]);

        // iu = _mm256_set1_ps(matrix[31]);
        iu_0 = _mm_set1_ps(matrix[31]);
        iu_1 = _mm_set1_ps(matrix[31]);

        // rm = _mm256_fmadd_ps(r3, ru, rm);
        temp = _mm_mul_ps(r3_0, ru_0);
        rm_0 = _mm_add_ps(rm_0, temp);
        temp = _mm_mul_ps(r3_1, ru_1);
        rm_1 = _mm_add_ps(rm_1, temp);

        // im = _mm256_fmadd_ps(r3, iu, im);
        temp = _mm_mul_ps(r3_0, iu_0);
        im_0 = _mm_add_ps(im_0, temp);
        temp = _mm_mul_ps(r3_1, iu_1);
        im_1 = _mm_add_ps(im_1, temp);

        // rm = _mm256_fnmadd_ps(i3, iu, rm);
        temp = _mm_mul_ps(i3_0, iu_0);
        rm_0 = _mm_sub_ps(rm_0, temp);
        temp = _mm_mul_ps(i3_1, iu_1);
        rm_1 = _mm_sub_ps(rm_1, temp);

        // im = _mm256_fmadd_ps(i3, ru, im);
        temp = _mm_mul_ps(i3_0, ru_0);
        im_0 = _mm_add_ps(im_0, temp);
        temp = _mm_mul_ps(i3_1, ru_1);
        im_1 = _mm_add_ps(im_1, temp);

        // rm = _mm256_permutevar8x32_ps(rm, ml3);
        temp = _mm_shuffle_ps(rm_0, rm_1, 221);
        temp2 = _mm_shuffle_ps(temp, temp, 114);

        temp = _mm_shuffle_ps(rm_0, rm_1, 136);
        rm_1 = _mm_shuffle_ps(temp, temp, 114);
        rm_0 = temp2;

        // im = _mm256_permutevar8x32_ps(im, ml3);
        temp = _mm_shuffle_ps(im_0, im_1, 221);
        temp2 = _mm_shuffle_ps(temp, temp, 114);

        temp = _mm_shuffle_ps(im_0, im_1, 136);
        im_1 = _mm_shuffle_ps(temp, temp, 114);
        im_0 = temp2;

        // rn = _mm256_blendv_ps(rn, rm, mb3);
        // rn_0 = rn_0;
        rn_1 = _mm_blendv_ps(rn_1, rm_1, mb3);

        // in = _mm256_blendv_ps(in, im, mb3);
        // in_0 = in_0;
        in_1 = _mm_blendv_ps(in_1, im_1, mb3);

        //_mm256_store_ps(p, rn);
        _mm_store_ps(p, rn_0);
        _mm_store_ps(p + 4, rn_1);

        //_mm256_store_ps(p + 8, in);
        _mm_store_ps(p + 8, in_0);
        _mm_store_ps(p + 12, in_1);
      }

      break;
    case 3:
      mb1 = _mm_castsi128_ps(_mm_set_epi32(-1, -1, 0, 0));
      mb2 = _mm_castsi128_ps(_mm_set_epi32(0, 0, -1, -1));
      mb3 = _mm_castsi128_ps(_mm_set_epi32(-1, -1, 0, 0));

      for (uint64_t i = 0; i < sizei; i += 16) {
        //__m256 r0, i0, r1, i1, r2, i2, r3, i3, ru, iu, rn, in, rm, im;
        __m128 r0_0, i0_0, r1_0, i1_0, r2_0, i2_0, r3_0, i3_0, ru_0, iu_0, rn_0,
            in_0, rm_0, im_0;
        __m128 r0_1, i0_1, r1_1, i1_1, r2_1, i2_1, r3_1, i3_1, ru_1, iu_1, rn_1,
            in_1, rm_1, im_1;

        // holder for fnmadd and fmadd.
        __m128 temp, temp2;

        auto p = rstate + i;

        // r0 = _mm256_load_ps(p);
        r0_0 = _mm_load_ps(p);
        r0_1 = _mm_load_ps(p + 4);

        // i0 = _mm256_load_ps(p + 8);
        i0_0 = _mm_load_ps(p + 8);
        i0_1 = _mm_load_ps(p + 8 + 4);

        // r1 = _mm256_permutevar8x32_ps(r0, ml1);
        r1_0 = _mm_shuffle_ps(r0_0, r0_0, 78);
        r1_1 = r0_1;

        // i1 = _mm256_permutevar8x32_ps(i0, ml1);
        i1_0 = _mm_shuffle_ps(i0_0, i0_0, 78);
        i1_1 = i0_1;

        // r2 = _mm256_permutevar8x32_ps(r0, ml2);
        r2_0 = _mm_shuffle_ps(r0_0, r0_1, 78);
        r2_0 = _mm_shuffle_ps(r2_0, r2_0, 78);

        r2_1 = _mm_shuffle_ps(r0_0, r0_1, 228);

        // i2 = _mm256_permutevar8x32_ps(i0, ml2);
        i2_0 = _mm_shuffle_ps(i0_0, i0_1, 78);
        i2_0 = _mm_shuffle_ps(i2_0, i2_0, 78);

        i2_1 = _mm_shuffle_ps(i0_0, i0_1, 228);

        // r3 = _mm256_permutevar8x32_ps(r0, ml3);
        r3_0 = _mm_shuffle_ps(r0_0, r0_1, 238);
        r3_0 = _mm_shuffle_ps(r3_0, r3_0, 78);

        r3_1 = _mm_shuffle_ps(r0_0, r0_1, 68);
        r3_1 = _mm_shuffle_ps(r3_1, r3_1, 78);

        // i3 = _mm256_permutevar8x32_ps(i0, ml3);
        i3_0 = _mm_shuffle_ps(i0_0, i0_1, 238);
        i3_0 = _mm_shuffle_ps(i3_0, i3_0, 78);

        i3_1 = _mm_shuffle_ps(i0_0, i0_1, 68);
        i3_1 = _mm_shuffle_ps(i3_1, i3_1, 78);

        // ru = _mm256_set1_ps(matrix[0]);
        ru_0 = _mm_set1_ps(matrix[0]);
        ru_1 = _mm_set1_ps(matrix[0]);

        // iu = _mm256_set1_ps(matrix[1]);
        iu_0 = _mm_set1_ps(matrix[1]);
        iu_1 = _mm_set1_ps(matrix[1]);

        // rn = _mm256_mul_ps(r0, ru);
        rn_0 = _mm_mul_ps(r0_0, ru_0);
        rn_1 = _mm_mul_ps(r0_1, ru_1);

        // in = _mm256_mul_ps(r0, iu);
        in_0 = _mm_mul_ps(r0_0, iu_0);
        in_1 = _mm_mul_ps(r0_1, iu_1);

        // rn = _mm256_fnmadd_ps(i0, iu, rn);
        temp = _mm_mul_ps(i0_0, iu_0);
        rn_0 = _mm_sub_ps(rn_0, temp);
        temp = _mm_mul_ps(i0_1, iu_1);
        rn_1 = _mm_sub_ps(rn_1, temp);

        // in = _mm256_fmadd_ps(i0, ru, in);
        temp = _mm_mul_ps(i0_0, ru_0);
        in_0 = _mm_add_ps(in_0, temp);
        temp = _mm_mul_ps(i0_1, ru_1);
        in_1 = _mm_add_ps(in_1, temp);

        // ru = _mm256_set1_ps(matrix[2]);
        ru_0 = _mm_set1_ps(matrix[2]);
        ru_1 = _mm_set1_ps(matrix[2]);

        // iu = _mm256_set1_ps(matrix[3]);
        iu_0 = _mm_set1_ps(matrix[3]);
        iu_1 = _mm_set1_ps(matrix[3]);

        // rn = _mm256_fmadd_ps(r1, ru, rn);
        temp = _mm_mul_ps(r1_0, ru_0);
        rn_0 = _mm_add_ps(rn_0, temp);
        temp = _mm_mul_ps(r1_1, ru_1);
        rn_1 = _mm_add_ps(rn_1, temp);

        // in = _mm256_fmadd_ps(r1, iu, in);
        temp = _mm_mul_ps(r1_0, iu_0);
        in_0 = _mm_add_ps(in_0, temp);
        temp = _mm_mul_ps(r1_1, iu_1);
        in_1 = _mm_add_ps(in_1, temp);

        // rn = _mm256_fnmadd_ps(i1, iu, rn);
        temp = _mm_mul_ps(i1_0, iu_0);
        rn_0 = _mm_sub_ps(rn_0, temp);
        temp = _mm_mul_ps(i1_1, iu_1);
        rn_1 = _mm_sub_ps(rn_1, temp);

        // in = _mm256_fmadd_ps(i1, ru, in);
        temp = _mm_mul_ps(i1_0, ru_0);
        in_0 = _mm_add_ps(in_0, temp);
        temp = _mm_mul_ps(i1_1, ru_1);
        in_1 = _mm_add_ps(in_1, temp);

        // ru = _mm256_set1_ps(matrix[4]);
        ru_0 = _mm_set1_ps(matrix[4]);
        ru_1 = _mm_set1_ps(matrix[4]);

        // iu = _mm256_set1_ps(matrix[5]);
        iu_0 = _mm_set1_ps(matrix[5]);
        iu_1 = _mm_set1_ps(matrix[5]);

        // rn = _mm256_fmadd_ps(r2, ru, rn);
        temp = _mm_mul_ps(r2_0, ru_0);
        rn_0 = _mm_add_ps(rn_0, temp);
        temp = _mm_mul_ps(r2_1, ru_1);
        rn_1 = _mm_add_ps(rn_1, temp);

        // in = _mm256_fmadd_ps(r2, iu, in);
        temp = _mm_mul_ps(r2_0, iu_0);
        in_0 = _mm_add_ps(in_0, temp);
        temp = _mm_mul_ps(r2_1, iu_1);
        in_1 = _mm_add_ps(in_1, temp);

        // rn = _mm256_fnmadd_ps(i2, iu, rn);
        temp = _mm_mul_ps(i2_0, iu_0);
        rn_0 = _mm_sub_ps(rn_0, temp);
        temp = _mm_mul_ps(i2_1, iu_1);
        rn_1 = _mm_sub_ps(rn_1, temp);

        // in = _mm256_fmadd_ps(i2, ru, in);
        temp = _mm_mul_ps(i2_0, ru_0);
        in_0 = _mm_add_ps(in_0, temp);
        temp = _mm_mul_ps(i2_1, ru_1);
        in_1 = _mm_add_ps(in_1, temp);

        // ru = _mm256_set1_ps(matrix[6]);
        ru_0 = _mm_set1_ps(matrix[6]);
        ru_1 = _mm_set1_ps(matrix[6]);

        // iu = _mm256_set1_ps(matrix[7]);
        iu_0 = _mm_set1_ps(matrix[7]);
        iu_1 = _mm_set1_ps(matrix[7]);

        // rn = _mm256_fmadd_ps(r3, ru, rn);
        temp = _mm_mul_ps(r3_0, ru_0);
        rn_0 = _mm_add_ps(rn_0, temp);
        temp = _mm_mul_ps(r3_1, ru_1);
        rn_1 = _mm_add_ps(rn_1, temp);

        // in = _mm256_fmadd_ps(r3, iu, in);
        temp = _mm_mul_ps(r3_0, iu_0);
        in_0 = _mm_add_ps(in_0, temp);
        temp = _mm_mul_ps(r3_1, iu_1);
        in_1 = _mm_add_ps(in_1, temp);

        // rn = _mm256_fnmadd_ps(i3, iu, rn);
        temp = _mm_mul_ps(i3_0, iu_0);
        rn_0 = _mm_sub_ps(rn_0, temp);
        temp = _mm_mul_ps(i3_1, iu_1);
        rn_1 = _mm_sub_ps(rn_1, temp);

        // in = _mm256_fmadd_ps(i3, ru, in);
        temp = _mm_mul_ps(i3_0, ru_0);
        in_0 = _mm_add_ps(in_0, temp);
        temp = _mm_mul_ps(i3_1, ru_1);
        in_1 = _mm_add_ps(in_1, temp);

        // ru = _mm256_set1_ps(matrix[8]);
        ru_0 = _mm_set1_ps(matrix[8]);
        ru_1 = _mm_set1_ps(matrix[8]);

        // iu = _mm256_set1_ps(matrix[9]);
        iu_0 = _mm_set1_ps(matrix[9]);
        iu_1 = _mm_set1_ps(matrix[9]);

        // rm = _mm256_mul_ps(r0, ru);
        rm_0 = _mm_mul_ps(r0_0, ru_0);
        rm_1 = _mm_mul_ps(r0_1, ru_1);

        // im = _mm256_mul_ps(r0, iu);
        im_0 = _mm_mul_ps(r0_0, iu_0);
        im_1 = _mm_mul_ps(r0_1, iu_1);

        // rm = _mm256_fnmadd_ps(i0, iu, rm);
        temp = _mm_mul_ps(i0_0, iu_0);
        rm_0 = _mm_sub_ps(rm_0, temp);
        temp = _mm_mul_ps(i0_1, iu_1);
        rm_1 = _mm_sub_ps(rm_1, temp);

        // im = _mm256_fmadd_ps(i0, ru, im);
        temp = _mm_mul_ps(i0_0, ru_0);
        im_0 = _mm_add_ps(im_0, temp);
        temp = _mm_mul_ps(i0_1, ru_1);
        im_1 = _mm_add_ps(im_1, temp);

        // ru = _mm256_set1_ps(matrix[10]);
        ru_0 = _mm_set1_ps(matrix[10]);
        ru_1 = _mm_set1_ps(matrix[10]);

        // iu = _mm256_set1_ps(matrix[11]);
        iu_0 = _mm_set1_ps(matrix[11]);
        iu_1 = _mm_set1_ps(matrix[11]);

        // rm = _mm256_fmadd_ps(r1, ru, rm);
        temp = _mm_mul_ps(r1_0, ru_0);
        rm_0 = _mm_add_ps(rm_0, temp);
        temp = _mm_mul_ps(r1_1, ru_1);
        rm_1 = _mm_add_ps(rm_1, temp);

        // im = _mm256_fmadd_ps(r1, iu, im);
        temp = _mm_mul_ps(r1_0, iu_0);
        im_0 = _mm_add_ps(im_0, temp);
        temp = _mm_mul_ps(r1_1, iu_1);
        im_1 = _mm_add_ps(im_1, temp);

        // rm = _mm256_fnmadd_ps(i1, iu, rm);
        temp = _mm_mul_ps(i1_0, iu_0);
        rm_0 = _mm_sub_ps(rm_0, temp);
        temp = _mm_mul_ps(i1_1, iu_1);
        rm_1 = _mm_sub_ps(rm_1, temp);

        // im = _mm256_fmadd_ps(i1, ru, im);
        temp = _mm_mul_ps(i1_0, ru_0);
        im_0 = _mm_add_ps(im_0, temp);
        temp = _mm_mul_ps(i1_1, ru_1);
        im_1 = _mm_add_ps(im_1, temp);

        // ru = _mm256_set1_ps(matrix[12]);
        ru_0 = _mm_set1_ps(matrix[12]);
        ru_1 = _mm_set1_ps(matrix[12]);

        // iu = _mm256_set1_ps(matrix[13]);
        iu_0 = _mm_set1_ps(matrix[13]);
        iu_1 = _mm_set1_ps(matrix[13]);

        // rm = _mm256_fmadd_ps(r2, ru, rm);
        temp = _mm_mul_ps(r2_0, ru_0);
        rm_0 = _mm_add_ps(rm_0, temp);
        temp = _mm_mul_ps(r2_1, ru_1);
        rm_1 = _mm_add_ps(rm_1, temp);

        // im = _mm256_fmadd_ps(r2, iu, im);
        temp = _mm_mul_ps(r2_0, iu_0);
        im_0 = _mm_add_ps(im_0, temp);
        temp = _mm_mul_ps(r2_1, iu_1);
        im_1 = _mm_add_ps(im_1, temp);

        // rm = _mm256_fnmadd_ps(i2, iu, rm);
        temp = _mm_mul_ps(i2_0, iu_0);
        rm_0 = _mm_sub_ps(rm_0, temp);
        temp = _mm_mul_ps(i2_1, iu_1);
        rm_1 = _mm_sub_ps(rm_1, temp);

        // im = _mm256_fmadd_ps(i2, ru, im);
        temp = _mm_mul_ps(i2_0, ru_0);
        im_0 = _mm_add_ps(im_0, temp);
        temp = _mm_mul_ps(i2_1, ru_1);
        im_1 = _mm_add_ps(im_1, temp);

        // ru = _mm256_set1_ps(matrix[14]);
        ru_0 = _mm_set1_ps(matrix[14]);
        ru_1 = _mm_set1_ps(matrix[14]);

        // iu = _mm256_set1_ps(matrix[15]);
        iu_0 = _mm_set1_ps(matrix[15]);
        iu_1 = _mm_set1_ps(matrix[15]);

        // rm = _mm256_fmadd_ps(r3, ru, rm);
        temp = _mm_mul_ps(r3_0, ru_0);
        rm_0 = _mm_add_ps(rm_0, temp);
        temp = _mm_mul_ps(r3_1, ru_1);
        rm_1 = _mm_add_ps(rm_1, temp);

        // im = _mm256_fmadd_ps(r3, iu, im);
        temp = _mm_mul_ps(r3_0, iu_0);
        im_0 = _mm_add_ps(im_0, temp);
        temp = _mm_mul_ps(r3_1, iu_1);
        im_1 = _mm_add_ps(im_1, temp);

        // rm = _mm256_fnmadd_ps(i3, iu, rm);
        temp = _mm_mul_ps(i3_0, iu_0);
        rm_0 = _mm_sub_ps(rm_0, temp);
        temp = _mm_mul_ps(i3_1, iu_1);
        rm_1 = _mm_sub_ps(rm_1, temp);

        // im = _mm256_fmadd_ps(i3, ru, im);
        temp = _mm_mul_ps(i3_0, ru_0);
        im_0 = _mm_add_ps(im_0, temp);
        temp = _mm_mul_ps(i3_1, ru_1);
        im_1 = _mm_add_ps(im_1, temp);

        // rm = _mm256_permutevar8x32_ps(rm, ml1);
        rm_0 = _mm_shuffle_ps(rm_0, rm_0, 78);
        // rm_1 = nothing

        // im = _mm256_permutevar8x32_ps(im, ml1);
        im_0 = _mm_shuffle_ps(im_0, im_0, 78);
        // im_1 = nothing

        // rn = _mm256_blendv_ps(rn, rm, mb1);
        rn_0 = _mm_blendv_ps(rn_0, rm_0, mb1);
        // rn_1 = rn_1;

        // in = _mm256_blendv_ps(in, im, mb1);
        in_0 = _mm_blendv_ps(in_0, im_0, mb1);
        // in_1 = in_1;

        // ru = _mm256_set1_ps(matrix[16]);
        ru_0 = _mm_set1_ps(matrix[16]);
        ru_1 = _mm_set1_ps(matrix[16]);

        // iu = _mm256_set1_ps(matrix[17]);
        iu_0 = _mm_set1_ps(matrix[17]);
        iu_1 = _mm_set1_ps(matrix[17]);

        // rm = _mm256_mul_ps(r0, ru);
        rm_0 = _mm_mul_ps(r0_0, ru_0);
        rm_1 = _mm_mul_ps(r0_1, ru_1);

        // im = _mm256_mul_ps(r0, iu);
        im_0 = _mm_mul_ps(r0_0, iu_0);
        im_1 = _mm_mul_ps(r0_1, iu_1);

        // rm = _mm256_fnmadd_ps(i0, iu, rm);
        temp = _mm_mul_ps(i0_0, iu_0);
        rm_0 = _mm_sub_ps(rm_0, temp);
        temp = _mm_mul_ps(i0_1, iu_1);
        rm_1 = _mm_sub_ps(rm_1, temp);

        // im = _mm256_fmadd_ps(i0, ru, im);
        temp = _mm_mul_ps(i0_0, ru_0);
        im_0 = _mm_add_ps(im_0, temp);
        temp = _mm_mul_ps(i0_1, ru_1);
        im_1 = _mm_add_ps(im_1, temp);

        // ru = _mm256_set1_ps(matrix[18]);
        ru_0 = _mm_set1_ps(matrix[18]);
        ru_1 = _mm_set1_ps(matrix[18]);

        // iu = _mm256_set1_ps(matrix[19]);
        iu_0 = _mm_set1_ps(matrix[19]);
        iu_1 = _mm_set1_ps(matrix[19]);

        // rm = _mm256_fmadd_ps(r1, ru, rm);
        temp = _mm_mul_ps(r1_0, ru_0);
        rm_0 = _mm_add_ps(rm_0, temp);
        temp = _mm_mul_ps(r1_1, ru_1);
        rm_1 = _mm_add_ps(rm_1, temp);

        // im = _mm256_fmadd_ps(r1, iu, im);
        temp = _mm_mul_ps(r1_0, iu_0);
        im_0 = _mm_add_ps(im_0, temp);
        temp = _mm_mul_ps(r1_1, iu_1);
        im_1 = _mm_add_ps(im_1, temp);

        // rm = _mm256_fnmadd_ps(i1, iu, rm);
        temp = _mm_mul_ps(i1_0, iu_0);
        rm_0 = _mm_sub_ps(rm_0, temp);
        temp = _mm_mul_ps(i1_1, iu_1);
        rm_1 = _mm_sub_ps(rm_1, temp);

        // im = _mm256_fmadd_ps(i1, ru, im);
        temp = _mm_mul_ps(i1_0, ru_0);
        im_0 = _mm_add_ps(im_0, temp);
        temp = _mm_mul_ps(i1_1, ru_1);
        im_1 = _mm_add_ps(im_1, temp);

        // ru = _mm256_set1_ps(matrix[20]);
        ru_0 = _mm_set1_ps(matrix[20]);
        ru_1 = _mm_set1_ps(matrix[20]);

        // iu = _mm256_set1_ps(matrix[21]);
        iu_0 = _mm_set1_ps(matrix[21]);
        iu_1 = _mm_set1_ps(matrix[21]);

        // rm = _mm256_fmadd_ps(r2, ru, rm);
        temp = _mm_mul_ps(r2_0, ru_0);
        rm_0 = _mm_add_ps(rm_0, temp);
        temp = _mm_mul_ps(r2_1, ru_1);
        rm_1 = _mm_add_ps(rm_1, temp);

        // im = _mm256_fmadd_ps(r2, iu, im);
        temp = _mm_mul_ps(r2_0, iu_0);
        im_0 = _mm_add_ps(im_0, temp);
        temp = _mm_mul_ps(r2_1, iu_1);
        im_1 = _mm_add_ps(im_1, temp);

        // rm = _mm256_fnmadd_ps(i2, iu, rm);
        temp = _mm_mul_ps(i2_0, iu_0);
        rm_0 = _mm_sub_ps(rm_0, temp);
        temp = _mm_mul_ps(i2_1, iu_1);
        rm_1 = _mm_sub_ps(rm_1, temp);

        // im = _mm256_fmadd_ps(i2, ru, im);
        temp = _mm_mul_ps(i2_0, ru_0);
        im_0 = _mm_add_ps(im_0, temp);
        temp = _mm_mul_ps(i2_1, ru_1);
        im_1 = _mm_add_ps(im_1, temp);

        // ru = _mm256_set1_ps(matrix[22]);
        ru_0 = _mm_set1_ps(matrix[22]);
        ru_1 = _mm_set1_ps(matrix[22]);

        // iu = _mm256_set1_ps(matrix[23]);
        iu_0 = _mm_set1_ps(matrix[23]);
        iu_1 = _mm_set1_ps(matrix[23]);

        // rm = _mm256_fmadd_ps(r3, ru, rm);
        temp = _mm_mul_ps(r3_0, ru_0);
        rm_0 = _mm_add_ps(rm_0, temp);
        temp = _mm_mul_ps(r3_1, ru_1);
        rm_1 = _mm_add_ps(rm_1, temp);

        // im = _mm256_fmadd_ps(r3, iu, im);
        temp = _mm_mul_ps(r3_0, iu_0);
        im_0 = _mm_add_ps(im_0, temp);
        temp = _mm_mul_ps(r3_1, iu_1);
        im_1 = _mm_add_ps(im_1, temp);

        // rm = _mm256_fnmadd_ps(i3, iu, rm);
        temp = _mm_mul_ps(i3_0, iu_0);
        rm_0 = _mm_sub_ps(rm_0, temp);
        temp = _mm_mul_ps(i3_1, iu_1);
        rm_1 = _mm_sub_ps(rm_1, temp);

        // im = _mm256_fmadd_ps(i3, ru, im);
        temp = _mm_mul_ps(i3_0, ru_0);
        im_0 = _mm_add_ps(im_0, temp);
        temp = _mm_mul_ps(i3_1, ru_1);
        im_1 = _mm_add_ps(im_1, temp);

        // rm = _mm256_permutevar8x32_ps(rm, ml2);
        temp = _mm_shuffle_ps(rm_0, rm_1, 78);
        temp2 = _mm_shuffle_ps(temp, temp, 78);
        // re-order this to save an instruction ?
        rm_1 = _mm_shuffle_ps(rm_0, rm_1, 228);
        rm_0 = temp2;

        // im = _mm256_permutevar8x32_ps(im, ml2);
        temp = _mm_shuffle_ps(im_0, im_1, 78);
        temp2 = _mm_shuffle_ps(temp, temp, 78);
        // re-order this to save an instruction ?
        im_1 = _mm_shuffle_ps(im_0, im_1, 228);
        im_0 = temp2;

        // rn = _mm256_blendv_ps(rn, rm, mb2);
        // rn_0 = rn_0;
        rn_1 = _mm_blendv_ps(rn_1, rm_1, mb2);

        // in = _mm256_blendv_ps(in, im, mb2);
        // in_0 = in_0;
        in_1 = _mm_blendv_ps(in_1, im_1, mb2);

        // ru = _mm256_set1_ps(matrix[24]);
        ru_0 = _mm_set1_ps(matrix[24]);
        ru_1 = _mm_set1_ps(matrix[24]);

        // iu = _mm256_set1_ps(matrix[25]);
        iu_0 = _mm_set1_ps(matrix[25]);
        iu_1 = _mm_set1_ps(matrix[25]);

        // rm = _mm256_mul_ps(r0, ru);
        rm_0 = _mm_mul_ps(r0_0, ru_0);
        rm_1 = _mm_mul_ps(r0_1, ru_1);

        // im = _mm256_mul_ps(r0, iu);
        im_0 = _mm_mul_ps(r0_0, iu_0);
        im_1 = _mm_mul_ps(r0_1, iu_1);

        // rm = _mm256_fnmadd_ps(i0, iu, rm);
        temp = _mm_mul_ps(i0_0, iu_0);
        rm_0 = _mm_sub_ps(rm_0, temp);
        temp = _mm_mul_ps(i0_1, iu_1);
        rm_1 = _mm_sub_ps(rm_1, temp);

        // im = _mm256_fmadd_ps(i0, ru, im);
        temp = _mm_mul_ps(i0_0, ru_0);
        im_0 = _mm_add_ps(im_0, temp);
        temp = _mm_mul_ps(i0_1, ru_1);
        im_1 = _mm_add_ps(im_1, temp);

        // ru = _mm256_set1_ps(matrix[26]);
        ru_0 = _mm_set1_ps(matrix[26]);
        ru_1 = _mm_set1_ps(matrix[26]);

        // iu = _mm256_set1_ps(matrix[27]);
        iu_0 = _mm_set1_ps(matrix[27]);
        iu_1 = _mm_set1_ps(matrix[27]);

        // rm = _mm256_fmadd_ps(r1, ru, rm);
        temp = _mm_mul_ps(r1_0, ru_0);
        rm_0 = _mm_add_ps(rm_0, temp);
        temp = _mm_mul_ps(r1_1, ru_1);
        rm_1 = _mm_add_ps(rm_1, temp);

        // im = _mm256_fmadd_ps(r1, iu, im);
        temp = _mm_mul_ps(r1_0, iu_0);
        im_0 = _mm_add_ps(im_0, temp);
        temp = _mm_mul_ps(r1_1, iu_1);
        im_1 = _mm_add_ps(im_1, temp);

        // rm = _mm256_fnmadd_ps(i1, iu, rm);
        temp = _mm_mul_ps(i1_0, iu_0);
        rm_0 = _mm_sub_ps(rm_0, temp);
        temp = _mm_mul_ps(i1_1, iu_1);
        rm_1 = _mm_sub_ps(rm_1, temp);

        // im = _mm256_fmadd_ps(i1, ru, im);
        temp = _mm_mul_ps(i1_0, ru_0);
        im_0 = _mm_add_ps(im_0, temp);
        temp = _mm_mul_ps(i1_1, ru_1);
        im_1 = _mm_add_ps(im_1, temp);

        // ru = _mm256_set1_ps(matrix[28]);
        ru_0 = _mm_set1_ps(matrix[28]);
        ru_1 = _mm_set1_ps(matrix[28]);

        // iu = _mm256_set1_ps(matrix[29]);
        iu_0 = _mm_set1_ps(matrix[29]);
        iu_1 = _mm_set1_ps(matrix[29]);

        // rm = _mm256_fmadd_ps(r2, ru, rm);
        temp = _mm_mul_ps(r2_0, ru_0);
        rm_0 = _mm_add_ps(rm_0, temp);
        temp = _mm_mul_ps(r2_1, ru_1);
        rm_1 = _mm_add_ps(rm_1, temp);

        // im = _mm256_fmadd_ps(r2, iu, im);
        temp = _mm_mul_ps(r2_0, iu_0);
        im_0 = _mm_add_ps(im_0, temp);
        temp = _mm_mul_ps(r2_1, iu_1);
        im_1 = _mm_add_ps(im_1, temp);

        // rm = _mm256_fnmadd_ps(i2, iu, rm);
        temp = _mm_mul_ps(i2_0, iu_0);
        rm_0 = _mm_sub_ps(rm_0, temp);
        temp = _mm_mul_ps(i2_1, iu_1);
        rm_1 = _mm_sub_ps(rm_1, temp);

        // im = _mm256_fmadd_ps(i2, ru, im);
        temp = _mm_mul_ps(i2_0, ru_0);
        im_0 = _mm_add_ps(im_0, temp);
        temp = _mm_mul_ps(i2_1, ru_1);
        im_1 = _mm_add_ps(im_1, temp);

        // ru = _mm256_set1_ps(matrix[30]);
        ru_0 = _mm_set1_ps(matrix[30]);
        ru_1 = _mm_set1_ps(matrix[30]);

        // iu = _mm256_set1_ps(matrix[31]);
        iu_0 = _mm_set1_ps(matrix[31]);
        iu_1 = _mm_set1_ps(matrix[31]);

        // rm = _mm256_fmadd_ps(r3, ru, rm);
        temp = _mm_mul_ps(r3_0, ru_0);
        rm_0 = _mm_add_ps(rm_0, temp);
        temp = _mm_mul_ps(r3_1, ru_1);
        rm_1 = _mm_add_ps(rm_1, temp);

        // im = _mm256_fmadd_ps(r3, iu, im);
        temp = _mm_mul_ps(r3_0, iu_0);
        im_0 = _mm_add_ps(im_0, temp);
        temp = _mm_mul_ps(r3_1, iu_1);
        im_1 = _mm_add_ps(im_1, temp);

        // rm = _mm256_fnmadd_ps(i3, iu, rm);
        temp = _mm_mul_ps(i3_0, iu_0);
        rm_0 = _mm_sub_ps(rm_0, temp);
        temp = _mm_mul_ps(i3_1, iu_1);
        rm_1 = _mm_sub_ps(rm_1, temp);

        // im = _mm256_fmadd_ps(i3, ru, im);
        temp = _mm_mul_ps(i3_0, ru_0);
        im_0 = _mm_add_ps(im_0, temp);
        temp = _mm_mul_ps(i3_1, ru_1);
        im_1 = _mm_add_ps(im_1, temp);

        // rm = _mm256_permutevar8x32_ps(rm, ml3);
        temp = _mm_shuffle_ps(rm_0, rm_1, 238);
        temp2 = _mm_shuffle_ps(temp, temp, 78);

        temp = _mm_shuffle_ps(rm_0, rm_1, 68);
        rm_1 = _mm_shuffle_ps(temp, temp, 78);
        rm_0 = temp2;

        // im = _mm256_permutevar8x32_ps(im, ml3);
        temp = _mm_shuffle_ps(im_0, im_1, 238);
        temp2 = _mm_shuffle_ps(temp, temp, 78);

        temp = _mm_shuffle_ps(im_0, im_1, 68);
        im_1 = _mm_shuffle_ps(temp, temp, 78);
        im_0 = temp2;

        // rn = _mm256_blendv_ps(rn, rm, mb3);
        // rn_0 = rn_0;
        rn_1 = _mm_blendv_ps(rn_1, rm_1, mb3);

        // in = _mm256_blendv_ps(in, im, mb3);
        // in_0 = in_0;
        in_1 = _mm_blendv_ps(in_1, im_1, mb3);

        //_mm256_store_ps(p, rn);
        _mm_store_ps(p, rn_0);
        _mm_store_ps(p + 4, rn_1);

        //_mm256_store_ps(p + 8, in);
        _mm_store_ps(p + 8, in_0);
        _mm_store_ps(p + 12, in_1);
      }

      break;
  }
}

void Simulator2SSE::ApplyGate2HL(const unsigned int q0, const unsigned int q1,
                                 const float* matrix, State* state) const {
  __m128 mb;

  uint64_t sizei = uint64_t(1) << (num_qubits_ + 1);
  uint64_t sizej = uint64_t(1) << (q1 + 1);

  auto rstate = RawData(state);

  switch (q0) {
    case 0:
      mb = _mm_castsi128_ps(_mm_set_epi32(-1, 0, -1, 0));

      for (uint64_t i = 0; i < sizei; i += 2 * sizej) {
        for (uint64_t j = 0; j < sizej; j += 16) {
          uint64_t si = i | j;

          //__m256 r0, i0, r1, i1, r2, i2, r3, i3, ru, iu, rn, in, rm, im;
          __m128 r0_0, i0_0, r1_0, i1_0, r2_0, i2_0, r3_0, i3_0, ru_0, iu_0,
              rn_0, in_0, rm_0, im_0;
          __m128 r0_1, i0_1, r1_1, i1_1, r2_1, i2_1, r3_1, i3_1, ru_1, iu_1,
              rn_1, in_1, rm_1, im_1;

          // for fnmadd and fmadd.
          __m128 temp;

          uint64_t p = si;

          // r0 = _mm256_load_ps(rstate + p);
          r0_0 = _mm_load_ps(rstate + p);
          r0_1 = _mm_load_ps(rstate + p + 4);

          // i0 = _mm256_load_ps(rstate + p + 8);
          i0_0 = _mm_load_ps(rstate + p + 8);
          i0_1 = _mm_load_ps(rstate + p + 12);

          // r1 = _mm256_permutevar8x32_ps(r0, ml);
          r1_0 = _mm_shuffle_ps(r0_0, r0_0, 177);
          r1_1 = _mm_shuffle_ps(r0_1, r0_1, 177);

          // i1 = _mm256_permutevar8x32_ps(i0, ml);
          i1_0 = _mm_shuffle_ps(i0_0, i0_0, 177);
          i1_1 = _mm_shuffle_ps(i0_1, i0_1, 177);

          p = si | sizej;

          // r2 = _mm256_load_ps(rstate + p);
          r2_0 = _mm_load_ps(rstate + p);
          r2_1 = _mm_load_ps(rstate + p + 4);

          // i2 = _mm256_load_ps(rstate + p + 8);
          i2_0 = _mm_load_ps(rstate + p + 8);
          i2_1 = _mm_load_ps(rstate + p + 12);

          // r3 = _mm256_permutevar8x32_ps(r2, ml);
          r3_0 = _mm_shuffle_ps(r2_0, r2_0, 177);
          r3_1 = _mm_shuffle_ps(r2_1, r2_1, 177);

          // i3 = _mm256_permutevar8x32_ps(i2, ml);
          i3_0 = _mm_shuffle_ps(i2_0, i2_0, 177);
          i3_1 = _mm_shuffle_ps(i2_1, i2_1, 177);

          // ru = _mm256_set1_ps(matrix[0]);
          ru_0 = _mm_set1_ps(matrix[0]);
          ru_1 = _mm_set1_ps(matrix[0]);

          // iu = _mm256_set1_ps(matrix[1]);
          iu_0 = _mm_set1_ps(matrix[1]);
          iu_1 = _mm_set1_ps(matrix[1]);

          // rn = _mm256_mul_ps(r0, ru);
          rn_0 = _mm_mul_ps(r0_0, ru_0);
          rn_1 = _mm_mul_ps(r0_1, ru_1);

          // in = _mm256_mul_ps(r0, iu);
          in_0 = _mm_mul_ps(r0_0, iu_0);
          in_1 = _mm_mul_ps(r0_1, iu_1);

          // rn = _mm256_fnmadd_ps(i0, iu, rn);
          temp = _mm_mul_ps(i0_0, iu_0);
          rn_0 = _mm_sub_ps(rn_0, temp);
          temp = _mm_mul_ps(i0_1, iu_1);
          rn_1 = _mm_sub_ps(rn_1, temp);

          // in = _mm256_fmadd_ps(i0, ru, in);
          temp = _mm_mul_ps(i0_0, ru_0);
          in_0 = _mm_add_ps(in_0, temp);
          temp = _mm_mul_ps(i0_1, ru_1);
          in_1 = _mm_add_ps(in_1, temp);

          // ru = _mm256_set1_ps(matrix[2]);
          ru_0 = _mm_set1_ps(matrix[2]);
          ru_1 = _mm_set1_ps(matrix[2]);

          // iu = _mm256_set1_ps(matrix[3]);
          iu_0 = _mm_set1_ps(matrix[3]);
          iu_1 = _mm_set1_ps(matrix[3]);

          // rn = _mm256_fmadd_ps(r1, ru, rn);
          temp = _mm_mul_ps(r1_0, ru_0);
          rn_0 = _mm_add_ps(rn_0, temp);
          temp = _mm_mul_ps(r1_1, ru_1);
          rn_1 = _mm_add_ps(rn_1, temp);

          // in = _mm256_fmadd_ps(r1, iu, in);
          temp = _mm_mul_ps(r1_0, iu_0);
          in_0 = _mm_add_ps(in_0, temp);
          temp = _mm_mul_ps(r1_1, iu_1);
          in_1 = _mm_add_ps(in_1, temp);

          // rn = _mm256_fnmadd_ps(i1, iu, rn);
          temp = _mm_mul_ps(i1_0, iu_0);
          rn_0 = _mm_sub_ps(rn_0, temp);
          temp = _mm_mul_ps(i1_1, iu_1);
          rn_1 = _mm_sub_ps(rn_1, temp);

          // in = _mm256_fmadd_ps(i1, ru, in);
          temp = _mm_mul_ps(i1_0, ru_0);
          in_0 = _mm_add_ps(in_0, temp);
          temp = _mm_mul_ps(i1_1, ru_1);
          in_1 = _mm_add_ps(in_1, temp);

          // ru = _mm256_set1_ps(matrix[4]);
          ru_0 = _mm_set1_ps(matrix[4]);
          ru_1 = _mm_set1_ps(matrix[4]);

          // iu = _mm256_set1_ps(matrix[5]);
          iu_0 = _mm_set1_ps(matrix[5]);
          iu_1 = _mm_set1_ps(matrix[5]);

          // rn = _mm256_fmadd_ps(r2, ru, rn);
          temp = _mm_mul_ps(r2_0, ru_0);
          rn_0 = _mm_add_ps(rn_0, temp);
          temp = _mm_mul_ps(r2_1, ru_1);
          rn_1 = _mm_add_ps(rn_1, temp);

          // in = _mm256_fmadd_ps(r2, iu, in);
          temp = _mm_mul_ps(r2_0, iu_0);
          in_0 = _mm_add_ps(in_0, temp);
          temp = _mm_mul_ps(r2_1, iu_1);
          in_1 = _mm_add_ps(in_1, temp);

          // rn = _mm256_fnmadd_ps(i2, iu, rn);
          temp = _mm_mul_ps(i2_0, iu_0);
          rn_0 = _mm_sub_ps(rn_0, temp);
          temp = _mm_mul_ps(i2_1, iu_1);
          rn_1 = _mm_sub_ps(rn_1, temp);

          // in = _mm256_fmadd_ps(i2, ru, in);
          temp = _mm_mul_ps(i2_0, ru_0);
          in_0 = _mm_add_ps(in_0, temp);
          temp = _mm_mul_ps(i2_1, ru_1);
          in_1 = _mm_add_ps(in_1, temp);

          // ru = _mm256_set1_ps(matrix[6]);
          ru_0 = _mm_set1_ps(matrix[6]);
          ru_1 = _mm_set1_ps(matrix[6]);

          // iu = _mm256_set1_ps(matrix[7]);
          iu_0 = _mm_set1_ps(matrix[7]);
          iu_1 = _mm_set1_ps(matrix[7]);

          // rn = _mm256_fmadd_ps(r3, ru, rn);
          temp = _mm_mul_ps(r3_0, ru_0);
          rn_0 = _mm_add_ps(rn_0, temp);
          temp = _mm_mul_ps(r3_1, ru_1);
          rn_1 = _mm_add_ps(rn_1, temp);

          // in = _mm256_fmadd_ps(r3, iu, in);
          temp = _mm_mul_ps(r3_0, iu_0);
          in_0 = _mm_add_ps(in_0, temp);
          temp = _mm_mul_ps(r3_1, iu_1);
          in_1 = _mm_add_ps(in_1, temp);

          // rn = _mm256_fnmadd_ps(i3, iu, rn);
          temp = _mm_mul_ps(i3_0, iu_0);
          rn_0 = _mm_sub_ps(rn_0, temp);
          temp = _mm_mul_ps(i3_1, iu_1);
          rn_1 = _mm_sub_ps(rn_1, temp);

          // in = _mm256_fmadd_ps(i3, ru, in);
          temp = _mm_mul_ps(i3_0, ru_0);
          in_0 = _mm_add_ps(in_0, temp);
          temp = _mm_mul_ps(i3_1, ru_1);
          in_1 = _mm_add_ps(in_1, temp);

          // ru = _mm256_set1_ps(matrix[8]);
          ru_0 = _mm_set1_ps(matrix[8]);
          ru_1 = _mm_set1_ps(matrix[8]);

          // iu = _mm256_set1_ps(matrix[9]);
          iu_0 = _mm_set1_ps(matrix[9]);
          iu_1 = _mm_set1_ps(matrix[9]);

          // rm = _mm256_mul_ps(r0, ru);
          rm_0 = _mm_mul_ps(r0_0, ru_0);
          rm_1 = _mm_mul_ps(r0_1, ru_1);

          // im = _mm256_mul_ps(r0, iu);
          im_0 = _mm_mul_ps(r0_0, iu_0);
          im_1 = _mm_mul_ps(r0_1, iu_1);

          // rm = _mm256_fnmadd_ps(i0, iu, rm);
          temp = _mm_mul_ps(i0_0, iu_0);
          rm_0 = _mm_sub_ps(rm_0, temp);
          temp = _mm_mul_ps(i0_1, iu_1);
          rm_1 = _mm_sub_ps(rm_1, temp);

          // im = _mm256_fmadd_ps(i0, ru, im);
          temp = _mm_mul_ps(i0_0, ru_0);
          im_0 = _mm_add_ps(im_0, temp);
          temp = _mm_mul_ps(i0_1, ru_1);
          im_1 = _mm_add_ps(im_1, temp);

          // ru = _mm256_set1_ps(matrix[10]);
          ru_0 = _mm_set1_ps(matrix[10]);
          ru_1 = _mm_set1_ps(matrix[10]);

          // iu = _mm256_set1_ps(matrix[11]);
          iu_0 = _mm_set1_ps(matrix[11]);
          iu_1 = _mm_set1_ps(matrix[11]);

          // rm = _mm256_fmadd_ps(r1, ru, rm);
          temp = _mm_mul_ps(r1_0, ru_0);
          rm_0 = _mm_add_ps(rm_0, temp);
          temp = _mm_mul_ps(r1_1, ru_1);
          rm_1 = _mm_add_ps(rm_1, temp);

          // im = _mm256_fmadd_ps(r1, iu, im);
          temp = _mm_mul_ps(r1_0, iu_0);
          im_0 = _mm_add_ps(im_0, temp);
          temp = _mm_mul_ps(r1_1, iu_1);
          im_1 = _mm_add_ps(im_1, temp);

          // rm = _mm256_fnmadd_ps(i1, iu, rm);
          temp = _mm_mul_ps(i1_0, iu_0);
          rm_0 = _mm_sub_ps(rm_0, temp);
          temp = _mm_mul_ps(i1_1, iu_1);
          rm_1 = _mm_sub_ps(rm_1, temp);

          // im = _mm256_fmadd_ps(i1, ru, im);
          temp = _mm_mul_ps(i1_0, ru_0);
          im_0 = _mm_add_ps(im_0, temp);
          temp = _mm_mul_ps(i1_1, ru_1);
          im_1 = _mm_add_ps(im_1, temp);

          // ru = _mm256_set1_ps(matrix[12]);
          ru_0 = _mm_set1_ps(matrix[12]);
          ru_1 = _mm_set1_ps(matrix[12]);

          // iu = _mm256_set1_ps(matrix[13]);
          iu_0 = _mm_set1_ps(matrix[13]);
          iu_1 = _mm_set1_ps(matrix[13]);

          // rm = _mm256_fmadd_ps(r2, ru, rm);
          temp = _mm_mul_ps(r2_0, ru_0);
          rm_0 = _mm_add_ps(rm_0, temp);
          temp = _mm_mul_ps(r2_1, ru_1);
          rm_1 = _mm_add_ps(rm_1, temp);

          // im = _mm256_fmadd_ps(r2, iu, im);
          temp = _mm_mul_ps(r2_0, iu_0);
          im_0 = _mm_add_ps(im_0, temp);
          temp = _mm_mul_ps(r2_1, iu_1);
          im_1 = _mm_add_ps(im_1, temp);

          // rm = _mm256_fnmadd_ps(i2, iu, rm);
          temp = _mm_mul_ps(i2_0, iu_0);
          rm_0 = _mm_sub_ps(rm_0, temp);
          temp = _mm_mul_ps(i2_1, iu_1);
          rm_1 = _mm_sub_ps(rm_1, temp);

          // im = _mm256_fmadd_ps(i2, ru, im);
          temp = _mm_mul_ps(i2_0, ru_0);
          im_0 = _mm_add_ps(im_0, temp);
          temp = _mm_mul_ps(i2_1, ru_1);
          im_1 = _mm_add_ps(im_1, temp);

          // ru = _mm256_set1_ps(matrix[14]);
          ru_0 = _mm_set1_ps(matrix[14]);
          ru_1 = _mm_set1_ps(matrix[14]);

          // iu = _mm256_set1_ps(matrix[15]);
          iu_0 = _mm_set1_ps(matrix[15]);
          iu_1 = _mm_set1_ps(matrix[15]);

          // rm = _mm256_fmadd_ps(r3, ru, rm);
          temp = _mm_mul_ps(r3_0, ru_0);
          rm_0 = _mm_add_ps(rm_0, temp);
          temp = _mm_mul_ps(r3_1, ru_1);
          rm_1 = _mm_add_ps(rm_1, temp);

          // im = _mm256_fmadd_ps(r3, iu, im);
          temp = _mm_mul_ps(r3_0, iu_0);
          im_0 = _mm_add_ps(im_0, temp);
          temp = _mm_mul_ps(r3_1, iu_1);
          im_1 = _mm_add_ps(im_1, temp);

          // rm = _mm256_fnmadd_ps(i3, iu, rm);
          temp = _mm_mul_ps(i3_0, iu_0);
          rm_0 = _mm_sub_ps(rm_0, temp);
          temp = _mm_mul_ps(i3_1, iu_1);
          rm_1 = _mm_sub_ps(rm_1, temp);

          // im = _mm256_fmadd_ps(i3, ru, im);
          temp = _mm_mul_ps(i3_0, ru_0);
          im_0 = _mm_add_ps(im_0, temp);
          temp = _mm_mul_ps(i3_1, ru_1);
          im_1 = _mm_add_ps(im_1, temp);

          // rm = _mm256_permutevar8x32_ps(rm, ml);
          rm_0 = _mm_shuffle_ps(rm_0, rm_0, 177);
          rm_1 = _mm_shuffle_ps(rm_1, rm_1, 177);

          // im = _mm256_permutevar8x32_ps(im, ml);
          im_0 = _mm_shuffle_ps(im_0, im_0, 177);
          im_1 = _mm_shuffle_ps(im_1, im_1, 177);

          // rn = _mm256_blendv_ps(rn, rm, mb);
          rn_0 = _mm_blendv_ps(rn_0, rm_0, mb);
          rn_1 = _mm_blendv_ps(rn_1, rm_1, mb);

          // in = _mm256_blendv_ps(in, im, mb);
          in_0 = _mm_blendv_ps(in_0, im_0, mb);
          in_1 = _mm_blendv_ps(in_1, im_1, mb);

          p = si;
          //_mm256_store_ps(rstate + p, rn);
          _mm_store_ps(rstate + p, rn_0);
          _mm_store_ps(rstate + p + 4, rn_1);

          //_mm256_store_ps(rstate + p + 8, in);
          _mm_store_ps(rstate + p + 8, in_0);
          _mm_store_ps(rstate + p + 12, in_1);

          // ru = _mm256_set1_ps(matrix[16]);
          ru_0 = _mm_set1_ps(matrix[16]);
          ru_1 = _mm_set1_ps(matrix[16]);

          // iu = _mm256_set1_ps(matrix[17]);
          iu_0 = _mm_set1_ps(matrix[17]);
          iu_1 = _mm_set1_ps(matrix[17]);

          // rn = _mm256_mul_ps(r0, ru);
          rn_0 = _mm_mul_ps(r0_0, ru_0);
          rn_1 = _mm_mul_ps(r0_1, ru_1);

          // in = _mm256_mul_ps(r0, iu);
          in_0 = _mm_mul_ps(r0_0, iu_0);
          in_1 = _mm_mul_ps(r0_1, iu_1);

          // rn = _mm256_fnmadd_ps(i0, iu, rn);
          temp = _mm_mul_ps(i0_0, iu_0);
          rn_0 = _mm_sub_ps(rn_0, temp);
          temp = _mm_mul_ps(i0_1, iu_1);
          rn_1 = _mm_sub_ps(rn_1, temp);

          // in = _mm256_fmadd_ps(i0, ru, in);
          temp = _mm_mul_ps(i0_0, ru_0);
          in_0 = _mm_add_ps(in_0, temp);
          temp = _mm_mul_ps(i0_1, ru_1);
          in_1 = _mm_add_ps(in_1, temp);

          // ru = _mm256_set1_ps(matrix[18]);
          ru_0 = _mm_set1_ps(matrix[18]);
          ru_1 = _mm_set1_ps(matrix[18]);

          // iu = _mm256_set1_ps(matrix[19]);
          iu_0 = _mm_set1_ps(matrix[19]);
          iu_1 = _mm_set1_ps(matrix[19]);

          // rn = _mm256_fmadd_ps(r1, ru, rn);
          temp = _mm_mul_ps(r1_0, ru_0);
          rn_0 = _mm_add_ps(rn_0, temp);
          temp = _mm_mul_ps(r1_1, ru_1);
          rn_1 = _mm_add_ps(rn_1, temp);

          // in = _mm256_fmadd_ps(r1, iu, in);
          temp = _mm_mul_ps(r1_0, iu_0);
          in_0 = _mm_add_ps(in_0, temp);
          temp = _mm_mul_ps(r1_1, iu_1);
          in_1 = _mm_add_ps(in_1, temp);

          // rn = _mm256_fnmadd_ps(i1, iu, rn);
          temp = _mm_mul_ps(i1_0, iu_0);
          rn_0 = _mm_sub_ps(rn_0, temp);
          temp = _mm_mul_ps(i1_1, iu_1);
          rn_1 = _mm_sub_ps(rn_1, temp);

          // in = _mm256_fmadd_ps(i1, ru, in);
          temp = _mm_mul_ps(i1_0, ru_0);
          in_0 = _mm_add_ps(in_0, temp);
          temp = _mm_mul_ps(i1_1, ru_1);
          in_1 = _mm_add_ps(in_1, temp);

          // ru = _mm256_set1_ps(matrix[20]);
          ru_0 = _mm_set1_ps(matrix[20]);
          ru_1 = _mm_set1_ps(matrix[20]);

          // iu = _mm256_set1_ps(matrix[21]);
          iu_0 = _mm_set1_ps(matrix[21]);
          iu_1 = _mm_set1_ps(matrix[21]);

          // rn = _mm256_fmadd_ps(r2, ru, rn);
          temp = _mm_mul_ps(r2_0, ru_0);
          rn_0 = _mm_add_ps(rn_0, temp);
          temp = _mm_mul_ps(r2_1, ru_1);
          rn_1 = _mm_add_ps(rn_1, temp);

          // in = _mm256_fmadd_ps(r2, iu, in);
          temp = _mm_mul_ps(r2_0, iu_0);
          in_0 = _mm_add_ps(in_0, temp);
          temp = _mm_mul_ps(r2_1, iu_1);
          in_1 = _mm_add_ps(in_1, temp);

          // rn = _mm256_fnmadd_ps(i2, iu, rn);
          temp = _mm_mul_ps(i2_0, iu_0);
          rn_0 = _mm_sub_ps(rn_0, temp);
          temp = _mm_mul_ps(i2_1, iu_1);
          rn_1 = _mm_sub_ps(rn_1, temp);

          // in = _mm256_fmadd_ps(i2, ru, in);
          temp = _mm_mul_ps(i2_0, ru_0);
          in_0 = _mm_add_ps(in_0, temp);
          temp = _mm_mul_ps(i2_1, ru_1);
          in_1 = _mm_add_ps(in_1, temp);

          // ru = _mm256_set1_ps(matrix[22]);
          ru_0 = _mm_set1_ps(matrix[22]);
          ru_1 = _mm_set1_ps(matrix[22]);

          // iu = _mm256_set1_ps(matrix[23]);
          iu_0 = _mm_set1_ps(matrix[23]);
          iu_1 = _mm_set1_ps(matrix[23]);

          // rn = _mm256_fmadd_ps(r3, ru, rn);
          temp = _mm_mul_ps(r3_0, ru_0);
          rn_0 = _mm_add_ps(rn_0, temp);
          temp = _mm_mul_ps(r3_1, ru_1);
          rn_1 = _mm_add_ps(rn_1, temp);

          // in = _mm256_fmadd_ps(r3, iu, in);
          temp = _mm_mul_ps(r3_0, iu_0);
          in_0 = _mm_add_ps(in_0, temp);
          temp = _mm_mul_ps(r3_1, iu_1);
          in_1 = _mm_add_ps(in_1, temp);

          // rn = _mm256_fnmadd_ps(i3, iu, rn);
          temp = _mm_mul_ps(i3_0, iu_0);
          rn_0 = _mm_sub_ps(rn_0, temp);
          temp = _mm_mul_ps(i3_1, iu_1);
          rn_1 = _mm_sub_ps(rn_1, temp);

          // in = _mm256_fmadd_ps(i3, ru, in);
          temp = _mm_mul_ps(i3_0, ru_0);
          in_0 = _mm_add_ps(in_0, temp);
          temp = _mm_mul_ps(i3_1, ru_1);
          in_1 = _mm_add_ps(in_1, temp);

          // ru = _mm256_set1_ps(matrix[24]);
          ru_0 = _mm_set1_ps(matrix[24]);
          ru_1 = _mm_set1_ps(matrix[24]);

          // iu = _mm256_set1_ps(matrix[25]);
          iu_0 = _mm_set1_ps(matrix[25]);
          iu_1 = _mm_set1_ps(matrix[25]);

          // rm = _mm256_mul_ps(r0, ru);
          rm_0 = _mm_mul_ps(r0_0, ru_0);
          rm_1 = _mm_mul_ps(r0_1, ru_1);

          // im = _mm256_mul_ps(r0, iu);
          im_0 = _mm_mul_ps(r0_0, iu_0);
          im_1 = _mm_mul_ps(r0_1, iu_1);

          // rm = _mm256_fnmadd_ps(i0, iu, rm);
          temp = _mm_mul_ps(i0_0, iu_0);
          rm_0 = _mm_sub_ps(rm_0, temp);
          temp = _mm_mul_ps(i0_1, iu_1);
          rm_1 = _mm_sub_ps(rm_1, temp);

          // im = _mm256_fmadd_ps(i0, ru, im);
          temp = _mm_mul_ps(i0_0, ru_0);
          im_0 = _mm_add_ps(im_0, temp);
          temp = _mm_mul_ps(i0_1, ru_1);
          im_1 = _mm_add_ps(im_1, temp);

          // ru = _mm256_set1_ps(matrix[26]);
          ru_0 = _mm_set1_ps(matrix[26]);
          ru_1 = _mm_set1_ps(matrix[26]);

          // iu = _mm256_set1_ps(matrix[27]);
          iu_0 = _mm_set1_ps(matrix[27]);
          iu_1 = _mm_set1_ps(matrix[27]);

          // rm = _mm256_fmadd_ps(r1, ru, rm);
          temp = _mm_mul_ps(r1_0, ru_0);
          rm_0 = _mm_add_ps(rm_0, temp);
          temp = _mm_mul_ps(r1_1, ru_1);
          rm_1 = _mm_add_ps(rm_1, temp);

          // im = _mm256_fmadd_ps(r1, iu, im);
          temp = _mm_mul_ps(r1_0, iu_0);
          im_0 = _mm_add_ps(im_0, temp);
          temp = _mm_mul_ps(r1_1, iu_1);
          im_1 = _mm_add_ps(im_1, temp);

          // rm = _mm256_fnmadd_ps(i1, iu, rm);
          temp = _mm_mul_ps(i1_0, iu_0);
          rm_0 = _mm_sub_ps(rm_0, temp);
          temp = _mm_mul_ps(i1_1, iu_1);
          rm_1 = _mm_sub_ps(rm_1, temp);

          // im = _mm256_fmadd_ps(i1, ru, im);
          temp = _mm_mul_ps(i1_0, ru_0);
          im_0 = _mm_add_ps(im_0, temp);
          temp = _mm_mul_ps(i1_1, ru_1);
          im_1 = _mm_add_ps(im_1, temp);

          // ru = _mm256_set1_ps(matrix[28]);
          ru_0 = _mm_set1_ps(matrix[28]);
          ru_1 = _mm_set1_ps(matrix[28]);

          // iu = _mm256_set1_ps(matrix[29]);
          iu_0 = _mm_set1_ps(matrix[29]);
          iu_1 = _mm_set1_ps(matrix[29]);

          // rm = _mm256_fmadd_ps(r2, ru, rm);
          temp = _mm_mul_ps(r2_0, ru_0);
          rm_0 = _mm_add_ps(rm_0, temp);
          temp = _mm_mul_ps(r2_1, ru_1);
          rm_1 = _mm_add_ps(rm_1, temp);

          // im = _mm256_fmadd_ps(r2, iu, im);
          temp = _mm_mul_ps(r2_0, iu_0);
          im_0 = _mm_add_ps(im_0, temp);
          temp = _mm_mul_ps(r2_1, iu_1);
          im_1 = _mm_add_ps(im_1, temp);

          // rm = _mm256_fnmadd_ps(i2, iu, rm);
          temp = _mm_mul_ps(i2_0, iu_0);
          rm_0 = _mm_sub_ps(rm_0, temp);
          temp = _mm_mul_ps(i2_1, iu_1);
          rm_1 = _mm_sub_ps(rm_1, temp);

          // im = _mm256_fmadd_ps(i2, ru, im);
          temp = _mm_mul_ps(i2_0, ru_0);
          im_0 = _mm_add_ps(im_0, temp);
          temp = _mm_mul_ps(i2_1, ru_1);
          im_1 = _mm_add_ps(im_1, temp);

          // ru = _mm256_set1_ps(matrix[30]);
          ru_0 = _mm_set1_ps(matrix[30]);
          ru_1 = _mm_set1_ps(matrix[30]);

          // iu = _mm256_set1_ps(matrix[31]);
          iu_0 = _mm_set1_ps(matrix[31]);
          iu_1 = _mm_set1_ps(matrix[31]);

          // rm = _mm256_fmadd_ps(r3, ru, rm);
          temp = _mm_mul_ps(r3_0, ru_0);
          rm_0 = _mm_add_ps(rm_0, temp);
          temp = _mm_mul_ps(r3_1, ru_1);
          rm_1 = _mm_add_ps(rm_1, temp);

          // im = _mm256_fmadd_ps(r3, iu, im);
          temp = _mm_mul_ps(r3_0, iu_0);
          im_0 = _mm_add_ps(im_0, temp);
          temp = _mm_mul_ps(r3_1, iu_1);
          im_1 = _mm_add_ps(im_1, temp);

          // rm = _mm256_fnmadd_ps(i3, iu, rm);
          temp = _mm_mul_ps(i3_0, iu_0);
          rm_0 = _mm_sub_ps(rm_0, temp);
          temp = _mm_mul_ps(i3_1, iu_1);
          rm_1 = _mm_sub_ps(rm_1, temp);

          // im = _mm256_fmadd_ps(i3, ru, im);
          temp = _mm_mul_ps(i3_0, ru_0);
          im_0 = _mm_add_ps(im_0, temp);
          temp = _mm_mul_ps(i3_1, ru_1);
          im_1 = _mm_add_ps(im_1, temp);

          // rm = _mm256_permutevar8x32_ps(rm, ml);
          rm_0 = _mm_shuffle_ps(rm_0, rm_0, 177);
          rm_1 = _mm_shuffle_ps(rm_1, rm_1, 177);

          // im = _mm256_permutevar8x32_ps(im, ml);
          im_0 = _mm_shuffle_ps(im_0, im_0, 177);
          im_1 = _mm_shuffle_ps(im_1, im_1, 177);

          // rn = _mm256_blendv_ps(rn, rm, mb);
          rn_0 = _mm_blendv_ps(rn_0, rm_0, mb);
          rn_1 = _mm_blendv_ps(rn_1, rm_1, mb);

          // in = _mm256_blendv_ps(in, im, mb);
          in_0 = _mm_blendv_ps(in_0, im_0, mb);
          in_1 = _mm_blendv_ps(in_1, im_1, mb);

          p = si | sizej;
          //_mm256_store_ps(rstate + p, rn);
          _mm_store_ps(rstate + p, rn_0);
          _mm_store_ps(rstate + p + 4, rn_1);

          //_mm256_store_ps(rstate + p + 8, in);
          _mm_store_ps(rstate + p + 8, in_0);
          _mm_store_ps(rstate + p + 12, in_1);
        }
      }

      break;
    case 1:
      mb = _mm_castsi128_ps(_mm_set_epi32(-1, -1, 0, 0));

      for (uint64_t i = 0; i < sizei; i += 2 * sizej) {
        for (uint64_t j = 0; j < sizej; j += 16) {
          uint64_t si = i | j;

          //__m256 r0, i0, r1, i1, r2, i2, r3, i3, ru, iu, rn, in, rm, im;
          __m128 r0_0, i0_0, r1_0, i1_0, r2_0, i2_0, r3_0, i3_0, ru_0, iu_0,
              rn_0, in_0, rm_0, im_0;
          __m128 r0_1, i0_1, r1_1, i1_1, r2_1, i2_1, r3_1, i3_1, ru_1, iu_1,
              rn_1, in_1, rm_1, im_1;

          // for fnmadd and fmadd.
          __m128 temp;

          uint64_t p = si;

          // r0 = _mm256_load_ps(rstate + p);
          r0_0 = _mm_load_ps(rstate + p);
          r0_1 = _mm_load_ps(rstate + p + 4);

          // i0 = _mm256_load_ps(rstate + p + 8);
          i0_0 = _mm_load_ps(rstate + p + 8);
          i0_1 = _mm_load_ps(rstate + p + 12);

          // r1 = _mm256_permutevar8x32_ps(r0, ml);
          r1_0 = _mm_shuffle_ps(r0_0, r0_0, 78);
          r1_1 = _mm_shuffle_ps(r0_1, r0_1, 78);

          // i1 = _mm256_permutevar8x32_ps(i0, ml);
          i1_0 = _mm_shuffle_ps(i0_0, i0_0, 78);
          i1_1 = _mm_shuffle_ps(i0_1, i0_1, 78);

          p = si | sizej;

          // r2 = _mm256_load_ps(rstate + p);
          r2_0 = _mm_load_ps(rstate + p);
          r2_1 = _mm_load_ps(rstate + p + 4);

          // i2 = _mm256_load_ps(rstate + p + 8);
          i2_0 = _mm_load_ps(rstate + p + 8);
          i2_1 = _mm_load_ps(rstate + p + 12);

          // r3 = _mm256_permutevar8x32_ps(r2, ml);
          r3_0 = _mm_shuffle_ps(r2_0, r2_0, 78);
          r3_1 = _mm_shuffle_ps(r2_1, r2_1, 78);

          // i3 = _mm256_permutevar8x32_ps(i2, ml);
          i3_0 = _mm_shuffle_ps(i2_0, i2_0, 78);
          i3_1 = _mm_shuffle_ps(i2_1, i2_1, 78);

          // ru = _mm256_set1_ps(matrix[0]);
          ru_0 = _mm_set1_ps(matrix[0]);
          ru_1 = _mm_set1_ps(matrix[0]);

          // iu = _mm256_set1_ps(matrix[1]);
          iu_0 = _mm_set1_ps(matrix[1]);
          iu_1 = _mm_set1_ps(matrix[1]);

          // rn = _mm256_mul_ps(r0, ru);
          rn_0 = _mm_mul_ps(r0_0, ru_0);
          rn_1 = _mm_mul_ps(r0_1, ru_1);

          // in = _mm256_mul_ps(r0, iu);
          in_0 = _mm_mul_ps(r0_0, iu_0);
          in_1 = _mm_mul_ps(r0_1, iu_1);

          // rn = _mm256_fnmadd_ps(i0, iu, rn);
          temp = _mm_mul_ps(i0_0, iu_0);
          rn_0 = _mm_sub_ps(rn_0, temp);
          temp = _mm_mul_ps(i0_1, iu_1);
          rn_1 = _mm_sub_ps(rn_1, temp);

          // in = _mm256_fmadd_ps(i0, ru, in);
          temp = _mm_mul_ps(i0_0, ru_0);
          in_0 = _mm_add_ps(in_0, temp);
          temp = _mm_mul_ps(i0_1, ru_1);
          in_1 = _mm_add_ps(in_1, temp);

          // ru = _mm256_set1_ps(matrix[2]);
          ru_0 = _mm_set1_ps(matrix[2]);
          ru_1 = _mm_set1_ps(matrix[2]);

          // iu = _mm256_set1_ps(matrix[3]);
          iu_0 = _mm_set1_ps(matrix[3]);
          iu_1 = _mm_set1_ps(matrix[3]);

          // rn = _mm256_fmadd_ps(r1, ru, rn);
          temp = _mm_mul_ps(r1_0, ru_0);
          rn_0 = _mm_add_ps(rn_0, temp);
          temp = _mm_mul_ps(r1_1, ru_1);
          rn_1 = _mm_add_ps(rn_1, temp);

          // in = _mm256_fmadd_ps(r1, iu, in);
          temp = _mm_mul_ps(r1_0, iu_0);
          in_0 = _mm_add_ps(in_0, temp);
          temp = _mm_mul_ps(r1_1, iu_1);
          in_1 = _mm_add_ps(in_1, temp);

          // rn = _mm256_fnmadd_ps(i1, iu, rn);
          temp = _mm_mul_ps(i1_0, iu_0);
          rn_0 = _mm_sub_ps(rn_0, temp);
          temp = _mm_mul_ps(i1_1, iu_1);
          rn_1 = _mm_sub_ps(rn_1, temp);

          // in = _mm256_fmadd_ps(i1, ru, in);
          temp = _mm_mul_ps(i1_0, ru_0);
          in_0 = _mm_add_ps(in_0, temp);
          temp = _mm_mul_ps(i1_1, ru_1);
          in_1 = _mm_add_ps(in_1, temp);

          // ru = _mm256_set1_ps(matrix[4]);
          ru_0 = _mm_set1_ps(matrix[4]);
          ru_1 = _mm_set1_ps(matrix[4]);

          // iu = _mm256_set1_ps(matrix[5]);
          iu_0 = _mm_set1_ps(matrix[5]);
          iu_1 = _mm_set1_ps(matrix[5]);

          // rn = _mm256_fmadd_ps(r2, ru, rn);
          temp = _mm_mul_ps(r2_0, ru_0);
          rn_0 = _mm_add_ps(rn_0, temp);
          temp = _mm_mul_ps(r2_1, ru_1);
          rn_1 = _mm_add_ps(rn_1, temp);

          // in = _mm256_fmadd_ps(r2, iu, in);
          temp = _mm_mul_ps(r2_0, iu_0);
          in_0 = _mm_add_ps(in_0, temp);
          temp = _mm_mul_ps(r2_1, iu_1);
          in_1 = _mm_add_ps(in_1, temp);

          // rn = _mm256_fnmadd_ps(i2, iu, rn);
          temp = _mm_mul_ps(i2_0, iu_0);
          rn_0 = _mm_sub_ps(rn_0, temp);
          temp = _mm_mul_ps(i2_1, iu_1);
          rn_1 = _mm_sub_ps(rn_1, temp);

          // in = _mm256_fmadd_ps(i2, ru, in);
          temp = _mm_mul_ps(i2_0, ru_0);
          in_0 = _mm_add_ps(in_0, temp);
          temp = _mm_mul_ps(i2_1, ru_1);
          in_1 = _mm_add_ps(in_1, temp);

          // ru = _mm256_set1_ps(matrix[6]);
          ru_0 = _mm_set1_ps(matrix[6]);
          ru_1 = _mm_set1_ps(matrix[6]);

          // iu = _mm256_set1_ps(matrix[7]);
          iu_0 = _mm_set1_ps(matrix[7]);
          iu_1 = _mm_set1_ps(matrix[7]);

          // rn = _mm256_fmadd_ps(r3, ru, rn);
          temp = _mm_mul_ps(r3_0, ru_0);
          rn_0 = _mm_add_ps(rn_0, temp);
          temp = _mm_mul_ps(r3_1, ru_1);
          rn_1 = _mm_add_ps(rn_1, temp);

          // in = _mm256_fmadd_ps(r3, iu, in);
          temp = _mm_mul_ps(r3_0, iu_0);
          in_0 = _mm_add_ps(in_0, temp);
          temp = _mm_mul_ps(r3_1, iu_1);
          in_1 = _mm_add_ps(in_1, temp);

          // rn = _mm256_fnmadd_ps(i3, iu, rn);
          temp = _mm_mul_ps(i3_0, iu_0);
          rn_0 = _mm_sub_ps(rn_0, temp);
          temp = _mm_mul_ps(i3_1, iu_1);
          rn_1 = _mm_sub_ps(rn_1, temp);

          // in = _mm256_fmadd_ps(i3, ru, in);
          temp = _mm_mul_ps(i3_0, ru_0);
          in_0 = _mm_add_ps(in_0, temp);
          temp = _mm_mul_ps(i3_1, ru_1);
          in_1 = _mm_add_ps(in_1, temp);

          // ru = _mm256_set1_ps(matrix[8]);
          ru_0 = _mm_set1_ps(matrix[8]);
          ru_1 = _mm_set1_ps(matrix[8]);

          // iu = _mm256_set1_ps(matrix[9]);
          iu_0 = _mm_set1_ps(matrix[9]);
          iu_1 = _mm_set1_ps(matrix[9]);

          // rm = _mm256_mul_ps(r0, ru);
          rm_0 = _mm_mul_ps(r0_0, ru_0);
          rm_1 = _mm_mul_ps(r0_1, ru_1);

          // im = _mm256_mul_ps(r0, iu);
          im_0 = _mm_mul_ps(r0_0, iu_0);
          im_1 = _mm_mul_ps(r0_1, iu_1);

          // rm = _mm256_fnmadd_ps(i0, iu, rm);
          temp = _mm_mul_ps(i0_0, iu_0);
          rm_0 = _mm_sub_ps(rm_0, temp);
          temp = _mm_mul_ps(i0_1, iu_1);
          rm_1 = _mm_sub_ps(rm_1, temp);

          // im = _mm256_fmadd_ps(i0, ru, im);
          temp = _mm_mul_ps(i0_0, ru_0);
          im_0 = _mm_add_ps(im_0, temp);
          temp = _mm_mul_ps(i0_1, ru_1);
          im_1 = _mm_add_ps(im_1, temp);

          // ru = _mm256_set1_ps(matrix[10]);
          ru_0 = _mm_set1_ps(matrix[10]);
          ru_1 = _mm_set1_ps(matrix[10]);

          // iu = _mm256_set1_ps(matrix[11]);
          iu_0 = _mm_set1_ps(matrix[11]);
          iu_1 = _mm_set1_ps(matrix[11]);

          // rm = _mm256_fmadd_ps(r1, ru, rm);
          temp = _mm_mul_ps(r1_0, ru_0);
          rm_0 = _mm_add_ps(rm_0, temp);
          temp = _mm_mul_ps(r1_1, ru_1);
          rm_1 = _mm_add_ps(rm_1, temp);

          // im = _mm256_fmadd_ps(r1, iu, im);
          temp = _mm_mul_ps(r1_0, iu_0);
          im_0 = _mm_add_ps(im_0, temp);
          temp = _mm_mul_ps(r1_1, iu_1);
          im_1 = _mm_add_ps(im_1, temp);

          // rm = _mm256_fnmadd_ps(i1, iu, rm);
          temp = _mm_mul_ps(i1_0, iu_0);
          rm_0 = _mm_sub_ps(rm_0, temp);
          temp = _mm_mul_ps(i1_1, iu_1);
          rm_1 = _mm_sub_ps(rm_1, temp);

          // im = _mm256_fmadd_ps(i1, ru, im);
          temp = _mm_mul_ps(i1_0, ru_0);
          im_0 = _mm_add_ps(im_0, temp);
          temp = _mm_mul_ps(i1_1, ru_1);
          im_1 = _mm_add_ps(im_1, temp);

          // ru = _mm256_set1_ps(matrix[12]);
          ru_0 = _mm_set1_ps(matrix[12]);
          ru_1 = _mm_set1_ps(matrix[12]);

          // iu = _mm256_set1_ps(matrix[13]);
          iu_0 = _mm_set1_ps(matrix[13]);
          iu_1 = _mm_set1_ps(matrix[13]);

          // rm = _mm256_fmadd_ps(r2, ru, rm);
          temp = _mm_mul_ps(r2_0, ru_0);
          rm_0 = _mm_add_ps(rm_0, temp);
          temp = _mm_mul_ps(r2_1, ru_1);
          rm_1 = _mm_add_ps(rm_1, temp);

          // im = _mm256_fmadd_ps(r2, iu, im);
          temp = _mm_mul_ps(r2_0, iu_0);
          im_0 = _mm_add_ps(im_0, temp);
          temp = _mm_mul_ps(r2_1, iu_1);
          im_1 = _mm_add_ps(im_1, temp);

          // rm = _mm256_fnmadd_ps(i2, iu, rm);
          temp = _mm_mul_ps(i2_0, iu_0);
          rm_0 = _mm_sub_ps(rm_0, temp);
          temp = _mm_mul_ps(i2_1, iu_1);
          rm_1 = _mm_sub_ps(rm_1, temp);

          // im = _mm256_fmadd_ps(i2, ru, im);
          temp = _mm_mul_ps(i2_0, ru_0);
          im_0 = _mm_add_ps(im_0, temp);
          temp = _mm_mul_ps(i2_1, ru_1);
          im_1 = _mm_add_ps(im_1, temp);

          // ru = _mm256_set1_ps(matrix[14]);
          ru_0 = _mm_set1_ps(matrix[14]);
          ru_1 = _mm_set1_ps(matrix[14]);

          // iu = _mm256_set1_ps(matrix[15]);
          iu_0 = _mm_set1_ps(matrix[15]);
          iu_1 = _mm_set1_ps(matrix[15]);

          // rm = _mm256_fmadd_ps(r3, ru, rm);
          temp = _mm_mul_ps(r3_0, ru_0);
          rm_0 = _mm_add_ps(rm_0, temp);
          temp = _mm_mul_ps(r3_1, ru_1);
          rm_1 = _mm_add_ps(rm_1, temp);

          // im = _mm256_fmadd_ps(r3, iu, im);
          temp = _mm_mul_ps(r3_0, iu_0);
          im_0 = _mm_add_ps(im_0, temp);
          temp = _mm_mul_ps(r3_1, iu_1);
          im_1 = _mm_add_ps(im_1, temp);

          // rm = _mm256_fnmadd_ps(i3, iu, rm);
          temp = _mm_mul_ps(i3_0, iu_0);
          rm_0 = _mm_sub_ps(rm_0, temp);
          temp = _mm_mul_ps(i3_1, iu_1);
          rm_1 = _mm_sub_ps(rm_1, temp);

          // im = _mm256_fmadd_ps(i3, ru, im);
          temp = _mm_mul_ps(i3_0, ru_0);
          im_0 = _mm_add_ps(im_0, temp);
          temp = _mm_mul_ps(i3_1, ru_1);
          im_1 = _mm_add_ps(im_1, temp);

          // rm = _mm256_permutevar8x32_ps(rm, ml);
          rm_0 = _mm_shuffle_ps(rm_0, rm_0, 78);
          rm_1 = _mm_shuffle_ps(rm_1, rm_1, 78);

          // im = _mm256_permutevar8x32_ps(im, ml);
          im_0 = _mm_shuffle_ps(im_0, im_0, 78);
          im_1 = _mm_shuffle_ps(im_1, im_1, 78);

          // rn = _mm256_blendv_ps(rn, rm, mb);
          rn_0 = _mm_blendv_ps(rn_0, rm_0, mb);
          rn_1 = _mm_blendv_ps(rn_1, rm_1, mb);

          // in = _mm256_blendv_ps(in, im, mb);
          in_0 = _mm_blendv_ps(in_0, im_0, mb);
          in_1 = _mm_blendv_ps(in_1, im_1, mb);

          p = si;
          //_mm256_store_ps(rstate + p, rn);
          _mm_store_ps(rstate + p, rn_0);
          _mm_store_ps(rstate + p + 4, rn_1);

          //_mm256_store_ps(rstate + p + 8, in);
          _mm_store_ps(rstate + p + 8, in_0);
          _mm_store_ps(rstate + p + 12, in_1);

          // ru = _mm256_set1_ps(matrix[16]);
          ru_0 = _mm_set1_ps(matrix[16]);
          ru_1 = _mm_set1_ps(matrix[16]);

          // iu = _mm256_set1_ps(matrix[17]);
          iu_0 = _mm_set1_ps(matrix[17]);
          iu_1 = _mm_set1_ps(matrix[17]);

          // rn = _mm256_mul_ps(r0, ru);
          rn_0 = _mm_mul_ps(r0_0, ru_0);
          rn_1 = _mm_mul_ps(r0_1, ru_1);

          // in = _mm256_mul_ps(r0, iu);
          in_0 = _mm_mul_ps(r0_0, iu_0);
          in_1 = _mm_mul_ps(r0_1, iu_1);

          // rn = _mm256_fnmadd_ps(i0, iu, rn);
          temp = _mm_mul_ps(i0_0, iu_0);
          rn_0 = _mm_sub_ps(rn_0, temp);
          temp = _mm_mul_ps(i0_1, iu_1);
          rn_1 = _mm_sub_ps(rn_1, temp);

          // in = _mm256_fmadd_ps(i0, ru, in);
          temp = _mm_mul_ps(i0_0, ru_0);
          in_0 = _mm_add_ps(in_0, temp);
          temp = _mm_mul_ps(i0_1, ru_1);
          in_1 = _mm_add_ps(in_1, temp);

          // ru = _mm256_set1_ps(matrix[18]);
          ru_0 = _mm_set1_ps(matrix[18]);
          ru_1 = _mm_set1_ps(matrix[18]);

          // iu = _mm256_set1_ps(matrix[19]);
          iu_0 = _mm_set1_ps(matrix[19]);
          iu_1 = _mm_set1_ps(matrix[19]);

          // rn = _mm256_fmadd_ps(r1, ru, rn);
          temp = _mm_mul_ps(r1_0, ru_0);
          rn_0 = _mm_add_ps(rn_0, temp);
          temp = _mm_mul_ps(r1_1, ru_1);
          rn_1 = _mm_add_ps(rn_1, temp);

          // in = _mm256_fmadd_ps(r1, iu, in);
          temp = _mm_mul_ps(r1_0, iu_0);
          in_0 = _mm_add_ps(in_0, temp);
          temp = _mm_mul_ps(r1_1, iu_1);
          in_1 = _mm_add_ps(in_1, temp);

          // rn = _mm256_fnmadd_ps(i1, iu, rn);
          temp = _mm_mul_ps(i1_0, iu_0);
          rn_0 = _mm_sub_ps(rn_0, temp);
          temp = _mm_mul_ps(i1_1, iu_1);
          rn_1 = _mm_sub_ps(rn_1, temp);

          // in = _mm256_fmadd_ps(i1, ru, in);
          temp = _mm_mul_ps(i1_0, ru_0);
          in_0 = _mm_add_ps(in_0, temp);
          temp = _mm_mul_ps(i1_1, ru_1);
          in_1 = _mm_add_ps(in_1, temp);

          // ru = _mm256_set1_ps(matrix[20]);
          ru_0 = _mm_set1_ps(matrix[20]);
          ru_1 = _mm_set1_ps(matrix[20]);

          // iu = _mm256_set1_ps(matrix[21]);
          iu_0 = _mm_set1_ps(matrix[21]);
          iu_1 = _mm_set1_ps(matrix[21]);

          // rn = _mm256_fmadd_ps(r2, ru, rn);
          temp = _mm_mul_ps(r2_0, ru_0);
          rn_0 = _mm_add_ps(rn_0, temp);
          temp = _mm_mul_ps(r2_1, ru_1);
          rn_1 = _mm_add_ps(rn_1, temp);

          // in = _mm256_fmadd_ps(r2, iu, in);
          temp = _mm_mul_ps(r2_0, iu_0);
          in_0 = _mm_add_ps(in_0, temp);
          temp = _mm_mul_ps(r2_1, iu_1);
          in_1 = _mm_add_ps(in_1, temp);

          // rn = _mm256_fnmadd_ps(i2, iu, rn);
          temp = _mm_mul_ps(i2_0, iu_0);
          rn_0 = _mm_sub_ps(rn_0, temp);
          temp = _mm_mul_ps(i2_1, iu_1);
          rn_1 = _mm_sub_ps(rn_1, temp);

          // in = _mm256_fmadd_ps(i2, ru, in);
          temp = _mm_mul_ps(i2_0, ru_0);
          in_0 = _mm_add_ps(in_0, temp);
          temp = _mm_mul_ps(i2_1, ru_1);
          in_1 = _mm_add_ps(in_1, temp);

          // ru = _mm256_set1_ps(matrix[22]);
          ru_0 = _mm_set1_ps(matrix[22]);
          ru_1 = _mm_set1_ps(matrix[22]);

          // iu = _mm256_set1_ps(matrix[23]);
          iu_0 = _mm_set1_ps(matrix[23]);
          iu_1 = _mm_set1_ps(matrix[23]);

          // rn = _mm256_fmadd_ps(r3, ru, rn);
          temp = _mm_mul_ps(r3_0, ru_0);
          rn_0 = _mm_add_ps(rn_0, temp);
          temp = _mm_mul_ps(r3_1, ru_1);
          rn_1 = _mm_add_ps(rn_1, temp);

          // in = _mm256_fmadd_ps(r3, iu, in);
          temp = _mm_mul_ps(r3_0, iu_0);
          in_0 = _mm_add_ps(in_0, temp);
          temp = _mm_mul_ps(r3_1, iu_1);
          in_1 = _mm_add_ps(in_1, temp);

          // rn = _mm256_fnmadd_ps(i3, iu, rn);
          temp = _mm_mul_ps(i3_0, iu_0);
          rn_0 = _mm_sub_ps(rn_0, temp);
          temp = _mm_mul_ps(i3_1, iu_1);
          rn_1 = _mm_sub_ps(rn_1, temp);

          // in = _mm256_fmadd_ps(i3, ru, in);
          temp = _mm_mul_ps(i3_0, ru_0);
          in_0 = _mm_add_ps(in_0, temp);
          temp = _mm_mul_ps(i3_1, ru_1);
          in_1 = _mm_add_ps(in_1, temp);

          // ru = _mm256_set1_ps(matrix[24]);
          ru_0 = _mm_set1_ps(matrix[24]);
          ru_1 = _mm_set1_ps(matrix[24]);

          // iu = _mm256_set1_ps(matrix[25]);
          iu_0 = _mm_set1_ps(matrix[25]);
          iu_1 = _mm_set1_ps(matrix[25]);

          // rm = _mm256_mul_ps(r0, ru);
          rm_0 = _mm_mul_ps(r0_0, ru_0);
          rm_1 = _mm_mul_ps(r0_1, ru_1);

          // im = _mm256_mul_ps(r0, iu);
          im_0 = _mm_mul_ps(r0_0, iu_0);
          im_1 = _mm_mul_ps(r0_1, iu_1);

          // rm = _mm256_fnmadd_ps(i0, iu, rm);
          temp = _mm_mul_ps(i0_0, iu_0);
          rm_0 = _mm_sub_ps(rm_0, temp);
          temp = _mm_mul_ps(i0_1, iu_1);
          rm_1 = _mm_sub_ps(rm_1, temp);

          // im = _mm256_fmadd_ps(i0, ru, im);
          temp = _mm_mul_ps(i0_0, ru_0);
          im_0 = _mm_add_ps(im_0, temp);
          temp = _mm_mul_ps(i0_1, ru_1);
          im_1 = _mm_add_ps(im_1, temp);

          // ru = _mm256_set1_ps(matrix[26]);
          ru_0 = _mm_set1_ps(matrix[26]);
          ru_1 = _mm_set1_ps(matrix[26]);

          // iu = _mm256_set1_ps(matrix[27]);
          iu_0 = _mm_set1_ps(matrix[27]);
          iu_1 = _mm_set1_ps(matrix[27]);

          // rm = _mm256_fmadd_ps(r1, ru, rm);
          temp = _mm_mul_ps(r1_0, ru_0);
          rm_0 = _mm_add_ps(rm_0, temp);
          temp = _mm_mul_ps(r1_1, ru_1);
          rm_1 = _mm_add_ps(rm_1, temp);

          // im = _mm256_fmadd_ps(r1, iu, im);
          temp = _mm_mul_ps(r1_0, iu_0);
          im_0 = _mm_add_ps(im_0, temp);
          temp = _mm_mul_ps(r1_1, iu_1);
          im_1 = _mm_add_ps(im_1, temp);

          // rm = _mm256_fnmadd_ps(i1, iu, rm);
          temp = _mm_mul_ps(i1_0, iu_0);
          rm_0 = _mm_sub_ps(rm_0, temp);
          temp = _mm_mul_ps(i1_1, iu_1);
          rm_1 = _mm_sub_ps(rm_1, temp);

          // im = _mm256_fmadd_ps(i1, ru, im);
          temp = _mm_mul_ps(i1_0, ru_0);
          im_0 = _mm_add_ps(im_0, temp);
          temp = _mm_mul_ps(i1_1, ru_1);
          im_1 = _mm_add_ps(im_1, temp);

          // ru = _mm256_set1_ps(matrix[28]);
          ru_0 = _mm_set1_ps(matrix[28]);
          ru_1 = _mm_set1_ps(matrix[28]);

          // iu = _mm256_set1_ps(matrix[29]);
          iu_0 = _mm_set1_ps(matrix[29]);
          iu_1 = _mm_set1_ps(matrix[29]);

          // rm = _mm256_fmadd_ps(r2, ru, rm);
          temp = _mm_mul_ps(r2_0, ru_0);
          rm_0 = _mm_add_ps(rm_0, temp);
          temp = _mm_mul_ps(r2_1, ru_1);
          rm_1 = _mm_add_ps(rm_1, temp);

          // im = _mm256_fmadd_ps(r2, iu, im);
          temp = _mm_mul_ps(r2_0, iu_0);
          im_0 = _mm_add_ps(im_0, temp);
          temp = _mm_mul_ps(r2_1, iu_1);
          im_1 = _mm_add_ps(im_1, temp);

          // rm = _mm256_fnmadd_ps(i2, iu, rm);
          temp = _mm_mul_ps(i2_0, iu_0);
          rm_0 = _mm_sub_ps(rm_0, temp);
          temp = _mm_mul_ps(i2_1, iu_1);
          rm_1 = _mm_sub_ps(rm_1, temp);

          // im = _mm256_fmadd_ps(i2, ru, im);
          temp = _mm_mul_ps(i2_0, ru_0);
          im_0 = _mm_add_ps(im_0, temp);
          temp = _mm_mul_ps(i2_1, ru_1);
          im_1 = _mm_add_ps(im_1, temp);

          // ru = _mm256_set1_ps(matrix[30]);
          ru_0 = _mm_set1_ps(matrix[30]);
          ru_1 = _mm_set1_ps(matrix[30]);

          // iu = _mm256_set1_ps(matrix[31]);
          iu_0 = _mm_set1_ps(matrix[31]);
          iu_1 = _mm_set1_ps(matrix[31]);

          // rm = _mm256_fmadd_ps(r3, ru, rm);
          temp = _mm_mul_ps(r3_0, ru_0);
          rm_0 = _mm_add_ps(rm_0, temp);
          temp = _mm_mul_ps(r3_1, ru_1);
          rm_1 = _mm_add_ps(rm_1, temp);

          // im = _mm256_fmadd_ps(r3, iu, im);
          temp = _mm_mul_ps(r3_0, iu_0);
          im_0 = _mm_add_ps(im_0, temp);
          temp = _mm_mul_ps(r3_1, iu_1);
          im_1 = _mm_add_ps(im_1, temp);

          // rm = _mm256_fnmadd_ps(i3, iu, rm);
          temp = _mm_mul_ps(i3_0, iu_0);
          rm_0 = _mm_sub_ps(rm_0, temp);
          temp = _mm_mul_ps(i3_1, iu_1);
          rm_1 = _mm_sub_ps(rm_1, temp);

          // im = _mm256_fmadd_ps(i3, ru, im);
          temp = _mm_mul_ps(i3_0, ru_0);
          im_0 = _mm_add_ps(im_0, temp);
          temp = _mm_mul_ps(i3_1, ru_1);
          im_1 = _mm_add_ps(im_1, temp);

          // rm = _mm256_permutevar8x32_ps(rm, ml);
          rm_0 = _mm_shuffle_ps(rm_0, rm_0, 78);
          rm_1 = _mm_shuffle_ps(rm_1, rm_1, 78);

          // im = _mm256_permutevar8x32_ps(im, ml);
          im_0 = _mm_shuffle_ps(im_0, im_0, 78);
          im_1 = _mm_shuffle_ps(im_1, im_1, 78);

          // rn = _mm256_blendv_ps(rn, rm, mb);
          rn_0 = _mm_blendv_ps(rn_0, rm_0, mb);
          rn_1 = _mm_blendv_ps(rn_1, rm_1, mb);

          // in = _mm256_blendv_ps(in, im, mb);
          in_0 = _mm_blendv_ps(in_0, im_0, mb);
          in_1 = _mm_blendv_ps(in_1, im_1, mb);

          p = si | sizej;
          //_mm256_store_ps(rstate + p, rn);
          _mm_store_ps(rstate + p, rn_0);
          _mm_store_ps(rstate + p + 4, rn_1);

          //_mm256_store_ps(rstate + p + 8, in);
          _mm_store_ps(rstate + p + 8, in_0);
          _mm_store_ps(rstate + p + 12, in_1);
        }
      }

      break;
    case 2:

      mb = _mm_castsi128_ps(_mm_set_epi32(-1, -1, -1, -1));

      for (uint64_t i = 0; i < sizei; i += 2 * sizej) {
        for (uint64_t j = 0; j < sizej; j += 16) {
          uint64_t si = i | j;

          //__m256 r0, i0, r1, i1, r2, i2, r3, i3, ru, iu, rn, in, rm, im;
          __m128 r0_0, i0_0, r1_0, i1_0, r2_0, i2_0, r3_0, i3_0, ru_0, iu_0,
              rn_0, in_0, rm_0, im_0;
          __m128 r0_1, i0_1, r1_1, i1_1, r2_1, i2_1, r3_1, i3_1, ru_1, iu_1,
              rn_1, in_1, rm_1, im_1;

          // for fnmadd and fmadd.
          __m128 temp;

          uint64_t p = si;

          // r0 = _mm256_load_ps(rstate + p);
          r0_0 = _mm_load_ps(rstate + p);
          r0_1 = _mm_load_ps(rstate + p + 4);

          // i0 = _mm256_load_ps(rstate + p + 8);
          i0_0 = _mm_load_ps(rstate + p + 8);
          i0_1 = _mm_load_ps(rstate + p + 12);

          // r1 = _mm256_permutevar8x32_ps(r0, ml);
          r1_0 = r0_1;
          r1_1 = r0_0;

          // i1 = _mm256_permutevar8x32_ps(i0, ml);
          i1_0 = i0_1;
          i1_1 = i0_0;

          p = si | sizej;

          // r2 = _mm256_load_ps(rstate + p);
          r2_0 = _mm_load_ps(rstate + p);
          r2_1 = _mm_load_ps(rstate + p + 4);

          // i2 = _mm256_load_ps(rstate + p + 8);
          i2_0 = _mm_load_ps(rstate + p + 8);
          i2_1 = _mm_load_ps(rstate + p + 12);

          // r3 = _mm256_permutevar8x32_ps(r2, ml);
          r3_0 = r2_1;
          r3_1 = r2_0;

          // i3 = _mm256_permutevar8x32_ps(i2, ml);
          i3_0 = i2_1;
          i3_1 = i2_0;

          // ru = _mm256_set1_ps(matrix[0]);
          ru_0 = _mm_set1_ps(matrix[0]);
          ru_1 = _mm_set1_ps(matrix[0]);

          // iu = _mm256_set1_ps(matrix[1]);
          iu_0 = _mm_set1_ps(matrix[1]);
          iu_1 = _mm_set1_ps(matrix[1]);

          // rn = _mm256_mul_ps(r0, ru);
          rn_0 = _mm_mul_ps(r0_0, ru_0);
          rn_1 = _mm_mul_ps(r0_1, ru_1);

          // in = _mm256_mul_ps(r0, iu);
          in_0 = _mm_mul_ps(r0_0, iu_0);
          in_1 = _mm_mul_ps(r0_1, iu_1);

          // rn = _mm256_fnmadd_ps(i0, iu, rn);
          temp = _mm_mul_ps(i0_0, iu_0);
          rn_0 = _mm_sub_ps(rn_0, temp);
          temp = _mm_mul_ps(i0_1, iu_1);
          rn_1 = _mm_sub_ps(rn_1, temp);

          // in = _mm256_fmadd_ps(i0, ru, in);
          temp = _mm_mul_ps(i0_0, ru_0);
          in_0 = _mm_add_ps(in_0, temp);
          temp = _mm_mul_ps(i0_1, ru_1);
          in_1 = _mm_add_ps(in_1, temp);

          // ru = _mm256_set1_ps(matrix[2]);
          ru_0 = _mm_set1_ps(matrix[2]);
          ru_1 = _mm_set1_ps(matrix[2]);

          // iu = _mm256_set1_ps(matrix[3]);
          iu_0 = _mm_set1_ps(matrix[3]);
          iu_1 = _mm_set1_ps(matrix[3]);

          // rn = _mm256_fmadd_ps(r1, ru, rn);
          temp = _mm_mul_ps(r1_0, ru_0);
          rn_0 = _mm_add_ps(rn_0, temp);
          temp = _mm_mul_ps(r1_1, ru_1);
          rn_1 = _mm_add_ps(rn_1, temp);

          // in = _mm256_fmadd_ps(r1, iu, in);
          temp = _mm_mul_ps(r1_0, iu_0);
          in_0 = _mm_add_ps(in_0, temp);
          temp = _mm_mul_ps(r1_1, iu_1);
          in_1 = _mm_add_ps(in_1, temp);

          // rn = _mm256_fnmadd_ps(i1, iu, rn);
          temp = _mm_mul_ps(i1_0, iu_0);
          rn_0 = _mm_sub_ps(rn_0, temp);
          temp = _mm_mul_ps(i1_1, iu_1);
          rn_1 = _mm_sub_ps(rn_1, temp);

          // in = _mm256_fmadd_ps(i1, ru, in);
          temp = _mm_mul_ps(i1_0, ru_0);
          in_0 = _mm_add_ps(in_0, temp);
          temp = _mm_mul_ps(i1_1, ru_1);
          in_1 = _mm_add_ps(in_1, temp);

          // ru = _mm256_set1_ps(matrix[4]);
          ru_0 = _mm_set1_ps(matrix[4]);
          ru_1 = _mm_set1_ps(matrix[4]);

          // iu = _mm256_set1_ps(matrix[5]);
          iu_0 = _mm_set1_ps(matrix[5]);
          iu_1 = _mm_set1_ps(matrix[5]);

          // rn = _mm256_fmadd_ps(r2, ru, rn);
          temp = _mm_mul_ps(r2_0, ru_0);
          rn_0 = _mm_add_ps(rn_0, temp);
          temp = _mm_mul_ps(r2_1, ru_1);
          rn_1 = _mm_add_ps(rn_1, temp);

          // in = _mm256_fmadd_ps(r2, iu, in);
          temp = _mm_mul_ps(r2_0, iu_0);
          in_0 = _mm_add_ps(in_0, temp);
          temp = _mm_mul_ps(r2_1, iu_1);
          in_1 = _mm_add_ps(in_1, temp);

          // rn = _mm256_fnmadd_ps(i2, iu, rn);
          temp = _mm_mul_ps(i2_0, iu_0);
          rn_0 = _mm_sub_ps(rn_0, temp);
          temp = _mm_mul_ps(i2_1, iu_1);
          rn_1 = _mm_sub_ps(rn_1, temp);

          // in = _mm256_fmadd_ps(i2, ru, in);
          temp = _mm_mul_ps(i2_0, ru_0);
          in_0 = _mm_add_ps(in_0, temp);
          temp = _mm_mul_ps(i2_1, ru_1);
          in_1 = _mm_add_ps(in_1, temp);

          // ru = _mm256_set1_ps(matrix[6]);
          ru_0 = _mm_set1_ps(matrix[6]);
          ru_1 = _mm_set1_ps(matrix[6]);

          // iu = _mm256_set1_ps(matrix[7]);
          iu_0 = _mm_set1_ps(matrix[7]);
          iu_1 = _mm_set1_ps(matrix[7]);

          // rn = _mm256_fmadd_ps(r3, ru, rn);
          temp = _mm_mul_ps(r3_0, ru_0);
          rn_0 = _mm_add_ps(rn_0, temp);
          temp = _mm_mul_ps(r3_1, ru_1);
          rn_1 = _mm_add_ps(rn_1, temp);

          // in = _mm256_fmadd_ps(r3, iu, in);
          temp = _mm_mul_ps(r3_0, iu_0);
          in_0 = _mm_add_ps(in_0, temp);
          temp = _mm_mul_ps(r3_1, iu_1);
          in_1 = _mm_add_ps(in_1, temp);

          // rn = _mm256_fnmadd_ps(i3, iu, rn);
          temp = _mm_mul_ps(i3_0, iu_0);
          rn_0 = _mm_sub_ps(rn_0, temp);
          temp = _mm_mul_ps(i3_1, iu_1);
          rn_1 = _mm_sub_ps(rn_1, temp);

          // in = _mm256_fmadd_ps(i3, ru, in);
          temp = _mm_mul_ps(i3_0, ru_0);
          in_0 = _mm_add_ps(in_0, temp);
          temp = _mm_mul_ps(i3_1, ru_1);
          in_1 = _mm_add_ps(in_1, temp);

          // ru = _mm256_set1_ps(matrix[8]);
          ru_0 = _mm_set1_ps(matrix[8]);
          ru_1 = _mm_set1_ps(matrix[8]);

          // iu = _mm256_set1_ps(matrix[9]);
          iu_0 = _mm_set1_ps(matrix[9]);
          iu_1 = _mm_set1_ps(matrix[9]);

          // rm = _mm256_mul_ps(r0, ru);
          rm_0 = _mm_mul_ps(r0_0, ru_0);
          rm_1 = _mm_mul_ps(r0_1, ru_1);

          // im = _mm256_mul_ps(r0, iu);
          im_0 = _mm_mul_ps(r0_0, iu_0);
          im_1 = _mm_mul_ps(r0_1, iu_1);

          // rm = _mm256_fnmadd_ps(i0, iu, rm);
          temp = _mm_mul_ps(i0_0, iu_0);
          rm_0 = _mm_sub_ps(rm_0, temp);
          temp = _mm_mul_ps(i0_1, iu_1);
          rm_1 = _mm_sub_ps(rm_1, temp);

          // im = _mm256_fmadd_ps(i0, ru, im);
          temp = _mm_mul_ps(i0_0, ru_0);
          im_0 = _mm_add_ps(im_0, temp);
          temp = _mm_mul_ps(i0_1, ru_1);
          im_1 = _mm_add_ps(im_1, temp);

          // ru = _mm256_set1_ps(matrix[10]);
          ru_0 = _mm_set1_ps(matrix[10]);
          ru_1 = _mm_set1_ps(matrix[10]);

          // iu = _mm256_set1_ps(matrix[11]);
          iu_0 = _mm_set1_ps(matrix[11]);
          iu_1 = _mm_set1_ps(matrix[11]);

          // rm = _mm256_fmadd_ps(r1, ru, rm);
          temp = _mm_mul_ps(r1_0, ru_0);
          rm_0 = _mm_add_ps(rm_0, temp);
          temp = _mm_mul_ps(r1_1, ru_1);
          rm_1 = _mm_add_ps(rm_1, temp);

          // im = _mm256_fmadd_ps(r1, iu, im);
          temp = _mm_mul_ps(r1_0, iu_0);
          im_0 = _mm_add_ps(im_0, temp);
          temp = _mm_mul_ps(r1_1, iu_1);
          im_1 = _mm_add_ps(im_1, temp);

          // rm = _mm256_fnmadd_ps(i1, iu, rm);
          temp = _mm_mul_ps(i1_0, iu_0);
          rm_0 = _mm_sub_ps(rm_0, temp);
          temp = _mm_mul_ps(i1_1, iu_1);
          rm_1 = _mm_sub_ps(rm_1, temp);

          // im = _mm256_fmadd_ps(i1, ru, im);
          temp = _mm_mul_ps(i1_0, ru_0);
          im_0 = _mm_add_ps(im_0, temp);
          temp = _mm_mul_ps(i1_1, ru_1);
          im_1 = _mm_add_ps(im_1, temp);

          // ru = _mm256_set1_ps(matrix[12]);
          ru_0 = _mm_set1_ps(matrix[12]);
          ru_1 = _mm_set1_ps(matrix[12]);

          // iu = _mm256_set1_ps(matrix[13]);
          iu_0 = _mm_set1_ps(matrix[13]);
          iu_1 = _mm_set1_ps(matrix[13]);

          // rm = _mm256_fmadd_ps(r2, ru, rm);
          temp = _mm_mul_ps(r2_0, ru_0);
          rm_0 = _mm_add_ps(rm_0, temp);
          temp = _mm_mul_ps(r2_1, ru_1);
          rm_1 = _mm_add_ps(rm_1, temp);

          // im = _mm256_fmadd_ps(r2, iu, im);
          temp = _mm_mul_ps(r2_0, iu_0);
          im_0 = _mm_add_ps(im_0, temp);
          temp = _mm_mul_ps(r2_1, iu_1);
          im_1 = _mm_add_ps(im_1, temp);

          // rm = _mm256_fnmadd_ps(i2, iu, rm);
          temp = _mm_mul_ps(i2_0, iu_0);
          rm_0 = _mm_sub_ps(rm_0, temp);
          temp = _mm_mul_ps(i2_1, iu_1);
          rm_1 = _mm_sub_ps(rm_1, temp);

          // im = _mm256_fmadd_ps(i2, ru, im);
          temp = _mm_mul_ps(i2_0, ru_0);
          im_0 = _mm_add_ps(im_0, temp);
          temp = _mm_mul_ps(i2_1, ru_1);
          im_1 = _mm_add_ps(im_1, temp);

          // ru = _mm256_set1_ps(matrix[14]);
          ru_0 = _mm_set1_ps(matrix[14]);
          ru_1 = _mm_set1_ps(matrix[14]);

          // iu = _mm256_set1_ps(matrix[15]);
          iu_0 = _mm_set1_ps(matrix[15]);
          iu_1 = _mm_set1_ps(matrix[15]);

          // rm = _mm256_fmadd_ps(r3, ru, rm);
          temp = _mm_mul_ps(r3_0, ru_0);
          rm_0 = _mm_add_ps(rm_0, temp);
          temp = _mm_mul_ps(r3_1, ru_1);
          rm_1 = _mm_add_ps(rm_1, temp);

          // im = _mm256_fmadd_ps(r3, iu, im);
          temp = _mm_mul_ps(r3_0, iu_0);
          im_0 = _mm_add_ps(im_0, temp);
          temp = _mm_mul_ps(r3_1, iu_1);
          im_1 = _mm_add_ps(im_1, temp);

          // rm = _mm256_fnmadd_ps(i3, iu, rm);
          temp = _mm_mul_ps(i3_0, iu_0);
          rm_0 = _mm_sub_ps(rm_0, temp);
          temp = _mm_mul_ps(i3_1, iu_1);
          rm_1 = _mm_sub_ps(rm_1, temp);

          // im = _mm256_fmadd_ps(i3, ru, im);
          temp = _mm_mul_ps(i3_0, ru_0);
          im_0 = _mm_add_ps(im_0, temp);
          temp = _mm_mul_ps(i3_1, ru_1);
          im_1 = _mm_add_ps(im_1, temp);

          // rm = _mm256_permutevar8x32_ps(rm, ml);
          temp = rm_0;
          rm_0 = rm_1;
          rm_1 = temp;

          // im = _mm256_permutevar8x32_ps(im, ml);
          temp = im_0;
          im_0 = im_1;
          im_1 = temp;

          // rn = _mm256_blendv_ps(rn, rm, mb);
          // rn_0 = _mm_blendv_ps(rn_0, rm_0, mb);
          rn_1 = _mm_blendv_ps(rn_1, rm_1, mb);

          // in = _mm256_blendv_ps(in, im, mb);
          // in_0 = _mm_blendv_ps(in_0, im_0, mb);
          in_1 = _mm_blendv_ps(in_1, im_1, mb);

          p = si;
          //_mm256_store_ps(rstate + p, rn);
          _mm_store_ps(rstate + p, rn_0);
          _mm_store_ps(rstate + p + 4, rn_1);

          //_mm256_store_ps(rstate + p + 8, in);
          _mm_store_ps(rstate + p + 8, in_0);
          _mm_store_ps(rstate + p + 12, in_1);

          // ru = _mm256_set1_ps(matrix[16]);
          ru_0 = _mm_set1_ps(matrix[16]);
          ru_1 = _mm_set1_ps(matrix[16]);

          // iu = _mm256_set1_ps(matrix[17]);
          iu_0 = _mm_set1_ps(matrix[17]);
          iu_1 = _mm_set1_ps(matrix[17]);

          // rn = _mm256_mul_ps(r0, ru);
          rn_0 = _mm_mul_ps(r0_0, ru_0);
          rn_1 = _mm_mul_ps(r0_1, ru_1);

          // in = _mm256_mul_ps(r0, iu);
          in_0 = _mm_mul_ps(r0_0, iu_0);
          in_1 = _mm_mul_ps(r0_1, iu_1);

          // rn = _mm256_fnmadd_ps(i0, iu, rn);
          temp = _mm_mul_ps(i0_0, iu_0);
          rn_0 = _mm_sub_ps(rn_0, temp);
          temp = _mm_mul_ps(i0_1, iu_1);
          rn_1 = _mm_sub_ps(rn_1, temp);

          // in = _mm256_fmadd_ps(i0, ru, in);
          temp = _mm_mul_ps(i0_0, ru_0);
          in_0 = _mm_add_ps(in_0, temp);
          temp = _mm_mul_ps(i0_1, ru_1);
          in_1 = _mm_add_ps(in_1, temp);

          // ru = _mm256_set1_ps(matrix[18]);
          ru_0 = _mm_set1_ps(matrix[18]);
          ru_1 = _mm_set1_ps(matrix[18]);

          // iu = _mm256_set1_ps(matrix[19]);
          iu_0 = _mm_set1_ps(matrix[19]);
          iu_1 = _mm_set1_ps(matrix[19]);

          // rn = _mm256_fmadd_ps(r1, ru, rn);
          temp = _mm_mul_ps(r1_0, ru_0);
          rn_0 = _mm_add_ps(rn_0, temp);
          temp = _mm_mul_ps(r1_1, ru_1);
          rn_1 = _mm_add_ps(rn_1, temp);

          // in = _mm256_fmadd_ps(r1, iu, in);
          temp = _mm_mul_ps(r1_0, iu_0);
          in_0 = _mm_add_ps(in_0, temp);
          temp = _mm_mul_ps(r1_1, iu_1);
          in_1 = _mm_add_ps(in_1, temp);

          // rn = _mm256_fnmadd_ps(i1, iu, rn);
          temp = _mm_mul_ps(i1_0, iu_0);
          rn_0 = _mm_sub_ps(rn_0, temp);
          temp = _mm_mul_ps(i1_1, iu_1);
          rn_1 = _mm_sub_ps(rn_1, temp);

          // in = _mm256_fmadd_ps(i1, ru, in);
          temp = _mm_mul_ps(i1_0, ru_0);
          in_0 = _mm_add_ps(in_0, temp);
          temp = _mm_mul_ps(i1_1, ru_1);
          in_1 = _mm_add_ps(in_1, temp);

          // ru = _mm256_set1_ps(matrix[20]);
          ru_0 = _mm_set1_ps(matrix[20]);
          ru_1 = _mm_set1_ps(matrix[20]);

          // iu = _mm256_set1_ps(matrix[21]);
          iu_0 = _mm_set1_ps(matrix[21]);
          iu_1 = _mm_set1_ps(matrix[21]);

          // rn = _mm256_fmadd_ps(r2, ru, rn);
          temp = _mm_mul_ps(r2_0, ru_0);
          rn_0 = _mm_add_ps(rn_0, temp);
          temp = _mm_mul_ps(r2_1, ru_1);
          rn_1 = _mm_add_ps(rn_1, temp);

          // in = _mm256_fmadd_ps(r2, iu, in);
          temp = _mm_mul_ps(r2_0, iu_0);
          in_0 = _mm_add_ps(in_0, temp);
          temp = _mm_mul_ps(r2_1, iu_1);
          in_1 = _mm_add_ps(in_1, temp);

          // rn = _mm256_fnmadd_ps(i2, iu, rn);
          temp = _mm_mul_ps(i2_0, iu_0);
          rn_0 = _mm_sub_ps(rn_0, temp);
          temp = _mm_mul_ps(i2_1, iu_1);
          rn_1 = _mm_sub_ps(rn_1, temp);

          // in = _mm256_fmadd_ps(i2, ru, in);
          temp = _mm_mul_ps(i2_0, ru_0);
          in_0 = _mm_add_ps(in_0, temp);
          temp = _mm_mul_ps(i2_1, ru_1);
          in_1 = _mm_add_ps(in_1, temp);

          // ru = _mm256_set1_ps(matrix[22]);
          ru_0 = _mm_set1_ps(matrix[22]);
          ru_1 = _mm_set1_ps(matrix[22]);

          // iu = _mm256_set1_ps(matrix[23]);
          iu_0 = _mm_set1_ps(matrix[23]);
          iu_1 = _mm_set1_ps(matrix[23]);

          // rn = _mm256_fmadd_ps(r3, ru, rn);
          temp = _mm_mul_ps(r3_0, ru_0);
          rn_0 = _mm_add_ps(rn_0, temp);
          temp = _mm_mul_ps(r3_1, ru_1);
          rn_1 = _mm_add_ps(rn_1, temp);

          // in = _mm256_fmadd_ps(r3, iu, in);
          temp = _mm_mul_ps(r3_0, iu_0);
          in_0 = _mm_add_ps(in_0, temp);
          temp = _mm_mul_ps(r3_1, iu_1);
          in_1 = _mm_add_ps(in_1, temp);

          // rn = _mm256_fnmadd_ps(i3, iu, rn);
          temp = _mm_mul_ps(i3_0, iu_0);
          rn_0 = _mm_sub_ps(rn_0, temp);
          temp = _mm_mul_ps(i3_1, iu_1);
          rn_1 = _mm_sub_ps(rn_1, temp);

          // in = _mm256_fmadd_ps(i3, ru, in);
          temp = _mm_mul_ps(i3_0, ru_0);
          in_0 = _mm_add_ps(in_0, temp);
          temp = _mm_mul_ps(i3_1, ru_1);
          in_1 = _mm_add_ps(in_1, temp);

          // ru = _mm256_set1_ps(matrix[24]);
          ru_0 = _mm_set1_ps(matrix[24]);
          ru_1 = _mm_set1_ps(matrix[24]);

          // iu = _mm256_set1_ps(matrix[25]);
          iu_0 = _mm_set1_ps(matrix[25]);
          iu_1 = _mm_set1_ps(matrix[25]);

          // rm = _mm256_mul_ps(r0, ru);
          rm_0 = _mm_mul_ps(r0_0, ru_0);
          rm_1 = _mm_mul_ps(r0_1, ru_1);

          // im = _mm256_mul_ps(r0, iu);
          im_0 = _mm_mul_ps(r0_0, iu_0);
          im_1 = _mm_mul_ps(r0_1, iu_1);

          // rm = _mm256_fnmadd_ps(i0, iu, rm);
          temp = _mm_mul_ps(i0_0, iu_0);
          rm_0 = _mm_sub_ps(rm_0, temp);
          temp = _mm_mul_ps(i0_1, iu_1);
          rm_1 = _mm_sub_ps(rm_1, temp);

          // im = _mm256_fmadd_ps(i0, ru, im);
          temp = _mm_mul_ps(i0_0, ru_0);
          im_0 = _mm_add_ps(im_0, temp);
          temp = _mm_mul_ps(i0_1, ru_1);
          im_1 = _mm_add_ps(im_1, temp);

          // ru = _mm256_set1_ps(matrix[26]);
          ru_0 = _mm_set1_ps(matrix[26]);
          ru_1 = _mm_set1_ps(matrix[26]);

          // iu = _mm256_set1_ps(matrix[27]);
          iu_0 = _mm_set1_ps(matrix[27]);
          iu_1 = _mm_set1_ps(matrix[27]);

          // rm = _mm256_fmadd_ps(r1, ru, rm);
          temp = _mm_mul_ps(r1_0, ru_0);
          rm_0 = _mm_add_ps(rm_0, temp);
          temp = _mm_mul_ps(r1_1, ru_1);
          rm_1 = _mm_add_ps(rm_1, temp);

          // im = _mm256_fmadd_ps(r1, iu, im);
          temp = _mm_mul_ps(r1_0, iu_0);
          im_0 = _mm_add_ps(im_0, temp);
          temp = _mm_mul_ps(r1_1, iu_1);
          im_1 = _mm_add_ps(im_1, temp);

          // rm = _mm256_fnmadd_ps(i1, iu, rm);
          temp = _mm_mul_ps(i1_0, iu_0);
          rm_0 = _mm_sub_ps(rm_0, temp);
          temp = _mm_mul_ps(i1_1, iu_1);
          rm_1 = _mm_sub_ps(rm_1, temp);

          // im = _mm256_fmadd_ps(i1, ru, im);
          temp = _mm_mul_ps(i1_0, ru_0);
          im_0 = _mm_add_ps(im_0, temp);
          temp = _mm_mul_ps(i1_1, ru_1);
          im_1 = _mm_add_ps(im_1, temp);

          // ru = _mm256_set1_ps(matrix[28]);
          ru_0 = _mm_set1_ps(matrix[28]);
          ru_1 = _mm_set1_ps(matrix[28]);

          // iu = _mm256_set1_ps(matrix[29]);
          iu_0 = _mm_set1_ps(matrix[29]);
          iu_1 = _mm_set1_ps(matrix[29]);

          // rm = _mm256_fmadd_ps(r2, ru, rm);
          temp = _mm_mul_ps(r2_0, ru_0);
          rm_0 = _mm_add_ps(rm_0, temp);
          temp = _mm_mul_ps(r2_1, ru_1);
          rm_1 = _mm_add_ps(rm_1, temp);

          // im = _mm256_fmadd_ps(r2, iu, im);
          temp = _mm_mul_ps(r2_0, iu_0);
          im_0 = _mm_add_ps(im_0, temp);
          temp = _mm_mul_ps(r2_1, iu_1);
          im_1 = _mm_add_ps(im_1, temp);

          // rm = _mm256_fnmadd_ps(i2, iu, rm);
          temp = _mm_mul_ps(i2_0, iu_0);
          rm_0 = _mm_sub_ps(rm_0, temp);
          temp = _mm_mul_ps(i2_1, iu_1);
          rm_1 = _mm_sub_ps(rm_1, temp);

          // im = _mm256_fmadd_ps(i2, ru, im);
          temp = _mm_mul_ps(i2_0, ru_0);
          im_0 = _mm_add_ps(im_0, temp);
          temp = _mm_mul_ps(i2_1, ru_1);
          im_1 = _mm_add_ps(im_1, temp);

          // ru = _mm256_set1_ps(matrix[30]);
          ru_0 = _mm_set1_ps(matrix[30]);
          ru_1 = _mm_set1_ps(matrix[30]);

          // iu = _mm256_set1_ps(matrix[31]);
          iu_0 = _mm_set1_ps(matrix[31]);
          iu_1 = _mm_set1_ps(matrix[31]);

          // rm = _mm256_fmadd_ps(r3, ru, rm);
          temp = _mm_mul_ps(r3_0, ru_0);
          rm_0 = _mm_add_ps(rm_0, temp);
          temp = _mm_mul_ps(r3_1, ru_1);
          rm_1 = _mm_add_ps(rm_1, temp);

          // im = _mm256_fmadd_ps(r3, iu, im);
          temp = _mm_mul_ps(r3_0, iu_0);
          im_0 = _mm_add_ps(im_0, temp);
          temp = _mm_mul_ps(r3_1, iu_1);
          im_1 = _mm_add_ps(im_1, temp);

          // rm = _mm256_fnmadd_ps(i3, iu, rm);
          temp = _mm_mul_ps(i3_0, iu_0);
          rm_0 = _mm_sub_ps(rm_0, temp);
          temp = _mm_mul_ps(i3_1, iu_1);
          rm_1 = _mm_sub_ps(rm_1, temp);

          // im = _mm256_fmadd_ps(i3, ru, im);
          temp = _mm_mul_ps(i3_0, ru_0);
          im_0 = _mm_add_ps(im_0, temp);
          temp = _mm_mul_ps(i3_1, ru_1);
          im_1 = _mm_add_ps(im_1, temp);

          // rm = _mm256_permutevar8x32_ps(rm, ml);
          temp = rm_0;
          rm_0 = rm_1;
          rm_1 = temp;

          // im = _mm256_permutevar8x32_ps(im, ml);
          temp = im_0;
          im_0 = im_1;
          im_1 = temp;

          // rn = _mm256_blendv_ps(rn, rm, mb);
          // rn_0 = _mm_blendv_ps(rn_0, rm_0, mb);
          rn_1 = _mm_blendv_ps(rn_1, rm_1, mb);

          // in = _mm256_blendv_ps(in, im, mb);
          // in_0 = _mm_blendv_ps(in_0, im_0, mb);
          in_1 = _mm_blendv_ps(in_1, im_1, mb);

          p = si | sizej;
          //_mm256_store_ps(rstate + p, rn);
          _mm_store_ps(rstate + p, rn_0);
          _mm_store_ps(rstate + p + 4, rn_1);

          //_mm256_store_ps(rstate + p + 8, in);
          _mm_store_ps(rstate + p + 8, in_0);
          _mm_store_ps(rstate + p + 12, in_1);
        }
      }

      break;
  }
}

}  // namespace qsim
}  // namespace tfq

#endif
