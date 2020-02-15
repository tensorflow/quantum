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

#ifdef __AVX2__

#include "tensorflow_quantum/core/qsim/state_space_avx.h"

#include <immintrin.h>

#include <cmath>
#include <cstdint>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_quantum/core/qsim/util.h"

namespace tfq {
namespace qsim {

StateSpaceAVX::StateSpaceAVX(const uint64_t num_qubits,
                             const uint64_t num_threads)
    : StateSpace(num_qubits, num_threads) {}

StateSpaceAVX::~StateSpaceAVX() { DeleteState(); }

StateSpaceType StateSpaceAVX::GetType() const {
  return StateSpaceType::AVX;
}

void StateSpaceAVX::CreateState() {
  SetRawState((float*)qsim::_aligned_malloc(sizeof(float) * 2 * GetDimension()));
}

void StateSpaceAVX::DeleteState() { qsim::_aligned_free(GetRawState()); }

StateSpace* StateSpaceAVX::Clone() const {
  StateSpaceAVX* state_copy =
      new StateSpaceAVX(GetNumQubits(), GetNumThreads());
  return state_copy;
}

void StateSpaceAVX::CopyFrom(const StateSpace& other) const {
  auto data = GetRawState();
  auto copy_data = other.GetRawState();

  uint64_t size2 = GetDimension() / 8;
  __m256 tmp1, tmp2;
  for (uint64_t i = 0; i < size2; ++i) {
    tmp1 = _mm256_load_ps(copy_data + 16 * i);
    tmp2 = _mm256_load_ps(copy_data + 16 * i + 8);

    _mm256_store_ps(data + 16 * i, tmp1);
    _mm256_store_ps(data + 16 * i + 8, tmp2);
  }
}

void StateSpaceAVX::ApplyGate2(const unsigned int q0, const unsigned int q1,
                               const float* matrix) {
  // Assume q0 < q1.
  if (q0 > 2) {
    ApplyGate2HH(q0, q1, matrix);
  } else if (q1 > 2) {
    ApplyGate2HL(q0, q1, matrix);
  } else {
    ApplyGate2LL(q0, q1, matrix);
  }
}

tensorflow::Status StateSpaceAVX::ApplyGate1(const float* matrix) {
  return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                            "AVX simulator doesn't support small circuits.");
}

void StateSpaceAVX::SetStateZero() {
  uint64_t size2 = GetDimension() / 8;
  __m256 val0 = _mm256_setzero_ps();

  auto data = GetRawState();

  for (uint64_t i = 0; i < size2; ++i) {
    _mm256_store_ps(data + 16 * i, val0);
    _mm256_store_ps(data + 16 * i + 8, val0);
  }
  data[0] = 1;
}

float StateSpaceAVX::GetRealInnerProduct(const StateSpace& other) const {
  uint64_t size2 = GetDimension() / 4;
  __m256d expv = _mm256_setzero_pd();
  __m256d rs, is;

  auto statea = GetRawState();
  auto stateb = other.GetRawState();

  // Currently not a thread safe implementation of inner product!
  for (uint64_t i = 0; i < size2; ++i) {
    rs = _mm256_cvtps_pd(_mm_load_ps(statea + 8 * i));
    is = _mm256_cvtps_pd(_mm_load_ps(stateb + 8 * i));
    expv = _mm256_fmadd_pd(rs, is, expv);
    rs = _mm256_cvtps_pd(_mm_load_ps(statea + 8 * i + 4));
    is = _mm256_cvtps_pd(_mm_load_ps(stateb + 8 * i + 4));
    expv = _mm256_fmadd_pd(rs, is, expv);
  }
  double buffer[4];
  _mm256_storeu_pd(buffer, expv);
  return (float)(buffer[0] + buffer[1] + buffer[2] + buffer[3]);
}

std::complex<float> StateSpaceAVX::GetAmpl(const uint64_t i) const {
  uint64_t p = (16 * (i / 8)) + (i % 8);
  return std::complex<float>(GetRawState()[p], GetRawState()[p + 8]);
}

void StateSpaceAVX::SetAmpl(const uint64_t i, const std::complex<float>& val) {
  uint64_t p = (16 * (i / 8)) + (i % 8);
  GetRawState()[p] = val.real();
  GetRawState()[p + 8] = val.imag();
}

void StateSpaceAVX::ApplyGate2HH(const unsigned int q0, const unsigned int q1,
                                 const float* matrix) {
  uint64_t sizei = uint64_t(1) << (GetNumQubits() + 1);
  uint64_t sizej = uint64_t(1) << (q1 + 1);
  uint64_t sizek = uint64_t(1) << (q0 + 1);

  auto rstate = GetRawState();

  for (uint64_t i = 0; i < sizei; i += 2 * sizej) {
    for (uint64_t j = 0; j < sizej; j += 2 * sizek) {
      for (uint64_t k = 0; k < sizek; k += 16) {
        uint64_t si = i | j | k;

        __m256 r0, i0, r1, i1, r2, i2, r3, i3, ru, iu, rn, in;

        uint64_t p = si;
        r0 = _mm256_load_ps(rstate + p);
        i0 = _mm256_load_ps(rstate + p + 8);
        ru = _mm256_set1_ps(matrix[0]);
        iu = _mm256_set1_ps(matrix[1]);
        rn = _mm256_mul_ps(r0, ru);
        in = _mm256_mul_ps(r0, iu);
        rn = _mm256_fnmadd_ps(i0, iu, rn);
        in = _mm256_fmadd_ps(i0, ru, in);
        p = si | sizek;
        r1 = _mm256_load_ps(rstate + p);
        i1 = _mm256_load_ps(rstate + p + 8);
        ru = _mm256_set1_ps(matrix[2]);
        iu = _mm256_set1_ps(matrix[3]);
        rn = _mm256_fmadd_ps(r1, ru, rn);
        in = _mm256_fmadd_ps(r1, iu, in);
        rn = _mm256_fnmadd_ps(i1, iu, rn);
        in = _mm256_fmadd_ps(i1, ru, in);
        p = si | sizej;
        r2 = _mm256_load_ps(rstate + p);
        i2 = _mm256_load_ps(rstate + p + 8);
        ru = _mm256_set1_ps(matrix[4]);
        iu = _mm256_set1_ps(matrix[5]);
        rn = _mm256_fmadd_ps(r2, ru, rn);
        in = _mm256_fmadd_ps(r2, iu, in);
        rn = _mm256_fnmadd_ps(i2, iu, rn);
        in = _mm256_fmadd_ps(i2, ru, in);
        p |= sizek;
        r3 = _mm256_load_ps(rstate + p);
        i3 = _mm256_load_ps(rstate + p + 8);
        ru = _mm256_set1_ps(matrix[6]);
        iu = _mm256_set1_ps(matrix[7]);
        rn = _mm256_fmadd_ps(r3, ru, rn);
        in = _mm256_fmadd_ps(r3, iu, in);
        rn = _mm256_fnmadd_ps(i3, iu, rn);
        in = _mm256_fmadd_ps(i3, ru, in);
        p = si;
        _mm256_store_ps(rstate + p, rn);
        _mm256_store_ps(rstate + p + 8, in);

        ru = _mm256_set1_ps(matrix[8]);
        iu = _mm256_set1_ps(matrix[9]);
        rn = _mm256_mul_ps(r0, ru);
        in = _mm256_mul_ps(r0, iu);
        rn = _mm256_fnmadd_ps(i0, iu, rn);
        in = _mm256_fmadd_ps(i0, ru, in);
        ru = _mm256_set1_ps(matrix[10]);
        iu = _mm256_set1_ps(matrix[11]);
        rn = _mm256_fmadd_ps(r1, ru, rn);
        in = _mm256_fmadd_ps(r1, iu, in);
        rn = _mm256_fnmadd_ps(i1, iu, rn);
        in = _mm256_fmadd_ps(i1, ru, in);
        ru = _mm256_set1_ps(matrix[12]);
        iu = _mm256_set1_ps(matrix[13]);
        rn = _mm256_fmadd_ps(r2, ru, rn);
        in = _mm256_fmadd_ps(r2, iu, in);
        rn = _mm256_fnmadd_ps(i2, iu, rn);
        in = _mm256_fmadd_ps(i2, ru, in);
        ru = _mm256_set1_ps(matrix[14]);
        iu = _mm256_set1_ps(matrix[15]);
        rn = _mm256_fmadd_ps(r3, ru, rn);
        in = _mm256_fmadd_ps(r3, iu, in);
        rn = _mm256_fnmadd_ps(i3, iu, rn);
        in = _mm256_fmadd_ps(i3, ru, in);
        p = si | sizek;
        _mm256_store_ps(rstate + p, rn);
        _mm256_store_ps(rstate + p + 8, in);

        ru = _mm256_set1_ps(matrix[16]);
        iu = _mm256_set1_ps(matrix[17]);
        rn = _mm256_mul_ps(r0, ru);
        in = _mm256_mul_ps(r0, iu);
        rn = _mm256_fnmadd_ps(i0, iu, rn);
        in = _mm256_fmadd_ps(i0, ru, in);
        ru = _mm256_set1_ps(matrix[18]);
        iu = _mm256_set1_ps(matrix[19]);
        rn = _mm256_fmadd_ps(r1, ru, rn);
        in = _mm256_fmadd_ps(r1, iu, in);
        rn = _mm256_fnmadd_ps(i1, iu, rn);
        in = _mm256_fmadd_ps(i1, ru, in);
        ru = _mm256_set1_ps(matrix[20]);
        iu = _mm256_set1_ps(matrix[21]);
        rn = _mm256_fmadd_ps(r2, ru, rn);
        in = _mm256_fmadd_ps(r2, iu, in);
        rn = _mm256_fnmadd_ps(i2, iu, rn);
        in = _mm256_fmadd_ps(i2, ru, in);
        ru = _mm256_set1_ps(matrix[22]);
        iu = _mm256_set1_ps(matrix[23]);
        rn = _mm256_fmadd_ps(r3, ru, rn);
        in = _mm256_fmadd_ps(r3, iu, in);
        rn = _mm256_fnmadd_ps(i3, iu, rn);
        in = _mm256_fmadd_ps(i3, ru, in);
        p = si | sizej;
        _mm256_store_ps(rstate + p, rn);
        _mm256_store_ps(rstate + p + 8, in);

        ru = _mm256_set1_ps(matrix[24]);
        iu = _mm256_set1_ps(matrix[25]);
        rn = _mm256_mul_ps(r0, ru);
        in = _mm256_mul_ps(r0, iu);
        rn = _mm256_fnmadd_ps(i0, iu, rn);
        in = _mm256_fmadd_ps(i0, ru, in);
        ru = _mm256_set1_ps(matrix[26]);
        iu = _mm256_set1_ps(matrix[27]);
        rn = _mm256_fmadd_ps(r1, ru, rn);
        in = _mm256_fmadd_ps(r1, iu, in);
        rn = _mm256_fnmadd_ps(i1, iu, rn);
        in = _mm256_fmadd_ps(i1, ru, in);
        ru = _mm256_set1_ps(matrix[28]);
        iu = _mm256_set1_ps(matrix[29]);
        rn = _mm256_fmadd_ps(r2, ru, rn);
        in = _mm256_fmadd_ps(r2, iu, in);
        rn = _mm256_fnmadd_ps(i2, iu, rn);
        in = _mm256_fmadd_ps(i2, ru, in);
        ru = _mm256_set1_ps(matrix[30]);
        iu = _mm256_set1_ps(matrix[31]);
        rn = _mm256_fmadd_ps(r3, ru, rn);
        in = _mm256_fmadd_ps(r3, iu, in);
        rn = _mm256_fnmadd_ps(i3, iu, rn);
        in = _mm256_fmadd_ps(i3, ru, in);
        p |= sizek;
        _mm256_store_ps(rstate + p, rn);
        _mm256_store_ps(rstate + p + 8, in);
      }
    }
  }
}

void StateSpaceAVX::ApplyGate2HL(const unsigned int q0, const unsigned int q1,
                                 const float* matrix) {
  __m256 mb;
  __m256i ml;

  uint64_t sizei = uint64_t(1) << (GetNumQubits() + 1);
  uint64_t sizej = uint64_t(1) << (q1 + 1);

  auto rstate = GetRawState();

  switch (q0) {
    case 0:
      ml = _mm256_set_epi32(6, 7, 4, 5, 2, 3, 0, 1);
      mb = _mm256_castsi256_ps(_mm256_set_epi32(-1, 0, -1, 0, -1, 0, -1, 0));
      break;
    case 1:
      ml = _mm256_set_epi32(5, 4, 7, 6, 1, 0, 3, 2);
      mb = _mm256_castsi256_ps(_mm256_set_epi32(-1, -1, 0, 0, -1, -1, 0, 0));
      break;
    case 2:
      ml = _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4);
      mb = _mm256_castsi256_ps(_mm256_set_epi32(-1, -1, -1, -1, 0, 0, 0, 0));
      break;
  }

  for (uint64_t i = 0; i < sizei; i += 2 * sizej) {
    for (uint64_t j = 0; j < sizej; j += 16) {
      uint64_t si = i | j;

      __m256 r0, i0, r1, i1, r2, i2, r3, i3, ru, iu, rn, in, rm, im;

      uint64_t p = si;

      r0 = _mm256_load_ps(rstate + p);
      i0 = _mm256_load_ps(rstate + p + 8);

      r1 = _mm256_permutevar8x32_ps(r0, ml);
      i1 = _mm256_permutevar8x32_ps(i0, ml);

      p = si | sizej;

      r2 = _mm256_load_ps(rstate + p);
      i2 = _mm256_load_ps(rstate + p + 8);

      r3 = _mm256_permutevar8x32_ps(r2, ml);
      i3 = _mm256_permutevar8x32_ps(i2, ml);

      ru = _mm256_set1_ps(matrix[0]);
      iu = _mm256_set1_ps(matrix[1]);
      rn = _mm256_mul_ps(r0, ru);
      in = _mm256_mul_ps(r0, iu);
      rn = _mm256_fnmadd_ps(i0, iu, rn);
      in = _mm256_fmadd_ps(i0, ru, in);
      ru = _mm256_set1_ps(matrix[2]);
      iu = _mm256_set1_ps(matrix[3]);
      rn = _mm256_fmadd_ps(r1, ru, rn);
      in = _mm256_fmadd_ps(r1, iu, in);
      rn = _mm256_fnmadd_ps(i1, iu, rn);
      in = _mm256_fmadd_ps(i1, ru, in);
      ru = _mm256_set1_ps(matrix[4]);
      iu = _mm256_set1_ps(matrix[5]);
      rn = _mm256_fmadd_ps(r2, ru, rn);
      in = _mm256_fmadd_ps(r2, iu, in);
      rn = _mm256_fnmadd_ps(i2, iu, rn);
      in = _mm256_fmadd_ps(i2, ru, in);
      ru = _mm256_set1_ps(matrix[6]);
      iu = _mm256_set1_ps(matrix[7]);
      rn = _mm256_fmadd_ps(r3, ru, rn);
      in = _mm256_fmadd_ps(r3, iu, in);
      rn = _mm256_fnmadd_ps(i3, iu, rn);
      in = _mm256_fmadd_ps(i3, ru, in);

      ru = _mm256_set1_ps(matrix[8]);
      iu = _mm256_set1_ps(matrix[9]);
      rm = _mm256_mul_ps(r0, ru);
      im = _mm256_mul_ps(r0, iu);
      rm = _mm256_fnmadd_ps(i0, iu, rm);
      im = _mm256_fmadd_ps(i0, ru, im);
      ru = _mm256_set1_ps(matrix[10]);
      iu = _mm256_set1_ps(matrix[11]);
      rm = _mm256_fmadd_ps(r1, ru, rm);
      im = _mm256_fmadd_ps(r1, iu, im);
      rm = _mm256_fnmadd_ps(i1, iu, rm);
      im = _mm256_fmadd_ps(i1, ru, im);
      ru = _mm256_set1_ps(matrix[12]);
      iu = _mm256_set1_ps(matrix[13]);
      rm = _mm256_fmadd_ps(r2, ru, rm);
      im = _mm256_fmadd_ps(r2, iu, im);
      rm = _mm256_fnmadd_ps(i2, iu, rm);
      im = _mm256_fmadd_ps(i2, ru, im);
      ru = _mm256_set1_ps(matrix[14]);
      iu = _mm256_set1_ps(matrix[15]);
      rm = _mm256_fmadd_ps(r3, ru, rm);
      im = _mm256_fmadd_ps(r3, iu, im);
      rm = _mm256_fnmadd_ps(i3, iu, rm);
      im = _mm256_fmadd_ps(i3, ru, im);

      rm = _mm256_permutevar8x32_ps(rm, ml);
      im = _mm256_permutevar8x32_ps(im, ml);
      rn = _mm256_blendv_ps(rn, rm, mb);
      in = _mm256_blendv_ps(in, im, mb);
      p = si;
      _mm256_store_ps(rstate + p, rn);
      _mm256_store_ps(rstate + p + 8, in);

      ru = _mm256_set1_ps(matrix[16]);
      iu = _mm256_set1_ps(matrix[17]);
      rn = _mm256_mul_ps(r0, ru);
      in = _mm256_mul_ps(r0, iu);
      rn = _mm256_fnmadd_ps(i0, iu, rn);
      in = _mm256_fmadd_ps(i0, ru, in);
      ru = _mm256_set1_ps(matrix[18]);
      iu = _mm256_set1_ps(matrix[19]);
      rn = _mm256_fmadd_ps(r1, ru, rn);
      in = _mm256_fmadd_ps(r1, iu, in);
      rn = _mm256_fnmadd_ps(i1, iu, rn);
      in = _mm256_fmadd_ps(i1, ru, in);
      ru = _mm256_set1_ps(matrix[20]);
      iu = _mm256_set1_ps(matrix[21]);
      rn = _mm256_fmadd_ps(r2, ru, rn);
      in = _mm256_fmadd_ps(r2, iu, in);
      rn = _mm256_fnmadd_ps(i2, iu, rn);
      in = _mm256_fmadd_ps(i2, ru, in);
      ru = _mm256_set1_ps(matrix[22]);
      iu = _mm256_set1_ps(matrix[23]);
      rn = _mm256_fmadd_ps(r3, ru, rn);
      in = _mm256_fmadd_ps(r3, iu, in);
      rn = _mm256_fnmadd_ps(i3, iu, rn);
      in = _mm256_fmadd_ps(i3, ru, in);

      ru = _mm256_set1_ps(matrix[24]);
      iu = _mm256_set1_ps(matrix[25]);
      rm = _mm256_mul_ps(r0, ru);
      im = _mm256_mul_ps(r0, iu);
      rm = _mm256_fnmadd_ps(i0, iu, rm);
      im = _mm256_fmadd_ps(i0, ru, im);
      ru = _mm256_set1_ps(matrix[26]);
      iu = _mm256_set1_ps(matrix[27]);
      rm = _mm256_fmadd_ps(r1, ru, rm);
      im = _mm256_fmadd_ps(r1, iu, im);
      rm = _mm256_fnmadd_ps(i1, iu, rm);
      im = _mm256_fmadd_ps(i1, ru, im);
      ru = _mm256_set1_ps(matrix[28]);
      iu = _mm256_set1_ps(matrix[29]);
      rm = _mm256_fmadd_ps(r2, ru, rm);
      im = _mm256_fmadd_ps(r2, iu, im);
      rm = _mm256_fnmadd_ps(i2, iu, rm);
      im = _mm256_fmadd_ps(i2, ru, im);
      ru = _mm256_set1_ps(matrix[30]);
      iu = _mm256_set1_ps(matrix[31]);
      rm = _mm256_fmadd_ps(r3, ru, rm);
      im = _mm256_fmadd_ps(r3, iu, im);
      rm = _mm256_fnmadd_ps(i3, iu, rm);
      im = _mm256_fmadd_ps(i3, ru, im);

      rm = _mm256_permutevar8x32_ps(rm, ml);
      im = _mm256_permutevar8x32_ps(im, ml);
      rn = _mm256_blendv_ps(rn, rm, mb);
      in = _mm256_blendv_ps(in, im, mb);
      p = si | sizej;
      _mm256_store_ps(rstate + p, rn);
      _mm256_store_ps(rstate + p + 8, in);
    }
  }
}

void StateSpaceAVX::ApplyGate2LL(const unsigned int q0, const unsigned int q1,
                                 const float* matrix) {
  const unsigned int q = q0 + q1;

  __m256 mb1, mb2, mb3;
  __m256i ml1, ml2, ml3;

  uint64_t sizei = uint64_t(1) << (GetNumQubits() + 1);
  auto rstate = GetRawState();

  switch (q) {
    case 1:
      ml1 = _mm256_set_epi32(7, 6, 4, 5, 3, 2, 0, 1);
      ml2 = _mm256_set_epi32(7, 4, 5, 6, 3, 0, 1, 2);
      ml3 = _mm256_set_epi32(4, 6, 5, 7, 0, 2, 1, 3);
      mb1 = _mm256_castsi256_ps(_mm256_set_epi32(0, 0, -1, 0, 0, 0, -1, 0));
      mb2 = _mm256_castsi256_ps(_mm256_set_epi32(0, -1, 0, 0, 0, -1, 0, 0));
      mb3 = _mm256_castsi256_ps(_mm256_set_epi32(-1, 0, 0, 0, -1, 0, 0, 0));
      break;
    case 2:
      ml1 = _mm256_set_epi32(7, 6, 5, 4, 2, 3, 0, 1);
      ml2 = _mm256_set_epi32(7, 2, 5, 0, 3, 6, 1, 4);
      ml3 = _mm256_set_epi32(2, 6, 0, 4, 3, 7, 1, 5);
      mb1 = _mm256_castsi256_ps(_mm256_set_epi32(0, 0, 0, 0, -1, 0, -1, 0));
      mb2 = _mm256_castsi256_ps(_mm256_set_epi32(0, -1, 0, -1, 0, 0, 0, 0));
      mb3 = _mm256_castsi256_ps(_mm256_set_epi32(-1, 0, -1, 0, 0, 0, 0, 0));
      break;
    case 3:
      ml1 = _mm256_set_epi32(7, 6, 5, 4, 1, 0, 3, 2);
      ml2 = _mm256_set_epi32(7, 6, 1, 0, 3, 2, 5, 4);
      ml3 = _mm256_set_epi32(1, 0, 5, 4, 3, 2, 7, 6);
      mb1 = _mm256_castsi256_ps(_mm256_set_epi32(0, 0, 0, 0, -1, -1, 0, 0));
      mb2 = _mm256_castsi256_ps(_mm256_set_epi32(0, 0, -1, -1, 0, 0, 0, 0));
      mb3 = _mm256_castsi256_ps(_mm256_set_epi32(-1, -1, 0, 0, 0, 0, 0, 0));
      break;
  }

  for (uint64_t i = 0; i < sizei; i += 16) {
    __m256 r0, i0, r1, i1, r2, i2, r3, i3, ru, iu, rn, in, rm, im;

    auto p = rstate + i;

    r0 = _mm256_load_ps(p);
    i0 = _mm256_load_ps(p + 8);

    r1 = _mm256_permutevar8x32_ps(r0, ml1);
    i1 = _mm256_permutevar8x32_ps(i0, ml1);

    r2 = _mm256_permutevar8x32_ps(r0, ml2);
    i2 = _mm256_permutevar8x32_ps(i0, ml2);

    r3 = _mm256_permutevar8x32_ps(r0, ml3);
    i3 = _mm256_permutevar8x32_ps(i0, ml3);

    ru = _mm256_set1_ps(matrix[0]);
    iu = _mm256_set1_ps(matrix[1]);
    rn = _mm256_mul_ps(r0, ru);
    in = _mm256_mul_ps(r0, iu);
    rn = _mm256_fnmadd_ps(i0, iu, rn);
    in = _mm256_fmadd_ps(i0, ru, in);
    ru = _mm256_set1_ps(matrix[2]);
    iu = _mm256_set1_ps(matrix[3]);
    rn = _mm256_fmadd_ps(r1, ru, rn);
    in = _mm256_fmadd_ps(r1, iu, in);
    rn = _mm256_fnmadd_ps(i1, iu, rn);
    in = _mm256_fmadd_ps(i1, ru, in);
    ru = _mm256_set1_ps(matrix[4]);
    iu = _mm256_set1_ps(matrix[5]);
    rn = _mm256_fmadd_ps(r2, ru, rn);
    in = _mm256_fmadd_ps(r2, iu, in);
    rn = _mm256_fnmadd_ps(i2, iu, rn);
    in = _mm256_fmadd_ps(i2, ru, in);
    ru = _mm256_set1_ps(matrix[6]);
    iu = _mm256_set1_ps(matrix[7]);
    rn = _mm256_fmadd_ps(r3, ru, rn);
    in = _mm256_fmadd_ps(r3, iu, in);
    rn = _mm256_fnmadd_ps(i3, iu, rn);
    in = _mm256_fmadd_ps(i3, ru, in);

    ru = _mm256_set1_ps(matrix[8]);
    iu = _mm256_set1_ps(matrix[9]);
    rm = _mm256_mul_ps(r0, ru);
    im = _mm256_mul_ps(r0, iu);
    rm = _mm256_fnmadd_ps(i0, iu, rm);
    im = _mm256_fmadd_ps(i0, ru, im);
    ru = _mm256_set1_ps(matrix[10]);
    iu = _mm256_set1_ps(matrix[11]);
    rm = _mm256_fmadd_ps(r1, ru, rm);
    im = _mm256_fmadd_ps(r1, iu, im);
    rm = _mm256_fnmadd_ps(i1, iu, rm);
    im = _mm256_fmadd_ps(i1, ru, im);
    ru = _mm256_set1_ps(matrix[12]);
    iu = _mm256_set1_ps(matrix[13]);
    rm = _mm256_fmadd_ps(r2, ru, rm);
    im = _mm256_fmadd_ps(r2, iu, im);
    rm = _mm256_fnmadd_ps(i2, iu, rm);
    im = _mm256_fmadd_ps(i2, ru, im);
    ru = _mm256_set1_ps(matrix[14]);
    iu = _mm256_set1_ps(matrix[15]);
    rm = _mm256_fmadd_ps(r3, ru, rm);
    im = _mm256_fmadd_ps(r3, iu, im);
    rm = _mm256_fnmadd_ps(i3, iu, rm);
    im = _mm256_fmadd_ps(i3, ru, im);

    rm = _mm256_permutevar8x32_ps(rm, ml1);
    im = _mm256_permutevar8x32_ps(im, ml1);
    rn = _mm256_blendv_ps(rn, rm, mb1);
    in = _mm256_blendv_ps(in, im, mb1);

    ru = _mm256_set1_ps(matrix[16]);
    iu = _mm256_set1_ps(matrix[17]);
    rm = _mm256_mul_ps(r0, ru);
    im = _mm256_mul_ps(r0, iu);
    rm = _mm256_fnmadd_ps(i0, iu, rm);
    im = _mm256_fmadd_ps(i0, ru, im);
    ru = _mm256_set1_ps(matrix[18]);
    iu = _mm256_set1_ps(matrix[19]);
    rm = _mm256_fmadd_ps(r1, ru, rm);
    im = _mm256_fmadd_ps(r1, iu, im);
    rm = _mm256_fnmadd_ps(i1, iu, rm);
    im = _mm256_fmadd_ps(i1, ru, im);
    ru = _mm256_set1_ps(matrix[20]);
    iu = _mm256_set1_ps(matrix[21]);
    rm = _mm256_fmadd_ps(r2, ru, rm);
    im = _mm256_fmadd_ps(r2, iu, im);
    rm = _mm256_fnmadd_ps(i2, iu, rm);
    im = _mm256_fmadd_ps(i2, ru, im);
    ru = _mm256_set1_ps(matrix[22]);
    iu = _mm256_set1_ps(matrix[23]);
    rm = _mm256_fmadd_ps(r3, ru, rm);
    im = _mm256_fmadd_ps(r3, iu, im);
    rm = _mm256_fnmadd_ps(i3, iu, rm);
    im = _mm256_fmadd_ps(i3, ru, im);

    rm = _mm256_permutevar8x32_ps(rm, ml2);
    im = _mm256_permutevar8x32_ps(im, ml2);
    rn = _mm256_blendv_ps(rn, rm, mb2);
    in = _mm256_blendv_ps(in, im, mb2);

    ru = _mm256_set1_ps(matrix[24]);
    iu = _mm256_set1_ps(matrix[25]);
    rm = _mm256_mul_ps(r0, ru);
    im = _mm256_mul_ps(r0, iu);
    rm = _mm256_fnmadd_ps(i0, iu, rm);
    im = _mm256_fmadd_ps(i0, ru, im);
    ru = _mm256_set1_ps(matrix[26]);
    iu = _mm256_set1_ps(matrix[27]);
    rm = _mm256_fmadd_ps(r1, ru, rm);
    im = _mm256_fmadd_ps(r1, iu, im);
    rm = _mm256_fnmadd_ps(i1, iu, rm);
    im = _mm256_fmadd_ps(i1, ru, im);
    ru = _mm256_set1_ps(matrix[28]);
    iu = _mm256_set1_ps(matrix[29]);
    rm = _mm256_fmadd_ps(r2, ru, rm);
    im = _mm256_fmadd_ps(r2, iu, im);
    rm = _mm256_fnmadd_ps(i2, iu, rm);
    im = _mm256_fmadd_ps(i2, ru, im);
    ru = _mm256_set1_ps(matrix[30]);
    iu = _mm256_set1_ps(matrix[31]);
    rm = _mm256_fmadd_ps(r3, ru, rm);
    im = _mm256_fmadd_ps(r3, iu, im);
    rm = _mm256_fnmadd_ps(i3, iu, rm);
    im = _mm256_fmadd_ps(i3, ru, im);

    rm = _mm256_permutevar8x32_ps(rm, ml3);
    im = _mm256_permutevar8x32_ps(im, ml3);
    rn = _mm256_blendv_ps(rn, rm, mb3);
    in = _mm256_blendv_ps(in, im, mb3);

    _mm256_store_ps(p, rn);
    _mm256_store_ps(p + 8, in);
  }
}

}  // namespace qsim
}  // namespace tfq

#endif
