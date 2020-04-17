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

#include "tensorflow_quantum/core/qsim/unitary_space_slow.h"

#include <cmath>
#include <complex>
#include <cstdint>
#include <memory>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow_quantum/core/qsim/matrix.h"

namespace tfq {
namespace qsim {
namespace {

inline const float dot_one_r(const float m_0i_r, const float m_0i_i,
                             const float m_1i_r, const float m_1i_i,
                             const int i, const float* gate) {
  return m_0i_r * gate[i * 4 + 0] - m_0i_i * gate[i * 4 + 1] +
         m_1i_r * gate[i * 4 + 2] - m_1i_i * gate[i * 4 + 3];
}

inline const float dot_one_i(const float m_0i_r, const float m_0i_i,
                             const float m_1i_r, const float m_1i_i,
                             const int i, const float* gate) {
  return m_0i_r * gate[i * 4 + 1] + m_0i_i * gate[i * 4 + 0] +
         m_1i_r * gate[i * 4 + 3] + m_1i_i * gate[i * 4 + 2];
}

inline const float dot_two_r(const float m_0i_r, const float m_0i_i,
                             const float m_1i_r, const float m_1i_i,
                             const float m_2i_r, const float m_2i_i,
                             const float m_3i_r, const float m_3i_i,
                             const int i, const float* gate) {
  return m_0i_r * gate[i * 8 + 0] - m_0i_i * gate[i * 8 + 1] +
         m_1i_r * gate[i * 8 + 2] - m_1i_i * gate[i * 8 + 3] +
         m_2i_r * gate[i * 8 + 4] - m_2i_i * gate[i * 8 + 5] +
         m_3i_r * gate[i * 8 + 6] - m_3i_i * gate[i * 8 + 7];
}

inline const float dot_two_i(const float m_0i_r, const float m_0i_i,
                             const float m_1i_r, const float m_1i_i,
                             const float m_2i_r, const float m_2i_i,
                             const float m_3i_r, const float m_3i_i,
                             const int i, const float* gate) {
  return m_0i_r * gate[i * 8 + 1] + m_0i_i * gate[i * 8 + 0] +
         m_1i_r * gate[i * 8 + 3] + m_1i_i * gate[i * 8 + 2] +
         m_2i_r * gate[i * 8 + 5] + m_2i_i * gate[i * 8 + 4] +
         m_3i_r * gate[i * 8 + 7] + m_3i_i * gate[i * 8 + 6];
}

}  // namespace

UnitarySpaceSlow::UnitarySpaceSlow(const uint64_t num_qubits,
                                   const uint64_t num_threads)
    : UnitarySpace(num_qubits, num_threads) {}

UnitarySpaceSlow::~UnitarySpaceSlow() { DeleteUnitary(); }

UnitarySpaceType UnitarySpaceSlow::GetType() const {
  return UnitarySpaceType::USLOW;
}

void UnitarySpaceSlow::CreateUnitary() {
  state_ = (float*)malloc(sizeof(float) * size_);
}

void UnitarySpaceSlow::DeleteUnitary() {
  if (GetRawUnitary() != NULL) {
    free(state_);
    state_ = NULL;
  }
}

void UnitarySpaceSlow::ApplyGate2(const unsigned int q0_be,
                                  const unsigned int q1_be,
                                  const float* matrix) {
  // Assume q0 < q1.
  const unsigned int q0_le = GetNumQubits() - q1_be - 1;
  const unsigned int q1_le = GetNumQubits() - q0_be - 1;

  const uint64_t sizei = uint64_t(1) << (GetNumQubits());
  const uint64_t sizej = uint64_t(1) << (q1_le);
  const uint64_t sizek = uint64_t(1) << (q0_le);

  auto data = GetRawUnitary();

  float m00_r, m00_i, m01_r, m01_i, m02_r, m02_i, m03_r, m03_i;
  float m10_r, m10_i, m11_r, m11_i, m12_r, m12_i, m13_r, m13_i;
  float m20_r, m20_i, m21_r, m21_i, m22_r, m22_i, m23_r, m23_i;
  float m30_r, m30_i, m31_r, m31_i, m32_r, m32_i, m33_r, m33_i;

  for (uint64_t i = 0; i < sizei; i += 2 * sizej) {
    for (uint64_t j = 0; j < sizej; j += 2 * sizek) {
      for (uint64_t k = 0; k < sizek; k += 1) {
        for (uint64_t ii = 0; ii < sizei; ii += 2 * sizej) {
          for (uint64_t jj = 0; jj < sizej; jj += 2 * sizek) {
            for (uint64_t kk = 0; kk < sizek; kk += 1) {
              uint64_t si = i | j | k;
              uint64_t si2 = ii | jj | kk;
              uint64_t p = si;
              uint64_t pp = si2;
              m00_r = data[2 * p * sizei + 2 * pp];
              m00_i = data[2 * p * sizei + 2 * pp + 1];
              pp = si2 | sizek;
              m01_r = data[2 * p * sizei + 2 * pp];
              m01_i = data[2 * p * sizei + 2 * pp + 1];
              pp = si2 | sizej;
              m02_r = data[2 * p * sizei + 2 * pp];
              m02_i = data[2 * p * sizei + 2 * pp + 1];
              pp |= sizek;
              m03_r = data[2 * p * sizei + 2 * pp];
              m03_i = data[2 * p * sizei + 2 * pp + 1];

              p = si | sizek;
              pp = si2;
              m10_r = data[2 * p * sizei + 2 * pp];
              m10_i = data[2 * p * sizei + 2 * pp + 1];
              pp = si2 | sizek;
              m11_r = data[2 * p * sizei + 2 * pp];
              m11_i = data[2 * p * sizei + 2 * pp + 1];
              pp = si2 | sizej;
              m12_r = data[2 * p * sizei + 2 * pp];
              m12_i = data[2 * p * sizei + 2 * pp + 1];
              pp |= sizek;
              m13_r = data[2 * p * sizei + 2 * pp];
              m13_i = data[2 * p * sizei + 2 * pp + 1];

              p = si | sizej;
              pp = si2;
              m20_r = data[2 * p * sizei + 2 * pp];
              m20_i = data[2 * p * sizei + 2 * pp + 1];
              pp = si2 | sizek;
              m21_r = data[2 * p * sizei + 2 * pp];
              m21_i = data[2 * p * sizei + 2 * pp + 1];
              pp = si2 | sizej;
              m22_r = data[2 * p * sizei + 2 * pp];
              m22_i = data[2 * p * sizei + 2 * pp + 1];
              pp |= sizek;
              m23_r = data[2 * p * sizei + 2 * pp];
              m23_i = data[2 * p * sizei + 2 * pp + 1];

              p |= sizek;
              pp = si2;
              m30_r = data[2 * p * sizei + 2 * pp];
              m30_i = data[2 * p * sizei + 2 * pp + 1];
              pp = si2 | sizek;
              m31_r = data[2 * p * sizei + 2 * pp];
              m31_i = data[2 * p * sizei + 2 * pp + 1];
              pp = si2 | sizej;
              m32_r = data[2 * p * sizei + 2 * pp];
              m32_i = data[2 * p * sizei + 2 * pp + 1];
              pp |= sizek;
              m33_r = data[2 * p * sizei + 2 * pp];
              m33_i = data[2 * p * sizei + 2 * pp + 1];

              // End of extraction. Begin computation.
              p = si;
              pp = si2;
              data[2 * p * sizei + 2 * pp] =
                  dot_two_r(m00_r, m00_i, m10_r, m10_i, m20_r, m20_i, m30_r,
                            m30_i, 0, matrix);
              data[2 * p * sizei + 2 * pp + 1] =
                  dot_two_i(m00_r, m00_i, m10_r, m10_i, m20_r, m20_i, m30_r,
                            m30_i, 0, matrix);
              pp = si2 | sizek;
              data[2 * p * sizei + 2 * pp] =
                  dot_two_r(m01_r, m01_i, m11_r, m11_i, m21_r, m21_i, m31_r,
                            m31_i, 0, matrix);
              data[2 * p * sizei + 2 * pp + 1] =
                  dot_two_i(m01_r, m01_i, m11_r, m11_i, m21_r, m21_i, m31_r,
                            m31_i, 0, matrix);
              pp = si2 | sizej;
              data[2 * p * sizei + 2 * pp] =
                  dot_two_r(m02_r, m02_i, m12_r, m12_i, m22_r, m22_i, m32_r,
                            m32_i, 0, matrix);
              data[2 * p * sizei + 2 * pp + 1] =
                  dot_two_i(m02_r, m02_i, m12_r, m12_i, m22_r, m22_i, m32_r,
                            m32_i, 0, matrix);
              pp |= sizek;
              data[2 * p * sizei + 2 * pp] =
                  dot_two_r(m03_r, m03_i, m13_r, m13_i, m23_r, m23_i, m33_r,
                            m33_i, 0, matrix);
              data[2 * p * sizei + 2 * pp + 1] =
                  dot_two_i(m03_r, m03_i, m13_r, m13_i, m23_r, m23_i, m33_r,
                            m33_i, 0, matrix);

              p = si | sizek;
              pp = si2;
              data[2 * p * sizei + 2 * pp] =
                  dot_two_r(m00_r, m00_i, m10_r, m10_i, m20_r, m20_i, m30_r,
                            m30_i, 1, matrix);
              data[2 * p * sizei + 2 * pp + 1] =
                  dot_two_i(m00_r, m00_i, m10_r, m10_i, m20_r, m20_i, m30_r,
                            m30_i, 1, matrix);
              pp = si2 | sizek;
              data[2 * p * sizei + 2 * pp] =
                  dot_two_r(m01_r, m01_i, m11_r, m11_i, m21_r, m21_i, m31_r,
                            m31_i, 1, matrix);
              data[2 * p * sizei + 2 * pp + 1] =
                  dot_two_i(m01_r, m01_i, m11_r, m11_i, m21_r, m21_i, m31_r,
                            m31_i, 1, matrix);
              pp = si2 | sizej;
              data[2 * p * sizei + 2 * pp] =
                  dot_two_r(m02_r, m02_i, m12_r, m12_i, m22_r, m22_i, m32_r,
                            m32_i, 1, matrix);
              data[2 * p * sizei + 2 * pp + 1] =
                  dot_two_i(m02_r, m02_i, m12_r, m12_i, m22_r, m22_i, m32_r,
                            m32_i, 1, matrix);
              pp |= sizek;
              data[2 * p * sizei + 2 * pp] =
                  dot_two_r(m03_r, m03_i, m13_r, m13_i, m23_r, m23_i, m33_r,
                            m33_i, 1, matrix);
              data[2 * p * sizei + 2 * pp + 1] =
                  dot_two_i(m03_r, m03_i, m13_r, m13_i, m23_r, m23_i, m33_r,
                            m33_i, 1, matrix);

              p = si | sizej;
              pp = si2;
              data[2 * p * sizei + 2 * pp] =
                  dot_two_r(m00_r, m00_i, m10_r, m10_i, m20_r, m20_i, m30_r,
                            m30_i, 2, matrix);
              data[2 * p * sizei + 2 * pp + 1] =
                  dot_two_i(m00_r, m00_i, m10_r, m10_i, m20_r, m20_i, m30_r,
                            m30_i, 2, matrix);
              pp = si2 | sizek;
              data[2 * p * sizei + 2 * pp] =
                  dot_two_r(m01_r, m01_i, m11_r, m11_i, m21_r, m21_i, m31_r,
                            m31_i, 2, matrix);
              data[2 * p * sizei + 2 * pp + 1] =
                  dot_two_i(m01_r, m01_i, m11_r, m11_i, m21_r, m21_i, m31_r,
                            m31_i, 2, matrix);
              pp = si2 | sizej;
              data[2 * p * sizei + 2 * pp] =
                  dot_two_r(m02_r, m02_i, m12_r, m12_i, m22_r, m22_i, m32_r,
                            m32_i, 2, matrix);
              data[2 * p * sizei + 2 * pp + 1] =
                  dot_two_i(m02_r, m02_i, m12_r, m12_i, m22_r, m22_i, m32_r,
                            m32_i, 2, matrix);
              pp |= sizek;
              data[2 * p * sizei + 2 * pp] =
                  dot_two_r(m03_r, m03_i, m13_r, m13_i, m23_r, m23_i, m33_r,
                            m33_i, 2, matrix);
              data[2 * p * sizei + 2 * pp + 1] =
                  dot_two_i(m03_r, m03_i, m13_r, m13_i, m23_r, m23_i, m33_r,
                            m33_i, 2, matrix);

              p |= sizek;
              pp = si2;
              data[2 * p * sizei + 2 * pp] =
                  dot_two_r(m00_r, m00_i, m10_r, m10_i, m20_r, m20_i, m30_r,
                            m30_i, 3, matrix);
              data[2 * p * sizei + 2 * pp + 1] =
                  dot_two_i(m00_r, m00_i, m10_r, m10_i, m20_r, m20_i, m30_r,
                            m30_i, 3, matrix);
              pp = si2 | sizek;
              data[2 * p * sizei + 2 * pp] =
                  dot_two_r(m01_r, m01_i, m11_r, m11_i, m21_r, m21_i, m31_r,
                            m31_i, 3, matrix);
              data[2 * p * sizei + 2 * pp + 1] =
                  dot_two_i(m01_r, m01_i, m11_r, m11_i, m21_r, m21_i, m31_r,
                            m31_i, 3, matrix);
              pp = si2 | sizej;
              data[2 * p * sizei + 2 * pp] =
                  dot_two_r(m02_r, m02_i, m12_r, m12_i, m22_r, m22_i, m32_r,
                            m32_i, 3, matrix);
              data[2 * p * sizei + 2 * pp + 1] =
                  dot_two_i(m02_r, m02_i, m12_r, m12_i, m22_r, m22_i, m32_r,
                            m32_i, 3, matrix);
              pp |= sizek;
              data[2 * p * sizei + 2 * pp] =
                  dot_two_r(m03_r, m03_i, m13_r, m13_i, m23_r, m23_i, m33_r,
                            m33_i, 3, matrix);
              data[2 * p * sizei + 2 * pp + 1] =
                  dot_two_i(m03_r, m03_i, m13_r, m13_i, m23_r, m23_i, m33_r,
                            m33_i, 3, matrix);
            }
          }
        }
      }
    }
  }
}

tensorflow::Status UnitarySpaceSlow::ApplyGate1(const float* matrix) {
  float mat_00_r, mat_01_r, mat_10_r, mat_11_r;
  float mat_00_i, mat_01_i, mat_10_i, mat_11_i;

  auto state = GetRawUnitary();
  mat_00_r = dot_one_r(state[0], state[1], state[4], state[5], 0, matrix);
  mat_00_i = dot_one_i(state[0], state[1], state[4], state[5], 0, matrix);

  mat_01_r = dot_one_r(state[2], state[3], state[6], state[7], 0, matrix);
  mat_01_i = dot_one_i(state[2], state[3], state[6], state[7], 0, matrix);

  mat_10_r = dot_one_r(state[0], state[1], state[4], state[5], 1, matrix);
  mat_10_i = dot_one_i(state[0], state[1], state[4], state[5], 1, matrix);

  mat_11_r = dot_one_r(state[2], state[3], state[6], state[7], 1, matrix);
  mat_11_i = dot_one_i(state[2], state[3], state[6], state[7], 1, matrix);

  state[0] = mat_00_r;
  state[1] = mat_00_i;
  state[2] = mat_01_r;
  state[3] = mat_01_i;
  state[4] = mat_10_r;
  state[5] = mat_10_i;
  state[6] = mat_11_r;
  state[7] = mat_11_i;

  return tensorflow::Status::OK();
}

void UnitarySpaceSlow::SetIdentity() {
  //#pragma omp parallel for num_threads(num_threads_)
  auto data = GetRawUnitary();
  uint64_t dim = uint64_t(1) << GetNumQubits();
  for (uint64_t i = 0; i < dim; i++) {
    for (uint64_t j = 0; j < dim; j++) {
      data[2 * i * dim + 2 * j] = 0;
      data[2 * i * dim + 2 * j + 1] = 0;
    }
    data[2 * i * dim + 2 * i] = 1;
  }
}

std::complex<float> UnitarySpaceSlow::GetEntry(const uint64_t i,
                                               const uint64_t j) const {
  uint64_t dim = uint64_t(1) << (GetNumQubits() + 1);
  return std::complex<float>(GetRawUnitary()[i * dim + 2 * j],
                             GetRawUnitary()[i * dim + 2 * j + 1]);
}

void UnitarySpaceSlow::SetEntry(const uint64_t i, const uint64_t j,
                                const std::complex<float>& val) {
  uint64_t dim = uint64_t(1) << (GetNumQubits() + 1);
  GetRawUnitary()[i * dim + 2 * j] = val.real();
  GetRawUnitary()[i * dim + 2 * j + 1] = val.imag();
}

}  // namespace qsim
}  // namespace tfq
