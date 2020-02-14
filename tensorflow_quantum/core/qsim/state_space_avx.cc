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

StateSpaceSlow::StateSpaceSlow(const unsigned int num_qubits,
                               const unsigned int num_threads)
    : StateSpace(num_qubits, num_threads) {
  CreateState();
}

void StateSpace::CreateState() {
  state_ =
      (float*)qsim::_aligned_malloc(sizeof(float) * 2 * this->GetDimension());
}

void StateSpace::DeleteState() { qsim::_aligned_free(state_); }

StateSpace* StateSpaceSlow::Copy() const {
  StateSpace* state_copy =
      new StateSpaceSlow(this->GetNumQubits(), this->GetNumThreads());
  for (uint64_t i = 0; i < this->GetDimension(); ++i) {
    state_copy->SetAmpl(i, this->GetAmpl(i));
  }
  return state_copy;
}

void StateSpaceSlow::ApplyGate2(const unsigned int q0, const unsigned int q1,
                                const float* m) {
  // Assume q0 < q1.
  uint64_t sizei = uint64_t(1) << (this->num_qubits_ + 1);
  uint64_t sizej = uint64_t(1) << (q1 + 1);
  uint64_t sizek = uint64_t(1) << (q0 + 1);

  for (uint64_t i = 0; i < sizei; i += 2 * sizej) {
    for (uint64_t j = 0; j < sizej; j += 2 * sizek) {
      for (uint64_t k = 0; k < sizek; k += 2) {
        uint64_t si = i | j | k;

        uint64_t p = si;
        float s0r = this->state_[p + 0];
        float s0i = this->state_[p + 1];
        p = si | sizek;
        float s1r = this->state_[p + 0];
        float s1i = this->state_[p + 1];
        p = si | sizej;
        float s2r = this->state_[p + 0];
        float s2i = this->state_[p + 1];
        p |= sizek;
        float s3r = this->state_[p + 0];
        float s3i = this->state_[p + 1];

        p = si;
        this->state_[p + 0] = s0r * m[0] - s0i * m[1] + s1r * m[2] -
                              s1i * m[3] + s2r * m[4] - s2i * m[5] +
                              s3r * m[6] - s3i * m[7];
        this->state_[p + 1] = s0r * m[1] + s0i * m[0] + s1r * m[3] +
                              s1i * m[2] + s2r * m[5] + s2i * m[4] +
                              s3r * m[7] + s3i * m[6];
        p = si | sizek;
        this->state_[p + 0] = s0r * m[8] - s0i * m[9] + s1r * m[10] -
                              s1i * m[11] + s2r * m[12] - s2i * m[13] +
                              s3r * m[14] - s3i * m[15];
        this->state_[p + 1] = s0r * m[9] + s0i * m[8] + s1r * m[11] +
                              s1i * m[10] + s2r * m[13] + s2i * m[12] +
                              s3r * m[15] + s3i * m[14];
        p = si | sizej;
        this->state_[p + 0] = s0r * m[16] - s0i * m[17] + s1r * m[18] -
                              s1i * m[19] + s2r * m[20] - s2i * m[21] +
                              s3r * m[22] - s3i * m[23];
        this->state_[p + 1] = s0r * m[17] + s0i * m[16] + s1r * m[19] +
                              s1i * m[18] + s2r * m[21] + s2i * m[20] +
                              s3r * m[23] + s3i * m[22];
        p |= sizek;
        this->state_[p + 0] = s0r * m[24] - s0i * m[25] + s1r * m[26] -
                              s1i * m[27] + s2r * m[28] - s2i * m[29] +
                              s3r * m[30] - s3i * m[31];
        this->state_[p + 1] = s0r * m[25] + s0i * m[24] + s1r * m[27] +
                              s1i * m[26] + s2r * m[29] + s2i * m[28] +
                              s3r * m[31] + s3i * m[30];
      }
    }
  }
}

tensorflow::Status StateSpaceSlow::ApplyGate1(const float* matrix) {
  // Workaround function to apply single qubit gates if the
  // circuit only has one qubit.

  float r_0, i_0, r_1, i_1;
  r_0 = this->state_[0] * matrix[0] - this->state_[1] * matrix[1] +
        this->state_[2] * matrix[2] - this->state_[3] * matrix[3];
  i_0 = this->state_[0] * matrix[1] + this->state_[1] * matrix[0] +
        this->state_[2] * matrix[3] + this->state_[3] * matrix[2];

  r_1 = this->state_[0] * matrix[4] - this->state_[1] * matrix[5] +
        this->state_[2] * matrix[6] - this->state_[3] * matrix[7];
  i_1 = this->state_[0] * matrix[5] + this->state_[1] * matrix[4] +
        this->state_[2] * matrix[7] + this->state_[3] * matrix[6];

  this->state_[0] = r_0;
  this->state_[1] = i_0;
  this->state_[2] = r_1;
  this->state_[3] = i_1;

  return tensorflow::Status::OK();
}

void StateSpaceSlow::SetStateZero() {
  //#pragma omp parallel for num_threads(num_threads_)
  for (uint64_t i = 0; i < size_; ++i) {
    this->state_[i] = 0;
  }
  this->state_[0] = 1;
}

float StateSpaceSlow::GetRealInnerProduct(const StateSpace* other) const {
  uint64_t size2 = this->GetDimension();
  double result = 0.0;

  // Currently not a thread safe implementation of inner product!
  for (uint64_t i = 0; i < size2; ++i) {
    const std::complex<float> amp_a = this->GetAmpl(i);
    const std::complex<float> amp_other = other->GetAmpl(i);

    const std::complex<double> amp_a_d = std::complex<double>(
        static_cast<double>(amp_a.real()), static_cast<double>(amp_a.imag()));

    const std::complex<double> amp_other_d =
        std::complex<double>(static_cast<double>(amp_other.real()),
                             static_cast<double>(amp_other.imag()));

    result += (std::conj(amp_a_d) * amp_other_d).real();
  }

  return static_cast<float>(result);
}

std::complex<float> StateSpaceSlow::GetAmpl(const uint64_t i) const {
  return std::complex<float>(this->state_[2 * i], this->state_[2 * i + 1]);
}

void StateSpaceSlow::SetAmpl(const uint64_t i, const std::complex<float>& val) {
  this->state_[2 * i] = val.real();
  this->state_[2 * i + 1] = val.imag();
}

}  // namespace qsim
}  // namespace tfq
