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

#include "tensorflow_quantum/core/qsim/simulator2_slow.h"

#include <cmath>
#include <cstdint>

#include "tensorflow_quantum/core/qsim/simulator.h"
#include "tensorflow_quantum/core/qsim/statespace_slow.h"

namespace tfq {
namespace qsim {

Simulator2Slow::Simulator2Slow(const unsigned int num_qubits,
                               const unsigned int num_threads)
    : Simulator(num_qubits, num_threads) {}

void Simulator2Slow::ApplyGate1(const float* matrix, State* state) const {
  // Workaround function to apply single qubit gates if the
  // circuit only has one qubit.
  auto data = StateSpace::RawData(state);

  float r_0, i_0, r_1, i_1;
  r_0 = data[0] * matrix[0] - data[1] * matrix[1] + data[2] * matrix[2] -
        data[3] * matrix[3];
  i_0 = data[0] * matrix[1] + data[1] * matrix[0] + data[2] * matrix[3] +
        data[3] * matrix[2];

  r_1 = data[0] * matrix[4] - data[1] * matrix[5] + data[2] * matrix[6] -
        data[3] * matrix[7];
  i_1 = data[0] * matrix[5] + data[1] * matrix[4] + data[2] * matrix[7] +
        data[3] * matrix[6];

  data[0] = r_0;
  data[1] = i_0;
  data[2] = r_1;
  data[3] = i_1;
}

void Simulator2Slow::ApplyGate2(const unsigned int q0, const unsigned int q1,
                                const float* m, State* state) const {
  // Assume q0 < q1.

  uint64_t sizei = uint64_t(1) << (num_qubits_ + 1);
  uint64_t sizej = uint64_t(1) << (q1 + 1);
  uint64_t sizek = uint64_t(1) << (q0 + 1);

  auto state_ = StateSpace::RawData(state);

  for (uint64_t i = 0; i < sizei; i += 2 * sizej) {
    for (uint64_t j = 0; j < sizej; j += 2 * sizek) {
      for (uint64_t k = 0; k < sizek; k += 2) {
        uint64_t si = i | j | k;

        uint64_t p = si;
        float s0r = state_[p + 0];
        float s0i = state_[p + 1];
        p = si | sizek;
        float s1r = state_[p + 0];
        float s1i = state_[p + 1];
        p = si | sizej;
        float s2r = state_[p + 0];
        float s2i = state_[p + 1];
        p |= sizek;
        float s3r = state_[p + 0];
        float s3i = state_[p + 1];

        p = si;
        state_[p + 0] = s0r * m[0] - s0i * m[1] + s1r * m[2] - s1i * m[3] +
                        s2r * m[4] - s2i * m[5] + s3r * m[6] - s3i * m[7];
        state_[p + 1] = s0r * m[1] + s0i * m[0] + s1r * m[3] + s1i * m[2] +
                        s2r * m[5] + s2i * m[4] + s3r * m[7] + s3i * m[6];
        p = si | sizek;
        state_[p + 0] = s0r * m[8] - s0i * m[9] + s1r * m[10] - s1i * m[11] +
                        s2r * m[12] - s2i * m[13] + s3r * m[14] - s3i * m[15];
        state_[p + 1] = s0r * m[9] + s0i * m[8] + s1r * m[11] + s1i * m[10] +
                        s2r * m[13] + s2i * m[12] + s3r * m[15] + s3i * m[14];
        p = si | sizej;
        state_[p + 0] = s0r * m[16] - s0i * m[17] + s1r * m[18] - s1i * m[19] +
                        s2r * m[20] - s2i * m[21] + s3r * m[22] - s3i * m[23];
        state_[p + 1] = s0r * m[17] + s0i * m[16] + s1r * m[19] + s1i * m[18] +
                        s2r * m[21] + s2i * m[20] + s3r * m[23] + s3i * m[22];
        p |= sizek;
        state_[p + 0] = s0r * m[24] - s0i * m[25] + s1r * m[26] - s1i * m[27] +
                        s2r * m[28] - s2i * m[29] + s3r * m[30] - s3i * m[31];
        state_[p + 1] = s0r * m[25] + s0i * m[24] + s1r * m[27] + s1i * m[26] +
                        s2r * m[29] + s2i * m[28] + s3r * m[31] + s3i * m[30];
      }
    }
  }
}

std::complex<float> Simulator2Slow::GetAmpl(const State& state,
                                            const uint64_t i) const {
  auto data = RawData(state);
  return std::complex<float>(data[2 * i], data[2 * i + 1]);
}

void Simulator2Slow::SetAmpl(State* state, const uint64_t i,
                             const std::complex<float>& val) const {
  auto data = RawData(state);
  data[2 * i] = val.real();
  data[2 * i + 1] = val.imag();
}

}  // namespace qsim
}  // namespace tfq
