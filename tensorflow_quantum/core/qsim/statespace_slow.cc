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

#include "tensorflow_quantum/core/qsim/statespace_slow.h"

#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>

#include "tensorflow_quantum/core/qsim/statespace.h"

namespace tfq {
namespace qsim {

StateSpaceSlow::StateSpaceSlow(const unsigned int num_qubits,
                               const unsigned int num_threads)
    : StateSpace(num_qubits, num_threads) {}

void StateSpaceSlow::CopyState(const State& src, State* dest) const {
  for (uint64_t i = 0; i < size_; ++i) {
    dest->get()[i] = src.get()[i];
  }
}

void StateSpaceSlow::SetStateZero(State* state) const {
  uint64_t size = size_ / 2;

  auto data = RawData(state);

  //#pragma omp parallel for num_threads(num_threads_)
  for (uint64_t i = 0; i < size; ++i) {
    data[2 * i + 0] = 0;
    data[2 * i + 1] = 0;
  }

  data[0] = 1;
}

float StateSpaceSlow::GetRealInnerProduct(const State& a,
                                          const State& b) const {
  uint64_t size2 = (size_ / 2);
  double result = 0.0;

  // Currently not a thread safe implementation of inner product!
  for (uint64_t i = 0; i < size2; ++i) {
    const std::complex<float> amp_a = GetAmpl(a, i);
    const std::complex<float> amp_b = GetAmpl(b, i);

    const std::complex<double> amp_a_d = std::complex<double>(
        static_cast<double>(amp_a.real()), static_cast<double>(amp_a.imag()));

    const std::complex<double> amp_b_d = std::complex<double>(
        static_cast<double>(amp_b.real()), static_cast<double>(amp_b.imag()));

    result += (std::conj(amp_a_d) * amp_b_d).real();
  }

  return static_cast<float>(result);
}

}  // namespace qsim
}  // namespace tfq
